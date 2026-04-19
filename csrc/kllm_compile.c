/*
 * kllm_compile.c — C compiler for kllm.
 *
 * Reads pre-downloaded model weights + tokenizer from disk and produces
 * a compiled chip directory ready for C-native inference.
 *
 * Pipeline:
 *   1. Read config.json + weight .npy files
 *   2. Build tape instructions directly (fixed LLaMA decode structure)
 *   3. Truth-table compress weight matrices (sort unique → codebook + uint16)
 *   4. Build BPE ROMs from tokenizer.json
 *   5. Write chip directory (tape.bin, slots.bin, const files, bpe ROMs, etc.)
 *
 * Only the weight download requires Python (kllm_download.py).
 * Everything in this file is pure C — no Python, no NumPy.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "tape_runner.h"

/* ================================================================
 * Config read from weights/config.json
 * ================================================================ */

typedef struct {
    int num_layers;
    int hidden_size;
    int num_heads;
    int num_kv_heads;
    int intermediate_size;
    int vocab_size;
    double rms_norm_eps;
    double rope_theta;
    int head_dim;        /* computed: hidden_size / num_heads */
    int num_groups;      /* computed: num_heads / num_kv_heads */
} ModelConfig;

/* ================================================================
 * File helpers (shared with kllm_infer.c but self-contained here)
 * ================================================================ */

static void *read_file_bin(const char *path, size_t *out_size) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    size_t sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *buf = malloc(sz);
    if (fread(buf, 1, sz, f) != sz) {
        fprintf(stderr, "Read error: %s\n", path); exit(1);
    }
    fclose(f);
    if (out_size) *out_size = sz;
    return buf;
}

static char *read_file_text(const char *path) {
    size_t sz;
    char *buf = (char *)read_file_bin(path, &sz);
    buf = (char *)realloc(buf, sz + 1);
    buf[sz] = '\0';
    return buf;
}

static char *path_join(const char *dir, const char *file) {
    size_t dlen = strlen(dir), flen = strlen(file);
    char *p = (char *)malloc(dlen + 1 + flen + 1);
    memcpy(p, dir, dlen);
    p[dlen] = '/';
    memcpy(p + dlen + 1, file, flen);
    p[dlen + 1 + flen] = '\0';
    return p;
}

static void mkdirs(const char *path) {
    char tmp[4096];
    snprintf(tmp, sizeof(tmp), "mkdir -p '%s'", path);
    system(tmp);
}

static void write_file_bin(const char *path, const void *data, size_t size) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot write %s\n", path); exit(1); }
    fwrite(data, 1, size, f);
    fclose(f);
}

static int file_exists(const char *path) {
    FILE *f = fopen(path, "rb");
    if (f) { fclose(f); return 1; }
    return 0;
}

/* ================================================================
 * Minimal JSON parser
 * ================================================================ */

static int json_get_int(const char *json, const char *key) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(json, pattern);
    if (!p) return -1;
    p += strlen(pattern);
    while (*p == ' ' || *p == ':' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    return atoi(p);
}

static double json_get_double(const char *json, const char *key) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(json, pattern);
    if (!p) return -1.0;
    p += strlen(pattern);
    while (*p == ' ' || *p == ':' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    return atof(p);
}

/* ================================================================
 * NPY file reader — reads NumPy .npy format
 *
 * Format: \x93NUMPY + major + minor + header_len + header_str + raw data
 * We only support float32 and little-endian.
 * ================================================================ */

typedef struct {
    float *data;
    int    shape[8];
    int    ndim;
    int    size;    /* total number of elements */
} NpyArray;

static NpyArray npy_load(const char *path) {
    NpyArray arr = {0};
    size_t file_sz;
    uint8_t *raw = (uint8_t *)read_file_bin(path, &file_sz);

    /* Verify magic */
    if (raw[0] != 0x93 || raw[1] != 'N' || raw[2] != 'U' ||
        raw[3] != 'M' || raw[4] != 'P' || raw[5] != 'Y') {
        fprintf(stderr, "Not a .npy file: %s\n", path);
        exit(1);
    }

    int major = raw[6];
    int header_len;
    int header_offset;

    if (major == 1) {
        header_len = *(uint16_t *)(raw + 8);
        header_offset = 10;
    } else {
        header_len = *(uint32_t *)(raw + 8);
        header_offset = 12;
    }

    char *header = (char *)malloc(header_len + 1);
    memcpy(header, raw + header_offset, header_len);
    header[header_len] = '\0';

    /* Parse shape from header: look for 'shape': (N, M, ...) */
    const char *sp = strstr(header, "'shape'");
    if (!sp) sp = strstr(header, "\"shape\"");
    if (!sp) { fprintf(stderr, "No shape in .npy header: %s\n", path); exit(1); }
    sp = strchr(sp, '(');
    if (!sp) { fprintf(stderr, "Bad shape in .npy header: %s\n", path); exit(1); }
    sp++;

    arr.ndim = 0;
    arr.size = 1;
    while (*sp && *sp != ')') {
        while (*sp == ' ' || *sp == ',') sp++;
        if (*sp == ')') break;
        int dim = atoi(sp);
        arr.shape[arr.ndim++] = dim;
        arr.size *= dim;
        while (*sp >= '0' && *sp <= '9') sp++;
    }

    /* Handle scalar: shape=() */
    if (arr.ndim == 0) {
        arr.ndim = 1;
        arr.shape[0] = 1;
        arr.size = 1;
    }

    free(header);

    /* Data starts after header */
    size_t data_offset = header_offset + header_len;
    arr.data = (float *)malloc(arr.size * sizeof(float));
    memcpy(arr.data, raw + data_offset, arr.size * sizeof(float));

    free(raw);
    return arr;
}

/* ================================================================
 * Tokenizer JSON parser — extract vocab + merges
 * ================================================================ */

/* Simple hash: FNV-1a 32-bit (must match Python _fnv1a) */
static uint32_t fnv1a(const uint8_t *data, int len) {
    uint32_t h = 2166136261u;
    for (int i = 0; i < len; i++) {
        h ^= data[i];
        h *= 16777619u;
    }
    return h;
}

/*
 * Vocab entry for building hash table.
 * We parse piece strings (possibly with unicode escapes) from tokenizer.json.
 */
typedef struct {
    uint8_t *bytes;   /* UTF-8 bytes of the piece */
    int      len;
    int      token_id;
} VocabEntry;

typedef struct {
    uint8_t *bytes;
    int      len;
    int      a_id;
    int      b_id;
    int      result_id;
} MergeEntry;

/*
 * Parse a JSON string (starting after opening quote) into raw bytes.
 * Handles \uXXXX escapes and \n, \t, \\, etc.
 * Returns length of decoded bytes. Writes into buf.
 */
static int json_decode_string(const char *src, uint8_t *buf, int buf_size) {
    int out = 0;
    while (*src && *src != '"' && out < buf_size - 1) {
        if (*src == '\\') {
            src++;
            switch (*src) {
                case 'n': buf[out++] = '\n'; src++; break;
                case 't': buf[out++] = '\t'; src++; break;
                case 'r': buf[out++] = '\r'; src++; break;
                case '\\': buf[out++] = '\\'; src++; break;
                case '"': buf[out++] = '"'; src++; break;
                case '/': buf[out++] = '/'; src++; break;
                case 'u': {
                    /* \uXXXX — decode as UTF-8 */
                    src++;
                    uint32_t cp = 0;
                    for (int i = 0; i < 4 && *src; i++, src++) {
                        cp <<= 4;
                        if (*src >= '0' && *src <= '9') cp |= *src - '0';
                        else if (*src >= 'a' && *src <= 'f') cp |= *src - 'a' + 10;
                        else if (*src >= 'A' && *src <= 'F') cp |= *src - 'A' + 10;
                    }
                    /* Check for surrogate pair \uD800-\uDBFF \uDC00-\uDFFF */
                    if (cp >= 0xD800 && cp <= 0xDBFF && src[0] == '\\' && src[1] == 'u') {
                        src += 2;
                        uint32_t lo = 0;
                        for (int i = 0; i < 4 && *src; i++, src++) {
                            lo <<= 4;
                            if (*src >= '0' && *src <= '9') lo |= *src - '0';
                            else if (*src >= 'a' && *src <= 'f') lo |= *src - 'a' + 10;
                            else if (*src >= 'A' && *src <= 'F') lo |= *src - 'A' + 10;
                        }
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    /* Encode as UTF-8 */
                    if (cp < 0x80) {
                        buf[out++] = (uint8_t)cp;
                    } else if (cp < 0x800) {
                        buf[out++] = 0xC0 | (cp >> 6);
                        buf[out++] = 0x80 | (cp & 0x3F);
                    } else if (cp < 0x10000) {
                        buf[out++] = 0xE0 | (cp >> 12);
                        buf[out++] = 0x80 | ((cp >> 6) & 0x3F);
                        buf[out++] = 0x80 | (cp & 0x3F);
                    } else {
                        buf[out++] = 0xF0 | (cp >> 18);
                        buf[out++] = 0x80 | ((cp >> 12) & 0x3F);
                        buf[out++] = 0x80 | ((cp >> 6) & 0x3F);
                        buf[out++] = 0x80 | (cp & 0x3F);
                    }
                    break;
                }
                default:
                    buf[out++] = *src; src++; break;
            }
        } else {
            buf[out++] = (uint8_t)*src;
            src++;
        }
    }
    return out;
}

/*
 * Parse tokenizer.json: extract vocab entries and merge rules.
 *
 * We look for:
 *   "model": { "vocab": { "piece": id, ... }, "merges": ["a b", ...] }
 *   "added_tokens": [{ "content": "...", "id": N }, ...]
 */

typedef struct {
    VocabEntry *vocab;
    int         vocab_count;
    int         vocab_size;    /* max(token_ids) + 1 */
    MergeEntry *merges;
    int         merge_count;
    int32_t    *special_ids;
    int         num_special;
    int         bos_token_id;
    int         eos_token_id;
} TokenizerData;

/*
 * Simple vocab parser: find "vocab": { ... } and parse key:value pairs.
 * This is simplistic but works for HuggingFace tokenizer.json format.
 */
static TokenizerData parse_tokenizer_json(const char *tok_path,
                                          const char *tok_cfg_path) {
    TokenizerData td = {0};

    char *json = read_file_text(tok_path);
    size_t json_len = strlen(json);
    (void)json_len;

    /* ---- Parse vocab ---- */
    /* Find "vocab": { */
    const char *vp = strstr(json, "\"vocab\"");
    if (!vp) { fprintf(stderr, "No vocab in tokenizer.json\n"); exit(1); }
    vp = strchr(vp, '{');
    if (!vp) { fprintf(stderr, "Bad vocab format\n"); exit(1); }
    vp++; /* skip opening brace */

    /* Count entries first (rough upper bound) */
    int vocab_cap = 64000;
    td.vocab = (VocabEntry *)calloc(vocab_cap, sizeof(VocabEntry));
    td.vocab_count = 0;
    td.vocab_size = 0;

    int brace_depth = 1;
    const char *p = vp;
    uint8_t piece_buf[512];

    while (*p && brace_depth > 0) {
        if (*p == '{') { brace_depth++; p++; continue; }
        if (*p == '}') { brace_depth--; p++; continue; }

        /* Look for "piece": id */
        if (*p == '"') {
            p++; /* skip opening quote */
            int piece_len = json_decode_string(p, piece_buf, sizeof(piece_buf));

            /* Skip to closing quote */
            while (*p && *p != '"') {
                if (*p == '\\') p++; /* skip escaped char */
                p++;
            }
            if (*p == '"') p++; /* skip closing quote */

            /* Skip colon + whitespace */
            while (*p && (*p == ' ' || *p == ':' || *p == '\t' ||
                          *p == '\n' || *p == '\r')) p++;

            /* Parse integer token ID */
            int id = atoi(p);
            while (*p >= '0' && *p <= '9') p++;

            if (td.vocab_count >= vocab_cap) {
                vocab_cap *= 2;
                td.vocab = (VocabEntry *)realloc(td.vocab, vocab_cap * sizeof(VocabEntry));
            }
            VocabEntry *e = &td.vocab[td.vocab_count++];
            e->bytes = (uint8_t *)malloc(piece_len);
            memcpy(e->bytes, piece_buf, piece_len);
            e->len = piece_len;
            e->token_id = id;
            if (id + 1 > td.vocab_size) td.vocab_size = id + 1;
        } else {
            p++;
        }
    }

    /* ---- Parse merges ---- */
    /* Handle both formats: ["a b", ...] and [["a", "b"], ...] */
    const char *mp = strstr(json, "\"merges\"");
    if (mp) {
        mp = strchr(mp, '[');
        if (mp) {
            mp++;
            int merge_cap = 64000;
            td.merges = (MergeEntry *)calloc(merge_cap, sizeof(MergeEntry));
            td.merge_count = 0;

            while (*mp && *mp != ']') {
                /* Skip whitespace and commas */
                while (*mp && (*mp == ' ' || *mp == '\t' || *mp == '\n' ||
                               *mp == '\r' || *mp == ',')) mp++;
                if (*mp == ']') break;

                if (*mp == '[') {
                    /* Array form: ["piece_a", "piece_b"] */
                    mp++; /* skip [ */
                    /* Parse piece_a */
                    while (*mp && *mp != '"') mp++;
                    if (*mp != '"') break;
                    mp++; /* skip opening quote */
                    uint8_t a_buf[256];
                    int a_len = json_decode_string(mp, a_buf, sizeof(a_buf));
                    while (*mp && *mp != '"') { if (*mp == '\\') mp++; mp++; }
                    if (*mp == '"') mp++;

                    /* Skip comma */
                    while (*mp && *mp != '"') mp++;
                    if (*mp != '"') break;
                    mp++; /* skip opening quote */
                    uint8_t b_buf[256];
                    int b_len = json_decode_string(mp, b_buf, sizeof(b_buf));
                    while (*mp && *mp != '"') { if (*mp == '\\') mp++; mp++; }
                    if (*mp == '"') mp++;

                    /* Skip to closing ] */
                    while (*mp && *mp != ']') mp++;
                    if (*mp == ']') mp++;

                    if (td.merge_count < merge_cap) {
                        MergeEntry *me = &td.merges[td.merge_count++];
                        me->bytes = (uint8_t *)malloc(a_len + 1 + b_len);
                        memcpy(me->bytes, a_buf, a_len);
                        me->bytes[a_len] = ' ';
                        memcpy(me->bytes + a_len + 1, b_buf, b_len);
                        me->len = a_len + 1 + b_len;
                        me->a_id = -1;
                        me->b_id = -1;
                        me->result_id = -1;
                    }
                } else if (*mp == '"') {
                    /* String form: "piece_a piece_b" */
                    mp++;
                    int entry_len = json_decode_string(mp, piece_buf, sizeof(piece_buf));

                    while (*mp && *mp != '"') {
                        if (*mp == '\\') mp++;
                        mp++;
                    }
                    if (*mp == '"') mp++;

                    int sp_pos = -1;
                    for (int i = 0; i < entry_len; i++) {
                        if (piece_buf[i] == ' ') { sp_pos = i; break; }
                    }
                    if (sp_pos > 0 && td.merge_count < merge_cap) {
                        MergeEntry *me = &td.merges[td.merge_count++];
                        me->bytes = (uint8_t *)malloc(entry_len);
                        memcpy(me->bytes, piece_buf, entry_len);
                        me->len = entry_len;
                        me->a_id = -1;
                        me->b_id = -1;
                        me->result_id = -1;
                    }
                } else {
                    mp++;
                }
            }
        }
    }

    /* ---- Parse added_tokens (special tokens) ---- */
    const char *atp = strstr(json, "\"added_tokens\"");
    int special_cap = 64;
    td.special_ids = (int32_t *)calloc(special_cap, sizeof(int32_t));
    td.num_special = 0;
    if (atp) {
        atp = strchr(atp, '[');
        if (atp) {
            atp++;
            while (*atp && *atp != ']') {
                const char *content_p = strstr(atp, "\"content\"");
                if (!content_p || content_p > strchr(atp, ']')) break;
                const char *id_p = strstr(atp, "\"id\"");
                if (!id_p) break;

                /* Get the id value */
                id_p += 4;
                while (*id_p && (*id_p == ' ' || *id_p == ':' ||
                                 *id_p == '\t')) id_p++;
                int sp_id = atoi(id_p);

                if (td.num_special < special_cap) {
                    td.special_ids[td.num_special++] = sp_id;
                }

                /* Move past this entry's closing brace */
                const char *next_brace = strchr(atp + 1, '}');
                if (next_brace) atp = next_brace + 1;
                else break;
            }
        }
    }

    /* Sort special_ids */
    for (int i = 0; i < td.num_special - 1; i++) {
        for (int j = i + 1; j < td.num_special; j++) {
            if (td.special_ids[j] < td.special_ids[i]) {
                int32_t tmp = td.special_ids[i];
                td.special_ids[i] = td.special_ids[j];
                td.special_ids[j] = tmp;
            }
        }
    }

    free(json);

    /* ---- Parse tokenizer_config.json for BOS/EOS ---- */
    if (file_exists(tok_cfg_path)) {
        char *cfg = read_file_text(tok_cfg_path);

        /* Get eos_token string and look up in vocab */
        const char *eos_p = strstr(cfg, "\"eos_token\"");
        td.eos_token_id = 2;  /* default */
        if (eos_p) {
            eos_p = strchr(eos_p + 11, '"');
            if (eos_p) {
                eos_p++;
                uint8_t eos_buf[64];
                int eos_len = json_decode_string(eos_p, eos_buf, sizeof(eos_buf));
                /* Look up in vocab */
                for (int i = 0; i < td.vocab_count; i++) {
                    if (td.vocab[i].len == eos_len &&
                        memcmp(td.vocab[i].bytes, eos_buf, eos_len) == 0) {
                        td.eos_token_id = td.vocab[i].token_id;
                        break;
                    }
                }
            }
        }

        /* BOS = second special token (LLaMA convention) */
        td.bos_token_id = (td.num_special > 1) ? td.special_ids[1] : 1;

        /* Try explicit bos_token lookup */
        const char *bos_p = strstr(cfg, "\"bos_token\"");
        if (bos_p) {
            bos_p = strchr(bos_p + 11, '"');
            if (bos_p) {
                bos_p++;
                uint8_t bos_buf[64];
                int bos_len = json_decode_string(bos_p, bos_buf, sizeof(bos_buf));
                for (int i = 0; i < td.vocab_count; i++) {
                    if (td.vocab[i].len == bos_len &&
                        memcmp(td.vocab[i].bytes, bos_buf, bos_len) == 0) {
                        td.bos_token_id = td.vocab[i].token_id;
                        break;
                    }
                }
            }
        }

        free(cfg);
    }

    /* ---- Resolve merge a_id, b_id, result_id ---- */
    /* Build a hash map from piece bytes → token ID for O(1) lookup */
    /* Use the same hash table approach we'll build for the chip */
    for (int mi = 0; mi < td.merge_count; mi++) {
        MergeEntry *me = &td.merges[mi];
        /* Find space separator */
        int sp = -1;
        for (int i = 0; i < me->len; i++) {
            if (me->bytes[i] == ' ') { sp = i; break; }
        }
        if (sp < 0) continue;

        uint8_t *a_bytes = me->bytes;
        int a_len = sp;
        uint8_t *b_bytes = me->bytes + sp + 1;
        int b_len = me->len - sp - 1;

        /* Concatenated result */
        uint8_t merged[512];
        memcpy(merged, a_bytes, a_len);
        memcpy(merged + a_len, b_bytes, b_len);
        int merged_len = a_len + b_len;

        /* Lookup a, b, result in vocab */
        for (int vi = 0; vi < td.vocab_count; vi++) {
            VocabEntry *ve = &td.vocab[vi];
            if (me->a_id < 0 && ve->len == a_len &&
                memcmp(ve->bytes, a_bytes, a_len) == 0)
                me->a_id = ve->token_id;
            if (me->b_id < 0 && ve->len == b_len &&
                memcmp(ve->bytes, b_bytes, b_len) == 0)
                me->b_id = ve->token_id;
            if (me->result_id < 0 && ve->len == merged_len &&
                memcmp(ve->bytes, merged, merged_len) == 0)
                me->result_id = ve->token_id;
            if (me->a_id >= 0 && me->b_id >= 0 && me->result_id >= 0)
                break;
        }
    }

    return td;
}

static void free_tokenizer_data(TokenizerData *td) {
    for (int i = 0; i < td->vocab_count; i++)
        free(td->vocab[i].bytes);
    free(td->vocab);
    for (int i = 0; i < td->merge_count; i++)
        free(td->merges[i].bytes);
    free(td->merges);
    free(td->special_ids);
}


/* ================================================================
 * BPE ROM builder — same as Python _compile_vocab_hash
 * ================================================================ */

typedef struct {
    uint8_t  *hash_keys;     /* [table_size * max_piece_len] */
    int32_t  *hash_vals;     /* [table_size] */
    int32_t  *hash_lens;     /* [table_size] */
    int       table_size;
    int       max_piece_len;
    /* Decode ROMs */
    uint8_t  *id_to_bytes;
    int32_t  *id_to_offsets;  /* [vocab_size * 2] — offset, length */
    int       id_to_bytes_len;
    /* Merge ROMs */
    int32_t  *merge_a;
    int32_t  *merge_b;
    int32_t  *merge_result;
    int       num_merges;
    /* Special tokens */
    int32_t  *special_ids;
    int       num_special;
} BpeRoms;

static BpeRoms build_bpe_roms(TokenizerData *td) {
    BpeRoms roms = {0};
    int max_piece_len = 64;
    roms.max_piece_len = max_piece_len;

    /* ---- Decode ROM: token_id → UTF-8 bytes ---- */
    int vocab_size = td->vocab_size;
    size_t all_bytes_cap = 1024 * 1024;
    uint8_t *all_bytes = (uint8_t *)calloc(all_bytes_cap, 1);
    int32_t *offsets = (int32_t *)calloc(vocab_size * 2, sizeof(int32_t));
    int all_bytes_len = 0;

    /* Build id_to_token mapping */
    uint8_t **id_to_token_bytes = (uint8_t **)calloc(vocab_size, sizeof(uint8_t *));
    int *id_to_token_len = (int *)calloc(vocab_size, sizeof(int));

    for (int i = 0; i < td->vocab_count; i++) {
        int id = td->vocab[i].token_id;
        if (id >= 0 && id < vocab_size) {
            id_to_token_bytes[id] = td->vocab[i].bytes;
            id_to_token_len[id] = td->vocab[i].len;
        }
    }

    for (int tid = 0; tid < vocab_size; tid++) {
        int tlen = id_to_token_len[tid];
        uint8_t *tbytes = id_to_token_bytes[tid];
        offsets[tid * 2] = all_bytes_len;
        offsets[tid * 2 + 1] = tlen;
        if (tlen > 0 && tbytes) {
            if (all_bytes_len + tlen > (int)all_bytes_cap) {
                all_bytes_cap *= 2;
                all_bytes = (uint8_t *)realloc(all_bytes, all_bytes_cap);
            }
            memcpy(all_bytes + all_bytes_len, tbytes, tlen);
            all_bytes_len += tlen;
        }
    }

    free(id_to_token_bytes);
    free(id_to_token_len);

    roms.id_to_bytes = all_bytes;
    roms.id_to_bytes_len = all_bytes_len;
    roms.id_to_offsets = offsets;

    /* ---- Merge ROMs ---- */
    roms.num_merges = td->merge_count;
    roms.merge_a = (int32_t *)malloc(td->merge_count * sizeof(int32_t));
    roms.merge_b = (int32_t *)malloc(td->merge_count * sizeof(int32_t));
    roms.merge_result = (int32_t *)malloc(td->merge_count * sizeof(int32_t));
    for (int i = 0; i < td->merge_count; i++) {
        roms.merge_a[i] = td->merges[i].a_id;
        roms.merge_b[i] = td->merges[i].b_id;
        roms.merge_result[i] = td->merges[i].result_id;
    }

    /* ---- Hash table (open addressing, linear probing) ---- */
    int n = td->vocab_count;
    int table_size = (int)(n / 0.6) + 1;
    /* Next power of 2 */
    int ts = 1;
    while (ts < table_size) ts <<= 1;
    table_size = ts;
    roms.table_size = table_size;

    roms.hash_keys = (uint8_t *)calloc(table_size * max_piece_len, 1);
    roms.hash_vals = (int32_t *)malloc(table_size * sizeof(int32_t));
    roms.hash_lens = (int32_t *)calloc(table_size, sizeof(int32_t));
    for (int i = 0; i < table_size; i++) roms.hash_vals[i] = -1;

    for (int i = 0; i < td->vocab_count; i++) {
        VocabEntry *e = &td->vocab[i];
        if (e->len > max_piece_len) continue;
        uint32_t h = fnv1a(e->bytes, e->len);
        int idx = (int)(h % (uint32_t)table_size);
        for (int probe = 0; probe < table_size; probe++) {
            if (roms.hash_lens[idx] == 0) {
                memcpy(roms.hash_keys + idx * max_piece_len, e->bytes, e->len);
                roms.hash_vals[idx] = e->token_id;
                roms.hash_lens[idx] = e->len;
                break;
            }
            idx = (idx + 1) % table_size;
        }
    }

    /* ---- Special tokens ---- */
    roms.num_special = td->num_special;
    roms.special_ids = (int32_t *)malloc(td->num_special * sizeof(int32_t));
    memcpy(roms.special_ids, td->special_ids, td->num_special * sizeof(int32_t));

    return roms;
}

static void free_bpe_roms(BpeRoms *roms) {
    free(roms->hash_keys);
    free(roms->hash_vals);
    free(roms->hash_lens);
    free(roms->id_to_bytes);
    free(roms->id_to_offsets);
    free(roms->merge_a);
    free(roms->merge_b);
    free(roms->merge_result);
    free(roms->special_ids);
}


/* ================================================================
 * Truth table compression — sort unique → codebook + uint16 indices
 * ================================================================ */

static int float_cmp(const void *a, const void *b) {
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    if (fa < fb) return -1;
    if (fa > fb) return 1;
    return 0;
}

typedef struct {
    float    *codebook;
    uint16_t *indices;
    int       n_codes;
    int       K, N;
} TruthTable;

/*
 * Compress a float32 matrix to truth table form.
 * Returns 1 on success (≤65536 unique values), 0 on failure.
 */
static int tt_compress(const float *data, int K, int N, TruthTable *out) {
    int total = K * N;
    float *sorted = (float *)malloc(total * sizeof(float));
    memcpy(sorted, data, total * sizeof(float));
    qsort(sorted, total, sizeof(float), float_cmp);

    /* Deduplicate */
    int n_unique = 0;
    float *codebook = (float *)malloc(total * sizeof(float));
    codebook[0] = sorted[0];
    n_unique = 1;
    for (int i = 1; i < total; i++) {
        if (sorted[i] != codebook[n_unique - 1]) {
            codebook[n_unique++] = sorted[i];
        }
    }
    free(sorted);

    if (n_unique > 65536) {
        free(codebook);
        return 0;
    }

    codebook = (float *)realloc(codebook, n_unique * sizeof(float));

    /* Build index mapping (binary search) */
    uint16_t *indices = (uint16_t *)malloc(total * sizeof(uint16_t));
    for (int i = 0; i < total; i++) {
        /* Binary search in codebook */
        int lo = 0, hi = n_unique - 1;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            if (codebook[mid] == data[i]) { indices[i] = (uint16_t)mid; break; }
            else if (codebook[mid] < data[i]) lo = mid + 1;
            else hi = mid - 1;
        }
    }

    out->codebook = codebook;
    out->indices = indices;
    out->n_codes = n_unique;
    out->K = K;
    out->N = N;
    return 1;
}


/* ================================================================
 * RoPE precomputation
 * ================================================================ */

static void build_rope(int max_seq, int head_dim, double theta,
                       float **out_cos, float **out_sin) {
    int half = head_dim / 2;
    float *rope_cos = (float *)malloc(max_seq * head_dim * sizeof(float));
    float *rope_sin = (float *)malloc(max_seq * head_dim * sizeof(float));

    float *inv_freq = (float *)malloc(half * sizeof(float));
    for (int i = 0; i < half; i++) {
        inv_freq[i] = 1.0f / powf((float)theta, (float)(i * 2) / head_dim);
    }

    for (int t = 0; t < max_seq; t++) {
        for (int i = 0; i < half; i++) {
            float angle = t * inv_freq[i];
            float c = cosf(angle);
            float s = sinf(angle);
            rope_cos[t * head_dim + i] = c;
            rope_cos[t * head_dim + half + i] = c;
            rope_sin[t * head_dim + i] = s;
            rope_sin[t * head_dim + half + i] = s;
        }
    }
    free(inv_freq);

    *out_cos = rope_cos;
    *out_sin = rope_sin;
}


/* ================================================================
 * Tape compiler — builds TapeInstr tape directly from model config
 *
 * Tags (must match _tape_runner.c):
 *   0=SILU  1=EXP  2=RSQRT  3=COS  4=SIN
 *   5=ADD   6=SUB  7=MUL    8=DIV  9=MAX
 *   10=NEG  11=SQUARE  12=MATMUL  13=MATMUL_TT
 *   14=SUM  15=MAX_RED  16=MEAN  17=ARGMAX
 *   18=RESHAPE  19=TRANSPOSE  20=CONCAT  21=REPEAT
 *   22=SLICE  23=COPY
 * ================================================================ */

enum {
    T_LUT_SILU  = 0,  T_LUT_EXP   = 1,  T_LUT_RSQRT = 2,
    T_LUT_COS   = 3,  T_LUT_SIN   = 4,
    T_ADD = 5, T_SUB = 6, T_MUL = 7, T_DIV = 8, T_MAX = 9,
    T_NEG = 10, T_SQUARE = 11,
    T_MATMUL = 12, T_MATMUL_TT = 13,
    T_SUM = 14, T_MAX_RED = 15, T_MEAN = 16, T_ARGMAX = 17,
    T_RESHAPE = 18, T_TRANSPOSE = 19, T_CONCAT = 20, T_REPEAT = 21,
    T_SLICE = 22, T_COPY = 23,
};

/*
 * The tape builder uses a slot allocator. Each "node" gets a slot ID.
 * CONST slots hold weight data. INPUT slots hold variable data.
 * Intermediate slots are reused across the tape.
 */

typedef struct {
    /* Slot tracking */
    int next_slot;

    /* Instruction array */
    TapeInstr *instrs;
    int n_instrs;
    int instrs_cap;

    /* Slot metadata for serialization */
    int  *slot_type;    /* 0=unused, 1=const, 2=input, 3=intermediate, 4=tt */
    int (*slot_shape)[MAX_DIMS];
    int  *slot_ndim;

    /* Weight data ownership */
    float    **const_data;    /* [slot] → data pointer (owned) */
    TruthTable *tt_data;      /* [slot] → TT data (for TT slots) */
    int         const_cap;

    ModelConfig cfg;
} TapeBuilder;

static void tb_init(TapeBuilder *tb, ModelConfig cfg, int slot_cap) {
    memset(tb, 0, sizeof(*tb));
    tb->cfg = cfg;
    tb->next_slot = 0;
    tb->instrs_cap = 4096;
    tb->instrs = (TapeInstr *)calloc(tb->instrs_cap, sizeof(TapeInstr));
    tb->n_instrs = 0;

    tb->slot_type = (int *)calloc(slot_cap, sizeof(int));
    tb->slot_shape = calloc(slot_cap, sizeof(int[MAX_DIMS]));
    tb->slot_ndim = (int *)calloc(slot_cap, sizeof(int));
    tb->const_data = (float **)calloc(slot_cap, sizeof(float *));
    tb->tt_data = (TruthTable *)calloc(slot_cap, sizeof(TruthTable));
    tb->const_cap = slot_cap;
}

static int tb_alloc_slot(TapeBuilder *tb, int type,
                         const int *shape, int ndim) {
    int s = tb->next_slot++;
    tb->slot_type[s] = type;
    tb->slot_ndim[s] = ndim;
    for (int d = 0; d < ndim && d < MAX_DIMS; d++)
        tb->slot_shape[s][d] = shape[d];
    return s;
}

/* Allocate a const slot and take ownership of data */
static int tb_const(TapeBuilder *tb, float *data, const int *shape, int ndim) {
    int s = tb_alloc_slot(tb, 1, shape, ndim);
    tb->const_data[s] = data;
    return s;
}

/* Allocate a TT const slot */
static int tb_const_tt(TapeBuilder *tb, TruthTable *tt) {
    int shape[2] = {tt->K, tt->N};
    int s = tb_alloc_slot(tb, 4, shape, 2);
    tb->tt_data[s] = *tt;
    return s;
}

/* Allocate an input slot */
static int tb_input(TapeBuilder *tb, const int *shape, int ndim) {
    return tb_alloc_slot(tb, 2, shape, ndim);
}

/* Compute output shape and allocate intermediate slot */
static void tb_compute_size(TapeInstr *instr) {
    instr->out_size = 1;
    for (int d = 0; d < instr->out_ndim; d++)
        instr->out_size *= instr->out_shape[d];
}

/* Emit an instruction */
static TapeInstr *tb_emit(TapeBuilder *tb) {
    if (tb->n_instrs >= tb->instrs_cap) {
        tb->instrs_cap *= 2;
        tb->instrs = (TapeInstr *)realloc(tb->instrs,
            tb->instrs_cap * sizeof(TapeInstr));
    }
    TapeInstr *instr = &tb->instrs[tb->n_instrs++];
    memset(instr, 0, sizeof(*instr));
    return instr;
}

/* ---- Helper: emit binop ---- */
static int tb_binop(TapeBuilder *tb, int tag, int a, int b,
                    const int *shape, int ndim) {
    int out = tb_alloc_slot(tb, 3, shape, ndim);
    TapeInstr *i = tb_emit(tb);
    i->tag = tag;
    i->out_slot = out;
    i->in0 = a;
    i->in1 = b;
    i->out_ndim = ndim;
    for (int d = 0; d < ndim; d++) i->out_shape[d] = shape[d];
    tb_compute_size(i);
    return out;
}

/* ---- Helper: emit unary op ---- */
static int tb_unary(TapeBuilder *tb, int tag, int a,
                    const int *shape, int ndim) {
    int out = tb_alloc_slot(tb, 3, shape, ndim);
    TapeInstr *i = tb_emit(tb);
    i->tag = tag;
    i->out_slot = out;
    i->in0 = a;
    i->out_ndim = ndim;
    for (int d = 0; d < ndim; d++) i->out_shape[d] = shape[d];
    tb_compute_size(i);
    return out;
}

/* ---- Helper: emit matmul ---- */
static int tb_matmul(TapeBuilder *tb, int a, int b,
                     const int *out_shape, int out_ndim) {
    int out = tb_alloc_slot(tb, 3, out_shape, out_ndim);
    TapeInstr *i = tb_emit(tb);
    i->tag = T_MATMUL;
    i->out_slot = out;
    i->in0 = a;
    i->in1 = b;
    return out;
}

/* ---- Helper: emit matmul_tt ---- */
static int tb_matmul_tt(TapeBuilder *tb, int a, int b,
                        const int *out_shape, int out_ndim) {
    int out = tb_alloc_slot(tb, 3, out_shape, out_ndim);
    TapeInstr *i = tb_emit(tb);
    i->tag = T_MATMUL_TT;
    i->out_slot = out;
    i->in0 = a;
    i->in1 = b;
    return out;
}

/* ---- Helper: emit reshape ---- */
static int tb_reshape(TapeBuilder *tb, int a, const int *shape, int ndim) {
    int out = tb_alloc_slot(tb, 3, shape, ndim);
    TapeInstr *i = tb_emit(tb);
    i->tag = T_RESHAPE;
    i->out_slot = out;
    i->in0 = a;
    i->out_ndim = ndim;
    for (int d = 0; d < ndim; d++) i->out_shape[d] = shape[d];
    tb_compute_size(i);
    return out;
}

/* ---- Helper: emit transpose ---- */
static int tb_transpose(TapeBuilder *tb, int a, const int *axes,
                        const int *out_shape, int out_ndim) {
    int out = tb_alloc_slot(tb, 3, out_shape, out_ndim);
    TapeInstr *i = tb_emit(tb);
    i->tag = T_TRANSPOSE;
    i->out_slot = out;
    i->in0 = a;
    for (int d = 0; d < out_ndim; d++) i->axes[d] = axes[d];
    return out;
}

/* ---- Helper: emit concat ---- */
static int tb_concat(TapeBuilder *tb, const int *inputs, int n_inputs,
                     int axis, const int *out_shape, int out_ndim) {
    int out = tb_alloc_slot(tb, 3, out_shape, out_ndim);
    TapeInstr *i = tb_emit(tb);
    i->tag = T_CONCAT;
    i->out_slot = out;
    i->in0 = inputs[0]; /* not used directly, concat reads concat_inputs */
    i->axis = axis;
    i->concat_n_inputs = n_inputs;
    for (int d = 0; d < n_inputs && d < MAX_CONCAT_INPUTS; d++)
        i->concat_inputs[d] = inputs[d];
    return out;
}

/* ---- Helper: emit repeat ---- */
static int tb_repeat(TapeBuilder *tb, int a, int repeats, int axis,
                     const int *out_shape, int out_ndim) {
    int out = tb_alloc_slot(tb, 3, out_shape, out_ndim);
    TapeInstr *i = tb_emit(tb);
    i->tag = T_REPEAT;
    i->out_slot = out;
    i->in0 = a;
    i->repeats = repeats;
    i->axis = axis;
    return out;
}

/* ---- Helper: emit slice ---- */
static int tb_slice(TapeBuilder *tb, int a, int ndim,
                    const int *starts, const int *stops, const int *steps,
                    const int *out_shape, int out_ndim) {
    int out = tb_alloc_slot(tb, 3, out_shape, out_ndim);
    TapeInstr *i = tb_emit(tb);
    i->tag = T_SLICE;
    i->out_slot = out;
    i->in0 = a;
    for (int d = 0; d < ndim; d++) {
        i->starts[d] = starts[d];
        i->steps[d] = steps[d];
        i->slice_out_shape[d] = stops[d] - starts[d];
    }
    i->slice_out_ndim = ndim;
    return out;
}

/* ---- Helper: emit expand_dims (copy with new shape) ---- */
static int tb_expand_dims(TapeBuilder *tb, int a,
                          const int *out_shape, int out_ndim) {
    int out = tb_alloc_slot(tb, 3, out_shape, out_ndim);
    TapeInstr *i = tb_emit(tb);
    i->tag = T_COPY;
    i->out_slot = out;
    i->in0 = a;
    i->out_ndim = out_ndim;
    for (int d = 0; d < out_ndim; d++) i->out_shape[d] = out_shape[d];
    tb_compute_size(i);
    return out;
}

/* ---- Helper: emit reduce ---- */
static int tb_reduce(TapeBuilder *tb, int tag, int a, int axis,
                     int keepdims, const int *out_shape, int out_ndim) {
    int out = tb_alloc_slot(tb, 3, out_shape, out_ndim);
    TapeInstr *i = tb_emit(tb);
    i->tag = tag;
    i->out_slot = out;
    i->in0 = a;
    i->axis = axis;
    i->keepdims = keepdims;
    return out;
}

/* ---- Helper: emit argmax ---- */
static int tb_argmax(TapeBuilder *tb, int a, int axis,
                     const int *out_shape, int out_ndim) {
    int out = tb_alloc_slot(tb, 3, out_shape, out_ndim);
    TapeInstr *i = tb_emit(tb);
    i->tag = T_ARGMAX;
    i->out_slot = out;
    i->in0 = a;
    i->axis = axis;
    return out;
}


/* ================================================================
 * Build RMSNorm subgraph
 *   hidden = (hidden / sqrt(mean(hidden^2) + eps)) * weight
 * ================================================================ */

static int tb_rms_norm(TapeBuilder *tb, int hidden, int weight_slot,
                       double eps, int seq_len, int hidden_size) {
    int shape_sh[] = {seq_len, hidden_size};
    int shape_s1[] = {seq_len, 1};

    /* squared = hidden * hidden */
    int sq = tb_binop(tb, T_MUL, hidden, hidden, shape_sh, 2);
    /* mean_sq = mean(squared, axis=-1, keepdims=True) */
    int mean_sq = tb_reduce(tb, T_MEAN, sq, 1, 1, shape_s1, 2);
    /* Add eps constant */
    float *eps_val = (float *)malloc(sizeof(float));
    *eps_val = (float)eps;
    int eps_shape[] = {1};
    int eps_slot = tb_const(tb, eps_val, eps_shape, 1);
    int added = tb_binop(tb, T_ADD, mean_sq, eps_slot, shape_s1, 2);
    /* rsqrt */
    int rsq = tb_unary(tb, T_LUT_RSQRT, added, shape_s1, 2);
    /* multiply */
    int normed = tb_binop(tb, T_MUL, hidden, rsq, shape_sh, 2);
    /* scale by weight */
    return tb_binop(tb, T_MUL, normed, weight_slot, shape_sh, 2);
}


/* ================================================================
 * Build RoPE subgraph
 *   x * cos + rotate(x) * sin
 * ================================================================ */

static int tb_rope(TapeBuilder *tb, int x, int cos_slot, int sin_slot,
                   int num_heads, int head_dim) {
    int half = head_dim / 2;
    /* x is (num_heads, 1, head_dim) */
    int shape3[] = {num_heads, 1, head_dim};
    int shape3h[] = {num_heads, 1, half};

    /* x1 = x[:, :, :half] */
    int starts_x1[] = {0, 0, 0};
    int stops_x1[] = {num_heads, 1, half};
    int steps_x1[] = {1, 1, 1};
    int x1 = tb_slice(tb, x, 3, starts_x1, stops_x1, steps_x1, shape3h, 3);

    /* x2 = x[:, :, half:] */
    int starts_x2[] = {0, 0, half};
    int stops_x2[] = {num_heads, 1, head_dim};
    int steps_x2[] = {1, 1, 1};
    int x2 = tb_slice(tb, x, 3, starts_x2, stops_x2, steps_x2, shape3h, 3);

    /* neg_x2 = -x2 */
    int neg_x2 = tb_unary(tb, T_NEG, x2, shape3h, 3);

    /* rotated = concat([-x2, x1], axis=-1) */
    int cat_ins[] = {neg_x2, x1};
    int rotated = tb_concat(tb, cat_ins, 2, 2, shape3, 3);

    /* x * cos + rotated * sin */
    /* cos/sin are (1, 1, head_dim) — broadcast */
    int xcos = tb_binop(tb, T_MUL, x, cos_slot, shape3, 3);
    int rsin = tb_binop(tb, T_MUL, rotated, sin_slot, shape3, 3);
    return tb_binop(tb, T_ADD, xcos, rsin, shape3, 3);
}


/* ================================================================
 * Build attention subgraph for one layer
 * ================================================================ */

typedef struct {
    int output;
    int new_k;
    int new_v;
} AttnResult;

static AttnResult tb_attention(
    TapeBuilder *tb, int hidden,
    int qkv_slot, int o_proj_slot,
    int cos_b_slot, int sin_b_slot, int scale_slot,
    int kv_cache_k, int kv_cache_v,
    int is_tt_qkv, int is_tt_o)
{
    ModelConfig *c = &tb->cfg;
    int num_heads = c->num_heads;
    int num_kv = c->num_kv_heads;
    int hd = c->head_dim;
    int groups = c->num_groups;
    int q_dim = num_heads * hd;
    int kv_dim = num_kv * hd;

    /* QKV projection: hidden @ qkv_t → (1, q_dim + 2*kv_dim) */
    int qkv_out_shape[] = {1, q_dim + 2 * kv_dim};
    int qkv;
    if (is_tt_qkv)
        qkv = tb_matmul_tt(tb, hidden, qkv_slot, qkv_out_shape, 2);
    else
        qkv = tb_matmul(tb, hidden, qkv_slot, qkv_out_shape, 2);

    /* Split Q, K, V */
    int q_shape[] = {1, q_dim};
    int starts_q[] = {0, 0}, stops_q[] = {1, q_dim}, steps_1[] = {1, 1};
    int q = tb_slice(tb, qkv, 2, starts_q, stops_q, steps_1, q_shape, 2);

    int k_shape[] = {1, kv_dim};
    int starts_k[] = {0, q_dim}, stops_k[] = {1, q_dim + kv_dim};
    int k = tb_slice(tb, qkv, 2, starts_k, stops_k, steps_1, k_shape, 2);

    int v_shape[] = {1, kv_dim};
    int starts_v[] = {0, q_dim + kv_dim}, stops_v[] = {1, q_dim + 2*kv_dim};
    int v = tb_slice(tb, qkv, 2, starts_v, stops_v, steps_1, v_shape, 2);

    /* Reshape Q: (1, q_dim) → (1, nH, hd) → transpose → (nH, 1, hd) */
    int q_r_shape[] = {1, num_heads, hd};
    q = tb_reshape(tb, q, q_r_shape, 3);
    int axes_102[] = {1, 0, 2};
    int q_t_shape[] = {num_heads, 1, hd};
    q = tb_transpose(tb, q, axes_102, q_t_shape, 3);

    /* Reshape K: (1, kv_dim) → (1, nKV, hd) → transpose → (nKV, 1, hd) */
    int k_r_shape[] = {1, num_kv, hd};
    k = tb_reshape(tb, k, k_r_shape, 3);
    int k_t_shape[] = {num_kv, 1, hd};
    k = tb_transpose(tb, k, axes_102, k_t_shape, 3);

    /* Reshape V similarly */
    int v_r_shape[] = {1, num_kv, hd};
    v = tb_reshape(tb, v, v_r_shape, 3);
    int v_t_shape[] = {num_kv, 1, hd};
    v = tb_transpose(tb, v, axes_102, v_t_shape, 3);

    /* RoPE on Q and K */
    q = tb_rope(tb, q, cos_b_slot, sin_b_slot, num_heads, hd);
    k = tb_rope(tb, k, cos_b_slot, sin_b_slot, num_kv, hd);

    /* KV cache concat: cat(cache, new) along axis=1 */
    /* Output seq dim is dynamic at runtime — C handles this.
     * For shape tracking we use placeholder seq=2 but the concat
     * instruction uses axis=1 and the C runner computes output size
     * from the actual slot shapes at runtime. */
    int cat_k_ins[] = {kv_cache_k, k};
    int cat_k_shape[] = {num_kv, 2, hd};  /* placeholder */
    int k_full = tb_concat(tb, cat_k_ins, 2, 1, cat_k_shape, 3);

    int cat_v_ins[] = {kv_cache_v, v};
    int cat_v_shape[] = {num_kv, 2, hd};
    int v_full = tb_concat(tb, cat_v_ins, 2, 1, cat_v_shape, 3);

    AttnResult res;
    res.new_k = k_full;
    res.new_v = v_full;

    /* GQA expansion */
    int k_exp, v_exp;
    if (groups > 1) {
        int gqa_k_shape[] = {num_heads, 2, hd};
        k_exp = tb_repeat(tb, k_full, groups, 0, gqa_k_shape, 3);
        int gqa_v_shape[] = {num_heads, 2, hd};
        v_exp = tb_repeat(tb, v_full, groups, 0, gqa_v_shape, 3);
    } else {
        k_exp = k_full;
        v_exp = v_full;
    }

    /* Attention scores: Q @ K^T */
    int axes_021[] = {0, 2, 1};
    int kt_shape[] = {num_heads, hd, 2};
    int k_trans = tb_transpose(tb, k_exp, axes_021, kt_shape, 3);

    int scores_shape[] = {num_heads, 1, 2};
    int scores = tb_matmul(tb, q, k_trans, scores_shape, 3);

    /* Scale by 1/sqrt(head_dim) */
    scores = tb_binop(tb, T_MUL, scores, scale_slot, scores_shape, 3);

    /* No causal mask for seq_len=1 decode */

    /* Softmax: exp(x - max(x)) / sum(exp(x - max(x))) */
    int max_shape[] = {num_heads, 1, 1};
    int mx = tb_reduce(tb, T_MAX_RED, scores, 2, 1, max_shape, 3);
    int sub = tb_binop(tb, T_SUB, scores, mx, scores_shape, 3);
    int ex = tb_unary(tb, T_LUT_EXP, sub, scores_shape, 3);
    int sm = tb_reduce(tb, T_SUM, ex, 2, 1, max_shape, 3);
    int attn_w = tb_binop(tb, T_DIV, ex, sm, scores_shape, 3);

    /* Context: attn @ V → (nH, 1, hd) */
    int ctx_shape[] = {num_heads, 1, hd};
    int ctx = tb_matmul(tb, attn_w, v_exp, ctx_shape, 3);

    /* Reshape: (nH, 1, hd) → (1, nH, hd) → (1, q_dim) */
    int ctx_t_shape[] = {1, num_heads, hd};
    ctx = tb_transpose(tb, ctx, axes_102, ctx_t_shape, 3);
    int ctx_flat_shape[] = {1, q_dim};
    ctx = tb_reshape(tb, ctx, ctx_flat_shape, 2);

    /* Output projection */
    int out_shape[] = {1, num_heads * hd};
    if (is_tt_o)
        res.output = tb_matmul_tt(tb, ctx, o_proj_slot, out_shape, 2);
    else
        res.output = tb_matmul(tb, ctx, o_proj_slot, out_shape, 2);

    return res;
}


/* ================================================================
 * Build MLP subgraph for one layer
 * ================================================================ */

static int tb_mlp(TapeBuilder *tb, int hidden,
                  int gate_up_slot, int down_slot,
                  int is_tt_gu, int is_tt_down) {
    ModelConfig *c = &tb->cfg;
    int inter = c->intermediate_size;

    /* gate+up projection: hidden @ gate_up_t → (1, 2*inter) */
    int gu_shape[] = {1, 2 * inter};
    int gu;
    if (is_tt_gu)
        gu = tb_matmul_tt(tb, hidden, gate_up_slot, gu_shape, 2);
    else
        gu = tb_matmul(tb, hidden, gate_up_slot, gu_shape, 2);

    /* Split gate and up */
    int g_shape[] = {1, inter};
    int starts_g[] = {0, 0}, stops_g[] = {1, inter}, steps[] = {1, 1};
    int gate = tb_slice(tb, gu, 2, starts_g, stops_g, steps, g_shape, 2);

    int u_shape[] = {1, inter};
    int starts_u[] = {0, inter}, stops_u[] = {1, 2*inter};
    int up = tb_slice(tb, gu, 2, starts_u, stops_u, steps, u_shape, 2);

    /* SiLU on gate */
    int gate_act = tb_unary(tb, T_LUT_SILU, gate, g_shape, 2);

    /* gate * up */
    int gated = tb_binop(tb, T_MUL, gate_act, up, g_shape, 2);

    /* down projection */
    int hidden_shape[] = {1, c->hidden_size};
    if (is_tt_down)
        return tb_matmul_tt(tb, gated, down_slot, hidden_shape, 2);
    else
        return tb_matmul(tb, gated, down_slot, hidden_shape, 2);
}


/* ================================================================
 * Main compile entry point
 * ================================================================ */

int compile_chip(const char *chip_dir, int max_seq, const char *model_name) {
    char *weights_dir = path_join(chip_dir, "weights");
    char *circuit_dir = path_join(chip_dir, "circuit");
    char *bpe_dir = path_join(chip_dir, "bpe");
    char *tables_dir = path_join(chip_dir, "tables");
    char *tok_dir = path_join(chip_dir, "tokenizer");

    mkdirs(circuit_dir);
    mkdirs(bpe_dir);
    mkdirs(tables_dir);

    /* ---- 1. Read config ---- */
    char *cfg_path = path_join(weights_dir, "config.json");
    char *cfg_json = read_file_text(cfg_path);
    free(cfg_path);

    ModelConfig cfg;
    cfg.num_layers = json_get_int(cfg_json, "num_layers");
    cfg.hidden_size = json_get_int(cfg_json, "hidden_size");
    cfg.num_heads = json_get_int(cfg_json, "num_attention_heads");
    cfg.num_kv_heads = json_get_int(cfg_json, "num_key_value_heads");
    cfg.intermediate_size = json_get_int(cfg_json, "intermediate_size");
    cfg.vocab_size = json_get_int(cfg_json, "vocab_size");
    cfg.rms_norm_eps = json_get_double(cfg_json, "rms_norm_eps");
    cfg.rope_theta = json_get_double(cfg_json, "rope_theta");
    cfg.head_dim = cfg.hidden_size / cfg.num_heads;
    cfg.num_groups = cfg.num_heads / cfg.num_kv_heads;
    free(cfg_json);

    printf("[compile] Model: %d layers, hidden=%d, heads=%d/%d, inter=%d, vocab=%d\n",
           cfg.num_layers, cfg.hidden_size, cfg.num_heads, cfg.num_kv_heads,
           cfg.intermediate_size, cfg.vocab_size);

    /* ---- 2. Compile tokenizer ---- */
    printf("[compile] Building BPE ROMs …\n");
    char *tok_json_path = path_join(tok_dir, "tokenizer.json");
    char *tok_cfg_path = path_join(tok_dir, "tokenizer_config.json");
    TokenizerData tok = parse_tokenizer_json(tok_json_path, tok_cfg_path);
    free(tok_json_path);
    free(tok_cfg_path);

    BpeRoms bpe = build_bpe_roms(&tok);

    printf("[compile] Vocab: %d tokens, %d merges, %d special\n",
           tok.vocab_size, tok.merge_count, tok.num_special);

    /* ---- 3. Load weights and build tape ---- */
    printf("[compile] Loading weights and building tape …\n");

    /* Estimate slot count: per layer ~30 weight slots + ~70 intermediates */
    int slot_cap = cfg.num_layers * 120 + 200;
    TapeBuilder tb;
    tb_init(&tb, cfg, slot_cap);

    /* INPUT slots — same names as Python compile_decode_template */
    int hidden_shape[] = {1, cfg.hidden_size};
    int input_token_embed = tb_input(&tb, hidden_shape, 2);

    int rope_shape[] = {1, cfg.head_dim};
    int input_rope_cos = tb_input(&tb, rope_shape, 2);
    int input_rope_sin = tb_input(&tb, rope_shape, 2);

    /* Per-layer KV cache inputs */
    int *kv_in_slots = (int *)malloc(cfg.num_layers * 2 * sizeof(int));
    int kv_cache_shape[] = {cfg.num_kv_heads, 1, cfg.head_dim};
    for (int li = 0; li < cfg.num_layers; li++) {
        kv_in_slots[li * 2] = tb_input(&tb, kv_cache_shape, 3);
        kv_in_slots[li * 2 + 1] = tb_input(&tb, kv_cache_shape, 3);
    }

    /* Fixed CONST: attention scale */
    float *scale_val = (float *)malloc(sizeof(float));
    *scale_val = 1.0f / sqrtf((float)cfg.head_dim);
    int scale_shape[] = {1};
    int scale_slot = tb_const(&tb, scale_val, scale_shape, 1);

    /* cos/sin for expand_dims(input, axis=0) → (1, 1, head_dim) for broadcast */
    int cos_b_shape[] = {1, 1, cfg.head_dim};
    int cos_b = tb_expand_dims(&tb, input_rope_cos, cos_b_shape, 3);
    int sin_b = tb_expand_dims(&tb, input_rope_sin, cos_b_shape, 3);

    /* KV output IDs for outputs */
    int *kv_out_slots = (int *)malloc(cfg.num_layers * 2 * sizeof(int));

    int hidden = input_token_embed;

    /* ---- Per-layer weights + subgraphs ---- */
    int n_tt = 0;
    for (int li = 0; li < cfg.num_layers; li++) {
        char layer_dir[256];
        snprintf(layer_dir, sizeof(layer_dir), "%s/layer_%d", weights_dir, li);

        /* Load weight arrays */
        char wpath[512];

        /* Fused Q/K/V: [Wq | Wk | Wv].T → (hidden, q_dim+2*kv_dim) */
        snprintf(wpath, sizeof(wpath), "%s/q_proj.npy", layer_dir);
        NpyArray wq = npy_load(wpath);
        snprintf(wpath, sizeof(wpath), "%s/k_proj.npy", layer_dir);
        NpyArray wk = npy_load(wpath);
        snprintf(wpath, sizeof(wpath), "%s/v_proj.npy", layer_dir);
        NpyArray wv = npy_load(wpath);

        int q_dim = cfg.num_heads * cfg.head_dim;
        int kv_dim = cfg.num_kv_heads * cfg.head_dim;
        int fused_cols = q_dim + 2 * kv_dim;
        int H = cfg.hidden_size;

        /* Transpose: (out, in) → (in, out), then concatenate */
        float *qkv_t = (float *)malloc(H * fused_cols * sizeof(float));
        /* Wq: (q_dim, H) → col 0..q_dim-1 of qkv_t (H, fused_cols) */
        for (int r = 0; r < H; r++) {
            for (int c = 0; c < q_dim; c++)
                qkv_t[r * fused_cols + c] = wq.data[c * H + r];
            for (int c = 0; c < kv_dim; c++)
                qkv_t[r * fused_cols + q_dim + c] = wk.data[c * H + r];
            for (int c = 0; c < kv_dim; c++)
                qkv_t[r * fused_cols + q_dim + kv_dim + c] = wv.data[c * H + r];
        }
        free(wq.data); free(wk.data); free(wv.data);

        /* TT compress QKV */
        TruthTable tt_qkv;
        int qkv_slot, is_tt_qkv = 0;
        if (tt_compress(qkv_t, H, fused_cols, &tt_qkv)) {
            qkv_slot = tb_const_tt(&tb, &tt_qkv);
            free(qkv_t);
            is_tt_qkv = 1;
            n_tt++;
        } else {
            int qkv_shape[] = {H, fused_cols};
            qkv_slot = tb_const(&tb, qkv_t, qkv_shape, 2);
        }

        /* O projection: (q_dim, H) → transpose → (H, q_dim) */
        snprintf(wpath, sizeof(wpath), "%s/o_proj.npy", layer_dir);
        NpyArray wo = npy_load(wpath);
        float *o_t = (float *)malloc(q_dim * H * sizeof(float));
        for (int r = 0; r < H; r++)
            for (int c = 0; c < q_dim; c++)
                o_t[r * q_dim + c] = wo.data[c * H + r];
        free(wo.data);

        TruthTable tt_o;
        int o_slot, is_tt_o = 0;
        if (tt_compress(o_t, H, q_dim, &tt_o)) {
            o_slot = tb_const_tt(&tb, &tt_o);
            free(o_t);
            is_tt_o = 1;
            n_tt++;
        } else {
            int o_shape[] = {H, q_dim};
            o_slot = tb_const(&tb, o_t, o_shape, 2);
        }

        /* Fused gate+up: [Wgate | Wup].T → (H, 2*inter) */
        snprintf(wpath, sizeof(wpath), "%s/gate_proj.npy", layer_dir);
        NpyArray wg = npy_load(wpath);
        snprintf(wpath, sizeof(wpath), "%s/up_proj.npy", layer_dir);
        NpyArray wu = npy_load(wpath);

        int inter = cfg.intermediate_size;
        float *gu_t = (float *)malloc(H * 2 * inter * sizeof(float));
        for (int r = 0; r < H; r++) {
            for (int c = 0; c < inter; c++)
                gu_t[r * 2*inter + c] = wg.data[c * H + r];
            for (int c = 0; c < inter; c++)
                gu_t[r * 2*inter + inter + c] = wu.data[c * H + r];
        }
        free(wg.data); free(wu.data);

        TruthTable tt_gu;
        int gu_slot, is_tt_gu = 0;
        if (tt_compress(gu_t, H, 2*inter, &tt_gu)) {
            gu_slot = tb_const_tt(&tb, &tt_gu);
            free(gu_t);
            is_tt_gu = 1;
            n_tt++;
        } else {
            int gu_shape[] = {H, 2*inter};
            gu_slot = tb_const(&tb, gu_t, gu_shape, 2);
        }

        /* Down projection: (inter, H) → transpose → (H reversed... no)
         * Actually (inter, H) → T → (H, inter) ... wait:
         * MLP does gated @ down_t where gated is (1, inter),
         * so down_t is (inter, H) transposed = ... let me check:
         * In Python: down_proj_t = get_transposed(li, "down_proj")
         * get_transposed returns layers[li]["down_proj"].T
         * down_proj shape is (H, inter) → .T → (inter, H)
         * So gated (1, inter) @ down_t (inter, H) → (1, H). Correct. */
        snprintf(wpath, sizeof(wpath), "%s/down_proj.npy", layer_dir);
        NpyArray wd = npy_load(wpath);
        /* wd is (H, inter), we need (inter, H) = wd.T */
        float *d_t = (float *)malloc(inter * H * sizeof(float));
        for (int r = 0; r < inter; r++)
            for (int c = 0; c < H; c++)
                d_t[r * H + c] = wd.data[c * inter + r];
        free(wd.data);

        TruthTable tt_d;
        int d_slot, is_tt_d = 0;
        if (tt_compress(d_t, inter, H, &tt_d)) {
            d_slot = tb_const_tt(&tb, &tt_d);
            free(d_t);
            is_tt_d = 1;
            n_tt++;
        } else {
            int d_shape[] = {inter, H};
            d_slot = tb_const(&tb, d_t, d_shape, 2);
        }

        /* Layer norm weights */
        snprintf(wpath, sizeof(wpath), "%s/input_layernorm_weight.npy",
                 layer_dir);
        NpyArray ln1_w = npy_load(wpath);
        int ln_shape[] = {cfg.hidden_size};
        int ln1_slot = tb_const(&tb, ln1_w.data, ln_shape, 1);

        snprintf(wpath, sizeof(wpath),
                 "%s/post_attention_layernorm_weight.npy", layer_dir);
        NpyArray ln2_w = npy_load(wpath);
        int ln2_slot = tb_const(&tb, ln2_w.data, ln_shape, 1);

        /* ---- Build layer subgraph ---- */
        int residual = hidden;

        /* Pre-attention RMSNorm */
        hidden = tb_rms_norm(&tb, hidden, ln1_slot,
                             cfg.rms_norm_eps, 1, cfg.hidden_size);

        /* Attention */
        AttnResult attn = tb_attention(
            &tb, hidden, qkv_slot, o_slot,
            cos_b, sin_b, scale_slot,
            kv_in_slots[li * 2], kv_in_slots[li * 2 + 1],
            is_tt_qkv, is_tt_o);

        kv_out_slots[li * 2] = attn.new_k;
        kv_out_slots[li * 2 + 1] = attn.new_v;

        /* Residual add */
        hidden = tb_binop(&tb, T_ADD, residual, attn.output, hidden_shape, 2);

        /* Pre-MLP RMSNorm */
        residual = hidden;
        hidden = tb_rms_norm(&tb, hidden, ln2_slot,
                             cfg.rms_norm_eps, 1, cfg.hidden_size);

        /* MLP */
        int mlp_out = tb_mlp(&tb, hidden, gu_slot, d_slot,
                             is_tt_gu, is_tt_d);

        /* Residual add */
        hidden = tb_binop(&tb, T_ADD, residual, mlp_out, hidden_shape, 2);

        printf("[compile]   Layer %d/%d: %s %s %s %s\n",
               li + 1, cfg.num_layers,
               is_tt_qkv ? "TT" : "F32",
               is_tt_o   ? "TT" : "F32",
               is_tt_gu  ? "TT" : "F32",
               is_tt_d   ? "TT" : "F32");
    }

    /* ---- Final norm ---- */
    char wpath[512];
    snprintf(wpath, sizeof(wpath), "%s/final_norm_weight.npy", weights_dir);
    NpyArray final_ln = npy_load(wpath);
    int ln_shape[] = {cfg.hidden_size};
    int final_ln_slot = tb_const(&tb, final_ln.data, ln_shape, 1);
    hidden = tb_rms_norm(&tb, hidden, final_ln_slot,
                         cfg.rms_norm_eps, 1, cfg.hidden_size);

    /* ---- lm_head projection ---- */
    snprintf(wpath, sizeof(wpath), "%s/lm_head.npy", weights_dir);
    NpyArray lm_head;
    int lm_head_slot, is_tt_lm = 0;
    if (file_exists(wpath)) {
        lm_head = npy_load(wpath);
    } else {
        /* Tied: lm_head = embed_tokens */
        snprintf(wpath, sizeof(wpath), "%s/embed_tokens.npy", weights_dir);
        lm_head = npy_load(wpath);
    }
    /* lm_head is (vocab, H) → T → (H, vocab) */
    float *lm_t = (float *)malloc(cfg.hidden_size * cfg.vocab_size * sizeof(float));
    for (int r = 0; r < cfg.hidden_size; r++)
        for (int c = 0; c < cfg.vocab_size; c++)
            lm_t[r * cfg.vocab_size + c] = lm_head.data[c * cfg.hidden_size + r];
    free(lm_head.data);

    TruthTable tt_lm;
    if (tt_compress(lm_t, cfg.hidden_size, cfg.vocab_size, &tt_lm)) {
        lm_head_slot = tb_const_tt(&tb, &tt_lm);
        free(lm_t);
        is_tt_lm = 1;
        n_tt++;
    } else {
        int lm_shape[] = {cfg.hidden_size, cfg.vocab_size};
        lm_head_slot = tb_const(&tb, lm_t, lm_shape, 2);
    }

    int logits_shape[] = {1, cfg.vocab_size};
    int logits_slot;
    if (is_tt_lm)
        logits_slot = tb_matmul_tt(&tb, hidden, lm_head_slot, logits_shape, 2);
    else
        logits_slot = tb_matmul(&tb, hidden, lm_head_slot, logits_shape, 2);

    printf("[compile] Converted %d weight matrices to truth tables\n", n_tt);
    printf("[compile] Tape: %d instructions, %d slots\n",
           tb.n_instrs, tb.next_slot);

    /* ---- 4. Write tape.bin ---- */
    {
        char *p = path_join(circuit_dir, "tape.bin");
        FILE *f = fopen(p, "wb");
        free(p);
        int32_t header[2] = {tb.n_instrs, (int32_t)sizeof(TapeInstr)};
        fwrite(header, sizeof(int32_t), 2, f);
        fwrite(tb.instrs, sizeof(TapeInstr), tb.n_instrs, f);
        fclose(f);
    }

    /* ---- 5. Write slots.bin + const data ---- */
    {
        char *p = path_join(circuit_dir, "slots.bin");
        FILE *f = fopen(p, "wb");
        free(p);
        int32_t n_slots = tb.next_slot;
        fwrite(&n_slots, sizeof(int32_t), 1, f);
        for (int s = 0; s < tb.next_slot; s++) {
            int32_t stype = tb.slot_type[s];
            int32_t ndim = tb.slot_ndim[s];
            fwrite(&stype, sizeof(int32_t), 1, f);
            fwrite(&ndim, sizeof(int32_t), 1, f);
            for (int d = 0; d < MAX_DIMS; d++) {
                int32_t dim = (d < ndim) ? tb.slot_shape[s][d] : 0;
                fwrite(&dim, sizeof(int32_t), 1, f);
            }
        }
        fclose(f);

        /* Write const data files */
        for (int s = 0; s < tb.next_slot; s++) {
            if (tb.slot_type[s] == 1) {
                /* Regular float32 const */
                int sz = 1;
                for (int d = 0; d < tb.slot_ndim[s]; d++)
                    sz *= tb.slot_shape[s][d];
                char name[128];
                snprintf(name, sizeof(name), "const_%d.bin", s);
                char *cp = path_join(circuit_dir, name);
                write_file_bin(cp, tb.const_data[s], sz * sizeof(float));
                free(cp);
            } else if (tb.slot_type[s] == 4) {
                /* TT const: codebook + indices */
                TruthTable *tt = &tb.tt_data[s];
                char name[128];
                snprintf(name, sizeof(name), "const_%d_codebook.bin", s);
                char *cp = path_join(circuit_dir, name);
                write_file_bin(cp, tt->codebook, tt->n_codes * sizeof(float));
                free(cp);

                snprintf(name, sizeof(name), "const_%d_indices.bin", s);
                cp = path_join(circuit_dir, name);
                write_file_bin(cp, tt->indices, tt->K * tt->N * sizeof(uint16_t));
                free(cp);
            }
        }
    }

    /* ---- 6. Write BPE ROMs ---- */
    {
        char *p;
        /* hash keys: (table_size, max_piece_len) uint8 */
        p = path_join(bpe_dir, "vocab_hash_keys.bin");
        write_file_bin(p, bpe.hash_keys,
                       bpe.table_size * bpe.max_piece_len);
        free(p);

        /* Write metadata JSON for hash keys shape */
        p = path_join(bpe_dir, "vocab_hash_keys.json");
        FILE *mf = fopen(p, "w");
        fprintf(mf, "{\"shape\": [%d, %d], \"dtype\": \"uint8\"}\n",
                bpe.table_size, bpe.max_piece_len);
        fclose(mf);
        free(p);

        p = path_join(bpe_dir, "vocab_hash_vals.bin");
        write_file_bin(p, bpe.hash_vals, bpe.table_size * sizeof(int32_t));
        free(p);

        p = path_join(bpe_dir, "vocab_hash_lens.bin");
        write_file_bin(p, bpe.hash_lens, bpe.table_size * sizeof(int32_t));
        free(p);

        p = path_join(bpe_dir, "merge_a.bin");
        write_file_bin(p, bpe.merge_a, bpe.num_merges * sizeof(int32_t));
        free(p);

        p = path_join(bpe_dir, "merge_b.bin");
        write_file_bin(p, bpe.merge_b, bpe.num_merges * sizeof(int32_t));
        free(p);

        p = path_join(bpe_dir, "merge_result.bin");
        write_file_bin(p, bpe.merge_result, bpe.num_merges * sizeof(int32_t));
        free(p);

        p = path_join(bpe_dir, "special_ids.bin");
        write_file_bin(p, bpe.special_ids, bpe.num_special * sizeof(int32_t));
        free(p);

        p = path_join(bpe_dir, "id_to_bytes.bin");
        write_file_bin(p, bpe.id_to_bytes, bpe.id_to_bytes_len);
        free(p);

        p = path_join(bpe_dir, "id_to_offsets.bin");
        write_file_bin(p, bpe.id_to_offsets,
                       tok.vocab_size * 2 * sizeof(int32_t));
        free(p);
    }

    /* ---- 7. Write tables (embed, rope) ---- */
    {
        snprintf(wpath, sizeof(wpath), "%s/embed_tokens.npy", weights_dir);
        NpyArray embed = npy_load(wpath);
        char *p = path_join(tables_dir, "embed_table.bin");
        write_file_bin(p, embed.data, embed.size * sizeof(float));
        free(p);
        free(embed.data);

        float *rope_cos, *rope_sin;
        build_rope(max_seq, cfg.head_dim, cfg.rope_theta, &rope_cos, &rope_sin);
        p = path_join(tables_dir, "rope_cos.bin");
        write_file_bin(p, rope_cos, max_seq * cfg.head_dim * sizeof(float));
        free(p);
        p = path_join(tables_dir, "rope_sin.bin");
        write_file_bin(p, rope_sin, max_seq * cfg.head_dim * sizeof(float));
        free(p);
        free(rope_cos);
        free(rope_sin);
    }

    /* ---- 8. Write processor.json ---- */
    {
        char *p = path_join(chip_dir, "processor.json");
        FILE *f = fopen(p, "w");
        free(p);

        fprintf(f, "{\n");
        fprintf(f, "  \"vocab_size\": %d,\n", cfg.vocab_size);
        fprintf(f, "  \"hidden_dim\": %d,\n", cfg.hidden_size);
        fprintf(f, "  \"num_layers\": %d,\n", cfg.num_layers);
        fprintf(f, "  \"num_kv_heads\": %d,\n", cfg.num_kv_heads);
        fprintf(f, "  \"head_dim\": %d,\n", cfg.head_dim);
        fprintf(f, "  \"max_seq_len\": %d,\n", max_seq);
        fprintf(f, "  \"eos_token_id\": %d,\n", tok.eos_token_id);

        fprintf(f, "  \"input_map\": {\n");
        fprintf(f, "    \"token_embed\": %d,\n", input_token_embed);
        fprintf(f, "    \"rope_cos\": %d,\n", input_rope_cos);
        fprintf(f, "    \"rope_sin\": %d,\n", input_rope_sin);
        for (int li = 0; li < cfg.num_layers; li++) {
            fprintf(f, "    \"L%d/cache_k\": %d,\n", li, kv_in_slots[li*2]);
            fprintf(f, "    \"L%d/cache_v\": %d%s\n", li, kv_in_slots[li*2+1],
                    (li < cfg.num_layers - 1) ? "," : "");
        }
        fprintf(f, "  },\n");

        fprintf(f, "  \"output_map\": {\n");
        fprintf(f, "    \"logits\": %d,\n", logits_slot);
        for (int li = 0; li < cfg.num_layers; li++) {
            fprintf(f, "    \"L%d/new_k\": %d,\n", li, kv_out_slots[li*2]);
            fprintf(f, "    \"L%d/new_v\": %d%s\n", li, kv_out_slots[li*2+1],
                    (li < cfg.num_layers - 1) ? "," : "");
        }
        fprintf(f, "  }\n");
        fprintf(f, "}\n");
        fclose(f);
    }

    /* ---- 9. Write chip.json ---- */
    {
        char *p = path_join(chip_dir, "chip.json");
        FILE *f = fopen(p, "w");
        free(p);
        fprintf(f, "{\n");
        fprintf(f, "  \"kllm_version\": \"0.3.0\",\n");
        fprintf(f, "  \"model_name\": \"%s\",\n", model_name ? model_name : "compiled");
        fprintf(f, "  \"eos_token_id\": %d,\n", tok.eos_token_id);
        fprintf(f, "  \"max_seq_len\": %d,\n", max_seq);
        fprintf(f, "  \"vocab_size\": %d,\n", cfg.vocab_size);
        fprintf(f, "  \"hidden_dim\": %d,\n", cfg.hidden_size);
        fprintf(f, "  \"num_layers\": %d,\n", cfg.num_layers);
        fprintf(f, "  \"head_dim\": %d\n", cfg.head_dim);
        fprintf(f, "}\n");
        fclose(f);
    }

    printf("[compile] Done — chip saved to %s\n", chip_dir);

    /* Cleanup */
    free(kv_in_slots);
    free(kv_out_slots);
    free(tb.instrs);
    /* Note: weight data in const_data[] and tt_data[] is needed for
     * the chip directory, so we keep those files but free tracking arrays */
    for (int s = 0; s < tb.next_slot; s++) {
        if (tb.slot_type[s] == 1 && tb.const_data[s])
            free(tb.const_data[s]);
        if (tb.slot_type[s] == 4) {
            free(tb.tt_data[s].codebook);
            free(tb.tt_data[s].indices);
        }
    }
    free(tb.slot_type);
    free(tb.slot_shape);
    free(tb.slot_ndim);
    free(tb.const_data);
    free(tb.tt_data);

    free_bpe_roms(&bpe);
    free_tokenizer_data(&tok);

    free(weights_dir);
    free(circuit_dir);
    free(bpe_dir);
    free(tables_dir);
    free(tok_dir);

    return 0;
}
