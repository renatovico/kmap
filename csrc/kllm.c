/*
 * kllm.c — Main CLI binary for kllm.
 *
 * Subcommands:
 *   kllm create  --model <hf_name> <chip_dir>              (Python download + C compile)
 *   kllm infer   <chip_dir> [--max-tokens N] <prompt...>   (native C)
 *   kllm compare <chip_dir> [--max-tokens N] [prompt...]   (C infer + Python HF)
 *
 * Build:
 *   make
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#include "tape_runner.h"


/* ================================================================
 * Minimal JSON parser (just for processor.json)
 * ================================================================ */

/* Find integer value for key in JSON string. Returns -1 if not found. */
static int json_get_int(const char *json, const char *key) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(json, pattern);
    if (!p) return -1;
    p += strlen(pattern);
    while (*p == ' ' || *p == ':' || *p == '\t' || *p == '\n') p++;
    return atoi(p);
}

/* Find integer value for key within a sub-object.
 * Searches for "parent_key": { ... "child_key": N ... }
 * Returns -1 if not found. */
static int json_get_nested_int(const char *json,
                               const char *parent, const char *child) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\"", parent);
    const char *p = strstr(json, pattern);
    if (!p) return -1;
    /* Find the opening brace */
    p = strchr(p, '{');
    if (!p) return -1;
    /* Find closing brace */
    const char *end = strchr(p, '}');
    if (!end) return -1;
    /* Search within the sub-object */
    snprintf(pattern, sizeof(pattern), "\"%s\"", child);
    const char *q = strstr(p, pattern);
    if (!q || q > end) return -1;
    q += strlen(pattern);
    while (*q == ' ' || *q == ':' || *q == '\t' || *q == '\n') q++;
    return atoi(q);
}


/* ================================================================
 * File I/O helpers
 * ================================================================ */

static void *read_file_bin(const char *path, size_t *out_size) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    size_t sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *buf = malloc(sz);
    if (fread(buf, 1, sz, f) != sz) {
        fprintf(stderr, "Read error: %s\n", path);
        exit(1);
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

static int file_exists(const char *path) {
    FILE *f = fopen(path, "rb");
    if (f) { fclose(f); return 1; }
    return 0;
}


/* ================================================================
 * Chip loader
 * ================================================================ */

typedef struct {
    TapeCtx   *ctx;

    /* Machine config */
    int vocab_size, hidden_dim, num_layers, num_kv_heads, head_dim;
    int max_seq_len, eos_token_id;

    /* Slot mappings */
    int token_embed_slot, rope_cos_slot, rope_sin_slot;
    int *kv_in_slots;   /* [num_layers * 2] */
    int *kv_out_slots;  /* [num_layers * 2] */
    int logits_slot;

    /* Tables (owned) */
    float *embed_table;
    float *rope_cos;
    float *rope_sin;

    /* BPE ROMs (owned) */
    uint8_t  *hash_keys;
    int32_t  *hash_vals;
    int32_t  *hash_lens;
    int       table_size, max_piece_len;
    int32_t  *merge_a, *merge_b, *merge_result;
    int       num_merges;
    uint8_t  *id_to_bytes;
    int32_t  *id_to_offsets;
    int32_t  *special_ids;
    int       num_special;
    int       bos_token_id;

    /* Weight data buffers (kept alive for slots) */
    void    **weight_bufs;
    int       n_weight_bufs;
} ChipCtx;


static ChipCtx *chip_load(const char *chip_dir) {
    ChipCtx *chip = (ChipCtx *)calloc(1, sizeof(ChipCtx));
    char *path;

    /* --- 1. Read processor.json --- */
    path = path_join(chip_dir, "processor.json");
    char *proc_json = read_file_text(path);
    free(path);

    chip->vocab_size  = json_get_int(proc_json, "vocab_size");
    chip->hidden_dim  = json_get_int(proc_json, "hidden_dim");
    chip->num_layers  = json_get_int(proc_json, "num_layers");
    chip->num_kv_heads = json_get_int(proc_json, "num_kv_heads");
    chip->head_dim    = json_get_int(proc_json, "head_dim");
    chip->max_seq_len = json_get_int(proc_json, "max_seq_len");
    chip->eos_token_id = json_get_int(proc_json, "eos_token_id");

    int n_layers = chip->num_layers;

    /* Slot mappings from input_map / output_map */
    chip->token_embed_slot = json_get_nested_int(proc_json,
                                "input_map", "token_embed");
    chip->rope_cos_slot = json_get_nested_int(proc_json,
                                "input_map", "rope_cos");
    chip->rope_sin_slot = json_get_nested_int(proc_json,
                                "input_map", "rope_sin");
    chip->logits_slot = json_get_nested_int(proc_json,
                                "output_map", "logits");

    chip->kv_in_slots = (int *)malloc(n_layers * 2 * sizeof(int));
    chip->kv_out_slots = (int *)malloc(n_layers * 2 * sizeof(int));
    for (int li = 0; li < n_layers; li++) {
        char key[64];
        snprintf(key, sizeof(key), "L%d/cache_k", li);
        chip->kv_in_slots[li * 2] = json_get_nested_int(proc_json,
                                        "input_map", key);
        snprintf(key, sizeof(key), "L%d/cache_v", li);
        chip->kv_in_slots[li * 2 + 1] = json_get_nested_int(proc_json,
                                        "input_map", key);
        snprintf(key, sizeof(key), "L%d/new_k", li);
        chip->kv_out_slots[li * 2] = json_get_nested_int(proc_json,
                                        "output_map", key);
        snprintf(key, sizeof(key), "L%d/new_v", li);
        chip->kv_out_slots[li * 2 + 1] = json_get_nested_int(proc_json,
                                        "output_map", key);
    }
    free(proc_json);

    /* --- 2. Read tables (raw float32 binary) --- */
    char *tables_dir = path_join(chip_dir, "tables");

    path = path_join(tables_dir, "embed_table.bin");
    size_t embed_sz;
    chip->embed_table = (float *)read_file_bin(path, &embed_sz);
    free(path);

    path = path_join(tables_dir, "rope_cos.bin");
    chip->rope_cos = (float *)read_file_bin(path, NULL);
    free(path);

    path = path_join(tables_dir, "rope_sin.bin");
    chip->rope_sin = (float *)read_file_bin(path, NULL);
    free(path);

    free(tables_dir);

    /* --- 3. Read BPE ROMs --- */
    char *bpe_dir = path_join(chip_dir, "bpe");

    path = path_join(bpe_dir, "vocab_hash_keys.bin");
    size_t hk_sz;
    chip->hash_keys = (uint8_t *)read_file_bin(path, &hk_sz);
    free(path);

    path = path_join(bpe_dir, "vocab_hash_vals.bin");
    chip->hash_vals = (int32_t *)read_file_bin(path, NULL);
    free(path);

    path = path_join(bpe_dir, "vocab_hash_lens.bin");
    chip->hash_lens = (int32_t *)read_file_bin(path, NULL);
    free(path);

    /* Parse hash table dimensions from metadata */
    path = path_join(bpe_dir, "vocab_hash_keys.json");
    char *hk_meta = read_file_text(path);
    free(path);
    /* shape: [table_size, max_piece_len] */
    {
        const char *p = strstr(hk_meta, "shape");
        if (p) {
            p = strchr(p, '[');
            if (p) {
                p++;
                chip->table_size = atoi(p);
                p = strchr(p, ',');
                if (p) chip->max_piece_len = atoi(p + 1);
            }
        }
    }
    free(hk_meta);

    path = path_join(bpe_dir, "merge_a.bin");
    size_t ma_sz;
    chip->merge_a = (int32_t *)read_file_bin(path, &ma_sz);
    free(path);
    chip->num_merges = (int)(ma_sz / sizeof(int32_t));

    path = path_join(bpe_dir, "merge_b.bin");
    chip->merge_b = (int32_t *)read_file_bin(path, NULL);
    free(path);

    path = path_join(bpe_dir, "merge_result.bin");
    chip->merge_result = (int32_t *)read_file_bin(path, NULL);
    free(path);

    path = path_join(bpe_dir, "special_ids.bin");
    size_t sp_sz;
    chip->special_ids = (int32_t *)read_file_bin(path, &sp_sz);
    free(path);
    chip->num_special = (int)(sp_sz / sizeof(int32_t));

    /* BOS = second special token (LLaMA convention) */
    chip->bos_token_id = (chip->num_special > 1)
                         ? chip->special_ids[1] : 1;

    path = path_join(bpe_dir, "id_to_bytes.bin");
    chip->id_to_bytes = (uint8_t *)read_file_bin(path, NULL);
    free(path);

    path = path_join(bpe_dir, "id_to_offsets.bin");
    chip->id_to_offsets = (int32_t *)read_file_bin(path, NULL);
    free(path);

    free(bpe_dir);

    /* --- 4. Read pre-compiled tape --- */
    char *circuit_dir = path_join(chip_dir, "circuit");

    /* tape.bin: [n_instrs(u32), instr_size(u32), instr_data...] */
    path = path_join(circuit_dir, "tape.bin");
    size_t tape_sz;
    uint8_t *tape_raw = (uint8_t *)read_file_bin(path, &tape_sz);
    free(path);

    int n_instrs = *(int *)tape_raw;
    int instr_size = *(int *)(tape_raw + 4);

    /* slots.bin: [n_slots(u32), per-slot: type(u32) ndim(u32) shape(i32×8)] */
    path = path_join(circuit_dir, "slots.bin");
    size_t slots_sz;
    uint8_t *slots_raw = (uint8_t *)read_file_bin(path, &slots_sz);
    free(path);

    int n_slots = *(int *)slots_raw;

    /* Create TapeCtx */
    TapeCtx *ctx = tape_ctx_create(n_slots, n_instrs);
    chip->ctx = ctx;

    /* Copy tape instructions */
    if (instr_size != (int)sizeof(TapeInstr)) {
        fprintf(stderr, "ERROR: tape instr size mismatch: file=%d, C=%zu\n",
                instr_size, sizeof(TapeInstr));
        exit(1);
    }
    memcpy(ctx->tape, tape_raw + 8, n_instrs * sizeof(TapeInstr));
    free(tape_raw);

    /* --- 5. Set up slots --- */
    /* Track allocated weight buffers for cleanup */
    int max_weight_bufs = n_slots;
    chip->weight_bufs = (void **)calloc(max_weight_bufs, sizeof(void *));
    chip->n_weight_bufs = 0;

    uint8_t *sp = slots_raw + 4;  /* skip n_slots header */
    int slot_record_size = 4 + 4 + 8 * 4;  /* type + ndim + shape[8] */

    for (int nid = 0; nid < n_slots; nid++) {
        int stype = *(int *)(sp);
        int ndim = *(int *)(sp + 4);
        int shape[MAX_DIMS];
        for (int d = 0; d < MAX_DIMS; d++)
            shape[d] = *(int *)(sp + 8 + d * 4);
        sp += slot_record_size;

        if (stype == 0) continue;  /* UNUSED */

        if (stype == 4) {
            /* TT weight: load codebook + indices */
            char cb_name[128], idx_name[128];
            snprintf(cb_name, sizeof(cb_name),
                     "const_%d_codebook.bin", nid);
            snprintf(idx_name, sizeof(idx_name),
                     "const_%d_indices.bin", nid);

            char *cb_path = path_join(circuit_dir, cb_name);
            char *idx_path = path_join(circuit_dir, idx_name);

            size_t cb_sz, idx_sz;
            float *codebook = (float *)read_file_bin(cb_path, &cb_sz);
            uint16_t *indices = (uint16_t *)read_file_bin(idx_path, &idx_sz);
            free(cb_path);
            free(idx_path);

            int n_codes = (int)(cb_sz / sizeof(float));
            int K = shape[0], N = shape[1];
            tape_slot_set_truth_table(ctx, nid, codebook, n_codes,
                                      indices, K, N);

            chip->weight_bufs[chip->n_weight_bufs++] = codebook;
            chip->weight_bufs[chip->n_weight_bufs++] = indices;
        }
        else if (stype == 1) {
            /* CONST float32: load from const_NNN.bin */
            char const_name[128];
            snprintf(const_name, sizeof(const_name),
                     "const_%d.bin", nid);
            char *const_path = path_join(circuit_dir, const_name);

            size_t c_sz;
            float *data = (float *)read_file_bin(const_path, &c_sz);
            free(const_path);

            tape_slot_set_external(ctx, nid, data, shape, ndim);
            chip->weight_bufs[chip->n_weight_bufs++] = data;
        }
        else {
            /* INPUT (2) or INTERMEDIATE (3): allocate buffer */
            tape_slot_alloc(ctx, nid, shape, ndim);
        }
    }

    free(slots_raw);
    free(circuit_dir);

    return chip;
}


static void chip_destroy(ChipCtx *chip) {
    if (!chip) return;
    if (chip->ctx) tape_ctx_destroy(chip->ctx);
    free(chip->kv_in_slots);
    free(chip->kv_out_slots);
    free(chip->embed_table);
    free(chip->rope_cos);
    free(chip->rope_sin);
    free(chip->hash_keys);
    free(chip->hash_vals);
    free(chip->hash_lens);
    free(chip->merge_a);
    free(chip->merge_b);
    free(chip->merge_result);
    free(chip->special_ids);
    free(chip->id_to_bytes);
    free(chip->id_to_offsets);
    for (int i = 0; i < chip->n_weight_bufs; i++)
        free(chip->weight_bufs[i]);
    free(chip->weight_bufs);
    free(chip);
}


/* ================================================================
 * Streaming callback — prints decoded bytes to stdout
 * ================================================================ */

static int print_callback(const uint8_t *bytes, int byte_len,
                           void *user_data) {
    (void)user_data;
    fwrite(bytes, 1, byte_len, stdout);
    fflush(stdout);
    return 0;
}


/* ================================================================
 * Chat template formatting (TinyLlama format)
 * ================================================================ */

static char *format_chat_prompt(const char *user_msg) {
    const char *prefix = "<|user|>\n";
    const char *suffix = "</s>\n<|assistant|>\n";
    size_t plen = strlen(prefix);
    size_t mlen = strlen(user_msg);
    size_t slen = strlen(suffix);
    char *buf = (char *)malloc(plen + mlen + slen + 1);
    memcpy(buf, prefix, plen);
    memcpy(buf + plen, user_msg, mlen);
    memcpy(buf + plen + mlen, suffix, slen);
    buf[plen + mlen + slen] = '\0';
    return buf;
}


/* ================================================================
 * Extern: C compiler from kllm_compile.c
 * ================================================================ */

extern int compile_chip(const char *chip_dir, int max_seq, const char *model_name);


/* ================================================================
 * Subcommand: create
 * ================================================================ */

static int cmd_create(int argc, char **argv) {
    const char *model_name = NULL;
    const char *chip_dir = NULL;

    /* Parse args: kllm create --model <name> <chip_dir> */
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_name = argv[++i];
        } else if (argv[i][0] != '-') {
            chip_dir = argv[i];
        }
    }

    if (!model_name || !chip_dir) {
        fprintf(stderr,
            "Usage: kllm create --model <hf_model_name> <chip_dir>\n"
            "\nExample:\n"
            "  kllm create --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 ./mychip\n");
        return 1;
    }

    /* Step 1: Call Python to download model weights + tokenizer */
    printf("[kllm] Downloading %s …\n", model_name);
    char cmd[4096];
    snprintf(cmd, sizeof(cmd),
        "python3 -c \""
        "from kllm.compiler.fabric import Fabric; "
        "Fabric.from_pretrained('%s', '%s')"
        "\"",
        model_name, chip_dir);
    int ret = system(cmd);
    if (ret != 0) {
        fprintf(stderr, "[kllm] Download failed (exit %d)\n", ret);
        return 1;
    }

    /* Step 2: Compile in C */
    printf("[kllm] Compiling chip …\n");
    return compile_chip(chip_dir, 2048, model_name);
}


/* ================================================================
 * Subcommand: infer
 * ================================================================ */

static int cmd_infer(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
            "Usage: kllm infer <chip_dir> [--max-tokens N] <prompt...>\n"
            "\nExample:\n"
            "  kllm infer ./mychip What is the capital of France?\n"
            "  kllm infer ./mychip --max-tokens 100 Hello world\n");
        return 1;
    }

    const char *chip_dir = argv[0];
    int max_tokens = 50;
    int prompt_start = 1;

    /* Parse --max-tokens */
    if (argc > 2 && strcmp(argv[1], "--max-tokens") == 0) {
        max_tokens = atoi(argv[2]);
        prompt_start = 3;
    }

    /* Join remaining args as prompt */
    if (prompt_start >= argc) {
        fprintf(stderr, "Error: no prompt provided\n");
        return 1;
    }
    size_t prompt_len = 0;
    for (int i = prompt_start; i < argc; i++)
        prompt_len += strlen(argv[i]) + 1;
    char *prompt = (char *)malloc(prompt_len + 1);
    prompt[0] = '\0';
    for (int i = prompt_start; i < argc; i++) {
        if (i > prompt_start) strcat(prompt, " ");
        strcat(prompt, argv[i]);
    }

    /* Format as chat */
    char *formatted = format_chat_prompt(prompt);
    free(prompt);

    /* Load chip */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    ChipCtx *chip = chip_load(chip_dir);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double load_s = (t1.tv_sec - t0.tv_sec)
                  + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    fprintf(stderr, "[kllm] Loaded chip in %.2fs\n", load_s);

    /* Run inference */
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int total_generated = 0;
    processor_infer_bytes(
        chip->ctx,
        chip->hash_keys, chip->hash_vals, chip->hash_lens,
        chip->table_size, chip->max_piece_len,
        chip->merge_a, chip->merge_b, chip->merge_result,
        chip->num_merges,
        chip->id_to_bytes, chip->id_to_offsets,
        chip->special_ids, chip->num_special,
        chip->bos_token_id,
        chip->embed_table, chip->vocab_size, chip->hidden_dim,
        chip->rope_cos, chip->rope_sin, chip->head_dim,
        chip->token_embed_slot, chip->rope_cos_slot, chip->rope_sin_slot,
        chip->kv_in_slots, chip->kv_out_slots,
        chip->logits_slot, chip->num_layers, chip->num_kv_heads,
        (const uint8_t *)formatted, (int)strlen(formatted),
        max_tokens, chip->eos_token_id,
        chip->max_seq_len, 8192,
        print_callback, NULL,
        &total_generated);

    printf("\n");

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double infer_s = (t1.tv_sec - t0.tv_sec)
                   + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    fprintf(stderr, "[kllm] Generated %d tokens in %.2fs (%.1f tok/s)\n",
            total_generated, infer_s,
            total_generated / infer_s);

    free(formatted);
    chip_destroy(chip);
    return 0;
}


/* ================================================================
 * Subcommand: compare
 * ================================================================ */

typedef struct { char buf[8192]; int len; } BufCtx;

static int buf_callback(const uint8_t *bytes, int byte_len, void *ud) {
    BufCtx *b = (BufCtx *)ud;
    if (b->len + byte_len < 8192) {
        memcpy(b->buf + b->len, bytes, byte_len);
        b->len += byte_len;
    }
    return 0;
}

static int cmd_compare(int argc, char **argv) {
    if (argc < 1) {
        fprintf(stderr,
            "Usage: kllm compare <chip_dir> [--max-tokens N] [prompt...]\n"
            "\nExample:\n"
            "  kllm compare ./mychip\n"
            "  kllm compare ./mychip --max-tokens 20 Hello world\n");
        return 1;
    }

    const char *chip_dir = argv[0];
    int max_tokens = 50;
    int prompt_start = 1;
    char *prompt = NULL;

    /* Parse --max-tokens */
    if (argc > 2 && strcmp(argv[1], "--max-tokens") == 0) {
        max_tokens = atoi(argv[2]);
        prompt_start = 3;
    }

    /* Optional prompt */
    if (prompt_start < argc) {
        size_t prompt_len = 0;
        for (int i = prompt_start; i < argc; i++)
            prompt_len += strlen(argv[i]) + 1;
        prompt = (char *)malloc(prompt_len + 1);
        prompt[0] = '\0';
        for (int i = prompt_start; i < argc; i++) {
            if (i > prompt_start) strcat(prompt, " ");
            strcat(prompt, argv[i]);
        }
    }

    /* Read model_name from chip.json */
    char *chip_json_path = path_join(chip_dir, "chip.json");
    char *chip_json = read_file_text(chip_json_path);
    free(chip_json_path);

    /* Extract model_name */
    const char *mn = strstr(chip_json, "\"model_name\"");
    char model_name[512] = "unknown";
    if (mn) {
        mn = strchr(mn + 12, '"');
        if (mn) {
            mn++;
            int ml = 0;
            while (mn[ml] && mn[ml] != '"' && ml < 511)
                model_name[ml] = mn[ml], ml++;
            model_name[ml] = '\0';
        }
    }
    free(chip_json);

    /* Benchmark prompts */
    const char *prompts[] = {
        "Hello",
        "What is the capital of France?",
        "Explain quantum computing in simple terms for a beginner.",
    };
    int n_prompts = prompt ? 1 : 3;

    printf("\n================================================================\n");
    printf("  kllm — Benchmark: Chip vs HuggingFace\n");
    printf("================================================================\n\n");

    double total_kllm = 0, total_hf = 0;

    for (int pi = 0; pi < n_prompts; pi++) {
        const char *p = prompt ? prompt : prompts[pi];

        /* ---- kllm (C) inference ---- */
        char *formatted = format_chat_prompt(p);
        ChipCtx *chip = chip_load(chip_dir);

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        /* Capture kllm output to buffer */
        BufCtx bctx = {{0}, 0};

        int total_gen = 0;
        processor_infer_bytes(
            chip->ctx,
            chip->hash_keys, chip->hash_vals, chip->hash_lens,
            chip->table_size, chip->max_piece_len,
            chip->merge_a, chip->merge_b, chip->merge_result,
            chip->num_merges,
            chip->id_to_bytes, chip->id_to_offsets,
            chip->special_ids, chip->num_special,
            chip->bos_token_id,
            chip->embed_table, chip->vocab_size, chip->hidden_dim,
            chip->rope_cos, chip->rope_sin, chip->head_dim,
            chip->token_embed_slot, chip->rope_cos_slot, chip->rope_sin_slot,
            chip->kv_in_slots, chip->kv_out_slots,
            chip->logits_slot, chip->num_layers, chip->num_kv_heads,
            (const uint8_t *)formatted, (int)strlen(formatted),
            max_tokens, chip->eos_token_id,
            chip->max_seq_len, 8192,
            buf_callback, &bctx,
            &total_gen);

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double kllm_s = (t1.tv_sec - t0.tv_sec)
                      + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        total_kllm += kllm_s;

        bctx.buf[bctx.len] = '\0';
        char kllm_output[8192];
        strncpy(kllm_output, bctx.buf, sizeof(kllm_output));

        free(formatted);
        chip_destroy(chip);

        /* ---- HuggingFace inference (Python) ---- */
        char hf_cmd[8192];
        char escaped_prompt[4096];
        /* Escape single quotes in prompt */
        int ep = 0;
        for (int i = 0; p[i] && ep < 4090; i++) {
            if (p[i] == '\'') {
                escaped_prompt[ep++] = '\'';
                escaped_prompt[ep++] = '\\';
                escaped_prompt[ep++] = '\'';
                escaped_prompt[ep++] = '\'';
            } else {
                escaped_prompt[ep++] = p[i];
            }
        }
        escaped_prompt[ep] = '\0';

        snprintf(hf_cmd, sizeof(hf_cmd),
            "python3 -c \""
            "import time, torch; "
            "from transformers import AutoModelForCausalLM, AutoTokenizer; "
            "tok = AutoTokenizer.from_pretrained('%s'); "
            "model = AutoModelForCausalLM.from_pretrained('%s', dtype=torch.float32, attn_implementation='eager'); "
            "msgs = [{'role': 'user', 'content': '%s'}]; "
            "pt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False); "
            "ids = tok(pt, return_tensors='pt')['input_ids']; "
            "t0 = time.perf_counter(); "
            "gen = ids.clone()\n"
            "with torch.no_grad():\n"
            "    for _ in range(%d):\n"
            "        logits = model(gen).logits[0, -1]\n"
            "        nxt = int(logits.argmax())\n"
            "        if nxt == tok.eos_token_id: break\n"
            "        gen = torch.cat([gen, torch.tensor([[nxt]])], dim=1)\n"
            "hf_time = time.perf_counter() - t0; "
            "out = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True); "
            "print(f'{hf_time:.4f}|{out}')"
            "\"",
            model_name, model_name, escaped_prompt, max_tokens);

        FILE *hf_pipe = popen(hf_cmd, "r");
        char hf_result[8192] = {0};
        double hf_s = 0;
        char hf_output[8192] = {0};

        if (hf_pipe) {
            if (fgets(hf_result, sizeof(hf_result), hf_pipe)) {
                char *sep = strchr(hf_result, '|');
                if (sep) {
                    *sep = '\0';
                    hf_s = atof(hf_result);
                    strncpy(hf_output, sep + 1, sizeof(hf_output));
                    /* Remove trailing newline */
                    int hl = strlen(hf_output);
                    if (hl > 0 && hf_output[hl-1] == '\n')
                        hf_output[hl-1] = '\0';
                }
            }
            pclose(hf_pipe);
        }
        total_hf += hf_s;

        printf("  [%d] Prompt: \"%s\"\n", pi + 1, p);
        printf("      HF   (%.2fs): %.60s\n", hf_s, hf_output);
        printf("      kllm (%.2fs): %.60s\n\n", kllm_s, kllm_output);
    }

    printf("----------------------------------------------------------------\n");
    printf("  Average: HF=%.2fs  kllm=%.2fs\n", total_hf / n_prompts,
           total_kllm / n_prompts);
    printf("================================================================\n");

    free(prompt);
    return 0;
}


/* ================================================================
 * main() — dispatch subcommand
 * ================================================================ */

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s <command> [options]\n"
        "\n"
        "Commands:\n"
        "  create   --model <hf_model> <chip_dir>    Download + compile chip\n"
        "  infer    <chip_dir> [--max-tokens N] <prompt...>   Run inference\n"
        "  compare  <chip_dir> [--max-tokens N] [prompt...]   Benchmark vs HF\n"
        "\n"
        "Examples:\n"
        "  %s create --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 ./mychip\n"
        "  %s infer ./mychip What is the capital of France?\n"
        "  %s compare ./mychip\n",
        prog, prog, prog, prog);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char *cmd = argv[1];

    if (strcmp(cmd, "create") == 0) {
        return cmd_create(argc - 2, argv + 2);
    } else if (strcmp(cmd, "infer") == 0) {
        return cmd_infer(argc - 2, argv + 2);
    } else if (strcmp(cmd, "compare") == 0) {
        return cmd_compare(argc - 2, argv + 2);
    } else {
        fprintf(stderr, "Unknown command: %s\n\n", cmd);
        print_usage(argv[0]);
        return 1;
    }
}
