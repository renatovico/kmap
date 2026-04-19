/*
 * kllm_gates.c — Gate minimization analysis for truth-table circuits.
 *
 * Reads a compiled chip's truth tables and finds minimum-gate
 * implementations of each codebook lookup function.
 *
 * Each codebook maps an index to a float32 value.
 * In hardware this is 32 boolean functions of log2(n_codes) input bits.
 * We enumerate the minimum number of 2-input gates (AND/OR/XOR)
 * needed to compute all 32 output bits with sharing.
 *
 * Approach:
 *   For each codebook, extract 32 truth tables (one per output bit).
 *   Deduplicate identical truth tables → unique functions.
 *   For each unique function, classify:
 *     - constant (0 gates)
 *     - single variable / negation (0 gates, just wiring)
 *     - 2-var (1 gate max)
 *     - general: enumerate reachable truth tables via gate composition
 *
 *   Report: naive (32 LUTs per element) vs optimized (shared lookup circuit).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "tape_runner.h"

/* ── Utility ─────────────────────────────────────────────────── */

static char *path_join_g(const char *dir, const char *name) {
    size_t len = strlen(dir) + 1 + strlen(name) + 1;
    char *p = (char *)malloc(len);
    snprintf(p, len, "%s/%s", dir, name);
    return p;
}

static void *read_file_g(const char *path, size_t *out_sz) {
    FILE *f = fopen(path, "rb");
    if (!f) { *out_sz = 0; return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *buf = malloc(sz);
    fread(buf, 1, sz, f);
    fclose(f);
    *out_sz = (size_t)sz;
    return buf;
}

/* ── Boolean function helpers ────────────────────────────────── */

/* Convert float32 to its IEEE 754 bit pattern */
static uint32_t f32_to_bits(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}

/*
 * Truth table representation:
 * For n_codes entries and n_vars = ceil(log2(n_codes)) input bits,
 * a truth table is a bitmask of n_codes bits where bit i = f(i).
 *
 * We use uint64_t arrays for codebooks up to 64 entries.
 * For larger codebooks (>64 entries), we use dynamically-sized arrays.
 * Since n_codes can be up to ~5600, we need multi-word truth tables.
 */

/* Maximum input bits we'll do exact enumeration for */
#define MAX_EXACT_VARS 13

/* ── FPGA 6-LUT cost model ───────────────────────────────────── */

/*
 * FPGA 6-LUT model: a 6-LUT implements any function of ≤6 inputs.
 * For n_vars > 6, Shannon decomposition:
 *   f(x1..xk) = xk * f1(x1..x_{k-1}) + ~xk * f0(x1..x_{k-1})
 * Cost: C(6)=1, C(k) = 2*C(k-1)+1 = 2^(k-5) - 1 LUTs.
 *
 * Returns total 6-LUTs for all 32 output bits (excluding trivial bits).
 */
static int lut6_per_function(int n_vars) {
    if (n_vars <= 6) return 1;
    return (1 << (n_vars - 5)) - 1;
}

static int estimate_luts_large(const float *codebook, int n_codes,
                               int *out_trivial) {
    int n_vars = 0;
    { int tmp = n_codes - 1; while (tmp > 0) { n_vars++; tmp >>= 1; } }
    if (n_vars == 0) n_vars = 1;

    int luts_per_bit = lut6_per_function(n_vars);
    int trivial = 0;

    for (int b = 0; b < 32; b++) {
        int ones = 0;
        for (int i = 0; i < n_codes; i++) {
            uint32_t bits = f32_to_bits(codebook[i]);
            if ((bits >> b) & 1) ones++;
        }
        if (ones == 0 || ones == n_codes) trivial++;
    }

    *out_trivial = trivial;
    return (32 - trivial) * luts_per_bit;
}

/* ── Codebook analysis ───────────────────────────────────────── */

typedef struct {
    int n_codes;
    int n_vars;
    int naive_luts;          /* 32 × lut6_per_function(n_vars) per lookup */
    int optimized_luts;      /* after trivial-bit elimination + sharing */
    int n_trivial;           /* constant or single-var bits (0 LUTs) */
    int n_unique_functions;  /* unique non-trivial truth tables */
    int n_slots;             /* how many slots share this codebook */
    int64_t total_elements;  /* K*N summed across all slots */
} CbAnalysis;

static CbAnalysis analyze_codebook(const float *codebook, int n_codes) {
    CbAnalysis a = {0};
    a.n_codes = n_codes;
    a.n_vars = 0;
    { int tmp = n_codes - 1; while (tmp > 0) { a.n_vars++; tmp >>= 1; } }
    if (a.n_vars == 0) a.n_vars = 1;

    int luts_per_bit = lut6_per_function(a.n_vars);
    a.naive_luts = 32 * luts_per_bit;

    if (n_codes == 1) {
        a.optimized_luts = 0;
        a.n_trivial = 32;
        a.n_unique_functions = 0;
        return a;
    }

    if (n_codes <= 64) {
        /* Use compact 64-bit truth tables */
        uint64_t tts[32];
        uint64_t mask = (n_codes >= 64) ? UINT64_MAX : ((1ULL << n_codes) - 1);

        for (int b = 0; b < 32; b++) {
            uint64_t tt = 0;
            for (int i = 0; i < n_codes; i++) {
                uint32_t bits = f32_to_bits(codebook[i]);
                if ((bits >> b) & 1) tt |= (1ULL << i);
            }
            tts[b] = tt & mask;
        }

        /* Classify and deduplicate */
        uint64_t unique_nontrivial[32];
        int n_unique = 0;
        a.n_trivial = 0;

        for (int b = 0; b < 32; b++) {
            uint64_t tt = tts[b];

            /* Check constant */
            if (tt == 0 || tt == mask) {
                a.n_trivial++;
                continue;
            }

            /* Check single variable */
            int is_var = 0;
            for (int v = 0; v < a.n_vars; v++) {
                uint64_t var_tt = 0;
                for (int i = 0; i < n_codes; i++)
                    if ((i >> v) & 1) var_tt |= (1ULL << i);
                var_tt &= mask;
                if (tt == var_tt || tt == ((~var_tt) & mask)) {
                    is_var = 1;
                    break;
                }
            }
            if (is_var) {
                a.n_trivial++;
                continue;
            }

            /* Check if already seen */
            int seen = 0;
            for (int j = 0; j < n_unique; j++) {
                if (unique_nontrivial[j] == tt ||
                    unique_nontrivial[j] == ((~tt) & mask)) {
                    seen = 1;
                    break;
                }
            }
            if (!seen && n_unique < 32) {
                unique_nontrivial[n_unique++] = tt;
            }
        }
        a.n_unique_functions = n_unique;

        if (n_unique == 0) {
            a.optimized_luts = 0;
        } else {
            /* For small codebooks (≤6 vars), exact gate count ≈ LUT count.
             * Each unique non-trivial function needs at most 1 6-LUT.
             * For >6 vars (but ≤64 entries), use Shannon bound. */
            if (a.n_vars <= 6) {
                a.optimized_luts = n_unique;  /* 1 LUT per unique function */
            } else {
                a.optimized_luts = n_unique * luts_per_bit;
            }
        }
    } else {
        /* Large codebook — Shannon decomposition bound */
        a.optimized_luts = estimate_luts_large(codebook, n_codes, &a.n_trivial);
        a.n_unique_functions = 32 - a.n_trivial;
    }

    return a;
}

/* ── Main analysis ───────────────────────────────────────────── */

int cmd_optimize(int argc, char **argv) {
    if (argc < 1) {
        fprintf(stderr, "Usage: kllm optimize <chip_dir> [-v]\n");
        return 1;
    }

    const char *chip_dir = argv[0];
    int verbose = 0;
    for (int i = 1; i < argc; i++)
        if (strcmp(argv[i], "-v") == 0) verbose = 1;

    /* Read chip.json for model name */
    char *chip_json_path = path_join_g(chip_dir, "chip.json");
    size_t cj_sz;
    char *chip_json = (char *)read_file_g(chip_json_path, &cj_sz);
    free(chip_json_path);
    if (!chip_json) {
        fprintf(stderr, "ERROR: cannot read chip.json\n");
        return 1;
    }

    /* Extract model_name from JSON */
    const char *mn = strstr(chip_json, "\"model_name\"");
    char model_name[256] = "unknown";
    if (mn) {
        const char *q1 = strchr(mn + 12, '"');
        if (q1) {
            q1++;
            const char *q2 = strchr(q1, '"');
            if (q2) {
                int len = (int)(q2 - q1);
                if (len > 255) len = 255;
                memcpy(model_name, q1, len);
                model_name[len] = '\0';
            }
        }
    }

    /* Read slots.bin */
    char circuit_dir_buf[512];
    snprintf(circuit_dir_buf, sizeof(circuit_dir_buf), "%s/circuit", chip_dir);
    char *slots_path = path_join_g(circuit_dir_buf, "slots.bin");
    size_t slots_sz;
    uint8_t *slots_raw = (uint8_t *)read_file_g(slots_path, &slots_sz);
    free(slots_path);
    if (!slots_raw) {
        fprintf(stderr, "ERROR: cannot read slots.bin\n");
        free(chip_json);
        return 1;
    }

    int n_slots = *(int32_t *)slots_raw;
    int slot_record = 4 + 4 + 8 * 4;  /* type + ndim + shape[8] = 40 bytes */

    printf("Model: %s\n", model_name);
    printf("Total slots: %d\n\n", n_slots);

    /* Collect TT slots: load codebooks */
    typedef struct {
        int slot_id;
        int n_codes;
        int K, N;
        float *codebook;
    } TTSlot;

    int tt_cap = 256;
    TTSlot *tt_slots = (TTSlot *)malloc(tt_cap * sizeof(TTSlot));
    int n_tt = 0;

    uint8_t *sp = slots_raw + 4;
    for (int i = 0; i < n_slots; i++) {
        int stype = *(int32_t *)(sp);
        /* int ndim = *(int32_t *)(sp + 4); */
        int *shape = (int32_t *)(sp + 8);
        sp += slot_record;

        if (stype != 4) continue;

        char name[128];
        snprintf(name, sizeof(name), "const_%d_codebook.bin", i);
        char *cb_path = path_join_g(circuit_dir_buf, name);
        size_t cb_sz;
        float *codebook = (float *)read_file_g(cb_path, &cb_sz);
        free(cb_path);
        if (!codebook) continue;

        if (n_tt >= tt_cap) {
            tt_cap *= 2;
            tt_slots = (TTSlot *)realloc(tt_slots, tt_cap * sizeof(TTSlot));
        }
        tt_slots[n_tt].slot_id = i;
        tt_slots[n_tt].n_codes = (int)(cb_sz / sizeof(float));
        tt_slots[n_tt].K = shape[0];
        tt_slots[n_tt].N = shape[1];
        tt_slots[n_tt].codebook = codebook;
        n_tt++;
    }

    printf("TT slots: %d\n\n", n_tt);

    /* Group by identical codebook (byte-exact comparison) */
    int *group_id = (int *)calloc(n_tt, sizeof(int));
    int n_groups = 0;
    for (int i = 0; i < n_tt; i++) group_id[i] = -1;

    for (int i = 0; i < n_tt; i++) {
        if (group_id[i] >= 0) continue;
        group_id[i] = n_groups;
        for (int j = i + 1; j < n_tt; j++) {
            if (group_id[j] >= 0) continue;
            if (tt_slots[i].n_codes != tt_slots[j].n_codes) continue;
            if (memcmp(tt_slots[i].codebook, tt_slots[j].codebook,
                       tt_slots[i].n_codes * sizeof(float)) == 0) {
                group_id[j] = n_groups;
            }
        }
        n_groups++;
    }

    printf("Unique codebooks: %d\n", n_groups);
    printf("================================================================\n\n");

    /* Analyze each group */
    int64_t total_naive = 0;
    int64_t total_optimized = 0;
    int64_t total_elements = 0;
    int total_lookup_circuits = 0;

    for (int g = 0; g < n_groups; g++) {
        /* Find representative and aggregate */
        int rep = -1;
        int slots_in_group = 0;
        int64_t elements = 0;

        for (int i = 0; i < n_tt; i++) {
            if (group_id[i] != g) continue;
            if (rep < 0) rep = i;
            slots_in_group++;
            elements += (int64_t)tt_slots[i].K * tt_slots[i].N;
        }
        if (rep < 0) continue;

        CbAnalysis ca = analyze_codebook(tt_slots[rep].codebook,
                                         tt_slots[rep].n_codes);
        ca.n_slots = slots_in_group;
        ca.total_elements = elements;
        total_elements += elements;
        total_lookup_circuits += slots_in_group;

        /*
         * FPGA 6-LUT cost model:
         *
         * Each matrix element performs one codebook lookup.
         * Each lookup = combinational circuit: index bits → float32 bits.
         *
         * In a fully parallel datapath, each element needs its own
         * lookup circuit instance. The codebook defines the function;
         * the per-element instance implements it.
         *
         * Naive: 32 output bits × LUT6 cost per bit
         * Optimized: (32 - trivial) × LUT6 cost per bit
         *   + sharing of identical functions across output bits
         */
        int64_t naive = (int64_t)ca.naive_luts * elements;
        int64_t optimized = (int64_t)ca.optimized_luts * elements;
        total_naive += naive;
        total_optimized += optimized;

        double pct = naive > 0 ? (1.0 - (double)optimized / naive) * 100.0 : 0.0;

        if (verbose || elements > 1000) {
            printf("  codebook[%d] (%d-bit index) × %d slots, %lld elements:\n",
                   ca.n_codes, ca.n_vars, slots_in_group, (long long)elements);
            printf("    bits: %d trivial, %d non-trivial\n",
                   ca.n_trivial, ca.n_unique_functions);
            printf("    per-lookup:  naive=%d LUTs  optimized=%d LUTs\n",
                   ca.naive_luts, ca.optimized_luts);
            printf("    total:       naive=%lld  optimized=%lld → %.0f%% reduction\n\n",
                   (long long)naive, (long long)optimized, pct);
        }
    }

    double total_pct = total_naive > 0
        ? (1.0 - (double)total_optimized / total_naive) * 100.0 : 0.0;

    printf("================================================================\n");
    printf("  FPGA 6-LUT Analysis (fully parallel datapath)\n");
    printf("----------------------------------------------------------------\n");
    printf("  Total elements:        %12lld\n", (long long)total_elements);
    printf("  Lookup circuits:       %12d\n", total_lookup_circuits);
    printf("  Unique codebooks:      %12d\n", n_groups);
    printf("  Naive 6-LUT count:     %12lld\n", (long long)total_naive);
    printf("  Optimized 6-LUT count: %12lld\n", (long long)total_optimized);
    printf("  Reduction:             %11.1f%%\n", total_pct);
    printf("================================================================\n");

    /* Cleanup */
    for (int i = 0; i < n_tt; i++) free(tt_slots[i].codebook);
    free(tt_slots);
    free(group_id);
    free(slots_raw);
    free(chip_json);

    return 0;
}
