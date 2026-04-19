/*
 * tape_runner.h — Public API for the tape execution engine.
 */

#ifndef TAPE_RUNNER_H
#define TAPE_RUNNER_H

#include <stdint.h>

#define MAX_DIMS 8
#define MAX_CONCAT_INPUTS 8

/* ================================================================
 * Data structures
 * ================================================================ */

typedef struct {
    float    *data;
    int       shape[MAX_DIMS];
    int       ndim;
    int       size;
    int       owns_data;
    float    *codebook;
    uint16_t *w_indices;
    int       n_codes;
    int       tt_K;
    int       tt_N;
} Slot;

typedef struct {
    int tag;
    int out_slot;
    int in0, in1, in2;
    int axis;
    int keepdims;
    int repeats;
    int out_shape[MAX_DIMS];
    int out_ndim;
    int out_size;
    int axes[MAX_DIMS];
    int starts[MAX_DIMS];
    int stops[MAX_DIMS];
    int steps[MAX_DIMS];
    int slice_out_shape[MAX_DIMS];
    int slice_out_ndim;
    int concat_inputs[MAX_CONCAT_INPUTS];
    int concat_n_inputs;
    int tt_weight_slot;
} TapeInstr;

typedef struct {
    Slot      *slots;
    int        n_slots;
    TapeInstr *tape;
    int        n_instrs;
} TapeCtx;

/* ================================================================
 * Tape context API
 * ================================================================ */

TapeCtx *tape_ctx_create(int n_slots, int n_instrs);
void tape_ctx_destroy(TapeCtx *ctx);
int tape_ctx_n_instrs(TapeCtx *ctx);
int tape_ctx_n_slots(TapeCtx *ctx);

/* ================================================================
 * Slot API
 * ================================================================ */

void tape_slot_alloc(TapeCtx *ctx, int slot_idx,
                     const int *shape, int ndim);
void tape_slot_set_external(TapeCtx *ctx, int slot_idx,
                            float *data, const int *shape, int ndim);
void tape_slot_set_truth_table(TapeCtx *ctx, int slot_idx,
                               float *codebook, int n_codes,
                               uint16_t *w_indices, int K, int N);
void tape_slot_write(TapeCtx *ctx, int slot_idx,
                     const float *data, int n);
void tape_slot_set_shape(TapeCtx *ctx, int slot_idx,
                         const int *shape, int ndim);
const float *tape_slot_read(TapeCtx *ctx, int slot_idx, int *out_size);
int tape_slot_read_shape(TapeCtx *ctx, int slot_idx, int *out_shape);
TapeInstr *tape_get_instr(TapeCtx *ctx, int instr_idx);

/* ================================================================
 * Tape runner
 * ================================================================ */

void tape_run(TapeCtx *ctx);

/* ================================================================
 * Processor inference
 * ================================================================ */

typedef int (*token_callback_fn)(
    const uint8_t *bytes, int byte_len, void *user_data);

int processor_infer(
    TapeCtx *ctx,
    const float *embed_table, int vocab_size, int hidden_dim,
    const float *rope_cos, const float *rope_sin, int head_dim,
    int token_embed_slot, int rope_cos_slot, int rope_sin_slot,
    const int *kv_in_slots, const int *kv_out_slots,
    int logits_slot, int n_layers, int num_kv_heads,
    const int *prompt_tokens, int prompt_len,
    int max_new_tokens, int eos_token_id,
    int *output_tokens, int *output_len);

int processor_infer_bytes(
    TapeCtx *ctx,
    const uint8_t *hash_keys, const int32_t *hash_vals,
    const int32_t *hash_lens,
    int table_size, int max_piece_len,
    const int32_t *merge_a, const int32_t *merge_b,
    const int32_t *merge_result, int num_merges,
    const uint8_t *id_to_bytes_rom, const int32_t *id_to_offsets,
    const int32_t *special_ids, int num_special,
    int bos_token_id,
    const float *embed_table, int vocab_size, int hidden_dim,
    const float *rope_cos, const float *rope_sin, int head_dim,
    int token_embed_slot, int rope_cos_slot, int rope_sin_slot,
    const int *kv_in_slots, const int *kv_out_slots,
    int logits_slot, int n_layers, int num_kv_heads,
    const uint8_t *input_bytes, int input_len,
    int max_new_tokens, int eos_token_id,
    int max_seq_tokens, int max_bpe_bytes,
    token_callback_fn callback, void *user_data,
    int *total_generated);

/* BPE encode/decode */
int bpe_encode(
    const uint8_t *raw_bytes, int raw_len,
    const uint8_t *hash_keys, const int32_t *hash_vals,
    const int32_t *hash_lens,
    int table_size, int max_piece_len,
    const int32_t *merge_a, const int32_t *merge_b,
    const int32_t *merge_result, int num_merges,
    int bos_token_id, int max_tokens,
    const uint8_t *id_to_bytes, const int32_t *id_to_offsets,
    const int32_t *special_ids, int num_special, int vocab_size,
    int32_t *out_ids);

int bpe_decode(
    const int32_t *token_ids, int num_tokens,
    const uint8_t *id_to_bytes, const int32_t *id_to_offsets,
    int vocab_size,
    const int32_t *special_ids, int num_special,
    int max_bytes, int skip_leading_space,
    uint8_t *out_bytes);

#endif /* TAPE_RUNNER_H */
