/*
 * _tape_runner.c — C tape execution engine for kllm.
 *
 * Executes a pre-compiled instruction tape entirely in C,
 * eliminating all Python loop overhead (64K function calls → 1).
 *
 * The tape and slot table are set up once by Python, then
 * ceval_tape_run() is called each decode step with only the
 * INPUT slot data changing.
 *
 * Compile (macOS):
 *   cc -O3 -shared -fPIC -march=native -o csrc/_tape_runner.so \
 *      csrc/_tape_runner.c -lm -framework Accelerate
 *
 * Compile (Linux):
 *   cc -O3 -shared -fPIC -march=native -o csrc/_tape_runner.so \
 *      csrc/_tape_runner.c -lm -lopenblas
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

#define MAX_DIMS 8
#define MAX_CONCAT_INPUTS 8

/* ================================================================
 * Tape instruction tags (must match Python _T_* constants)
 * ================================================================ */
enum {
    T_LUT_SILU  = 0,
    T_LUT_EXP   = 1,
    T_LUT_RSQRT = 2,
    T_LUT_COS   = 3,
    T_LUT_SIN   = 4,
    T_ADD       = 5,
    T_SUB       = 6,
    T_MUL       = 7,
    T_DIV       = 8,
    T_MAX       = 9,
    T_NEG       = 10,
    T_SQUARE    = 11,
    T_MATMUL    = 12,
    T_MATMUL_Q8 = 13,
    T_SUM       = 14,
    T_MAX_RED   = 15,
    T_MEAN      = 16,
    T_ARGMAX    = 17,
    T_RESHAPE   = 18,
    T_TRANSPOSE = 19,
    T_CONCAT    = 20,
    T_REPEAT    = 21,
    T_SLICE     = 22,
    T_COPY      = 23,   /* for expand_dims, cast — just copy */
};

/* ================================================================
 * Slot: a tensor buffer with metadata
 * ================================================================ */
typedef struct {
    float    *data;          /* points into a big arena OR external */
    int       shape[MAX_DIMS];
    int       ndim;
    int       size;          /* total number of float elements */
    int       owns_data;     /* 1 if we allocated data (arena) */
} Slot;

/* ================================================================
 * Tape instruction: one op with all params inlined
 * ================================================================ */
typedef struct {
    int tag;
    int out_slot;            /* output slot index */
    int in0, in1, in2;       /* input slot indices (unused = -1) */

    /* Op-specific params */
    int axis;
    int keepdims;
    int repeats;

    /* Shape params for binary ops (broadcasting) */
    int out_shape[MAX_DIMS];
    int out_ndim;
    int out_size;

    /* For MATMUL_Q8: pre-cached raw pointers */
    const int8_t *wq8_ptr;
    const float  *scales_ptr;
    int K_q8, N_q8;

    /* For TRANSPOSE: axes */
    int axes[MAX_DIMS];

    /* For SLICE: start/stop/step per dim */
    int starts[MAX_DIMS];
    int stops[MAX_DIMS];
    int steps[MAX_DIMS];
    int slice_out_shape[MAX_DIMS];
    int slice_out_ndim;

    /* For CONCAT: input slot indices, count, axis */
    int concat_inputs[MAX_CONCAT_INPUTS];
    int concat_n_inputs;
} TapeInstr;

/* ================================================================
 * Tape context: holds all state for repeated execution
 * ================================================================ */
typedef struct {
    Slot      *slots;
    int        n_slots;
    TapeInstr *tape;
    int        n_instrs;
} TapeCtx;

/* ================================================================
 * Helper: shape utilities (duplicated from _circuit_eval.c to
 * keep this file self-contained)
 * ================================================================ */

static int total_size(const int *shape, int ndim) {
    int s = 1;
    for (int i = 0; i < ndim; i++) s *= shape[i];
    return s;
}

static void compute_strides(const int *shape, int ndim, int *strides) {
    if (ndim == 0) return;
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--)
        strides[i] = strides[i + 1] * shape[i + 1];
}

static void linear_to_multi(int idx, const int *shape, int ndim, int *multi) {
    for (int i = ndim - 1; i >= 0; i--) {
        multi[i] = idx % shape[i];
        idx /= shape[i];
    }
}

static inline int broadcast_linear(const int *inp_shape, int inp_ndim,
                                   const int *inp_strides,
                                   const int *out_idx, int out_ndim) {
    int offset = out_ndim - inp_ndim;
    int linear = 0;
    for (int i = 0; i < inp_ndim; i++) {
        int oi = i + offset;
        int dim_idx = (inp_shape[i] == 1) ? 0 : out_idx[oi];
        linear += dim_idx * inp_strides[i];
    }
    return linear;
}

/* ================================================================
 * Inline ops (small ops inlined to avoid function call overhead)
 * ================================================================ */

static void do_silu(float *out, const float *x, int n) {
    for (int i = 0; i < n; i++) {
        double xd = (double)x[i];
        out[i] = (float)(xd / (1.0 + exp(-xd)));
    }
}

static void do_exp(float *out, const float *x, int n) {
    for (int i = 0; i < n; i++)
        out[i] = (float)exp((double)x[i]);
}

static void do_rsqrt(float *out, const float *x, int n) {
    for (int i = 0; i < n; i++)
        out[i] = (float)(1.0 / sqrt((double)x[i]));
}

static void do_cos(float *out, const float *x, int n) {
    for (int i = 0; i < n; i++)
        out[i] = (float)cos((double)x[i]);
}

static void do_sin(float *out, const float *x, int n) {
    for (int i = 0; i < n; i++)
        out[i] = (float)sin((double)x[i]);
}

typedef float (*BinaryFn)(float, float);
static float fn_add(float a, float b) { return a + b; }
static float fn_sub(float a, float b) { return a - b; }
static float fn_mul(float a, float b) { return a * b; }
static float fn_div(float a, float b) { return a / b; }
static float fn_max(float a, float b) { return a > b ? a : b; }

static void do_binop(float *out,
                     const float *a, const int *a_shape, int a_ndim,
                     const float *b, const int *b_shape, int b_ndim,
                     const int *out_shape, int out_ndim, int out_size,
                     BinaryFn fn) {
    /* Fast path: same shape (no broadcasting needed) */
    if (a_ndim == b_ndim && a_ndim == out_ndim) {
        int same = 1;
        for (int d = 0; d < a_ndim; d++) {
            if (a_shape[d] != b_shape[d]) { same = 0; break; }
        }
        if (same) {
            for (int i = 0; i < out_size; i++)
                out[i] = fn(a[i], b[i]);
            return;
        }
    }

    /* General broadcasting path */
    int a_strides[MAX_DIMS], b_strides[MAX_DIMS];
    compute_strides(a_shape, a_ndim, a_strides);
    compute_strides(b_shape, b_ndim, b_strides);
    int idx[MAX_DIMS];
    for (int i = 0; i < out_size; i++) {
        linear_to_multi(i, out_shape, out_ndim, idx);
        int ai = broadcast_linear(a_shape, a_ndim, a_strides, idx, out_ndim);
        int bi = broadcast_linear(b_shape, b_ndim, b_strides, idx, out_ndim);
        out[i] = fn(a[ai], b[bi]);
    }
}

static void do_sum(float *out, const float *x,
                   const int *shape, int ndim, int axis) {
    int n = total_size(shape, ndim);
    int axis_size = shape[axis];
    int strides[MAX_DIMS];
    compute_strides(shape, ndim, strides);
    int axis_stride = strides[axis];
    int out_size = n / axis_size;

    int out_shape[MAX_DIMS];
    int out_ndim = 0;
    for (int d = 0; d < ndim; d++)
        if (d != axis) out_shape[out_ndim++] = shape[d];

    for (int oi = 0; oi < out_size; oi++) {
        int oidx[MAX_DIMS];
        if (out_ndim > 0) linear_to_multi(oi, out_shape, out_ndim, oidx);
        int in_base[MAX_DIMS];
        int od = 0;
        for (int d = 0; d < ndim; d++) {
            if (d == axis) in_base[d] = 0;
            else in_base[d] = oidx[od++];
        }
        int base_lin = 0;
        for (int d = 0; d < ndim; d++)
            base_lin += in_base[d] * strides[d];
        float acc = 0.0f;
        for (int a = 0; a < axis_size; a++)
            acc += x[base_lin + a * axis_stride];
        out[oi] = acc;
    }
}

static void do_max_reduce(float *out, const float *x,
                          const int *shape, int ndim, int axis) {
    int n = total_size(shape, ndim);
    int axis_size = shape[axis];
    int strides[MAX_DIMS];
    compute_strides(shape, ndim, strides);
    int axis_stride = strides[axis];
    int out_size = n / axis_size;

    int out_shape[MAX_DIMS];
    int out_ndim = 0;
    for (int d = 0; d < ndim; d++)
        if (d != axis) out_shape[out_ndim++] = shape[d];

    for (int oi = 0; oi < out_size; oi++) {
        int oidx[MAX_DIMS];
        if (out_ndim > 0) linear_to_multi(oi, out_shape, out_ndim, oidx);
        int in_base[MAX_DIMS];
        int od = 0;
        for (int d = 0; d < ndim; d++) {
            if (d == axis) in_base[d] = 0;
            else in_base[d] = oidx[od++];
        }
        int base_lin = 0;
        for (int d = 0; d < ndim; d++)
            base_lin += in_base[d] * strides[d];
        float mx = x[base_lin];
        for (int a = 1; a < axis_size; a++) {
            float v = x[base_lin + a * axis_stride];
            if (v > mx) mx = v;
        }
        out[oi] = mx;
    }
}

static void do_mean(float *out, const float *x,
                    const int *shape, int ndim, int axis) {
    int axis_size = shape[axis];
    do_sum(out, x, shape, ndim, axis);
    int out_size = total_size(shape, ndim) / axis_size;
    float scale = 1.0f / (float)axis_size;
    for (int i = 0; i < out_size; i++)
        out[i] *= scale;
}

static void do_transpose(float *out, const float *x,
                         const int *shape, int ndim, const int *axes) {
    int n = total_size(shape, ndim);
    int strides[MAX_DIMS];
    compute_strides(shape, ndim, strides);
    int out_shape[MAX_DIMS], out_strides[MAX_DIMS];
    for (int i = 0; i < ndim; i++)
        out_shape[i] = shape[axes[i]];
    compute_strides(out_shape, ndim, out_strides);
    int out_idx[MAX_DIMS];
    for (int oi = 0; oi < n; oi++) {
        linear_to_multi(oi, out_shape, ndim, out_idx);
        int in_idx[MAX_DIMS];
        for (int d = 0; d < ndim; d++)
            in_idx[axes[d]] = out_idx[d];
        int in_lin = 0;
        for (int d = 0; d < ndim; d++)
            in_lin += in_idx[d] * strides[d];
        out[oi] = x[in_lin];
    }
}

static void do_repeat(float *out, const float *x,
                      const int *shape, int ndim, int repeats, int axis) {
    int strides[MAX_DIMS];
    compute_strides(shape, ndim, strides);
    int out_shape[MAX_DIMS];
    for (int d = 0; d < ndim; d++) out_shape[d] = shape[d];
    out_shape[axis] *= repeats;
    int out_n = total_size(out_shape, ndim);
    int out_idx[MAX_DIMS];
    for (int oi = 0; oi < out_n; oi++) {
        linear_to_multi(oi, out_shape, ndim, out_idx);
        int in_lin = 0;
        for (int d = 0; d < ndim; d++) {
            int didx = (d == axis) ? out_idx[d] / repeats : out_idx[d];
            in_lin += didx * strides[d];
        }
        out[oi] = x[in_lin];
    }
}

static void do_slice(float *out, const float *x,
                     const int *shape, int ndim,
                     const int *starts, const int *steps,
                     const int *out_shape, int out_ndim) {
    int out_n = total_size(out_shape, out_ndim);
    int strides[MAX_DIMS];
    compute_strides(shape, ndim, strides);
    int out_idx[MAX_DIMS];
    for (int oi = 0; oi < out_n; oi++) {
        linear_to_multi(oi, out_shape, out_ndim, out_idx);
        int in_lin = 0;
        for (int d = 0; d < ndim; d++) {
            int in_d = starts[d] + out_idx[d] * steps[d];
            in_lin += in_d * strides[d];
        }
        out[oi] = x[in_lin];
    }
}

static void do_matmul_q8(float *out, const float *x, int M, int K,
                         const int8_t *W, int N, const float *scales) {
    for (int i = 0; i < M; i++) {
        const float *x_row = x + i * K;
        float *o_row = out + i * N;
        memset(o_row, 0, N * sizeof(float));
        for (int k = 0; k < K; k++) {
            float xk = x_row[k];
            const int8_t *w_row = W + k * N;
            for (int j = 0; j < N; j++)
                o_row[j] += xk * (float)w_row[j];
        }
        for (int j = 0; j < N; j++)
            o_row[j] *= scales[j];
    }
}

/* ================================================================
 * Public API: Tape context management
 * ================================================================ */

TapeCtx *tape_ctx_create(int n_slots, int n_instrs) {
    TapeCtx *ctx = (TapeCtx *)calloc(1, sizeof(TapeCtx));
    ctx->n_slots = n_slots;
    ctx->n_instrs = n_instrs;
    ctx->slots = (Slot *)calloc(n_slots, sizeof(Slot));
    ctx->tape = (TapeInstr *)calloc(n_instrs, sizeof(TapeInstr));
    return ctx;
}

void tape_ctx_destroy(TapeCtx *ctx) {
    if (!ctx) return;
    for (int i = 0; i < ctx->n_slots; i++) {
        if (ctx->slots[i].owns_data && ctx->slots[i].data)
            free(ctx->slots[i].data);
    }
    free(ctx->slots);
    free(ctx->tape);
    free(ctx);
}

/* Set slot shape and allocate data buffer */
void tape_slot_alloc(TapeCtx *ctx, int slot_idx,
                     const int *shape, int ndim) {
    Slot *s = &ctx->slots[slot_idx];
    s->ndim = ndim;
    s->size = 1;
    for (int d = 0; d < ndim; d++) {
        s->shape[d] = shape[d];
        s->size *= shape[d];
    }
    s->data = (float *)calloc(s->size, sizeof(float));
    s->owns_data = 1;
}

/* Point slot at external data (no ownership) */
void tape_slot_set_external(TapeCtx *ctx, int slot_idx,
                            float *data, const int *shape, int ndim) {
    Slot *s = &ctx->slots[slot_idx];
    if (s->owns_data && s->data) free(s->data);
    s->data = data;
    s->ndim = ndim;
    s->size = 1;
    for (int d = 0; d < ndim; d++) {
        s->shape[d] = shape[d];
        s->size *= shape[d];
    }
    s->owns_data = 0;
}

/* Copy data into a slot (for INPUT nodes) */
void tape_slot_write(TapeCtx *ctx, int slot_idx,
                     const float *data, int n) {
    Slot *s = &ctx->slots[slot_idx];
    /* Reallocate if input size changed (KV cache grows) */
    if (n > s->size) {
        if (s->owns_data && s->data) free(s->data);
        s->data = (float *)malloc(n * sizeof(float));
        s->size = n;
        s->owns_data = 1;
    }
    memcpy(s->data, data, n * sizeof(float));
}

/* Update slot shape (for concat results that grow) */
void tape_slot_set_shape(TapeCtx *ctx, int slot_idx,
                         const int *shape, int ndim) {
    Slot *s = &ctx->slots[slot_idx];
    s->ndim = ndim;
    s->size = 1;
    for (int d = 0; d < ndim; d++) {
        s->shape[d] = shape[d];
        s->size *= shape[d];
    }
}

/* Read slot data pointer and size (for output extraction) */
const float *tape_slot_read(TapeCtx *ctx, int slot_idx, int *out_size) {
    *out_size = ctx->slots[slot_idx].size;
    return ctx->slots[slot_idx].data;
}

/* Read slot shape (ndim + shape array). Returns ndim. */
int tape_slot_read_shape(TapeCtx *ctx, int slot_idx, int *out_shape) {
    Slot *s = &ctx->slots[slot_idx];
    for (int d = 0; d < s->ndim; d++) out_shape[d] = s->shape[d];
    return s->ndim;
}

/* Get pointer to tape instruction for in-place setup */
TapeInstr *tape_get_instr(TapeCtx *ctx, int instr_idx) {
    return &ctx->tape[instr_idx];
}

/* ================================================================
 * Main tape runner — one C call per decode step
 * ================================================================ */

/* Ensure a slot has at least 'need' float elements allocated */
static void ensure_slot(Slot *s, int need) {
    if (need > s->size) {
        if (s->owns_data && s->data) free(s->data);
        s->data = (float *)malloc(need * sizeof(float));
        s->owns_data = 1;
        s->size = need;
    }
}

/* Copy shape from src to dst, update size */
static void copy_shape(Slot *dst, const Slot *src) {
    dst->ndim = src->ndim;
    int sz = 1;
    for (int d = 0; d < src->ndim; d++) {
        dst->shape[d] = src->shape[d];
        sz *= src->shape[d];
    }
    dst->size = sz;
}

/* Compute broadcast output shape from two inputs (right-aligned) */
static int broadcast_shapes(const int *a, int an, const int *b, int bn,
                            int *out, int *out_ndim) {
    int nd = (an > bn) ? an : bn;
    *out_ndim = nd;
    int sz = 1;
    for (int d = 0; d < nd; d++) {
        int ad = (d < an) ? a[an - 1 - d] : 1;
        int bd = (d < bn) ? b[bn - 1 - d] : 1;
        int od = (ad > bd) ? ad : bd;
        out[nd - 1 - d] = od;
        sz *= od;
    }
    return sz;
}

void tape_run(TapeCtx *ctx) {
    Slot *slots = ctx->slots;
    TapeInstr *tape = ctx->tape;
    int n = ctx->n_instrs;

    for (int i = 0; i < n; i++) {
        TapeInstr *op = &tape[i];
        Slot *out = &slots[op->out_slot];
        Slot *s0  = (op->in0 >= 0) ? &slots[op->in0] : NULL;
        Slot *s1  = (op->in1 >= 0) ? &slots[op->in1] : NULL;

        switch (op->tag) {
        case T_LUT_SILU:
            ensure_slot(out, s0->size); copy_shape(out, s0);
            do_silu(out->data, s0->data, s0->size);
            break;
        case T_LUT_EXP:
            ensure_slot(out, s0->size); copy_shape(out, s0);
            do_exp(out->data, s0->data, s0->size);
            break;
        case T_LUT_RSQRT:
            ensure_slot(out, s0->size); copy_shape(out, s0);
            do_rsqrt(out->data, s0->data, s0->size);
            break;
        case T_LUT_COS:
            ensure_slot(out, s0->size); copy_shape(out, s0);
            do_cos(out->data, s0->data, s0->size);
            break;
        case T_LUT_SIN:
            ensure_slot(out, s0->size); copy_shape(out, s0);
            do_sin(out->data, s0->data, s0->size);
            break;

        case T_ADD: case T_SUB: case T_MUL: case T_DIV: case T_MAX: {
            int bs[MAX_DIMS], bnd, bsz;
            bsz = broadcast_shapes(s0->shape, s0->ndim, s1->shape, s1->ndim,
                                   bs, &bnd);
            ensure_slot(out, bsz);
            float (*fn)(float, float) =
                (op->tag == T_ADD) ? fn_add :
                (op->tag == T_SUB) ? fn_sub :
                (op->tag == T_MUL) ? fn_mul :
                (op->tag == T_DIV) ? fn_div : fn_max;
            do_binop(out->data, s0->data, s0->shape, s0->ndim,
                     s1->data, s1->shape, s1->ndim,
                     bs, bnd, bsz, fn);
            out->ndim = bnd;
            for (int d = 0; d < bnd; d++) out->shape[d] = bs[d];
            out->size = bsz;
            break;
        }

        case T_NEG: {
            ensure_slot(out, s0->size); copy_shape(out, s0);
            const uint32_t *xi = (const uint32_t *)s0->data;
            uint32_t *oi = (uint32_t *)out->data;
            for (int j = 0; j < s0->size; j++)
                oi[j] = xi[j] ^ 0x80000000u;
            break;
        }
        case T_SQUARE:
            ensure_slot(out, s0->size); copy_shape(out, s0);
            for (int j = 0; j < s0->size; j++)
                out->data[j] = s0->data[j] * s0->data[j];
            break;

        case T_MATMUL: {
            /* BLAS sgemm — handles batched via loop */
            int a_ndim = s0->ndim, b_ndim = s1->ndim;
            int m = s0->shape[a_ndim - 2];
            int k = s0->shape[a_ndim - 1];
            int nn = s1->shape[b_ndim - 1];
            int batch = 1;
            for (int d = 0; d < a_ndim - 2; d++) batch *= s0->shape[d];
            int o_total = batch * m * nn;
            ensure_slot(out, o_total);
            /* Set output shape */
            out->ndim = a_ndim;
            for (int d = 0; d < a_ndim - 2; d++) out->shape[d] = s0->shape[d];
            out->shape[a_ndim - 2] = m;
            out->shape[a_ndim - 1] = nn;
            out->size = o_total;
            int a_mat = m * k, b_mat = k * nn, o_mat = m * nn;
            for (int bi = 0; bi < batch; bi++) {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            m, nn, k,
                            1.0f, s0->data + bi * a_mat, k,
                                  s1->data + bi * b_mat, nn,
                            0.0f, out->data + bi * o_mat, nn);
            }
            break;
        }

        case T_MATMUL_Q8: {
            int M_q8 = s0->shape[0];
            int out_sz = M_q8 * op->N_q8;
            ensure_slot(out, out_sz);
            out->ndim = 2; out->shape[0] = M_q8; out->shape[1] = op->N_q8;
            out->size = out_sz;
            do_matmul_q8(out->data, s0->data,
                         M_q8, op->K_q8,
                         op->wq8_ptr, op->N_q8,
                         op->scales_ptr);
            break;
        }

        case T_SUM: case T_MAX_RED: case T_MEAN: {
            /* Compute reduced output size */
            int red_out = s0->size / s0->shape[op->axis];
            if (op->keepdims) red_out = s0->size / s0->shape[op->axis];
            ensure_slot(out, red_out);
            /* Set output shape */
            int od = 0;
            for (int d = 0; d < s0->ndim; d++) {
                if (d == op->axis) {
                    if (op->keepdims) out->shape[od++] = 1;
                } else {
                    out->shape[od++] = s0->shape[d];
                }
            }
            out->ndim = od; out->size = red_out;
            if (op->tag == T_SUM)
                do_sum(out->data, s0->data, s0->shape, s0->ndim, op->axis);
            else if (op->tag == T_MAX_RED)
                do_max_reduce(out->data, s0->data, s0->shape, s0->ndim, op->axis);
            else
                do_mean(out->data, s0->data, s0->shape, s0->ndim, op->axis);
            break;
        }

        case T_ARGMAX: {
            /* Output is int, stored as float for uniform slot type */
            int axis_size = s0->shape[op->axis];
            int strides[MAX_DIMS];
            compute_strides(s0->shape, s0->ndim, strides);
            int axis_stride = strides[op->axis];
            int out_size_a = s0->size / axis_size;
            ensure_slot(out, out_size_a);
            int o_shape[MAX_DIMS]; int o_ndim = 0;
            for (int d = 0; d < s0->ndim; d++)
                if (d != op->axis) o_shape[o_ndim++] = s0->shape[d];
            out->ndim = o_ndim; out->size = out_size_a;
            for (int d = 0; d < o_ndim; d++) out->shape[d] = o_shape[d];
            for (int oi = 0; oi < out_size_a; oi++) {
                int oidx[MAX_DIMS];
                if (o_ndim > 0) linear_to_multi(oi, o_shape, o_ndim, oidx);
                int in_base[MAX_DIMS]; int od = 0;
                for (int d = 0; d < s0->ndim; d++) {
                    if (d == op->axis) in_base[d] = 0;
                    else in_base[d] = oidx[od++];
                }
                int base_lin = 0;
                for (int d = 0; d < s0->ndim; d++)
                    base_lin += in_base[d] * strides[d];
                float mx = s0->data[base_lin]; int mi = 0;
                for (int a = 1; a < axis_size; a++) {
                    float v = s0->data[base_lin + a * axis_stride];
                    if (v > mx) { mx = v; mi = a; }
                }
                out->data[oi] = (float)mi;
            }
            break;
        }

        case T_RESHAPE:
            ensure_slot(out, s0->size);
            if (out->data != s0->data)
                memcpy(out->data, s0->data, s0->size * sizeof(float));
            /* Shape comes from op->out_shape (static) but size from input */
            out->ndim = op->out_ndim;
            for (int d = 0; d < op->out_ndim; d++)
                out->shape[d] = op->out_shape[d];
            out->size = s0->size;
            break;

        case T_TRANSPOSE: {
            int t_sz = s0->size;
            ensure_slot(out, t_sz);
            out->ndim = s0->ndim; out->size = t_sz;
            for (int d = 0; d < s0->ndim; d++)
                out->shape[d] = s0->shape[op->axes[d]];
            do_transpose(out->data, s0->data,
                         s0->shape, s0->ndim, op->axes);
            break;
        }

        case T_CONCAT: {
            int axis = op->axis;
            int nc = op->concat_n_inputs;
            /* Compute required output size from input slots */
            int need = 0;
            for (int ci = 0; ci < nc; ci++)
                need += slots[op->concat_inputs[ci]].size;
            /* Grow output slot if needed (KV cache grows each step) */
            if (need > out->size) {
                if (out->owns_data) free(out->data);
                out->data = (float *)malloc(need * sizeof(float));
                out->owns_data = 1;
                out->size = need;
            }
            /* Copy shape from first input, then fix concat axis */
            Slot *first_in = &slots[op->concat_inputs[0]];
            out->ndim = first_in->ndim;
            int total_ax = 0;
            for (int ci = 0; ci < nc; ci++)
                total_ax += slots[op->concat_inputs[ci]].shape[axis];
            for (int d = 0; d < first_in->ndim; d++)
                out->shape[d] = first_in->shape[d];
            out->shape[axis] = total_ax;

            if (axis == 0 || first_in->ndim == 1) {
                /* axis=0: just concatenate flat */
                int off = 0;
                for (int ci = 0; ci < nc; ci++) {
                    Slot *si = &slots[op->concat_inputs[ci]];
                    memcpy(out->data + off, si->data,
                           si->size * sizeof(float));
                    off += si->size;
                }
            } else {
                /* General axis concat */
                int outer = 1, inner = 1;
                for (int d = 0; d < axis; d++) outer *= first_in->shape[d];
                for (int d = axis + 1; d < first_in->ndim; d++)
                    inner *= first_in->shape[d];
                int out_off = 0;
                for (int o = 0; o < outer; o++) {
                    for (int ci = 0; ci < nc; ci++) {
                        Slot *si = &slots[op->concat_inputs[ci]];
                        int chunk = si->shape[axis] * inner;
                        int in_off = o * chunk;
                        memcpy(out->data + out_off, si->data + in_off,
                               chunk * sizeof(float));
                        out_off += chunk;
                    }
                }
            }
            break;
        }

        case T_REPEAT: {
            int r_sz = s0->size * op->repeats;
            ensure_slot(out, r_sz);
            copy_shape(out, s0);
            out->shape[op->axis] *= op->repeats;
            out->size = r_sz;
            do_repeat(out->data, s0->data,
                      s0->shape, s0->ndim,
                      op->repeats, op->axis);
            break;
        }

        case T_SLICE: {
            /* Recompute slice output shape from current input shape */
            int sl_shape[MAX_DIMS], sl_nd = s0->ndim, sl_sz = 1;
            for (int d = 0; d < s0->ndim; d++) {
                int start = op->starts[d];
                int step = op->steps[d];
                int stop = s0->shape[d]; /* default: full dim */
                if (d < op->slice_out_ndim) {
                    /* Use static stop if available */
                    int static_dim = op->slice_out_shape[d];
                    if (static_dim > 0 && static_dim <= s0->shape[d])
                        stop = start + static_dim * step;
                }
                int dim = (stop - start + step - 1) / step;
                if (dim < 0) dim = 0;
                sl_shape[d] = dim;
                sl_sz *= dim;
            }
            ensure_slot(out, sl_sz);
            out->ndim = sl_nd; out->size = sl_sz;
            for (int d = 0; d < sl_nd; d++) out->shape[d] = sl_shape[d];
            do_slice(out->data, s0->data,
                     s0->shape, s0->ndim,
                     op->starts, op->steps,
                     sl_shape, sl_nd);
            break;
        }

        case T_COPY:
            ensure_slot(out, s0->size); copy_shape(out, s0);
            memcpy(out->data, s0->data, s0->size * sizeof(float));
            break;
        }
    }
}
