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
#include <stdio.h>
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
        if (ctx->slots[i].owns_data && ctx->slots[i].data) {
            free(ctx->slots[i].data);
        }
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
    if (n <= 0) return; /* nothing to write */
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
    if (need < 1) need = 1; /* always allocate at least 1 element */
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
            /* Guard against zero-size dimensions (Apple Accelerate
               BLAS doesn't support N=0 in cblas_sgemm). */
            if (m == 0 || k == 0 || nn == 0 || batch == 0) {
                if (o_total > 0)
                    memset(out->data, 0, o_total * sizeof(float));
                break;
            }
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


/* ================================================================
 * BPE Tokenizer — encode (bytes → token IDs) / decode (IDs → bytes)
 *
 * All data comes from ROM arrays compiled into the chip:
 *   hash_keys  : uint8[table_size][max_piece_len] — vocab hash table
 *   hash_vals  : int32[table_size]                 — token IDs
 *   hash_lens  : int32[table_size]                 — piece lengths
 *   merge_a/b  : int32[num_merges]                 — merge pair IDs
 *   merge_result : int32[num_merges]               — merged token ID
 *   id_to_bytes  : uint8[N]                        — flat byte pool
 *   id_to_offsets: int32[vocab_size][2]             — (offset, length)
 *   special_ids  : int32[num_special]               — skip set
 * ================================================================ */

/* FNV-1a hash (32-bit), same as Python _fnv1a() */
static uint32_t fnv1a(const uint8_t *data, int len) {
    uint32_t h = 2166136261u;
    for (int i = 0; i < len; i++) {
        h ^= data[i];
        h *= 16777619u;
    }
    return h;
}

/* Vocab hash lookup — open-addressing, linear probing.
 * Returns token ID or -1 if not found. */
static int vocab_lookup(
    const uint8_t *piece, int piece_len,
    const uint8_t *hash_keys, const int32_t *hash_vals,
    const int32_t *hash_lens,
    int table_size, int max_piece_len)
{
    if (table_size == 0 || piece_len > max_piece_len) return -1;
    uint32_t h = fnv1a(piece, piece_len);
    int idx = (int)(h % (uint32_t)table_size);
    for (int probe = 0; probe < table_size; probe++) {
        int stored_len = hash_lens[idx];
        if (stored_len == 0) return -1; /* empty slot */
        if (stored_len == piece_len) {
            const uint8_t *stored = hash_keys + idx * max_piece_len;
            if (memcmp(stored, piece, piece_len) == 0)
                return hash_vals[idx];
        }
        idx = (idx + 1) % table_size;
    }
    return -1;
}

/* Metaspace: ▁ = UTF-8 bytes 0xE2 0x96 0x81 */
#define META_B0 0xE2
#define META_B1 0x96
#define META_B2 0x81
#define META_LEN 3

/* Check if byte at pos starts a metaspace sequence */
static int is_meta(const uint8_t *buf, int pos, int len) {
    return (pos + 2 < len &&
            buf[pos] == META_B0 &&
            buf[pos+1] == META_B1 &&
            buf[pos+2] == META_B2);
}

/* Count UTF-8 character byte length starting at buf[pos] */
static int utf8_char_len(uint8_t b) {
    if (b < 0x80) return 1;
    if ((b & 0xE0) == 0xC0) return 2;
    if ((b & 0xF0) == 0xE0) return 3;
    if ((b & 0xF8) == 0xF0) return 4;
    return 1; /* invalid — treat as single byte */
}

/* Encode a segment of regular text (no special tokens) using BPE.
 * If prepend_meta is true, prepend metaspace ▁ to the text.
 * Returns number of tokens written to out_ids[]. */
static int _bpe_encode_segment(
    const uint8_t *raw_bytes, int raw_len,
    const uint8_t *hash_keys, const int32_t *hash_vals,
    const int32_t *hash_lens,
    int table_size, int max_piece_len,
    const int32_t *merge_a, const int32_t *merge_b,
    const int32_t *merge_result, int num_merges,
    int max_tokens, int prepend_meta,
    int32_t *out_ids)
{
    if (raw_len <= 0 || max_tokens <= 0) return 0;

    /* Phase 1: build metaspace-transformed buffer.
     * Optionally prepend metaspace, then replace all spaces with metaspace. */
    int max_transformed = raw_len * 3 + META_LEN + 16;
    uint8_t *tbuf = (uint8_t *)malloc(max_transformed);
    int tlen = 0;

    if (prepend_meta) {
        tbuf[tlen++] = META_B0; tbuf[tlen++] = META_B1; tbuf[tlen++] = META_B2;
    }

    for (int i = 0; i < raw_len; i++) {
        if (raw_bytes[i] == ' ') {
            tbuf[tlen++] = META_B0; tbuf[tlen++] = META_B1; tbuf[tlen++] = META_B2;
        } else {
            tbuf[tlen++] = raw_bytes[i];
        }
    }

    /* Phase 2: split into words at metaspace boundaries. */
    int n_output = 0;
    int *word_ids = (int *)malloc((tlen + 1) * sizeof(int));

    int wstart = 0;
    while (wstart < tlen) {
        int wend = wstart;

        if (is_meta(tbuf, wend, tlen))
            wend += META_LEN;

        while (wend < tlen) {
            if (is_meta(tbuf, wend, tlen))
                break;
            int clen = utf8_char_len(tbuf[wend]);
            wend += clen;
        }

        if (wend <= wstart) { wstart = wend + 1; continue; }

        /* Phase 3: char-level tokenization of this word */
        int n_word = 0;
        int cpos = wstart;
        while (cpos < wend) {
            int clen;
            if (is_meta(tbuf, cpos, tlen)) {
                clen = META_LEN;
            } else {
                clen = utf8_char_len(tbuf[cpos]);
                if (cpos + clen > wend) clen = wend - cpos;
            }
            int tid = vocab_lookup(tbuf + cpos, clen,
                                   hash_keys, hash_vals, hash_lens,
                                   table_size, max_piece_len);
            if (tid >= 0) {
                word_ids[n_word++] = tid;
            } else {
                for (int b = 0; b < clen; b++) {
                    char hex_str[8];
                    int slen = snprintf(hex_str, sizeof(hex_str),
                                       "<0x%02X>", tbuf[cpos + b]);
                    int fb_id = vocab_lookup(
                        (const uint8_t *)hex_str, slen,
                        hash_keys, hash_vals, hash_lens,
                        table_size, max_piece_len);
                    if (fb_id >= 0) word_ids[n_word++] = fb_id;
                }
            }
            cpos += clen;
        }

        /* Phase 4: BPE merge loop */
        while (n_word > 1) {
            int best_rank = num_merges;
            int best_a = -1, best_b = -1, best_result = -1;

            for (int i = 0; i < n_word - 1; i++) {
                int a = word_ids[i], b = word_ids[i + 1];
                for (int r = 0; r < best_rank; r++) {
                    if (merge_a[r] == a && merge_b[r] == b) {
                        best_rank = r;
                        best_a = a;
                        best_b = b;
                        best_result = merge_result[r];
                        break;
                    }
                }
            }

            if (best_a < 0) break;

            int new_n = 0;
            int i = 0;
            while (i < n_word) {
                if (i < n_word - 1 &&
                    word_ids[i] == best_a && word_ids[i+1] == best_b) {
                    word_ids[new_n++] = best_result;
                    i += 2;
                } else {
                    word_ids[new_n++] = word_ids[i];
                    i++;
                }
            }
            n_word = new_n;
        }

        for (int i = 0; i < n_word && n_output < max_tokens; i++) {
            out_ids[n_output++] = word_ids[i];
        }

        wstart = wend;
    }

    free(word_ids);
    free(tbuf);
    return n_output;
}

/* Pre-tokenize + BPE encode: raw UTF-8 bytes → token IDs.
 *
 * Splits the input at special token boundaries (e.g. </s>) and
 * encodes each regular text segment with BPE.  Special tokens
 * are emitted directly by their ID.
 *
 * Returns number of tokens written to out_ids[].
 */
int bpe_encode(
    const uint8_t *raw_bytes, int raw_len,
    /* ROMs */
    const uint8_t *hash_keys, const int32_t *hash_vals,
    const int32_t *hash_lens,
    int table_size, int max_piece_len,
    const int32_t *merge_a, const int32_t *merge_b,
    const int32_t *merge_result, int num_merges,
    int bos_token_id, int max_tokens,
    /* Special token handling */
    const uint8_t *id_to_bytes_rom, const int32_t *id_to_offsets,
    const int32_t *special_ids, int num_special, int vocab_size,
    /* Output */
    int32_t *out_ids)
{
    int n_output = 0;
    out_ids[n_output++] = bos_token_id;

    if (raw_len <= 0) return n_output;

    /* Build table of special token byte strings (skip BOS). */
    const uint8_t **sp_str = NULL;
    int *sp_len = NULL;
    int *sp_tok_ids = NULL;
    int n_sp = 0;
    if (id_to_bytes_rom && id_to_offsets && special_ids && num_special > 0) {
        sp_str = (const uint8_t **)malloc(num_special * sizeof(uint8_t *));
        sp_len = (int *)malloc(num_special * sizeof(int));
        sp_tok_ids = (int *)malloc(num_special * sizeof(int));
        for (int s = 0; s < num_special; s++) {
            int sid = special_ids[s];
            if (sid == bos_token_id) continue;
            if (sid < 0 || sid >= vocab_size) continue;
            int off = id_to_offsets[sid * 2];
            int slen = id_to_offsets[sid * 2 + 1];
            if (slen <= 0) continue;
            sp_str[n_sp] = id_to_bytes_rom + off;
            sp_len[n_sp] = slen;
            sp_tok_ids[n_sp] = sid;
            n_sp++;
        }
    }

    int pos = 0;
    int first_segment = 1;

    while (pos < raw_len && n_output < max_tokens) {
        /* Try to match a special token at current position */
        int matched = -1;
        int matched_len = 0;
        for (int s = 0; s < n_sp; s++) {
            int slen = sp_len[s];
            if (pos + slen > raw_len) continue;
            if (slen > matched_len &&
                memcmp(raw_bytes + pos, sp_str[s], slen) == 0) {
                matched = s;
                matched_len = slen;
            }
        }

        if (matched >= 0) {
            out_ids[n_output++] = sp_tok_ids[matched];
            pos += matched_len;
            first_segment = 0;
        } else {
            /* Find extent of regular text until next special token */
            int seg_end = pos + 1;
            while (seg_end < raw_len) {
                int found = 0;
                for (int s = 0; s < n_sp && !found; s++) {
                    int slen = sp_len[s];
                    if (seg_end + slen > raw_len) continue;
                    if (memcmp(raw_bytes + seg_end, sp_str[s], slen) == 0)
                        found = 1;
                }
                if (found) break;
                seg_end++;
            }

            int wrote = _bpe_encode_segment(
                raw_bytes + pos, seg_end - pos,
                hash_keys, hash_vals, hash_lens,
                table_size, max_piece_len,
                merge_a, merge_b, merge_result, num_merges,
                max_tokens - n_output, first_segment,
                out_ids + n_output);
            n_output += wrote;
            pos = seg_end;
            first_segment = 0;
        }
    }

    free(sp_str);
    free(sp_len);
    free(sp_tok_ids);
    return n_output;
}


/* BPE decode: token IDs → UTF-8 bytes.
 * Returns number of bytes written to out_bytes[]. */
int bpe_decode(
    const int32_t *token_ids, int num_tokens,
    /* ROMs */
    const uint8_t *id_to_bytes, const int32_t *id_to_offsets,
    int vocab_size,
    const int32_t *special_ids, int num_special,
    int max_bytes,
    int skip_leading_space,
    /* Output */
    uint8_t *out_bytes)
{
    /* First pass: concatenate raw token pieces (with metaspace) */
    uint8_t *raw = (uint8_t *)malloc(max_bytes + 256);
    int raw_len = 0;

    for (int i = 0; i < num_tokens; i++) {
        int tid = token_ids[i];

        /* Skip special tokens */
        int is_special = 0;
        for (int s = 0; s < num_special; s++) {
            if (tid == special_ids[s]) { is_special = 1; break; }
        }
        if (is_special) continue;

        if (tid < 0 || tid >= vocab_size) continue;
        int offset = id_to_offsets[tid * 2];
        int length = id_to_offsets[tid * 2 + 1];
        if (length == 0) continue;

        if (raw_len + length < max_bytes + 256) {
            memcpy(raw + raw_len, id_to_bytes + offset, length);
            raw_len += length;
        }
    }

    /* Second pass: replace metaspace ▁ with space, skip leading space */
    int out_len = 0;
    int skip_first_space = skip_leading_space;
    int i = 0;
    while (i < raw_len && out_len < max_bytes) {
        if (is_meta(raw, i, raw_len)) {
            if (skip_first_space) {
                skip_first_space = 0;
            } else {
                out_bytes[out_len++] = ' ';
            }
            i += META_LEN;
        } else {
            out_bytes[out_len++] = raw[i++];
            skip_first_space = 0;
        }
    }

    /* Third pass: handle <0xHH> byte-fallback sequences.
     * Scan for patterns like <0x41> and replace inline. */
    /* For simplicity, do a copy pass */
    uint8_t *final = (uint8_t *)malloc(out_len + 1);
    int final_len = 0;
    i = 0;
    while (i < out_len) {
        if (i + 5 < out_len &&
            out_bytes[i] == '<' && out_bytes[i+1] == '0' &&
            out_bytes[i+2] == 'x') {
            /* Find closing > */
            int j = i + 3;
            while (j < out_len && out_bytes[j] != '>') j++;
            if (j < out_len && j - (i+3) == 2) {
                /* Parse hex */
                int hi = 0, lo = 0;
                uint8_t c1 = out_bytes[i+3], c2 = out_bytes[i+4];
                if (c1 >= '0' && c1 <= '9') hi = c1 - '0';
                else if (c1 >= 'A' && c1 <= 'F') hi = c1 - 'A' + 10;
                else if (c1 >= 'a' && c1 <= 'f') hi = c1 - 'a' + 10;
                if (c2 >= '0' && c2 <= '9') lo = c2 - '0';
                else if (c2 >= 'A' && c2 <= 'F') lo = c2 - 'A' + 10;
                else if (c2 >= 'a' && c2 <= 'f') lo = c2 - 'a' + 10;
                final[final_len++] = (uint8_t)(hi * 16 + lo);
                i = j + 1;
                continue;
            }
        }
        final[final_len++] = out_bytes[i++];
    }
    memcpy(out_bytes, final, final_len);
    free(final);
    free(raw);
    return final_len;
}


/* ================================================================
 * processor_infer_bytes() — Full bytes-in → streaming bytes-out.
 *
 * Single C call that:
 * 1. BPE-encodes input bytes → token IDs
 * 2. Runs prefill + decode (same as processor_infer)
 * 3. BPE-decodes each generated token → calls callback
 *
 * The callback receives decoded bytes for each generated token,
 * enabling streaming output.
 * ================================================================ */

/* Callback type: called per generated token with decoded bytes.
 * user_data is an opaque pointer passed through.
 * Return 0 to continue, non-zero to stop generation. */
typedef int (*token_callback_fn)(
    const uint8_t *bytes, int byte_len, void *user_data);

int processor_infer_bytes(
    TapeCtx *ctx,
    /* BPE ROMs */
    const uint8_t *hash_keys, const int32_t *hash_vals,
    const int32_t *hash_lens,
    int table_size, int max_piece_len,
    const int32_t *merge_a, const int32_t *merge_b,
    const int32_t *merge_result, int num_merges,
    const uint8_t *id_to_bytes_rom, const int32_t *id_to_offsets,
    const int32_t *special_ids, int num_special,
    int bos_token_id,
    /* Processor resources */
    const float *embed_table, int vocab_size, int hidden_dim,
    const float *rope_cos, const float *rope_sin, int head_dim,
    /* Slot mappings */
    int token_embed_slot, int rope_cos_slot, int rope_sin_slot,
    const int *kv_in_slots, const int *kv_out_slots,
    int logits_slot, int n_layers, int num_kv_heads,
    /* Inference params */
    const uint8_t *input_bytes, int input_len,
    int max_new_tokens, int eos_token_id,
    int max_seq_tokens, int max_bpe_bytes,
    /* Streaming callback */
    token_callback_fn callback, void *user_data,
    /* Output */
    int *total_generated)
{
    /* --- Phase 1: BPE encode --- */
    int32_t *prompt_tokens = (int32_t *)malloc(max_seq_tokens * sizeof(int32_t));
    int prompt_len = bpe_encode(
        input_bytes, input_len,
        hash_keys, hash_vals, hash_lens,
        table_size, max_piece_len,
        merge_a, merge_b, merge_result, num_merges,
        bos_token_id, max_seq_tokens,
        id_to_bytes_rom, id_to_offsets,
        special_ids, num_special, vocab_size,
        prompt_tokens);

    /* --- Phase 2: inference (same logic as processor_infer) --- */
    int kv_dim = num_kv_heads * head_dim;
    float **kv_k_buf = (float **)calloc(n_layers, sizeof(float *));
    float **kv_v_buf = (float **)calloc(n_layers, sizeof(float *));
    int *kv_seq_len = (int *)calloc(n_layers, sizeof(int));
    int *kv_cap = (int *)calloc(n_layers, sizeof(int));

    /* Pre-allocate KV cache with 1 slot (avoids zero-dim) */
    for (int li = 0; li < n_layers; li++) {
        kv_cap[li] = kv_dim * 4; /* initial capacity: 4 positions */
        kv_k_buf[li] = (float *)calloc(kv_cap[li], sizeof(float));
        kv_v_buf[li] = (float *)calloc(kv_cap[li], sizeof(float));
        kv_seq_len[li] = 0;
    }

    float *embed_buf = (float *)malloc(hidden_dim * sizeof(float));
    float *cos_buf = (float *)malloc(head_dim * sizeof(float));
    float *sin_buf = (float *)malloc(head_dim * sizeof(float));

    int embed_shape[2] = {1, hidden_dim};
    int rope_shape[2] = {1, head_dim};

    int position = 0;
    int generated = 0;

    /* Temp buffer for single-token BPE decode */
    uint8_t *dec_buf = (uint8_t *)malloc(max_bpe_bytes);

    #define FEED_TOKEN2(token_id, pos) do { \
        memcpy(embed_buf, embed_table + (token_id) * hidden_dim, \
               hidden_dim * sizeof(float)); \
        tape_slot_write(ctx, token_embed_slot, embed_buf, hidden_dim); \
        tape_slot_set_shape(ctx, token_embed_slot, embed_shape, 2); \
        memcpy(cos_buf, rope_cos + (pos) * head_dim, \
               head_dim * sizeof(float)); \
        memcpy(sin_buf, rope_sin + (pos) * head_dim, \
               head_dim * sizeof(float)); \
        tape_slot_write(ctx, rope_cos_slot, cos_buf, head_dim); \
        tape_slot_set_shape(ctx, rope_cos_slot, rope_shape, 2); \
        tape_slot_write(ctx, rope_sin_slot, sin_buf, head_dim); \
        tape_slot_set_shape(ctx, rope_sin_slot, rope_shape, 2); \
        for (int li = 0; li < n_layers; li++) { \
            int k_in = kv_in_slots[li * 2]; \
            int v_in = kv_in_slots[li * 2 + 1]; \
            int seq = kv_seq_len[li]; \
            int sz = num_kv_heads * seq * head_dim; \
            if (sz > 0) { \
                tape_slot_write(ctx, k_in, kv_k_buf[li], sz); \
                int kv_shape[3] = {num_kv_heads, seq, head_dim}; \
                tape_slot_set_shape(ctx, k_in, kv_shape, 3); \
                tape_slot_write(ctx, v_in, kv_v_buf[li], sz); \
                tape_slot_set_shape(ctx, v_in, kv_shape, 3); \
            } else { \
                float zero = 0.0f; \
                int kv_shape[3] = {num_kv_heads, 0, head_dim}; \
                tape_slot_write(ctx, k_in, &zero, 0); \
                tape_slot_set_shape(ctx, k_in, kv_shape, 3); \
                tape_slot_write(ctx, v_in, &zero, 0); \
                tape_slot_set_shape(ctx, v_in, kv_shape, 3); \
            } \
        } \
        tape_run(ctx); \
        for (int li = 0; li < n_layers; li++) { \
            int k_out = kv_out_slots[li * 2]; \
            int v_out = kv_out_slots[li * 2 + 1]; \
            int out_sz = 0; \
            const float *new_k = tape_slot_read(ctx, k_out, &out_sz); \
            int new_seq = kv_seq_len[li] + 1; \
            int new_total = num_kv_heads * new_seq * head_dim; \
            if (new_total > kv_cap[li]) { \
                int cap = new_total * 2; \
                kv_k_buf[li] = (float *)realloc(kv_k_buf[li], \
                    cap * sizeof(float)); \
                kv_v_buf[li] = (float *)realloc(kv_v_buf[li], \
                    cap * sizeof(float)); \
                kv_cap[li] = cap; \
            } \
            memcpy(kv_k_buf[li], new_k, new_total * sizeof(float)); \
            const float *new_v = tape_slot_read(ctx, v_out, &out_sz); \
            memcpy(kv_v_buf[li], new_v, new_total * sizeof(float)); \
            kv_seq_len[li] = new_seq; \
        } \
    } while(0)

    /* PREFILL */
    for (int i = 0; i < prompt_len; i++) {
        FEED_TOKEN2(prompt_tokens[i], position);
        position++;
    }
    /* DECODE */
    for (int step = 0; step < max_new_tokens; step++) {
        if (position >= max_seq_tokens) break;

        int logits_sz = 0;
        const float *logits = tape_slot_read(ctx, logits_slot, &logits_sz);
        int last_row_start = logits_sz - vocab_size;
        if (last_row_start < 0) last_row_start = 0;
        float best_val = logits[last_row_start];
        int best_id = 0;
        for (int v = 1; v < vocab_size; v++) {
            float val = logits[last_row_start + v];
            if (val > best_val) { best_val = val; best_id = v; }
        }

        if (best_id == eos_token_id) {
            break;
        }

        generated++;

        /* BPE decode this single token and stream via callback */
        if (callback) {
            int32_t single_tok = best_id;
            int dec_len = bpe_decode(
                &single_tok, 1,
                id_to_bytes_rom, id_to_offsets, vocab_size,
                special_ids, num_special,
                max_bpe_bytes, (generated == 1) ? 1 : 0,
                dec_buf);
            int stop = callback(dec_buf, dec_len, user_data);
            if (stop) break;
        }

        FEED_TOKEN2(best_id, position);
        position++;
    }

    #undef FEED_TOKEN2

    *total_generated = generated;

    /* Cleanup */
    for (int li = 0; li < n_layers; li++) {
        free(kv_k_buf[li]);
        free(kv_v_buf[li]);
    }
    free(kv_k_buf); free(kv_v_buf);
    free(kv_seq_len); free(kv_cap);
    free(embed_buf); free(cos_buf); free(sin_buf);
    free(prompt_tokens); free(dec_buf);

    return 0;
}


/* ================================================================
 * processor_infer() — Complete inference loop in C.
 *
 * Runs prefill (one token at a time) then autoregressive decode.
 * Same FSM as the Verilog inference_controller:
 *   PREFILL: for each prompt token → embed → set rope → tape_run → accumulate KV
 *   DECODE:  argmax → eos check → embed → set rope → tape_run → accumulate KV → loop
 *
 * Returns number of generated tokens in *output_len.
 * ================================================================ */

int processor_infer(
    TapeCtx *ctx,
    /* Processor resources (ROMs baked into the chip) */
    const float *embed_table,   /* (vocab_size, hidden_dim) row-major */
    int vocab_size,
    int hidden_dim,
    const float *rope_cos,      /* (max_seq, head_dim) row-major */
    const float *rope_sin,      /* (max_seq, head_dim) row-major */
    int head_dim,
    /* Datapath slot mappings */
    int token_embed_slot,
    int rope_cos_slot,
    int rope_sin_slot,
    const int *kv_in_slots,     /* [n_layers * 2]: K_in, V_in, K_in, V_in, ... */
    const int *kv_out_slots,    /* [n_layers * 2]: K_out, V_out, K_out, V_out, ... */
    int logits_slot,
    int n_layers,
    int num_kv_heads,
    /* Inference parameters */
    const int *prompt_tokens,
    int prompt_len,
    int max_new_tokens,
    int eos_token_id,
    /* Output buffer (caller-allocated, size >= max_new_tokens) */
    int *output_tokens,
    int *output_len
)
{
    /* KV cache buffers per layer: (num_kv_heads, seq_so_far, head_dim) */
    int kv_dim = num_kv_heads * head_dim;
    float **kv_k_buf = (float **)calloc(n_layers, sizeof(float *));
    float **kv_v_buf = (float **)calloc(n_layers, sizeof(float *));
    int *kv_seq_len = (int *)calloc(n_layers, sizeof(int));
    int *kv_cap = (int *)calloc(n_layers, sizeof(int));

    /* Temporary buffer for embedding (1, hidden_dim) */
    float *embed_buf = (float *)malloc(hidden_dim * sizeof(float));
    /* Temporary buffers for RoPE (1, head_dim) */
    float *cos_buf = (float *)malloc(head_dim * sizeof(float));
    float *sin_buf = (float *)malloc(head_dim * sizeof(float));

    int embed_shape[2] = {1, hidden_dim};
    int rope_shape[2] = {1, head_dim};

    int position = 0;
    int generated = 0;

    /* ------ Helper: feed one token through the datapath ------ */
    #define FEED_TOKEN(token_id, pos) do { \
        /* Embed lookup */ \
        memcpy(embed_buf, embed_table + (token_id) * hidden_dim, \
               hidden_dim * sizeof(float)); \
        tape_slot_write(ctx, token_embed_slot, embed_buf, hidden_dim); \
        tape_slot_set_shape(ctx, token_embed_slot, embed_shape, 2); \
        /* RoPE for this position */ \
        memcpy(cos_buf, rope_cos + (pos) * head_dim, \
               head_dim * sizeof(float)); \
        memcpy(sin_buf, rope_sin + (pos) * head_dim, \
               head_dim * sizeof(float)); \
        tape_slot_write(ctx, rope_cos_slot, cos_buf, head_dim); \
        tape_slot_set_shape(ctx, rope_cos_slot, rope_shape, 2); \
        tape_slot_write(ctx, rope_sin_slot, sin_buf, head_dim); \
        tape_slot_set_shape(ctx, rope_sin_slot, rope_shape, 2); \
        /* Feed KV cache for each layer */ \
        for (int li = 0; li < n_layers; li++) { \
            int k_in = kv_in_slots[li * 2]; \
            int v_in = kv_in_slots[li * 2 + 1]; \
            int seq = kv_seq_len[li]; \
            int sz = num_kv_heads * seq * head_dim; \
            if (sz > 0) { \
                tape_slot_write(ctx, k_in, kv_k_buf[li], sz); \
                int kv_shape[3] = {num_kv_heads, seq, head_dim}; \
                tape_slot_set_shape(ctx, k_in, kv_shape, 3); \
                tape_slot_write(ctx, v_in, kv_v_buf[li], sz); \
                tape_slot_set_shape(ctx, v_in, kv_shape, 3); \
            } else { \
                /* Empty cache: write a tiny placeholder */ \
                float zero = 0.0f; \
                int kv_shape[3] = {num_kv_heads, 0, head_dim}; \
                tape_slot_write(ctx, k_in, &zero, 0); \
                tape_slot_set_shape(ctx, k_in, kv_shape, 3); \
                tape_slot_write(ctx, v_in, &zero, 0); \
                tape_slot_set_shape(ctx, v_in, kv_shape, 3); \
            } \
        } \
        /* Run datapath */ \
        tape_run(ctx); \
        /* Extract new KV and append to cache */ \
        for (int li = 0; li < n_layers; li++) { \
            int k_out = kv_out_slots[li * 2]; \
            int v_out = kv_out_slots[li * 2 + 1]; \
            int out_sz = 0; \
            const float *new_k = tape_slot_read(ctx, k_out, &out_sz); \
            /* out_sz = num_kv_heads * (seq+1) * head_dim (concat result) */ \
            int new_seq = kv_seq_len[li] + 1; \
            int new_total = num_kv_heads * new_seq * head_dim; \
            /* Grow buffer if needed */ \
            if (new_total > kv_cap[li]) { \
                int cap = new_total * 2; /* double to amortize */ \
                kv_k_buf[li] = (float *)realloc(kv_k_buf[li], \
                    cap * sizeof(float)); \
                kv_v_buf[li] = (float *)realloc(kv_v_buf[li], \
                    cap * sizeof(float)); \
                kv_cap[li] = cap; \
            } \
            memcpy(kv_k_buf[li], new_k, new_total * sizeof(float)); \
            const float *new_v = tape_slot_read(ctx, v_out, &out_sz); \
            memcpy(kv_v_buf[li], new_v, new_total * sizeof(float)); \
            kv_seq_len[li] = new_seq; \
        } \
    } while(0)

    /* ------ PREFILL: process each prompt token ------ */
    for (int i = 0; i < prompt_len; i++) {
        FEED_TOKEN(prompt_tokens[i], position);
        position++;
    }

    /* ------ DECODE: autoregressive generation ------ */
    for (int step = 0; step < max_new_tokens; step++) {
        /* Argmax over logits[-1] (last row, vocab_size elements) */
        int logits_sz = 0;
        const float *logits = tape_slot_read(ctx, logits_slot, &logits_sz);
        /* logits shape is (1, vocab_size) — last row is at end */
        int last_row_start = logits_sz - vocab_size;
        if (last_row_start < 0) last_row_start = 0;
        float best_val = logits[last_row_start];
        int best_id = 0;
        for (int v = 1; v < vocab_size; v++) {
            float val = logits[last_row_start + v];
            if (val > best_val) {
                best_val = val;
                best_id = v;
            }
        }

        /* EOS check */
        if (best_id == eos_token_id) break;

        output_tokens[step] = best_id;
        generated++;

        /* Feed new token */
        FEED_TOKEN(best_id, position);
        position++;
    }

    #undef FEED_TOKEN

    *output_len = generated;

    /* Cleanup */
    for (int li = 0; li < n_layers; li++) {
        free(kv_k_buf[li]);
        free(kv_v_buf[li]);
    }
    free(kv_k_buf);
    free(kv_v_buf);
    free(kv_seq_len);
    free(kv_cap);
    free(embed_buf);
    free(cos_buf);
    free(sin_buf);

    return 0; /* success */
}
