/*
 * _circuit_eval.c — C tensor operations for kllm circuit evaluation.
 *
 * Replaces NumPy as the compute backend for CircuitGraph evaluation.
 * Each function implements one primitive operation from the graph.
 * The Python wrapper calls these via ctypes, managing the evaluation
 * loop and memory allocation.
 *
 * Compile:
 *   cc -O3 -shared -fPIC -march=native -o _circuit_eval.so _circuit_eval.c -lm
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define MAX_DIMS 8

/* ================================================================
 * Shape / stride utilities
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

/* Convert linear index to multi-dim index */
static void linear_to_multi(int idx, const int *shape, int ndim, int *multi) {
    for (int i = ndim - 1; i >= 0; i--) {
        multi[i] = idx % shape[i];
        idx /= shape[i];
    }
}

/* Compute linear index into tensor with broadcasting.
 * out_idx is a multi-dim index into the output shape (out_ndim dims).
 * Returns the linear index into the input tensor (inp_ndim dims). */
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
 * Element-wise binary ops with broadcasting
 * ================================================================ */

typedef float (*BinaryFn)(float, float);

static float fn_add(float a, float b) { return a + b; }
static float fn_sub(float a, float b) { return a - b; }
static float fn_mul(float a, float b) { return a * b; }
static float fn_div(float a, float b) { return a / b; }
static float fn_max(float a, float b) { return a > b ? a : b; }

static void binary_op(float *out,
                      const float *a, const int *a_shape, int a_ndim,
                      const float *b, const int *b_shape, int b_ndim,
                      const int *out_shape, int out_ndim,
                      BinaryFn fn) {
    int n = total_size(out_shape, out_ndim);

    int a_strides[MAX_DIMS], b_strides[MAX_DIMS];
    compute_strides(a_shape, a_ndim, a_strides);
    compute_strides(b_shape, b_ndim, b_strides);

    int idx[MAX_DIMS];
    for (int i = 0; i < n; i++) {
        linear_to_multi(i, out_shape, out_ndim, idx);
        int ai = broadcast_linear(a_shape, a_ndim, a_strides, idx, out_ndim);
        int bi = broadcast_linear(b_shape, b_ndim, b_strides, idx, out_ndim);
        out[i] = fn(a[ai], b[bi]);
    }
}

/* Exported binary ops */

void ceval_add(float *out,
               const float *a, const int *a_shape, int a_ndim,
               const float *b, const int *b_shape, int b_ndim,
               const int *out_shape, int out_ndim) {
    binary_op(out, a, a_shape, a_ndim, b, b_shape, b_ndim,
              out_shape, out_ndim, fn_add);
}

void ceval_sub(float *out,
               const float *a, const int *a_shape, int a_ndim,
               const float *b, const int *b_shape, int b_ndim,
               const int *out_shape, int out_ndim) {
    binary_op(out, a, a_shape, a_ndim, b, b_shape, b_ndim,
              out_shape, out_ndim, fn_sub);
}

void ceval_mul(float *out,
               const float *a, const int *a_shape, int a_ndim,
               const float *b, const int *b_shape, int b_ndim,
               const int *out_shape, int out_ndim) {
    binary_op(out, a, a_shape, a_ndim, b, b_shape, b_ndim,
              out_shape, out_ndim, fn_mul);
}

void ceval_div(float *out,
               const float *a, const int *a_shape, int a_ndim,
               const float *b, const int *b_shape, int b_ndim,
               const int *out_shape, int out_ndim) {
    binary_op(out, a, a_shape, a_ndim, b, b_shape, b_ndim,
              out_shape, out_ndim, fn_div);
}

void ceval_max(float *out,
               const float *a, const int *a_shape, int a_ndim,
               const float *b, const int *b_shape, int b_ndim,
               const int *out_shape, int out_ndim) {
    binary_op(out, a, a_shape, a_ndim, b, b_shape, b_ndim,
              out_shape, out_ndim, fn_max);
}

/* ================================================================
 * Comparison / selection
 * ================================================================ */

void ceval_cmp_le(float *out,
                  const float *a, const int *a_shape, int a_ndim,
                  const float *b, const int *b_shape, int b_ndim,
                  const int *out_shape, int out_ndim) {
    int n = total_size(out_shape, out_ndim);
    int a_strides[MAX_DIMS], b_strides[MAX_DIMS];
    compute_strides(a_shape, a_ndim, a_strides);
    compute_strides(b_shape, b_ndim, b_strides);
    int idx[MAX_DIMS];
    for (int i = 0; i < n; i++) {
        linear_to_multi(i, out_shape, out_ndim, idx);
        int ai = broadcast_linear(a_shape, a_ndim, a_strides, idx, out_ndim);
        int bi = broadcast_linear(b_shape, b_ndim, b_strides, idx, out_ndim);
        out[i] = (a[ai] <= b[bi]) ? 1.0f : 0.0f;
    }
}

void ceval_mux(float *out, const float *cond, const float *a, const float *b,
               int n) {
    /* cond==0 → a, cond!=0 → b  (same semantics as np.where(cond, b, a)) */
    for (int i = 0; i < n; i++) {
        out[i] = (cond[i] != 0.0f) ? b[i] : a[i];
    }
}

/* ================================================================
 * Unary ops
 * ================================================================ */

void ceval_neg(float *out, const float *x, int n) {
    const uint32_t *xi = (const uint32_t *)x;
    uint32_t *oi = (uint32_t *)out;
    for (int i = 0; i < n; i++)
        oi[i] = xi[i] ^ 0x80000000u;
}

void ceval_abs(float *out, const float *x, int n) {
    const uint32_t *xi = (const uint32_t *)x;
    uint32_t *oi = (uint32_t *)out;
    for (int i = 0; i < n; i++)
        oi[i] = xi[i] & 0x7FFFFFFFu;
}

void ceval_square(float *out, const float *x, int n) {
    for (int i = 0; i < n; i++)
        out[i] = x[i] * x[i];
}

/* ================================================================
 * LUT functions (activation circuits)
 * ================================================================ */

void ceval_silu(float *out, const float *x, int n) {
    for (int i = 0; i < n; i++) {
        double xd = (double)x[i];
        out[i] = (float)(xd / (1.0 + exp(-xd)));
    }
}

void ceval_exp(float *out, const float *x, int n) {
    for (int i = 0; i < n; i++)
        out[i] = (float)exp((double)x[i]);
}

void ceval_rsqrt(float *out, const float *x, int n) {
    for (int i = 0; i < n; i++)
        out[i] = (float)(1.0 / sqrt((double)x[i]));
}

void ceval_cos(float *out, const float *x, int n) {
    for (int i = 0; i < n; i++)
        out[i] = (float)cos((double)x[i]);
}

void ceval_sin(float *out, const float *x, int n) {
    for (int i = 0; i < n; i++)
        out[i] = (float)sin((double)x[i]);
}

/* ================================================================
 * Matrix multiply
 * ================================================================
 *
 * Handles arbitrary-dimension batched matmul:
 *   (..., m, k) @ (..., k, n) → (..., m, n)
 * Batch dims are broadcast.
 */

void ceval_matmul(float *out,
                  const float *a, const int *a_shape, int a_ndim,
                  const float *b, const int *b_shape, int b_ndim,
                  const int *out_shape, int out_ndim) {
    /* Matrix dimensions from the last 2 dims */
    int m = a_shape[a_ndim - 2];
    int k = a_shape[a_ndim - 1];
    int n = b_shape[b_ndim - 1];

    /* Batch dimensions: out_shape[0..out_ndim-3] */
    int batch_ndim = out_ndim - 2;
    int batch_size = 1;
    for (int i = 0; i < batch_ndim; i++)
        batch_size *= out_shape[i];

    /* Strides for batch dims of a and b */
    int a_batch_shape[MAX_DIMS], b_batch_shape[MAX_DIMS];
    int a_batch_strides[MAX_DIMS], b_batch_strides[MAX_DIMS];
    int out_batch_strides[MAX_DIMS];

    /* a batch shape: first (a_ndim - 2) dims */
    int a_batch_ndim = a_ndim - 2;
    int b_batch_ndim = b_ndim - 2;
    for (int i = 0; i < a_batch_ndim; i++)
        a_batch_shape[i] = a_shape[i];
    for (int i = 0; i < b_batch_ndim; i++)
        b_batch_shape[i] = b_shape[i];

    /* Compute batch strides (in terms of matrix blocks, not elements) */
    int a_mat_size = m * k;
    int b_mat_size = k * n;
    int o_mat_size = m * n;

    if (a_batch_ndim > 0) {
        a_batch_strides[a_batch_ndim - 1] = a_mat_size;
        for (int i = a_batch_ndim - 2; i >= 0; i--)
            a_batch_strides[i] = a_batch_strides[i + 1] * a_shape[i + 1];
    }
    if (b_batch_ndim > 0) {
        b_batch_strides[b_batch_ndim - 1] = b_mat_size;
        for (int i = b_batch_ndim - 2; i >= 0; i--)
            b_batch_strides[i] = b_batch_strides[i + 1] * b_shape[i + 1];
    }
    if (batch_ndim > 0) {
        out_batch_strides[batch_ndim - 1] = o_mat_size;
        for (int i = batch_ndim - 2; i >= 0; i--)
            out_batch_strides[i] = out_batch_strides[i + 1] * out_shape[i + 1];
    }

    int batch_idx[MAX_DIMS];
    for (int bi = 0; bi < batch_size; bi++) {
        /* Convert linear batch index to multi-dim index */
        if (batch_ndim > 0) {
            int tmp = bi;
            for (int d = batch_ndim - 1; d >= 0; d--) {
                batch_idx[d] = tmp % out_shape[d];
                tmp /= out_shape[d];
            }
        }

        /* Compute offsets into a and b (with broadcasting) */
        int a_off = 0;
        for (int d = 0; d < a_batch_ndim; d++) {
            int od = d + (batch_ndim - a_batch_ndim);
            int dim_idx = (a_batch_shape[d] == 1) ? 0 : batch_idx[od];
            a_off += dim_idx * a_batch_strides[d];
        }
        int b_off = 0;
        for (int d = 0; d < b_batch_ndim; d++) {
            int od = d + (batch_ndim - b_batch_ndim);
            int dim_idx = (b_batch_shape[d] == 1) ? 0 : batch_idx[od];
            b_off += dim_idx * b_batch_strides[d];
        }
        int o_off = bi * o_mat_size;

        /* Standard 2D matmul: out[i,j] = sum_p a[i,p] * b[p,j] */
        const float *A = a + a_off;
        const float *B = b + b_off;
        float *O = out + o_off;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float acc = 0.0f;
                for (int p = 0; p < k; p++)
                    acc += A[i * k + p] * B[p * n + j];
                O[i * n + j] = acc;
            }
        }
    }
}

/* ================================================================
 * Reductions
 * ================================================================
 *
 * All reductions take:
 *   x:      input data (row-major)
 *   shape:  input shape
 *   ndim:   number of dims
 *   axis:   reduction axis (already normalised to [0, ndim) by Python)
 *   out:    output data (pre-allocated by Python)
 *
 * For keepdims, the output shape has the reduced axis size = 1.
 * For non-keepdims, the reduced axis is removed.
 * Python handles both cases in shape computation; here we just
 * need the original shape and the axis.
 */

void ceval_sum(float *out, const float *x,
               const int *shape, int ndim, int axis) {
    int n = total_size(shape, ndim);
    int axis_size = shape[axis];

    /* Compute stride of the reduction axis */
    int strides[MAX_DIMS];
    compute_strides(shape, ndim, strides);
    int axis_stride = strides[axis];

    /* Number of output elements */
    int out_size = n / axis_size;

    /* For each output element, sum over the axis */
    int idx[MAX_DIMS];
    int oi = 0;

    /* Build output shape (shape with axis removed) */
    int out_shape[MAX_DIMS];
    int out_ndim = 0;
    for (int d = 0; d < ndim; d++) {
        if (d != axis) out_shape[out_ndim++] = shape[d];
    }

    int out_strides[MAX_DIMS];
    if (out_ndim > 0) compute_strides(out_shape, out_ndim, out_strides);

    for (oi = 0; oi < out_size; oi++) {
        /* Convert output linear index to multi-dim (in reduced shape) */
        int oidx[MAX_DIMS];
        if (out_ndim > 0) linear_to_multi(oi, out_shape, out_ndim, oidx);

        /* Map back to input: insert the axis dimension */
        int in_base[MAX_DIMS];
        int od = 0;
        for (int d = 0; d < ndim; d++) {
            if (d == axis) in_base[d] = 0;
            else in_base[d] = oidx[od++];
        }

        /* Compute base linear index */
        int base_lin = 0;
        for (int d = 0; d < ndim; d++)
            base_lin += in_base[d] * strides[d];

        float acc = 0.0f;
        for (int a = 0; a < axis_size; a++)
            acc += x[base_lin + a * axis_stride];
        out[oi] = acc;
    }
}

void ceval_max_reduce(float *out, const float *x,
                      const int *shape, int ndim, int axis) {
    int n = total_size(shape, ndim);
    int axis_size = shape[axis];

    int strides[MAX_DIMS];
    compute_strides(shape, ndim, strides);
    int axis_stride = strides[axis];
    int out_size = n / axis_size;

    int out_shape[MAX_DIMS];
    int out_ndim = 0;
    for (int d = 0; d < ndim; d++) {
        if (d != axis) out_shape[out_ndim++] = shape[d];
    }

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

void ceval_mean(float *out, const float *x,
                const int *shape, int ndim, int axis) {
    int axis_size = shape[axis];
    ceval_sum(out, x, shape, ndim, axis);
    int out_size = total_size(shape, ndim) / axis_size;
    float scale = 1.0f / (float)axis_size;
    for (int i = 0; i < out_size; i++)
        out[i] *= scale;
}

void ceval_argmax(int *out, const float *x,
                  const int *shape, int ndim, int axis) {
    int n = total_size(shape, ndim);
    int axis_size = shape[axis];

    int strides[MAX_DIMS];
    compute_strides(shape, ndim, strides);
    int axis_stride = strides[axis];
    int out_size = n / axis_size;

    int out_shape[MAX_DIMS];
    int out_ndim = 0;
    for (int d = 0; d < ndim; d++) {
        if (d != axis) out_shape[out_ndim++] = shape[d];
    }

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
        int mi = 0;
        for (int a = 1; a < axis_size; a++) {
            float v = x[base_lin + a * axis_stride];
            if (v > mx) { mx = v; mi = a; }
        }
        out[oi] = mi;
    }
}

/* ================================================================
 * Wiring ops
 * ================================================================ */

/* Transpose: permute dimensions and copy data */
void ceval_transpose(float *out, const float *x,
                     const int *shape, int ndim,
                     const int *axes) {
    int n = total_size(shape, ndim);

    int strides[MAX_DIMS];
    compute_strides(shape, ndim, strides);

    /* Output shape: out_shape[i] = shape[axes[i]] */
    int out_shape[MAX_DIMS], out_strides[MAX_DIMS];
    for (int i = 0; i < ndim; i++)
        out_shape[i] = shape[axes[i]];
    compute_strides(out_shape, ndim, out_strides);

    int out_idx[MAX_DIMS];
    for (int oi = 0; oi < n; oi++) {
        linear_to_multi(oi, out_shape, ndim, out_idx);

        /* Map output index back to input: in_idx[axes[i]] = out_idx[i] */
        int in_idx[MAX_DIMS];
        for (int d = 0; d < ndim; d++)
            in_idx[axes[d]] = out_idx[d];

        int in_lin = 0;
        for (int d = 0; d < ndim; d++)
            in_lin += in_idx[d] * strides[d];

        out[oi] = x[in_lin];
    }
}

/* Concat: join arrays along axis */
void ceval_concat(float *out,
                  const float **inputs, const int *sizes,
                  int num_inputs) {
    /* Simple: flatten and copy sequentially (Python pre-computes
     * the correct interleaved copy for axis != 0).
     * For axis == 0, this is just appending.
     *
     * Actually, for general concat along any axis, Python should
     * handle the layout since we need per-input shape info.
     * Here we do the simple case: contiguous copy. */
    int offset = 0;
    for (int i = 0; i < num_inputs; i++) {
        memcpy(out + offset, inputs[i], sizes[i] * sizeof(float));
        offset += sizes[i];
    }
}

/* General concat along axis */
void ceval_concat_axis(float *out,
                       const float **inputs,
                       const int **input_shapes,
                       const int *input_axis_sizes,
                       int num_inputs,
                       const int *first_shape, int ndim, int axis) {
    int strides[MAX_DIMS];
    compute_strides(first_shape, ndim, strides);

    /* Number of elements in dimensions before and after axis */
    int outer = 1, inner = 1;
    for (int d = 0; d < axis; d++) outer *= first_shape[d];
    for (int d = axis + 1; d < ndim; d++) inner *= first_shape[d];

    int out_off = 0;
    for (int o = 0; o < outer; o++) {
        for (int inp = 0; inp < num_inputs; inp++) {
            int axis_sz = input_axis_sizes[inp];
            int chunk = axis_sz * inner;

            /* Compute offset into input: o * input_axis_sizes[inp] * inner */
            int in_off = o * chunk;
            memcpy(out + out_off, inputs[inp] + in_off,
                   chunk * sizeof(float));
            out_off += chunk;
        }
    }
}

/* Repeat: repeat elements along axis */
void ceval_repeat(float *out, const float *x,
                  const int *shape, int ndim,
                  int repeats, int axis) {
    int n = total_size(shape, ndim);
    int strides[MAX_DIMS];
    compute_strides(shape, ndim, strides);

    int axis_size = shape[axis];
    int axis_stride = strides[axis];

    /* Output shape: same but shape[axis] *= repeats */
    int out_shape[MAX_DIMS];
    for (int d = 0; d < ndim; d++) out_shape[d] = shape[d];
    out_shape[axis] *= repeats;
    int out_n = total_size(out_shape, ndim);

    int out_strides[MAX_DIMS];
    compute_strides(out_shape, ndim, out_strides);
    int out_axis_stride = out_strides[axis];

    int out_idx[MAX_DIMS];
    for (int oi = 0; oi < out_n; oi++) {
        linear_to_multi(oi, out_shape, ndim, out_idx);

        /* Map output axis index to input axis index */
        /* np.repeat: each element repeated 'repeats' times */
        int in_axis_idx = out_idx[axis] / repeats;

        int in_lin = 0;
        for (int d = 0; d < ndim; d++) {
            int didx = (d == axis) ? in_axis_idx : out_idx[d];
            in_lin += didx * strides[d];
        }
        out[oi] = x[in_lin];
    }
}

/* Slice: extract subarray with start/stop/step per dimension */
void ceval_slice(float *out, const float *x,
                 const int *shape, int ndim,
                 const int *starts, const int *stops, const int *steps,
                 const int *out_shape, int out_ndim) {
    int out_n = total_size(out_shape, out_ndim);

    int strides[MAX_DIMS];
    compute_strides(shape, ndim, strides);

    int out_idx[MAX_DIMS];
    for (int oi = 0; oi < out_n; oi++) {
        linear_to_multi(oi, out_shape, out_ndim, out_idx);

        /* Map output index to input index */
        int in_lin = 0;
        for (int d = 0; d < ndim; d++) {
            int in_d = starts[d] + out_idx[d] * steps[d];
            in_lin += in_d * strides[d];
        }
        out[oi] = x[in_lin];
    }
}

/* Copy (for reshape, expand_dims, cast — just memcpy) */
void ceval_copy(float *out, const float *x, int n) {
    memcpy(out, x, n * sizeof(float));
}
