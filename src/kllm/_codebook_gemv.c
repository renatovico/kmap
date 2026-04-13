/*
 * Gate Engine — LUT-based matrix-vector multiply.
 *
 * Every "multiplication" is replaced by a table lookup:
 *
 *   out[j] = Σ_k  LUT[ w_q[j,k] ][ x_q[k] ]
 *
 * where:
 *   w_q  = uint8 quantized weights  (per-row affine: 256 levels)
 *   x_q  = uint8 quantized input    (per-vector affine: 256 levels)
 *   LUT  = 256×256 float32 precomputed products
 *
 * The LUT (256 KB) lives entirely in L1 cache after the first access.
 * The inner loop is: load byte, load byte, index 2D array, f32 add.
 * No floating-point multiply anywhere — this IS a gate circuit.
 *
 * For the accumulator we use float32 (not int32) because the LUT
 * entries are the exact dequantized products — this avoids a separate
 * rescale pass and keeps the accumulation numerically clean.
 *
 * Weight memory: 1 byte/element (4× smaller than float32).
 * On M1 Pro (37 GB/s): theoretical 36 tok/s for TinyLlama-1.1B.
 * On DDR5 x86 (80 GB/s): theoretical 73 tok/s.
 *
 * Compile:
 *   cc -O3 -shared -fPIC -march=native [-fopenmp] \
 *      -o _codebook_gemv.so _codebook_gemv.c
 */

#include <stdint.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/*
 * Build the 256×256 LUT for one projection.
 *
 * LUT[wi][xi] = w_dequant(wi) * x_dequant(xi)
 *
 * where:
 *   w_dequant(wi) = wi * w_scale + w_zero   (per-row — but we fold
 *       the row scale into the output rescaling, so here we use the
 *       "unit row" where w_scale=1, w_zero=0, i.e. just the integer)
 *
 * Actually, we build one LUT per *row* incorporating that row's
 * scale+zero, which would be 2048 × 256 × 256 = too large.
 *
 * Better: factor the affine quantization.
 *
 *   out[j] = Σ_k  (wq[j,k]*ws[j] + wz[j]) * (xq[k]*xs + xz)
 *          = ws[j]*xs * Σ wq*xq  +  ws[j]*xz * Σ wq
 *            + wz[j]*xs * Σ xq   +  wz[j]*xz * K
 *
 * The hot term is Σ wq[j,k]*xq[k] — a uint8·uint8 dot product.
 * We accumulate in int32, then apply the 4 affine correction terms.
 *
 * THIS is the gate-native form: integer multiply-accumulate.
 * On ARM: UDOT does 4× uint8·uint8→uint32 per cycle.
 * On x86: VPDPBUSD (AVX-VNNI) does the same.
 * Fallback: uint8→uint16 multiply + uint32 accumulate.
 */

/* ---- Integer GEMV (decode: seq=1) ---- */
void gate_gemv_u8(
    const uint8_t  *wq,       /* (M, K) uint8 quantized weights     */
    const float    *w_scale,  /* (M,)   per-row scale                */
    const float    *w_zero,   /* (M,)   per-row zero-point           */
    const uint8_t  *xq,       /* (K,)   uint8 quantized input        */
    float           x_scale,  /* scalar input scale                  */
    float           x_zero,   /* scalar input zero-point             */
    float          *out,      /* (M,)   output (float32)             */
    int M, int K)
{
    /* Precompute input-side sums (shared across all rows) */
    int32_t xq_sum = 0;
    for (int k = 0; k < K; k++)
        xq_sum += (int32_t)xq[k];

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int j = 0; j < M; j++) {
        const uint8_t *row = wq + (long)j * K;
        int32_t dot = 0;
        int32_t wq_sum = 0;

        /* Inner loop: uint8 * uint8 → int32 accumulate.
         * Compiler auto-vectorises this to UDOT (ARM) or
         * VPDPBUSD (x86 AVX-VNNI).                        */
        for (int k = 0; k < K; k++) {
            int32_t a = (int32_t)row[k];
            int32_t b = (int32_t)xq[k];
            dot += a * b;
            wq_sum += a;
        }

        float ws = w_scale[j];
        float wz = w_zero[j];
        out[j] = ws * x_scale * (float)dot
               + ws * x_zero  * (float)wq_sum
               + wz * x_scale * (float)xq_sum
               + wz * x_zero  * (float)K;
    }
}

/* ---- Integer GEMM (prefill: seq>1) ---- */
void gate_gemm_u8(
    const uint8_t  *wq,       /* (M, K)         */
    const float    *w_scale,  /* (M,)            */
    const float    *w_zero,   /* (M,)            */
    const uint8_t  *Xq,       /* (seq, K)        */
    const float    *x_scales, /* (seq,)           */
    const float    *x_zeros,  /* (seq,)           */
    float          *out,      /* (seq, M)         */
    int M, int K, int seq)
{
    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
    #endif
    for (int s = 0; s < seq; s++) {
        for (int j = 0; j < M; j++) {
            const uint8_t *xq = Xq + (long)s * K;
            const uint8_t *row = wq + (long)j * K;
            int32_t dot = 0;
            int32_t wq_sum = 0;
            int32_t xq_sum = 0;

            for (int k = 0; k < K; k++) {
                int32_t a = (int32_t)row[k];
                int32_t b = (int32_t)xq[k];
                dot += a * b;
                wq_sum += a;
                xq_sum += b;
            }

            float ws = w_scale[j];
            float wz = w_zero[j];
            float xs = x_scales[s];
            float xz = x_zeros[s];
            out[(long)s * M + j] = ws * xs * (float)dot
                                  + ws * xz * (float)wq_sum
                                  + wz * xs * (float)xq_sum
                                  + wz * xz * (float)K;
        }
    }
}

/* ---- Fused float32-in GEMV: quantize activation + integer matmul ---- */
/*
 * Takes float32 activations directly, quantizes to uint8 inline,
 * then runs the integer dot product.  Eliminates Python-side
 * quantize_activation_u8() call overhead.
 */
void gate_gemv_f32in(
    const uint8_t  *wq,       /* (M, K) uint8 weights       */
    const float    *w_scale,  /* (M,) per-row scale           */
    const float    *w_zero,   /* (M,) per-row zero            */
    const float    *x,        /* (K,) float32 input           */
    float          *out,      /* (M,) float32 output          */
    uint8_t        *xq_buf,   /* (K,) scratch for quantized x */
    int M, int K)
{
    /* Quantize x to uint8 */
    float xmin = x[0], xmax = x[0];
    for (int k = 1; k < K; k++) {
        if (x[k] < xmin) xmin = x[k];
        if (x[k] > xmax) xmax = x[k];
    }
    float x_scale = (xmax - xmin) / 255.0f;
    if (x_scale == 0.0f) x_scale = 1.0f;
    float inv_xs = 1.0f / x_scale;

    int32_t xq_sum = 0;
    for (int k = 0; k < K; k++) {
        float v = (x[k] - xmin) * inv_xs;
        int iv = (int)(v + 0.5f);
        if (iv < 0) iv = 0;
        if (iv > 255) iv = 255;
        xq_buf[k] = (uint8_t)iv;
        xq_sum += iv;
    }

    float x_zero = xmin;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int j = 0; j < M; j++) {
        const uint8_t *row = wq + (long)j * K;
        int32_t dot = 0;
        int32_t wq_sum = 0;

        for (int k = 0; k < K; k++) {
            int32_t a = (int32_t)row[k];
            int32_t b = (int32_t)xq_buf[k];
            dot += a * b;
            wq_sum += a;
        }

        float ws = w_scale[j];
        float wz = w_zero[j];
        out[j] = ws * x_scale * (float)dot
               + ws * x_zero  * (float)wq_sum
               + wz * x_scale * (float)xq_sum
               + wz * x_zero  * (float)K;
    }
}

/* ---- Fused LUT GEMV (alternative: no int multiply at all) ---- */
/*
 * For architectures without fast integer multiply (FPGA, WASM),
 * use a 256×256 LUT where every product is precomputed.
 *
 * LUT[w_idx][x_idx] = dequant(w_idx) * dequant(x_idx)
 * Inner loop: load byte, load byte, table lookup, float add.
 * Zero multiplications. Pure combinational logic.
 */
void gate_gemv_lut(
    const uint8_t  *wq,    /* (M, K)                */
    const uint8_t  *xq,    /* (K,)                  */
    const float    *lut,   /* (256, 256) float32    */
    float          *out,   /* (M,)                  */
    int M, int K)
{
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int j = 0; j < M; j++) {
        const uint8_t *row = wq + (long)j * K;
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            acc += lut[(int)row[k] * 256 + (int)xq[k]];
        }
        out[j] = acc;
    }
}

/* ---- Lossless uint16 codebook GEMV (kept for exact mode) ---- */

#define TILE 256

void codebook_gemv(
    const uint16_t *idx,
    const float    *cb,
    const float    *x,
    float          *out,
    int M, int K)
{
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int j = 0; j < M; j++) {
        const uint16_t *row = idx + (long)j * K;
        float acc = 0.0f;
        float tile[TILE];
        int k = 0;
        for (; k + TILE <= K; k += TILE) {
            for (int t = 0; t < TILE; t++)
                tile[t] = cb[row[k + t]];
            for (int t = 0; t < TILE; t++)
                acc += tile[t] * x[k + t];
        }
        for (; k < K; k++)
            acc += cb[row[k]] * x[k];
        out[j] = acc;
    }
}

void codebook_gemm(
    const uint16_t *idx,
    const float    *cb,
    const float    *X,
    float          *out,
    int M, int K, int seq)
{
    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
    #endif
    for (int s = 0; s < seq; s++) {
        for (int j = 0; j < M; j++) {
            const float *xr = X + (long)s * K;
            const uint16_t *row = idx + (long)j * K;
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += cb[row[k]] * xr[k];
            }
            out[(long)s * M + j] = acc;
        }
    }
}
