"""Gate Engine — LUT / integer–only matrix multiplication.

Two execution modes, both **multiply-free**:

1. **Integer mode** (``gate_matmul``):
   Weights are ``uint8`` per-row affine-quantised.  Activations are
   ``uint8`` per-vector affine-quantised.  The inner loop computes
   ``Σ wq·xq`` in ``int32`` — the compiler auto-vectorises this to
   ARM UDOT / x86 AVX-VNNI (4× uint8 ops per cycle).  Four affine
   correction terms are applied per row in float32 at the end.

2. **LUT mode** (``gate_matmul_lut``):
   A 256×256 float32 table (256 KB, fits L1) contains every possible
   product.  The inner loop is ``acc += LUT[w][x]`` — zero multiplies.
   Ideal for FPGA / WASM targets without fast integer multiply.

Both modes read **1 byte per weight** (4× less than float32).

Also exposes the lossless uint16 codebook path from the previous
iteration (``codebook_matmul``).
"""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess

import numpy as np

# ---------------------------------------------------------------
# Compile / load the C shared library
# ---------------------------------------------------------------

_LIB = None

_SRC = os.path.join(os.path.dirname(__file__), "_codebook_gemv.c")
_SO = os.path.join(os.path.dirname(__file__), "_codebook_gemv.so")


def _compile_if_needed() -> str:
    """Return path to compiled shared library, building it if necessary."""
    if os.path.isfile(_SO) and os.path.getmtime(_SO) >= os.path.getmtime(_SRC):
        return _SO
    cc = os.environ.get("CC", "cc")
    base = [cc, "-O3", "-shared", "-fPIC", "-march=native"]

    # Try with OpenMP for multi-core parallelism
    if platform.system() == "Darwin":
        omp_inc = "/opt/homebrew/opt/libomp/include"
        omp_lib = "/opt/homebrew/opt/libomp/lib"
        omp_flags = [
            "-Xpreprocessor", "-fopenmp",
            f"-I{omp_inc}", f"-L{omp_lib}", "-lomp",
        ]
    else:
        omp_flags = ["-fopenmp"]

    for flags in [omp_flags, []]:
        try:
            cmd = base + flags + ["-o", _SO, _SRC]
            subprocess.check_call(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return _SO
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue

    raise RuntimeError("Failed to compile _codebook_gemv.c")
    return _SO


def _get_lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    so_path = _compile_if_needed()
    _LIB = ctypes.CDLL(so_path)

    c_u8p = ctypes.POINTER(ctypes.c_uint8)
    c_u16p = ctypes.POINTER(ctypes.c_uint16)
    c_fp = ctypes.POINTER(ctypes.c_float)
    c_f = ctypes.c_float
    c_i = ctypes.c_int

    # gate_gemv_u8
    _LIB.gate_gemv_u8.restype = None
    _LIB.gate_gemv_u8.argtypes = [
        c_u8p, c_fp, c_fp, c_u8p, c_f, c_f, c_fp, c_i, c_i,
    ]

    # gate_gemm_u8
    _LIB.gate_gemm_u8.restype = None
    _LIB.gate_gemm_u8.argtypes = [
        c_u8p, c_fp, c_fp, c_u8p, c_fp, c_fp, c_fp, c_i, c_i, c_i,
    ]

    # gate_gemv_lut
    _LIB.gate_gemv_lut.restype = None
    _LIB.gate_gemv_lut.argtypes = [c_u8p, c_u8p, c_fp, c_fp, c_i, c_i]

    # gate_gemv_f32in (fused: float32 in → quantize → int matmul)
    _LIB.gate_gemv_f32in.restype = None
    _LIB.gate_gemv_f32in.argtypes = [
        c_u8p, c_fp, c_fp, c_fp, c_fp, c_u8p, c_i, c_i,
    ]

    # codebook_gemv / codebook_gemm (lossless path)
    _LIB.codebook_gemv.restype = None
    _LIB.codebook_gemv.argtypes = [c_u16p, c_fp, c_fp, c_fp, c_i, c_i]
    _LIB.codebook_gemm.restype = None
    _LIB.codebook_gemm.argtypes = [c_u16p, c_fp, c_fp, c_fp, c_i, c_i, c_i]

    return _LIB


# ---------------------------------------------------------------
# Weight quantization helpers
# ---------------------------------------------------------------

def quantize_weight_u8(
    w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-row affine quantise float32 weights to uint8.

    Returns ``(wq, scale, zero)`` where
    ``wq * scale[:, None] + zero[:, None] ≈ w``.
    """
    row_min = w.min(axis=1)
    row_max = w.max(axis=1)
    scale = (row_max - row_min) / 255.0
    scale[scale == 0] = 1.0
    zero = row_min.copy()
    wq = np.round((w - zero[:, None]) / scale[:, None]).astype(np.uint8)
    return wq, scale.astype(np.float32), zero.astype(np.float32)


def quantize_activation_u8(
    x: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Per-vector affine quantise float32 activation to uint8.

    Returns ``(xq, scale, zero)`` where ``xq * scale + zero ≈ x``.
    """
    xmin = float(x.min())
    xmax = float(x.max())
    scale = (xmax - xmin) / 255.0 if xmax != xmin else 1.0
    xq = np.round((x - xmin) / scale).astype(np.uint8)
    return xq, np.float32(scale), np.float32(xmin)


# ---------------------------------------------------------------
# Gate-based matmul (integer / LUT)
# ---------------------------------------------------------------

def gate_matmul(
    wq: np.ndarray,
    w_scale: np.ndarray,
    w_zero: np.ndarray,
    x: np.ndarray,
    *,
    _cache: dict | None = None,
) -> np.ndarray:
    """Integer-only GEMV via the C gate engine.

    Parameters
    ----------
    wq      : (M, K) uint8 quantised weights.
    w_scale : (M,) float32 per-row scale.
    w_zero  : (M,) float32 per-row zero.
    x       : (K,) or (seq, K) float32 activation.
    _cache  : optional pre-built ctypes pointer dict (avoids re-wrapping).

    Returns
    -------
    out : (M,) or (seq, M) float32.
    """
    lib = _get_lib()
    M, K = wq.shape

    u8p = ctypes.POINTER(ctypes.c_uint8)
    fp = ctypes.POINTER(ctypes.c_float)

    if _cache is not None:
        wq_p = _cache["wq"]
        ws_p = _cache["ws"]
        wz_p = _cache["wz"]
    else:
        wq = np.ascontiguousarray(wq, dtype=np.uint8)
        w_scale = np.ascontiguousarray(w_scale, dtype=np.float32)
        w_zero = np.ascontiguousarray(w_zero, dtype=np.float32)
        wq_p = wq.ctypes.data_as(u8p)
        ws_p = w_scale.ctypes.data_as(fp)
        wz_p = w_zero.ctypes.data_as(fp)

    if x.ndim == 1:
        x = np.ascontiguousarray(x, dtype=np.float32)
        out = np.empty(M, dtype=np.float32)
        xq_buf = np.empty(K, dtype=np.uint8)
        lib.gate_gemv_f32in(
            wq_p, ws_p, wz_p,
            x.ctypes.data_as(fp),
            out.ctypes.data_as(fp),
            xq_buf.ctypes.data_as(u8p),
            M, K,
        )
        return out

    # Batched (seq, K)
    seq = x.shape[0]
    x_scales = np.empty(seq, dtype=np.float32)
    x_zeros = np.empty(seq, dtype=np.float32)
    Xq = np.empty((seq, K), dtype=np.uint8)
    for s in range(seq):
        Xq[s], x_scales[s], x_zeros[s] = quantize_activation_u8(x[s])
    Xq = np.ascontiguousarray(Xq)
    out = np.empty((seq, M), dtype=np.float32)
    lib.gate_gemm_u8(
        wq_p if _cache else wq.ctypes.data_as(u8p),
        ws_p if _cache else w_scale.ctypes.data_as(fp),
        wz_p if _cache else w_zero.ctypes.data_as(fp),
        Xq.ctypes.data_as(u8p),
        x_scales.ctypes.data_as(fp),
        x_zeros.ctypes.data_as(fp),
        out.ctypes.data_as(fp),
        M, K, seq,
    )
    return out


def build_gate_cache(gw: dict) -> dict:
    """Pre-compute ctypes pointers for a gate weight dict."""
    u8p = ctypes.POINTER(ctypes.c_uint8)
    fp = ctypes.POINTER(ctypes.c_float)
    wq = np.ascontiguousarray(gw["wq"], dtype=np.uint8)
    ws = np.ascontiguousarray(gw["scale"], dtype=np.float32)
    wz = np.ascontiguousarray(gw["zero"], dtype=np.float32)
    return {
        "wq_arr": wq, "ws_arr": ws, "wz_arr": wz,
        "wq": wq.ctypes.data_as(u8p),
        "ws": ws.ctypes.data_as(fp),
        "wz": wz.ctypes.data_as(fp),
        "M": wq.shape[0], "K": wq.shape[1],
    }


# ---------------------------------------------------------------
# Lossless uint16 codebook matmul (exact mode)
# ---------------------------------------------------------------

def build_codebook(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build a codebook and uint16 index matrix (lossless)."""
    cb = np.unique(w)
    if cb.size > 65535:
        raise ValueError(
            f"Weight has {cb.size} unique values — exceeds uint16 (65 535)."
        )
    idx = np.searchsorted(cb, w).astype(np.uint16)
    assert np.array_equal(cb[idx], w), "codebook reconstruction mismatch"
    return cb, idx


def codebook_matmul(
    idx: np.ndarray,
    cb: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """Lossless GEMV via uint16 codebook lookup + float32 FMA."""
    idx = np.ascontiguousarray(idx, dtype=np.uint16)
    cb = np.ascontiguousarray(cb, dtype=np.float32)
    x = np.ascontiguousarray(x, dtype=np.float32)
    M, K = idx.shape
    lib = _get_lib()
    u16p = ctypes.POINTER(ctypes.c_uint16)
    fp = ctypes.POINTER(ctypes.c_float)

    if x.ndim == 1:
        out = np.empty(M, dtype=np.float32)
        lib.codebook_gemv(
            idx.ctypes.data_as(u16p), cb.ctypes.data_as(fp),
            x.ctypes.data_as(fp), out.ctypes.data_as(fp), M, K,
        )
        return out

    seq = x.shape[0]
    x = np.ascontiguousarray(x)
    out = np.empty((seq, M), dtype=np.float32)
    lib.codebook_gemm(
        idx.ctypes.data_as(u16p), cb.ctypes.data_as(fp),
        x.ctypes.data_as(fp), out.ctypes.data_as(fp), M, K, seq,
    )
    return out
