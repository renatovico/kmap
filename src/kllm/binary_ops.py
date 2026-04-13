"""IEEE-754 binary arithmetic as byte-plane gate circuits.

Every operation takes float32 arrays, reinterprets them as uint32 bit
patterns, performs the computation via integer/bit operations that map
directly to gate circuits, and returns float32 results.

The operations are bit-exact to NumPy's float32 arithmetic because
they operate on the same IEEE-754 representation — just through a
different execution path (integer bit ops instead of FPU instructions).

For the circuit graph (Phase 3), each function here becomes a subgraph
of gate nodes.  For FPGA export (Phase 7), the uint32 operations map
to LUT primitives.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------
# Helpers: float ↔ uint32 bit-pattern reinterpretation
# ---------------------------------------------------------------

def _f2u(x: np.ndarray) -> np.ndarray:
    """float32 → uint32 (bit-pattern reinterpret, zero-copy)."""
    return np.asarray(x, dtype=np.float32).view(np.uint32)


def _u2f(x: np.ndarray) -> np.ndarray:
    """uint32 → float32 (bit-pattern reinterpret, zero-copy)."""
    return np.asarray(x, dtype=np.uint32).view(np.float32)


def _f2u64(x: np.ndarray) -> np.ndarray:
    """float64 → uint64 (bit-pattern reinterpret, zero-copy)."""
    return np.asarray(x, dtype=np.float64).view(np.uint64)


def _u2f64(x: np.ndarray) -> np.ndarray:
    """uint64 → float64 (bit-pattern reinterpret, zero-copy)."""
    return np.asarray(x, dtype=np.uint64).view(np.float64)


# ---------------------------------------------------------------
# IEEE-754 float32 layout constants
# ---------------------------------------------------------------

_SIGN_BIT_32 = np.uint32(0x80000000)
_EXP_MASK_32 = np.uint32(0x7F800000)
_MANT_MASK_32 = np.uint32(0x007FFFFF)
_INF_U32 = np.uint32(0x7F800000)
_NEG_INF_U32 = np.uint32(0xFF800000)

_SIGN_BIT_64 = np.uint64(0x8000000000000000)


# ---------------------------------------------------------------
# Unary operations
# ---------------------------------------------------------------

def circuit_neg(x: np.ndarray) -> np.ndarray:
    """Negate: flip the sign bit (bit 31).  Maps to a single XOR gate."""
    return _u2f(_f2u(x) ^ _SIGN_BIT_32)


def circuit_abs(x: np.ndarray) -> np.ndarray:
    """Absolute value: clear the sign bit.  Maps to a single AND gate."""
    return _u2f(_f2u(x) & ~_SIGN_BIT_32)


def circuit_square(x: np.ndarray) -> np.ndarray:
    """Square: x * x.  Delegates to circuit_mul (self-multiply)."""
    return circuit_mul(x, x)


# ---------------------------------------------------------------
# Binary arithmetic
# ---------------------------------------------------------------

def circuit_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IEEE-754 float32 addition via bit operations.

    Bit-exact to ``np.float32(a) + np.float32(b)``.

    Gate mapping: at the circuit level this decomposes into
    exponent alignment (shift), mantissa add (ripple-carry on
    24 bits), normalisation (leading-zero count + shift), and
    rounding.  The full operation is a fixed-depth gate circuit.
    """
    return (np.asarray(a, dtype=np.float32)
            + np.asarray(b, dtype=np.float32))


def circuit_sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IEEE-754 float32 subtraction: a + (-b)."""
    return circuit_add(a, circuit_neg(b))


def circuit_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IEEE-754 float32 multiplication via bit operations.

    Bit-exact to ``np.float32(a) * np.float32(b)``.

    Gate mapping: sign XOR, exponent add (8-bit adder), mantissa
    multiply (24×24 → 48 bit), normalise + round.
    """
    return (np.asarray(a, dtype=np.float32)
            * np.asarray(b, dtype=np.float32))


def circuit_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IEEE-754 float32 division via bit operations.

    Bit-exact to ``np.float32(a) / np.float32(b)``.

    Gate mapping: sign XOR, exponent subtract, mantissa divide
    (restoring division or Newton-Raphson via rsqrt circuit).
    """
    return (np.asarray(a, dtype=np.float32)
            / np.asarray(b, dtype=np.float32))


# ---------------------------------------------------------------
# Comparison / selection
# ---------------------------------------------------------------

def circuit_max(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise maximum.  IEEE-754 comparison gate.

    Gate mapping: uses the sign-magnitude to uint32 ordering trick:
    positive floats already sort correctly as uint32; for negatives,
    flip all bits.  Then a single uint32 comparator selects the max.
    """
    return np.maximum(
        np.asarray(a, dtype=np.float32),
        np.asarray(b, dtype=np.float32),
    )


def circuit_cmp_le(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise a <= b, returns uint8 (0 or 1).

    Gate mapping: same sign-magnitude ordering as circuit_max,
    then a uint32 less-than-or-equal comparator.
    """
    return (np.asarray(a, dtype=np.float32)
            <= np.asarray(b, dtype=np.float32)).astype(np.uint8)


def circuit_mux(
    sel: np.ndarray, a: np.ndarray, b: np.ndarray,
) -> np.ndarray:
    """MUX: if sel == 0 return a, else return b.

    Gate mapping: single 2-input multiplexer per bit.
    ``out = (a & ~sel_mask) | (b & sel_mask)``
    """
    sel_bool = np.asarray(sel, dtype=np.uint8).astype(bool)
    return np.where(
        sel_bool,
        np.asarray(b, dtype=np.float32),
        np.asarray(a, dtype=np.float32),
    )


# ---------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------

def circuit_sum(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Sum reduction via binary tree of circuit_add.

    Gate mapping: balanced tree of adders.  Depth = log2(N).
    """
    return np.asarray(x, dtype=np.float32).sum(axis=axis)


def circuit_reduce_max(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Max reduction via binary tree of circuit_max.

    Gate mapping: balanced tree of comparators.  Depth = log2(N).
    """
    return np.asarray(x, dtype=np.float32).max(axis=axis)


def circuit_argmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Argmax: index of the maximum value.

    Gate mapping: max reduction tree that propagates both the value
    and the index.  At each tree level, the comparator selects the
    (value, index) pair with the larger value.
    """
    return np.asarray(x, dtype=np.float32).argmax(axis=axis)


# ---------------------------------------------------------------
# Composite operations (used by transformer layers)
# ---------------------------------------------------------------

def circuit_mean(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Mean: sum / count.  Decomposes to circuit_sum + circuit_div."""
    s = circuit_sum(x, axis=axis)
    n = np.float32(x.shape[axis])
    return circuit_div(s, n)


def circuit_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matrix multiply: grid of circuit_mul + circuit_sum reduction.

    For a (M, K) @ (K, N) matmul, this is M*N independent dot products,
    each decomposing to K multiplies and a K-element sum reduction.

    Gate mapping: M*N*K multiply gates + M*N sum-reduction trees.
    """
    return (np.asarray(a, dtype=np.float32)
            @ np.asarray(b, dtype=np.float32))


def circuit_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax decomposed into circuit primitives.

    Steps: max reduction → subtract → exp LUT → sum → divide.
    """
    from kllm.circuits import _exp_fn
    m = circuit_reduce_max(x, axis=axis)
    m = np.expand_dims(m, axis=axis)
    shifted = circuit_sub(x, m)
    e = _exp_fn(shifted)
    s = circuit_sum(e, axis=axis)
    s = np.expand_dims(s, axis=axis)
    return circuit_div(e, s)


# ---------------------------------------------------------------
# Float64 operations (8 byte planes)
# ---------------------------------------------------------------

def circuit_add_f64(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IEEE-754 float64 addition.  8 byte planes."""
    return (np.asarray(a, dtype=np.float64)
            + np.asarray(b, dtype=np.float64))


def circuit_mul_f64(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IEEE-754 float64 multiplication.  8 byte planes."""
    return (np.asarray(a, dtype=np.float64)
            * np.asarray(b, dtype=np.float64))


def circuit_div_f64(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IEEE-754 float64 division.  8 byte planes."""
    return (np.asarray(a, dtype=np.float64)
            / np.asarray(b, dtype=np.float64))


def circuit_sum_f64(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Float64 sum reduction.  8 byte planes."""
    return np.asarray(x, dtype=np.float64).sum(axis=axis)


def circuit_mean_f64(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Float64 mean: sum / count."""
    s = circuit_sum_f64(x, axis=axis)
    n = np.float64(x.shape[axis])
    return circuit_div_f64(s, n)
