"""IEEE-754 arithmetic operations — reference implementation.

These functions define WHAT each operation computes, not HOW
the executor runs them.  They use NumPy as the reference (golden
model) that the C gate executor must match bit-for-bit.

Architecture
------------
- ``circuit_graph.py`` defines the circuit DAG (WHAT to compute).
- ``binary_ops.py`` (this module) provides standalone functions
  that the tests call to verify correctness.  Each function uses
  NumPy — the REFERENCE — not a Python FPU simulation.
- The C gate executor (Phase 4) walks the DAG and uses byte-plane
  shift+XOR gates.  It must produce the same output as these
  reference functions.

Why not implement IEEE-754 in Python?
-------------------------------------
The previous version tried to re-implement the FPU algorithm using
Python integer operations (shift, XOR, mask on uint32).  That was
wrong — it was a hypothetical "how to run" that:

1. Was fragile and incomplete (edge cases, rounding, denormals).
2. Was the executor's job, not the graph's.
3. Added no value — NumPy already IS the reference FPU.

The real execution path is:
  Circuit graph → C executor (byte-plane gates) → FPGA synthesis
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------
# Helpers: float ↔ uint bit-pattern reinterpretation
# ---------------------------------------------------------------

def _f2u(x: np.ndarray) -> np.ndarray:
    """float32 → uint32 (bit-pattern reinterpret, zero-copy)."""
    return np.asarray(x, dtype=np.float32).view(np.uint32)


def _u2f(x: np.ndarray) -> np.ndarray:
    """uint32 → float32 (bit-pattern reinterpret, zero-copy)."""
    return np.asarray(x, dtype=np.uint32).view(np.float32)


# ---------------------------------------------------------------
# IEEE-754 float32 constants
# ---------------------------------------------------------------

_SIGN_BIT_32 = np.uint32(0x80000000)


# ---------------------------------------------------------------
# Unary operations
# ---------------------------------------------------------------

def circuit_neg(x: np.ndarray) -> np.ndarray:
    """Negate: flip the sign bit (bit 31).  Gate cost: 1 XOR."""
    return _u2f(_f2u(x) ^ _SIGN_BIT_32)


def circuit_abs(x: np.ndarray) -> np.ndarray:
    """Absolute value: clear the sign bit.  Gate cost: 1 AND."""
    return _u2f(_f2u(x) & ~_SIGN_BIT_32)


def circuit_square(x: np.ndarray) -> np.ndarray:
    """Square: x * x."""
    return circuit_mul(x, x)


# ---------------------------------------------------------------
# Binary arithmetic (reference: NumPy FPU)
# ---------------------------------------------------------------

def circuit_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IEEE-754 float32 addition.  Reference implementation."""
    return (np.asarray(a, dtype=np.float32)
            + np.asarray(b, dtype=np.float32))


def circuit_sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IEEE-754 float32 subtraction."""
    return (np.asarray(a, dtype=np.float32)
            - np.asarray(b, dtype=np.float32))


def circuit_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IEEE-754 float32 multiplication.  Reference implementation."""
    return (np.asarray(a, dtype=np.float32)
            * np.asarray(b, dtype=np.float32))


def circuit_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IEEE-754 float32 division.  Reference implementation."""
    return (np.asarray(a, dtype=np.float32)
            / np.asarray(b, dtype=np.float32))


# ---------------------------------------------------------------
# Comparison / selection
# ---------------------------------------------------------------

def circuit_max(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise maximum.

    Gate strategy: sign-magnitude → sortable uint32 comparison.
    """
    return np.maximum(
        np.asarray(a, dtype=np.float32),
        np.asarray(b, dtype=np.float32),
    )


def circuit_cmp_le(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise a <= b -> uint8 (0 or 1)."""
    return (np.asarray(a, dtype=np.float32)
            <= np.asarray(b, dtype=np.float32)).astype(np.uint8)


def circuit_mux(
    sel: np.ndarray, a: np.ndarray, b: np.ndarray,
) -> np.ndarray:
    """MUX: sel==0 -> a, sel!=0 -> b.

    Gate strategy: bitwise (a & ~mask) | (b & mask) on uint32.
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
    """Sum reduction.  Gate strategy: balanced tree of adders."""
    return np.asarray(x, dtype=np.float32).sum(axis=axis)


def circuit_reduce_max(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Max reduction.  Gate strategy: balanced tree of comparators."""
    return np.asarray(x, dtype=np.float32).max(axis=axis)


def circuit_argmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Argmax: index of the maximum value."""
    return np.asarray(x, dtype=np.float32).argmax(axis=axis)


def circuit_mean(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Mean: sum / count."""
    return np.asarray(x, dtype=np.float32).mean(axis=axis)


def circuit_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matrix multiply.  Gate strategy: mul grid + sum-reduction tree."""
    return (np.asarray(a, dtype=np.float32)
            @ np.asarray(b, dtype=np.float32))


def circuit_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax: max -> sub -> exp -> sum -> div."""
    x = np.asarray(x, dtype=np.float32)
    m = x.max(axis=axis, keepdims=True)
    with np.errstate(over="ignore"):
        e = np.exp((x - m).astype(np.float64)).astype(np.float32)
    return e / e.sum(axis=axis, keepdims=True)


# ---------------------------------------------------------------
# Float64 operations (8 byte planes)
# ---------------------------------------------------------------

def circuit_add_f64(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IEEE-754 float64 addition."""
    return (np.asarray(a, dtype=np.float64)
            + np.asarray(b, dtype=np.float64))


def circuit_mul_f64(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IEEE-754 float64 multiplication."""
    return (np.asarray(a, dtype=np.float64)
            * np.asarray(b, dtype=np.float64))


def circuit_div_f64(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IEEE-754 float64 division."""
    return (np.asarray(a, dtype=np.float64)
            / np.asarray(b, dtype=np.float64))


def circuit_sum_f64(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Float64 sum reduction."""
    return np.asarray(x, dtype=np.float64).sum(axis=axis)


def circuit_mean_f64(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Float64 mean."""
    return np.asarray(x, dtype=np.float64).mean(axis=axis)
