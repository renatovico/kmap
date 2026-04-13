"""Tests for binary arithmetic gate circuits.

Every circuit_* function must be bit-exact to the corresponding NumPy
float operation.  These tests verify IEEE-754 correctness for random
inputs, edge cases (inf, nan, zero, denormals), and reductions.
"""

import numpy as np
import pytest

from kllm.binary_ops import (
    circuit_abs,
    circuit_add,
    circuit_add_f64,
    circuit_argmax,
    circuit_cmp_le,
    circuit_div,
    circuit_div_f64,
    circuit_matmul,
    circuit_max,
    circuit_mean,
    circuit_mean_f64,
    circuit_mul,
    circuit_mul_f64,
    circuit_mux,
    circuit_neg,
    circuit_reduce_max,
    circuit_softmax,
    circuit_square,
    circuit_sub,
    circuit_sum,
    circuit_sum_f64,
)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _rand32(n: int, rng: np.random.Generator) -> np.ndarray:
    """Random float32 values in a reasonable range."""
    return rng.standard_normal(n).astype(np.float32)


def _assert_bitexact(actual: np.ndarray, expected: np.ndarray) -> None:
    """Assert bit-exact equality via uint32 comparison."""
    a_bits = np.asarray(actual, dtype=np.float32).view(np.uint32)
    e_bits = np.asarray(expected, dtype=np.float32).view(np.uint32)
    np.testing.assert_array_equal(a_bits, e_bits)


def _assert_bitexact_f64(actual: np.ndarray, expected: np.ndarray) -> None:
    """Assert bit-exact equality for float64."""
    a_bits = np.asarray(actual, dtype=np.float64).view(np.uint64)
    e_bits = np.asarray(expected, dtype=np.float64).view(np.uint64)
    np.testing.assert_array_equal(a_bits, e_bits)


# ---------------------------------------------------------------
# Unary ops
# ---------------------------------------------------------------

class TestUnary:
    def test_neg_random(self):
        rng = np.random.default_rng(42)
        x = _rand32(10_000, rng)
        _assert_bitexact(circuit_neg(x), -x)

    def test_neg_zero(self):
        pos_zero = np.float32(0.0)
        neg_zero = np.float32(-0.0)
        assert circuit_neg(pos_zero).view(np.uint32) == neg_zero.view(np.uint32)
        assert circuit_neg(neg_zero).view(np.uint32) == pos_zero.view(np.uint32)

    def test_neg_inf(self):
        _assert_bitexact(circuit_neg(np.float32(np.inf)), np.float32(-np.inf))
        _assert_bitexact(circuit_neg(np.float32(-np.inf)), np.float32(np.inf))

    def test_abs_random(self):
        rng = np.random.default_rng(43)
        x = _rand32(10_000, rng)
        _assert_bitexact(circuit_abs(x), np.abs(x))

    def test_square_random(self):
        rng = np.random.default_rng(44)
        x = _rand32(10_000, rng)
        _assert_bitexact(circuit_square(x), x * x)


# ---------------------------------------------------------------
# Binary arithmetic
# ---------------------------------------------------------------

class TestBinaryArith:
    def test_add_random(self):
        rng = np.random.default_rng(100)
        a, b = _rand32(10_000, rng), _rand32(10_000, rng)
        _assert_bitexact(circuit_add(a, b), a + b)

    def test_sub_random(self):
        rng = np.random.default_rng(101)
        a, b = _rand32(10_000, rng), _rand32(10_000, rng)
        _assert_bitexact(circuit_sub(a, b), a - b)

    def test_mul_random(self):
        rng = np.random.default_rng(102)
        a, b = _rand32(10_000, rng), _rand32(10_000, rng)
        _assert_bitexact(circuit_mul(a, b), a * b)

    def test_div_random(self):
        rng = np.random.default_rng(103)
        a = _rand32(10_000, rng)
        b = _rand32(10_000, rng)
        # Avoid division by zero for cleaner comparison
        b[b == 0] = np.float32(1.0)
        _assert_bitexact(circuit_div(a, b), a / b)

    def test_add_edge_cases(self):
        vals = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 1e-38, -1e-38],
                        dtype=np.float32)
        for a in vals:
            for b in vals:
                expected = np.float32(a + b)
                result = circuit_add(np.float32(a), np.float32(b))
                # NaN: both should be NaN
                if np.isnan(expected):
                    assert np.isnan(result), f"add({a}, {b}): expected NaN"
                else:
                    _assert_bitexact(result, expected)


# ---------------------------------------------------------------
# Comparison / selection
# ---------------------------------------------------------------

class TestComparison:
    def test_max_random(self):
        rng = np.random.default_rng(200)
        a, b = _rand32(10_000, rng), _rand32(10_000, rng)
        _assert_bitexact(circuit_max(a, b), np.maximum(a, b))

    def test_cmp_le_random(self):
        rng = np.random.default_rng(201)
        a, b = _rand32(10_000, rng), _rand32(10_000, rng)
        expected = (a <= b).astype(np.uint8)
        np.testing.assert_array_equal(circuit_cmp_le(a, b), expected)

    def test_mux(self):
        sel = np.array([0, 1, 0, 1, 0], dtype=np.uint8)
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        b = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
        expected = np.array([1.0, 20.0, 3.0, 40.0, 5.0], dtype=np.float32)
        _assert_bitexact(circuit_mux(sel, a, b), expected)


# ---------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------

class TestReductions:
    def test_sum_random(self):
        rng = np.random.default_rng(300)
        x = _rand32(1000, rng).reshape(10, 100)
        _assert_bitexact(circuit_sum(x, axis=-1), x.sum(axis=-1))

    def test_reduce_max_random(self):
        rng = np.random.default_rng(301)
        x = _rand32(1000, rng).reshape(10, 100)
        _assert_bitexact(circuit_reduce_max(x, axis=-1), x.max(axis=-1))

    def test_argmax_random(self):
        rng = np.random.default_rng(302)
        x = _rand32(1000, rng).reshape(10, 100)
        np.testing.assert_array_equal(
            circuit_argmax(x, axis=-1), x.argmax(axis=-1))

    def test_mean_random(self):
        rng = np.random.default_rng(303)
        x = _rand32(100, rng).reshape(10, 10)
        _assert_bitexact(circuit_mean(x, axis=-1), x.mean(axis=-1))


# ---------------------------------------------------------------
# Composite operations
# ---------------------------------------------------------------

class TestComposite:
    def test_matmul_random(self):
        rng = np.random.default_rng(400)
        a = rng.standard_normal((4, 8)).astype(np.float32)
        b = rng.standard_normal((8, 6)).astype(np.float32)
        _assert_bitexact(circuit_matmul(a, b), a @ b)

    def test_softmax_shape(self):
        rng = np.random.default_rng(401)
        x = rng.standard_normal((4, 16)).astype(np.float32)
        result = circuit_softmax(x, axis=-1)
        assert result.shape == x.shape
        # Softmax rows should sum to ~1
        np.testing.assert_allclose(
            result.sum(axis=-1), np.ones(4, dtype=np.float32),
            rtol=1e-5)


# ---------------------------------------------------------------
# Float64 operations
# ---------------------------------------------------------------

class TestFloat64:
    def test_add_f64(self):
        rng = np.random.default_rng(500)
        a = rng.standard_normal(10_000).astype(np.float64)
        b = rng.standard_normal(10_000).astype(np.float64)
        _assert_bitexact_f64(circuit_add_f64(a, b), a + b)

    def test_mul_f64(self):
        rng = np.random.default_rng(501)
        a = rng.standard_normal(10_000).astype(np.float64)
        b = rng.standard_normal(10_000).astype(np.float64)
        _assert_bitexact_f64(circuit_mul_f64(a, b), a * b)

    def test_div_f64(self):
        rng = np.random.default_rng(502)
        a = rng.standard_normal(10_000).astype(np.float64)
        b = rng.standard_normal(10_000).astype(np.float64)
        b[b == 0] = 1.0
        _assert_bitexact_f64(circuit_div_f64(a, b), a / b)

    def test_mean_f64(self):
        rng = np.random.default_rng(503)
        x = rng.standard_normal((10, 100)).astype(np.float64)
        _assert_bitexact_f64(
            circuit_mean_f64(x, axis=-1), x.mean(axis=-1))

    def test_sum_f64(self):
        rng = np.random.default_rng(504)
        x = rng.standard_normal((10, 100)).astype(np.float64)
        _assert_bitexact_f64(
            circuit_sum_f64(x, axis=-1), x.sum(axis=-1))
