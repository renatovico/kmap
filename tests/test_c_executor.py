"""Tests for the C circuit evaluator.

Verifies that evaluate_c() matches evaluate() (the NumPy reference)
for every op type and for full compiled transformer graphs.
"""

import numpy as np
import pytest

from kllm.graph.circuit_graph import CircuitGraph, Op
from kllm.graph.evaluator import evaluate
from kllm.graph.circuit_executor import evaluate_c


# ---------------------------------------------------------------
# Helper
# ---------------------------------------------------------------

def _assert_values_match(ref: dict, c_vals: dict, rtol=1e-5, atol=1e-6):
    """Assert all node values from C match the reference."""
    for nid in ref:
        r = np.asarray(ref[nid])
        c = np.asarray(c_vals[nid])
        np.testing.assert_allclose(c, r, rtol=rtol, atol=atol,
                                   err_msg=f"Node {nid} mismatch")


# ---------------------------------------------------------------
# Individual op tests
# ---------------------------------------------------------------

class TestUnaryOps:
    def test_neg(self):
        g = CircuitGraph()
        x = g.const(np.array([1.0, -2.0, 0.0, 3.14], dtype=np.float32))
        y = g.neg(x)
        _assert_values_match(evaluate(g), evaluate_c(g))

    def test_abs(self):
        g = CircuitGraph()
        x = g.const(np.array([-1.0, 2.0, -0.0, -3.14], dtype=np.float32))
        y = g.abs(x)
        _assert_values_match(evaluate(g), evaluate_c(g))

    def test_square(self):
        g = CircuitGraph()
        x = g.const(np.array([2.0, -3.0, 0.5], dtype=np.float32))
        y = g.square(x)
        _assert_values_match(evaluate(g), evaluate_c(g))


class TestLUTOps:
    def test_silu(self):
        g = CircuitGraph()
        x = g.const(np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32))
        y = g.lut(x, "silu")
        _assert_values_match(evaluate(g), evaluate_c(g))

    def test_exp(self):
        g = CircuitGraph()
        x = g.const(np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32))
        y = g.lut(x, "exp")
        _assert_values_match(evaluate(g), evaluate_c(g))

    def test_rsqrt(self):
        g = CircuitGraph()
        x = g.const(np.array([1.0, 4.0, 9.0, 0.25], dtype=np.float32))
        y = g.lut(x, "rsqrt")
        _assert_values_match(evaluate(g), evaluate_c(g))

    def test_cos(self):
        g = CircuitGraph()
        x = g.const(np.array([0.0, 1.0, 3.14], dtype=np.float32))
        y = g.lut(x, "cos")
        _assert_values_match(evaluate(g), evaluate_c(g))

    def test_sin(self):
        g = CircuitGraph()
        x = g.const(np.array([0.0, 1.0, 3.14], dtype=np.float32))
        y = g.lut(x, "sin")
        _assert_values_match(evaluate(g), evaluate_c(g))


class TestBinaryOps:
    def test_add_same_shape(self):
        g = CircuitGraph()
        a = g.const(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        b = g.const(np.array([4.0, 5.0, 6.0], dtype=np.float32))
        g.add(a, b)
        _assert_values_match(evaluate(g), evaluate_c(g))

    def test_add_broadcast_scalar(self):
        g = CircuitGraph()
        a = g.const(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        b = g.const(np.float32(10.0))
        g.add(a, b)
        _assert_values_match(evaluate(g), evaluate_c(g))

    def test_add_broadcast_dims(self):
        g = CircuitGraph()
        a = g.const(np.ones((2, 3), dtype=np.float32))
        b = g.const(np.array([10.0, 20.0, 30.0], dtype=np.float32))
        g.add(a, b)
        _assert_values_match(evaluate(g), evaluate_c(g))

    def test_add_broadcast_keepdims(self):
        """Broadcast pattern from reductions with keepdims=True."""
        g = CircuitGraph()
        a = g.const(np.ones((2, 3), dtype=np.float32))
        b = g.const(np.array([[10.0], [20.0]], dtype=np.float32))
        g.add(a, b)
        _assert_values_match(evaluate(g), evaluate_c(g))

    def test_sub(self):
        g = CircuitGraph()
        a = g.const(np.array([5.0, 3.0, 1.0], dtype=np.float32))
        b = g.const(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        g.sub(a, b)
        _assert_values_match(evaluate(g), evaluate_c(g))

    def test_mul_broadcast(self):
        g = CircuitGraph()
        a = g.const(np.ones((2, 3), dtype=np.float32) * 2)
        b = g.const(np.array([3.0, 4.0, 5.0], dtype=np.float32))
        g.mul(a, b)
        _assert_values_match(evaluate(g), evaluate_c(g))

    def test_div(self):
        g = CircuitGraph()
        a = g.const(np.array([6.0, 12.0, 9.0], dtype=np.float32))
        b = g.const(np.array([2.0, 3.0, 3.0], dtype=np.float32))
        g.div(a, b)
        _assert_values_match(evaluate(g), evaluate_c(g))

    def test_max(self):
        g = CircuitGraph()
        a = g.const(np.array([1.0, 5.0, 3.0], dtype=np.float32))
        b = g.const(np.array([4.0, 2.0, 6.0], dtype=np.float32))
        g.max(a, b)
        _assert_values_match(evaluate(g), evaluate_c(g))


class TestComparison:
    def test_cmp_le(self):
        g = CircuitGraph()
        a = g.const(np.array([1.0, 5.0, 3.0], dtype=np.float32))
        b = g.const(np.array([4.0, 2.0, 3.0], dtype=np.float32))
        g.cmp_le(a, b)
        ref = evaluate(g)
        c_vals = evaluate_c(g)
        # cmp_le returns uint8 in ref, float→uint8 in C
        for nid in ref:
            np.testing.assert_array_equal(
                np.asarray(c_vals[nid]),
                np.asarray(ref[nid]))

    def test_mux(self):
        g = CircuitGraph()
        a = g.const(np.array([1.0, 5.0, 3.0], dtype=np.float32))
        b = g.const(np.array([4.0, 2.0, 3.0], dtype=np.float32))
        cond = g.cmp_le(a, b)
        x = g.const(np.array([10.0, 20.0, 30.0], dtype=np.float32))
        y = g.const(np.array([40.0, 50.0, 60.0], dtype=np.float32))
        g.mux(cond, x, y)
        _assert_values_match(evaluate(g), evaluate_c(g))


class TestMatmul:
    def test_2d(self):
        g = CircuitGraph()
        a = g.const(np.random.randn(3, 4).astype(np.float32))
        b = g.const(np.random.randn(4, 5).astype(np.float32))
        g.matmul(a, b)
        _assert_values_match(evaluate(g), evaluate_c(g), rtol=1e-4)

    def test_3d_batched(self):
        g = CircuitGraph()
        a = g.const(np.random.randn(2, 3, 4).astype(np.float32))
        b = g.const(np.random.randn(2, 4, 5).astype(np.float32))
        g.matmul(a, b)
        _assert_values_match(evaluate(g), evaluate_c(g), rtol=1e-4)

    def test_4d_batched(self):
        """4D matmul as used in attention: (B, H, S, D) @ (B, H, D, S)."""
        g = CircuitGraph()
        a = g.const(np.random.randn(1, 2, 3, 4).astype(np.float32))
        b = g.const(np.random.randn(1, 2, 4, 3).astype(np.float32))
        g.matmul(a, b)
        _assert_values_match(evaluate(g), evaluate_c(g), rtol=1e-4)


class TestReductions:
    def test_sum_axis1(self):
        g = CircuitGraph()
        x = g.const(np.array([[1.0, 2.0, 3.0],
                               [4.0, 5.0, 6.0]], dtype=np.float32))
        g.sum(x, axis=1)
        _assert_values_match(evaluate(g), evaluate_c(g))

    def test_sum_keepdims(self):
        g = CircuitGraph()
        x = g.const(np.array([[1.0, 2.0, 3.0],
                               [4.0, 5.0, 6.0]], dtype=np.float32))
        g.sum(x, axis=-1, keepdims=True)
        _assert_values_match(evaluate(g), evaluate_c(g))

    def test_max_reduce(self):
        g = CircuitGraph()
        x = g.const(np.array([[1.0, 5.0, 3.0],
                               [4.0, 2.0, 6.0]], dtype=np.float32))
        g.max_reduce(x, axis=-1, keepdims=True)
        _assert_values_match(evaluate(g), evaluate_c(g))

    def test_mean(self):
        g = CircuitGraph()
        x = g.const(np.array([[2.0, 4.0, 6.0],
                               [8.0, 10.0, 12.0]], dtype=np.float32))
        g.mean(x, axis=-1, keepdims=True)
        _assert_values_match(evaluate(g), evaluate_c(g))

    def test_argmax(self):
        g = CircuitGraph()
        x = g.const(np.array([[1.0, 5.0, 3.0],
                               [4.0, 2.0, 6.0]], dtype=np.float32))
        g.argmax(x, axis=-1)
        ref = evaluate(g)
        c_vals = evaluate_c(g)
        for nid in ref:
            np.testing.assert_array_equal(
                np.asarray(c_vals[nid]),
                np.asarray(ref[nid]))


class TestWiring:
    def test_reshape(self):
        g = CircuitGraph()
        x = g.const(np.arange(6, dtype=np.float32))
        g.reshape(x, (2, 3))
        _assert_values_match(evaluate(g), evaluate_c(g))

    def test_transpose(self):
        g = CircuitGraph()
        x = g.const(np.arange(6, dtype=np.float32).reshape(2, 3))
        g.transpose(x, (1, 0))
        _assert_values_match(evaluate(g), evaluate_c(g))

    def test_transpose_3d(self):
        g = CircuitGraph()
        x = g.const(np.arange(24, dtype=np.float32).reshape(2, 3, 4))
        g.transpose(x, (1, 0, 2))
        _assert_values_match(evaluate(g), evaluate_c(g))

    def test_slice(self):
        g = CircuitGraph()
        x = g.const(np.arange(12, dtype=np.float32).reshape(3, 4))
        g.slice(x, (slice(0, 2), slice(1, 3)))
        _assert_values_match(evaluate(g), evaluate_c(g))

    def test_concat(self):
        g = CircuitGraph()
        a = g.const(np.array([[1.0, 2.0]], dtype=np.float32))
        b = g.const(np.array([[3.0, 4.0]], dtype=np.float32))
        g.concat([a, b], axis=0)
        _assert_values_match(evaluate(g), evaluate_c(g))

    def test_repeat(self):
        g = CircuitGraph()
        x = g.const(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        g.repeat(x, repeats=3, axis=0)
        _assert_values_match(evaluate(g), evaluate_c(g))

    def test_expand_dims(self):
        g = CircuitGraph()
        x = g.const(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        g.expand_dims(x, axis=0)
        _assert_values_match(evaluate(g), evaluate_c(g))


class TestComposite:
    def test_softmax(self):
        """Softmax decomposition: max_reduce → sub → exp → sum → div."""
        g = CircuitGraph()
        x = g.const(np.random.randn(2, 5).astype(np.float32))
        g.softmax(x, axis=-1)
        _assert_values_match(evaluate(g), evaluate_c(g), rtol=1e-4)

    def test_rms_norm(self):
        """RMSNorm: square → mean → add eps → rsqrt → mul → mul."""
        g = CircuitGraph()
        x = g.const(np.random.randn(2, 4).astype(np.float32))
        w = g.const(np.ones(4, dtype=np.float32))
        g.rms_norm(x, w, eps_val=1e-5)
        _assert_values_match(evaluate(g), evaluate_c(g), rtol=1e-4)


# ---------------------------------------------------------------
# Full transformer graph test
# ---------------------------------------------------------------

class TestFullGraph:
    """Test that C evaluator matches reference on a compiled model."""

    def _make_mock_fabric(self):
        """Tiny LLaMA-like model for testing."""
        from tests.test_circuit_compiler import MockFabric
        return MockFabric(num_layers=1, hidden_size=8, num_heads=2,
                          num_kv_heads=2, intermediate_size=16,
                          vocab_size=32)

    def test_compiled_model_matches_reference(self):
        """Compiled graph: C eval == NumPy eval."""
        from kllm.compiler.circuit_compiler import compile_model
        from tests.test_circuit_compiler import MockFabric

        fabric = MockFabric(num_layers=1, hidden_size=8, num_heads=2,
                            num_kv_heads=2, intermediate_size=16,
                            vocab_size=32)
        token_ids = [1, 5, 10]
        graph, logits_id, _kv = compile_model(fabric, token_ids)

        ref_vals = evaluate(graph)
        c_vals = evaluate_c(graph)

        # Check logits match
        np.testing.assert_allclose(
            c_vals[logits_id], ref_vals[logits_id],
            rtol=1e-3, atol=1e-5,
            err_msg="Logits mismatch between C and reference evaluator")


# ---------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------

class TestSerialization:
    def test_round_trip(self, tmp_path):
        """Serialize → deserialize → evaluate gives same results."""
        g = CircuitGraph()
        a = g.const(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        b = g.const(np.array([4.0, 5.0, 6.0], dtype=np.float32))
        c = g.add(a, b)

        ref = evaluate(g)

        path = str(tmp_path / "test_graph")
        g.serialize(path)

        g2 = CircuitGraph.deserialize(path)
        ref2 = evaluate(g2)

        np.testing.assert_array_equal(ref2[c], ref[c])

    def test_round_trip_complex(self, tmp_path):
        """Round-trip with slices, reductions, and LUTs."""
        g = CircuitGraph()
        x = g.const(np.random.randn(4, 6).astype(np.float32))
        s = g.slice(x, (slice(0, 2), slice(1, 4)))
        y = g.lut(s, "silu")
        z = g.sum(y, axis=-1, keepdims=True)

        ref = evaluate(g)

        path = str(tmp_path / "complex_graph")
        g.serialize(path)

        g2 = CircuitGraph.deserialize(path)
        ref2 = evaluate(g2)

        np.testing.assert_allclose(ref2[z], ref[z], rtol=1e-5)
