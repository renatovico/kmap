"""Tests for the circuit graph — DAG representation and reference evaluator.

The circuit graph is the core data structure: it defines the entire
computation as a DAG of gate nodes.  The reference evaluator (NumPy)
produces the golden output that the C executor must match.
"""

import numpy as np
import pytest

from kllm.circuit_graph import CircuitGraph, Op
from kllm.evaluator import evaluate


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _rand32(shape, rng):
    return rng.standard_normal(shape).astype(np.float32)


# ---------------------------------------------------------------
# Basic node construction
# ---------------------------------------------------------------

class TestGraphConstruction:
    def test_const_node(self):
        g = CircuitGraph()
        w = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        nid = g.const(w, name="weights")
        assert nid == 0
        assert len(g) == 1
        assert g.nodes[0].op == Op.CONST

    def test_add_two_consts(self):
        g = CircuitGraph()
        a = g.const(np.float32(3.0))
        b = g.const(np.float32(4.0))
        c = g.add(a, b)
        assert c == 2
        assert g.nodes[c].inputs == [0, 1]

    def test_topological_order(self):
        g = CircuitGraph()
        a = g.const(np.float32(1.0))
        b = g.const(np.float32(2.0))
        c = g.add(a, b)
        d = g.mul(c, b)
        order = g.topological_order()
        # a and b before c, c before d
        assert order.index(a) < order.index(c)
        assert order.index(b) < order.index(c)
        assert order.index(c) < order.index(d)

    def test_repr(self):
        g = CircuitGraph()
        g.const(np.float32(1.0))
        g.const(np.float32(2.0))
        g.add(0, 1)
        r = repr(g)
        assert "3 nodes" in r
        assert "add=1" in r

    def test_gate_count(self):
        g = CircuitGraph()
        a = g.const(np.float32(1.0))
        b = g.const(np.float32(2.0))
        c = g.add(a, b)
        g.reshape(c, (1,))
        counts = g.gate_count()
        assert counts["gate"] == 1  # only the add
        assert counts["wire"] == 3  # 2 consts + 1 reshape
        assert counts["total"] == 4


# ---------------------------------------------------------------
# Reference evaluator — arithmetic
# ---------------------------------------------------------------

class TestEvalArithmetic:
    def test_add(self):
        g = CircuitGraph()
        a = g.const(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        b = g.const(np.array([4.0, 5.0, 6.0], dtype=np.float32))
        c = g.add(a, b)
        result = evaluate(g)[c]
        np.testing.assert_array_equal(result, [5.0, 7.0, 9.0])

    def test_sub(self):
        g = CircuitGraph()
        a = g.const(np.array([10.0, 20.0], dtype=np.float32))
        b = g.const(np.array([3.0, 7.0], dtype=np.float32))
        c = g.sub(a, b)
        result = evaluate(g)[c]
        np.testing.assert_array_equal(result, [7.0, 13.0])

    def test_mul(self):
        g = CircuitGraph()
        a = g.const(np.array([2.0, 3.0], dtype=np.float32))
        b = g.const(np.array([4.0, 5.0], dtype=np.float32))
        c = g.mul(a, b)
        result = evaluate(g)[c]
        np.testing.assert_array_equal(result, [8.0, 15.0])

    def test_div(self):
        g = CircuitGraph()
        a = g.const(np.array([10.0, 21.0], dtype=np.float32))
        b = g.const(np.array([2.0, 7.0], dtype=np.float32))
        c = g.div(a, b)
        result = evaluate(g)[c]
        np.testing.assert_array_equal(result, [5.0, 3.0])

    def test_neg(self):
        g = CircuitGraph()
        a = g.const(np.array([1.0, -2.0, 0.0], dtype=np.float32))
        c = g.neg(a)
        result = evaluate(g)[c]
        expected = np.array([-1.0, 2.0, -0.0], dtype=np.float32)
        # Check bit patterns (neg zero)
        np.testing.assert_array_equal(
            result.view(np.uint32), expected.view(np.uint32))

    def test_abs(self):
        g = CircuitGraph()
        a = g.const(np.array([-5.0, 3.0, -0.0], dtype=np.float32))
        c = g.abs(a)
        result = evaluate(g)[c]
        np.testing.assert_array_equal(result, [5.0, 3.0, 0.0])

    def test_square(self):
        g = CircuitGraph()
        a = g.const(np.array([3.0, -4.0], dtype=np.float32))
        c = g.square(a)
        result = evaluate(g)[c]
        np.testing.assert_array_equal(result, [9.0, 16.0])

    def test_random_add_bitexact(self):
        """Random values must match NumPy bit-for-bit."""
        rng = np.random.default_rng(42)
        av = _rand32(10_000, rng)
        bv = _rand32(10_000, rng)
        g = CircuitGraph()
        a = g.const(av)
        b = g.const(bv)
        c = g.add(a, b)
        result = evaluate(g)[c]
        np.testing.assert_array_equal(
            result.view(np.uint32), (av + bv).view(np.uint32))


# ---------------------------------------------------------------
# Reference evaluator — comparison / selection
# ---------------------------------------------------------------

class TestEvalComparison:
    def test_max(self):
        g = CircuitGraph()
        a = g.const(np.array([1.0, 5.0, 3.0], dtype=np.float32))
        b = g.const(np.array([4.0, 2.0, 3.0], dtype=np.float32))
        c = g.max(a, b)
        result = evaluate(g)[c]
        np.testing.assert_array_equal(result, [4.0, 5.0, 3.0])

    def test_cmp_le(self):
        g = CircuitGraph()
        a = g.const(np.array([1.0, 5.0, 3.0], dtype=np.float32))
        b = g.const(np.array([4.0, 2.0, 3.0], dtype=np.float32))
        c = g.cmp_le(a, b)
        result = evaluate(g)[c]
        np.testing.assert_array_equal(result, [1, 0, 1])

    def test_mux(self):
        g = CircuitGraph()
        cond = g.const(np.array([0, 1, 0], dtype=np.uint8))
        a = g.const(np.array([10.0, 20.0, 30.0], dtype=np.float32))
        b = g.const(np.array([100.0, 200.0, 300.0], dtype=np.float32))
        c = g.mux(cond, a, b)
        result = evaluate(g)[c]
        np.testing.assert_array_equal(result, [10.0, 200.0, 30.0])


# ---------------------------------------------------------------
# Reference evaluator — reductions
# ---------------------------------------------------------------

class TestEvalReductions:
    def test_sum(self):
        g = CircuitGraph()
        x = g.const(np.array([[1.0, 2.0, 3.0],
                               [4.0, 5.0, 6.0]], dtype=np.float32))
        s = g.sum(x, axis=-1)
        result = evaluate(g)[s]
        np.testing.assert_array_equal(result, [6.0, 15.0])

    def test_max_reduce(self):
        g = CircuitGraph()
        x = g.const(np.array([[3.0, 1.0, 2.0]], dtype=np.float32))
        m = g.max_reduce(x, axis=-1)
        result = evaluate(g)[m]
        np.testing.assert_array_equal(result, [3.0])

    def test_argmax(self):
        g = CircuitGraph()
        x = g.const(np.array([[1.0, 5.0, 3.0]], dtype=np.float32))
        idx = g.argmax(x, axis=-1)
        result = evaluate(g)[idx]
        np.testing.assert_array_equal(result, [1])

    def test_mean(self):
        g = CircuitGraph()
        x = g.const(np.array([[2.0, 4.0, 6.0]], dtype=np.float32))
        m = g.mean(x, axis=-1)
        result = evaluate(g)[m]
        np.testing.assert_array_equal(result, [4.0])

    def test_matmul(self):
        rng = np.random.default_rng(100)
        av = rng.standard_normal((3, 4)).astype(np.float32)
        bv = rng.standard_normal((4, 5)).astype(np.float32)
        g = CircuitGraph()
        a = g.const(av)
        b = g.const(bv)
        c = g.matmul(a, b)
        result = evaluate(g)[c]
        np.testing.assert_array_equal(
            result.view(np.uint32), (av @ bv).view(np.uint32))


# ---------------------------------------------------------------
# Reference evaluator — wire routing (zero gates)
# ---------------------------------------------------------------

class TestEvalWiring:
    def test_reshape(self):
        g = CircuitGraph()
        x = g.const(np.arange(6, dtype=np.float32))
        r = g.reshape(x, (2, 3))
        result = evaluate(g)[r]
        assert result.shape == (2, 3)

    def test_transpose(self):
        g = CircuitGraph()
        x = g.const(np.arange(6, dtype=np.float32).reshape(2, 3))
        t = g.transpose(x, (1, 0))
        result = evaluate(g)[t]
        assert result.shape == (3, 2)

    def test_concat(self):
        g = CircuitGraph()
        a = g.const(np.array([1.0, 2.0], dtype=np.float32))
        b = g.const(np.array([3.0, 4.0], dtype=np.float32))
        c = g.concat([a, b], axis=0)
        result = evaluate(g)[c]
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0, 4.0])

    def test_repeat(self):
        g = CircuitGraph()
        x = g.const(np.array([[1.0, 2.0]], dtype=np.float32))
        r = g.repeat(x, repeats=3, axis=0)
        result = evaluate(g)[r]
        assert result.shape == (3, 2)
        np.testing.assert_array_equal(result[0], result[1])

    def test_slice(self):
        g = CircuitGraph()
        x = g.const(np.arange(10, dtype=np.float32))
        s = g.slice(x, (slice(2, 5),))
        result = evaluate(g)[s]
        np.testing.assert_array_equal(result, [2.0, 3.0, 4.0])


# ---------------------------------------------------------------
# Reference evaluator — LUT (activation functions)
# ---------------------------------------------------------------

class TestEvalLUT:
    def test_exp_lut(self):
        g = CircuitGraph()
        x = g.const(np.array([0.0, 1.0, -1.0], dtype=np.float32))
        e = g.lut(x, "exp")
        result = evaluate(g)[e]
        expected = np.exp(np.array([0.0, 1.0, -1.0], dtype=np.float64)
                          ).astype(np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_silu_lut(self):
        g = CircuitGraph()
        x = g.const(np.array([0.0, 1.0, -1.0], dtype=np.float32))
        s = g.lut(x, "silu")
        result = evaluate(g)[s]
        assert result.shape == (3,)
        # SiLU(0) = 0
        assert result[0] == 0.0

    def test_unknown_lut_raises(self):
        g = CircuitGraph()
        x = g.const(np.float32(1.0))
        g.lut(x, "unknown_fn")
        with pytest.raises(ValueError, match="Unknown LUT"):
            evaluate(g)


# ---------------------------------------------------------------
# Composite subgraphs
# ---------------------------------------------------------------

class TestCompositeSubgraphs:
    def test_softmax(self):
        g = CircuitGraph()
        x = g.const(np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32))
        s = g.softmax(x, axis=-1)
        result = evaluate(g)[s]
        assert result.shape == (1, 4)
        np.testing.assert_allclose(result.sum(axis=-1), [1.0], rtol=1e-5)
        # Softmax produces multiple intermediate nodes
        assert len(g) > 2  # not just const + one node

    def test_rms_norm(self):
        g = CircuitGraph()
        x_val = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        w_val = np.ones(4, dtype=np.float32)
        x = g.const(x_val)
        w = g.const(w_val)
        n = g.rms_norm(x, w, eps_val=1e-5)
        result = evaluate(g)[n]
        assert result.shape == (1, 4)
        # RMSNorm should normalize — check not all-zeros
        assert np.any(result != 0)

    def test_softmax_decomposes_to_primitives(self):
        """Softmax should create max_reduce, sub, lut, sum, div nodes."""
        g = CircuitGraph()
        x = g.const(np.ones((2, 4), dtype=np.float32))
        g.softmax(x)
        ops = {n.op for n in g.nodes}
        assert Op.MAX_REDUCE in ops
        assert Op.SUB in ops
        assert Op.LUT in ops
        assert Op.SUM in ops
        assert Op.DIV in ops


# ---------------------------------------------------------------
# Input nodes
# ---------------------------------------------------------------

class TestInputNodes:
    def test_input_evaluation(self):
        g = CircuitGraph()
        x = g.input((4,), name="tokens")
        w = g.const(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        y = g.add(x, w)
        result = evaluate(g, {x: np.array([10.0, 20.0, 30.0, 40.0],
                                           dtype=np.float32)})[y]
        np.testing.assert_array_equal(result, [11.0, 22.0, 33.0, 44.0])

    def test_missing_input_raises(self):
        g = CircuitGraph()
        g.input((4,), name="tokens")
        with pytest.raises(ValueError, match="Missing input"):
            evaluate(g)


# ---------------------------------------------------------------
# Cast
# ---------------------------------------------------------------

class TestCast:
    def test_float32_to_float64(self):
        g = CircuitGraph()
        x = g.const(np.array([1.5, 2.5], dtype=np.float32))
        c = g.cast(x, np.dtype(np.float64))
        result = evaluate(g)[c]
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, [1.5, 2.5])
