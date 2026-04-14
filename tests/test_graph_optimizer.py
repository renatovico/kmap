"""Tests for graph-level DAG optimizations.

Verifies that each optimization pass preserves output values
while reducing node count.
"""

import numpy as np
import pytest

from kllm.circuit_graph import CircuitGraph, Op, evaluate
from kllm.graph_optimizer import (
    constant_fold,
    dead_node_elimination,
    identity_elimination,
    optimize_graph,
    optimization_stats,
)


# ---------------------------------------------------------------
# Constant folding
# ---------------------------------------------------------------

class TestConstantFold:
    def test_fold_add_of_two_constants(self):
        g = CircuitGraph()
        a = g.const(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        b = g.const(np.array([4.0, 5.0, 6.0], dtype=np.float32))
        c = g.add(a, b)

        ref = evaluate(g)
        g2 = constant_fold(g)
        ref2 = evaluate(g2)

        np.testing.assert_array_equal(ref2[c], ref[c])
        # The add node should now be CONST
        assert g2.nodes[c].op == Op.CONST

    def test_fold_chain(self):
        """Chain: const → mul → add → neg → all should fold."""
        g = CircuitGraph()
        a = g.const(np.array([2.0, 3.0], dtype=np.float32))
        b = g.const(np.array([4.0, 5.0], dtype=np.float32))
        c = g.mul(a, b)
        d = g.const(np.float32(1.0))
        e = g.add(c, d)
        f = g.neg(e)

        ref = evaluate(g)
        g2 = constant_fold(g)
        ref2 = evaluate(g2)

        np.testing.assert_array_equal(ref2[f], ref[f])
        assert g2.nodes[f].op == Op.CONST

    def test_fold_preserves_inputs(self):
        """Nodes depending on INPUT should NOT be folded."""
        g = CircuitGraph()
        x = g.input((3,), name="x")
        w = g.const(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        y = g.mul(x, w)

        # Only const 'w' should be marked const; mul depends on input
        g2 = constant_fold(g)
        assert g2.nodes[w].op == Op.CONST
        assert g2.nodes[y].op == Op.MUL  # NOT folded

    def test_fold_lut(self):
        """LUT on constant input should fold."""
        g = CircuitGraph()
        x = g.const(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        y = g.lut(x, "silu")

        ref = evaluate(g)
        g2 = constant_fold(g)
        ref2 = evaluate(g2)

        np.testing.assert_allclose(ref2[y], ref[y], rtol=1e-6)
        assert g2.nodes[y].op == Op.CONST

    def test_fold_softmax(self):
        """Softmax composite on const input: all primitives should fold."""
        g = CircuitGraph()
        x = g.const(np.random.randn(2, 5).astype(np.float32))
        s = g.softmax(x, axis=-1)

        ref = evaluate(g)
        g2 = constant_fold(g)
        ref2 = evaluate(g2)

        np.testing.assert_allclose(ref2[s], ref[s], rtol=1e-5)
        assert g2.nodes[s].op == Op.CONST

    def test_fold_rms_norm(self):
        """RMSNorm composite on const input should fold completely."""
        g = CircuitGraph()
        x = g.const(np.random.randn(2, 4).astype(np.float32))
        w = g.const(np.ones(4, dtype=np.float32))
        r = g.rms_norm(x, w, eps_val=1e-5)

        ref = evaluate(g)
        g2 = constant_fold(g)
        ref2 = evaluate(g2)

        np.testing.assert_allclose(ref2[r], ref[r], rtol=1e-5)
        assert g2.nodes[r].op == Op.CONST

    def test_fold_matmul(self):
        g = CircuitGraph()
        a = g.const(np.random.randn(3, 4).astype(np.float32))
        b = g.const(np.random.randn(4, 5).astype(np.float32))
        c = g.matmul(a, b)

        ref = evaluate(g)
        g2 = constant_fold(g)
        ref2 = evaluate(g2)

        np.testing.assert_allclose(ref2[c], ref[c], rtol=1e-5)
        assert g2.nodes[c].op == Op.CONST

    def test_does_not_mutate_original(self):
        """Constant folding should not modify the original graph."""
        g = CircuitGraph()
        a = g.const(np.array([1.0], dtype=np.float32))
        b = g.const(np.array([2.0], dtype=np.float32))
        c = g.add(a, b)

        assert g.nodes[c].op == Op.ADD
        _ = constant_fold(g)
        assert g.nodes[c].op == Op.ADD  # original unchanged


# ---------------------------------------------------------------
# Dead node elimination
# ---------------------------------------------------------------

class TestDeadNodeElimination:
    def test_removes_unused_nodes(self):
        g = CircuitGraph()
        a = g.const(np.array([1.0], dtype=np.float32))
        b = g.const(np.array([2.0], dtype=np.float32))
        c = g.add(a, b)
        dead = g.const(np.array([99.0], dtype=np.float32))  # unused
        also_dead = g.mul(a, dead)  # unused

        assert len(g) == 5
        g2, id_map = dead_node_elimination(g, keep_ids=[c])
        assert len(g2) == 3  # a, b, c
        assert c in id_map

        ref = evaluate(g)
        ref2 = evaluate(g2)
        np.testing.assert_array_equal(ref2[id_map[c]], ref[c])

    def test_preserves_entire_chain(self):
        g = CircuitGraph()
        a = g.const(np.array([1.0], dtype=np.float32))
        b = g.neg(a)
        c = g.neg(b)
        d = g.neg(c)

        g2, id_map = dead_node_elimination(g, keep_ids=[d])
        assert len(g2) == 4  # all needed

    def test_multiple_outputs(self):
        g = CircuitGraph()
        a = g.const(np.array([1.0], dtype=np.float32))
        b = g.const(np.array([2.0], dtype=np.float32))
        c = g.add(a, b)
        d = g.mul(a, b)

        g2, id_map = dead_node_elimination(g, keep_ids=[c, d])
        assert len(g2) == 4  # a, b, c, d
        assert c in id_map and d in id_map

    def test_renumbers_contiguously(self):
        g = CircuitGraph()
        a = g.const(np.array([1.0], dtype=np.float32))
        b = g.const(np.array([2.0], dtype=np.float32))
        c = g.add(a, b)
        dead = g.const(np.array([99.0], dtype=np.float32))

        g2, id_map = dead_node_elimination(g, keep_ids=[c])
        # Node IDs should be 0, 1, 2
        assert [n.id for n in g2.nodes] == [0, 1, 2]


# ---------------------------------------------------------------
# Identity elimination
# ---------------------------------------------------------------

class TestIdentityElimination:
    def test_add_zero_right(self):
        g = CircuitGraph()
        x = g.const(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        zero = g.const(np.float32(0.0))
        y = g.add(x, zero)

        ref = evaluate(g)
        g2 = identity_elimination(g)
        ref2 = evaluate(g2)
        np.testing.assert_array_equal(ref2[y], ref[y])
        # The add should be redirected, so something downstream
        # would point to x's value

    def test_add_zero_left(self):
        g = CircuitGraph()
        zero = g.const(np.float32(0.0))
        x = g.const(np.array([1.0, 2.0], dtype=np.float32))
        y = g.add(zero, x)

        ref = evaluate(g)
        g2 = identity_elimination(g)
        ref2 = evaluate(g2)
        np.testing.assert_array_equal(ref2[y], ref[y])

    def test_mul_one(self):
        g = CircuitGraph()
        x = g.const(np.array([5.0, 6.0], dtype=np.float32))
        one = g.const(np.float32(1.0))
        y = g.mul(x, one)

        ref = evaluate(g)
        g2 = identity_elimination(g)
        ref2 = evaluate(g2)
        np.testing.assert_array_equal(ref2[y], ref[y])

    def test_div_one(self):
        g = CircuitGraph()
        x = g.const(np.array([5.0, 6.0], dtype=np.float32))
        one = g.const(np.float32(1.0))
        y = g.div(x, one)

        ref = evaluate(g)
        g2 = identity_elimination(g)
        ref2 = evaluate(g2)
        np.testing.assert_array_equal(ref2[y], ref[y])

    def test_sub_zero(self):
        g = CircuitGraph()
        x = g.const(np.array([5.0, 6.0], dtype=np.float32))
        zero = g.const(np.float32(0.0))
        y = g.sub(x, zero)

        ref = evaluate(g)
        g2 = identity_elimination(g)
        ref2 = evaluate(g2)
        np.testing.assert_array_equal(ref2[y], ref[y])

    def test_double_neg(self):
        g = CircuitGraph()
        x = g.const(np.array([1.0, -2.0, 3.0], dtype=np.float32))
        y = g.neg(x)
        z = g.neg(y)

        ref = evaluate(g)
        g2 = identity_elimination(g)
        ref2 = evaluate(g2)
        np.testing.assert_array_equal(ref2[z], ref[z])


# ---------------------------------------------------------------
# Combined optimizer
# ---------------------------------------------------------------

class TestOptimizeGraph:
    def test_full_pipeline(self):
        """All-constant graph folds to just the output const."""
        g = CircuitGraph()
        a = g.const(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        b = g.const(np.array([4.0, 5.0, 6.0], dtype=np.float32))
        c = g.add(a, b)
        d = g.mul(c, b)
        e = g.neg(d)
        dead = g.const(np.array([99.0], dtype=np.float32))

        ref = evaluate(g)
        opt_g, id_map = optimize_graph(g, output_ids=[e])
        opt_ref = evaluate(opt_g)

        np.testing.assert_array_equal(opt_ref[id_map[e]], ref[e])
        # Should be a single const node
        assert len(opt_g) == 1
        assert opt_g.nodes[0].op == Op.CONST

    def test_with_input_preserves_structure(self):
        """Graph with input nodes: only constant subgraphs fold."""
        g = CircuitGraph()
        x = g.input((3,), name="x")
        w = g.const(np.array([2.0, 3.0, 4.0], dtype=np.float32))
        b = g.const(np.array([0.1, 0.2, 0.3], dtype=np.float32))
        # w * x + b — w and b are const, but x is input
        wx = g.mul(w, x)
        out = g.add(wx, b)

        opt_g, id_map = optimize_graph(g, output_ids=[out])

        # x, w, b should survive (needed for mul/add)
        # mul and add cannot be folded
        # Count non-CONST nodes
        non_const = [n for n in opt_g.nodes if n.op != Op.CONST]
        assert len(non_const) >= 2  # input + mul + add (at least input and ops)

        # Verify evaluation with providing input
        inp = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        ref = evaluate(g, inputs={x: inp})
        opt_ref = evaluate(opt_g, inputs={id_map[x]: inp})
        np.testing.assert_allclose(opt_ref[id_map[out]], ref[out])

    def test_compiled_model_folds_completely(self):
        """A compiled model (all constants) folds to one output node."""
        from kllm.circuit_compiler import compile_model
        from tests.test_circuit_compiler import MockFabric

        fabric = MockFabric(num_layers=1, hidden_size=8, num_heads=2,
                            num_kv_heads=2, intermediate_size=16,
                            vocab_size=32)
        token_ids = [1, 5, 10]
        graph, logits_id, _kv = compile_model(fabric, token_ids)

        original_count = len(graph)
        ref = evaluate(graph)

        opt_g, id_map = optimize_graph(graph, output_ids=[logits_id])
        opt_ref = evaluate(opt_g)

        # Output must match
        np.testing.assert_allclose(
            opt_ref[id_map[logits_id]], ref[logits_id],
            rtol=1e-4, atol=1e-6)

        # Should have collapsed to 1 node (the logits constant)
        assert len(opt_g) == 1
        assert opt_g.nodes[0].op == Op.CONST

        # Massive reduction
        assert original_count > 50  # compiled model has many nodes
        stats = optimization_stats(graph, opt_g)
        assert stats["nodes_removed"] > 50

    def test_optimization_stats(self):
        g = CircuitGraph()
        a = g.const(np.array([1.0], dtype=np.float32))
        b = g.const(np.array([2.0], dtype=np.float32))
        c = g.add(a, b)
        d = g.mul(c, b)
        dead = g.const(np.array([99.0], dtype=np.float32))

        opt_g, _ = optimize_graph(g, output_ids=[d])
        stats = optimization_stats(g, opt_g)

        assert stats["original_nodes"] == 5
        assert stats["optimized_nodes"] == 1
        assert stats["nodes_removed"] == 4
        assert stats["gate_reduction_pct"] == 100.0  # all gates gone


class TestOptimizeGraphInputId:
    def test_id_map_correct(self):
        """Verify that id_map maps old output IDs to correct new IDs."""
        g = CircuitGraph()
        a = g.const(np.array([1.0, 2.0], dtype=np.float32))
        b = g.const(np.array([3.0, 4.0], dtype=np.float32))
        c = g.add(a, b)
        d = g.mul(a, b)

        ref = evaluate(g)

        opt_g, id_map = optimize_graph(g, output_ids=[c, d])
        opt_ref = evaluate(opt_g)

        np.testing.assert_array_equal(opt_ref[id_map[c]], ref[c])
        np.testing.assert_array_equal(opt_ref[id_map[d]], ref[d])
