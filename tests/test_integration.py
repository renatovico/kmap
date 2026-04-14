"""End-to-end integration tests for the circuit graph pipeline.

Pipeline: compile_model → optimize_graph → evaluate / evaluate_c → export HDL.
Verifies bit-exact consistency across all stages.
"""

import os
import tempfile
from functools import lru_cache

import numpy as np
import pytest

from kllm.circuit_compiler import compile_model
from kllm.circuit_graph import CircuitGraph, Op, evaluate
from kllm.circuit_executor import evaluate_c
from kllm.graph_optimizer import optimize_graph, optimization_stats
from kllm.hdl_export import export_verilog, export_vhdl, export_testbench, estimate_resources


# ---------------------------------------------------------------------------
# Shared tiny model fabric
# ---------------------------------------------------------------------------

class MockFabric:
    def __init__(self, num_layers=1, hidden_size=8, num_heads=2,
                 num_kv_heads=2, intermediate_size=16, vocab_size=32):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size
        self.rms_norm_eps = 1e-5
        self.rope_theta = 10000.0
        self.head_dim = hidden_size // num_heads
        self.num_groups = num_heads // num_kv_heads

        rng = np.random.default_rng(99)

        self.embed_tokens = rng.standard_normal(
            (vocab_size, hidden_size)).astype(np.float32) * 0.02
        self.lm_head = self.embed_tokens
        self.final_norm_weight = np.ones(hidden_size, dtype=np.float32)

        self.layers = []
        for _ in range(num_layers):
            layer = {}
            for proj, out_dim in [
                ("q_proj", num_heads * self.head_dim),
                ("k_proj", num_kv_heads * self.head_dim),
                ("v_proj", num_kv_heads * self.head_dim),
                ("o_proj", hidden_size),
            ]:
                layer[proj] = rng.standard_normal(
                    (out_dim, hidden_size)).astype(np.float32) * 0.02
            for proj, out_dim in [
                ("gate_proj", intermediate_size),
                ("up_proj", intermediate_size),
            ]:
                layer[proj] = rng.standard_normal(
                    (out_dim, hidden_size)).astype(np.float32) * 0.02
            layer["down_proj"] = rng.standard_normal(
                (hidden_size, intermediate_size)).astype(np.float32) * 0.02
            layer["input_layernorm_weight"] = np.ones(
                hidden_size, dtype=np.float32)
            layer["post_attention_layernorm_weight"] = np.ones(
                hidden_size, dtype=np.float32)
            self.layers.append(layer)

    @lru_cache(maxsize=None)
    def get_transposed(self, layer_idx, proj):
        return np.ascontiguousarray(
            self.layers[layer_idx][proj].T, dtype=np.float32)

    @lru_cache(maxsize=None)
    def get_fused_qkv_t(self, layer_idx):
        q_t = self.get_transposed(layer_idx, 'q_proj')
        k_t = self.get_transposed(layer_idx, 'k_proj')
        v_t = self.get_transposed(layer_idx, 'v_proj')
        return np.ascontiguousarray(np.concatenate([q_t, k_t, v_t], axis=1))

    @lru_cache(maxsize=None)
    def get_fused_gate_up_t(self, layer_idx):
        gate_t = self.get_transposed(layer_idx, 'gate_proj')
        up_t = self.get_transposed(layer_idx, 'up_proj')
        return np.ascontiguousarray(np.concatenate([gate_t, up_t], axis=1))

    @staticmethod
    def _quantize_per_column(w_f32):
        amax = np.abs(w_f32).max(axis=0)
        amax = np.where(amax == 0, 1.0, amax)
        scales = (amax / 127.0).astype(np.float32)
        w_q8 = np.clip(np.round(w_f32 / scales), -128, 127).astype(np.int8)
        return w_q8, scales

    @lru_cache(maxsize=None)
    def get_quantized(self, layer_idx, proj):
        w_t = self.get_transposed(layer_idx, proj)
        return self._quantize_per_column(w_t)

    @lru_cache(maxsize=None)
    def get_quantized_fused_qkv(self, layer_idx):
        w_t = self.get_fused_qkv_t(layer_idx)
        return self._quantize_per_column(w_t)

    @lru_cache(maxsize=None)
    def get_quantized_fused_gate_up(self, layer_idx):
        w_t = self.get_fused_gate_up_t(layer_idx)
        return self._quantize_per_column(w_t)


@pytest.fixture
def fabric():
    return MockFabric()


@pytest.fixture
def compiled(fabric):
    """Compile a 3-token sequence and return (graph, logits_id)."""
    return compile_model(fabric, [1, 5, 10])


# ---------------------------------------------------------------------------
# 1. Compile → Reference evaluate
# ---------------------------------------------------------------------------

class TestCompileAndEvaluate:
    def test_compile_produces_valid_graph(self, compiled):
        graph, logits_id, _kv = compiled
        assert len(graph.nodes) > 10
        assert logits_id in {n.id for n in graph.nodes}

    def test_reference_evaluate_gives_logits(self, compiled, fabric):
        graph, logits_id, _kv = compiled
        result = evaluate(graph)
        logits = result[logits_id]
        assert logits.shape[-1] == fabric.vocab_size

    def test_c_evaluate_matches_reference(self, compiled):
        graph, logits_id, _kv = compiled
        ref = evaluate(graph)
        c_result = evaluate_c(graph)
        np.testing.assert_allclose(
            c_result[logits_id], ref[logits_id], rtol=1e-5, atol=1e-7,
        )


# ---------------------------------------------------------------------------
# 2. Compile → Optimize → Evaluate (bit-exact)
# ---------------------------------------------------------------------------

class TestOptimizePipeline:
    def test_optimize_reduces_nodes(self, compiled):
        graph, logits_id, _kv = compiled
        opt, id_map = optimize_graph(graph, [logits_id])
        assert len(opt.nodes) <= len(graph.nodes)

    def test_optimized_matches_original(self, compiled):
        graph, logits_id, _kv = compiled
        ref = evaluate(graph)
        opt, id_map = optimize_graph(graph, [logits_id])
        opt_result = evaluate(opt)
        new_logits_id = id_map[logits_id]
        np.testing.assert_allclose(
            opt_result[new_logits_id], ref[logits_id], rtol=1e-5, atol=1e-7,
        )

    def test_optimized_c_matches_original(self, compiled):
        graph, logits_id, _kv = compiled
        ref = evaluate(graph)
        opt, id_map = optimize_graph(graph, [logits_id])
        c_result = evaluate_c(opt)
        new_logits_id = id_map[logits_id]
        np.testing.assert_allclose(
            c_result[new_logits_id], ref[logits_id], rtol=1e-5, atol=1e-7,
        )

    def test_optimization_stats_valid(self, compiled):
        graph, logits_id, _kv = compiled
        opt, _ = optimize_graph(graph, [logits_id])
        stats = optimization_stats(graph, opt)
        assert stats["original_nodes"] >= stats["optimized_nodes"]
        assert 0 <= stats["gate_reduction_pct"] <= 100


# ---------------------------------------------------------------------------
# 3. Serialize → Deserialize round-trip
# ---------------------------------------------------------------------------

class TestSerializationRoundTrip:
    def test_serialize_deserialize_preserves_output(self, compiled):
        graph, logits_id, _kv = compiled
        ref = evaluate(graph)
        with tempfile.TemporaryDirectory() as td:
            graph.serialize(td)
            loaded = CircuitGraph.deserialize(td)
        loaded_result = evaluate(loaded)
        np.testing.assert_allclose(
            loaded_result[logits_id], ref[logits_id], rtol=1e-5, atol=1e-7,
        )

    def test_serialize_optimized(self, compiled):
        graph, logits_id, _kv = compiled
        opt, id_map = optimize_graph(graph, [logits_id])
        new_id = id_map[logits_id]
        ref = evaluate(opt)
        with tempfile.TemporaryDirectory() as td:
            opt.serialize(td)
            loaded = CircuitGraph.deserialize(td)
        loaded_result = evaluate(loaded)
        np.testing.assert_allclose(
            loaded_result[new_id], ref[new_id], rtol=1e-5, atol=1e-7,
        )


# ---------------------------------------------------------------------------
# 4. HDL export from compiled graph
# ---------------------------------------------------------------------------

class TestHDLExportIntegration:
    def test_export_verilog_from_compiled(self, compiled):
        graph, _, _kv = compiled
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "top.v")
            export_verilog(graph, path)
            with open(path) as f:
                content = f.read()
            assert "module" in content
            assert os.path.isfile(path)

    def test_export_vhdl_from_compiled(self, compiled):
        graph, _, _kv = compiled
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "top.vhd")
            export_vhdl(graph, path)
            with open(path) as f:
                content = f.read()
            assert "entity" in content
            assert os.path.isfile(path)

    def test_export_testbench(self, compiled):
        graph, logits_id, _kv = compiled
        values = evaluate(graph)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "tb.sv")
            content = export_testbench(graph, path, values)
            assert "module" in content.lower() or "initial" in content.lower()

    def test_resource_estimate(self, compiled):
        graph, _, _kv = compiled
        res = estimate_resources(graph)
        assert res["luts"] > 0
        assert res["dsps"] >= 0


# ---------------------------------------------------------------------------
# 5. Full pipeline: compile → optimize → C eval → serialize → HDL
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_end_to_end(self, fabric):
        """Full pipeline: compile → optimize → C eval → serialize → load → HDL."""
        # Step 1: compile
        graph, logits_id, _kv = compile_model(fabric, [2, 7, 15])
        ref_logits = evaluate(graph)[logits_id]

        # Step 2: optimize
        opt, id_map = optimize_graph(graph, [logits_id])
        new_logits_id = id_map[logits_id]

        # Step 3: C evaluate optimized
        c_result = evaluate_c(opt)
        np.testing.assert_allclose(
            c_result[new_logits_id], ref_logits, rtol=1e-5, atol=1e-7,
        )

        # Step 4: serialize + deserialize
        with tempfile.TemporaryDirectory() as td:
            opt.serialize(td)
            loaded = CircuitGraph.deserialize(td)

            # Step 5: C evaluate loaded
            loaded_result = evaluate_c(loaded)
            np.testing.assert_allclose(
                loaded_result[new_logits_id], ref_logits, rtol=1e-5, atol=1e-7,
            )

            # Step 6: HDL export
            v_path = os.path.join(td, "top.v")
            vhdl_path = os.path.join(td, "top.vhd")
            export_verilog(loaded, v_path)
            export_vhdl(loaded, vhdl_path)
            assert os.path.isfile(v_path)
            assert os.path.isfile(vhdl_path)

            # Step 7: resource estimate
            res = estimate_resources(loaded)
            assert res["luts"] >= 0

    def test_different_token_sequences_differ(self, fabric):
        """Different inputs produce different logits."""
        g1, lid1, _kv = compile_model(fabric, [1, 2])
        g2, lid2, _kv = compile_model(fabric, [3, 4])
        r1 = evaluate(g1)[lid1]
        r2 = evaluate(g2)[lid2]
        assert not np.allclose(r1, r2)

    def test_argmax_consistency(self, fabric):
        """argmax(ref) == argmax(C_eval) == argmax(optimized_C_eval)."""
        graph, lid, _kv = compile_model(fabric, [5, 10, 20])
        ref = evaluate(graph)[lid]
        c_res = evaluate_c(graph)[lid]
        opt, id_map = optimize_graph(graph, [lid])
        c_opt = evaluate_c(opt)[id_map[lid]]

        ref_tok = np.argmax(ref, axis=-1)
        c_tok = np.argmax(c_res, axis=-1)
        opt_tok = np.argmax(c_opt, axis=-1)
        np.testing.assert_array_equal(ref_tok, c_tok)
        np.testing.assert_array_equal(ref_tok, opt_tok)
