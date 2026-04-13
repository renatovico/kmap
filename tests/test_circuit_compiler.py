"""Tests for the circuit compiler — transformer → CircuitGraph.

Uses a tiny mock Fabric (2 layers, dim=16, 2 heads) to test that
the compiler produces a valid graph whose reference evaluation
matches numpy execution.
"""

import numpy as np
import pytest

from kllm.circuit_compiler import compile_model, _build_rope_const
from kllm.circuit_graph import CircuitGraph, Op, evaluate


# ---------------------------------------------------------------
# Minimal mock Fabric
# ---------------------------------------------------------------

class MockFabric:
    """Tiny LLaMA-like config for testing the compiler."""

    def __init__(self, num_layers: int = 2, hidden_size: int = 16,
                 num_heads: int = 2, num_kv_heads: int = 2,
                 intermediate_size: int = 32, vocab_size: int = 64,
                 rms_norm_eps: float = 1e-5, rope_theta: float = 10000.0):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.head_dim = hidden_size // num_heads
        self.num_groups = num_heads // num_kv_heads

        rng = np.random.default_rng(42)

        self.embed_tokens = rng.standard_normal(
            (vocab_size, hidden_size)).astype(np.float32) * 0.02
        self.lm_head = self.embed_tokens  # tied
        self.final_norm_weight = np.ones(hidden_size, dtype=np.float32)

        self.layers = []
        for _ in range(num_layers):
            layer = {}
            # Projections
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
            # Norm weights
            layer["input_layernorm_weight"] = np.ones(
                hidden_size, dtype=np.float32)
            layer["post_attention_layernorm_weight"] = np.ones(
                hidden_size, dtype=np.float32)
            self.layers.append(layer)


# ---------------------------------------------------------------
# Reference forward pass (plain NumPy — same as circuit_model.py)
# ---------------------------------------------------------------

def _np_silu(x):
    with np.errstate(over="ignore", invalid="ignore"):
        return (x / (1.0 + np.exp(-x.astype(np.float64)))).astype(np.float32)


def _np_exp(x):
    with np.errstate(over="ignore"):
        return np.exp(x.astype(np.float64)).astype(np.float32)


def _np_rsqrt(x):
    with np.errstate(invalid="ignore", divide="ignore"):
        return (1.0 / np.sqrt(x.astype(np.float64))).astype(np.float32)


def _reference_forward(fabric, token_ids, start_pos=0):
    """Plain NumPy forward pass matching circuit_model.py logic."""
    f = fabric
    hidden = f.embed_tokens[token_ids].astype(np.float32)
    seq = len(token_ids)

    # RoPE
    cos_all, sin_all = _build_rope_const(2048, f.head_dim, f.rope_theta)
    cos = cos_all[start_pos:start_pos + seq]
    sin = sin_all[start_pos:start_pos + seq]
    cos_b = cos[np.newaxis, :, :]
    sin_b = sin[np.newaxis, :, :]
    half = f.head_dim // 2
    scale = np.float32(1.0 / np.sqrt(f.head_dim))

    for li in range(f.num_layers):
        w = f.layers[li]

        # Pre-attention RMSNorm
        residual = hidden
        var = np.mean(hidden.astype(np.float64) ** 2, axis=-1, keepdims=True)
        rscale = _np_rsqrt((var + f.rms_norm_eps).astype(np.float32))
        normed = (hidden * rscale).astype(np.float32) * w["input_layernorm_weight"]

        # QKV
        q = normed @ w["q_proj"].T
        k_val = normed @ w["k_proj"].T
        v_val = normed @ w["v_proj"].T

        q = q.reshape(seq, f.num_heads, f.head_dim).transpose(1, 0, 2)
        k_val = k_val.reshape(seq, f.num_kv_heads, f.head_dim).transpose(1, 0, 2)
        v_val = v_val.reshape(seq, f.num_kv_heads, f.head_dim).transpose(1, 0, 2)

        # RoPE
        q1, q2 = q[..., :half], q[..., half:]
        q = q * cos_b + np.concatenate([-q2, q1], axis=-1) * sin_b
        k1, k2 = k_val[..., :half], k_val[..., half:]
        k_val = k_val * cos_b + np.concatenate([-k2, k1], axis=-1) * sin_b

        # GQA
        if f.num_groups > 1:
            k_exp = np.repeat(k_val, f.num_groups, axis=0)
            v_exp = np.repeat(v_val, f.num_groups, axis=0)
        else:
            k_exp, v_exp = k_val, v_val

        # Attention scores
        scores = np.matmul(q, k_exp.transpose(0, 2, 1)) * scale

        if seq > 1:
            q_pos = np.arange(start_pos, start_pos + seq)[:, None]
            k_pos = np.arange(start_pos + seq)[None, :]
            causal = np.where(
                k_pos <= q_pos, np.float32(0.0), np.float32(-np.inf))
            scores += causal[np.newaxis, :, :]

        # Softmax
        m = scores.max(axis=-1, keepdims=True)
        e = _np_exp((scores - m).astype(np.float32))
        attn_w = e / e.sum(axis=-1, keepdims=True)

        # Context
        context = np.matmul(attn_w, v_exp)
        context = context.transpose(1, 0, 2).reshape(seq, -1)
        hidden = residual + context @ w["o_proj"].T

        # Pre-MLP RMSNorm
        residual = hidden
        var = np.mean(hidden.astype(np.float64) ** 2, axis=-1, keepdims=True)
        rscale = _np_rsqrt((var + f.rms_norm_eps).astype(np.float32))
        normed = (hidden * rscale).astype(np.float32) * w["post_attention_layernorm_weight"]

        # MLP
        gate = normed @ w["gate_proj"].T
        up = normed @ w["up_proj"].T
        hidden = residual + (_np_silu(gate) * up) @ w["down_proj"].T

    # Final norm
    var = np.mean(hidden.astype(np.float64) ** 2, axis=-1, keepdims=True)
    rscale = _np_rsqrt((var + f.rms_norm_eps).astype(np.float32))
    hidden = (hidden * rscale).astype(np.float32) * f.final_norm_weight

    # Logits
    return hidden @ f.lm_head.T


# ---------------------------------------------------------------
# Tests
# ---------------------------------------------------------------

class TestCompileModel:
    """Test that compile_model produces a valid graph."""

    def test_compiles_without_error(self):
        fab = MockFabric(num_layers=1, hidden_size=16, num_heads=2,
                         num_kv_heads=2, intermediate_size=32)
        g, logits_id = compile_model(fab, token_ids=[0, 1, 2])
        assert isinstance(g, CircuitGraph)
        assert logits_id >= 0
        assert len(g) > 0

    def test_graph_has_expected_ops(self):
        fab = MockFabric(num_layers=1)
        g, _ = compile_model(fab, token_ids=[0])
        ops = {n.op for n in g.nodes}
        # Must contain at least these
        assert Op.CONST in ops
        assert Op.MATMUL in ops
        assert Op.ADD in ops
        assert Op.MUL in ops
        assert Op.LUT in ops      # silu, exp, rsqrt

    def test_single_token_graph_evaluates(self):
        fab = MockFabric(num_layers=1)
        g, logits_id = compile_model(fab, token_ids=[5])
        vals = evaluate(g)
        logits = vals[logits_id]
        assert logits.shape == (1, fab.vocab_size)
        assert np.isfinite(logits).all()

    def test_multi_token_graph_evaluates(self):
        fab = MockFabric(num_layers=1)
        tokens = [1, 2, 3, 4]
        g, logits_id = compile_model(fab, token_ids=tokens)
        vals = evaluate(g)
        logits = vals[logits_id]
        assert logits.shape == (len(tokens), fab.vocab_size)
        assert np.isfinite(logits).all()

    def test_matches_reference_single_token(self):
        """Graph evaluation must match the NumPy reference."""
        fab = MockFabric(num_layers=2)
        tokens = [10]
        g, logits_id = compile_model(fab, token_ids=tokens)
        graph_logits = evaluate(g)[logits_id]
        ref_logits = _reference_forward(fab, tokens)
        np.testing.assert_allclose(
            graph_logits, ref_logits, rtol=1e-5, atol=1e-6)

    def test_matches_reference_multi_token(self):
        """Multi-token: graph must match reference with causal mask."""
        fab = MockFabric(num_layers=2)
        tokens = [1, 5, 10]
        g, logits_id = compile_model(fab, token_ids=tokens)
        graph_logits = evaluate(g)[logits_id]
        ref_logits = _reference_forward(fab, tokens)
        np.testing.assert_allclose(
            graph_logits, ref_logits, rtol=1e-5, atol=1e-6)

    def test_gate_count_grows_with_layers(self):
        fab1 = MockFabric(num_layers=1)
        fab2 = MockFabric(num_layers=2)
        g1, _ = compile_model(fab1, token_ids=[0])
        g2, _ = compile_model(fab2, token_ids=[0])
        assert g2.gate_count()["total"] > g1.gate_count()["total"]

    def test_argmax_on_logits(self):
        """Can append argmax to get predicted token."""
        fab = MockFabric(num_layers=1)
        g, logits_id = compile_model(fab, token_ids=[0])
        # Take last row's argmax
        last_row = g.slice(logits_id, (slice(-1, None),),
                           name="last_logit")
        pred = g.argmax(last_row, axis=-1, name="pred_token")
        vals = evaluate(g)
        assert vals[pred].shape == (1,)
        assert 0 <= vals[pred][0] < fab.vocab_size

    def test_graph_repr(self):
        fab = MockFabric(num_layers=1)
        g, _ = compile_model(fab, token_ids=[0])
        r = repr(g)
        assert "nodes" in r


class TestRopeConst:
    def test_shapes(self):
        cos, sin = _build_rope_const(128, 8, 10000.0)
        assert cos.shape == (128, 8)
        assert sin.shape == (128, 8)
        assert cos.dtype == np.float32

    def test_cos_sin_range(self):
        cos, sin = _build_rope_const(64, 16, 10000.0)
        assert np.all(cos >= -1.0) and np.all(cos <= 1.0)
        assert np.all(sin >= -1.0) and np.all(sin <= 1.0)
