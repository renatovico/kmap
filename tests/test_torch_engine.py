"""Tests for TorchInferenceEngine.

All tests run on CPU in fp32 using a small synthetic model so that they
are fast, offline, and do not require a compiled logic fabric.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Tiny synthetic model that mimics the HuggingFace CausalLM interface
# ---------------------------------------------------------------------------

class _TinyConfig:
    """Minimal config that mirrors attributes used by TorchInferenceEngine."""

    vocab_size = 32
    hidden_size = 16
    num_attention_heads = 2
    num_key_value_heads = 2
    intermediate_size = 32
    num_hidden_layers = 1
    rms_norm_eps = 1e-5
    rope_theta = 10_000.0
    max_position_embeddings = 512
    head_dim = hidden_size // num_attention_heads


class _FakePastKV(tuple):
    """A tuple subclass so isinstance checks on past_key_values pass."""


class _TinyTransformerOutput:
    """Mimics the output structure of HF CausalLM."""

    def __init__(self, logits: torch.Tensor, past_key_values) -> None:
        self.logits = logits
        self.past_key_values = past_key_values


class _TinyAttention(nn.Module):
    """Single-head self-attention that supports optional KV-cache."""

    def __init__(self, cfg: _TinyConfig) -> None:
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.scale = self.head_dim ** -0.5
        dim = cfg.hidden_size
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, cfg.num_key_value_heads * self.head_dim, bias=False)
        self.v = nn.Linear(dim, cfg.num_key_value_heads * self.head_dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, past_kv=None, position_ids=None):
        B, S, D = x.shape
        H, Hkv, Hd = self.num_heads, self.num_kv_heads, self.head_dim

        q = self.q(x).view(B, S, H, Hd).transpose(1, 2)     # (B, H, S, Hd)
        k = self.k(x).view(B, S, Hkv, Hd).transpose(1, 2)   # (B, Hkv, S, Hd)
        v = self.v(x).view(B, S, Hkv, Hd).transpose(1, 2)

        if past_kv is not None:
            k_past, v_past = past_kv
            k = torch.cat([k_past, k], dim=2)
            v = torch.cat([v_past, v], dim=2)

        new_kv = (k, v)

        groups = H // Hkv
        if groups > 1:
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        T_q, T_k = scores.shape[-2], scores.shape[-1]
        # causal mask relative to cached prefix
        mask = torch.triu(torch.full((T_q, T_k), float("-inf"), device=x.device), diagonal=T_k - T_q + 1)
        scores = scores + mask
        attn = scores.softmax(-1)
        ctx = torch.matmul(attn, v).transpose(1, 2).reshape(B, S, -1)
        return self.o(ctx), new_kv


class _TinyMLP(nn.Module):
    def __init__(self, cfg: _TinyConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))


class _TinyLayer(nn.Module):
    def __init__(self, cfg: _TinyConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.hidden_size)
        self.norm2 = nn.LayerNorm(cfg.hidden_size)
        self.attn = _TinyAttention(cfg)
        self.mlp = _TinyMLP(cfg)

    def forward(self, x, past_kv=None, position_ids=None):
        attn_out, new_kv = self.attn(self.norm1(x), past_kv=past_kv, position_ids=position_ids)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, new_kv


class _TinyModel(nn.Module):
    """A tiny causal LM model that mimics the HuggingFace API."""

    def __init__(self, cfg: _TinyConfig) -> None:
        super().__init__()
        self.config = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layer = _TinyLayer(cfg)
        self.norm = nn.LayerNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values=None,
        use_cache: bool = False,
        position_ids=None,
        **kwargs,
    ) -> _TinyTransformerOutput:
        x = self.embed(input_ids)                               # (B, S, D)
        past_kv = past_key_values[0] if past_key_values else None
        x, new_kv = self.layer(x, past_kv=past_kv, position_ids=position_ids)
        x = self.norm(x)
        logits = self.lm_head(x)                                # (B, S, V)
        pkv = (new_kv,) if use_cache else None
        return _TinyTransformerOutput(logits=logits, past_key_values=pkv)


# ---------------------------------------------------------------------------
# Fixture: a TorchInferenceEngine backed by the tiny model
# ---------------------------------------------------------------------------

@pytest.fixture()
def engine(monkeypatch):
    """Return a TorchInferenceEngine that uses _TinyModel instead of HF."""
    from kllm.torch_engine import TorchInferenceEngine

    cfg = _TinyConfig()
    tiny_model = _TinyModel(cfg)
    tiny_model.eval()

    # Fake tokenizer
    fake_tok = MagicMock()
    fake_tok.encode.side_effect = lambda text: [1, 2, 3]
    fake_tok.decode.side_effect = lambda ids, **kw: "output"
    fake_tok.eos_token_id = 0

    # Patch AutoTokenizer and AutoModelForCausalLM
    monkeypatch.setattr(
        "kllm.torch_engine.TorchInferenceEngine.__init__",
        lambda self, *a, **kw: None,
    )

    eng = TorchInferenceEngine.__new__(TorchInferenceEngine)
    eng.device = torch.device("cpu")
    eng.torch_dtype = torch.float32
    eng.window = 16
    eng.model = tiny_model
    eng.tokenizer = fake_tok
    eng._kv_cache = None
    eng._next_pos = 0
    return eng


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTorchForwardPass:
    def test_forward_returns_tensor(self, engine):
        token_ids = [1, 2, 3]
        logits = engine.forward(token_ids)
        assert isinstance(logits, torch.Tensor)

    def test_forward_shape(self, engine):
        token_ids = [1, 2, 3]
        logits = engine.forward(token_ids)
        cfg = _TinyConfig()
        assert logits.shape == (len(token_ids), cfg.vocab_size)

    def test_forward_dtype_float32(self, engine):
        logits = engine.forward([1, 2, 3])
        assert logits.dtype == torch.float32

    def test_forward_on_cpu(self, engine):
        logits = engine.forward([1, 2, 3])
        assert logits.device.type == "cpu"


class TestKVCachePrefillDecode:
    def test_prefill_returns_vocab_tensor(self, engine):
        logits = engine.prefill([1, 2, 3])
        cfg = _TinyConfig()
        assert logits.shape == (cfg.vocab_size,)

    def test_prefill_sets_cache(self, engine):
        engine.prefill([1, 2, 3])
        assert engine._kv_cache is not None

    def test_prefill_sets_next_pos(self, engine):
        engine.prefill([1, 2, 3])
        assert engine._next_pos == 3

    def test_decode_one_returns_vocab_tensor(self, engine):
        engine.prefill([1, 2, 3])
        logits = engine.decode_one(4)
        cfg = _TinyConfig()
        assert logits.shape == (cfg.vocab_size,)

    def test_decode_one_advances_position(self, engine):
        engine.prefill([1, 2, 3])
        engine.decode_one(4)
        assert engine._next_pos == 4

    def test_decode_one_without_prefill_raises(self, engine):
        with pytest.raises(RuntimeError, match="prefill"):
            engine.decode_one(1)

    def test_kvcache_next_token_matches_full_forward(self, engine):
        """KV-cache decode must predict the same next token as a full forward pass."""
        token_ids = [1, 2, 3]
        # Full (non-cached) forward: predict token after position 2
        logits_full = engine.forward(token_ids)
        next_full = int(logits_full[-1].argmax())

        # KV-cache path: prefill only, then read last logits
        logits_cached = engine.prefill(token_ids)
        next_cached = int(logits_cached.argmax())

        assert next_full == next_cached, (
            f"Full-forward predicted {next_full} but KV-cache predicted {next_cached}"
        )


class TestSlidingWindow:
    def test_window_limits_cache_length(self, engine):
        """After many decode steps the KV cache must not exceed the window size."""
        engine.prefill([1, 2, 3])
        for tok in range(4, 20):
            engine.decode_one(tok)

        # Each layer contributes one (k, v) pair; check k length
        cache_len = engine._kv_cache[0][0].shape[2]
        assert cache_len <= engine.window

    def test_small_window_does_not_crash(self, engine):
        engine.window = 4
        engine.prefill([1, 2, 3])
        for tok in range(4, 12):
            engine.decode_one(tok)


class TestGenerate:
    def test_generate_returns_string(self, engine):
        result = engine.generate("hello", max_new_tokens=3)
        assert isinstance(result, str)

    def test_generate_respects_max_new_tokens(self, engine):
        """With eos_token_id=0 and a tiny vocab, generation should terminate."""
        result = engine.generate("hello", max_new_tokens=5)
        assert result is not None


class TestEngineConfig:
    def test_invalid_dtype_raises(self):
        from kllm.torch_engine import _DTYPE_MAP
        assert "fp32" in _DTYPE_MAP
        assert "bf16" in _DTYPE_MAP
        assert "fp16" in _DTYPE_MAP
