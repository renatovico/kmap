"""LLaMA-style transformer building blocks (pure NumPy).

Stateless functions and a thin ``Transformer`` class that holds the
KV cache and RoPE tables.  All weights come from a :class:`~kllm.fabric.Fabric`.
"""

import numpy as np


# ------------------------------------------------------------------
# Stateless ops
# ------------------------------------------------------------------

def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """LLaMA-style RMSNorm (no bias, no mean subtraction)."""
    variance = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    normed = x / np.sqrt(variance + eps).astype(np.float32)
    return normed * weight


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU (Swish) activation: x · σ(x)."""
    return x * (1.0 / (1.0 + np.exp(-x.astype(np.float64)))).astype(np.float32)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def build_rope_cache(
    seq_len: int, head_dim: int, theta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute ``(cos, sin)`` tables for rotary position embeddings.

    Returns two arrays of shape ``(seq_len, head_dim)``.
    """
    inv_freq = 1.0 / (
        theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)
    )
    t = np.arange(seq_len, dtype=np.float32)
    freqs = np.outer(t, inv_freq)                      # (seq, head_dim/2)
    emb = np.concatenate([freqs, freqs], axis=-1)       # (seq, head_dim)
    return np.cos(emb), np.sin(emb)


def apply_rotary_emb(
    x: np.ndarray, cos: np.ndarray, sin: np.ndarray,
) -> np.ndarray:
    """Apply RoPE to *x* ``(num_heads, seq, head_dim)``.

    Uses the *rotate-half* convention matching HuggingFace LLaMA.
    """
    cos = cos[np.newaxis, :, :]       # (1, seq, head_dim)
    sin = sin[np.newaxis, :, :]
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = np.concatenate([-x2, x1], axis=-1)
    return x * cos + rotated * sin


# ------------------------------------------------------------------
# Transformer with KV cache
# ------------------------------------------------------------------

class Transformer:
    """LLaMA-style transformer with KV cache (pure NumPy).

    Holds pre-computed RoPE tables and a per-layer KV cache.
    All weight data is provided by a :class:`~kllm.fabric.Fabric`.
    """

    def __init__(self, fabric: "kllm.fabric.Fabric") -> None:
        from kllm.fabric import Fabric  # noqa: F811 — for type checking

        self._f: Fabric = fabric

        # Pre-compute RoPE tables
        max_seq = 2048
        self._rope_cos, self._rope_sin = build_rope_cache(
            max_seq, fabric.head_dim, fabric.rope_theta,
        )

        # KV cache — populated during forward
        self._kv_cache: list[tuple[np.ndarray, np.ndarray]] = []

    # ------------------------------------------------------------------
    def reset_cache(self) -> None:
        """Clear the KV cache (call before a new sequence)."""
        self._kv_cache.clear()

    # ------------------------------------------------------------------
    # Attention
    # ------------------------------------------------------------------
    def _attention(
        self, hidden: np.ndarray, w: dict,
        layer_idx: int, start_pos: int,
    ) -> np.ndarray:
        f = self._f
        seq = hidden.shape[0]

        q = hidden @ w["q_proj"].T
        k = hidden @ w["k_proj"].T
        v = hidden @ w["v_proj"].T

        q = q.reshape(seq, f.num_heads, f.head_dim).transpose(1, 0, 2)
        k = k.reshape(seq, f.num_kv_heads, f.head_dim).transpose(1, 0, 2)
        v = v.reshape(seq, f.num_kv_heads, f.head_dim).transpose(1, 0, 2)

        # RoPE
        cos = self._rope_cos[start_pos:start_pos + seq]
        sin = self._rope_sin[start_pos:start_pos + seq]
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # KV cache
        if layer_idx < len(self._kv_cache):
            k_prev, v_prev = self._kv_cache[layer_idx]
            k = np.concatenate([k_prev, k], axis=1)
            v = np.concatenate([v_prev, v], axis=1)
            self._kv_cache[layer_idx] = (k, v)
        else:
            self._kv_cache.append((k, v))

        # GQA
        if f.num_groups > 1:
            k_exp = np.repeat(k, f.num_groups, axis=0)
            v_exp = np.repeat(v, f.num_groups, axis=0)
        else:
            k_exp, v_exp = k, v

        total_seq = k.shape[1]
        scale = np.float32(1.0 / np.sqrt(f.head_dim))
        scores = np.matmul(q, k_exp.transpose(0, 2, 1)) * scale

        # Causal mask
        if seq > 1:
            q_pos = np.arange(start_pos, start_pos + seq)[:, None]
            k_pos = np.arange(total_seq)[None, :]
            causal = np.where(
                k_pos <= q_pos, np.float32(0.0), np.float32(-np.inf),
            )
            scores += causal[np.newaxis, :, :]

        attn_weights = softmax(scores, axis=-1)
        context = np.matmul(attn_weights, v_exp)
        context = context.transpose(1, 0, 2).reshape(seq, -1)

        return context @ w["o_proj"].T

    # ------------------------------------------------------------------
    # MLP
    # ------------------------------------------------------------------
    @staticmethod
    def _mlp(hidden: np.ndarray, w: dict) -> np.ndarray:
        gate = hidden @ w["gate_proj"].T
        up = hidden @ w["up_proj"].T
        return (silu(gate) * up) @ w["down_proj"].T

    # ------------------------------------------------------------------
    # Single layer
    # ------------------------------------------------------------------
    def _forward_layer(
        self, hidden: np.ndarray, layer_idx: int, start_pos: int,
    ) -> np.ndarray:
        w = self._f.layers[layer_idx]

        residual = hidden
        hidden = rms_norm(
            hidden, w["input_layernorm_weight"], self._f.rms_norm_eps,
        )
        hidden = self._attention(hidden, w, layer_idx, start_pos)
        hidden = residual + hidden

        residual = hidden
        hidden = rms_norm(
            hidden, w["post_attention_layernorm_weight"], self._f.rms_norm_eps,
        )
        hidden = self._mlp(hidden, w)

        return residual + hidden

    # ------------------------------------------------------------------
    # Full forward → logits
    # ------------------------------------------------------------------
    def forward(
        self, token_ids: list[int], start_pos: int = 0,
    ) -> np.ndarray:
        """Run the transformer and return logits ``(seq, vocab)``."""
        hidden = self._f.embed_tokens[token_ids]

        for li in range(self._f.num_layers):
            hidden = self._forward_layer(hidden, li, start_pos)

        hidden = rms_norm(
            hidden, self._f.final_norm_weight, self._f.rms_norm_eps,
        )
        return hidden @ self._f.lm_head.T
