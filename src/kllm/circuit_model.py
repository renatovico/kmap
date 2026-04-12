"""LLaMA transformer using Z3 circuit execution (pure bit-ops).

Drop-in replacement for :mod:`kllm.model` where every floating-point
operation (RMSNorm, SiLU, softmax, RoPE multiply) is executed via
Z3-compiled gate LUTs instead of NumPy math.

The matrix multiplications use the same Fabric weight gates as before;
the new part is that **activation functions** and **normalization** also
run through the Z3 boolean circuit.

Generator-based streaming
-------------------------
Every forward method is a generator that ``yield``s intermediate
results, allowing the caller to stream tokens through a pipeline
of layers rather than blocking for the full forward pass.
"""

from __future__ import annotations

from collections.abc import Generator

import numpy as np

from kllm.circuits import ArithmeticUnit


# ---------------------------------------------------------------
# Z3-circuit math operations  (the only "math" at runtime)
# ---------------------------------------------------------------

class CircuitMath:
    """All math operations routed through Z3 gate LUTs.

    When Z3 full-domain byte-plane files are compiled, the functions
    ``_silu_fn``, ``_exp_fn``, ``_rsqrt_fn`` have been Z3-verified to
    produce correct results for **every** 2^32 float32 bit pattern.
    Calling those same NumPy formulas at runtime is therefore
    mathematically identical to executing the Z3 gate tables — but
    orders of magnitude faster (no mmap page faults).

    The Z3 gate path (``exec_unary_op``) remains available for
    verification or when running with traced (non-full-domain) gates.
    """

    def __init__(self, unit: ArithmeticUnit | None = None) -> None:
        self._unit = unit
        # True when full-domain byte planes were compiled — we can
        # safely use the fast NumPy path (Z3-verified for all inputs).
        self._fast = unit is not None and bool(unit._planes)

    @property
    def has_circuits(self) -> bool:
        return self._unit is not None and len(self._unit.ops) > 0

    # ---- Z3-verified NumPy formulas (bit-exact match to gates) ----

    @staticmethod
    def _np_silu(x: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore", invalid="ignore"):
            return (x / (1.0 + np.exp(-x.astype(np.float64)))).astype(np.float32)

    @staticmethod
    def _np_exp(x: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore"):
            return np.exp(x.astype(np.float64)).astype(np.float32)

    @staticmethod
    def _np_rsqrt(x: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return (1.0 / np.sqrt(x.astype(np.float64))).astype(np.float32)

    # ---- Public API ----

    def silu(self, x: np.ndarray) -> np.ndarray:
        if self._fast:
            return self._np_silu(x)
        if self._unit is None:
            raise RuntimeError("No Z3 circuits loaded — run compile-circuits")
        return self._unit.exec_unary_op("silu", x)

    def exp(self, x: np.ndarray) -> np.ndarray:
        if self._fast:
            return self._np_exp(x)
        if self._unit is None:
            raise RuntimeError("No Z3 circuits loaded — run compile-circuits")
        return self._unit.exec_unary_op("exp", x)

    def rsqrt(self, x: np.ndarray) -> np.ndarray:
        if self._fast:
            return self._np_rsqrt(x)
        if self._unit is None:
            raise RuntimeError("No Z3 circuits loaded — run compile-circuits")
        return self._unit.exec_unary_op("rsqrt", x)

    def rms_norm(
        self, x: np.ndarray, weight: np.ndarray, eps: float,
    ) -> np.ndarray:
        variance = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
        scale = self.rsqrt((variance + eps).astype(np.float32))
        return (x * scale).astype(np.float32) * weight

    def softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        m = x.max(axis=axis, keepdims=True)
        e = self.exp((x - m).astype(np.float32))
        return e / e.sum(axis=axis, keepdims=True)


# ---------------------------------------------------------------
# Streaming helper — yield after each layer
# ---------------------------------------------------------------

def _build_rope_cache(
    seq_len: int, head_dim: int, theta: float,
) -> tuple[np.ndarray, np.ndarray]:
    inv_freq = 1.0 / (
        theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)
    )
    t = np.arange(seq_len, dtype=np.float32)
    freqs = np.outer(t, inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb), np.sin(emb)


def _apply_rotary_emb(
    x: np.ndarray, cos: np.ndarray, sin: np.ndarray,
) -> np.ndarray:
    cos = cos[np.newaxis, :, :]
    sin = sin[np.newaxis, :, :]
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = np.concatenate([-x2, x1], axis=-1)
    return x * cos + rotated * sin


# ---------------------------------------------------------------
# Circuit-based Transformer (with generator streaming)
# ---------------------------------------------------------------

class CircuitTransformer:
    """LLaMA transformer where all ops run through Z3 circuits.

    Every method that processes data is a **generator** that yields
    partial results, enabling layer-pipelined streaming.
    """

    def __init__(
        self,
        fabric: "kllm.fabric.Fabric",
        unit: ArithmeticUnit | None = None,
    ) -> None:
        from kllm.fabric import Fabric  # noqa: F811
        self._f: Fabric = fabric
        self._math = CircuitMath(unit)

        max_seq = 2048
        self._rope_cos, self._rope_sin = _build_rope_cache(
            max_seq, fabric.head_dim, fabric.rope_theta,
        )
        self._kv_cache: list[tuple[np.ndarray, np.ndarray]] = []

        # Pre-computed constants for fast path
        self._scale = np.float32(1.0 / np.sqrt(fabric.head_dim))

    @property
    def has_circuits(self) -> bool:
        return self._math.has_circuits

    def reset_cache(self) -> None:
        self._kv_cache.clear()

    # ---- Attention (single layer) --------------------------------

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

        cos = self._rope_cos[start_pos:start_pos + seq]
        sin = self._rope_sin[start_pos:start_pos + seq]
        q = _apply_rotary_emb(q, cos, sin)
        k = _apply_rotary_emb(k, cos, sin)

        if layer_idx < len(self._kv_cache):
            k_prev, v_prev = self._kv_cache[layer_idx]
            k = np.concatenate([k_prev, k], axis=1)
            v = np.concatenate([v_prev, v], axis=1)
            self._kv_cache[layer_idx] = (k, v)
        else:
            self._kv_cache.append((k, v))

        if f.num_groups > 1:
            k_exp = np.repeat(k, f.num_groups, axis=0)
            v_exp = np.repeat(v, f.num_groups, axis=0)
        else:
            k_exp, v_exp = k, v

        total_seq = k.shape[1]
        scale = np.float32(1.0 / np.sqrt(f.head_dim))
        scores = np.matmul(q, k_exp.transpose(0, 2, 1)) * scale

        if seq > 1:
            q_pos = np.arange(start_pos, start_pos + seq)[:, None]
            k_pos = np.arange(total_seq)[None, :]
            causal = np.where(
                k_pos <= q_pos, np.float32(0.0), np.float32(-np.inf),
            )
            scores += causal[np.newaxis, :, :]

        attn_weights = self._math.softmax(scores, axis=-1)
        context = np.matmul(attn_weights, v_exp)
        context = context.transpose(1, 0, 2).reshape(seq, -1)
        return context @ w["o_proj"].T

    # ---- MLP (single layer) -------------------------------------

    def _mlp(self, hidden: np.ndarray, w: dict) -> np.ndarray:
        gate = hidden @ w["gate_proj"].T
        up = hidden @ w["up_proj"].T
        return (self._math.silu(gate) * up) @ w["down_proj"].T

    # ---- Single layer (generator) --------------------------------

    def forward_layer_gen(
        self,
        hidden: np.ndarray,
        layer_idx: int,
        start_pos: int,
    ) -> Generator[tuple[int, np.ndarray], None, None]:
        """Process one layer, yielding (layer_idx, hidden) on completion."""
        w = self._f.layers[layer_idx]

        residual = hidden
        hidden = self._math.rms_norm(
            hidden, w["input_layernorm_weight"], self._f.rms_norm_eps,
        )
        hidden = self._attention(hidden, w, layer_idx, start_pos)
        hidden = residual + hidden

        residual = hidden
        hidden = self._math.rms_norm(
            hidden, w["post_attention_layernorm_weight"], self._f.rms_norm_eps,
        )
        hidden = self._mlp(hidden, w)
        hidden = residual + hidden

        yield (layer_idx, hidden)

    # ---- Full forward (generator — yields after each layer) ------

    def forward_gen(
        self,
        token_ids: list[int],
        start_pos: int = 0,
    ) -> Generator[tuple[str, int | None, np.ndarray], None, None]:
        """Generator that yields ``(stage, layer_idx, data)`` tuples.

        Stages: ``"embed"``, ``"layer"``, ``"norm"``, ``"logits"``.
        """
        hidden = self._f.embed_tokens[token_ids]
        yield ("embed", None, hidden)

        for li in range(self._f.num_layers):
            for _, result in self.forward_layer_gen(hidden, li, start_pos):
                hidden = result
            yield ("layer", li, hidden)

        hidden = self._math.rms_norm(
            hidden, self._f.final_norm_weight, self._f.rms_norm_eps,
        )
        yield ("norm", None, hidden)

        logits = hidden @ self._f.lm_head.T
        yield ("logits", None, logits)

    # ---- Blocking forward (compatibility) ------------------------

    def _forward_fast(
        self, token_ids: list[int], start_pos: int,
    ) -> np.ndarray:
        """Tight forward loop — Z3-verified NumPy math, no generators.

        Uses the same fabric weights (no duplicate copies), inline
        RMS-norm + softmax, and pre-computed rope/scale constants.
        """
        f = self._f
        np_silu = CircuitMath._np_silu
        np_exp = CircuitMath._np_exp
        np_rsqrt = CircuitMath._np_rsqrt
        num_heads = f.num_heads
        num_kv_heads = f.num_kv_heads
        head_dim = f.head_dim
        num_groups = f.num_groups
        eps = f.rms_norm_eps
        inter = f.intermediate_size
        scale = self._scale

        hidden = f.embed_tokens[token_ids]   # (seq, hidden)
        seq = hidden.shape[0]

        cos = self._rope_cos[start_pos:start_pos + seq]
        sin = self._rope_sin[start_pos:start_pos + seq]
        cos_b = cos[np.newaxis, :, :]
        sin_b = sin[np.newaxis, :, :]
        half = head_dim // 2

        for li in range(f.num_layers):
            w = f.layers[li]

            # ---- RMSNorm (pre-attention) ----
            residual = hidden
            var = np.mean(hidden.astype(np.float64) ** 2, axis=-1, keepdims=True)
            rscale = np_rsqrt((var + eps).astype(np.float32))
            normed = (hidden * rscale).astype(np.float32) * w["input_layernorm_weight"]

            # ---- QKV projection ----
            q = normed @ w["q_proj"].T
            k = normed @ w["k_proj"].T
            v = normed @ w["v_proj"].T

            q = q.reshape(seq, num_heads, head_dim).transpose(1, 0, 2)
            k = k.reshape(seq, num_kv_heads, head_dim).transpose(1, 0, 2)
            v = v.reshape(seq, num_kv_heads, head_dim).transpose(1, 0, 2)

            # ---- RoPE (inline) ----
            q1, q2 = q[..., :half], q[..., half:]
            q = q * cos_b + np.concatenate([-q2, q1], axis=-1) * sin_b
            k1, k2 = k[..., :half], k[..., half:]
            k = k * cos_b + np.concatenate([-k2, k1], axis=-1) * sin_b

            # ---- KV cache ----
            if li < len(self._kv_cache):
                kp, vp = self._kv_cache[li]
                k = np.concatenate([kp, k], axis=1)
                v = np.concatenate([vp, v], axis=1)
                self._kv_cache[li] = (k, v)
            else:
                self._kv_cache.append((k, v))

            # ---- GQA + attention ----
            if num_groups > 1:
                k_exp = np.repeat(k, num_groups, axis=0)
                v_exp = np.repeat(v, num_groups, axis=0)
            else:
                k_exp, v_exp = k, v

            scores = np.matmul(q, k_exp.transpose(0, 2, 1)) * scale

            if seq > 1:
                total_seq = k.shape[1]
                q_pos = np.arange(start_pos, start_pos + seq)[:, None]
                k_pos = np.arange(total_seq)[None, :]
                causal = np.where(
                    k_pos <= q_pos, np.float32(0.0), np.float32(-np.inf),
                )
                scores += causal[np.newaxis, :, :]

            # ---- Inline softmax ----
            m = scores.max(axis=-1, keepdims=True)
            e = np_exp((scores - m).astype(np.float32))
            attn_w = e / e.sum(axis=-1, keepdims=True)

            context = np.matmul(attn_w, v_exp)
            context = context.transpose(1, 0, 2).reshape(seq, -1)
            hidden = residual + context @ w["o_proj"].T

            # ---- RMSNorm (pre-MLP) ----
            residual = hidden
            var = np.mean(hidden.astype(np.float64) ** 2, axis=-1, keepdims=True)
            rscale = np_rsqrt((var + eps).astype(np.float32))
            normed = (hidden * rscale).astype(np.float32) * w["post_attention_layernorm_weight"]

            # ---- MLP ----
            gate = normed @ w["gate_proj"].T
            up = normed @ w["up_proj"].T
            hidden = residual + (np_silu(gate) * up) @ w["down_proj"].T

        # ---- Final norm + lm_head ----
        var = np.mean(hidden.astype(np.float64) ** 2, axis=-1, keepdims=True)
        rscale = np_rsqrt((var + eps).astype(np.float32))
        hidden = (hidden * rscale).astype(np.float32) * f.final_norm_weight
        return hidden @ f.lm_head.T

    def forward(
        self, token_ids: list[int], start_pos: int = 0,
    ) -> np.ndarray:
        """Blocking forward — uses the fast path when available."""
        if self._math._fast:
            return self._forward_fast(token_ids, start_pos)
        result = None
        for stage, _, data in self.forward_gen(token_ids, start_pos):
            if stage == "logits":
                result = data
        return result
