"""Z3 Gate Inference Engine.

The Z3 compiler proves for every weight byte ``target``:

    (0xFF << s1) ^ mask == target       (8-bit arithmetic)

and stores the (s1, mask) pair.  ALL 256 byte values are solved —
no fallback.  At load time the gates are executed once (shift + XOR)
to recover the original float32 weights, which are cached in RAM.
Inference uses those cached weights for standard matmuls, with a
KV cache for efficient auto-regressive decoding:

    embed → (RMSNorm → Attention → + → RMSNorm → MLP → +) × N → Norm → LM head → logits
"""

import os
import time

import numpy as np

from kllm.bitops import repack_sub_masks

_NUM_PLANES = 4

def _rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """LLaMA-style RMSNorm (no bias, no mean subtraction)."""
    variance = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    normed = x / np.sqrt(variance + eps).astype(np.float32)
    return normed * weight


def _silu(x: np.ndarray) -> np.ndarray:
    """SiLU (Swish) activation: x · σ(x)."""
    return x * (1.0 / (1.0 + np.exp(-x.astype(np.float64)))).astype(np.float32)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def _build_rope_cache(
    seq_len: int, head_dim: int, theta: float
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute (cos, sin) tables for rotary position embeddings.

    Returns two arrays of shape ``(seq_len, head_dim)``.
    """
    inv_freq = 1.0 / (
        theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)
    )
    t = np.arange(seq_len, dtype=np.float32)
    freqs = np.outer(t, inv_freq)       # (seq, head_dim/2)
    emb = np.concatenate([freqs, freqs], axis=-1)  # (seq, head_dim)
    return np.cos(emb), np.sin(emb)


def _apply_rotary_emb(
    x: np.ndarray, cos: np.ndarray, sin: np.ndarray
) -> np.ndarray:
    """Apply RoPE to *x* of shape ``(num_heads, seq, head_dim)``.

    ``cos`` / ``sin`` are ``(seq, head_dim)`` and get broadcast over heads.
    Uses the *rotate-half* convention matching HuggingFace LLaMA.
    """
    cos = cos[np.newaxis, :, :]   # (1, seq, head_dim)
    sin = sin[np.newaxis, :, :]
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = np.concatenate([-x2, x1], axis=-1)
    return x * cos + rotated * sin


# Probe byte used to recover target: uint8(0xFF << s1) ^ mask == target.
_Z3_PROBE = np.uint8(0xFF)


def _z3_reconstruct_weight(
    gates: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> np.ndarray:
    """Fused Z3 gate execution + zero-copy float assembly.

    Executes all four gate pairs and writes the recovered bytes directly
    into a contiguous ``(…, 4)`` uint8 buffer, then reinterprets as
    float32 via ``view`` — no shift/OR arithmetic for the repack step.

    Pure bit operations: shift → XOR → view.
    """
    shape = gates[0][0].shape
    buf = np.empty(shape + (4,), dtype=np.uint8)
    for i, (s1, mask) in enumerate(gates):
        buf[..., i] = (_Z3_PROBE << s1).astype(np.uint8) ^ mask
    return buf.view(np.float32).reshape(shape)


class BitLogicInferenceEngine:
    """Inference engine backed by the Z3-solved gate fabric.

    **Load-time** — Z3 gates are executed (shift + XOR) to recover the
    original float32 weights.  The gate execution is the proof step:
    every weight byte is reconstructed via pure bit operations.  The
    fused ``_z3_reconstruct_weight`` path writes recovered bytes
    into a contiguous buffer and reinterprets as float32 with ``view``
    (no shift/OR arithmetic for repack).

    After reconstruction the float32 weights are cached in RAM and the
    raw gate arrays are discarded — cutting memory in half (4 B vs 8 B
    per element).

    **Inference-time** — cached float32 weights drive standard matmuls.
    A KV cache stores K/V projections from previous positions so that
    auto-regressive decoding only forwards the *new* token through
    each layer (O(1) per token instead of O(seq)).

    The canonical on-disk representation remains the Z3 gate pairs.
    """

    _LINEAR_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj")

    def __init__(self, model_name: str, save_dir: str = "./lossless_logic"):
        self.save_dir = save_dir

        # ---- Load metadata ----
        meta_path = os.path.join(save_dir, "meta.npz")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"No compiled fabric found at {save_dir}. "
                "Run `kllm --mode compile` first."
            )
        meta = np.load(meta_path, allow_pickle=True)
        self.num_layers = int(meta["num_layers"])
        self.hidden_size = int(meta["hidden_size"])
        self.num_heads = int(meta["num_attention_heads"])
        self.num_kv_heads = int(meta["num_key_value_heads"])
        self.intermediate_size = int(meta["intermediate_size"])
        self.vocab_size = int(meta["vocab_size"])
        self.rms_norm_eps = float(meta["rms_norm_eps"])
        self.rope_theta = float(meta["rope_theta"])
        self.head_dim = self.hidden_size // self.num_heads
        self.num_groups = self.num_heads // self.num_kv_heads

        # ---- Tokenizer (prefer local copy saved at compile time) ----
        from transformers import AutoTokenizer

        tok_dir = os.path.join(save_dir, "tokenizer")
        if os.path.isdir(tok_dir):
            self.tokenizer = AutoTokenizer.from_pretrained(tok_dir)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # ---- Pre-compute RoPE tables (up to max seq) ----
        max_seq = 2048
        self._rope_cos, self._rope_sin = _build_rope_cache(
            max_seq, self.head_dim, self.rope_theta
        )

        # ---- Execute Z3 gates → float32 (pure bitops, one-time) --------
        print(f"[*] Loading Z3 gate fabric for {model_name} …")
        t0 = time.perf_counter()

        gdata = np.load(
            os.path.join(save_dir, "globals.npz"), allow_pickle=True
        )
        self.embed_tokens = _z3_reconstruct_weight(tuple(
            (np.array(gdata[f"embed_tokens_m{i}_s1"]),
             np.array(gdata[f"embed_tokens_m{i}_mask"]))
            for i in range(_NUM_PLANES)
        ))
        self.final_norm_weight = np.array(gdata["final_norm_weight"])

        if bool(gdata["lm_head_tied"]):
            self._lm_head = self.embed_tokens
        else:
            self._lm_head = _z3_reconstruct_weight(tuple(
                (np.array(gdata[f"lm_head_m{i}_s1"]),
                 np.array(gdata[f"lm_head_m{i}_mask"]))
                for i in range(_NUM_PLANES)
            ))
        del gdata

        # Per-layer: execute gates → cache float32 → discard gates
        self._weights: list[dict] = []
        for li in range(self.num_layers):
            path = os.path.join(save_dir, f"layer_{li}.npz")
            raw = np.load(path)
            layer: dict = {}
            for name in self._LINEAR_NAMES:
                gates = tuple(
                    (np.array(raw[f"{name}_m{i}_s1"]),
                     np.array(raw[f"{name}_m{i}_mask"]))
                    for i in range(_NUM_PLANES)
                )
                layer[name] = _z3_reconstruct_weight(gates)
            layer["input_layernorm_weight"] = np.array(
                raw["input_layernorm_weight"]
            )
            layer["post_attention_layernorm_weight"] = np.array(
                raw["post_attention_layernorm_weight"]
            )
            self._weights.append(layer)

        elapsed = time.perf_counter() - t0
        print(f"[+] Weights reconstructed from Z3 gates in {elapsed:.1f}s "
              f"({self.num_layers} layers, {len(self._LINEAR_NAMES)} "
              f"projections each)")

        # ---- KV cache (populated during forward) ----
        self._kv_cache: list[tuple[np.ndarray, np.ndarray]] = []

    # ------------------------------------------------------------------
    def reset_cache(self) -> None:
        """Clear the KV cache (call before a new sequence)."""
        self._kv_cache.clear()

    # ------------------------------------------------------------------
    # Attention  (cached weights + KV cache)
    # ------------------------------------------------------------------
    def _attention(
        self, hidden: np.ndarray, w: dict,
        layer_idx: int, start_pos: int,
    ) -> np.ndarray:
        seq = hidden.shape[0]

        q = hidden @ w["q_proj"].T
        k = hidden @ w["k_proj"].T
        v = hidden @ w["v_proj"].T

        q = q.reshape(seq, self.num_heads, self.head_dim).transpose(1, 0, 2)
        k = k.reshape(seq, self.num_kv_heads, self.head_dim).transpose(1, 0, 2)
        v = v.reshape(seq, self.num_kv_heads, self.head_dim).transpose(1, 0, 2)

        # RoPE — slice from pre-computed tables
        cos = self._rope_cos[start_pos:start_pos + seq]
        sin = self._rope_sin[start_pos:start_pos + seq]
        q = _apply_rotary_emb(q, cos, sin)
        k = _apply_rotary_emb(k, cos, sin)

        # KV cache: append new K/V and use full history
        if layer_idx < len(self._kv_cache):
            k_prev, v_prev = self._kv_cache[layer_idx]
            k = np.concatenate([k_prev, k], axis=1)
            v = np.concatenate([v_prev, v], axis=1)
            self._kv_cache[layer_idx] = (k, v)
        else:
            self._kv_cache.append((k, v))

        # GQA — expand KV heads to match Q heads
        if self.num_groups > 1:
            k_exp = np.repeat(k, self.num_groups, axis=0)
            v_exp = np.repeat(v, self.num_groups, axis=0)
        else:
            k_exp, v_exp = k, v

        total_seq = k.shape[1]
        scale = np.float32(1.0 / np.sqrt(self.head_dim))
        scores = np.matmul(q, k_exp.transpose(0, 2, 1)) * scale

        # Causal mask (only needed when processing >1 token)
        if seq > 1:
            q_pos = np.arange(start_pos, start_pos + seq)[:, None]
            k_pos = np.arange(total_seq)[None, :]
            causal = np.where(
                k_pos <= q_pos, np.float32(0.0), np.float32(-np.inf),
            )
            scores += causal[np.newaxis, :, :]

        attn_weights = _softmax(scores, axis=-1)

        context = np.matmul(attn_weights, v_exp)
        context = context.transpose(1, 0, 2).reshape(seq, -1)

        return context @ w["o_proj"].T

    # ------------------------------------------------------------------
    # MLP  (cached weights + float SiLU)
    # ------------------------------------------------------------------
    def _mlp(self, hidden: np.ndarray, w: dict) -> np.ndarray:
        gate = hidden @ w["gate_proj"].T
        up = hidden @ w["up_proj"].T
        return (_silu(gate) * up) @ w["down_proj"].T

    # ------------------------------------------------------------------
    # Single-layer forward
    # ------------------------------------------------------------------
    def _forward_layer(
        self, hidden: np.ndarray, layer_idx: int, start_pos: int,
    ) -> np.ndarray:
        w = self._weights[layer_idx]

        residual = hidden
        hidden = _rms_norm(
            hidden, w["input_layernorm_weight"], self.rms_norm_eps
        )
        hidden = self._attention(hidden, w, layer_idx, start_pos)
        hidden = residual + hidden

        residual = hidden
        hidden = _rms_norm(
            hidden, w["post_attention_layernorm_weight"], self.rms_norm_eps
        )
        hidden = self._mlp(hidden, w)

        return residual + hidden

    # ------------------------------------------------------------------
    # Full forward pass  → logits
    # ------------------------------------------------------------------
    def forward(
        self, token_ids: list[int], start_pos: int = 0,
    ) -> np.ndarray:
        """Run the transformer and return logits.

        When *start_pos* > 0 the KV cache from previous positions is
        reused — only the new token(s) are processed through each layer.
        Call :meth:`reset_cache` before starting a new sequence.
        """
        hidden = self.embed_tokens[token_ids]

        for li in range(self.num_layers):
            hidden = self._forward_layer(hidden, li, start_pos)

        hidden = _rms_norm(hidden, self.final_norm_weight, self.rms_norm_eps)
        return hidden @ self._lm_head.T

    # ------------------------------------------------------------------
    # Text → logits convenience wrapper
    # ------------------------------------------------------------------
    def run(self, text: str) -> np.ndarray:
        """Tokenize *text*, run inference, return logits."""
        token_ids = self.tokenizer.encode(text)
        self.reset_cache()
        t0 = time.perf_counter()
        logits = self.forward(token_ids)
        elapsed = time.perf_counter() - t0
        print(f"[+] Inference done in {elapsed:.2f}s")
        return logits

    # ------------------------------------------------------------------
    # Greedy text generation (KV-cached)
    # ------------------------------------------------------------------
    def generate(self, text: str, max_new_tokens: int = 50) -> str:
        """Greedy auto-regressive generation with KV cache."""
        token_ids = self.tokenizer.encode(text)
        eos = self.tokenizer.eos_token_id

        self.reset_cache()

        # Prefill — process full prompt, populate KV cache
        t0 = time.perf_counter()
        logits = self.forward(token_ids, start_pos=0)
        elapsed = time.perf_counter() - t0
        next_id = int(logits[-1].argmax())
        tok_str = self.tokenizer.decode([next_id])
        print(f"  [prefill] {len(token_ids)} tok → "
              f"{next_id} ({tok_str!r}) in {elapsed:.2f}s")

        if next_id == eos:
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        token_ids.append(next_id)

        # Decode — one token at a time, reusing KV cache
        for i in range(1, max_new_tokens):
            t0 = time.perf_counter()
            pos = len(token_ids) - 1
            logits = self.forward([token_ids[-1]], start_pos=pos)
            elapsed = time.perf_counter() - t0
            next_id = int(logits[-1].argmax())
            tok_str = self.tokenizer.decode([next_id])
            print(f"  [decode] token {i+1}/{max_new_tokens}: "
                  f"{next_id} ({tok_str!r}) in {elapsed:.2f}s")
            if next_id == eos:
                break
            token_ids.append(next_id)

        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
