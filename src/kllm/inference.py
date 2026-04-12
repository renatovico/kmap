"""Lossless Bit-Sliced Inference Engine.

Streams the pre-compiled logic fabric from disk one layer at a time,
reconstructs the original FP32 weights from the four IEEE-754 byte
planes, and runs a full LLaMA-style transformer forward pass:

    embed → (RMSNorm → Attention → + → RMSNorm → MLP → +) × N → Norm → LM head → logits

Peak RAM stays close to one-layer's-worth of weights because each
layer is loaded, used, and freed before the next.
"""

import os
import time

import numpy as np

from kllm.bitops import extract_sub_masks, repack_sub_masks

_NUM_PLANES = 4

# Popcount lookup table for uint8 (fallback if np.bitwise_count missing)
_POPCOUNT_LUT = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
_HAS_BITWISE_COUNT = hasattr(np, 'bitwise_count')


def _popcount_bytes(x: np.ndarray) -> np.ndarray:
    """Element-wise popcount for a uint8 array."""
    if _HAS_BITWISE_COUNT:
        return np.bitwise_count(x)
    return _POPCOUNT_LUT[x]


def _reconstruct_weight(data, name: str) -> np.ndarray:
    """Reconstruct an FP32 weight matrix from its four byte planes."""
    planes = [data[f"{name}_m{i}"] for i in range(_NUM_PLANES)]
    return repack_sub_masks(*planes)


def _gf2_matmul_plane(x_plane: np.ndarray, w_plane: np.ndarray,
                      w_packed: list | None = None) -> np.ndarray:
    """Fast GF(2) matrix multiply on a single uint8 byte plane.

    Uses **bitpacking**: each row is packed into uint8 via ``np.packbits``,
    then AND + popcount replaces the integer matmul — 8× fewer core ops.

    If *w_packed* is provided (list of 8 packed arrays, one per bit), the
    weight packing step is skipped entirely, making repeated calls with
    the same weight essentially free.

    Parameters
    ----------
    x_plane : (seq, in_dim) uint8
    w_plane : (out_dim, in_dim) uint8  — may be None when *w_packed* given
    w_packed : list[np.ndarray] | None
        Pre-packed weight bits — ``w_packed[b]`` is
        ``np.packbits((w_plane >> b) & 1, axis=-1)``  shape (out_dim, K).

    Returns
    -------
    (seq, out_dim) uint8
    """
    seq = x_plane.shape[0]
    out_dim = w_packed[0].shape[0] if w_packed else w_plane.shape[0]
    result = np.zeros((seq, out_dim), dtype=np.uint8)

    for bit in range(8):
        x_b = (x_plane >> bit) & 1                          # (seq, in)
        x_pk = np.packbits(x_b, axis=-1)                    # (seq, K)

        if w_packed is not None:
            w_pk = w_packed[bit]                              # (out, K)
        else:
            w_b = (w_plane >> bit) & 1
            w_pk = np.packbits(w_b, axis=-1)                 # (out, K)

        # AND + XOR-reduce → parity = GF(2) inner product
        # parity(a XOR b) = parity(a) XOR parity(b), so we XOR-reduce
        # K packed bytes first, then popcount just one byte per element.
        anded = x_pk[:, np.newaxis, :] & w_pk[np.newaxis, :, :]   # (seq, out, K)
        xored = np.bitwise_xor.reduce(anded, axis=-1)             # (seq, out) uint8
        parity = _popcount_bytes(xored) & 1                       # (seq, out) uint8
        result |= parity.astype(np.uint8) << bit

    return result


def _pack_weight_bits(w_plane: np.ndarray) -> list[np.ndarray]:
    """Pre-pack a weight byte-plane into 8 bitpacked arrays (one per bit)."""
    return [np.packbits((w_plane >> b) & 1, axis=-1) for b in range(8)]


def _gate_matmul(x: np.ndarray, data, name: str,
                 packed_cache: dict | None = None) -> np.ndarray:
    """GF(2) 'matmul' on IEEE-754 byte planes — the Z3 inference kernel.

    When *packed_cache* is provided, it must map
    ``f"{name}_m{p}"`` → list of 8 packed arrays (from ``_pack_weight_bits``).
    This avoids repacking weights on every call (critical for generation).

    Parameters
    ----------
    x : (seq, in_dim) float32
    data : npz dict-like — must contain ``{name}_m{p}`` for p in 0..3.
    name : str — weight name prefix.
    packed_cache : dict | None — pre-packed weight bits.

    Returns
    -------
    (seq, out_dim) float32
    """
    x_planes = extract_sub_masks(x.astype(np.float32))
    result_planes = []

    for p in range(_NUM_PLANES):
        xp = x_planes[p]                     # (seq, in_dim) uint8
        key = f"{name}_m{p}"
        if packed_cache is not None and key in packed_cache:
            wp_packed = packed_cache[key]
            result_planes.append(_gf2_matmul_plane(xp, None, w_packed=wp_packed))
        else:
            wp = data[key]                    # (out_dim, in_dim) uint8
            result_planes.append(_gf2_matmul_plane(xp, wp))

    result = repack_sub_masks(*result_planes)
    np.nan_to_num(result, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return result


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


class LosslessInferenceEngine:
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

        # ---- Load global weights ----
        print(f"[*] Loading tokenizer and global weights for {model_name} …")
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        gpath = os.path.join(save_dir, "globals.npz")
        gdata = np.load(gpath, allow_pickle=True)
        self.embed_tokens = _reconstruct_weight(gdata, "embed_tokens")
        self.final_norm_weight = gdata["final_norm_weight"]

        if bool(gdata["lm_head_tied"]):
            self.lm_head = self.embed_tokens
        else:
            self.lm_head = _reconstruct_weight(gdata, "lm_head")
        del gdata

    # ------------------------------------------------------------------
    # Attention  (GQA + RoPE, causal mask)
    # ------------------------------------------------------------------
    def _attention(
        self,
        hidden: np.ndarray,
        q_w: np.ndarray,
        k_w: np.ndarray,
        v_w: np.ndarray,
        o_w: np.ndarray,
    ) -> np.ndarray:
        seq = hidden.shape[0]

        q = hidden @ q_w.T   # (seq, num_heads * head_dim)
        k = hidden @ k_w.T   # (seq, num_kv_heads * head_dim)
        v = hidden @ v_w.T   # (seq, num_kv_heads * head_dim)

        # Reshape → (num_heads, seq, head_dim)
        q = q.reshape(seq, self.num_heads, self.head_dim).transpose(1, 0, 2)
        k = k.reshape(seq, self.num_kv_heads, self.head_dim).transpose(1, 0, 2)
        v = v.reshape(seq, self.num_kv_heads, self.head_dim).transpose(1, 0, 2)

        # RoPE
        cos, sin = _build_rope_cache(seq, self.head_dim, self.rope_theta)
        q = _apply_rotary_emb(q, cos, sin)
        k = _apply_rotary_emb(k, cos, sin)

        # GQA — expand KV heads to match Q heads
        if self.num_groups > 1:
            k = np.repeat(k, self.num_groups, axis=0)
            v = np.repeat(v, self.num_groups, axis=0)

        # Scaled dot-product attention
        scale = np.float32(1.0 / np.sqrt(self.head_dim))
        scores = np.matmul(q, k.transpose(0, 2, 1)) * scale  # (heads, seq, seq)

        # Causal mask
        causal = np.triu(
            np.full((seq, seq), -np.inf, dtype=np.float32), k=1
        )
        scores += causal

        attn_weights = _softmax(scores, axis=-1)

        # Weighted sum → output projection
        context = np.matmul(attn_weights, v)            # (heads, seq, head_dim)
        context = context.transpose(1, 0, 2).reshape(seq, -1)  # (seq, hidden)
        return context @ o_w.T

    # ------------------------------------------------------------------
    # MLP  (SiLU-gated)
    # ------------------------------------------------------------------
    @staticmethod
    def _mlp(
        hidden: np.ndarray,
        gate_w: np.ndarray,
        up_w: np.ndarray,
        down_w: np.ndarray,
    ) -> np.ndarray:
        gate = hidden @ gate_w.T
        up = hidden @ up_w.T
        return (_silu(gate) * up) @ down_w.T

    # ------------------------------------------------------------------
    # Single-layer forward (loads from disk)
    # ------------------------------------------------------------------
    def _forward_layer(self, hidden: np.ndarray, layer_idx: int) -> np.ndarray:
        path = os.path.join(self.save_dir, f"layer_{layer_idx}.npz")
        data = np.load(path)

        # --- self-attention ---
        residual = hidden
        hidden = _rms_norm(hidden, data["input_layernorm_weight"], self.rms_norm_eps)

        q_w = _reconstruct_weight(data, "q_proj")
        k_w = _reconstruct_weight(data, "k_proj")
        v_w = _reconstruct_weight(data, "v_proj")
        o_w = _reconstruct_weight(data, "o_proj")
        hidden = self._attention(hidden, q_w, k_w, v_w, o_w)
        del q_w, k_w, v_w, o_w

        hidden = residual + hidden

        # --- MLP ---
        residual = hidden
        hidden = _rms_norm(hidden, data["post_attention_layernorm_weight"], self.rms_norm_eps)

        gate_w = _reconstruct_weight(data, "gate_proj")
        up_w = _reconstruct_weight(data, "up_proj")
        down_w = _reconstruct_weight(data, "down_proj")
        hidden = self._mlp(hidden, gate_w, up_w, down_w)
        del gate_w, up_w, down_w, data

        return residual + hidden

    # ------------------------------------------------------------------
    # Full forward pass  → logits
    # ------------------------------------------------------------------
    def forward(self, token_ids: list[int]) -> np.ndarray:
        """Run the full transformer and return logits ``(seq, vocab)``."""
        hidden = self.embed_tokens[token_ids]  # (seq, hidden)

        for li in range(self.num_layers):
            hidden = self._forward_layer(hidden, li)

        hidden = _rms_norm(hidden, self.final_norm_weight, self.rms_norm_eps)
        logits = hidden @ self.lm_head.T  # (seq, vocab)
        return logits

    # ------------------------------------------------------------------
    # Text → logits convenience wrapper
    # ------------------------------------------------------------------
    def run(self, text: str) -> np.ndarray:
        """Tokenize *text*, run the model, return logits ``(seq, vocab)``."""
        print(f"[*] Tokenizing: '{text}'")
        token_ids = self.tokenizer.encode(text)

        print(f"[*] Streaming {self.num_layers} layers through logic fabric …")
        t0 = time.perf_counter()
        logits = self.forward(token_ids)
        elapsed = time.perf_counter() - t0
        print(f"[+] Inference done in {elapsed:.4f}s")
        return logits

    # ------------------------------------------------------------------
    # Greedy text generation
    # ------------------------------------------------------------------
    def generate(self, text: str, max_new_tokens: int = 50) -> str:
        """Greedy auto-regressive generation (no KV-cache yet)."""
        token_ids = self.tokenizer.encode(text)
        eos = self.tokenizer.eos_token_id

        for _ in range(max_new_tokens):
            logits = self.forward(token_ids)
            next_id = int(logits[-1].argmax())
            if next_id == eos:
                break
            token_ids.append(next_id)

        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


# ======================================================================
# Z3 Gate Inference Engine
#
# The Z3 compiler proves for every weight byte ``target``:
#
#     (0xFF << s1) ^ mask == target       (8-bit arithmetic)
#
# and stores the (s1, mask) pair.  ALL 256 byte values are solved —
# no fallback.  Inference EXECUTES the gates to recover weights:
#
#   1. Execute Z3 gate: uint8(0xFF << s1) ^ mask  →  weight byte
#   2. Assemble float:  repack_sub_masks(planes)  →  float32 weight
#   3. Navigate:         x @ w.T                  →  output activations
#
# The entire weight-recovery path is pure bit operations:
#   shift → XOR → OR → shift  (no float until the matmul)
#
# The Z3 solver has already done the hard work.  Inference just
# navigates the pre-computed gate structure.
# ======================================================================

# Probe byte used to recover target: uint8(0xFF << s1) ^ mask == target.
_Z3_PROBE = np.uint8(0xFF)


def _execute_z3_gate(s1: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Execute a Z3-solved gate to recover the original byte plane.

    The Z3 compiler proved:  ``(0xFF << s1) ^ mask == target``
    for every weight byte.  This function evaluates that gate.

    Pure bit operations: one shift, one XOR.

    The ``.astype(np.uint8)`` after the shift is critical — Z3 solves
    in 8-bit bitvector arithmetic where shifts truncate, but numpy
    widens the result.  We must match Z3's semantics.
    """
    return (_Z3_PROBE << s1).astype(np.uint8) ^ mask


def _z3_reconstruct_weight(
    gates: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> np.ndarray:
    """Reconstruct a float32 weight from its four Z3 gate pairs.

    Executes the gates (shift + XOR) to recover byte planes, then
    assembles the float via ``repack_sub_masks`` (OR + shift).
    The entire path is bit operations — no float arithmetic.
    """
    planes = tuple(_execute_z3_gate(s1, mask) for s1, mask in gates)
    return repack_sub_masks(*planes)


def _z3_linear(
    x: np.ndarray,
    gates: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> np.ndarray:
    """Linear projection where the weight is stored as Z3 gate arrays.

    The weight is reconstructed by executing the Z3 gates:

        (0xFF << s1) ^ mask  →  byte plane   (shift + XOR)
        repack_sub_masks(p0, p1, p2, p3) → float32   (OR + shift)
        x @ w.T → output                     (matmul)

    The compiler pre-solved the gates; inference navigates them.

    Parameters
    ----------
    x : (seq, in_dim) float32
    gates : tuple of 4 × (s1, mask) pairs, each (out_dim, in_dim) uint8

    Returns
    -------
    (seq, out_dim) float32
    """
    w = _z3_reconstruct_weight(gates)
    return x @ w.T


class BitLogicInferenceEngine:
    """Inference engine that navigates the Z3-solved gate fabric.

    The Z3 compiler proves, for every weight byte ``target``:

        ``(0xFF << s1) ^ mask == target``   (8-bit arithmetic)

    and stores the ``(s1, mask)`` pair.  This engine loads those
    gate arrays and **executes** them at inference time:

    1. **Execute gates** — ``uint8(0xFF << s1) ^ mask`` recovers each
       weight byte plane.  Pure shift + XOR.
    2. **Assemble float** — ``repack_sub_masks`` joins 4 byte planes
       into float32.  Pure OR + shift.
    3. **Navigate** — ``x @ w.T`` produces the layer output.

    No weight is ever stored as a float.  The Z3 gate parameters
    ``(s1, mask)`` ARE the canonical weight representation.  Inference
    is a walk through the pre-computed gate fabric.
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

        # ---- Load tokenizer ----
        print(f"[*] Loading tokenizer and Z3 gate fabric for {model_name} …")
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # ---- Load global weights as Z3 gates ----
        gpath = os.path.join(save_dir, "globals.npz")
        gdata = np.load(gpath, allow_pickle=True)

        # Embedding table: execute gates to recover float (lookup, not matmul)
        embed_gates = tuple(
            (np.array(gdata[f"embed_tokens_m{i}_s1"]),
             np.array(gdata[f"embed_tokens_m{i}_mask"]))
            for i in range(_NUM_PLANES)
        )
        self.embed_tokens = _z3_reconstruct_weight(embed_gates)

        self.final_norm_weight = np.array(gdata["final_norm_weight"])

        # LM head as Z3 gates (used for matmul at output)
        if bool(gdata["lm_head_tied"]):
            self._lm_head_gates = tuple(
                (np.array(gdata[f"embed_tokens_m{i}_s1"]),
                 np.array(gdata[f"embed_tokens_m{i}_mask"]))
                for i in range(_NUM_PLANES)
            )
        else:
            self._lm_head_gates = tuple(
                (np.array(gdata[f"lm_head_m{i}_s1"]),
                 np.array(gdata[f"lm_head_m{i}_mask"]))
                for i in range(_NUM_PLANES)
            )
        del gdata

        # ---- Cache every layer as Z3 gate arrays ----
        print(f"[*] Caching {self.num_layers} layers as Z3 gate arrays …")
        self._layers: list[dict] = []
        for li in range(self.num_layers):
            path = os.path.join(save_dir, f"layer_{li}.npz")
            raw = np.load(path)
            layer: dict = {}
            for name in self._LINEAR_NAMES:
                # Each weight → 4 gate pairs (s1, mask), one per byte plane
                layer[name] = tuple(
                    (np.array(raw[f"{name}_m{i}_s1"]),
                     np.array(raw[f"{name}_m{i}_mask"]))
                    for i in range(_NUM_PLANES)
                )
            layer["input_layernorm_weight"] = np.array(
                raw["input_layernorm_weight"]
            )
            layer["post_attention_layernorm_weight"] = np.array(
                raw["post_attention_layernorm_weight"]
            )
            self._layers.append(layer)
        print(f"[+] All layers cached ({self.num_layers} × "
              f"{len(self._LINEAR_NAMES)} Z3 gate sets).")

    # ------------------------------------------------------------------
    # Attention  (Z3 gate linear projections + float RoPE/softmax)
    # ------------------------------------------------------------------
    def _attention(self, hidden: np.ndarray, layer: dict) -> np.ndarray:
        seq = hidden.shape[0]

        q = _z3_linear(hidden, layer["q_proj"])
        k = _z3_linear(hidden, layer["k_proj"])
        v = _z3_linear(hidden, layer["v_proj"])

        q = q.reshape(seq, self.num_heads, self.head_dim).transpose(1, 0, 2)
        k = k.reshape(seq, self.num_kv_heads, self.head_dim).transpose(1, 0, 2)
        v = v.reshape(seq, self.num_kv_heads, self.head_dim).transpose(1, 0, 2)

        cos, sin = _build_rope_cache(seq, self.head_dim, self.rope_theta)
        q = _apply_rotary_emb(q, cos, sin)
        k = _apply_rotary_emb(k, cos, sin)

        if self.num_groups > 1:
            k = np.repeat(k, self.num_groups, axis=0)
            v = np.repeat(v, self.num_groups, axis=0)

        scale = np.float32(1.0 / np.sqrt(self.head_dim))
        scores = np.matmul(q, k.transpose(0, 2, 1)) * scale
        causal = np.triu(
            np.full((seq, seq), -np.inf, dtype=np.float32), k=1
        )
        scores += causal
        attn_weights = _softmax(scores, axis=-1)

        context = np.matmul(attn_weights, v)
        context = context.transpose(1, 0, 2).reshape(seq, -1)

        return _z3_linear(context, layer["o_proj"])

    # ------------------------------------------------------------------
    # MLP  (Z3 gate linear projections + float SiLU)
    # ------------------------------------------------------------------
    def _mlp(self, hidden: np.ndarray, layer: dict) -> np.ndarray:
        gate = _z3_linear(hidden, layer["gate_proj"])
        up = _z3_linear(hidden, layer["up_proj"])
        return _z3_linear(_silu(gate) * up, layer["down_proj"])

    # ------------------------------------------------------------------
    # Single-layer forward (from cache)
    # ------------------------------------------------------------------
    def _forward_layer(self, hidden: np.ndarray, layer_idx: int) -> np.ndarray:
        layer = self._layers[layer_idx]

        residual = hidden
        hidden = _rms_norm(
            hidden, layer["input_layernorm_weight"], self.rms_norm_eps
        )
        hidden = self._attention(hidden, layer)
        hidden = residual + hidden

        residual = hidden
        hidden = _rms_norm(
            hidden, layer["post_attention_layernorm_weight"], self.rms_norm_eps
        )
        hidden = self._mlp(hidden, layer)

        return residual + hidden

    # ------------------------------------------------------------------
    # Full forward pass  → logits
    # ------------------------------------------------------------------
    def forward(self, token_ids: list[int]) -> np.ndarray:
        """Run the Z3 gate transformer and return logits."""
        hidden = self.embed_tokens[token_ids]

        for li in range(self.num_layers):
            hidden = self._forward_layer(hidden, li)

        hidden = _rms_norm(hidden, self.final_norm_weight, self.rms_norm_eps)
        logits = _z3_linear(hidden, self._lm_head_gates)
        return logits

    # ------------------------------------------------------------------
    # Text → logits convenience wrapper
    # ------------------------------------------------------------------
    def run(self, text: str) -> np.ndarray:
        """Tokenize *text*, run Z3 gate inference, return logits."""
        print(f"[*] Tokenizing: '{text}'")
        token_ids = self.tokenizer.encode(text)

        print(
            f"[*] Navigating {self.num_layers} layers through "
            f"Z3 gate fabric …"
        )
        t0 = time.perf_counter()
        logits = self.forward(token_ids)
        elapsed = time.perf_counter() - t0
        print(f"[+] Z3 gate inference done in {elapsed:.4f}s")
        return logits

    # ------------------------------------------------------------------
    # Greedy text generation
    # ------------------------------------------------------------------
    def generate(self, text: str, max_new_tokens: int = 50) -> str:
        """Greedy auto-regressive generation (no KV-cache yet)."""
        token_ids = self.tokenizer.encode(text)
        eos = self.tokenizer.eos_token_id

        for i in range(max_new_tokens):
            t0 = time.perf_counter()
            logits = self.forward(token_ids)
            elapsed = time.perf_counter() - t0
            next_id = int(logits[-1].argmax())
            tok_str = self.tokenizer.decode([next_id])
            print(f"  [z3-gate] token {i+1}/{max_new_tokens}: "
                  f"{next_id} ({tok_str!r}) in {elapsed:.1f}s")
            if next_id == eos:
                break
            token_ids.append(next_id)

        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
