"""Compile a LLaMA transformer into a CircuitGraph.

Takes a loaded ``Fabric`` (model weights + config) and emits a
complete ``CircuitGraph`` where every operation — embedding lookup,
matmul, RMSNorm, RoPE, attention, MLP, softmax, logits — is a node
in the DAG.

The resulting graph can be:
1. **Evaluated** by the reference evaluator (NumPy) — bit-exact to
   ``circuit_model.py``.
2. **Executed** by the C gate executor (Phase 4).
3. **Optimised** (constant folding, dead elimination) (Phase 5).
4. **Synthesised** to Verilog/VHDL for FPGA (Phase 7).
"""

from __future__ import annotations

import numpy as np

from kllm.circuit_graph import CircuitGraph


# ---------------------------------------------------------------
# RoPE precomputation
# ---------------------------------------------------------------

def _build_rope_const(
    max_seq: int, head_dim: int, theta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute cos/sin RoPE embeddings as float32 arrays."""
    inv_freq = 1.0 / (
        theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)
    )
    t = np.arange(max_seq, dtype=np.float32)
    freqs = np.outer(t, inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


# ---------------------------------------------------------------
# Subgraph builders
# ---------------------------------------------------------------

def _apply_rotary_emb(
    g: CircuitGraph,
    x: int,          # (num_heads, seq, head_dim)
    cos: int,        # (seq, head_dim)
    sin: int,        # (seq, head_dim)
    half: int,
    name: str,
) -> int:
    """RoPE: x * cos + rotate(x) * sin.  All nodes, no NumPy math."""
    # x1 = x[..., :half], x2 = x[..., half:]
    x1 = g.slice(x, (slice(None), slice(None), slice(0, half)),
                 name=f"{name}/x1")
    x2 = g.slice(x, (slice(None), slice(None), slice(half, None)),
                 name=f"{name}/x2")
    # rotated = [-x2, x1]
    neg_x2 = g.neg(x2, name=f"{name}/neg_x2")
    rotated = g.concat([neg_x2, x1], axis=-1, name=f"{name}/rotated")
    # x * cos + rotated * sin
    xcos = g.mul(x, cos, name=f"{name}/xcos")
    rsin = g.mul(rotated, sin, name=f"{name}/rsin")
    return g.add(xcos, rsin, name=f"{name}/rope_out")


def _attention_layer(
    g: CircuitGraph,
    hidden: int,      # (seq, hidden_size)
    w: dict[str, int],  # node IDs for weight constants
    cos: int,         # (seq, head_dim)  — already sliced for this position
    sin: int,         # (seq, head_dim)
    scale_node: int,
    kv_cache_k: int | None,  # previous K or None
    kv_cache_v: int | None,  # previous V or None
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    num_groups: int,
    seq_len: int,
    start_pos: int,
    name: str,
) -> tuple[int, int, int]:
    """Build attention subgraph.  Returns (output, new_k, new_v)."""

    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim

    # Fused Q/K/V projection — one matmul replaces three
    if "qkv_t" in w:
        qkv = g.matmul(hidden, w["qkv_t"], name=f"{name}/qkv_proj")
        q = g.slice(qkv, (slice(None), slice(0, q_dim)),
                    name=f"{name}/q_split")
        k = g.slice(qkv, (slice(None), slice(q_dim, q_dim + kv_dim)),
                    name=f"{name}/k_split")
        v = g.slice(qkv, (slice(None), slice(q_dim + kv_dim, None)),
                    name=f"{name}/v_split")
    else:
        q = g.matmul(hidden, w["q_proj_t"], name=f"{name}/q_proj")
        k = g.matmul(hidden, w["k_proj_t"], name=f"{name}/k_proj")
        v = g.matmul(hidden, w["v_proj_t"], name=f"{name}/v_proj")

    # Reshape to (heads, seq, head_dim)
    q = g.reshape(q, (seq_len, num_heads, head_dim), name=f"{name}/q_reshape")
    q = g.transpose(q, (1, 0, 2), name=f"{name}/q_trans")  # (nH, seq, hd)
    k = g.reshape(k, (seq_len, num_kv_heads, head_dim),
                  name=f"{name}/k_reshape")
    k = g.transpose(k, (1, 0, 2), name=f"{name}/k_trans")
    v = g.reshape(v, (seq_len, num_kv_heads, head_dim),
                  name=f"{name}/v_reshape")
    v = g.transpose(v, (1, 0, 2), name=f"{name}/v_trans")

    # RoPE
    # cos/sin are (seq, head_dim) — need (1, seq, head_dim) for broadcast
    cos_b = g.expand_dims(cos, axis=0, name=f"{name}/cos_b")
    sin_b = g.expand_dims(sin, axis=0, name=f"{name}/sin_b")
    half = head_dim // 2
    q = _apply_rotary_emb(g, q, cos_b, sin_b, half, f"{name}/rope_q")
    k = _apply_rotary_emb(g, k, cos_b, sin_b, half, f"{name}/rope_k")

    # KV cache: concatenate with previous
    if kv_cache_k is not None:
        k = g.concat([kv_cache_k, k], axis=1, name=f"{name}/kv_cat_k")
        v = g.concat([kv_cache_v, v], axis=1, name=f"{name}/kv_cat_v")
    new_k, new_v = k, v

    # GQA: expand k/v to match q's head count
    if num_groups > 1:
        k_exp = g.repeat(k, repeats=num_groups, axis=0,
                         name=f"{name}/gqa_k")
        v_exp = g.repeat(v, repeats=num_groups, axis=0,
                         name=f"{name}/gqa_v")
    else:
        k_exp, v_exp = k, v

    # Attention scores: Q @ K^T * scale
    k_t = g.transpose(k_exp, (0, 2, 1), name=f"{name}/k_t")
    scores = g.matmul(q, k_t, name=f"{name}/qk")
    scores = g.mul(scores, scale_node, name=f"{name}/scale")

    # Causal mask (only for prefill with seq > 1)
    if seq_len > 1:
        total_seq = start_pos + seq_len if kv_cache_k is None else -1
        # Build causal mask as constant
        if kv_cache_k is not None:
            # We don't know total_seq at graph build time for dynamic cache
            # For now, build for the specific seq_len (static graph)
            pass
        q_pos = np.arange(start_pos, start_pos + seq_len)[:, None]
        k_pos_len = start_pos + seq_len  # total positions
        k_pos = np.arange(k_pos_len)[None, :]
        causal = np.where(
            k_pos <= q_pos, np.float32(0.0), np.float32(-np.inf),
        ).astype(np.float32)
        # Shape: (1, seq, total_seq) for broadcast with (nH, seq, total_seq)
        causal_node = g.const(causal[np.newaxis, :, :],
                              name=f"{name}/causal_mask")
        scores = g.add(scores, causal_node, name=f"{name}/masked")

    # Softmax
    attn_w = g.softmax(scores, axis=-1, name=f"{name}/softmax")

    # Context: attn @ V
    context = g.matmul(attn_w, v_exp, name=f"{name}/context")
    # Reshape back: (nH, seq, hd) → (seq, nH, hd) → (seq, hidden)
    context = g.transpose(context, (1, 0, 2), name=f"{name}/ctx_trans")
    context = g.reshape(context, (seq_len, num_heads * head_dim),
                        name=f"{name}/ctx_flat")

    # Output projection
    out = g.matmul(context, w["o_proj_t"], name=f"{name}/o_proj")
    return out, new_k, new_v


def _mlp_layer(
    g: CircuitGraph,
    hidden: int,
    w: dict[str, int],
    name: str,
) -> int:
    """Build MLP subgraph: gate_proj → SiLU, up_proj, gate*up → down_proj."""
    if "gate_up_t" in w:
        gate_up = g.matmul(hidden, w["gate_up_t"],
                           name=f"{name}/gate_up_proj")
        mid = w.get("_intermediate_size")
        gate = g.slice(gate_up, (slice(None), slice(0, mid)),
                       name=f"{name}/gate_split")
        up = g.slice(gate_up, (slice(None), slice(mid, None)),
                     name=f"{name}/up_split")
    else:
        gate = g.matmul(hidden, w["gate_proj_t"], name=f"{name}/gate")
        up = g.matmul(hidden, w["up_proj_t"], name=f"{name}/up")
    gate_act = g.lut(gate, "silu", name=f"{name}/silu")
    gated = g.mul(gate_act, up, name=f"{name}/gate_up")
    return g.matmul(gated, w["down_proj_t"], name=f"{name}/down")


# ---------------------------------------------------------------
# Main compiler
# ---------------------------------------------------------------

def compile_model(
    fabric: "kllm.fabric.Fabric",
    token_ids: list[int],
    start_pos: int = 0,
    max_seq: int = 2048,
) -> tuple[CircuitGraph, int]:
    """Compile a full forward pass into a CircuitGraph.

    Parameters
    ----------
    fabric : Fabric
        Loaded model weights and config.
    token_ids : list[int]
        The token sequence to build the graph for.  The graph is
        static (fixed shape) — one graph per sequence length.
    start_pos : int
        Starting position in the sequence (for KV cache offset).
    max_seq : int
        Maximum sequence length for RoPE precomputation.

    Returns
    -------
    (graph, logits_node_id)
        The complete circuit graph and the node ID of the logits output.
    """
    f = fabric
    g = CircuitGraph()
    seq_len = len(token_ids)

    # ---- Embedding lookup (constant selection) ----
    # For a fixed input, this is just a constant slice
    embed_weights = g.const(f.embed_tokens, name="embed_tokens")
    # Select rows by token IDs — this is a MUX in gate terms,
    # but for a fixed graph (known token_ids) it's a constant.
    hidden_val = f.embed_tokens[token_ids].astype(np.float32)
    hidden = g.const(hidden_val, name="embed_out")

    # ---- RoPE precompute (all constants) ----
    rope_cos_all, rope_sin_all = _build_rope_const(
        max_seq, f.head_dim, f.rope_theta)
    rope_cos_slice = rope_cos_all[start_pos:start_pos + seq_len]
    rope_sin_slice = rope_sin_all[start_pos:start_pos + seq_len]
    cos_node = g.const(rope_cos_slice, name="rope_cos")
    sin_node = g.const(rope_sin_slice, name="rope_sin")

    # ---- Scale constant ----
    scale_val = np.float32(1.0 / np.sqrt(f.head_dim))
    scale_node = g.const(scale_val, name="attn_scale")

    # ---- Per-layer constant weights → graph nodes ----
    layer_weights: list[dict[str, int]] = []
    for li in range(f.num_layers):
        lw = f.layers[li]
        wn: dict[str, int] = {}
        # Fused QKV and GateUp — one matmul replaces 2-3 separate ones
        wn["qkv_t"] = g.const(
            f.get_fused_qkv_t(li), name=f"L{li}/qkv_t")
        wn["o_proj_t"] = g.const(
            f.get_transposed(li, "o_proj"), name=f"L{li}/o_proj_t")
        wn["gate_up_t"] = g.const(
            f.get_fused_gate_up_t(li), name=f"L{li}/gate_up_t")
        wn["down_proj_t"] = g.const(
            f.get_transposed(li, "down_proj"), name=f"L{li}/down_proj_t")
        wn["_intermediate_size"] = f.intermediate_size
        wn["input_ln_w"] = g.const(
            lw["input_layernorm_weight"].astype(np.float32),
            name=f"L{li}/input_ln_w")
        wn["post_ln_w"] = g.const(
            lw["post_attention_layernorm_weight"].astype(np.float32),
            name=f"L{li}/post_ln_w")
        layer_weights.append(wn)

    # Final norm + lm_head weights
    final_ln_w = g.const(f.final_norm_weight.astype(np.float32),
                         name="final_ln_w")
    lm_head_t = g.const(f.lm_head.T.astype(np.float32),
                        name="lm_head_t")

    # ---- Layer loop ----
    kv_caches: list[tuple[int | None, int | None]] = []

    for li in range(f.num_layers):
        wn = layer_weights[li]
        prefix = f"L{li}"

        # Pre-attention RMSNorm
        residual = hidden
        hidden = g.rms_norm(hidden, wn["input_ln_w"],
                            f.rms_norm_eps, name=f"{prefix}/pre_attn_norm")

        # Attention
        kv_k = kv_caches[li][0] if li < len(kv_caches) else None
        kv_v = kv_caches[li][1] if li < len(kv_caches) else None

        attn_out, new_k, new_v = _attention_layer(
            g, hidden, wn, cos_node, sin_node, scale_node,
            kv_k, kv_v,
            f.num_heads, f.num_kv_heads, f.head_dim, f.num_groups,
            seq_len, start_pos, name=f"{prefix}/attn")

        if li < len(kv_caches):
            kv_caches[li] = (new_k, new_v)
        else:
            kv_caches.append((new_k, new_v))

        # Residual add
        hidden = g.add(residual, attn_out, name=f"{prefix}/residual1")

        # Pre-MLP RMSNorm
        residual = hidden
        hidden = g.rms_norm(hidden, wn["post_ln_w"],
                            f.rms_norm_eps, name=f"{prefix}/pre_mlp_norm")

        # MLP
        mlp_out = _mlp_layer(g, hidden, wn, name=f"{prefix}/mlp")

        # Residual add
        hidden = g.add(residual, mlp_out, name=f"{prefix}/residual2")

    # ---- Final norm ----
    hidden = g.rms_norm(hidden, final_ln_w, f.rms_norm_eps,
                        name="final_norm")

    # ---- Logit projection ----
    logits = g.matmul(hidden, lm_head_t, name="logits")

    # kv_caches is a list of (k_node_id, v_node_id) per layer —
    # callers can use these to extract KV values without searching
    # by name.
    return g, logits, kv_caches


# ---------------------------------------------------------------
# Decode template — the persistent machine
# ---------------------------------------------------------------

class DecodeMachine:
    """A pre-built single-token decode graph with INPUT nodes.

    The machine's structure (matmuls, norms, attention, MLP) is fixed.
    Weights are baked in as CONST.  Variable state (token embedding,
    RoPE, KV cache) is fed through INPUT nodes each step.

    This avoids rebuilding and re-optimizing the graph every token.
    """

    def __init__(
        self,
        graph: CircuitGraph,
        logits_id: int,
        kv_ids: list[tuple[int, int]],   # (new_k_id, new_v_id) per layer
        input_ids: dict[str, int],        # name → input node ID
    ) -> None:
        self.graph = graph
        self.logits_id = logits_id
        self.kv_ids = kv_ids
        self.input_ids = input_ids


def compile_decode_template(
    fabric: "kllm.fabric.Fabric",
    max_seq: int = 2048,
) -> DecodeMachine:
    """Build the decode machine once.  Reuse every token.

    Weights are CONST (part of the machine).  Variable state is INPUT:
      - ``token_embed``: (1, hidden_size) — embedding of the new token
      - ``rope_cos``, ``rope_sin``: (1, head_dim) — position encoding
      - ``L{i}/cache_k``, ``L{i}/cache_v``: per-layer KV cache

    Returns a ``DecodeMachine`` with the graph and all handle IDs.
    """
    f = fabric
    g = CircuitGraph()
    seq_len = 1  # always single-token decode
    input_ids: dict[str, int] = {}

    # ---- Variable inputs ----
    hidden = g.input(
        shape=(1, f.num_heads * f.head_dim),
        name="token_embed")
    input_ids["token_embed"] = hidden

    cos_node = g.input(shape=(1, f.head_dim), name="rope_cos")
    sin_node = g.input(shape=(1, f.head_dim), name="rope_sin")
    input_ids["rope_cos"] = cos_node
    input_ids["rope_sin"] = sin_node

    # Per-layer KV cache inputs (shape varies with sequence length —
    # the evaluator doesn't check shapes, just passes the array through)
    kv_cache_inputs: list[tuple[int, int]] = []
    for li in range(f.num_layers):
        k_in = g.input(
            shape=(f.num_kv_heads, 1, f.head_dim),  # placeholder shape
            name=f"L{li}/cache_k")
        v_in = g.input(
            shape=(f.num_kv_heads, 1, f.head_dim),
            name=f"L{li}/cache_v")
        input_ids[f"L{li}/cache_k"] = k_in
        input_ids[f"L{li}/cache_v"] = v_in
        kv_cache_inputs.append((k_in, v_in))

    # ---- Fixed CONST: scale ----
    scale_val = np.float32(1.0 / np.sqrt(f.head_dim))
    scale_node = g.const(scale_val, name="attn_scale")

    # ---- Fixed CONST: all layer weights ----
    layer_weights: list[dict[str, int]] = []
    for li in range(f.num_layers):
        lw = f.layers[li]
        wn: dict[str, int] = {}
        # Fused QKV and GateUp — one matmul replaces 2-3 separate ones
        wn["qkv_t"] = g.const(
            f.get_fused_qkv_t(li), name=f"L{li}/qkv_t")
        wn["o_proj_t"] = g.const(
            f.get_transposed(li, "o_proj"), name=f"L{li}/o_proj_t")
        wn["gate_up_t"] = g.const(
            f.get_fused_gate_up_t(li), name=f"L{li}/gate_up_t")
        wn["down_proj_t"] = g.const(
            f.get_transposed(li, "down_proj"), name=f"L{li}/down_proj_t")
        wn["_intermediate_size"] = f.intermediate_size
        wn["input_ln_w"] = g.const(
            lw["input_layernorm_weight"].astype(np.float32),
            name=f"L{li}/input_ln_w")
        wn["post_ln_w"] = g.const(
            lw["post_attention_layernorm_weight"].astype(np.float32),
            name=f"L{li}/post_ln_w")
        layer_weights.append(wn)

    final_ln_w = g.const(f.final_norm_weight.astype(np.float32),
                         name="final_ln_w")
    lm_head_t = g.const(f.lm_head.T.astype(np.float32),
                        name="lm_head_t")

    # ---- Layer loop (seq_len=1, no causal mask needed) ----
    kv_out_ids: list[tuple[int, int]] = []

    for li in range(f.num_layers):
        wn = layer_weights[li]
        prefix = f"L{li}"

        residual = hidden
        hidden = g.rms_norm(hidden, wn["input_ln_w"],
                            f.rms_norm_eps, name=f"{prefix}/pre_attn_norm")

        kv_k_in, kv_v_in = kv_cache_inputs[li]
        attn_out, new_k, new_v = _attention_layer(
            g, hidden, wn, cos_node, sin_node, scale_node,
            kv_k_in, kv_v_in,
            f.num_heads, f.num_kv_heads, f.head_dim, f.num_groups,
            seq_len, 0, name=f"{prefix}/attn")  # start_pos=0 doesn't matter for seq_len=1

        kv_out_ids.append((new_k, new_v))

        hidden = g.add(residual, attn_out, name=f"{prefix}/residual1")

        residual = hidden
        hidden = g.rms_norm(hidden, wn["post_ln_w"],
                            f.rms_norm_eps, name=f"{prefix}/pre_mlp_norm")
        mlp_out = _mlp_layer(g, hidden, wn, name=f"{prefix}/mlp")
        hidden = g.add(residual, mlp_out, name=f"{prefix}/residual2")

    # ---- Final norm + logits ----
    hidden = g.rms_norm(hidden, final_ln_w, f.rms_norm_eps,
                        name="final_norm")
    logits = g.matmul(hidden, lm_head_t, name="logits")

    return DecodeMachine(
        graph=g,
        logits_id=logits,
        kv_ids=kv_out_ids,
        input_ids=input_ids,
    )
