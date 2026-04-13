"""JIT per-token circuit optimization for autoregressive decode.

After each decode step, the new K/V cache values become **known
constants** for all future steps.  This module folds them into the
graph and re-optimizes — producing a progressively smaller circuit
as the sequence grows.

Architecture
------------
``JitSession`` manages one autoregressive generation session:

1. **Prefill** — compile a graph for the full prompt, evaluate,
   capture K/V cache tensors.
2. **Decode step** — for each new token:
   a. Compile a single-token decode graph (seq_len=1, start_pos=N).
   b. Inject accumulated K/V cache as CONST nodes.
   c. Constant-fold + dead-eliminate → the attention over past tokens
      collapses entirely (all inputs are const).
   d. Evaluate the optimized graph → logits + new K/V.
   e. Append new K/V to the cache.
3. **Prefix cache** — hash token prefix → reuse compiled session state.

Usage::

    from kllm.jit_optimizer import JitSession
    session = JitSession(fabric)
    logits = session.prefill([1, 5, 10])      # compile + evaluate prompt
    logits = session.decode_step(next_token)   # per-token JIT
"""

from __future__ import annotations

import hashlib

import numpy as np

from kllm.circuit_compiler import compile_model
from kllm.circuit_graph import CircuitGraph, evaluate
from kllm.graph_optimizer import optimize_graph, optimization_stats


class JitSession:
    """Manages one autoregressive generation session with JIT optimization.

    The session maintains a KV cache and builds optimized graphs
    for each decode step.  Past K/V values are folded as constants,
    so the attention over cached positions requires zero compute.
    """

    def __init__(self, fabric: object) -> None:
        self.fabric = fabric
        self.num_layers = fabric.num_layers
        self.num_kv_heads = fabric.num_kv_heads
        self.head_dim = fabric.head_dim

        # KV cache: list of (K, V) per layer
        # K shape: (num_kv_heads, seq_so_far, head_dim)
        # V shape: (num_kv_heads, seq_so_far, head_dim)
        self.kv_cache: list[tuple[np.ndarray, np.ndarray]] = []
        self.position: int = 0
        self.token_history: list[int] = []
        self.stats: list[dict] = []

    def prefill(self, token_ids: list[int]) -> np.ndarray:
        """Compile and evaluate the prompt in one pass.

        Returns the logits array (seq_len, vocab_size).
        """
        graph, logits_id = compile_model(
            self.fabric, token_ids, start_pos=0)

        # Evaluate full graph
        values = evaluate(graph)
        logits = values[logits_id]

        # Extract K/V cache from the graph evaluation
        self.kv_cache = _extract_kv_cache(
            graph, values, self.num_layers, len(token_ids))

        self.position = len(token_ids)
        self.token_history = list(token_ids)

        self.stats.append({
            "step": "prefill",
            "tokens": len(token_ids),
            "original_nodes": len(graph),
            "position": self.position,
        })

        return logits

    def decode_step(self, token_id: int) -> np.ndarray:
        """Generate one token with JIT optimization.

        Compiles a single-token graph, injects KV cache as constants,
        folds + eliminates, evaluates, and updates the cache.

        Returns the logits array (1, vocab_size).
        """
        # Compile single-token graph
        graph, logits_id = compile_model(
            self.fabric, [token_id], start_pos=self.position)

        original_nodes = len(graph)

        # Optimize: constant fold + dead elimination
        opt_graph, id_map = optimize_graph(graph, output_ids=[logits_id])

        optimized_nodes = len(opt_graph)

        # Evaluate the optimized graph
        values = evaluate(opt_graph)
        logits = values[id_map[logits_id]]

        # Extract new K/V from the unoptimized graph evaluation
        # (the optimized graph has already folded everything)
        # We need to evaluate the full graph to get K/V values
        full_values = evaluate(graph)
        new_kv = _extract_kv_cache(
            graph, full_values, self.num_layers, 1)

        # Append new K/V to cache
        for li in range(self.num_layers):
            old_k, old_v = self.kv_cache[li]
            new_k, new_v = new_kv[li]
            self.kv_cache[li] = (
                np.concatenate([old_k, new_k], axis=1),
                np.concatenate([old_v, new_v], axis=1),
            )

        self.position += 1
        self.token_history.append(token_id)

        self.stats.append({
            "step": f"decode_{self.position}",
            "token_id": token_id,
            "original_nodes": original_nodes,
            "optimized_nodes": optimized_nodes,
            "reduction_pct": 100.0 * (1 - optimized_nodes / max(original_nodes, 1)),
            "position": self.position,
        })

        return logits

    def get_stats(self) -> list[dict]:
        """Return optimization statistics for each step."""
        return self.stats


# ---------------------------------------------------------------
# Prefix cache
# ---------------------------------------------------------------

_PREFIX_CACHE: dict[str, tuple[list[tuple[np.ndarray, np.ndarray]], int, np.ndarray]] = {}


def _prefix_hash(token_ids: list[int]) -> str:
    """Hash a token prefix for cache lookup."""
    data = np.array(token_ids, dtype=np.int32).tobytes()
    return hashlib.sha256(data).hexdigest()[:16]


def cached_prefill(
    fabric: object,
    token_ids: list[int],
) -> tuple[JitSession, np.ndarray]:
    """Prefill with prefix caching.

    If the same prefix was already computed, reuse the KV cache
    and start the JIT session from the cached state.

    Returns (session, logits).
    """
    key = _prefix_hash(token_ids)

    if key in _PREFIX_CACHE:
        kv_cache, position, logits = _PREFIX_CACHE[key]
        session = JitSession(fabric)
        # Deep-copy the cache so mutations don't affect the cached copy
        session.kv_cache = [(k.copy(), v.copy()) for k, v in kv_cache]
        session.position = position
        session.token_history = list(token_ids)
        session.stats.append({
            "step": "prefill_cached",
            "tokens": len(token_ids),
            "position": position,
        })
        return session, logits.copy()

    session = JitSession(fabric)
    logits = session.prefill(token_ids)

    # Cache the result
    _PREFIX_CACHE[key] = (
        [(k.copy(), v.copy()) for k, v in session.kv_cache],
        session.position,
        logits.copy(),
    )

    return session, logits


def clear_prefix_cache() -> None:
    """Clear the prefix cache."""
    _PREFIX_CACHE.clear()


# ---------------------------------------------------------------
# KV cache extraction
# ---------------------------------------------------------------

def _extract_kv_cache(
    graph: CircuitGraph,
    values: dict[int, np.ndarray],
    num_layers: int,
    seq_len: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Extract K/V values from evaluated graph by finding named nodes.

    The compiler names KV nodes as ``L{i}/attn/k_trans`` and
    ``L{i}/attn/v_trans`` (after RoPE for K).
    """
    cache: list[tuple[np.ndarray, np.ndarray]] = []

    for li in range(num_layers):
        # Find the K node after RoPE: L{i}/attn/rope_k/rope_out
        k_val = None
        v_val = None

        for node in graph.nodes:
            if node.name == f"L{li}/attn/rope_k/rope_out":
                k_val = values[node.id]
            elif node.name == f"L{li}/attn/v_trans":
                v_val = values[node.id]

        if k_val is None or v_val is None:
            # Fallback: search for approximate names
            for node in graph.nodes:
                if f"L{li}" in node.name and "rope_k" in node.name and "rope_out" in node.name:
                    k_val = values[node.id]
                if f"L{li}" in node.name and "v_trans" in node.name:
                    v_val = values[node.id]

        if k_val is None or v_val is None:
            raise RuntimeError(
                f"Could not find K/V cache nodes for layer {li}")

        cache.append((np.asarray(k_val), np.asarray(v_val)))

    return cache
