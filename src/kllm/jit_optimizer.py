"""Graph-based inference — the model IS the machine.

The circuit graph represents the transformer as a machine:
- Weights are CONST nodes — fixed parts of the machine.
- Variable state (token embedding, RoPE, KV cache) flows through
  INPUT nodes.
- The machine's structure (matmuls, norms, attention, MLP) is built
  once and reused every token — no recompilation.

Architecture
------------
``JitSession`` manages one autoregressive generation session:

1. **Build** — ``compile_decode_template()`` builds the machine once.
   All weights become CONST nodes.  Variable state becomes INPUT nodes.
2. **Prefill** — compile a static all-CONST graph for the full prompt,
   evaluate it, capture K/V cache arrays.
3. **Decode step** — for each new token:
   a. Compute the token embedding + RoPE for this position.
   b. Feed token_embed + rope + KV cache as INPUT values into the machine.
   c. Evaluate the machine → logits + updated K/V.
   d. Append new K/V to the cache.

No recompilation, no re-optimization per token — the machine runs.

Usage::

    from kllm.jit_optimizer import JitSession
    session = JitSession(fabric)
    logits = session.prefill([1, 5, 10])
    logits = session.decode_step(next_token)
"""

from __future__ import annotations

import hashlib

import numpy as np

from kllm.circuit_compiler import (
    DecodeMachine,
    compile_decode_template,
    compile_model,
    _build_rope_const,
)
from kllm.circuit_executor import evaluate_c, precompute_consts
from kllm.circuit_graph import CircuitGraph


class JitSession:
    """One autoregressive session.  The machine runs, not rebuilds.

    On init, builds the decode machine once (``compile_decode_template``).
    Each ``decode_step`` feeds new inputs and evaluates — zero
    compilation overhead per token.
    """

    def __init__(self, fabric: object) -> None:
        self.fabric = fabric
        self.num_layers = fabric.num_layers
        self.num_kv_heads = fabric.num_kv_heads
        self.head_dim = fabric.head_dim

        # Build the decode machine once — weights baked in as CONST,
        # variable state (token, position, KV) as INPUT nodes.
        self._machine: DecodeMachine = compile_decode_template(fabric)

        # Pre-cache all CONST values so evaluate_c skips them each step.
        self._const_cache = precompute_consts(self._machine.graph)

        # Precomputed RoPE table (all positions up to max_seq)
        self._rope_cos, self._rope_sin = _build_rope_const(
            2048, fabric.head_dim, fabric.rope_theta)

        # KV cache: list of (K, V) per layer
        # K shape: (num_kv_heads, seq_so_far, head_dim)
        self.kv_cache: list[tuple[np.ndarray, np.ndarray]] = []
        self.position: int = 0
        self.token_history: list[int] = []
        self.stats: list[dict] = []

    def prefill(self, token_ids: list[int]) -> np.ndarray:
        """Compile and evaluate the prompt in one pass.

        Returns the logits array (seq_len, vocab_size).
        """
        graph, logits_id, kv_node_ids = compile_model(
            self.fabric, token_ids, start_pos=0)

        # Evaluate full graph — one pass gives us everything
        values = evaluate_c(graph)
        logits = values[logits_id]

        # Extract K/V cache directly from known node IDs
        self.kv_cache = []
        for k_nid, v_nid in kv_node_ids:
            self.kv_cache.append((
                np.asarray(values[k_nid]),
                np.asarray(values[v_nid]),
            ))

        self.position = len(token_ids)
        self.token_history = list(token_ids)

        self.stats.append({
            "step": "prefill",
            "tokens": len(token_ids),
            "nodes": len(graph),
            "position": self.position,
        })

        return logits

    def decode_step(self, token_id: int) -> np.ndarray:
        """Run the machine for one token.  No recompilation.

        Feeds the new token embedding, RoPE, and KV cache into
        the machine's INPUT nodes.  Evaluates.  Updates cache.

        Returns logits array (1, vocab_size).
        """
        f = self.fabric
        m = self._machine

        # Prepare inputs for the machine
        token_embed = f.embed_tokens[token_id:token_id + 1].astype(np.float32)
        rope_cos = self._rope_cos[self.position:self.position + 1]
        rope_sin = self._rope_sin[self.position:self.position + 1]

        inputs: dict[int, np.ndarray] = {
            m.input_ids["token_embed"]: token_embed,
            m.input_ids["rope_cos"]: rope_cos,
            m.input_ids["rope_sin"]: rope_sin,
        }

        # Feed KV cache as inputs
        for li in range(self.num_layers):
            k_cache, v_cache = self.kv_cache[li]
            inputs[m.input_ids[f"L{li}/cache_k"]] = k_cache
            inputs[m.input_ids[f"L{li}/cache_v"]] = v_cache

        # Run the machine
        values = evaluate_c(m.graph, inputs, const_cache=self._const_cache)

        # Extract outputs
        logits = values[m.logits_id]

        # Update KV cache — new_k/new_v include the concat of old + new
        for li in range(self.num_layers):
            new_k_id, new_v_id = m.kv_ids[li]
            self.kv_cache[li] = (
                np.asarray(values[new_k_id]),
                np.asarray(values[new_v_id]),
            )

        self.position += 1
        self.token_history.append(token_id)

        self.stats.append({
            "step": f"decode_{self.position}",
            "token_id": token_id,
            "position": self.position,
        })

        return logits

    def get_stats(self) -> list[dict]:
        """Return statistics for each step."""
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
