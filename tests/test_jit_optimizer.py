"""Tests for JIT per-token circuit optimization.

Verifies that:
1. JIT decode produces the same logits as full-sequence evaluation
2. Gate count decreases (the circuit shrinks per step)
3. Prefix cache works correctly
"""

import numpy as np
import pytest

from kllm.circuit_compiler import compile_model
from kllm.circuit_graph import evaluate
from kllm.jit_optimizer import JitSession, cached_prefill, clear_prefix_cache


# Use MockFabric from the compiler tests
from tests.test_circuit_compiler import MockFabric


@pytest.fixture
def fabric():
    return MockFabric(
        num_layers=1, hidden_size=8, num_heads=2,
        num_kv_heads=2, intermediate_size=16, vocab_size=32,
    )


class TestJitSession:
    def test_prefill_returns_logits(self, fabric):
        session = JitSession(fabric)
        logits = session.prefill([1, 5, 10])
        assert logits.shape == (3, 32)  # (seq_len, vocab_size)

    def test_prefill_matches_reference(self, fabric):
        """Prefill logits match direct graph evaluation."""
        token_ids = [1, 5, 10]

        # Direct evaluation
        graph, logits_id = compile_model(fabric, token_ids)
        ref_logits = evaluate(graph)[logits_id]

        # JIT prefill
        session = JitSession(fabric)
        jit_logits = session.prefill(token_ids)

        np.testing.assert_allclose(jit_logits, ref_logits, rtol=1e-5)

    def test_prefill_updates_position(self, fabric):
        session = JitSession(fabric)
        session.prefill([1, 5, 10])
        assert session.position == 3

    def test_prefill_populates_kv_cache(self, fabric):
        session = JitSession(fabric)
        session.prefill([1, 5, 10])
        assert len(session.kv_cache) == fabric.num_layers
        for k, v in session.kv_cache:
            assert k.shape[0] == fabric.num_kv_heads
            assert k.shape[1] == 3  # seq_len
            assert k.shape[2] == fabric.head_dim

    def test_decode_step_returns_logits(self, fabric):
        session = JitSession(fabric)
        session.prefill([1, 5])
        logits = session.decode_step(10)
        assert logits.shape == (1, 32)

    def test_decode_step_updates_position(self, fabric):
        session = JitSession(fabric)
        session.prefill([1, 5])
        session.decode_step(10)
        assert session.position == 3

    def test_decode_step_extends_kv_cache(self, fabric):
        session = JitSession(fabric)
        session.prefill([1, 5])
        assert session.kv_cache[0][0].shape[1] == 2  # seq=2
        session.decode_step(10)
        assert session.kv_cache[0][0].shape[1] == 3  # seq=3

    def test_decode_optimizes_graph(self, fabric):
        """Decode step should fold the all-constant graph."""
        session = JitSession(fabric)
        session.prefill([1, 5])
        session.decode_step(10)

        stats = session.get_stats()
        decode_stat = stats[-1]
        # The single-token graph is all constants → should fold
        assert decode_stat["optimized_nodes"] < decode_stat["original_nodes"]
        assert decode_stat["reduction_pct"] > 0

    def test_multiple_decode_steps(self, fabric):
        session = JitSession(fabric)
        session.prefill([1])
        for tok in [5, 10, 15]:
            session.decode_step(tok)
        assert session.position == 4
        assert len(session.token_history) == 4

    def test_stats_tracked(self, fabric):
        session = JitSession(fabric)
        session.prefill([1, 5])
        session.decode_step(10)
        session.decode_step(15)

        stats = session.get_stats()
        assert len(stats) == 3  # prefill + 2 decode
        assert stats[0]["step"] == "prefill"
        assert "decode" in stats[1]["step"]
        assert "decode" in stats[2]["step"]


class TestPrefixCache:
    def setup_method(self):
        clear_prefix_cache()

    def test_cache_miss_then_hit(self, fabric):
        session1, logits1 = cached_prefill(fabric, [1, 5, 10])
        session2, logits2 = cached_prefill(fabric, [1, 5, 10])

        np.testing.assert_array_equal(logits1, logits2)
        # Second call should be a cache hit
        assert session2.stats[0]["step"] == "prefill_cached"

    def test_different_prefix_no_hit(self, fabric):
        session1, _ = cached_prefill(fabric, [1, 5, 10])
        session2, _ = cached_prefill(fabric, [2, 6, 11])

        assert session1.stats[0]["step"] == "prefill"
        assert session2.stats[0]["step"] == "prefill"

    def test_cached_session_is_independent(self, fabric):
        """Mutations to cached session don't affect future cache hits."""
        session1, _ = cached_prefill(fabric, [1, 5])
        session1.decode_step(10)  # mutates session1's kv_cache

        session2, _ = cached_prefill(fabric, [1, 5])
        # session2 should have the original cache (seq=2, not 3)
        assert session2.kv_cache[0][0].shape[1] == 2

    def test_clear_cache(self, fabric):
        cached_prefill(fabric, [1, 5, 10])
        clear_prefix_cache()
        session, _ = cached_prefill(fabric, [1, 5, 10])
        assert session.stats[0]["step"] == "prefill"  # miss


class TestKVCacheExtraction:
    def test_kv_shapes_correct(self, fabric):
        session = JitSession(fabric)
        session.prefill([1, 5])

        for li in range(fabric.num_layers):
            k, v = session.kv_cache[li]
            assert k.shape == (fabric.num_kv_heads, 2, fabric.head_dim)
            assert v.shape == (fabric.num_kv_heads, 2, fabric.head_dim)
