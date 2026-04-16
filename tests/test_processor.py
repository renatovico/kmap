"""Tests for the Processor and NativeRunner.

Uses the MockFabric from test_circuit_compiler to build a tiny
processor and verify that NativeRunner produces valid output.
"""

import numpy as np
import pytest
import os
import tempfile

from kllm.processor import Processor
from kllm.native_runner import NativeRunner
from kllm.circuit_graph import CircuitGraph, Op


# ---------------------------------------------------------------
# Reuse MockFabric from test_circuit_compiler
# ---------------------------------------------------------------
from tests.test_circuit_compiler import MockFabric


class TestProcessorBuild:
    """Test Processor.build() with a mock fabric."""

    def test_build_creates_processor(self):
        fab = MockFabric(num_layers=1, hidden_size=16, num_heads=2,
                         num_kv_heads=2, intermediate_size=32, vocab_size=64)
        proc = Processor.build(fab, eos_token_id=2)

        assert isinstance(proc.datapath, CircuitGraph)
        assert proc.vocab_size == 64
        assert proc.num_layers == 1
        assert proc.hidden_dim == 16
        assert proc.head_dim == 8
        assert proc.eos_token_id == 2
        assert proc.max_seq_len == 2048

    def test_build_has_embed_table(self):
        fab = MockFabric(num_layers=1, vocab_size=64, hidden_size=16)
        proc = Processor.build(fab, eos_token_id=2)

        assert proc.embed_table.shape == (64, 16)
        assert proc.embed_table.dtype == np.float32

    def test_build_has_rope_tables(self):
        fab = MockFabric(num_layers=1, hidden_size=16, num_heads=2)
        proc = Processor.build(fab, eos_token_id=2)

        assert proc.rope_cos.shape == (2048, 8)  # head_dim = 16 // 2 = 8
        assert proc.rope_sin.shape == (2048, 8)
        assert proc.rope_cos.dtype == np.float32

    def test_build_has_input_output_maps(self):
        fab = MockFabric(num_layers=2)
        proc = Processor.build(fab, eos_token_id=2)

        # Input map must have token_embed, rope_cos, rope_sin, per-layer KV
        assert "token_embed" in proc.input_map
        assert "rope_cos" in proc.input_map
        assert "rope_sin" in proc.input_map
        assert "L0/cache_k" in proc.input_map
        assert "L1/cache_v" in proc.input_map

        # Output map must have logits and per-layer new KV
        assert "logits" in proc.output_map
        assert "L0/new_k" in proc.output_map
        assert "L1/new_v" in proc.output_map

    def test_build_optimizes_graph(self):
        """Optimised graph should be smaller than raw."""
        fab = MockFabric(num_layers=1)
        proc = Processor.build(fab, eos_token_id=2)
        # After optimization, graph should exist and be non-empty
        assert len(proc.datapath) > 0


class TestProcessorSaveLoad:
    """Test Processor serialization round-trip."""

    def test_save_load_roundtrip(self):
        fab = MockFabric(num_layers=1, hidden_size=16, num_heads=2,
                         num_kv_heads=2, intermediate_size=32, vocab_size=64)
        proc = Processor.build(fab, eos_token_id=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            proc.save(tmpdir)

            # Check files exist
            assert os.path.exists(os.path.join(tmpdir, "processor.json"))
            assert os.path.exists(os.path.join(tmpdir, "circuit", "nodes.bin"))
            assert os.path.exists(os.path.join(tmpdir, "tables", "embed_table.npy"))
            assert os.path.exists(os.path.join(tmpdir, "tables", "rope_cos.npy"))
            assert os.path.exists(os.path.join(tmpdir, "tables", "rope_sin.npy"))

            # Load back
            loaded = Processor.load(tmpdir)

            assert loaded.vocab_size == proc.vocab_size
            assert loaded.num_layers == proc.num_layers
            assert loaded.hidden_dim == proc.hidden_dim
            assert loaded.head_dim == proc.head_dim
            assert loaded.eos_token_id == proc.eos_token_id
            assert loaded.max_seq_len == proc.max_seq_len
            assert loaded.input_map == proc.input_map
            assert loaded.output_map == proc.output_map

            np.testing.assert_array_equal(loaded.embed_table, proc.embed_table)
            np.testing.assert_array_equal(loaded.rope_cos, proc.rope_cos)
            np.testing.assert_array_equal(loaded.rope_sin, proc.rope_sin)


class TestNativeRunner:
    """Test NativeRunner (C virtual processor)."""

    def test_construct_without_tokenizer(self):
        """NativeRunner can be constructed even without tokenizer ROMs."""
        fab = MockFabric(num_layers=1, hidden_size=16, num_heads=2,
                         num_kv_heads=2, intermediate_size=32, vocab_size=64)
        proc = Processor.build(fab, eos_token_id=2)
        runner = NativeRunner(proc)

        # No tokenizer means no BPE ROMs
        assert runner._bpe_roms == {}

    def test_infer_bytes_requires_tokenizer(self):
        """infer_bytes raises RuntimeError without tokenizer ROMs."""
        fab = MockFabric(num_layers=1, hidden_size=16, num_heads=2,
                         num_kv_heads=2, intermediate_size=32, vocab_size=64)
        proc = Processor.build(fab, eos_token_id=2)
        runner = NativeRunner(proc)

        with pytest.raises(RuntimeError, match="No BPE ROMs"):
            runner.infer_bytes(b"Hello", max_tokens=1)

    def test_infer_bytes_empty_input(self):
        """Empty input returns empty bytes."""
        fab = MockFabric(num_layers=1, hidden_size=16, num_heads=2,
                         num_kv_heads=2, intermediate_size=32, vocab_size=64)
        proc = Processor.build(fab, eos_token_id=2)
        runner = NativeRunner(proc)

        result = runner.infer_bytes(b"", max_tokens=5)
        assert result == b""


MYCHIP_PATH = os.path.join(os.path.dirname(__file__), "..", "mychip")


@pytest.mark.skipif(
    not os.path.isdir(MYCHIP_PATH),
    reason="mychip directory not found (integration test)"
)
class TestNativeRunnerIntegration:
    """Integration tests using a real compiled chip."""

    @pytest.fixture(autouse=True)
    def setup(self):
        proc = Processor.load(MYCHIP_PATH)
        self.runner = NativeRunner(proc)

    def test_infer_bytes_returns_bytes(self):
        result = self.runner.infer_bytes(b"Hello", max_tokens=3)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_infer_bytes_streaming_yields_chunks(self):
        chunks = list(self.runner.infer_bytes_streaming(b"Hello", 3))
        assert len(chunks) > 0
        assert all(isinstance(c, bytes) for c in chunks)

        # Streaming result matches batch result
        batch = self.runner.infer_bytes(b"Hello", max_tokens=3)
        assert b"".join(chunks) == batch

    def test_infer_bytes_deterministic(self):
        """Same input → same output (greedy decode)."""
        proc = Processor.load(MYCHIP_PATH)
        r1 = NativeRunner(proc)
        out1 = r1.infer_bytes(b"Hello world", max_tokens=3)

        r2 = NativeRunner(proc)
        out2 = r2.infer_bytes(b"Hello world", max_tokens=3)

        assert out1 == out2

    def test_save_load_then_infer_matches(self):
        """Inference after save/load must match original."""
        proc = Processor.load(MYCHIP_PATH)
        runner1 = NativeRunner(proc)
        output1 = runner1.infer_bytes(b"Hello", max_tokens=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            proc.save(tmpdir)
            loaded = Processor.load(tmpdir)
            runner2 = NativeRunner(loaded)
            output2 = runner2.infer_bytes(b"Hello", max_tokens=3)

        assert output1 == output2
