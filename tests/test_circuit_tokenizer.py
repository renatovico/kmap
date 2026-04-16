"""Tests for the circuit-based BPE tokenizer.

Tests that:
1. ROM compilation produces correct hash tables and merge arrays.
2. The CircuitGraph BPE_ENCODE op produces correct token IDs via
   the reference evaluator (golden model).
3. The CircuitGraph BPE_DECODE op produces correct UTF-8 bytes.
4. Round-trip: decode(encode(text)) ≈ text.
5. The C executor (evaluate_c) matches the reference evaluator.
6. compile_tokenizer_graph() builds a valid graph.
7. compile_tokenizer_graph_from_json() builds from files.
"""

import json
import os
import tempfile

import numpy as np
import pytest

from kllm.graph.circuit_graph import CircuitGraph, Op
from kllm.compiler.circuit_tokenizer import (
    compile_tokenizer_roms,
    compile_tokenizer_graph,
    compile_tokenizer_graph_from_json,
    TokenizerGraphMaps,
    _fnv1a,
)
from kllm.graph.evaluator import (
    evaluate,
    _vocab_hash_lookup,
    _ref_bpe_encode,
    _ref_bpe_decode,
)
from kllm.graph.circuit_executor import evaluate_c


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

_MINI_VOCAB = {
    "<unk>": 0, "<s>": 1, "</s>": 2,
    "▁": 3, "▁Hello": 4, "▁world": 5,
    "H": 6, "e": 7, "l": 8, "o": 9,
    "▁w": 10, "r": 11, "d": 12,
    "\n": 13,
    # Intermediate BPE pieces (needed for merge chain)
    "▁H": 14, "▁He": 15, "▁Hel": 16, "▁Hell": 17,
    "▁wo": 18, "▁wor": 19, "▁worl": 20,
    # Individual characters needed for initial segmentation
    "w": 21,
}

_MINI_MERGES = {
    ("▁", "H"): 0,
    ("▁H", "e"): 1,
    ("▁He", "l"): 2,
    ("▁Hel", "l"): 3,
    ("▁Hell", "o"): 4,
    ("▁", "w"): 5,
    ("▁w", "o"): 6,
    ("▁wo", "r"): 7,
    ("▁wor", "l"): 8,
    ("▁worl", "d"): 9,
}

_SPECIAL_IDS = {0, 1, 2}


@pytest.fixture
def mini_roms():
    """Compiled ROM arrays for the mini vocabulary."""
    return compile_tokenizer_roms(_MINI_VOCAB, _MINI_MERGES, _SPECIAL_IDS)


@pytest.fixture
def mini_graph():
    """Compiled tokenizer CircuitGraph."""
    return compile_tokenizer_graph(
        _MINI_VOCAB, _MINI_MERGES, _SPECIAL_IDS,
        bos_token_id=1, max_tokens=64, max_bytes=256,
    )


@pytest.fixture
def mini_tok_dir(tmp_path):
    """Create a minimal tokenizer directory on disk."""
    tok_data = {
        "version": "1.0",
        "added_tokens": [
            {"id": 0, "content": "<unk>", "special": True,
             "single_word": False, "lstrip": False, "rstrip": False,
             "normalized": False},
            {"id": 1, "content": "<s>", "special": True,
             "single_word": False, "lstrip": False, "rstrip": False,
             "normalized": False},
            {"id": 2, "content": "</s>", "special": True,
             "single_word": False, "lstrip": False, "rstrip": False,
             "normalized": False},
        ],
        "pre_tokenizer": {
            "type": "Metaspace", "replacement": "▁",
            "add_prefix_space": True,
        },
        "model": {
            "type": "BPE",
            "vocab": _MINI_VOCAB,
            "merges": [[a, b] for (a, b) in sorted(
                _MINI_MERGES.keys(), key=lambda x: _MINI_MERGES[x])],
        },
    }
    config = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
    }
    tok_dir = str(tmp_path / "tokenizer")
    os.makedirs(tok_dir)
    with open(os.path.join(tok_dir, "tokenizer.json"), "w") as f:
        json.dump(tok_data, f)
    with open(os.path.join(tok_dir, "tokenizer_config.json"), "w") as f:
        json.dump(config, f)
    return tok_dir


# ------------------------------------------------------------------
# ROM Compilation Tests
# ------------------------------------------------------------------

class TestRomCompilation:
    def test_produces_all_roms(self, mini_roms):
        expected_keys = {
            "id_to_bytes", "id_to_offsets",
            "merge_a", "merge_b", "merge_result",
            "vocab_hash_keys", "vocab_hash_vals", "vocab_hash_lens",
            "special_ids", "vocab_size",
        }
        assert set(mini_roms.keys()) == expected_keys

    def test_vocab_size(self, mini_roms):
        assert int(mini_roms["vocab_size"][0]) == max(_MINI_VOCAB.values()) + 1

    def test_decode_rom_roundtrip(self, mini_roms):
        """Each token ID should decode to the correct string."""
        id_to_bytes = mini_roms["id_to_bytes"]
        id_to_offsets = mini_roms["id_to_offsets"]
        for piece, tid in _MINI_VOCAB.items():
            offset, length = id_to_offsets[tid]
            decoded = bytes(id_to_bytes[offset:offset + length]).decode("utf-8")
            assert decoded == piece, f"Token {tid}: expected {piece!r}, got {decoded!r}"

    def test_merge_table_sorted(self, mini_roms):
        """Merge table should be sorted by priority (rank)."""
        merge_a = mini_roms["merge_a"]
        merge_b = mini_roms["merge_b"]
        merge_result = mini_roms["merge_result"]
        assert len(merge_a) == len(_MINI_MERGES)
        # First merge should be (▁, H) → ▁H
        assert int(merge_a[0]) == _MINI_VOCAB["▁"]
        assert int(merge_b[0]) == _MINI_VOCAB["H"]
        assert int(merge_result[0]) == _MINI_VOCAB["▁H"]

    def test_special_ids(self, mini_roms):
        special = set(mini_roms["special_ids"].tolist())
        assert special == _SPECIAL_IDS

    def test_hash_table_lookup(self, mini_roms):
        """Each vocab piece should be findable in the hash table."""
        keys = mini_roms["vocab_hash_keys"]
        vals = mini_roms["vocab_hash_vals"]
        lens = mini_roms["vocab_hash_lens"]
        for piece, expected_id in _MINI_VOCAB.items():
            piece_bytes = piece.encode("utf-8")
            found_id = _vocab_hash_lookup(piece_bytes, keys, vals, lens)
            assert found_id == expected_id, (
                f"Hash lookup for {piece!r}: expected {expected_id}, got {found_id}"
            )


# ------------------------------------------------------------------
# Reference BPE Encode Tests
# ------------------------------------------------------------------

class TestRefBpeEncode:
    def test_hello_world(self, mini_roms):
        raw = b"Hello world"
        token_ids, num_tokens = _ref_bpe_encode(
            raw, len(raw),
            mini_roms["vocab_hash_keys"],
            mini_roms["vocab_hash_vals"],
            mini_roms["vocab_hash_lens"],
            mini_roms["merge_a"],
            mini_roms["merge_b"],
            mini_roms["merge_result"],
            mini_roms["special_ids"],
            bos_token_id=1, max_tokens=64,
        )
        n = int(num_tokens[0])
        ids = token_ids[:n].tolist()
        assert ids[0] == 1  # BOS
        assert _MINI_VOCAB["▁Hello"] in ids
        assert _MINI_VOCAB["▁world"] in ids

    def test_empty_input(self, mini_roms):
        raw = b""
        token_ids, num_tokens = _ref_bpe_encode(
            raw, 0,
            mini_roms["vocab_hash_keys"],
            mini_roms["vocab_hash_vals"],
            mini_roms["vocab_hash_lens"],
            mini_roms["merge_a"],
            mini_roms["merge_b"],
            mini_roms["merge_result"],
            mini_roms["special_ids"],
            bos_token_id=1, max_tokens=64,
        )
        n = int(num_tokens[0])
        assert n == 1  # just BOS
        assert int(token_ids[0]) == 1

    def test_bos_token_prepended(self, mini_roms):
        raw = b"Hello"
        token_ids, num_tokens = _ref_bpe_encode(
            raw, len(raw),
            mini_roms["vocab_hash_keys"],
            mini_roms["vocab_hash_vals"],
            mini_roms["vocab_hash_lens"],
            mini_roms["merge_a"],
            mini_roms["merge_b"],
            mini_roms["merge_result"],
            mini_roms["special_ids"],
            bos_token_id=1, max_tokens=64,
        )
        assert int(token_ids[0]) == 1  # BOS is first


# ------------------------------------------------------------------
# Reference BPE Decode Tests
# ------------------------------------------------------------------

class TestRefBpeDecode:
    def test_decode_hello_world(self, mini_roms):
        # Encode first
        token_ids = np.array(
            [_MINI_VOCAB["▁Hello"], _MINI_VOCAB["▁world"]],
            dtype=np.int32,
        )
        byte_output, byte_length = _ref_bpe_decode(
            token_ids, 2,
            mini_roms["id_to_bytes"],
            mini_roms["id_to_offsets"],
            mini_roms["special_ids"],
            max_bytes=256,
        )
        n = int(byte_length[0])
        text = bytes(byte_output[:n]).decode("utf-8")
        assert text == "Hello world"

    def test_skips_special_tokens(self, mini_roms):
        token_ids = np.array([1, _MINI_VOCAB["▁Hello"]], dtype=np.int32)
        byte_output, byte_length = _ref_bpe_decode(
            token_ids, 2,
            mini_roms["id_to_bytes"],
            mini_roms["id_to_offsets"],
            mini_roms["special_ids"],
            max_bytes=256,
        )
        n = int(byte_length[0])
        text = bytes(byte_output[:n]).decode("utf-8")
        assert "<s>" not in text
        assert "Hello" in text


# ------------------------------------------------------------------
# CircuitGraph Encode/Decode Tests
# ------------------------------------------------------------------

class TestCircuitGraphBpe:
    def test_graph_has_bpe_nodes(self, mini_graph):
        g, maps = mini_graph
        has_encode = any(n.op == Op.BPE_ENCODE for n in g.nodes)
        has_decode = any(n.op == Op.BPE_DECODE for n in g.nodes)
        assert has_encode
        assert has_decode

    def _all_inputs(self, g, maps, enc_raw=b"", dec_ids=None, dec_n=0):
        """Build inputs dict for entire graph (encode + decode)."""
        max_bytes = g.nodes[maps.byte_input].shape[0]
        max_tokens = g.nodes[maps.dec_token_ids].shape[0]

        byte_input = np.zeros(max_bytes, dtype=np.uint8)
        if enc_raw:
            n = min(len(enc_raw), max_bytes)
            byte_input[:n] = list(enc_raw[:n])

        d_ids = np.zeros(max_tokens, dtype=np.int32)
        if dec_ids is not None:
            nn = min(len(dec_ids), max_tokens)
            d_ids[:nn] = dec_ids[:nn]

        return {
            maps.byte_input: byte_input,
            maps.byte_length: np.array([len(enc_raw) if enc_raw else 0],
                                        dtype=np.int32),
            maps.dec_token_ids: d_ids,
            maps.dec_num_tokens: np.array([dec_n], dtype=np.int32),
        }

    def test_encode_via_evaluate(self, mini_graph):
        """BPE encode via reference evaluator matches expected output."""
        g, maps = mini_graph
        raw = b"Hello world"

        inputs = self._all_inputs(g, maps, enc_raw=raw)
        values = evaluate(g, inputs)

        token_ids = values[maps.token_ids]
        num_tokens = int(values[maps.num_tokens].flat[0])
        ids = token_ids[:num_tokens].tolist()

        assert ids[0] == 1  # BOS
        assert _MINI_VOCAB["▁Hello"] in ids
        assert _MINI_VOCAB["▁world"] in ids

    def test_decode_via_evaluate(self, mini_graph):
        """BPE decode via reference evaluator matches expected output."""
        g, maps = mini_graph

        dec_ids = np.array(
            [_MINI_VOCAB["▁Hello"], _MINI_VOCAB["▁world"]],
            dtype=np.int32,
        )
        inputs = self._all_inputs(g, maps, dec_ids=dec_ids, dec_n=2)
        values = evaluate(g, inputs)

        byte_output = values[maps.dec_byte_output]
        byte_length = int(values[maps.dec_byte_length].flat[0])
        text = bytes(byte_output[:byte_length]).decode("utf-8")

        assert text == "Hello world"

    def test_encode_via_evaluate_c(self, mini_graph):
        """BPE encode via C executor matches reference evaluator."""
        g, maps = mini_graph
        raw = b"Hello world"

        inputs = self._all_inputs(g, maps, enc_raw=raw)

        ref_values = evaluate(g, inputs)
        c_values = evaluate_c(g, inputs)

        ref_ids = ref_values[maps.token_ids]
        c_ids = c_values[maps.token_ids]
        ref_n = int(ref_values[maps.num_tokens].flat[0])
        c_n = int(c_values[maps.num_tokens].flat[0])

        assert ref_n == c_n
        np.testing.assert_array_equal(ref_ids[:ref_n], c_ids[:c_n])

    def test_roundtrip_encode_decode(self, mini_graph):
        """encode(text) → decode(tokens) ≈ text."""
        g, maps = mini_graph
        text = "Hello world"
        raw = text.encode("utf-8")

        # Encode
        enc_inputs = self._all_inputs(g, maps, enc_raw=raw)
        enc_values = evaluate(g, enc_inputs)
        token_ids = enc_values[maps.token_ids]
        num_tokens = int(enc_values[maps.num_tokens].flat[0])

        # Decode (pass the encoded tokens to the decode subgraph)
        dec_inputs = self._all_inputs(
            g, maps, dec_ids=token_ids[:num_tokens], dec_n=num_tokens,
        )
        dec_values = evaluate(g, dec_inputs)
        byte_output = dec_values[maps.dec_byte_output]
        byte_length = int(dec_values[maps.dec_byte_length].flat[0])
        result = bytes(byte_output[:byte_length]).decode("utf-8")

        assert result == text


# ------------------------------------------------------------------
# From-JSON compilation
# ------------------------------------------------------------------

class TestFromJson:
    def test_compile_from_json(self, mini_tok_dir):
        tok_json = os.path.join(mini_tok_dir, "tokenizer.json")
        tok_cfg = os.path.join(mini_tok_dir, "tokenizer_config.json")
        g, maps = compile_tokenizer_graph_from_json(
            tok_json, tok_cfg, max_tokens=64, max_bytes=256,
        )
        assert isinstance(g, CircuitGraph)
        assert isinstance(maps, TokenizerGraphMaps)
        assert any(n.op == Op.BPE_ENCODE for n in g.nodes)
        assert any(n.op == Op.BPE_DECODE for n in g.nodes)

    def test_from_json_encode(self, mini_tok_dir):
        tok_json = os.path.join(mini_tok_dir, "tokenizer.json")
        tok_cfg = os.path.join(mini_tok_dir, "tokenizer_config.json")
        g, maps = compile_tokenizer_graph_from_json(
            tok_json, tok_cfg, max_tokens=64, max_bytes=256,
        )

        raw = b"Hello world"
        max_bytes = g.nodes[maps.byte_input].shape[0]
        max_tokens = g.nodes[maps.dec_token_ids].shape[0]

        byte_input = np.zeros(max_bytes, dtype=np.uint8)
        byte_input[:len(raw)] = list(raw)

        inputs = {
            maps.byte_input: byte_input,
            maps.byte_length: np.array([len(raw)], dtype=np.int32),
            maps.dec_token_ids: np.zeros(max_tokens, dtype=np.int32),
            maps.dec_num_tokens: np.array([0], dtype=np.int32),
        }
        values = evaluate(g, inputs)
        num_tokens = int(values[maps.num_tokens].flat[0])
        ids = values[maps.token_ids][:num_tokens].tolist()

        assert ids[0] == 1  # BOS
        assert _MINI_VOCAB["▁Hello"] in ids
        assert _MINI_VOCAB["▁world"] in ids


# ------------------------------------------------------------------
# FNV-1a hash
# ------------------------------------------------------------------

class TestFnv1a:
    def test_deterministic(self):
        assert _fnv1a(b"hello") == _fnv1a(b"hello")

    def test_different_inputs(self):
        assert _fnv1a(b"hello") != _fnv1a(b"world")

    def test_empty(self):
        h = _fnv1a(b"")
        assert isinstance(h, int)
        assert h == 2166136261  # FNV offset basis
