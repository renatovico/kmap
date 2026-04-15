"""Tests for the pure-Python BPE tokenizer."""

import json
import os
import tempfile

import numpy as np
import pytest

from kllm.tokenizer import Tokenizer

# Minimal tokenizer fixture (subset of TinyLlama vocab)
_MINI_VOCAB = {
    "<unk>": 0, "<s>": 1, "</s>": 2,
    "▁": 3, "▁Hello": 4, "▁world": 5,
    "H": 6, "e": 7, "l": 8, "o": 9,
    "▁w": 10, "r": 11, "d": 12,
    "\n": 13,
}
_MINI_MERGES = [["▁", "H"], ["▁H", "e"], ["▁He", "l"], ["▁Hel", "l"], ["▁Hell", "o"],
                ["▁", "w"], ["▁w", "o"], ["▁wo", "r"], ["▁wor", "l"], ["▁worl", "d"]]


@pytest.fixture
def mini_tok(tmp_path):
    """Create a minimal tokenizer on disk."""
    tok_data = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {"id": 0, "content": "<unk>", "special": True, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False},
            {"id": 1, "content": "<s>", "special": True, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False},
            {"id": 2, "content": "</s>", "special": True, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False},
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "add_prefix_space": True},
        "post_processor": None,
        "decoder": None,
        "model": {"type": "BPE", "vocab": _MINI_VOCAB, "merges": _MINI_MERGES, "byte_fallback": False},
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
    return Tokenizer(tok_dir)


class TestTokenizerInit:
    def test_loads_vocab(self, mini_tok):
        assert mini_tok.bos_token_id == 1
        assert mini_tok.eos_token_id == 2

    def test_special_tokens(self, mini_tok):
        assert "<s>" in mini_tok._special_tokens
        assert "</s>" in mini_tok._special_tokens


class TestEncode:
    def test_hello_world(self, mini_tok):
        ids = mini_tok.encode("Hello world")
        assert ids[0] == 1  # BOS
        assert 4 in ids     # ▁Hello
        assert 5 in ids     # ▁world

    def test_add_bos(self, mini_tok):
        with_bos = mini_tok.encode("Hello", add_bos=True)
        without_bos = mini_tok.encode("Hello", add_bos=False)
        assert with_bos[0] == 1
        assert without_bos[0] != 1

    def test_empty_string(self, mini_tok):
        ids = mini_tok.encode("")
        assert ids == [1]  # just BOS

    def test_special_token_in_text(self, mini_tok):
        ids = mini_tok.encode("Hello</s>world")
        assert 2 in ids  # EOS token present


class TestDecode:
    def test_roundtrip(self, mini_tok):
        text = "Hello world"
        ids = mini_tok.encode(text)
        decoded = mini_tok.decode(ids, skip_special_tokens=True)
        assert decoded == text

    def test_skip_special(self, mini_tok):
        decoded = mini_tok.decode([1, 4, 5], skip_special_tokens=True)
        assert "<s>" not in decoded

    def test_keep_special(self, mini_tok):
        decoded = mini_tok.decode([1, 4, 5], skip_special_tokens=False)
        assert "<s>" in decoded


class TestChatTemplate:
    def test_user_message(self, mini_tok):
        msgs = [{"role": "user", "content": "Hello"}]
        result = mini_tok.apply_chat_template(msgs)
        assert "<|user|>" in result
        assert "Hello" in result
        assert mini_tok.eos_token in result

    def test_generation_prompt(self, mini_tok):
        msgs = [{"role": "user", "content": "Hello"}]
        result = mini_tok.apply_chat_template(
            msgs, add_generation_prompt=True,
        )
        assert result.endswith("<|assistant|>\n")


# ---- Integration test with real tokenizer (skip if not compiled) ----

_REAL_TOK_DIR = os.path.join(
    os.path.dirname(__file__), "..", "lossless_logic", "tokenizer",
)

_has_real_tok = os.path.isdir(_REAL_TOK_DIR)


@pytest.mark.skipif(not _has_real_tok, reason="No compiled tokenizer")
class TestRealTokenizer:
    @pytest.fixture(autouse=True)
    def _load(self):
        self.tok = Tokenizer(_REAL_TOK_DIR)

    def test_encode_simple(self):
        ids = self.tok.encode("Hello world")
        assert ids == [1, 15043, 3186]

    def test_encode_question(self):
        ids = self.tok.encode("How are you?")
        assert ids == [1, 1128, 526, 366, 29973]

    def test_decode_roundtrip(self):
        text = "Hello world"
        assert self.tok.decode(self.tok.encode(text), skip_special_tokens=True) == text

    def test_unicode_roundtrip(self):
        text = "Olá, como está?"
        decoded = self.tok.decode(self.tok.encode(text), skip_special_tokens=True)
        assert decoded == text


# ---- ROM compilation tests ----

class TestTokenizerROMs:
    def test_compile_roms_returns_dict(self, mini_tok):
        roms = mini_tok.compile_roms()
        assert isinstance(roms, dict)
        assert "id_to_bytes" in roms
        assert "id_to_offsets" in roms
        assert "merge_a" in roms
        assert "merge_b" in roms
        assert "merge_result" in roms
        assert "special_ids" in roms
        assert "vocab_size" in roms

    def test_compile_roms_vocab_size(self, mini_tok):
        roms = mini_tok.compile_roms()
        vocab_size = int(roms["vocab_size"][0])
        assert vocab_size > 0
        assert roms["id_to_offsets"].shape == (vocab_size, 2)

    def test_compile_roms_decode_roundtrip(self, mini_tok):
        """Token from ROM decode should match token from Python decode."""
        roms = mini_tok.compile_roms()
        offsets = roms["id_to_offsets"]
        raw_bytes = roms["id_to_bytes"]

        # Check known tokens
        for token_str, token_id in [("▁Hello", 4), ("▁world", 5)]:
            offset, length = int(offsets[token_id, 0]), int(offsets[token_id, 1])
            rom_str = bytes(raw_bytes[offset:offset + length]).decode("utf-8")
            assert rom_str == token_str

    def test_compile_roms_merges(self, mini_tok):
        roms = mini_tok.compile_roms()
        assert len(roms["merge_a"]) == len(roms["merge_b"])
        assert len(roms["merge_a"]) == len(roms["merge_result"])
        assert len(roms["merge_a"]) > 0

    def test_compile_roms_special_ids(self, mini_tok):
        roms = mini_tok.compile_roms()
        special = set(roms["special_ids"].tolist())
        assert mini_tok.bos_token_id in special
        assert mini_tok.eos_token_id in special

    def test_save_load_roms(self, mini_tok, tmp_path):
        roms_dir = str(tmp_path / "roms")
        mini_tok.save_roms(roms_dir)

        loaded = Tokenizer.load_roms(roms_dir)
        assert set(loaded.keys()) == set(mini_tok.compile_roms().keys())

        for key in loaded:
            np.testing.assert_array_equal(
                loaded[key], mini_tok.compile_roms()[key],
            )

    def test_compile_roms_cached(self, mini_tok):
        """Second call returns same object (cached)."""
        roms1 = mini_tok.compile_roms()
        roms2 = mini_tok.compile_roms()
        assert roms1 is roms2
