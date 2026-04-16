"""Circuit tokenizer — BPE as a CircuitGraph.

The tokenizer is a circuit component.  Its data (vocabulary, merge
priorities) compiles into ROM arrays stored as CONST nodes, and the
BPE algorithm is a ROM-backed FSM expressed as ``BPE_ENCODE`` /
``BPE_DECODE`` ops in the graph.

This module provides:

1. **ROM compilation** — tokenizer data → NumPy arrays (same as before).
2. **Graph compilation** — build a ``CircuitGraph`` for encode/decode.
3. **Vocab hash table** — open-addressing hash table ROM for O(1)
   piece→token_id lookup during encode.

The compiled graph can be:
- Evaluated by the reference evaluator (NumPy golden model).
- Executed by the C gate executor or tape runner.
- Synthesised to BRAM/ROM blocks on FPGA.

ROMs produced
-------------
* ``id_to_bytes`` + ``id_to_offsets`` : decode ROM (token_id → UTF-8 bytes)
* ``merge_a``, ``merge_b``, ``merge_result`` : encode ROM (sorted by rank)
* ``vocab_hash_keys``, ``vocab_hash_vals``, ``vocab_hash_lens`` : hash table
* ``special_ids`` : special token ids (bos, eos, …)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from kllm.circuit_graph import CircuitGraph


# ------------------------------------------------------------------
# ROM compilation — tokenizer data → chip ROM arrays
# ------------------------------------------------------------------

def compile_tokenizer_roms(
    vocab: dict[str, int],
    merges: dict[tuple[str, str], int],
    special_token_ids: set[int],
) -> dict[str, np.ndarray]:
    """Compile tokenizer tables into ROM-ready NumPy arrays.

    Returns a dict of arrays that can be saved as .npy files and
    loaded onto the chip (or synthesised into BRAM).
    """
    vocab_size = max(vocab.values()) + 1

    # --- Decode ROM: token_id → UTF-8 bytes ---
    all_bytes = bytearray()
    offsets = np.zeros((vocab_size, 2), dtype=np.int32)
    id_to_token = {v: k for k, v in vocab.items()}

    for tid in range(vocab_size):
        tok_str = id_to_token.get(tid, "")
        tok_bytes = tok_str.encode("utf-8")
        offsets[tid] = [len(all_bytes), len(tok_bytes)]
        all_bytes.extend(tok_bytes)

    id_to_bytes = np.frombuffer(bytes(all_bytes), dtype=np.uint8).copy()

    # --- Encode ROM: merge table (sorted by priority) ---
    num_merges = len(merges)
    merge_a = np.zeros(num_merges, dtype=np.int32)
    merge_b = np.zeros(num_merges, dtype=np.int32)
    merge_result = np.zeros(num_merges, dtype=np.int32)

    sorted_merges = sorted(merges.items(), key=lambda x: x[1])
    for rank, ((piece_a, piece_b), _) in enumerate(sorted_merges):
        a_id = vocab.get(piece_a, -1)
        b_id = vocab.get(piece_b, -1)
        merged_str = piece_a + piece_b
        result_id = vocab.get(merged_str, -1)
        merge_a[rank] = a_id
        merge_b[rank] = b_id
        merge_result[rank] = result_id

    # --- Vocab hash table (open addressing, linear probing) ---
    hash_roms = _compile_vocab_hash(vocab)

    # --- Special token IDs ---
    special_arr = np.array(sorted(special_token_ids), dtype=np.int32)

    return {
        "id_to_bytes": id_to_bytes,
        "id_to_offsets": offsets,
        "merge_a": merge_a,
        "merge_b": merge_b,
        "merge_result": merge_result,
        "vocab_hash_keys": hash_roms["keys"],
        "vocab_hash_vals": hash_roms["vals"],
        "vocab_hash_lens": hash_roms["lens"],
        "special_ids": special_arr,
        "vocab_size": np.array([vocab_size], dtype=np.int32),
    }


def _fnv1a(data: bytes) -> int:
    """FNV-1a hash (32-bit)."""
    h = 2166136261
    for b in data:
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def _compile_vocab_hash(
    vocab: dict[str, int],
    load_factor: float = 0.6,
    max_piece_len: int = 64,
) -> dict[str, np.ndarray]:
    """Build open-addressing hash table ROMs for piece→token_id.

    Table layout (all fixed-size for BRAM):
      keys  : uint8[table_size, max_piece_len]  — UTF-8 piece bytes
      vals  : int32[table_size]                 — token ID
      lens  : int32[table_size]                 — piece byte length (0=empty)
    """
    n = len(vocab)
    table_size = int(n / load_factor) + 1
    # Next power of 2 for fast modulo (optional, but good for hardware)
    ts = 1
    while ts < table_size:
        ts <<= 1
    table_size = ts

    keys = np.zeros((table_size, max_piece_len), dtype=np.uint8)
    vals = np.full(table_size, -1, dtype=np.int32)
    lens = np.zeros(table_size, dtype=np.int32)

    for piece_str, token_id in vocab.items():
        piece_bytes = piece_str.encode("utf-8")
        if len(piece_bytes) > max_piece_len:
            continue  # skip oversized pieces (shouldn't happen in practice)
        h = _fnv1a(piece_bytes)
        idx = h % table_size
        for _ in range(table_size):
            if lens[idx] == 0:  # empty slot
                keys[idx, :len(piece_bytes)] = list(piece_bytes)
                vals[idx] = token_id
                lens[idx] = len(piece_bytes)
                break
            idx = (idx + 1) % table_size

    return {"keys": keys, "vals": vals, "lens": lens}


def save_roms(roms: dict[str, np.ndarray], directory: str) -> None:
    """Save compiled ROM arrays to disk."""
    os.makedirs(directory, exist_ok=True)
    for name, arr in roms.items():
        np.save(os.path.join(directory, f"{name}.npy"), arr)


def load_roms(directory: str) -> dict[str, np.ndarray]:
    """Load previously compiled ROM arrays from disk."""
    roms = {}
    for fname in os.listdir(directory):
        if fname.endswith(".npy"):
            name = fname[:-4]
            roms[name] = np.load(os.path.join(directory, fname))
    return roms


# ------------------------------------------------------------------
# Graph compilation — tokenizer → CircuitGraph
# ------------------------------------------------------------------

@dataclass
class TokenizerGraphMaps:
    """Input/output node ID maps for a tokenizer CircuitGraph."""
    # Encode inputs/outputs
    byte_input: int          # INPUT: uint8[max_bytes]
    byte_length: int         # INPUT: int32 scalar
    token_ids: int           # OUTPUT: int32[max_tokens]
    num_tokens: int          # OUTPUT: int32 scalar
    # Decode inputs/outputs
    dec_token_ids: int       # INPUT: int32[max_tokens]
    dec_num_tokens: int      # INPUT: int32 scalar
    dec_byte_output: int     # OUTPUT: uint8[max_bytes]
    dec_byte_length: int     # OUTPUT: int32 scalar


def compile_tokenizer_graph(
    vocab: dict[str, int],
    merges: dict[tuple[str, str], int],
    special_token_ids: set[int],
    bos_token_id: int = 1,
    max_tokens: int = 2048,
    max_bytes: int = 8192,
) -> tuple[CircuitGraph, TokenizerGraphMaps]:
    """Build a CircuitGraph for BPE encode + decode.

    The graph has:
    - INPUT nodes for raw bytes (encode) and token IDs (decode)
    - CONST nodes for all ROM arrays
    - BPE_ENCODE / BPE_DECODE computation nodes

    Returns (graph, maps) where maps contains the node IDs.
    """
    roms = compile_tokenizer_roms(vocab, merges, special_token_ids)

    g = CircuitGraph()

    # --- CONST nodes (ROM arrays) ---
    c_hash_keys = g.const(roms["vocab_hash_keys"], name="vocab_hash_keys")
    c_hash_vals = g.const(roms["vocab_hash_vals"], name="vocab_hash_vals")
    c_hash_lens = g.const(roms["vocab_hash_lens"], name="vocab_hash_lens")
    c_merge_a = g.const(roms["merge_a"], name="merge_a")
    c_merge_b = g.const(roms["merge_b"], name="merge_b")
    c_merge_result = g.const(roms["merge_result"], name="merge_result")
    c_special = g.const(roms["special_ids"], name="special_ids")
    c_id_to_bytes = g.const(roms["id_to_bytes"], name="id_to_bytes")
    c_id_to_offsets = g.const(roms["id_to_offsets"], name="id_to_offsets")

    # --- Encode subgraph ---
    enc_byte_input = g.input(
        (max_bytes,), dtype=np.dtype(np.uint8), name="enc/byte_input")
    enc_byte_length = g.input(
        (1,), dtype=np.dtype(np.int32), name="enc/byte_length")

    enc_token_ids, enc_num_tokens = g.bpe_encode(
        byte_input=enc_byte_input,
        byte_length=enc_byte_length,
        vocab_hash_keys=c_hash_keys,
        vocab_hash_vals=c_hash_vals,
        vocab_hash_lens=c_hash_lens,
        merge_a=c_merge_a,
        merge_b=c_merge_b,
        merge_result=c_merge_result,
        special_ids=c_special,
        bos_token_id=bos_token_id,
        max_tokens=max_tokens,
        name="encode",
    )

    # --- Decode subgraph ---
    dec_token_ids_in = g.input(
        (max_tokens,), dtype=np.dtype(np.int32), name="dec/token_ids")
    dec_num_tokens_in = g.input(
        (1,), dtype=np.dtype(np.int32), name="dec/num_tokens")

    dec_byte_output, dec_byte_length = g.bpe_decode(
        token_ids=dec_token_ids_in,
        num_tokens=dec_num_tokens_in,
        id_to_bytes=c_id_to_bytes,
        id_to_offsets=c_id_to_offsets,
        special_ids=c_special,
        max_bytes=max_bytes,
        name="decode",
    )

    maps = TokenizerGraphMaps(
        byte_input=enc_byte_input,
        byte_length=enc_byte_length,
        token_ids=enc_token_ids,
        num_tokens=enc_num_tokens,
        dec_token_ids=dec_token_ids_in,
        dec_num_tokens=dec_num_tokens_in,
        dec_byte_output=dec_byte_output,
        dec_byte_length=dec_byte_length,
    )

    return g, maps


def compile_tokenizer_graph_from_json(
    tokenizer_json_path: str,
    tokenizer_config_path: str,
    bos_token_id: int | None = None,
    max_tokens: int = 2048,
    max_bytes: int = 8192,
) -> tuple[CircuitGraph, TokenizerGraphMaps]:
    """Build tokenizer CircuitGraph directly from tokenizer.json files.

    This replaces the need for the Python Tokenizer class — reads the
    JSON files directly and compiles the graph.
    """
    import json

    with open(tokenizer_json_path, encoding="utf-8") as f:
        data = json.load(f)
    with open(tokenizer_config_path, encoding="utf-8") as f:
        config = json.load(f)

    model = data["model"]

    # Vocabulary
    vocab: dict[str, int] = model["vocab"]

    # Merges
    merges: dict[tuple[str, str], int] = {}
    for rank, entry in enumerate(model.get("merges", [])):
        if isinstance(entry, list):
            a, b = entry
        else:
            a, b = entry.split(" ", 1)
        merges[(a, b)] = rank

    # Special tokens
    special_tokens: set[str] = {
        at["content"] for at in data.get("added_tokens", [])
    }
    special_token_ids: set[int] = {
        vocab[t] for t in special_tokens if t in vocab
    }

    # BOS token
    bos_token = config.get("bos_token", "<s>")
    if bos_token_id is None:
        bos_token_id = vocab.get(bos_token, 1)

    return compile_tokenizer_graph(
        vocab, merges, special_token_ids,
        bos_token_id=bos_token_id,
        max_tokens=max_tokens,
        max_bytes=max_bytes,
    )
