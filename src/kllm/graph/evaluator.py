"""Reference evaluator — golden output via NumPy.

The evaluator walks the graph in topological order and computes
each node using NumPy.  This is the REFERENCE implementation
that the C executor must match bit-for-bit.

It uses float arithmetic (NumPy FPU) — that's correct!  The
graph defines WHAT to compute; the evaluator defines HOW for
validation.  The C executor will use byte-plane gates instead.
"""

from __future__ import annotations

import re

import numpy as np

from kllm.graph.circuit_graph import CircuitGraph, Op


# LUT function registry (maps name → NumPy function)
_LUT_REGISTRY: dict[str, callable] = {}


def register_lut(name: str, fn: callable) -> None:
    """Register a NumPy function as a LUT evaluator."""
    _LUT_REGISTRY[name] = fn


def _silu_fn(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        return (x / (1.0 + np.exp(-x.astype(np.float64)))).astype(np.float32)


def _exp_fn(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        return np.exp(x.astype(np.float64)).astype(np.float32)


def _rsqrt_fn(x: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return (1.0 / np.sqrt(x.astype(np.float64))).astype(np.float32)


def _cos_fn(x: np.ndarray) -> np.ndarray:
    return np.cos(x.astype(np.float64)).astype(np.float32)


def _sin_fn(x: np.ndarray) -> np.ndarray:
    return np.sin(x.astype(np.float64)).astype(np.float32)


# Register built-in LUTs
for _name, _fn in [("silu", _silu_fn), ("exp", _exp_fn),
                    ("rsqrt", _rsqrt_fn), ("cos", _cos_fn),
                    ("sin", _sin_fn)]:
    register_lut(_name, _fn)


def evaluate(graph: CircuitGraph,
             inputs: dict[int, np.ndarray] | None = None,
             ) -> dict[int, np.ndarray]:
    """Reference evaluation of a circuit graph via NumPy.

    Parameters
    ----------
    graph : CircuitGraph
        The circuit to evaluate.
    inputs : dict mapping input node ID → NumPy array
        Values for ``INPUT`` nodes.

    Returns
    -------
    dict mapping node ID → NumPy array
        Computed value for every node.
    """
    inputs = inputs or {}
    values: dict[int, np.ndarray] = {}
    order = graph.topological_order()

    for nid in order:
        node = graph.nodes[nid]
        inp = [values[i] for i in node.inputs]

        if node.op == Op.CONST:
            values[nid] = node.params["value"]

        elif node.op == Op.INPUT:
            if nid not in inputs:
                raise ValueError(
                    f"Missing input for node {nid} ({node.name!r})")
            values[nid] = inputs[nid]

        elif node.op == Op.LUT:
            fn_name = node.params["fn"]
            if fn_name not in _LUT_REGISTRY:
                raise ValueError(f"Unknown LUT function: {fn_name!r}")
            values[nid] = _LUT_REGISTRY[fn_name](inp[0])

        elif node.op == Op.ADD:
            values[nid] = (np.asarray(inp[0], dtype=np.float32)
                           + np.asarray(inp[1], dtype=np.float32))

        elif node.op == Op.SUB:
            values[nid] = (np.asarray(inp[0], dtype=np.float32)
                           - np.asarray(inp[1], dtype=np.float32))

        elif node.op == Op.MUL:
            values[nid] = (np.asarray(inp[0], dtype=np.float32)
                           * np.asarray(inp[1], dtype=np.float32))

        elif node.op == Op.DIV:
            values[nid] = (np.asarray(inp[0], dtype=np.float32)
                           / np.asarray(inp[1], dtype=np.float32))

        elif node.op == Op.NEG:
            u = np.asarray(inp[0], dtype=np.float32).view(np.uint32)
            values[nid] = (u ^ np.uint32(0x80000000)).view(np.float32)

        elif node.op == Op.ABS:
            u = np.asarray(inp[0], dtype=np.float32).view(np.uint32)
            values[nid] = (u & np.uint32(0x7FFFFFFF)).view(np.float32)

        elif node.op == Op.SQUARE:
            x = np.asarray(inp[0], dtype=np.float32)
            values[nid] = x * x

        elif node.op == Op.MAX:
            values[nid] = np.maximum(
                np.asarray(inp[0], dtype=np.float32),
                np.asarray(inp[1], dtype=np.float32))

        elif node.op == Op.CMP_LE:
            values[nid] = (np.asarray(inp[0], dtype=np.float32)
                           <= np.asarray(inp[1], dtype=np.float32)
                           ).astype(np.uint8)

        elif node.op == Op.MUX:
            cond = np.asarray(inp[0]).astype(bool)
            values[nid] = np.where(
                cond,
                np.asarray(inp[2], dtype=np.float32),
                np.asarray(inp[1], dtype=np.float32))

        elif node.op == Op.MATMUL:
            values[nid] = (np.asarray(inp[0], dtype=np.float32)
                           @ np.asarray(inp[1], dtype=np.float32))

        elif node.op == Op.SUM:
            values[nid] = np.asarray(inp[0], dtype=np.float32).sum(
                axis=node.params["axis"],
                keepdims=node.params.get("keepdims", False))

        elif node.op == Op.MAX_REDUCE:
            values[nid] = np.asarray(inp[0], dtype=np.float32).max(
                axis=node.params["axis"],
                keepdims=node.params.get("keepdims", False))

        elif node.op == Op.ARGMAX:
            values[nid] = np.asarray(inp[0], dtype=np.float32).argmax(
                axis=node.params["axis"])

        elif node.op == Op.MEAN:
            values[nid] = np.asarray(inp[0], dtype=np.float32).mean(
                axis=node.params["axis"],
                keepdims=node.params.get("keepdims", False))

        elif node.op == Op.RESHAPE:
            values[nid] = np.asarray(inp[0]).reshape(node.params["shape"])

        elif node.op == Op.TRANSPOSE:
            values[nid] = np.asarray(inp[0]).transpose(node.params["axes"])

        elif node.op == Op.CONCAT:
            values[nid] = np.concatenate(inp, axis=node.params["axis"])

        elif node.op == Op.REPEAT:
            values[nid] = np.repeat(inp[0],
                                    node.params["repeats"],
                                    axis=node.params["axis"])

        elif node.op == Op.SLICE:
            values[nid] = inp[0][node.params["slices"]]

        elif node.op == Op.CAST:
            values[nid] = np.asarray(inp[0]).astype(node.params["dtype"])

        elif node.op == Op.EXPAND_DIMS:
            values[nid] = np.expand_dims(inp[0], axis=node.params["axis"])

        elif node.op == Op.BPE_ENCODE:
            values[nid] = _eval_bpe_encode(node, inp, values)

        elif node.op == Op.BPE_DECODE:
            values[nid] = _eval_bpe_decode(node, inp, values)

        else:
            raise ValueError(f"Unknown op: {node.op}")

    return values


# ------------------------------------------------------------------
# BPE reference implementation (ROM-based, matches hardware FSM)
# ------------------------------------------------------------------

_META = "▁"
_META_BYTES = _META.encode("utf-8")  # 0xE2 0x96 0x81


def _vocab_hash_lookup(
    piece_bytes: bytes,
    hash_keys: np.ndarray,
    hash_vals: np.ndarray,
    hash_lens: np.ndarray,
) -> int:
    """Look up a piece in the vocab hash ROM.

    Open-addressing hash table: hash_keys is uint8[table_size, max_piece_len],
    hash_vals is int32[table_size], hash_lens is int32[table_size].
    Returns token ID or -1 if not found.
    """
    table_size = len(hash_vals)
    if table_size == 0:
        return -1
    # FNV-1a hash
    h = 2166136261
    for b in piece_bytes:
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    idx = int(h % table_size)
    for _ in range(table_size):
        stored_len = int(hash_lens[idx])
        if stored_len == 0:
            return -1  # empty slot
        if stored_len == len(piece_bytes):
            if bytes(hash_keys[idx, :stored_len]) == piece_bytes:
                return int(hash_vals[idx])
        idx = (idx + 1) % table_size
    return -1


def _pre_tokenize_bytes(raw: bytes) -> list[bytes]:
    """Metaspace pre-tokenization on raw UTF-8 bytes.

    Adds ▁ prefix, replaces spaces with ▁, splits into words
    where each word starts with ▁.
    """
    if not raw:
        return []
    # Add prefix space then replace spaces with metaspace
    text = " " + raw.decode("utf-8", errors="replace")
    text = text.replace(" ", _META)
    if not text:
        return []
    # Split before ▁ only when preceded by a non-▁ character
    r = re.escape(_META)
    words = re.split(f"(?<=[^{r}])(?={r})", text)
    return [w.encode("utf-8") for w in words if w]


def _bpe_merge_loop(
    token_ids: list[int],
    merge_a: np.ndarray,
    merge_b: np.ndarray,
    merge_result: np.ndarray,
) -> list[int]:
    """Apply BPE merges using the merge-priority ROM.

    Iteratively finds the highest-priority (lowest-rank) merge pair
    in the current token sequence and merges all occurrences.
    """
    num_merges = len(merge_a)

    # Build a reverse lookup: (left_id, right_id) → (rank, result_id)
    merge_lookup: dict[tuple[int, int], tuple[int, int]] = {}
    for rank in range(num_merges):
        key = (int(merge_a[rank]), int(merge_b[rank]))
        if key not in merge_lookup:  # keep lowest rank
            merge_lookup[key] = (rank, int(merge_result[rank]))

    while len(token_ids) > 1:
        # Find the pair with lowest merge rank
        best_rank = num_merges
        best_pair = None
        best_result = -1
        for i in range(len(token_ids) - 1):
            pair = (token_ids[i], token_ids[i + 1])
            entry = merge_lookup.get(pair)
            if entry is not None and entry[0] < best_rank:
                best_rank = entry[0]
                best_pair = pair
                best_result = entry[1]

        if best_pair is None:
            break

        # Merge all occurrences of the best pair
        new_ids: list[int] = []
        i = 0
        while i < len(token_ids):
            if (i < len(token_ids) - 1
                    and token_ids[i] == best_pair[0]
                    and token_ids[i + 1] == best_pair[1]):
                new_ids.append(best_result)
                i += 2
            else:
                new_ids.append(token_ids[i])
                i += 1
        token_ids = new_ids

    return token_ids


def _ref_bpe_encode(
    raw_bytes: bytes,
    byte_length: int,
    hash_keys: np.ndarray,
    hash_vals: np.ndarray,
    hash_lens: np.ndarray,
    merge_a: np.ndarray,
    merge_b: np.ndarray,
    merge_result: np.ndarray,
    special_ids: np.ndarray,
    bos_token_id: int,
    max_tokens: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference BPE encode: UTF-8 bytes → token IDs.

    Returns (token_ids[max_tokens], num_tokens[1]) as int32 arrays.
    """
    actual_bytes = raw_bytes[:byte_length]

    # Pre-tokenize into words
    words = _pre_tokenize_bytes(actual_bytes)

    all_ids: list[int] = [bos_token_id]

    for word_bytes in words:
        # Split word into characters, look up each in vocab
        text = word_bytes.decode("utf-8", errors="replace")
        char_ids: list[int] = []
        for ch in text:
            ch_bytes = ch.encode("utf-8")
            tid = _vocab_hash_lookup(ch_bytes, hash_keys, hash_vals, hash_lens)
            if tid >= 0:
                char_ids.append(tid)
            else:
                # Byte-fallback: <0xHH> tokens
                for b in ch_bytes:
                    tok_str = f"<0x{b:02X}>"
                    tok_bytes = tok_str.encode("utf-8")
                    fallback_id = _vocab_hash_lookup(
                        tok_bytes, hash_keys, hash_vals, hash_lens)
                    if fallback_id >= 0:
                        char_ids.append(fallback_id)
                    # else: drop (unknown byte, shouldn't happen)

        # Apply BPE merges
        merged = _bpe_merge_loop(char_ids, merge_a, merge_b, merge_result)
        all_ids.extend(merged)

    # Pad and return
    n = min(len(all_ids), max_tokens)
    token_ids = np.zeros(max_tokens, dtype=np.int32)
    token_ids[:n] = all_ids[:n]
    num_tokens = np.array([n], dtype=np.int32)
    return token_ids, num_tokens


def _ref_bpe_decode(
    token_ids: np.ndarray,
    num_tokens: int,
    id_to_bytes: np.ndarray,
    id_to_offsets: np.ndarray,
    special_ids: np.ndarray,
    max_bytes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference BPE decode: token IDs → UTF-8 bytes.

    Returns (byte_output[max_bytes], byte_length[1]).
    """
    special_set = set(int(x) for x in special_ids)
    meta = _META

    pieces: list[str] = []
    for i in range(num_tokens):
        tid = int(token_ids[i])
        if tid in special_set:
            continue
        vocab_size = id_to_offsets.shape[0]
        if tid < 0 or tid >= vocab_size:
            continue
        offset = int(id_to_offsets[tid, 0])
        length = int(id_to_offsets[tid, 1])
        if length == 0:
            continue
        tok_bytes = bytes(id_to_bytes[offset:offset + length])
        pieces.append(tok_bytes.decode("utf-8", errors="replace"))

    text = "".join(pieces)

    # Byte-fallback decoding: <0xHH> sequences → actual bytes
    byte_pattern = re.compile(r"(<0x[0-9A-Fa-f]{2}>)+")
    def _decode_bytes(match):
        hex_tokens = re.findall(r"<0x([0-9A-Fa-f]{2})>", match.group())
        return bytes(int(h, 16) for h in hex_tokens).decode(
            "utf-8", errors="replace")
    text = byte_pattern.sub(_decode_bytes, text)

    # Undo metaspace
    text = text.replace(meta, " ")
    if text.startswith(" "):
        text = text[1:]

    result_bytes = text.encode("utf-8")
    n = min(len(result_bytes), max_bytes)
    byte_output = np.zeros(max_bytes, dtype=np.uint8)
    byte_output[:n] = list(result_bytes[:n])
    byte_length = np.array([n], dtype=np.int32)
    return byte_output, byte_length


def _eval_bpe_encode(node, inp, values):
    """Evaluate a BPE_ENCODE node."""
    # inp: [byte_input, byte_length, hash_keys, hash_vals, hash_lens,
    #       merge_a, merge_b, merge_result, special_ids]
    byte_input = np.asarray(inp[0], dtype=np.uint8)
    byte_length = int(np.asarray(inp[1]).flat[0])
    hash_keys = np.asarray(inp[2], dtype=np.uint8)
    hash_vals = np.asarray(inp[3], dtype=np.int32)
    hash_lens = np.asarray(inp[4], dtype=np.int32)
    merge_a = np.asarray(inp[5], dtype=np.int32)
    merge_b = np.asarray(inp[6], dtype=np.int32)
    merge_result = np.asarray(inp[7], dtype=np.int32)
    special_ids = np.asarray(inp[8], dtype=np.int32)

    bos_token_id = node.params.get("bos_token_id", 1)
    max_tokens = node.params.get("max_tokens", 2048)
    output_index = node.params.get("output_index", 0)

    # Check if we already computed the paired node
    paired = node.params.get("paired_node")
    if paired is not None and paired in values:
        # The paired node already computed both outputs — reuse
        pass

    raw = bytes(byte_input[:byte_length])
    token_ids, num_tokens = _ref_bpe_encode(
        raw, byte_length,
        hash_keys, hash_vals, hash_lens,
        merge_a, merge_b, merge_result, special_ids,
        bos_token_id, max_tokens,
    )

    if output_index == 0:
        return token_ids
    else:
        return num_tokens


def _eval_bpe_decode(node, inp, values):
    """Evaluate a BPE_DECODE node."""
    # inp: [token_ids, num_tokens, id_to_bytes, id_to_offsets, special_ids]
    token_ids = np.asarray(inp[0], dtype=np.int32)
    num_tokens = int(np.asarray(inp[1]).flat[0])
    id_to_bytes_rom = np.asarray(inp[2], dtype=np.uint8)
    id_to_offsets = np.asarray(inp[3], dtype=np.int32)
    special_ids = np.asarray(inp[4], dtype=np.int32)

    max_bytes = node.params.get("max_bytes", 8192)
    output_index = node.params.get("output_index", 0)

    byte_output, byte_length = _ref_bpe_decode(
        token_ids, num_tokens,
        id_to_bytes_rom, id_to_offsets, special_ids,
        max_bytes,
    )

    if output_index == 0:
        return byte_output
    else:
        return byte_length
