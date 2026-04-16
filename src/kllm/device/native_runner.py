"""NativeRunner — C virtual processor for chip simulation.

The NativeRunner is the CPU emulation of the compiled Processor chip.
Everything runs in C — tokenization, prefill, decode, detokenization,
KV cache management. Python is only the thin ctypes bridge.

The execution path mirrors what the physical FPGA would do:

1. BPE encode raw UTF-8 bytes → token IDs  (C)
2. Prefill prompt tokens through the datapath circuit  (C)
3. Autoregressive decode: embed → RoPE → tape_run → argmax → KV update  (C)
4. BPE decode each generated token → UTF-8 bytes  (C)

All data flows through the device's circuit tape — no Python in the loop.

Usage::

    from kllm.device.processor import Processor
    from kllm.device.native_runner import NativeRunner

    processor = Processor.load("./mychip")
    runner = NativeRunner(processor)

    # Streaming (bytes in, bytes out — each chunk as generated):
    for chunk in runner.infer_bytes_streaming(b"Hello world", max_tokens=50):
        sys.stdout.buffer.write(chunk)

    # Batch:
    result = runner.infer_bytes(b"Hello world", max_tokens=50)
"""

from __future__ import annotations

import ctypes
import sys
from typing import Iterator

import numpy as np

from kllm.graph.circuit_executor import (
    CTapeRunner,
    precompute_consts,
    _get_tape_lib,
)
from kllm.graph.circuit_graph import Op
from kllm.device.processor import Processor


# ctypes convenience
_FP = ctypes.POINTER(ctypes.c_float)
_IP = ctypes.POINTER(ctypes.c_int)
_BP = ctypes.POINTER(ctypes.c_ubyte)
_I32P = ctypes.POINTER(ctypes.c_int32)

# Callback type matching C: int (*)(const uint8_t*, int, void*)
_CALLBACK_TYPE = ctypes.CFUNCTYPE(
    ctypes.c_int,           # return: 0=continue, non-zero=stop
    ctypes.POINTER(ctypes.c_ubyte),  # bytes
    ctypes.c_int,           # byte_len
    ctypes.c_void_p,        # user_data
)


def _int_arr(vals):
    """Create a ctypes int array from a Python sequence."""
    return (ctypes.c_int * len(vals))(*vals)


class NativeRunner:
    """C virtual processor — emulates the chip entirely in C.

    All inference runs through ``processor_infer_bytes()`` in the C
    tape runner. Python only marshals data in/out via ctypes.
    """

    def __init__(self, processor: Processor) -> None:
        self.proc = processor

        # Load the C tape runner library
        self._lib = _get_tape_lib()
        if not hasattr(self._lib, 'processor_infer_bytes'):
            raise RuntimeError(
                "C tape runner missing processor_infer_bytes(). "
                "Recompile csrc/_tape_runner.c.")

        # Pre-compute datapath constants and build the C tape runner
        self._const_cache = precompute_consts(processor.datapath)

        # Bootstrap CTapeRunner with a sample input (size-1 KV cache
        # to avoid zero-dim issues during tape compilation)
        sample = self._build_sample_inputs()
        self._c_runner = CTapeRunner(
            processor.datapath, self._const_cache,
            sample_inputs=sample)

        # Extract BPE ROM arrays from the tokenizer circuit graph
        self._bpe_roms = self._extract_bpe_roms()

    # ------------------------------------------------------------------
    # Internal: extract BPE ROM data from tokenizer graph CONST nodes
    # ------------------------------------------------------------------

    def _extract_bpe_roms(self) -> dict[str, np.ndarray]:
        """Pull ROM arrays from the tokenizer circuit graph's CONST nodes."""
        p = self.proc
        if p.tokenizer_graph is None:
            return {}

        g = p.tokenizer_graph
        consts = precompute_consts(g)

        roms: dict[str, np.ndarray] = {}
        for nid, node in enumerate(g.nodes):
            if node is None or node.op != Op.CONST:
                continue
            name = node.name or ""
            if name in ("vocab_hash_keys", "vocab_hash_vals",
                        "vocab_hash_lens", "merge_a", "merge_b",
                        "merge_result", "special_ids",
                        "id_to_bytes", "id_to_offsets"):
                roms[name] = np.ascontiguousarray(consts[nid])
        return roms

    def _build_sample_inputs(self) -> dict[int, np.ndarray]:
        """Build sample inputs with size-1 KV cache for tape bootstrap."""
        p = self.proc
        token_embed = p.embed_table[0:1].astype(np.float32)
        rope_cos = p.rope_cos[0:1]
        rope_sin = p.rope_sin[0:1]

        inputs: dict[int, np.ndarray] = {
            p.input_map["token_embed"]: token_embed,
            p.input_map["rope_cos"]: rope_cos,
            p.input_map["rope_sin"]: rope_sin,
        }
        # Use size-1 KV cache (not zero!) so shapes are non-degenerate
        for li in range(p.num_layers):
            kv = np.zeros((p.num_kv_heads, 1, p.head_dim), dtype=np.float32)
            inputs[p.input_map[f"L{li}/cache_k"]] = kv.copy()
            inputs[p.input_map[f"L{li}/cache_v"]] = kv.copy()

        return inputs

    # ------------------------------------------------------------------
    # Public API: bytes-in → bytes-out (all in C)
    # ------------------------------------------------------------------

    def infer_bytes(self, input_bytes: bytes, max_tokens: int) -> bytes:
        """Full inference: raw UTF-8 bytes in → raw UTF-8 bytes out.

        Everything runs in one C call. Returns all output bytes at once.
        """
        chunks: list[bytes] = []
        for chunk in self.infer_bytes_streaming(input_bytes, max_tokens):
            chunks.append(chunk)
        return b"".join(chunks)

    def infer_bytes_streaming(
        self,
        input_bytes: bytes,
        max_tokens: int,
    ) -> Iterator[bytes]:
        """Yield decoded byte chunks as each token is generated.

        The entire inference loop (BPE encode → prefill → decode →
        BPE decode per token) runs in C. A ctypes callback yields
        each decoded chunk back to Python.
        """
        if not input_bytes:
            return

        roms = self._bpe_roms
        if not roms:
            raise RuntimeError(
                "No BPE ROMs found in tokenizer circuit. "
                "Re-create chip with `kllm create`.")

        p = self.proc
        lib = self._lib
        ctx = self._c_runner._ctx

        # Marshal BPE ROM pointers
        hash_keys = roms["vocab_hash_keys"]
        hash_vals = np.ascontiguousarray(roms["vocab_hash_vals"], dtype=np.int32)
        hash_lens = np.ascontiguousarray(roms["vocab_hash_lens"], dtype=np.int32)
        merge_a = np.ascontiguousarray(roms["merge_a"], dtype=np.int32)
        merge_b = np.ascontiguousarray(roms["merge_b"], dtype=np.int32)
        merge_result = np.ascontiguousarray(roms["merge_result"], dtype=np.int32)
        id_to_bytes_rom = np.ascontiguousarray(roms["id_to_bytes"], dtype=np.uint8)
        id_to_offsets = np.ascontiguousarray(roms["id_to_offsets"], dtype=np.int32)
        special_ids = np.ascontiguousarray(roms["special_ids"], dtype=np.int32)

        table_size = hash_keys.shape[0]
        max_piece_len = hash_keys.shape[1]
        num_merges = len(merge_a)
        num_special = len(special_ids)

        # BOS token ID (second special token = 1 for LLaMA-family models)
        bos_token_id = 1
        if num_special > 1:
            bos_token_id = int(special_ids[1])

        # Processor resources
        embed = np.ascontiguousarray(p.embed_table, dtype=np.float32)
        rope_cos = np.ascontiguousarray(p.rope_cos, dtype=np.float32)
        rope_sin = np.ascontiguousarray(p.rope_sin, dtype=np.float32)

        # Slot mappings
        token_embed_slot = p.input_map["token_embed"]
        rope_cos_slot = p.input_map["rope_cos"]
        rope_sin_slot = p.input_map["rope_sin"]

        kv_in = []
        kv_out = []
        for li in range(p.num_layers):
            kv_in.append(p.input_map[f"L{li}/cache_k"])
            kv_in.append(p.input_map[f"L{li}/cache_v"])
            kv_out.append(p.output_map[f"L{li}/new_k"])
            kv_out.append(p.output_map[f"L{li}/new_v"])

        logits_slot = p.output_map["logits"]

        # Input bytes as contiguous buffer
        input_buf = np.frombuffer(input_bytes, dtype=np.uint8).copy()

        # Collect chunks from C callback
        chunks: list[bytes] = []

        @_CALLBACK_TYPE
        def _on_token(byte_ptr, byte_len, _user_data):
            if byte_len > 0:
                chunk = bytes(byte_ptr[:byte_len])
                chunks.append(chunk)
            return 0

        # prevent GC
        self._callback_ref = _on_token

        total_generated = ctypes.c_int(0)
        max_bpe_bytes = 8192

        lib.processor_infer_bytes(
            ctx,
            # BPE ROMs
            hash_keys.ctypes.data_as(_BP),
            hash_vals.ctypes.data_as(_I32P),
            hash_lens.ctypes.data_as(_I32P),
            table_size, max_piece_len,
            merge_a.ctypes.data_as(_I32P),
            merge_b.ctypes.data_as(_I32P),
            merge_result.ctypes.data_as(_I32P),
            num_merges,
            id_to_bytes_rom.ctypes.data_as(_BP),
            id_to_offsets.ctypes.data_as(_I32P),
            special_ids.ctypes.data_as(_I32P),
            num_special,
            bos_token_id,
            # Processor resources
            embed.ctypes.data_as(_FP),
            p.vocab_size,
            p.hidden_dim,
            rope_cos.ctypes.data_as(_FP),
            rope_sin.ctypes.data_as(_FP),
            p.head_dim,
            # Slot mappings
            token_embed_slot,
            rope_cos_slot,
            rope_sin_slot,
            _int_arr(kv_in),
            _int_arr(kv_out),
            logits_slot,
            p.num_layers,
            p.num_kv_heads,
            # Inference params
            input_buf.ctypes.data_as(_BP),
            len(input_bytes),
            max_tokens,
            p.eos_token_id,
            p.max_seq_len,
            max_bpe_bytes,
            # Callback
            _on_token,
            None,
            # Output
            ctypes.byref(total_generated),
        )

        yield from chunks
