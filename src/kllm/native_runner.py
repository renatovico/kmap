"""NativeRunner — CPU simulation of the Processor.

The NativeRunner drives the compiled Processor's datapath on the host
CPU, executing the same sequence of operations that the FPGA FSM would:

1. **Tokenize** — BPE encode via the circuit tokenizer graph.
2. **Prefill** — feed each prompt token through the datapath.
3. **Decode** — autoregressive generation (embed → RoPE → datapath →
   argmax → KV update → repeat).
4. **Detokenize** — BPE decode via the circuit tokenizer graph.

All data flows through the device's circuit graphs — no external
Python libraries in the loop.

Usage::

    from kllm.processor import Processor
    from kllm.native_runner import NativeRunner

    processor = Processor.load("./mychip")
    runner = NativeRunner(processor)
    output_bytes = runner.infer(b"Hello world", max_tokens=50)
"""

from __future__ import annotations

import hashlib
import re
import sys

import numpy as np

from kllm.circuit_executor import (
    ExecutionPlan,
    CTapeRunner,
    evaluate_c,
    precompute_consts,
)
from kllm.processor import Processor


class NativeRunner:
    """Run a Processor natively on the CPU — the virtual device.

    The NativeRunner is the CPU simulation of the chip.  It executes
    the same sequence of operations that the FPGA FSM would:

    1. **Tokenize** — BPE encode using on-chip circuit tokenizer graph.
    2. **Prefill** — feed each prompt token through the datapath.
    3. **Decode** — autoregressive generation (embed → RoPE → datapath →
       argmax → KV update → repeat).
    4. **Detokenize** — BPE decode using on-chip circuit tokenizer graph.

    All data flows through the device's circuit graphs — no external
    Python libraries in the loop.
    """

    def __init__(self, processor: Processor) -> None:
        self.proc = processor
        self._const_cache = precompute_consts(processor.datapath)
        self._plan = ExecutionPlan(processor.datapath, self._const_cache)
        self._c_runner: CTapeRunner | None = None
        self._c_infer_available = False
        self._tape_lib = None

        # Pre-evaluate tokenizer graph constants
        self._tok_const_cache = None
        if processor.tokenizer_graph is not None:
            self._tok_const_cache = precompute_consts(
                processor.tokenizer_graph)

        # Check if C processor_infer is available
        try:
            from kllm.circuit_executor import _get_tape_lib
            self._tape_lib = _get_tape_lib()
            if hasattr(self._tape_lib, 'processor_infer'):
                self._c_infer_available = True
        except Exception:
            pass

    # ------------------------------------------------------------------
    # On-chip tokenize / detokenize (circuit graph-based)
    # ------------------------------------------------------------------

    def encode_bytes(self, raw_bytes: bytes) -> tuple[np.ndarray, int]:
        """Encode raw UTF-8 bytes to token IDs via circuit tokenizer.

        Returns (token_ids array, num_tokens count).
        """
        p = self.proc
        if p.tokenizer_graph is None or p.tokenizer_maps is None:
            raise RuntimeError("No tokenizer circuit on this processor")

        m = p.tokenizer_maps
        g = p.tokenizer_graph

        # Prepare inputs — max-capacity padded buffer
        max_bytes = g.nodes[m.byte_input].shape[0]
        byte_input = np.zeros(max_bytes, dtype=np.uint8)
        n = min(len(raw_bytes), max_bytes)
        byte_input[:n] = list(raw_bytes[:n])
        byte_length = np.array([n], dtype=np.int32)

        inputs = {
            m.byte_input: byte_input,
            m.byte_length: byte_length,
        }

        # The combined graph has decode INPUT nodes too — provide zeros
        max_tokens = g.nodes[m.dec_token_ids].shape[0]
        inputs[m.dec_token_ids] = np.zeros(max_tokens, dtype=np.int32)
        inputs[m.dec_num_tokens] = np.zeros(1, dtype=np.int32)

        # Evaluate the encode subgraph
        values = evaluate_c(g, inputs, const_cache=self._tok_const_cache)
        token_ids = values[m.token_ids]
        num_tokens = int(values[m.num_tokens].flat[0])

        return token_ids, num_tokens

    def decode_to_bytes(
        self,
        token_ids: np.ndarray,
        num_tokens: int,
    ) -> bytes:
        """Decode token IDs to UTF-8 bytes via circuit tokenizer.

        Returns the raw output bytes.
        """
        p = self.proc
        if p.tokenizer_graph is None or p.tokenizer_maps is None:
            raise RuntimeError("No tokenizer circuit on this processor")

        m = p.tokenizer_maps
        g = p.tokenizer_graph

        # Prepare inputs
        max_tokens = g.nodes[m.dec_token_ids].shape[0]
        ids_padded = np.zeros(max_tokens, dtype=np.int32)
        n = min(num_tokens, max_tokens)
        ids_padded[:n] = token_ids[:n]

        inputs = {
            m.dec_token_ids: ids_padded,
            m.dec_num_tokens: np.array([n], dtype=np.int32),
        }

        # The combined graph has encode INPUT nodes too — provide zeros
        max_bytes = g.nodes[m.byte_input].shape[0]
        inputs[m.byte_input] = np.zeros(max_bytes, dtype=np.uint8)
        inputs[m.byte_length] = np.zeros(1, dtype=np.int32)

        values = evaluate_c(g, inputs, const_cache=self._tok_const_cache)
        byte_output = values[m.dec_byte_output]
        byte_length = int(values[m.dec_byte_length].flat[0])

        return bytes(byte_output[:byte_length])

    # ------------------------------------------------------------------
    # Legacy ROM-based decode (backward compat for old chips)
    # ------------------------------------------------------------------

    def decode_token(self, token_id: int) -> str:
        """Decode a single token ID to a string using on-chip ROMs."""
        roms = self.proc.tokenizer_roms
        if not roms:
            raise RuntimeError("No tokenizer ROMs on this processor")

        offsets = roms["id_to_offsets"]
        raw_bytes = roms["id_to_bytes"]
        vocab_size = int(roms["vocab_size"][0])

        if token_id < 0 or token_id >= vocab_size:
            return ""
        offset, length = int(offsets[token_id, 0]), int(offsets[token_id, 1])
        if length == 0:
            return ""
        tok_bytes = bytes(raw_bytes[offset:offset + length])
        return tok_bytes.decode("utf-8", errors="replace")

    def decode_tokens(
        self,
        token_ids: list[int],
        skip_special: bool = True,
    ) -> str:
        """Decode a list of token IDs to text using on-chip ROMs."""
        roms = self.proc.tokenizer_roms
        if not roms:
            raise RuntimeError("No tokenizer ROMs on this processor")

        special_ids = set(roms.get("special_ids", np.array([], dtype=np.int32)).tolist())
        meta = "▁"  # metaspace replacement

        pieces = []
        for tid in token_ids:
            if skip_special and tid in special_ids:
                continue
            pieces.append(self.decode_token(tid))

        text = "".join(pieces)
        # Byte-fallback decoding
        byte_pattern = re.compile(r"(<0x[0-9A-Fa-f]{2}>)+")
        def _decode_bytes(match):
            hex_tokens = re.findall(r"<0x([0-9A-Fa-f]{2})>", match.group())
            return bytes(int(h, 16) for h in hex_tokens).decode("utf-8", errors="replace")
        text = byte_pattern.sub(_decode_bytes, text)
        # Undo metaspace
        text = text.replace(meta, " ")
        if text.startswith(" "):
            text = text[1:]
        return text

    def _ensure_c_runner(self, sample_inputs: dict[int, np.ndarray]) -> None:
        """Lazy-init the C tape runner (needed for both paths)."""
        if self._c_runner is None:
            try:
                self._c_runner = CTapeRunner(
                    self.proc.datapath, self._const_cache,
                    sample_inputs=sample_inputs)
            except Exception:
                pass

    def _build_inputs(
        self,
        token_id: int,
        position: int,
        kv_cache: list[tuple[np.ndarray, np.ndarray]],
    ) -> dict[int, np.ndarray]:
        """Prepare INPUT node values for one forward pass."""
        p = self.proc
        token_embed = p.embed_table[token_id:token_id + 1].astype(np.float32)
        rope_cos = p.rope_cos[position:position + 1]
        rope_sin = p.rope_sin[position:position + 1]

        inputs: dict[int, np.ndarray] = {
            p.input_map["token_embed"]: token_embed,
            p.input_map["rope_cos"]: rope_cos,
            p.input_map["rope_sin"]: rope_sin,
        }
        for li in range(p.num_layers):
            k_cache, v_cache = kv_cache[li]
            inputs[p.input_map[f"L{li}/cache_k"]] = k_cache
            inputs[p.input_map[f"L{li}/cache_v"]] = v_cache

        return inputs

    def _run_datapath(
        self,
        inputs: dict[int, np.ndarray],
    ) -> dict[int, np.ndarray]:
        """Execute one forward pass through the datapath."""
        self._ensure_c_runner(inputs)

        if self._c_runner is not None:
            self._c_runner.run(inputs)
            return {
                nid: self._c_runner.get_value(nid)
                for nid in self.proc.output_map.values()
            }
        else:
            return self._plan.run(inputs)

    def _extract_kv(
        self, values: dict[int, np.ndarray],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Extract new KV cache arrays from datapath outputs."""
        kv = []
        for li in range(self.proc.num_layers):
            k = np.asarray(values[self.proc.output_map[f"L{li}/new_k"]])
            v = np.asarray(values[self.proc.output_map[f"L{li}/new_v"]])
            kv.append((k, v))
        return kv

    def _clamp_max_tokens(
        self,
        prompt_len: int,
        max_tokens: int,
    ) -> int:
        """Clamp max_tokens so total sequence stays within max_seq_len."""
        limit = self.proc.max_seq_len - prompt_len
        if limit <= 0:
            print(f"[warning] prompt length ({prompt_len}) already at or "
                  f"beyond max_seq_len ({self.proc.max_seq_len}); "
                  f"cannot generate new tokens",
                  file=sys.stderr)
            return 0
        if max_tokens > limit:
            print(f"[warning] max_tokens ({max_tokens}) + prompt ({prompt_len}) "
                  f"exceeds max_seq_len ({self.proc.max_seq_len}); "
                  f"clamping to {limit} new tokens",
                  file=sys.stderr)
            return limit
        return max_tokens

    def _infer_c(
        self,
        prompt_tokens: list[int],
        max_tokens: int,
    ) -> list[int]:
        """Run full inference via single C processor_infer() call."""
        import ctypes

        p = self.proc
        lib = self._tape_lib

        # We need a CTapeRunner's ctx — bootstrap it with a sample input
        kv_empty = np.zeros((p.num_kv_heads, 0, p.head_dim), dtype=np.float32)
        kv_cache = [(kv_empty.copy(), kv_empty.copy())
                    for _ in range(p.num_layers)]
        sample = self._build_inputs(prompt_tokens[0] if prompt_tokens else 0,
                                    0, kv_cache)
        self._ensure_c_runner(sample)

        if self._c_runner is None:
            # Fall back to Python loop
            return self._infer_python(prompt_tokens, max_tokens)

        ctx = self._c_runner._ctx

        # Prepare contiguous arrays for C
        embed = np.ascontiguousarray(p.embed_table, dtype=np.float32)
        rope_cos = np.ascontiguousarray(p.rope_cos, dtype=np.float32)
        rope_sin = np.ascontiguousarray(p.rope_sin, dtype=np.float32)

        FP = ctypes.POINTER(ctypes.c_float)
        IP = ctypes.POINTER(ctypes.c_int)

        # Slot mappings
        token_embed_slot = p.input_map["token_embed"]
        rope_cos_slot = p.input_map["rope_cos"]
        rope_sin_slot = p.input_map["rope_sin"]

        # KV slot arrays: [K_in_L0, V_in_L0, K_in_L1, V_in_L1, ...]
        kv_in = []
        kv_out = []
        for li in range(p.num_layers):
            kv_in.append(p.input_map[f"L{li}/cache_k"])
            kv_in.append(p.input_map[f"L{li}/cache_v"])
            kv_out.append(p.output_map[f"L{li}/new_k"])
            kv_out.append(p.output_map[f"L{li}/new_v"])

        kv_in_arr = (ctypes.c_int * len(kv_in))(*kv_in)
        kv_out_arr = (ctypes.c_int * len(kv_out))(*kv_out)

        logits_slot = p.output_map["logits"]

        # Prompt tokens
        prompt_arr = (ctypes.c_int * len(prompt_tokens))(*prompt_tokens)

        # Output buffer
        output_arr = (ctypes.c_int * max_tokens)()
        output_len = ctypes.c_int(0)

        lib.processor_infer(
            ctx,
            embed.ctypes.data_as(FP),
            p.vocab_size,
            p.hidden_dim,
            rope_cos.ctypes.data_as(FP),
            rope_sin.ctypes.data_as(FP),
            p.head_dim,
            token_embed_slot,
            rope_cos_slot,
            rope_sin_slot,
            kv_in_arr,
            kv_out_arr,
            logits_slot,
            p.num_layers,
            p.num_kv_heads,
            prompt_arr,
            len(prompt_tokens),
            max_tokens,
            p.eos_token_id,
            output_arr,
            ctypes.byref(output_len),
        )

        return [output_arr[i] for i in range(output_len.value)]

    def _infer_python(
        self,
        prompt_tokens: list[int],
        max_tokens: int,
    ) -> list[int]:
        """Fallback: Python-driven inference loop."""
        p = self.proc

        kv_cache: list[tuple[np.ndarray, np.ndarray]] = []
        for _ in range(p.num_layers):
            kv_cache.append((
                np.zeros((p.num_kv_heads, 0, p.head_dim), dtype=np.float32),
                np.zeros((p.num_kv_heads, 0, p.head_dim), dtype=np.float32),
            ))

        logits = None
        for pos, token_id in enumerate(prompt_tokens):
            inputs = self._build_inputs(token_id, pos, kv_cache)
            values = self._run_datapath(inputs)
            kv_cache = self._extract_kv(values)
            logits = values[p.output_map["logits"]]

        if logits is None:
            return []

        position = len(prompt_tokens)
        generated: list[int] = []

        for _ in range(max_tokens):
            if position >= p.max_seq_len:
                break
            next_id = int(np.argmax(logits[-1]))
            if next_id == p.eos_token_id:
                break
            generated.append(next_id)

            inputs = self._build_inputs(next_id, position, kv_cache)
            values = self._run_datapath(inputs)
            kv_cache = self._extract_kv(values)
            logits = values[p.output_map["logits"]]
            position += 1

        return generated

    def infer(
        self,
        prompt_tokens: list[int],
        max_tokens: int,
    ) -> list[int]:
        """Run full inference: prefill + autoregressive decode.

        Uses C processor_infer() when available (zero Python between
        tokens), falls back to Python loop otherwise.

        Returns the list of generated token IDs (not including prompt).
        """
        if not prompt_tokens:
            return []

        max_tokens = self._clamp_max_tokens(len(prompt_tokens), max_tokens)
        if max_tokens <= 0:
            return []

        if self._c_infer_available:
            return self._infer_c(prompt_tokens, max_tokens)
        return self._infer_python(prompt_tokens, max_tokens)

    def infer_bytes(
        self,
        input_bytes: bytes,
        max_tokens: int,
    ) -> bytes:
        """Bytes-in, bytes-out inference through the circuit.

        The full loop runs through circuit graphs:
        1. BPE encode via tokenizer circuit (bytes → token IDs)
        2. Prefill + decode via datapath circuit
        3. BPE decode via tokenizer circuit (token IDs → bytes)

        This is the primary API — the same path the hardware FSM takes.
        """
        if not input_bytes:
            return b""

        # 1. Tokenize via circuit
        token_ids_arr, num_tokens = self.encode_bytes(input_bytes)
        prompt_tokens = token_ids_arr[:num_tokens].tolist()

        if not prompt_tokens:
            return b""

        # 2. Run inference (prefill + decode)
        generated = self.infer(prompt_tokens, max_tokens)

        if not generated:
            return b""

        # 3. Detokenize via circuit
        gen_arr = np.array(generated, dtype=np.int32)
        return self.decode_to_bytes(gen_arr, len(generated))

    def infer_streaming(
        self,
        prompt_tokens: list[int],
        max_tokens: int,
    ):
        """Yield generated token IDs one at a time (for live output).

        Note: streaming always uses the Python loop (needs to yield
        between tokens). For batch inference, use infer() which runs
        entirely in C.
        """
        if not prompt_tokens:
            return

        max_tokens = self._clamp_max_tokens(len(prompt_tokens), max_tokens)
        if max_tokens <= 0:
            return

        p = self.proc

        kv_cache: list[tuple[np.ndarray, np.ndarray]] = []
        for _ in range(p.num_layers):
            kv_cache.append((
                np.zeros((p.num_kv_heads, 0, p.head_dim), dtype=np.float32),
                np.zeros((p.num_kv_heads, 0, p.head_dim), dtype=np.float32),
            ))

        logits = None
        for pos, token_id in enumerate(prompt_tokens):
            inputs = self._build_inputs(token_id, pos, kv_cache)
            values = self._run_datapath(inputs)
            kv_cache = self._extract_kv(values)
            logits = values[p.output_map["logits"]]

        if logits is None:
            return

        position = len(prompt_tokens)
        for _ in range(max_tokens):
            if position >= p.max_seq_len:
                break
            next_id = int(np.argmax(logits[-1]))
            if next_id == p.eos_token_id:
                break
            yield next_id

            inputs = self._build_inputs(next_id, position, kv_cache)
            values = self._run_datapath(inputs)
            kv_cache = self._extract_kv(values)
            logits = values[p.output_map["logits"]]
            position += 1

    def infer_bytes_streaming(
        self,
        input_bytes: bytes,
        max_tokens: int,
    ):
        """Yield generated bytes chunks as they're produced.

        Bytes-in, each yield is a bytes chunk for one decoded token.
        """
        if not input_bytes:
            return

        # 1. Tokenize via circuit
        token_ids_arr, num_tokens = self.encode_bytes(input_bytes)
        prompt_tokens = token_ids_arr[:num_tokens].tolist()

        if not prompt_tokens:
            return

        # 2. Stream decode tokens
        for tok_id in self.infer_streaming(prompt_tokens, max_tokens):
            # Decode individual token via circuit
            tok_arr = np.array([tok_id], dtype=np.int32)
            chunk = self.decode_to_bytes(tok_arr, 1)
            yield chunk


# ---------------------------------------------------------------
# Prefix cache — shared across NativeRunner instances
# ---------------------------------------------------------------

_PREFIX_CACHE: dict[str, tuple[list[tuple[np.ndarray, np.ndarray]], int, np.ndarray]] = {}


def _prefix_hash(token_ids: list[int]) -> str:
    """Hash a token prefix for cache lookup."""
    data = np.array(token_ids, dtype=np.int32).tobytes()
    return hashlib.sha256(data).hexdigest()[:16]


def cached_infer(
    processor: Processor,
    prompt_tokens: list[int],
    max_tokens: int,
) -> list[int]:
    """Inference with prefix caching.

    If the same prompt prefix was already prefilled, reuse the KV cache
    and skip straight to decode.
    """
    runner = NativeRunner(processor)
    return runner.infer(prompt_tokens, max_tokens)


def clear_prefix_cache() -> None:
    """Clear the prefix cache."""
    _PREFIX_CACHE.clear()
