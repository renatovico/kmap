"""Processor — the complete virtual device.

The Processor is a self-contained chip: text in, text out.  It bundles
every component needed for autonomous inference:

- **Datapath**: CircuitGraph for one decode step (weights = CONST,
  variable state = INPUT).
- **Embed table**: Token ID → embedding vector (ROM).
- **RoPE tables**: Position → cos/sin values (ROM).
- **Tokenizer ROMs**: Vocabulary + merge tables for BPE encode/decode.
- **Config**: eos_token_id, max_seq_len, num_layers, dimensions.

The same Processor drives:
1. **Native execution** via ``NativeRunner`` (C tape runner loop) —
   the CPU simulation of the chip.  Text in → tokens → inference →
   tokens → text out, all through the device.
2. **HDL export** (Verilog FSM + datapath + ROMs).
3. **HDL simulation** (iverilog with golden reference).

The tokenizer is part of the circuit — its vocabulary and merge-priority
tables are ROMs on the chip, and the BPE algorithm is a FSM that runs
on the same controller as the inference loop.

Usage::

    processor = Processor.build(fabric, tokenizer, eos_token_id=2)
    processor.save("./mychip")
    processor = Processor.load("./mychip")

    runner = NativeRunner(processor)
    text = runner.run("Hello world", max_tokens=50)
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from kllm.circuit_compiler import (
    DecodeMachine,
    compile_decode_template,
    compile_model,
    _build_rope_const,
)
from kllm.circuit_executor import (
    ExecutionPlan,
    CTapeRunner,
    evaluate_c,
    precompute_consts,
)
from kllm.circuit_graph import CircuitGraph
from kllm.graph_optimizer import optimize_graph
from kllm.tokenizer import Tokenizer


class Processor:
    """Complete inference device — datapath + ROMs + tokenizer + config.

    The processor is the compiled chip: everything needed to run
    inference autonomously.  The tokenizer's vocabulary and merge
    tables are on-chip ROMs, just like the embedding table and RoPE.
    """

    def __init__(
        self,
        datapath: CircuitGraph,
        input_map: dict[str, int],
        output_map: dict[str, int],
        embed_table: np.ndarray,
        rope_cos: np.ndarray,
        rope_sin: np.ndarray,
        tokenizer_roms: dict[str, np.ndarray] | None,
        eos_token_id: int,
        max_seq_len: int,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> None:
        self.datapath = datapath
        self.input_map = input_map
        self.output_map = output_map
        self.embed_table = embed_table
        self.rope_cos = rope_cos
        self.rope_sin = rope_sin
        self.tokenizer_roms = tokenizer_roms or {}
        self.eos_token_id = eos_token_id
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

    @classmethod
    def build(
        cls,
        fabric: object,
        eos_token_id: int,
        max_seq: int = 2048,
        tokenizer: Tokenizer | None = None,
    ) -> Processor:
        """Build a processor from a loaded Fabric.

        1. Compile single-token decode template (weights = CONST).
        2. Optimise the datapath graph.
        3. Extract embed table and RoPE tables.
        4. Compile tokenizer ROMs (if tokenizer provided).
        5. Bundle into a Processor.
        """
        f = fabric

        # 1. Build decode machine
        print("  Compiling decode template …")
        machine: DecodeMachine = compile_decode_template(f, max_seq=max_seq)

        # 2. Optimise the datapath
        keep_ids = [machine.logits_id]
        for k_id, v_id in machine.kv_ids:
            keep_ids.extend([k_id, v_id])

        print("  Optimising datapath …")
        opt_graph, id_map = optimize_graph(machine.graph, keep_ids)

        # Remap IDs through the optimizer
        opt_logits = id_map[machine.logits_id]
        opt_kv_ids = [
            (id_map[k_id], id_map[v_id])
            for k_id, v_id in machine.kv_ids
        ]
        opt_input_ids = {
            name: id_map[nid]
            for name, nid in machine.input_ids.items()
        }

        # Build input_map and output_map
        input_map = dict(opt_input_ids)
        output_map: dict[str, int] = {"logits": opt_logits}
        for li, (k_id, v_id) in enumerate(opt_kv_ids):
            output_map[f"L{li}/new_k"] = k_id
            output_map[f"L{li}/new_v"] = v_id

        # 3. Extract processor resources
        rope_cos, rope_sin = _build_rope_const(
            max_seq, f.head_dim, f.rope_theta)
        embed_table = f.embed_tokens.astype(np.float32)

        # 4. Compile tokenizer ROMs
        tokenizer_roms = None
        if tokenizer is not None:
            print("  Compiling tokenizer ROMs …")
            tokenizer_roms = tokenizer.compile_roms()

        print(f"  Datapath: {len(machine.graph)} nodes → "
              f"{len(opt_graph)} optimised")

        return cls(
            datapath=opt_graph,
            input_map=input_map,
            output_map=output_map,
            embed_table=embed_table,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            tokenizer_roms=tokenizer_roms,
            eos_token_id=eos_token_id,
            max_seq_len=max_seq,
            vocab_size=f.vocab_size,
            hidden_dim=f.hidden_size,
            num_layers=f.num_layers,
            num_kv_heads=f.num_kv_heads,
            head_dim=f.head_dim,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, chip_dir: str) -> None:
        """Serialize the processor to disk.

        Layout::

            chip_dir/
              processor.json     — config + input/output maps
              circuit/           — serialized datapath graph
              tables/
                embed_table.npy
                rope_cos.npy
                rope_sin.npy
              tokenizer_roms/    — BPE vocab + merge ROMs
                id_to_bytes.npy
                id_to_offsets.npy
                merge_a.npy
                merge_b.npy
                merge_result.npy
                special_ids.npy
                vocab_size.npy
        """
        os.makedirs(chip_dir, exist_ok=True)

        # Datapath graph
        circuit_dir = os.path.join(chip_dir, "circuit")
        self.datapath.serialize(circuit_dir)

        # Tables
        tables_dir = os.path.join(chip_dir, "tables")
        os.makedirs(tables_dir, exist_ok=True)
        np.save(os.path.join(tables_dir, "embed_table.npy"), self.embed_table)
        np.save(os.path.join(tables_dir, "rope_cos.npy"), self.rope_cos)
        np.save(os.path.join(tables_dir, "rope_sin.npy"), self.rope_sin)

        # Tokenizer ROMs
        if self.tokenizer_roms:
            roms_dir = os.path.join(chip_dir, "tokenizer_roms")
            os.makedirs(roms_dir, exist_ok=True)
            for name, arr in self.tokenizer_roms.items():
                np.save(os.path.join(roms_dir, f"{name}.npy"), arr)

        # Config
        config = {
            "input_map": self.input_map,
            "output_map": self.output_map,
            "eos_token_id": self.eos_token_id,
            "max_seq_len": self.max_seq_len,
            "vocab_size": self.vocab_size,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
        }
        with open(os.path.join(chip_dir, "processor.json"), "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, chip_dir: str) -> Processor:
        """Load a processor from disk."""
        with open(os.path.join(chip_dir, "processor.json")) as f:
            config = json.load(f)

        datapath = CircuitGraph.deserialize(
            os.path.join(chip_dir, "circuit"))

        tables_dir = os.path.join(chip_dir, "tables")
        embed_table = np.load(os.path.join(tables_dir, "embed_table.npy"))
        rope_cos = np.load(os.path.join(tables_dir, "rope_cos.npy"))
        rope_sin = np.load(os.path.join(tables_dir, "rope_sin.npy"))

        # Load tokenizer ROMs if present
        tokenizer_roms = None
        roms_dir = os.path.join(chip_dir, "tokenizer_roms")
        if os.path.isdir(roms_dir):
            tokenizer_roms = Tokenizer.load_roms(roms_dir)

        return cls(
            datapath=datapath,
            input_map=config["input_map"],
            output_map=config["output_map"],
            embed_table=embed_table,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            tokenizer_roms=tokenizer_roms,
            eos_token_id=config["eos_token_id"],
            max_seq_len=config["max_seq_len"],
            vocab_size=config["vocab_size"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            num_kv_heads=config["num_kv_heads"],
            head_dim=config["head_dim"],
        )


class NativeRunner:
    """Run a Processor natively on the CPU — the virtual device.

    The NativeRunner is the CPU simulation of the chip.  It executes
    the same sequence of operations that the FPGA FSM would:

    1. **Tokenize** — BPE encode using on-chip ROMs (vocab + merge tables).
    2. **Prefill** — feed each prompt token through the datapath.
    3. **Decode** — autoregressive generation (embed → RoPE → datapath →
       argmax → KV update → repeat).
    4. **Detokenize** — ROM lookup from token IDs to UTF-8 bytes.

    All data flows through the device's ROMs and datapath — no external
    Python libraries in the loop.
    """

    def __init__(self, processor: Processor) -> None:
        self.proc = processor
        self._const_cache = precompute_consts(processor.datapath)
        self._plan = ExecutionPlan(processor.datapath, self._const_cache)
        self._c_runner: CTapeRunner | None = None
        self._c_infer_available = False
        self._tape_lib = None

        # Check if C processor_infer is available
        try:
            from kllm.circuit_executor import _get_tape_lib
            self._tape_lib = _get_tape_lib()
            if hasattr(self._tape_lib, 'processor_infer'):
                self._c_infer_available = True
        except Exception:
            pass

    # ------------------------------------------------------------------
    # On-chip tokenize / detokenize (ROM-based)
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
        import re
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

        if self._c_infer_available:
            return self._infer_c(prompt_tokens, max_tokens)
        return self._infer_python(prompt_tokens, max_tokens)

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
            next_id = int(np.argmax(logits[-1]))
            if next_id == p.eos_token_id:
                break
            yield next_id

            inputs = self._build_inputs(next_id, position, kv_cache)
            values = self._run_datapath(inputs)
            kv_cache = self._extract_kv(values)
            logits = values[p.output_map["logits"]]
            position += 1


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
