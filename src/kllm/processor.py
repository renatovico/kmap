"""Processor — a complete autonomous inference machine.

The Processor is a self-contained chip that takes prompt tokens and
produces output tokens.  It bundles:

- **Datapath**: A CircuitGraph for one decode step (weights baked in
  as CONST, variable state as INPUT).
- **Embed table**: Token ID → embedding vector (ROM on chip).
- **RoPE tables**: Position → cos/sin values (ROM on chip).
- **Config**: eos_token_id, max_seq_len, num_layers, dimensions.

The same Processor drives:
1. **Native execution** via ``NativeRunner`` (C tape runner loop).
2. **HDL export** (Verilog FSM + datapath + ROMs).
3. **HDL simulation** (iverilog with golden reference).

Usage::

    processor = Processor.build(fabric, eos_token_id=2)
    processor.save("./mychip")
    processor = Processor.load("./mychip")

    runner = NativeRunner(processor)
    output_ids = runner.infer([1, 5, 10], max_tokens=50)
"""

from __future__ import annotations

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


class Processor:
    """Complete inference processor — datapath + resources + config.

    The processor is the compiled chip: everything needed to run
    inference autonomously, without Python in the loop.
    """

    def __init__(
        self,
        datapath: CircuitGraph,
        input_map: dict[str, int],
        output_map: dict[str, int],
        embed_table: np.ndarray,
        rope_cos: np.ndarray,
        rope_sin: np.ndarray,
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
    ) -> Processor:
        """Build a processor from a loaded Fabric.

        1. Compile single-token decode template (weights = CONST).
        2. Optimise the datapath graph.
        3. Extract embed table and RoPE tables.
        4. Bundle into a Processor.
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

        print(f"  Datapath: {len(machine.graph)} nodes → "
              f"{len(opt_graph)} optimised")

        return cls(
            datapath=opt_graph,
            input_map=input_map,
            output_map=output_map,
            embed_table=embed_table,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
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

        return cls(
            datapath=datapath,
            input_map=config["input_map"],
            output_map=config["output_map"],
            embed_table=embed_table,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            eos_token_id=config["eos_token_id"],
            max_seq_len=config["max_seq_len"],
            vocab_size=config["vocab_size"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            num_kv_heads=config["num_kv_heads"],
            head_dim=config["head_dim"],
        )


class NativeRunner:
    """Run a Processor natively on the CPU.

    Manages the inference loop in Python+C: prefill all prompt tokens,
    then decode autoregressively.  Each forward pass goes through the
    C tape runner (or Python ExecutionPlan fallback).

    This mirrors what a hardware FSM would do — the same sequence of
    operations (embed lookup, RoPE select, feed datapath, read logits,
    argmax, update KV) happens here in C, and in Verilog as an FSM.
    """

    def __init__(self, processor: Processor) -> None:
        self.proc = processor
        self._const_cache = precompute_consts(processor.datapath)
        self._plan = ExecutionPlan(processor.datapath, self._const_cache)
        self._c_runner: CTapeRunner | None = None

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
        # Lazy-init C runner on first call (needs real input shapes)
        if self._c_runner is None:
            try:
                self._c_runner = CTapeRunner(
                    self.proc.datapath, self._const_cache,
                    sample_inputs=inputs)
            except Exception:
                pass

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

    def infer(
        self,
        prompt_tokens: list[int],
        max_tokens: int,
    ) -> list[int]:
        """Run full inference: prefill + autoregressive decode.

        This is the complete inference loop — the same sequence a
        hardware FSM would execute.  No compilation per token.

        Returns the list of generated token IDs (not including prompt).
        """
        p = self.proc

        # Initialize empty KV cache per layer
        kv_cache: list[tuple[np.ndarray, np.ndarray]] = []
        for _ in range(p.num_layers):
            kv_cache.append((
                np.zeros((p.num_kv_heads, 0, p.head_dim), dtype=np.float32),
                np.zeros((p.num_kv_heads, 0, p.head_dim), dtype=np.float32),
            ))

        # ---- Prefill: process each prompt token ----
        logits = None
        for pos, token_id in enumerate(prompt_tokens):
            inputs = self._build_inputs(token_id, pos, kv_cache)
            values = self._run_datapath(inputs)
            kv_cache = self._extract_kv(values)
            logits = values[p.output_map["logits"]]

        if logits is None:
            return []

        # ---- Decode: autoregressive generation ----
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

    def infer_streaming(
        self,
        prompt_tokens: list[int],
        max_tokens: int,
    ):
        """Yield generated token IDs one at a time (for live output)."""
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
