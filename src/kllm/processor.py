"""Processor — the complete virtual device.

The Processor is a self-contained chip definition.  It bundles every
component needed for autonomous inference:

- **Datapath**: CircuitGraph for one decode step (weights = CONST,
  variable state = INPUT).
- **Tokenizer graph**: CircuitGraph for BPE encode/decode (ROMs as CONST).
- **Embed table**: Token ID → embedding vector (ROM).
- **RoPE tables**: Position → cos/sin values (ROM).
- **Config**: eos_token_id, max_seq_len, num_layers, dimensions.

The same Processor drives:
1. **Native execution** via ``NativeRunner`` (in ``native_runner.py``)
   — the CPU simulation of the chip.
2. **HDL export** (Verilog FSM + datapath + ROMs).
3. **HDL simulation** (iverilog with golden reference).

The tokenizer is part of the circuit — its vocabulary and merge-priority
tables are CONST nodes in the tokenizer CircuitGraph, and the BPE
algorithm is a ROM-backed FSM expressed as BPE_ENCODE/BPE_DECODE ops.

Usage::

    processor = Processor.build(fabric, eos_token_id=2,
                                tokenizer_dir="./mychip/tokenizer")
    processor.save("./mychip")
    processor = Processor.load("./mychip")

    from kllm.native_runner import NativeRunner
    runner = NativeRunner(processor)
    output_bytes = runner.infer(b"Hello world", max_tokens=50)
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
from kllm.circuit_executor import evaluate_c
from kllm.circuit_graph import CircuitGraph
from kllm.circuit_tokenizer import (
    compile_tokenizer_graph_from_json,
    load_roms as load_tokenizer_roms,
    TokenizerGraphMaps,
)
from kllm.graph_optimizer import optimize_graph


class Processor:
    """Complete inference device — datapath + tokenizer graph + ROMs + config.

    The processor is the compiled chip: everything needed to run
    inference autonomously.  The tokenizer's vocabulary and merge
    tables are CONST nodes in the tokenizer CircuitGraph.
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
        tokenizer_graph: CircuitGraph | None = None,
        tokenizer_maps: TokenizerGraphMaps | None = None,
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
        self.tokenizer_graph = tokenizer_graph
        self.tokenizer_maps = tokenizer_maps

    @classmethod
    def build(
        cls,
        fabric: object,
        eos_token_id: int,
        max_seq: int = 2048,
        tokenizer_dir: str | None = None,
    ) -> "Processor":
        """Build a processor from a loaded Fabric.

        1. Compile single-token decode template (weights = CONST).
        2. Optimise the datapath graph.
        3. Extract embed table and RoPE tables.
        4. Compile tokenizer graph from JSON.
        5. Bundle into a Processor.

        Parameters
        ----------
        fabric : Fabric
            Loaded model weights and config.
        eos_token_id : int
            End-of-sequence token ID.
        max_seq : int
            Maximum sequence length.
        tokenizer_dir : str | None
            Path to the tokenizer directory (tokenizer.json + config).
            If provided, builds the circuit tokenizer graph directly.
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

        # 4. Compile tokenizer
        tokenizer_roms = None
        tokenizer_graph = None
        tokenizer_maps = None

        if tokenizer_dir is not None:
            tok_json = os.path.join(tokenizer_dir, "tokenizer.json")
            tok_cfg = os.path.join(tokenizer_dir, "tokenizer_config.json")
            if os.path.exists(tok_json) and os.path.exists(tok_cfg):
                print("  Compiling tokenizer circuit …")
                tokenizer_graph, tokenizer_maps = (
                    compile_tokenizer_graph_from_json(
                        tok_json, tok_cfg,
                        bos_token_id=None,
                        max_tokens=max_seq,
                    )
                )

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
            tokenizer_graph=tokenizer_graph,
            tokenizer_maps=tokenizer_maps,
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
              tokenizer_circuit/ — serialized tokenizer graph (new)
              tables/
                embed_table.npy
                rope_cos.npy
                rope_sin.npy
              tokenizer_roms/    — BPE ROMs (legacy, still saved for compat)
        """
        os.makedirs(chip_dir, exist_ok=True)

        # Datapath graph
        circuit_dir = os.path.join(chip_dir, "circuit")
        self.datapath.serialize(circuit_dir)

        # Tokenizer graph (new)
        if self.tokenizer_graph is not None:
            tok_circuit_dir = os.path.join(chip_dir, "tokenizer_circuit")
            self.tokenizer_graph.serialize(tok_circuit_dir)

        # Tables
        tables_dir = os.path.join(chip_dir, "tables")
        os.makedirs(tables_dir, exist_ok=True)
        np.save(os.path.join(tables_dir, "embed_table.npy"), self.embed_table)
        np.save(os.path.join(tables_dir, "rope_cos.npy"), self.rope_cos)
        np.save(os.path.join(tables_dir, "rope_sin.npy"), self.rope_sin)

        # Tokenizer ROMs (legacy path — still save for backward compat)
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

        # Save tokenizer maps if present
        if self.tokenizer_maps is not None:
            config["tokenizer_maps"] = {
                "byte_input": self.tokenizer_maps.byte_input,
                "byte_length": self.tokenizer_maps.byte_length,
                "token_ids": self.tokenizer_maps.token_ids,
                "num_tokens": self.tokenizer_maps.num_tokens,
                "dec_token_ids": self.tokenizer_maps.dec_token_ids,
                "dec_num_tokens": self.tokenizer_maps.dec_num_tokens,
                "dec_byte_output": self.tokenizer_maps.dec_byte_output,
                "dec_byte_length": self.tokenizer_maps.dec_byte_length,
            }

        with open(os.path.join(chip_dir, "processor.json"), "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, chip_dir: str) -> "Processor":
        """Load a processor from disk."""
        with open(os.path.join(chip_dir, "processor.json")) as f:
            config = json.load(f)

        datapath = CircuitGraph.deserialize(
            os.path.join(chip_dir, "circuit"))

        tables_dir = os.path.join(chip_dir, "tables")
        embed_table = np.load(os.path.join(tables_dir, "embed_table.npy"))
        rope_cos = np.load(os.path.join(tables_dir, "rope_cos.npy"))
        rope_sin = np.load(os.path.join(tables_dir, "rope_sin.npy"))

        # Load tokenizer graph (new path)
        tokenizer_graph = None
        tokenizer_maps = None
        tok_circuit_dir = os.path.join(chip_dir, "tokenizer_circuit")
        if os.path.isdir(tok_circuit_dir):
            tokenizer_graph = CircuitGraph.deserialize(tok_circuit_dir)
            tm = config.get("tokenizer_maps")
            if tm is not None:
                tokenizer_maps = TokenizerGraphMaps(
                    byte_input=tm["byte_input"],
                    byte_length=tm["byte_length"],
                    token_ids=tm["token_ids"],
                    num_tokens=tm["num_tokens"],
                    dec_token_ids=tm["dec_token_ids"],
                    dec_num_tokens=tm["dec_num_tokens"],
                    dec_byte_output=tm["dec_byte_output"],
                    dec_byte_length=tm["dec_byte_length"],
                )

        # Load tokenizer ROMs (legacy path)
        tokenizer_roms = None
        roms_dir = os.path.join(chip_dir, "tokenizer_roms")
        if os.path.isdir(roms_dir):
            tokenizer_roms = load_tokenizer_roms(roms_dir)

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
            tokenizer_graph=tokenizer_graph,
            tokenizer_maps=tokenizer_maps,
        )
