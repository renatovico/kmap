"""Chip — user-facing compiled model artifact.

A Chip is a self-contained directory containing everything needed to
run inference.  The Processor IS the chip — it contains the datapath,
all ROMs (embedding, RoPE, tokenizer vocabulary + merges), and config.

The chip is the unit the user creates, loads, and invokes::

    kllm create --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 ./mychip
    kllm infer ./mychip --max-tokens 50 Hello world

Internally, a Chip wraps:
- ``Processor`` — the complete device (datapath + ROMs + tokenizer ROMs)
- ``Tokenizer`` — reference BPE implementation (for encoding prompts)

The Processor's on-chip tokenizer ROMs handle detokenization directly
(ROM lookup from token_id → bytes).  The reference Tokenizer is kept
for encoding prompts and chat template rendering, but the core data
lives in the Processor's ROMs.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from kllm.processor import Processor, NativeRunner
from kllm.tokenizer import Tokenizer


class Chip:
    """A compiled chip — Processor + reference Tokenizer.

    Create with ``Chip.create()`` (downloads + compiles), or load an
    existing one with ``Chip.load()``.
    """

    def __init__(
        self,
        path: str,
        processor: Processor,
        tokenizer: Tokenizer,
    ) -> None:
        self.path = path
        self.processor = processor
        self.tokenizer = tokenizer
        self._runner: NativeRunner | None = None

    @classmethod
    def create(
        cls,
        model_name: str,
        chip_path: str,
    ) -> "Chip":
        """Download a HuggingFace model and compile it into a chip.

        This is the all-in-one command: download weights, extract
        tokenizer, compile the decode datapath, compile tokenizer ROMs,
        optimise, and serialize everything to ``chip_path``.
        """
        from kllm.fabric import Fabric

        os.makedirs(chip_path, exist_ok=True)

        # 1. Download and extract weights + tokenizer
        print(f"[chip] Downloading {model_name} …")
        fabric = Fabric.from_pretrained(model_name, chip_path)

        # 2. Load tokenizer
        tok_dir = os.path.join(chip_path, "tokenizer")
        tokenizer = Tokenizer(tok_dir)

        # 3. Build the processor — datapath + all ROMs (including tokenizer)
        print("[chip] Building processor …")
        processor = Processor.build(
            fabric, tokenizer.eos_token_id, tokenizer=tokenizer,
        )

        # 4. Save processor artifacts
        print("[chip] Saving processor …")
        processor.save(chip_path)

        # 5. Write chip metadata
        metadata = {
            "kllm_version": "0.1.0",
            "model_name": model_name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "eos_token_id": tokenizer.eos_token_id,
            "max_seq_len": processor.max_seq_len,
            "vocab_size": processor.vocab_size,
            "hidden_dim": processor.hidden_dim,
            "num_layers": processor.num_layers,
            "head_dim": processor.head_dim,
        }
        with open(os.path.join(chip_path, "chip.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[chip] Done — saved to {chip_path}")
        return cls(path=chip_path, processor=processor, tokenizer=tokenizer)

    @classmethod
    def load(cls, chip_path: str) -> "Chip":
        """Load a previously compiled chip from disk."""
        chip_json = os.path.join(chip_path, "chip.json")
        if not os.path.exists(chip_json):
            raise FileNotFoundError(
                f"No chip found at {chip_path!r}. "
                "Run `kllm create --model <model> <path>` first."
            )

        processor = Processor.load(chip_path)

        # Load reference tokenizer for encoding (if present)
        tok_dir = os.path.join(chip_path, "tokenizer")
        tokenizer = Tokenizer(tok_dir) if os.path.isdir(tok_dir) else None
        return cls(path=chip_path, processor=processor, tokenizer=tokenizer)

    @staticmethod
    def exists(chip_path: str) -> bool:
        """Check whether a compiled chip exists at the given path."""
        return os.path.exists(os.path.join(chip_path, "chip.json"))

    def _get_runner(self) -> NativeRunner:
        """Lazy-init the native runner (the virtual device)."""
        if self._runner is None:
            self._runner = NativeRunner(self.processor)
        return self._runner

    def infer(self, prompt: str, max_tokens: int = 50) -> str:
        """Run inference: text in → text out through the device.

        The full loop (tokenize → prefill → decode → detokenize) runs
        through the processor — the same operations a hardware FSM
        would perform.

        Detokenization uses the on-chip ROMs when available.
        """
        messages = [{"role": "user", "content": prompt}]
        prompt_str = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True,
        )
        token_ids = self.tokenizer.encode(prompt_str)

        runner = self._get_runner()
        generated = runner.infer(token_ids, max_tokens)

        # Decode using on-chip ROMs if available, else reference tokenizer
        if self.processor.tokenizer_roms:
            return runner.decode_tokens(generated, skip_special=True)
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def infer_streaming(self, prompt: str, max_tokens: int = 50):
        """Yield generated text chunks as they're produced."""
        messages = [{"role": "user", "content": prompt}]
        prompt_str = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True,
        )
        token_ids = self.tokenizer.encode(prompt_str)

        runner = self._get_runner()
        for tok_id in runner.infer_streaming(token_ids, max_tokens):
            if self.processor.tokenizer_roms:
                yield runner.decode_tokens([tok_id], skip_special=True)
            else:
                yield self.tokenizer.decode(
                    [tok_id], skip_special_tokens=True,
                )
