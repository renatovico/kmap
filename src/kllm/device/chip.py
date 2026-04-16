"""Chip — user-facing compiled model artifact.

A Chip is a self-contained directory containing everything needed to
run inference: the Processor (datapath + tokenizer circuit + ROMs).

The chip takes raw UTF-8 bytes in and produces UTF-8 bytes out —
everything (tokenization, inference, detokenization) runs through
circuit graphs on the device.

Usage::

    kllm create --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 ./mychip
    kllm infer ./mychip --max-tokens 50 Hello world

Internally, a Chip wraps:
- ``Processor`` — the complete device (datapath + tokenizer circuit + config)
- ``NativeRunner`` — CPU simulation of the chip

Chat template rendering is a simple string formatter — not a circuit op.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from kllm.device.processor import Processor
from kllm.device.native_runner import NativeRunner


# ------------------------------------------------------------------
# Chat template formatting (simple string formatter, not a circuit op)
# ------------------------------------------------------------------

_CHAT_ROLE_TAGS = {
    "user": "<|user|>",
    "system": "<|system|>",
    "assistant": "<|assistant|>",
}

# Default EOS token (TinyLlama format)
_DEFAULT_EOS = "</s>"


def format_chat(
    messages: list[dict[str, str]],
    eos_token: str = _DEFAULT_EOS,
    add_generation_prompt: bool = False,
) -> str:
    """Render a chat conversation into a prompt string (TinyLlama format).

    This is a pure string operation — no tokenization involved.
    """
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        tag = _CHAT_ROLE_TAGS.get(role, f"<|{role}|>")
        parts.append(f"{tag}\n{msg['content']}{eos_token}\n")
    if add_generation_prompt:
        parts.append("<|assistant|>\n")
    return "".join(parts)


class Chip:
    """A compiled chip — Processor with circuit tokenizer.

    Create with ``Chip.create()`` (downloads + compiles), or load an
    existing one with ``Chip.load()``.
    """

    def __init__(
        self,
        path: str,
        processor: Processor,
    ) -> None:
        self.path = path
        self.processor = processor
        self._runner: NativeRunner | None = None

    @classmethod
    def create(
        cls,
        model_name: str,
        chip_path: str,
    ) -> "Chip":
        """Download a HuggingFace model and compile it into a chip.

        This is the all-in-one command: download weights, extract
        tokenizer, compile the decode datapath, compile tokenizer
        circuit, optimise, and serialize everything to ``chip_path``.
        """
        from kllm.compiler.fabric import Fabric

        os.makedirs(chip_path, exist_ok=True)

        # 1. Download and extract weights + tokenizer
        print(f"[chip] Downloading {model_name} …")
        fabric = Fabric.from_pretrained(model_name, chip_path)

        # 2. Determine EOS token ID from tokenizer config
        tok_dir = os.path.join(chip_path, "tokenizer")
        tok_cfg_path = os.path.join(tok_dir, "tokenizer_config.json")
        eos_token_id = 2  # default
        if os.path.exists(tok_cfg_path):
            with open(tok_cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
            # Load vocab to resolve EOS token string → ID
            tok_json_path = os.path.join(tok_dir, "tokenizer.json")
            if os.path.exists(tok_json_path):
                with open(tok_json_path, encoding="utf-8") as f:
                    tok_data = json.load(f)
                vocab = tok_data["model"]["vocab"]
                eos_str = cfg.get("eos_token", "</s>")
                eos_token_id = vocab.get(eos_str, 2)

        # 3. Build the processor — datapath + tokenizer circuit
        print("[chip] Building processor …")
        processor = Processor.build(
            fabric, eos_token_id, tokenizer_dir=tok_dir,
        )

        # 4. Save processor artifacts
        print("[chip] Saving processor …")
        processor.save(chip_path)

        # 5. Write chip metadata
        metadata = {
            "kllm_version": "0.2.0",
            "model_name": model_name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "eos_token_id": eos_token_id,
            "max_seq_len": processor.max_seq_len,
            "vocab_size": processor.vocab_size,
            "hidden_dim": processor.hidden_dim,
            "num_layers": processor.num_layers,
            "head_dim": processor.head_dim,
        }
        with open(os.path.join(chip_path, "chip.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[chip] Done — saved to {chip_path}")
        return cls(path=chip_path, processor=processor)

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
        return cls(path=chip_path, processor=processor)

    def _get_runner(self) -> NativeRunner:
        """Lazy-init the native runner (the virtual device)."""
        if self._runner is None:
            self._runner = NativeRunner(self.processor)
        return self._runner

    def infer(self, prompt: str, max_tokens: int = 50) -> str:
        """Run inference: text in → text out through the device.

        The prompt goes through chat template formatting, then the full
        circuit pipeline: BPE encode → prefill → decode → BPE decode.

        Returns the generated text as a string.
        """
        messages = [{"role": "user", "content": prompt}]
        prompt_str = format_chat(messages, add_generation_prompt=True)

        runner = self._get_runner()

        if self.processor.tokenizer_graph is None:
            raise RuntimeError(
                "No circuit tokenizer found in this chip. "
                "Re-create the chip with `kllm create` to compile the "
                "tokenizer circuit."
            )

        prompt_bytes = prompt_str.encode("utf-8")
        result_bytes = runner.infer_bytes(prompt_bytes, max_tokens)
        return result_bytes.decode("utf-8", errors="replace")

    def infer_streaming(self, prompt: str, max_tokens: int = 50):
        """Yield generated text chunks as they're produced."""
        messages = [{"role": "user", "content": prompt}]
        prompt_str = format_chat(messages, add_generation_prompt=True)

        runner = self._get_runner()

        if self.processor.tokenizer_graph is None:
            raise RuntimeError(
                "No circuit tokenizer found in this chip. "
                "Re-create the chip with `kllm create` to compile the "
                "tokenizer circuit."
            )

        prompt_bytes = prompt_str.encode("utf-8")
        for chunk_bytes in runner.infer_bytes_streaming(
                prompt_bytes, max_tokens):
            yield chunk_bytes.decode("utf-8", errors="replace")
