"""Z3 Gate Inference Engine.

Thin orchestrator that wires together:

- :class:`~kllm.tokenizer.Tokenizer` — pure-Python BPE (no HuggingFace)
- :class:`~kllm.fabric.Fabric` — Z3 gate loader (shift + XOR → float32)
- :class:`~kllm.model.Transformer` — LLaMA-style forward pass with KV cache

No ``torch`` or ``transformers`` dependency at inference time.
"""

import os
import time

from kllm.fabric import Fabric
from kllm.model import Transformer
from kllm.tokenizer import Tokenizer


class BitLogicInferenceEngine:
    """End-to-end inference engine backed by Z3-solved gate fabric.

    Loads the compiled fabric, reconstructs float32 weights via pure
    bit operations, and runs greedy auto-regressive generation with
    a KV cache.  Uses a pure-Python BPE tokenizer — no HuggingFace
    dependency at inference time.
    """

    def __init__(self, save_dir: str = "./lossless_logic") -> None:
        self.save_dir = save_dir

        # ---- Tokenizer (pure Python) ----
        tok_dir = os.path.join(save_dir, "tokenizer")
        if not os.path.isdir(tok_dir):
            raise FileNotFoundError(
                f"No tokenizer found at {tok_dir}. "
                "Run `kllm --mode compile` to bundle it."
            )
        self.tokenizer = Tokenizer(tok_dir)

        # ---- Fabric (Z3 gates → float32 weights) ----
        print(f"[*] Loading Z3 gate fabric …")
        self.fabric = Fabric(save_dir)
        print(
            f"[+] Weights reconstructed from Z3 gates in "
            f"{self.fabric.load_time:.1f}s "
            f"({self.fabric.num_layers} layers, 7 projections each)"
        )

        # ---- Transformer (forward pass + KV cache) ----
        self.model = Transformer(self.fabric)

    # ------------------------------------------------------------------
    def reset_cache(self) -> None:
        """Clear the KV cache (call before a new sequence)."""
        self.model.reset_cache()

    # ------------------------------------------------------------------
    def forward(
        self, token_ids: list[int], start_pos: int = 0,
    ) -> "np.ndarray":
        """Run the transformer and return logits."""
        return self.model.forward(token_ids, start_pos)

    # ------------------------------------------------------------------
    def run(self, text: str) -> "np.ndarray":
        """Tokenize *text*, run inference, return logits."""
        token_ids = self.tokenizer.encode(text)
        self.reset_cache()
        t0 = time.perf_counter()
        logits = self.forward(token_ids)
        elapsed = time.perf_counter() - t0
        print(f"[+] Inference done in {elapsed:.2f}s")
        return logits

    # ------------------------------------------------------------------
    def generate(self, text: str, max_new_tokens: int = 50) -> str:
        """Greedy auto-regressive generation with KV cache."""
        token_ids = self.tokenizer.encode(text)
        eos = self.tokenizer.eos_token_id

        self.reset_cache()

        # Prefill — process full prompt
        t0 = time.perf_counter()
        logits = self.forward(token_ids, start_pos=0)
        elapsed = time.perf_counter() - t0
        next_id = int(logits[-1].argmax())
        tok_str = self.tokenizer.decode([next_id], skip_special_tokens=False)
        print(
            f"  [prefill] {len(token_ids)} tok → "
            f"{next_id} ({tok_str!r}) in {elapsed:.2f}s"
        )

        if next_id == eos:
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        token_ids.append(next_id)

        # Decode — one token at a time
        for i in range(1, max_new_tokens):
            t0 = time.perf_counter()
            pos = len(token_ids) - 1
            logits = self.forward([token_ids[-1]], start_pos=pos)
            elapsed = time.perf_counter() - t0
            next_id = int(logits[-1].argmax())
            tok_str = self.tokenizer.decode(
                [next_id], skip_special_tokens=False,
            )
            print(
                f"  [decode] token {i + 1}/{max_new_tokens}: "
                f"{next_id} ({tok_str!r}) in {elapsed:.2f}s"
            )
            if next_id == eos:
                break
            token_ids.append(next_id)

        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
