"""Z3 Gate Inference Engine.

Thin orchestrator that wires together:

- :class:`~kllm.tokenizer.Tokenizer` — pure-Python BPE (no HuggingFace)
- :class:`~kllm.fabric.Fabric` — Z3 gate loader (shift + XOR → float32)
- :class:`~kllm.circuit_model.CircuitTransformer` — LLaMA with Z3 circuits
- :class:`~kllm.circuits.ArithmeticUnit` — Z3-compiled activation functions

Supports two execution modes:

1. **Blocking** — ``generate()`` returns the full output string.
2. **Streaming** — ``stream()`` is a generator that yields each token
   as it is produced, and ``stream_layers()`` yields per-layer progress.

No ``torch`` or ``transformers`` dependency at inference time.
"""

from __future__ import annotations

import os
import time
from collections.abc import Generator

from kllm.circuit_model import CircuitTransformer
from kllm.circuits import ArithmeticUnit
from kllm.fabric import Fabric
from kllm.tokenizer import Tokenizer


class BitLogicInferenceEngine:
    """End-to-end inference engine backed by Z3-solved gate fabric.

    Loads the compiled fabric, optionally loads Z3 arithmetic circuits
    for activation functions, and runs greedy auto-regressive generation
    with a KV cache.

    If ``circuits.npz`` is present in *save_dir*, all activations
    (SiLU, exp, rsqrt) run through Z3 gate LUTs — the entire
    inference pipeline becomes pure shift+XOR operations.
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
        print("[*] Loading Z3 gate fabric …")
        self.fabric = Fabric(save_dir)
        print(
            f"[+] Weights reconstructed from Z3 gates in "
            f"{self.fabric.load_time:.1f}s "
            f"({self.fabric.num_layers} layers, 7 projections each)"
        )

        # ---- Arithmetic circuits (optional) ----
        circuit_path = os.path.join(save_dir, "circuits.npz")
        unit: ArithmeticUnit | None = None
        if os.path.exists(circuit_path):
            unit = ArithmeticUnit.load(circuit_path)
            ops = list(unit.ops.keys())
            print(f"[+] Z3 arithmetic circuits loaded: {', '.join(ops)}")
        else:
            print("[*] No circuits.npz — activations use NumPy fallback")

        # ---- Transformer (circuit-based, with generators) ----
        self.model = CircuitTransformer(self.fabric, unit)

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
    # Streaming generators
    # ------------------------------------------------------------------
    def stream(
        self, text: str, max_new_tokens: int = 50,
    ) -> Generator[str, None, None]:
        """Generator that yields each decoded token as it is produced.

        Usage::

            for tok in engine.stream("Hello", max_new_tokens=20):
                print(tok, end="", flush=True)
        """
        token_ids = self.tokenizer.encode(text)
        eos = self.tokenizer.eos_token_id

        self.reset_cache()

        # Prefill
        logits = self.forward(token_ids, start_pos=0)
        next_id = int(logits[-1].argmax())
        if next_id == eos:
            return
        token_ids.append(next_id)
        yield self.tokenizer.decode([next_id], skip_special_tokens=False)

        # Decode
        for _ in range(1, max_new_tokens):
            pos = len(token_ids) - 1
            logits = self.forward([token_ids[-1]], start_pos=pos)
            next_id = int(logits[-1].argmax())
            if next_id == eos:
                return
            token_ids.append(next_id)
            yield self.tokenizer.decode([next_id], skip_special_tokens=False)

    def stream_layers(
        self, text: str, max_new_tokens: int = 50,
    ) -> Generator[tuple[str, int | None, "np.ndarray | str"], None, None]:
        """Generator that yields per-layer progress and per-token output.

        Yields ``(stage, layer_idx_or_token_num, data)`` tuples:

        - ``("embed", None, hidden)``
        - ``("layer", layer_idx, hidden)``
        - ``("norm", None, hidden)``
        - ``("logits", None, logits)``
        - ``("token", token_num, decoded_str)``
        """
        token_ids = self.tokenizer.encode(text)
        eos = self.tokenizer.eos_token_id

        self.reset_cache()

        # Prefill — stream layer by layer
        for stage, li, data in self.model.forward_gen(token_ids, start_pos=0):
            yield (stage, li, data)
            if stage == "logits":
                logits = data

        next_id = int(logits[-1].argmax())
        if next_id == eos:
            return
        token_ids.append(next_id)
        tok_str = self.tokenizer.decode([next_id], skip_special_tokens=False)
        yield ("token", 1, tok_str)

        # Decode — stream each token + layers
        for i in range(1, max_new_tokens):
            pos = len(token_ids) - 1
            for stage, li, data in self.model.forward_gen(
                [token_ids[-1]], start_pos=pos,
            ):
                yield (stage, li, data)
                if stage == "logits":
                    logits = data

            next_id = int(logits[-1].argmax())
            if next_id == eos:
                return
            token_ids.append(next_id)
            tok_str = self.tokenizer.decode(
                [next_id], skip_special_tokens=False,
            )
            yield ("token", i + 1, tok_str)

    # ------------------------------------------------------------------
    # Blocking generation (uses streaming internally)
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
