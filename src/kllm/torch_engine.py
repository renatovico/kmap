"""PyTorch Inference Engine with KV-Cache and Sliding Window.

A new runtime backend that uses PyTorch (CPU or CUDA) instead of NumPy.
Supports:
- Device selection: ``cpu`` or ``cuda``
- Dtype selection: ``fp32``, ``bf16``, ``fp16``
- KV-cache for efficient autoregressive generation
- Configurable sliding window to bound KV-cache memory
"""

from __future__ import annotations

import time

import torch


_DTYPE_MAP: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


class TorchInferenceEngine:
    """PyTorch-based inference engine with KV-cache and sliding window.

    Loads the original HuggingFace model weights and runs forward /
    generation using torch tensors, keeping the same API as the NumPy
    engines so that CLI integration is seamless.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID or local path.
    device : str
        ``"cpu"`` or ``"cuda"`` (or any valid ``torch.device`` string).
    dtype : str
        One of ``"fp32"``, ``"bf16"``, ``"fp16"``.
    window : int | None
        Sliding-window size for KV-cache during generation.  When the
        cache exceeds *window* tokens the oldest entries are evicted.
        ``None`` disables the window (cache grows without bound).
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        dtype: str = "fp32",
        window: int | None = 131_072,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if dtype not in _DTYPE_MAP:
            raise ValueError(
                f"dtype must be one of {list(_DTYPE_MAP)}; got {dtype!r}"
            )

        self.device = torch.device(device)
        self.torch_dtype = _DTYPE_MAP[dtype]
        self.window = window

        print(f"[*] Loading tokenizer for {model_name} …")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"[*] Loading model {model_name} "
              f"(dtype={dtype}, device={device}) …")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
        )
        self.model.to(self.device)
        self.model.eval()
        print("[+] Model loaded.")

        # KV-cache state (set by prefill, updated by decode_one)
        self._kv_cache: tuple | None = None
        self._next_pos: int = 0  # absolute position index of the next token

    # ------------------------------------------------------------------
    # Full forward pass (no KV-cache — equivalent to NumPy engines)
    # ------------------------------------------------------------------
    def forward(self, token_ids: list[int]) -> "torch.Tensor":
        """Run a full non-cached forward pass.

        Returns
        -------
        torch.Tensor, shape ``(seq, vocab)``, always float32 on CPU regardless
        of the model's internal dtype.
        """
        input_ids = torch.tensor([token_ids], dtype=torch.long,
                                 device=self.device)
        with torch.no_grad():
            out = self.model(input_ids, use_cache=False)
        return out.logits[0].float().cpu()

    # ------------------------------------------------------------------
    # Text → logits convenience wrapper
    # ------------------------------------------------------------------
    def run(self, text: str) -> "torch.Tensor":
        """Tokenize *text*, run the model, return logits ``(seq, vocab)``."""
        print(f"[*] Tokenizing: '{text}'")
        token_ids = self.tokenizer.encode(text)
        print(f"[*] Running torch forward pass …")
        t0 = time.perf_counter()
        logits = self.forward(token_ids)
        elapsed = time.perf_counter() - t0
        print(f"[+] Torch inference done in {elapsed:.4f}s")
        return logits

    # ------------------------------------------------------------------
    # KV-cache prefill
    # ------------------------------------------------------------------
    def prefill(self, token_ids: list[int]) -> "torch.Tensor":
        """Run the prompt through the model and fill the KV-cache.

        If *window* is set, only the last *window* tokens are used.
        After this call the engine is ready for :meth:`decode_one`.

        Returns
        -------
        torch.Tensor, shape ``(vocab,)``, float32 on CPU — logits for the
        last prompt token (i.e., the prediction of the *first* new token).
        """
        if self.window is not None:
            token_ids = token_ids[-self.window :]

        input_ids = torch.tensor([token_ids], dtype=torch.long,
                                 device=self.device)
        with torch.no_grad():
            out = self.model(input_ids, use_cache=True)

        self._kv_cache = out.past_key_values
        self._next_pos = len(token_ids)
        return out.logits[0, -1].float().cpu()

    # ------------------------------------------------------------------
    # KV-cache decode step
    # ------------------------------------------------------------------
    def decode_one(self, token_id: int) -> "torch.Tensor":
        """Generate one new token using the existing KV-cache.

        Call :meth:`prefill` first.  If the KV-cache is longer than
        *window*, the oldest entries are evicted before the step.

        Returns
        -------
        torch.Tensor, shape ``(vocab,)``, float32 on CPU — logits for
        the new position (the prediction of the *next* token after
        *token_id*).
        """
        if self._kv_cache is None:
            raise RuntimeError("Call prefill() before decode_one().")

        # Sliding-window eviction
        if self.window is not None:
            # past_key_values: tuple of (key, value) per layer
            # key/value shape: (batch, kv_heads, cache_len, head_dim)
            cache_len = self._kv_cache[0][0].shape[2]
            if cache_len >= self.window:
                trim = cache_len - self.window + 1
                self._kv_cache = tuple(
                    (k[:, :, trim:, :].contiguous(),
                     v[:, :, trim:, :].contiguous())
                    for k, v in self._kv_cache
                )

        input_ids = torch.tensor([[token_id]], dtype=torch.long,
                                 device=self.device)
        position_ids = torch.tensor([[self._next_pos]], dtype=torch.long,
                                    device=self.device)

        with torch.no_grad():
            out = self.model(
                input_ids,
                past_key_values=self._kv_cache,
                use_cache=True,
                position_ids=position_ids,
            )

        self._kv_cache = out.past_key_values
        self._next_pos += 1
        return out.logits[0, -1].float().cpu()

    # ------------------------------------------------------------------
    # Greedy text generation (with KV-cache)
    # ------------------------------------------------------------------
    def generate(self, text: str, max_new_tokens: int = 50) -> str:
        """Greedy autoregressive generation using KV-cache.

        The prompt is run through :meth:`prefill` once; subsequent tokens
        each call :meth:`decode_one` — no recomputation of the prefix.

        Parameters
        ----------
        text : str
            Prompt text.
        max_new_tokens : int
            Maximum number of new tokens to generate.

        Returns
        -------
        str — the decoded text (prompt + generated tokens).
        """
        token_ids: list[int] = self.tokenizer.encode(text)
        eos = self.tokenizer.eos_token_id

        # Prefill
        logits = self.prefill(token_ids)
        next_id = int(logits.argmax())
        tok_str = self.tokenizer.decode([next_id])
        print(f"  [torch] token 1/{max_new_tokens}: {next_id} ({tok_str!r})")
        if next_id == eos:
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        token_ids.append(next_id)

        # Decode loop
        for i in range(1, max_new_tokens):
            t0 = time.perf_counter()
            logits = self.decode_one(next_id)
            elapsed = time.perf_counter() - t0
            next_id = int(logits.argmax())
            tok_str = self.tokenizer.decode([next_id])
            print(f"  [torch] token {i + 1}/{max_new_tokens}: "
                  f"{next_id} ({tok_str!r}) in {elapsed:.3f}s")
            if next_id == eos:
                break
            token_ids.append(next_id)

        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
