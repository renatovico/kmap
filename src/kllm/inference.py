"""Lossless Bit-Sliced Inference Engine.

Streams the pre-compiled logic fabric from disk through the GPU (or CPU)
one layer at a time so that VRAM never exceeds ~4 GB.  Accepts real
customer text via the HuggingFace tokenizer.
"""

import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kllm.bitops import extract_sub_masks
from kllm.device import free_vram, to_device, to_numpy, xp

_NUM_PLANES = 4


class LosslessInferenceEngine:
    def __init__(self, model_name: str, save_dir: str = "./lossless_logic"):
        self.save_dir = save_dir

        # Load metadata written by the compiler.
        meta_path = os.path.join(save_dir, "meta.npz")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"No compiled fabric found at {save_dir}. "
                "Run `kllm --mode compile` first."
            )
        meta = np.load(meta_path, allow_pickle=True)
        self.num_layers = int(meta["num_layers"])
        self.proj_names = list(meta["proj_names"])

        # Tokenizer + embedding table stay in system RAM (32 GB budget).
        print(f"[*] Loading tokenizer and embedding table for {model_name} …")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        self.embed_tokens = model.get_input_embeddings().weight.detach().numpy()
        self.num_layers = min(self.num_layers, len(model.model.layers))
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ------------------------------------------------------------------
    # Text → lossless bit-planes
    # ------------------------------------------------------------------
    def _text_to_planes(self, text: str) -> list[np.ndarray]:
        token_ids = self.tokenizer.encode(text, return_tensors="np")[0]
        embeddings = self.embed_tokens[token_ids]  # (seq_len, hidden_dim)  fp32
        return list(extract_sub_masks(embeddings))

    # ------------------------------------------------------------------
    # Single-layer logic execution (GPU / CPU)
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_logic_plane(plane, s1, mask):
        """(x << s1) ^ mask  —  SIMT parallel across the whole matrix."""
        return (plane << s1) ^ mask

    def _run_layer(self, planes: list, layer_idx: int) -> list:
        path = os.path.join(self.save_dir, f"layer_{layer_idx}.npz")
        data = np.load(path)

        # We process each projection independently and accumulate.
        accumulated = [xp.zeros_like(planes[p]) for p in range(_NUM_PLANES)]

        for proj_name in self.proj_names:
            key_prefix = f"{proj_name}_m"
            # Check if this projection was compiled for this layer.
            if f"{key_prefix}0_s1" not in data:
                continue

            for p in range(_NUM_PLANES):
                s1 = to_device(data[f"{key_prefix}{p}_s1"])
                mask = to_device(data[f"{key_prefix}{p}_mask"])
                result = self._apply_logic_plane(planes[p], s1, mask)
                accumulated[p] = accumulated[p] ^ result  # XOR accumulation
                del s1, mask

        del data
        free_vram()
        return accumulated

    # ------------------------------------------------------------------
    # Full forward pass
    # ------------------------------------------------------------------
    def run(self, text: str) -> np.ndarray:
        print(f"[*] Tokenizing: '{text}'")
        planes = self._text_to_planes(text)
        gpu_planes = [to_device(p) for p in planes]

        print(f"[*] Streaming {self.num_layers} layers through logic fabric …")
        t0 = time.perf_counter()

        for li in range(self.num_layers):
            gpu_planes = self._run_layer(gpu_planes, li)

        elapsed = time.perf_counter() - t0
        print(f"[+] Inference done in {elapsed:.4f}s")

        # Reconstruct IEEE-754 floats from the final bit-planes.
        result_uint32 = (
            gpu_planes[3].astype(xp.uint32) << 24
            | gpu_planes[2].astype(xp.uint32) << 16
            | gpu_planes[1].astype(xp.uint32) << 8
            | gpu_planes[0].astype(xp.uint32)
        )
        result_fp32 = result_uint32.view(xp.float32) if hasattr(result_uint32, "view") else result_uint32.view(np.float32)
        return to_numpy(result_fp32)
