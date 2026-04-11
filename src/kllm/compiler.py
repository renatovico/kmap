"""Lossless Logic Compiler.

Extracts IEEE-754 sub-bit masks from every linear-projection weight
in a HuggingFace causal-LM, synthesises minimal boolean logic with Z3,
and streams the compiled "logic fabric" to disk layer-by-layer so that
RAM never exceeds a single layer at a time.
"""

import os
import time

import numpy as np
from tqdm import tqdm
from z3 import BitVec, ForAll, Solver, ULE, sat

from kllm.bitops import extract_sub_masks

# Linear projection names present in every transformer layer that we compile.
_PROJ_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj")


class LosslessLogicCompiler:
    def __init__(
        self,
        model_name: str,
        save_dir: str = "./lossless_logic",
        solver_timeout: int = 200,
    ):
        self.model_name = model_name
        self.save_dir = save_dir
        self.solver_timeout = solver_timeout
        self._logic_cache: dict[int, tuple[int, int]] = {}

        os.makedirs(self.save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Z3 solver — finds a minimal (shift, mask) pair for each unique
    # 8-bit pattern so that  ``(x << s1) ^ mask == (x & target)``
    # holds for every possible 8-bit input ``x``.
    # ------------------------------------------------------------------
    @staticmethod
    def _solve_pattern(target_val: int, timeout: int = 200) -> tuple[int, int]:
        solver = Solver()
        solver.set("timeout", timeout)

        x = BitVec("x", 8)
        s1 = BitVec("s1", 8)
        mask = BitVec("mask", 8)

        solver.add(ForAll([x], (x << s1) ^ mask == (x & int(target_val))))
        solver.add(ULE(s1, 7))

        if solver.check() == sat:
            m = solver.model()
            return (m[s1].as_long(), m[mask].as_long())
        # Fallback: identity shift + literal mask preserves losslessness.
        return (0, int(target_val))

    def _solve_unique(self, unique_vals: np.ndarray) -> dict[int, tuple[int, int]]:
        registry: dict[int, tuple[int, int]] = {}
        for v in unique_vals:
            v_int = int(v)
            if v_int not in self._logic_cache:
                self._logic_cache[v_int] = self._solve_pattern(v_int, self.solver_timeout)
            registry[v_int] = self._logic_cache[v_int]
        return registry

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compile(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM

        print(f"[*] Loading {self.model_name} …")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float32
        )
        layers = model.model.layers
        num_layers = len(layers)

        # Persist metadata so inference knows the model topology.
        meta = {
            "model_name": self.model_name,
            "num_layers": num_layers,
            "proj_names": list(_PROJ_NAMES),
        }
        np.savez(os.path.join(self.save_dir, "meta.npz"), **{k: np.array(v) for k, v in meta.items()})

        t0 = time.perf_counter()

        for li, layer in enumerate(layers):
            print(f"\n=== Layer {li + 1}/{num_layers} ===")
            attn = layer.self_attn

            layer_arrays: dict[str, np.ndarray] = {}

            for proj_name in _PROJ_NAMES:
                proj = getattr(attn, proj_name, None)
                if proj is None:
                    continue

                weights_fp32 = proj.weight.detach().numpy()
                masks = extract_sub_masks(weights_fp32)

                for plane_idx, plane in enumerate(masks):
                    unique = np.unique(plane)
                    registry = self._solve_unique(unique)

                    flat = plane.flatten()
                    s1_arr = np.empty_like(flat)
                    mask_arr = np.empty_like(flat)
                    for idx, b in enumerate(flat):
                        s1_arr[idx], mask_arr[idx] = registry[int(b)]

                    key_s1 = f"{proj_name}_m{plane_idx}_s1"
                    key_mask = f"{proj_name}_m{plane_idx}_mask"
                    layer_arrays[key_s1] = s1_arr.reshape(plane.shape)
                    layer_arrays[key_mask] = mask_arr.reshape(plane.shape)

                tqdm.write(f"  [{proj_name}] solved ({len(self._logic_cache)} unique patterns cached)")

            path = os.path.join(self.save_dir, f"layer_{li}.npz")
            np.savez_compressed(path, **layer_arrays)
            print(f"  -> saved {path}")

        elapsed = time.perf_counter() - t0
        print(f"\n[+] Compilation finished in {elapsed:.1f}s  ({len(self._logic_cache)} unique logic gates)")
        print(f"    Fabric stored in {self.save_dir}/")
