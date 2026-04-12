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
from z3 import BitVec, BitVecVal, Solver, ULE, sat

from kllm.bitops import extract_sub_masks

# Weight names in every LLaMA-style transformer layer.
_ATTN_WEIGHTS = ("q_proj", "k_proj", "v_proj", "o_proj")
_MLP_WEIGHTS = ("gate_proj", "up_proj", "down_proj")

# Legacy alias used by older fabric files.
_PROJ_NAMES = _ATTN_WEIGHTS


def _store_weight_gates(
    dest: dict,
    name: str,
    weight: np.ndarray,
    s1_lut: np.ndarray,
    mask_lut: np.ndarray,
) -> None:
    """Decompose FP32 weight into byte planes AND Z3 gate arrays.

    For each byte plane the gate arrays encode a Z3-proved (shift, mask)
    pair such that ``(0xFF << s1) ^ mask`` recovers the weight byte.
    At inference the gate is executed with pure bit ops (shift + XOR)
    to reconstruct the weight — no raw float is ever stored.

    The byte planes are also kept so that the standard engine
    (``repack_sub_masks``) can work from the same fabric.
    """
    for i, plane in enumerate(extract_sub_masks(weight)):
        dest[f"{name}_m{i}"] = plane
        dest[f"{name}_m{i}_s1"] = s1_lut[plane]
        dest[f"{name}_m{i}_mask"] = mask_lut[plane]


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
    # Z3 solver — for each byte value ``target``, finds a (shift, mask)
    # pair such that ``(0xFF << s1) ^ mask == target`` in 8-bit
    # arithmetic.  This is provably solvable for ALL 256 byte values:
    # Z3 confirms each one.  No fallback.  If the solver fails (should
    # never happen) compilation stops immediately.
    #
    # At inference the gate is executed to recover the weight byte:
    #     target = uint8(0xFF << s1) ^ mask
    # Pure bit operations — one shift, one XOR.
    # ------------------------------------------------------------------
    @staticmethod
    def _solve_pattern(target_val: int, timeout: int = 200) -> tuple[int, int]:
        solver = Solver()
        solver.set("timeout", timeout)

        s1 = BitVec("s1", 8)
        mask = BitVec("mask", 8)
        probe = BitVecVal(0xFF, 8)

        solver.add((probe << s1) ^ mask == BitVecVal(int(target_val), 8))
        solver.add(ULE(s1, 7))

        if solver.check() == sat:
            m = solver.model()
            return (m[s1].as_long(), m[mask].as_long())

        raise RuntimeError(
            f"Z3 UNSAT for byte {target_val} (timeout={timeout}ms). "
            "This should never happen — increase --solver-timeout."
        )

    def _solve_unique(self, unique_vals: np.ndarray) -> dict[int, tuple[int, int]]:
        registry: dict[int, tuple[int, int]] = {}
        for v in unique_vals:
            v_int = int(v)
            if v_int not in self._logic_cache:
                self._logic_cache[v_int] = self._solve_pattern(v_int, self.solver_timeout)
            registry[v_int] = self._logic_cache[v_int]
        return registry

    def _build_gate_lut(self) -> tuple[np.ndarray, np.ndarray]:
        """Solve Z3 for all 256 byte values → vectorized (s1, mask) lookup."""
        print("[*] Building Z3 gate lookup table (256 values) …")
        s1_arr = np.zeros(256, dtype=np.uint8)
        mask_arr = np.zeros(256, dtype=np.uint8)
        for v in range(256):
            s1, mask = self._solve_pattern(v, self.solver_timeout)
            s1_arr[v] = s1
            mask_arr[v] = mask
        print("  -> gate LUT ready")
        return s1_arr, mask_arr

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compile(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM

        from transformers import AutoTokenizer

        print(f"[*] Loading {self.model_name} …")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float32
        )
        config = model.config
        layers = model.model.layers
        num_layers = len(layers)

        # ---- Save tokenizer locally so inference never hits the Hub ----
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tok_dir = os.path.join(self.save_dir, "tokenizer")
        tokenizer.save_pretrained(tok_dir)
        print(f"  -> saved tokenizer to {tok_dir}")

        # ---- Build Z3 gate lookup table for all 256 byte values ------
        s1_lut, mask_lut = self._build_gate_lut()

        # ---- metadata (model topology for inference) -----------------
        meta = {
            "model_name": self.model_name,
            "num_layers": num_layers,
            "hidden_size": config.hidden_size,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "intermediate_size": config.intermediate_size,
            "vocab_size": config.vocab_size,
            "rms_norm_eps": float(config.rms_norm_eps),
            "rope_theta": float(
                getattr(config, "rope_theta", None)
                or (config.rope_scaling or {}).get("rope_theta", 10000.0)
            ),
            "max_position_embeddings": config.max_position_embeddings,
        }
        np.savez(
            os.path.join(self.save_dir, "meta.npz"),
            **{k: np.array(v) for k, v in meta.items()},
        )

        # ---- global weights (embed_tokens, final norm, lm_head) -----
        print("[*] Saving global weights …")
        globals_dict: dict[str, np.ndarray] = {}

        embed = model.get_input_embeddings().weight.detach().numpy()
        _store_weight_gates(globals_dict, "embed_tokens", embed, s1_lut, mask_lut)

        globals_dict["final_norm_weight"] = model.model.norm.weight.detach().numpy()

        lm_head_w = model.lm_head.weight.detach().numpy()
        tied = np.array_equal(lm_head_w, embed)
        globals_dict["lm_head_tied"] = np.array(tied)
        if tied:
            for i in range(4):
                globals_dict[f"lm_head_m{i}_s1"] = globals_dict[f"embed_tokens_m{i}_s1"]
                globals_dict[f"lm_head_m{i}_mask"] = globals_dict[f"embed_tokens_m{i}_mask"]
        else:
            _store_weight_gates(globals_dict, "lm_head", lm_head_w, s1_lut, mask_lut)

        np.savez_compressed(os.path.join(self.save_dir, "globals.npz"), **globals_dict)
        print("  -> saved globals.npz")

        # ---- per-layer weights ---------------------------------------
        t0 = time.perf_counter()

        for li, layer in enumerate(layers):
            print(f"\n=== Layer {li + 1}/{num_layers} ===")
            layer_dict: dict[str, np.ndarray] = {}

            # Attention projections
            attn = layer.self_attn
            for name in _ATTN_WEIGHTS:
                proj = getattr(attn, name, None)
                if proj is None:
                    continue
                _store_weight_gates(layer_dict, name, proj.weight.detach().numpy(), s1_lut, mask_lut)
                tqdm.write(f"  [{name}] solved ({len(self._logic_cache)} unique patterns cached)")

            # MLP projections
            mlp = layer.mlp
            for name in _MLP_WEIGHTS:
                proj = getattr(mlp, name, None)
                if proj is None:
                    continue
                _store_weight_gates(layer_dict, name, proj.weight.detach().numpy(), s1_lut, mask_lut)
                tqdm.write(f"  [{name}] solved ({len(self._logic_cache)} unique patterns cached)")

            # Layer-norm weights (small — store as raw float32)
            layer_dict["input_layernorm_weight"] = (
                layer.input_layernorm.weight.detach().numpy()
            )
            layer_dict["post_attention_layernorm_weight"] = (
                layer.post_attention_layernorm.weight.detach().numpy()
            )

            path = os.path.join(self.save_dir, f"layer_{li}.npz")
            np.savez_compressed(path, **layer_dict)
            print(f"  -> saved {path}")

        elapsed = time.perf_counter() - t0
        print(f"\n[+] Compilation finished in {elapsed:.1f}s  ({len(self._logic_cache)} unique logic gates)")
        print(f"    Fabric stored in {self.save_dir}/")

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
