"""Z3 gate fabric loader.

Reads the compiled ``.npz`` files from disk, executes the Z3 gate
pairs (shift + XOR) to recover float32 weights, and exposes them
as plain NumPy arrays.

This module is the *only* place that knows about the on-disk layout
(``meta.npz``, ``globals.npz``, ``layer_<n>.npz``).
"""

import os
import time

import numpy as np

_NUM_PLANES = 4

# Probe byte used to recover target: uint8(0xFF << s1) ^ mask == target.
_Z3_PROBE = np.uint8(0xFF)


def _z3_reconstruct_weight(
    gates: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> np.ndarray:
    """Fused Z3 gate execution + zero-copy float assembly.

    Executes all four gate pairs and writes the recovered bytes directly
    into a contiguous ``(…, 4)`` uint8 buffer, then reinterprets as
    float32 via ``view`` — no shift/OR arithmetic for the repack step.

    Pure bit operations: shift → XOR → view.
    """
    shape = gates[0][0].shape
    buf = np.empty(shape + (4,), dtype=np.uint8)
    for i, (s1, mask) in enumerate(gates):
        buf[..., i] = (_Z3_PROBE << s1).astype(np.uint8) ^ mask
    return buf.view(np.float32).reshape(shape)


LINEAR_NAMES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


class Fabric:
    """Compiled Z3 gate fabric — loads and reconstructs model weights.

    On construction, executes every Z3 gate pair (shift + XOR) to
    recover the original float32 weights.  Gate arrays are then
    discarded, leaving only cached float32 tensors in RAM.
    """

    def __init__(self, save_dir: str) -> None:
        self.save_dir = save_dir

        # ---- Metadata ----
        meta_path = os.path.join(save_dir, "meta.npz")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"No compiled fabric found at {save_dir}. "
                "Run `kllm --mode compile` first."
            )
        meta = np.load(meta_path, allow_pickle=True)
        self.num_layers: int = int(meta["num_layers"])
        self.hidden_size: int = int(meta["hidden_size"])
        self.num_heads: int = int(meta["num_attention_heads"])
        self.num_kv_heads: int = int(meta["num_key_value_heads"])
        self.intermediate_size: int = int(meta["intermediate_size"])
        self.vocab_size: int = int(meta["vocab_size"])
        self.rms_norm_eps: float = float(meta["rms_norm_eps"])
        self.rope_theta: float = float(meta["rope_theta"])
        self.head_dim: int = self.hidden_size // self.num_heads
        self.num_groups: int = self.num_heads // self.num_kv_heads

        # ---- Execute Z3 gates → float32 ----
        t0 = time.perf_counter()

        gdata = np.load(
            os.path.join(save_dir, "globals.npz"), allow_pickle=True,
        )
        self.embed_tokens: np.ndarray = _z3_reconstruct_weight(tuple(
            (np.array(gdata[f"embed_tokens_m{i}_s1"]),
             np.array(gdata[f"embed_tokens_m{i}_mask"]))
            for i in range(_NUM_PLANES)
        ))
        self.final_norm_weight: np.ndarray = np.array(
            gdata["final_norm_weight"],
        )

        if bool(gdata["lm_head_tied"]):
            self.lm_head: np.ndarray = self.embed_tokens
        else:
            self.lm_head = _z3_reconstruct_weight(tuple(
                (np.array(gdata[f"lm_head_m{i}_s1"]),
                 np.array(gdata[f"lm_head_m{i}_mask"]))
                for i in range(_NUM_PLANES)
            ))
        del gdata

        # Per-layer weights
        self.layers: list[dict[str, np.ndarray]] = []
        for li in range(self.num_layers):
            path = os.path.join(save_dir, f"layer_{li}.npz")
            raw = np.load(path)
            layer: dict[str, np.ndarray] = {}
            for name in LINEAR_NAMES:
                gates = tuple(
                    (np.array(raw[f"{name}_m{i}_s1"]),
                     np.array(raw[f"{name}_m{i}_mask"]))
                    for i in range(_NUM_PLANES)
                )
                layer[name] = _z3_reconstruct_weight(gates)
            layer["input_layernorm_weight"] = np.array(
                raw["input_layernorm_weight"],
            )
            layer["post_attention_layernorm_weight"] = np.array(
                raw["post_attention_layernorm_weight"],
            )
            self.layers.append(layer)

        self.load_time: float = time.perf_counter() - t0
