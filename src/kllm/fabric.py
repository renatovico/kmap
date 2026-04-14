"""Model weight loader.

Loads transformer weights as plain NumPy float32 arrays.  Two paths:

1. ``Fabric.from_pretrained(model_name, save_dir)`` — download from
   HuggingFace, extract weights, cache as ``.npy`` files.
2. ``Fabric(save_dir)`` — load from previously cached ``.npy`` files.

This module is the *only* place that knows about the on-disk weight
layout (``config.json``, ``embed_tokens.npy``, ``layer_<n>/``).
"""

import json
import os
import time
from functools import lru_cache

import numpy as np

LINEAR_NAMES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)

_ATTN_WEIGHTS = ("q_proj", "k_proj", "v_proj", "o_proj")
_MLP_WEIGHTS = ("gate_proj", "up_proj", "down_proj")


class Fabric:
    """Model weight container — exposes config + float32 weight arrays.

    Attributes
    ----------
    num_layers, hidden_size, num_heads, num_kv_heads, intermediate_size,
    vocab_size, rms_norm_eps, rope_theta, head_dim, num_groups
        Model topology.
    embed_tokens : ndarray (vocab_size, hidden_size)
    lm_head : ndarray (vocab_size, hidden_size) — may be tied to embed_tokens
    final_norm_weight : ndarray (hidden_size,)
    layers : list[dict[str, ndarray]]
        Per-layer weight arrays (q_proj, k_proj, … , layernorm weights).
    """

    def __init__(self, save_dir: str) -> None:
        self.save_dir = save_dir

        weights_dir = os.path.join(save_dir, "weights")
        config_path = os.path.join(weights_dir, "config.json")

        # Support legacy meta.npz layout as fallback
        if not os.path.exists(config_path):
            legacy_meta = os.path.join(save_dir, "meta.npz")
            legacy_cache = os.path.join(save_dir, "optimized")
            if os.path.exists(legacy_meta) and os.path.isdir(legacy_cache):
                self._load_legacy(save_dir)
                return
            raise FileNotFoundError(
                f"No compiled model found at {save_dir}. "
                "Run `kllm --mode compile` first."
            )

        t0 = time.perf_counter()

        with open(config_path) as f:
            cfg = json.load(f)

        self.num_layers: int = cfg["num_layers"]
        self.hidden_size: int = cfg["hidden_size"]
        self.num_heads: int = cfg["num_attention_heads"]
        self.num_kv_heads: int = cfg["num_key_value_heads"]
        self.intermediate_size: int = cfg["intermediate_size"]
        self.vocab_size: int = cfg["vocab_size"]
        self.rms_norm_eps: float = cfg["rms_norm_eps"]
        self.rope_theta: float = cfg["rope_theta"]
        self.head_dim: int = self.hidden_size // self.num_heads
        self.num_groups: int = self.num_heads // self.num_kv_heads

        self._load_cached(weights_dir)
        self.load_time: float = time.perf_counter() - t0

    def _load_cached(self, weights_dir: str) -> None:
        """Load float32 weights from .npy cache."""
        self.embed_tokens = np.load(
            os.path.join(weights_dir, "embed_tokens.npy"),
        )
        self.final_norm_weight = np.load(
            os.path.join(weights_dir, "final_norm_weight.npy"),
        )

        lm_head_path = os.path.join(weights_dir, "lm_head.npy")
        if os.path.exists(lm_head_path):
            self.lm_head = np.load(lm_head_path)
        else:
            self.lm_head = self.embed_tokens  # tied

        self.layers = []
        for li in range(self.num_layers):
            layer_dir = os.path.join(weights_dir, f"layer_{li}")
            layer: dict[str, np.ndarray] = {}
            for name in LINEAR_NAMES:
                layer[name] = np.load(
                    os.path.join(layer_dir, f"{name}.npy"),
                )
            layer["input_layernorm_weight"] = np.load(
                os.path.join(layer_dir, "input_layernorm_weight.npy"),
            )
            layer["post_attention_layernorm_weight"] = np.load(
                os.path.join(layer_dir, "post_attention_layernorm_weight.npy"),
            )
            self.layers.append(layer)

    def _load_legacy(self, save_dir: str) -> None:
        """Load from legacy meta.npz + optimized/ layout."""
        meta = np.load(os.path.join(save_dir, "meta.npz"), allow_pickle=True)
        self.num_layers = int(meta["num_layers"])
        self.hidden_size = int(meta["hidden_size"])
        self.num_heads = int(meta["num_attention_heads"])
        self.num_kv_heads = int(meta["num_key_value_heads"])
        self.intermediate_size = int(meta["intermediate_size"])
        self.vocab_size = int(meta["vocab_size"])
        self.rms_norm_eps = float(meta["rms_norm_eps"])
        self.rope_theta = float(meta["rope_theta"])
        self.head_dim = self.hidden_size // self.num_heads
        self.num_groups = self.num_heads // self.num_kv_heads

        cache_dir = os.path.join(save_dir, "optimized")
        t0 = time.perf_counter()
        self._load_cached(cache_dir)
        self.load_time = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Transposed-weight cache (shared across all compile calls)
    # ------------------------------------------------------------------

    @lru_cache(maxsize=None)
    def get_transposed(self, layer_idx: int, proj: str) -> np.ndarray:
        """Return ``layers[layer_idx][proj].T`` as contiguous float32.

        Results are cached so that ``compile_model`` and
        ``compile_decode_template`` share the same array objects,
        avoiding duplicate multi-GB allocations.
        """
        return np.ascontiguousarray(
            self.layers[layer_idx][proj].T, dtype=np.float32)

    @lru_cache(maxsize=None)
    def get_fused_qkv_t(self, layer_idx: int) -> np.ndarray:
        """Return ``[Wq | Wk | Wv].T`` — fused QKV weight matrix.

        Shape: (hidden_size, q_dim + kv_dim + kv_dim).
        One matmul replaces three separate Q/K/V projections.
        """
        q_t = self.get_transposed(layer_idx, "q_proj")
        k_t = self.get_transposed(layer_idx, "k_proj")
        v_t = self.get_transposed(layer_idx, "v_proj")
        return np.ascontiguousarray(
            np.concatenate([q_t, k_t, v_t], axis=1))

    @lru_cache(maxsize=None)
    def get_fused_gate_up_t(self, layer_idx: int) -> np.ndarray:
        """Return ``[Wgate | Wup].T`` — fused Gate+Up weight matrix.

        Shape: (hidden_size, 2 * intermediate_size).
        One matmul replaces two separate gate/up projections.
        """
        gate_t = self.get_transposed(layer_idx, "gate_proj")
        up_t = self.get_transposed(layer_idx, "up_proj")
        return np.ascontiguousarray(
            np.concatenate([gate_t, up_t], axis=1))

    # ------------------------------------------------------------------
    # INT8 quantization cache (disk-backed)
    # ------------------------------------------------------------------

    @staticmethod
    def _quantize_per_column(w_f32: np.ndarray
                             ) -> tuple[np.ndarray, np.ndarray]:
        """Per-output-channel absmax quantization to int8.

        Returns (w_q8, scales) where ``w_f32 ≈ w_q8.astype(f32) * scales``.
        """
        amax = np.abs(w_f32).max(axis=0)
        amax = np.where(amax == 0, 1.0, amax)  # avoid div-by-zero
        scales = (amax / 127.0).astype(np.float32)
        w_q8 = np.clip(np.round(w_f32 / scales), -128, 127).astype(np.int8)
        return w_q8, scales

    def _q8_cache_path(self, name: str
                       ) -> tuple[str, str]:
        """Return disk paths for cached (w_q8, scales) arrays."""
        q8_dir = os.path.join(self.save_dir, "q8")
        return (os.path.join(q8_dir, f"{name}_q8.npy"),
                os.path.join(q8_dir, f"{name}_scales.npy"))

    def _load_or_quantize(self, name: str, w_f32_fn
                          ) -> tuple[np.ndarray, np.ndarray]:
        """Load cached INT8 weights from disk, or quantize and save."""
        q8_path, scales_path = self._q8_cache_path(name)
        if os.path.exists(q8_path) and os.path.exists(scales_path):
            return np.load(q8_path), np.load(scales_path)
        w_q8, scales = self._quantize_per_column(w_f32_fn())
        os.makedirs(os.path.dirname(q8_path), exist_ok=True)
        np.save(q8_path, w_q8)
        np.save(scales_path, scales)
        return w_q8, scales

    @lru_cache(maxsize=None)
    def get_quantized(self, layer_idx: int, proj: str
                      ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(W_q8, scales)`` for a single projection (transposed).

        W_q8 shape: (in_features, out_features) int8
        scales shape: (out_features,) float32
        """
        return self._load_or_quantize(
            f"L{layer_idx}_{proj}",
            lambda: self.get_transposed(layer_idx, proj))

    @lru_cache(maxsize=None)
    def get_quantized_fused_qkv(self, layer_idx: int
                                ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(W_q8, scales)`` for fused QKV (transposed)."""
        return self._load_or_quantize(
            f"L{layer_idx}_fused_qkv",
            lambda: self.get_fused_qkv_t(layer_idx))

    @lru_cache(maxsize=None)
    def get_quantized_fused_gate_up(self, layer_idx: int
                                    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(W_q8, scales)`` for fused GateUp (transposed)."""
        return self._load_or_quantize(
            f"L{layer_idx}_fused_gate_up",
            lambda: self.get_fused_gate_up_t(layer_idx))

    # ------------------------------------------------------------------
    # Class method: download from HuggingFace and cache
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        save_dir: str = "./lossless_logic",
    ) -> "Fabric":
        """Download a model from HuggingFace and extract float32 weights.

        Requires ``torch`` and ``transformers`` (compile-time only).
        After the first call, weights are cached as ``.npy`` files —
        subsequent loads via ``Fabric(save_dir)`` need only numpy.

        Parameters
        ----------
        model_name : str
            HuggingFace model ID (e.g. ``TinyLlama/TinyLlama-1.1B-Chat-v1.0``).
        save_dir : str
            Directory to save extracted weights and tokenizer.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[*] Loading {model_name} …")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float32,
        )
        config = model.config

        # ---- Save tokenizer ----
        tok = AutoTokenizer.from_pretrained(model_name)
        tok_dir = os.path.join(save_dir, "tokenizer")
        tok.save_pretrained(tok_dir)
        print(f"  -> tokenizer saved to {tok_dir}")

        # ---- Save config ----
        weights_dir = os.path.join(save_dir, "weights")
        os.makedirs(weights_dir, exist_ok=True)

        cfg = {
            "model_name": model_name,
            "num_layers": len(model.model.layers),
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
        }
        with open(os.path.join(weights_dir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

        # ---- Save global weights ----
        embed = model.get_input_embeddings().weight.detach().numpy()
        np.save(os.path.join(weights_dir, "embed_tokens.npy"), embed)

        norm_w = model.model.norm.weight.detach().numpy()
        np.save(os.path.join(weights_dir, "final_norm_weight.npy"), norm_w)

        lm_head_w = model.lm_head.weight.detach().numpy()
        tied = np.array_equal(lm_head_w, embed)
        if not tied:
            np.save(os.path.join(weights_dir, "lm_head.npy"), lm_head_w)
        print(f"  -> global weights saved (lm_head tied={tied})")

        # ---- Save per-layer weights ----
        for li, layer in enumerate(model.model.layers):
            layer_dir = os.path.join(weights_dir, f"layer_{li}")
            os.makedirs(layer_dir, exist_ok=True)

            attn = layer.self_attn
            for name in _ATTN_WEIGHTS:
                proj = getattr(attn, name, None)
                if proj is not None:
                    np.save(
                        os.path.join(layer_dir, f"{name}.npy"),
                        proj.weight.detach().numpy(),
                    )

            mlp = layer.mlp
            for name in _MLP_WEIGHTS:
                proj = getattr(mlp, name, None)
                if proj is not None:
                    np.save(
                        os.path.join(layer_dir, f"{name}.npy"),
                        proj.weight.detach().numpy(),
                    )

            np.save(
                os.path.join(layer_dir, "input_layernorm_weight.npy"),
                layer.input_layernorm.weight.detach().numpy(),
            )
            np.save(
                os.path.join(layer_dir, "post_attention_layernorm_weight.npy"),
                layer.post_attention_layernorm.weight.detach().numpy(),
            )

            print(f"  -> layer {li + 1}/{len(model.model.layers)} saved")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[+] Weights cached in {weights_dir}/")
        return cls(save_dir)
