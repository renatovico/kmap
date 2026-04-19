"""Model weight downloader.

Downloads transformer weights from HuggingFace and saves them as
plain NumPy float32 ``.npy`` files for the C compiler to consume.

Usage::

    from kllm.compiler.fabric import Fabric
    Fabric.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "./mychip")
"""

import json
import os

import numpy as np

_ATTN_WEIGHTS = ("q_proj", "k_proj", "v_proj", "o_proj")
_MLP_WEIGHTS = ("gate_proj", "up_proj", "down_proj")


class Fabric:
    """Download and cache HuggingFace model weights as .npy files."""

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        save_dir: str = "./lossless_logic",
    ) -> None:
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
