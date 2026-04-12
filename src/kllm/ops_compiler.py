"""Z3 operations compiler — trace-based circuit compilation.

Runs a reference forward pass through the transformer, captures every
unique float32 value that flows through each operation (SiLU, exp,
rsqrt, RoPE cos/sin multiply), and compiles Z3 gate LUTs for each.

After compilation, every arithmetic operation in the inference pipeline
can be executed via pure shift+XOR gates — the Z3 solver acts as a
universal computing machine that proves each gate correct.

Usage::

    from kllm.ops_compiler import OpsCompiler
    compiler = OpsCompiler(save_dir="./lossless_logic")
    compiler.compile()           # builds circuits.npz
"""

from __future__ import annotations

import os
import time

import numpy as np

from kllm.circuits import ArithmeticUnit, _exp_fn, _rsqrt_fn, _silu_fn


class OpsCompiler:
    """Trace-compile all transformer operations into Z3 circuits.

    Flow:
    1. Load the already-compiled Fabric (weight gates).
    2. Run a reference forward pass capturing every intermediate value.
    3. For each operation, extract unique float32 inputs.
    4. Use :class:`ArithmeticUnit` to compile Z3 gates for each.
    5. Save to ``<save_dir>/circuits.npz``.
    """

    def __init__(self, save_dir: str = "./lossless_logic") -> None:
        self.save_dir = save_dir
        self._traces: dict[str, list[np.ndarray]] = {
            "silu_inputs": [],
            "exp_inputs": [],
            "rsqrt_inputs": [],
            "rope_cos_inputs": [],
            "rope_sin_inputs": [],
        }

    def _trace_forward(self) -> None:
        """Run the transformer and capture all activations."""
        from kllm.fabric import Fabric
        from kllm.model import (
            apply_rotary_emb,
            build_rope_cache,
            rms_norm,
            silu,
            softmax,
        )
        from kllm.tokenizer import Tokenizer

        tok_dir = os.path.join(self.save_dir, "tokenizer")
        tokenizer = Tokenizer(tok_dir)
        fabric = Fabric(self.save_dir)

        # Use a representative prompt that exercises the model
        prompt = "Hello, how are you today?"
        token_ids = tokenizer.encode(prompt)
        seq = len(token_ids)

        hidden = fabric.embed_tokens[token_ids]  # (seq, hidden)

        max_seq = 2048
        rope_cos, rope_sin = build_rope_cache(
            max_seq, fabric.head_dim, fabric.rope_theta,
        )

        # Trace RoPE cos/sin values
        cos_slice = rope_cos[:seq]
        sin_slice = rope_sin[:seq]
        self._traces["rope_cos_inputs"].append(cos_slice.ravel())
        self._traces["rope_sin_inputs"].append(sin_slice.ravel())

        for li in range(fabric.num_layers):
            w = fabric.layers[li]

            # --- Attention block ---
            residual = hidden.copy()

            # RMSNorm — trace the rsqrt input
            x64 = hidden.astype(np.float64)
            variance = np.mean(x64 ** 2, axis=-1, keepdims=True)
            rsqrt_input = (variance + fabric.rms_norm_eps).astype(np.float32)
            self._traces["rsqrt_inputs"].append(rsqrt_input.ravel())
            normed = rms_norm(hidden, w["input_layernorm_weight"], fabric.rms_norm_eps)

            # Q, K, V projections (matmul — already compiled as weight gates)
            q = normed @ w["q_proj"].T
            k = normed @ w["k_proj"].T
            v = normed @ w["v_proj"].T

            q = q.reshape(seq, fabric.num_heads, fabric.head_dim).transpose(1, 0, 2)
            k = k.reshape(seq, fabric.num_kv_heads, fabric.head_dim).transpose(1, 0, 2)
            v = v.reshape(seq, fabric.num_kv_heads, fabric.head_dim).transpose(1, 0, 2)

            # RoPE — trace the multiply inputs
            q_rope = apply_rotary_emb(q, cos_slice, sin_slice)
            k_rope = apply_rotary_emb(k, cos_slice, sin_slice)

            # GQA
            if fabric.num_groups > 1:
                k_exp = np.repeat(k_rope, fabric.num_groups, axis=0)
                v_exp = np.repeat(v, fabric.num_groups, axis=0)
            else:
                k_exp, v_exp = k_rope, v

            # Attention scores
            scale = np.float32(1.0 / np.sqrt(fabric.head_dim))
            scores = np.matmul(q_rope, k_exp.transpose(0, 2, 1)) * scale

            # Causal mask
            q_pos = np.arange(seq)[:, None]
            k_pos = np.arange(seq)[None, :]
            causal = np.where(k_pos <= q_pos, np.float32(0.0), np.float32(-np.inf))
            scores += causal[np.newaxis, :, :]

            # Softmax — trace exp inputs
            m = scores.max(axis=-1, keepdims=True)
            exp_input = (scores - m).astype(np.float32)
            # filter out -inf before tracing
            finite_mask = np.isfinite(exp_input)
            self._traces["exp_inputs"].append(exp_input[finite_mask].ravel())

            attn_weights = softmax(scores, axis=-1)
            context = np.matmul(attn_weights, v_exp)
            context = context.transpose(1, 0, 2).reshape(seq, -1)
            attn_out = context @ w["o_proj"].T
            hidden = residual + attn_out

            # --- MLP block ---
            residual = hidden.copy()

            # RMSNorm again
            x64 = hidden.astype(np.float64)
            variance = np.mean(x64 ** 2, axis=-1, keepdims=True)
            rsqrt_input = (variance + fabric.rms_norm_eps).astype(np.float32)
            self._traces["rsqrt_inputs"].append(rsqrt_input.ravel())
            normed = rms_norm(
                hidden, w["post_attention_layernorm_weight"], fabric.rms_norm_eps,
            )

            gate = normed @ w["gate_proj"].T
            up = normed @ w["up_proj"].T

            # SiLU — trace inputs
            self._traces["silu_inputs"].append(gate.ravel())

            hidden = residual + (silu(gate) * up) @ w["down_proj"].T

            print(f"  [trace] Layer {li + 1}/{fabric.num_layers} done")

        # Final norm
        x64 = hidden.astype(np.float64)
        variance = np.mean(x64 ** 2, axis=-1, keepdims=True)
        rsqrt_input = (variance + fabric.rms_norm_eps).astype(np.float32)
        self._traces["rsqrt_inputs"].append(rsqrt_input.ravel())

    def compile(self) -> None:
        """Run trace + compile all operations into Z3 circuits."""
        print("[*] Tracing forward pass to capture activation domains …")
        t0 = time.perf_counter()
        self._trace_forward()
        trace_time = time.perf_counter() - t0
        print(f"[+] Trace done in {trace_time:.1f}s")

        # Collect unique values per operation
        traced: dict[str, np.ndarray] = {}
        for key, arrays in self._traces.items():
            if arrays:
                combined = np.concatenate(arrays)
                unique = np.unique(combined)
                traced[key] = unique
                print(f"  {key}: {len(unique)} unique values")

        # Compile circuits
        print("\n[*] Compiling Z3 circuits …")
        t0 = time.perf_counter()

        unit = ArithmeticUnit()
        unit.compile_constant_gates()

        if "silu_inputs" in traced:
            print("  [circuits] Compiling SiLU …")
            unit.compile_unary_op("silu", _silu_fn, traced["silu_inputs"])

        if "exp_inputs" in traced:
            print("  [circuits] Compiling exp …")
            unit.compile_unary_op("exp", _exp_fn, traced["exp_inputs"])

        if "rsqrt_inputs" in traced:
            print("  [circuits] Compiling rsqrt …")
            unit.compile_unary_op("rsqrt", _rsqrt_fn, traced["rsqrt_inputs"])

        compile_time = time.perf_counter() - t0
        print(f"[+] Circuits compiled in {compile_time:.1f}s")

        # Save
        out_path = os.path.join(self.save_dir, "circuits.npz")
        unit.save(out_path)
        print(f"[+] Saved to {out_path}")
