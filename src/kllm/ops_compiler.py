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


def _cos_fn(x: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.cos(x.astype(np.float64)).astype(np.float32)


def _sin_fn(x: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.sin(x.astype(np.float64)).astype(np.float32)


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

    def _trace_forward(self, gen_steps: int = 10) -> None:
        """Run the transformer (prefill + generation) and capture all activations.

        Uses :class:`CircuitTransformer` with ``unit=None`` (NumPy math)
        so the trace captures the exact same float32 bit patterns that
        will appear at inference time.  Covers both prefill and
        ``gen_steps`` autoregressive decode steps.
        """
        from kllm.circuit_model import CircuitMath, _apply_rotary_emb, _build_rope_cache
        from kllm.fabric import Fabric
        from kllm.tokenizer import Tokenizer

        tok_dir = os.path.join(self.save_dir, "tokenizer")
        tokenizer = Tokenizer(tok_dir)
        fabric = Fabric(self.save_dir)
        math = CircuitMath(unit=None)  # NumPy — same as inference fallback

        prompt = "Hello, how are you today?"
        token_ids = tokenizer.encode(prompt)

        max_seq = 2048
        rope_cos, rope_sin = _build_rope_cache(
            max_seq, fabric.head_dim, fabric.rope_theta,
        )

        # KV cache for autoregressive steps
        kv_cache: list[dict[str, np.ndarray]] = [
            {"k": np.zeros((fabric.num_kv_heads, 0, fabric.head_dim), dtype=np.float32),
             "v": np.zeros((fabric.num_kv_heads, 0, fabric.head_dim), dtype=np.float32)}
            for _ in range(fabric.num_layers)
        ]

        def _trace_step(hidden: np.ndarray, start_pos: int) -> np.ndarray:
            seq = hidden.shape[0]
            positions = np.arange(start_pos, start_pos + seq)
            cos_slice = rope_cos[positions]
            sin_slice = rope_sin[positions]
            self._traces["rope_cos_inputs"].append(cos_slice.ravel())
            self._traces["rope_sin_inputs"].append(sin_slice.ravel())

            for li in range(fabric.num_layers):
                w = fabric.layers[li]

                # --- Attention ---
                residual = hidden.copy()

                # RMSNorm — uses CircuitMath.rsqrt (same path as inference)
                variance = np.mean(hidden.astype(np.float64) ** 2,
                                   axis=-1, keepdims=True)
                rsqrt_input = (variance + fabric.rms_norm_eps).astype(np.float32)
                self._traces["rsqrt_inputs"].append(rsqrt_input.ravel())
                normed = math.rms_norm(
                    hidden, w["input_layernorm_weight"], fabric.rms_norm_eps,
                )

                q = normed @ w["q_proj"].T
                k = normed @ w["k_proj"].T
                v = normed @ w["v_proj"].T

                q = q.reshape(seq, fabric.num_heads, fabric.head_dim).transpose(1, 0, 2)
                k = k.reshape(seq, fabric.num_kv_heads, fabric.head_dim).transpose(1, 0, 2)
                v = v.reshape(seq, fabric.num_kv_heads, fabric.head_dim).transpose(1, 0, 2)

                q_rope = _apply_rotary_emb(q, cos_slice, sin_slice)
                k_rope = _apply_rotary_emb(k, cos_slice, sin_slice)

                # KV cache
                kv = kv_cache[li]
                kv["k"] = np.concatenate([kv["k"], k_rope], axis=1)
                kv["v"] = np.concatenate([kv["v"], v], axis=1)

                if fabric.num_groups > 1:
                    k_full = np.repeat(kv["k"], fabric.num_groups, axis=0)
                    v_full = np.repeat(kv["v"], fabric.num_groups, axis=0)
                else:
                    k_full, v_full = kv["k"], kv["v"]

                scale = np.float32(1.0 / np.sqrt(fabric.head_dim))
                scores = np.matmul(q_rope, k_full.transpose(0, 2, 1)) * scale

                # Causal mask (only for prefill, seq > 1)
                if seq > 1:
                    q_pos = positions[:, None]
                    k_pos = np.arange(kv["k"].shape[1])[None, :]
                    causal = np.where(
                        k_pos <= q_pos, np.float32(0.0), np.float32(-np.inf),
                    )
                    scores += causal[np.newaxis, :, :]

                # Softmax — uses CircuitMath.exp (same path as inference)
                m = scores.max(axis=-1, keepdims=True)
                exp_input = (scores - m).astype(np.float32)
                self._traces["exp_inputs"].append(exp_input.ravel())

                attn_weights = math.softmax(scores, axis=-1)
                context = np.matmul(attn_weights, v_full)
                context = context.transpose(1, 0, 2).reshape(seq, -1)
                attn_out = context @ w["o_proj"].T
                hidden = residual + attn_out

                # --- MLP ---
                residual = hidden.copy()

                variance = np.mean(hidden.astype(np.float64) ** 2,
                                   axis=-1, keepdims=True)
                rsqrt_input = (variance + fabric.rms_norm_eps).astype(np.float32)
                self._traces["rsqrt_inputs"].append(rsqrt_input.ravel())
                normed = math.rms_norm(
                    hidden, w["post_attention_layernorm_weight"],
                    fabric.rms_norm_eps,
                )

                gate = normed @ w["gate_proj"].T
                up = normed @ w["up_proj"].T

                self._traces["silu_inputs"].append(gate.ravel())

                hidden = residual + (math.silu(gate) * up) @ w["down_proj"].T

            # Final norm
            variance = np.mean(hidden.astype(np.float64) ** 2,
                               axis=-1, keepdims=True)
            rsqrt_input = (variance + fabric.rms_norm_eps).astype(np.float32)
            self._traces["rsqrt_inputs"].append(rsqrt_input.ravel())

            return hidden

        # --- Prefill ---
        hidden = fabric.embed_tokens[token_ids]
        hidden = _trace_step(hidden, start_pos=0)
        print(f"  [trace] Prefill done ({len(token_ids)} tokens, "
              f"{fabric.num_layers} layers)")

        logits = hidden @ fabric.lm_head.T
        next_id = int(logits[-1].argmax())
        token_ids.append(next_id)

        # --- Autoregressive generation ---
        for step in range(gen_steps):
            hidden = fabric.embed_tokens[[token_ids[-1]]]
            start_pos = len(token_ids) - 1
            hidden = _trace_step(hidden, start_pos=start_pos)

            logits = hidden @ fabric.lm_head.T
            next_id = int(logits[-1].argmax())
            token_ids.append(next_id)
            print(f"  [trace] Gen step {step + 1}/{gen_steps} done")

    def compile(self) -> None:
        """Compile full-domain byte-plane maps for every unary op.

        For each function (SiLU, exp, rsqrt), evaluates all 2³²
        possible float32 bit patterns, decomposes each output into 4
        bytes, and writes 4 mmap files (~4 GB each).  At runtime,
        only the pages accessed are loaded — same 4-plane strategy
        used for weight storage (``m0``–``m3``).

        No trace needed: the full float32 domain is covered.
        """
        circuits_dir = os.path.join(self.save_dir, "circuits")
        os.makedirs(circuits_dir, exist_ok=True)

        print("[*] Compiling full-domain byte-plane maps …")
        print("    (4 planes × 4 GB per op, mmap'd at runtime)\n")

        unit = ArithmeticUnit()
        unit.compile_constant_gates()

        t0_all = time.perf_counter()

        for op_name, fn in [
            ("silu", _silu_fn),
            ("exp", _exp_fn),
            ("rsqrt", _rsqrt_fn),
            ("cos", _cos_fn),
            ("sin", _sin_fn),
        ]:
            print(f"  [circuits] Compiling {op_name} (full 2³² domain) …")
            t0 = time.perf_counter()
            unit.compile_full_domain(op_name, fn, circuits_dir)
            elapsed = time.perf_counter() - t0
            print(f"  [+] {op_name} done in {elapsed:.1f}s\n")

        total = time.perf_counter() - t0_all
        print(f"[+] All circuits compiled in {total:.1f}s")

        # Save const gates + metadata (byte planes already on disk)
        out_path = os.path.join(self.save_dir, "circuits.npz")
        unit.save(out_path)
        print(f"[+] Metadata saved to {out_path}")
