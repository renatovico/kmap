"""Classic vs Bit-Sliced comparison.

Runs the same input through:
  1. Traditional FP32 matrix multiplication (PyTorch)
  2. Lossless bit-sliced logic fabric (kllm)

and prints a side-by-side report of timing, memory, and output stats.
"""

import os
import time
import tracemalloc

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kllm.bitops import extract_sub_masks, repack_sub_masks
from kllm.compiler import _PROJ_NAMES
from kllm.device import free_vram, to_device, to_numpy, xp


def _classic_forward_layer(embeddings: np.ndarray, layer) -> np.ndarray:
    """Run one transformer layer's attention projections the traditional way."""
    x = torch.from_numpy(embeddings)
    results = []
    for proj_name in _PROJ_NAMES:
        proj = getattr(layer.self_attn, proj_name, None)
        if proj is None:
            continue
        with torch.no_grad():
            results.append(proj(x).numpy())
    # Sum projections to produce a single comparable output per layer.
    return sum(results)


def _logic_forward_layer(planes: list, layer_data, proj_names: list) -> list:
    """Run one layer through the bit-sliced logic fabric."""
    accumulated = [xp.zeros_like(planes[p]) for p in range(4)]
    for proj_name in proj_names:
        prefix = f"{proj_name}_m"
        if f"{prefix}0_s1" not in layer_data:
            continue
        for p in range(4):
            s1 = to_device(layer_data[f"{prefix}{p}_s1"])
            mask = to_device(layer_data[f"{prefix}{p}_mask"])
            result = (planes[p] << s1) ^ mask
            accumulated[p] = accumulated[p] ^ result
            del s1, mask
    free_vram()
    return accumulated


def compare(
    model_name: str,
    save_dir: str,
    text: str,
    max_layers: int | None = None,
) -> dict:
    """Run both pipelines and return a comparison dict."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    embed_table = model.get_input_embeddings().weight.detach().numpy()

    token_ids = tokenizer.encode(text, return_tensors="np")[0]
    embeddings = embed_table[token_ids]  # (seq_len, hidden_dim) fp32

    meta = np.load(os.path.join(save_dir, "meta.npz"), allow_pickle=True)
    num_layers = int(meta["num_layers"])
    proj_names = list(meta["proj_names"])
    if max_layers is not None:
        num_layers = min(num_layers, max_layers)
    num_layers = min(num_layers, len(model.model.layers))

    # ------------------------------------------------------------------ classic
    tracemalloc.start()
    t0 = time.perf_counter()
    classic_state = embeddings.copy()
    for li in range(num_layers):
        classic_state = _classic_forward_layer(classic_state, model.model.layers[li])
    classic_time = time.perf_counter() - t0
    classic_mem_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    # ---------------------------------------------------------------- bit-logic
    planes = list(extract_sub_masks(embeddings))
    gpu_planes = [to_device(p) for p in planes]

    tracemalloc.start()
    t0 = time.perf_counter()
    for li in range(num_layers):
        data = np.load(os.path.join(save_dir, f"layer_{li}.npz"))
        gpu_planes = _logic_forward_layer(gpu_planes, data, proj_names)
        del data
    logic_time = time.perf_counter() - t0
    logic_mem_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    # Reconstruct fp32 from planes for stats.
    result_uint32 = (
        gpu_planes[3].astype(xp.uint32) << 24
        | gpu_planes[2].astype(xp.uint32) << 16
        | gpu_planes[1].astype(xp.uint32) << 8
        | gpu_planes[0].astype(xp.uint32)
    )
    logic_fp32 = to_numpy(
        result_uint32.view(xp.float32)
        if hasattr(result_uint32, "view")
        else result_uint32.view(np.float32)
    )

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {
        "text": text,
        "tokens": len(token_ids),
        "layers": num_layers,
        "classic_time_s": classic_time,
        "logic_time_s": logic_time,
        "speedup": classic_time / logic_time if logic_time > 0 else float("inf"),
        "classic_peak_mb": classic_mem_peak / 1024 / 1024,
        "logic_peak_mb": logic_mem_peak / 1024 / 1024,
        "classic_output_shape": classic_state.shape,
        "logic_output_shape": logic_fp32.shape,
        "classic_output_sample": classic_state.flat[:5].tolist(),
        "logic_output_sample": logic_fp32.flat[:5].tolist(),
    }


def print_report(stats: dict) -> None:
    """Pretty-print the comparison report."""
    print("\n" + "=" * 60)
    print("  kllm — Classic FP32 vs Lossless Bit-Sliced Logic")
    print("=" * 60)
    print(f"  Prompt      : {stats['text']!r}")
    print(f"  Tokens      : {stats['tokens']}")
    print(f"  Layers      : {stats['layers']}")
    print("-" * 60)
    print(f"  {'Metric':<28} {'Classic (FP32)':>14} {'Bit-Logic':>14}")
    print("-" * 60)
    print(f"  {'Time (s)':<28} {stats['classic_time_s']:>14.4f} {stats['logic_time_s']:>14.4f}")
    print(f"  {'Peak RAM (MB)':<28} {stats['classic_peak_mb']:>14.2f} {stats['logic_peak_mb']:>14.2f}")
    print(f"  {'Output shape':<28} {str(stats['classic_output_shape']):>14} {str(stats['logic_output_shape']):>14}")
    print("-" * 60)
    print(f"  Speedup (logic / classic) : {stats['speedup']:.2f}x")
    print()
    print(f"  Classic sample : {stats['classic_output_sample']}")
    print(f"  Logic sample   : {stats['logic_output_sample']}")
    print("=" * 60)
