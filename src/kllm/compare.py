"""HuggingFace vs kllm comparison.

Runs the same prompt through:
  1. HuggingFace ``AutoModelForCausalLM`` with greedy decoding
  2. kllm ``Chip`` — compiled processor inference

and prints a side-by-side report of generated text and timing.
"""

import os
import time

import numpy as np
import torch


# Standard benchmark prompts (short, medium, long)
_BENCHMARK_PROMPTS = [
    "Hello",
    "What is the capital of France?",
    "Explain quantum computing in simple terms for a beginner.",
]


def compare_chip(
    chip: object,
    text: str | None = None,
    max_tokens: int = 50,
) -> dict:
    """Compare chip inference vs HuggingFace for text generation.

    If *text* is None, runs a standard benchmark suite
    (multiple prompts). Otherwise compares the single prompt.
    """
    from kllm.device.chip import Chip

    # Read model_name from chip metadata
    import json
    with open(os.path.join(chip.path, "chip.json")) as f:
        meta = json.load(f)
    model_name = meta["model_name"]

    prompts = [text] if text else _BENCHMARK_PROMPTS

    results = []
    for prompt in prompts:
        stats = _compare_single(chip, model_name, prompt, max_tokens)
        results.append(stats)

    if len(results) == 1:
        return results[0]

    # Aggregate summary for benchmark suite
    return {
        "benchmark": True,
        "results": results,
        "avg_hf_time_s": sum(r["hf_time_s"] for r in results) / len(results),
        "avg_kllm_time_s": sum(r["kllm_time_s"] for r in results) / len(results),
    }


def _compare_single(
    chip: object,
    model_name: str,
    text: str,
    max_tokens: int,
) -> dict:
    """Compare a single prompt: HF vs chip."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    messages = [{"role": "user", "content": text}]

    # ---- HuggingFace ----
    hf_tok = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32,
        attn_implementation="eager",
    )

    prompt_text = hf_tok.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    ids = hf_tok(prompt_text, return_tensors="pt")["input_ids"]

    t0 = time.perf_counter()
    gen_ids = ids.clone()
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = hf_model(gen_ids).logits[0, -1]
            next_id = int(logits.argmax())
            if next_id == hf_tok.eos_token_id:
                break
            gen_ids = torch.cat(
                [gen_ids, torch.tensor([[next_id]])], dim=1,
            )
    hf_time = time.perf_counter() - t0

    hf_output = hf_tok.decode(
        gen_ids[0, ids.shape[1]:], skip_special_tokens=True,
    )

    del hf_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ---- kllm chip ----
    t0 = time.perf_counter()
    kllm_output = chip.infer(text, max_tokens)
    kllm_time = time.perf_counter() - t0

    return {
        "text": text,
        "max_tokens": max_tokens,
        "hf_time_s": hf_time,
        "kllm_time_s": kllm_time,
        "hf_output": hf_output,
        "kllm_output": kllm_output,
    }


def print_generate_report(stats: dict) -> None:
    """Pretty-print generation comparison report."""
    w = 64

    if stats.get("benchmark"):
        print("\n" + "=" * w)
        print("  kllm — Benchmark Suite: Chip vs HuggingFace")
        print("=" * w)
        for i, r in enumerate(stats["results"], 1):
            print(f"\n  [{i}] Prompt: {r['text']!r}")
            print(f"      HF  ({r['hf_time_s']:.2f}s): {r['hf_output'][:60]}")
            print(f"      kllm({r['kllm_time_s']:.2f}s): {r['kllm_output'][:60]}")
        print("-" * w)
        print(f"  Average: HF={stats['avg_hf_time_s']:.2f}s  "
              f"kllm={stats['avg_kllm_time_s']:.2f}s")
        print("=" * w)
        return

    print("\n" + "=" * w)
    print("  kllm — HuggingFace vs Chip Machine")
    print("=" * w)
    print(f"  Prompt     : {stats['text']!r}")
    print(f"  Max tokens : {stats['max_tokens']}")
    print("-" * w)
    print(f"  HuggingFace ({stats['hf_time_s']:.2f}s):")
    print(f"    {stats['hf_output']}")
    print("-" * w)
    print(f"  kllm ({stats['kllm_time_s']:.2f}s):")
    print(f"    {stats['kllm_output']}")
    print("=" * w)
