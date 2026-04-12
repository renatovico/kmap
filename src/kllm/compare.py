"""HuggingFace vs kllm comparison.

Runs the same prompt through:
  1. HuggingFace ``pipeline("text-generation")`` with greedy decoding
  2. kllm ``BitLogicInferenceEngine`` — Z3-gate-backed inference

and prints a side-by-side report of generated text and timing.
"""

import os
import time

import torch
from transformers import AutoTokenizer


def compare_generate(
    model_name: str,
    save_dir: str,
    text: str,
    max_tokens: int = 50,
) -> dict:
    """Compare HuggingFace pipeline vs kllm engine for text generation.

    *text* is treated as a user message in a chat template.
    """
    from transformers import pipeline as hf_pipeline

    # ---- HuggingFace pipeline ----
    messages = [{"role": "user", "content": text}]
    pipe = hf_pipeline(
        "text-generation", model=model_name,
        max_new_tokens=max_tokens, do_sample=False,
    )

    t0 = time.perf_counter()
    hf_result = pipe(messages)
    hf_time = time.perf_counter() - t0

    hf_text = hf_result[0]["generated_text"]
    if isinstance(hf_text, list):
        hf_output = hf_text[-1]["content"]
    else:
        hf_output = hf_text

    del pipe
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ---- kllm engine ----
    tok_dir = os.path.join(save_dir, "tokenizer")
    if os.path.isdir(tok_dir):
        tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    from kllm.inference import BitLogicInferenceEngine

    eng = BitLogicInferenceEngine(model_name, save_dir)

    t0 = time.perf_counter()
    kllm_full = eng.generate(prompt, max_new_tokens=max_tokens)
    kllm_time = time.perf_counter() - t0

    # generate() decodes with skip_special_tokens=True, so we must
    # compare against the prompt decoded the same way for correct slicing.
    prompt_decoded = tokenizer.decode(
        tokenizer.encode(prompt), skip_special_tokens=True,
    )
    kllm_output = kllm_full[len(prompt_decoded):]

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
    print("\n" + "=" * w)
    print("  kllm — HuggingFace Pipeline vs Z3 Gate Engine")
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
