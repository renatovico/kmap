"""HuggingFace vs kllm comparison.

Runs the same prompt through:
  1. HuggingFace ``pipeline("text-generation")`` with greedy decoding
  2. kllm ``BitLogicInferenceEngine`` — Z3-gate-backed inference

and prints a side-by-side report of generated text and timing.
"""

import os
import time

import torch


def compare_generate(
    model_name: str,
    save_dir: str,
    text: str,
    max_tokens: int = 50,
) -> dict:
    """Compare HuggingFace pipeline vs kllm engine for text generation.

    *text* is treated as a user message in a chat template.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # ---- HuggingFace (manual greedy decode) ----
    # Use float32 + eager attention to match kllm's exact numerical
    # path.  The pipeline API defaults to bfloat16 + SDPA which gives
    # different accumulation results.
    messages = [{"role": "user", "content": text}]
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

    # ---- kllm engine (pure Python, no HuggingFace) ----
    from kllm.tokenizer import Tokenizer as KllmTokenizer

    tok_dir = os.path.join(save_dir, "tokenizer")
    kllm_tok = KllmTokenizer(tok_dir)
    prompt = kllm_tok.apply_chat_template(
        messages, add_generation_prompt=True,
    )

    from kllm.inference import BitLogicInferenceEngine

    eng = BitLogicInferenceEngine(save_dir)

    t0 = time.perf_counter()
    kllm_full = eng.generate(prompt, max_new_tokens=max_tokens)
    kllm_time = time.perf_counter() - t0

    # generate() decodes with skip_special_tokens=True, so we must
    # compare against the prompt decoded the same way for correct slicing.
    prompt_decoded = kllm_tok.decode(
        kllm_tok.encode(prompt), skip_special_tokens=True,
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
