"""HuggingFace vs kllm comparison.

Runs the same prompt through:
  1. HuggingFace ``AutoModelForCausalLM`` with greedy decoding
  2. kllm ``JitSession`` — circuit graph inference via C executor

and prints a side-by-side report of generated text and timing.
"""

import os
import time

import numpy as np
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

    # ---- kllm engine (circuit graph + C executor) ----
    from kllm.fabric import Fabric
    from kllm.jit_optimizer import JitSession
    from kllm.tokenizer import Tokenizer as KllmTokenizer

    tok_dir = os.path.join(save_dir, "tokenizer")
    if not os.path.isdir(tok_dir):
        print("[compare] Weights not cached — running Fabric.from_pretrained …")
        Fabric.from_pretrained(model_name, save_dir)

    kllm_tok = KllmTokenizer(tok_dir)
    prompt_str = kllm_tok.apply_chat_template(
        messages, add_generation_prompt=True,
    )
    token_ids = kllm_tok.encode(prompt_str)

    fabric = Fabric(save_dir)
    session = JitSession(fabric)

    t0 = time.perf_counter()
    logits = session.prefill(token_ids)
    generated: list[int] = []
    for _ in range(max_tokens):
        next_id = int(np.argmax(logits[-1]))
        if next_id == kllm_tok.eos_token_id:
            break
        generated.append(next_id)
        logits = session.decode_step(next_id)
    kllm_time = time.perf_counter() - t0

    kllm_output = kllm_tok.decode(generated, skip_special_tokens=True)

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
    print("  kllm — HuggingFace vs Circuit Graph Engine")
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
