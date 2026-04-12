"""Classic vs kllm comparison.

Runs the same prompt through:
  1. Standard HuggingFace ``model(input_ids)`` forward pass
  2. kllm lossless fabric inference (weight reconstruction → matmul)

and prints a side-by-side report of timing, memory, output shape,
decoded text, and numerical equivalence.
"""

import os
import time
import tracemalloc

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kllm.inference import LosslessInferenceEngine


# ------------------------------------------------------------------
# Classic HuggingFace forward pass
# ------------------------------------------------------------------

def _classic_forward(
    model, tokenizer, text: str
) -> tuple[np.ndarray, list[int]]:
    """Run the standard HuggingFace causal-LM forward pass.

    Returns ``(logits_numpy, token_id_list)``.
    """
    inputs = tokenizer(text, return_tensors="pt")
    token_ids = inputs["input_ids"][0].tolist()

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0].numpy()  # (seq, vocab)
    return logits, token_ids


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def compare(
    model_name: str,
    save_dir: str,
    text: str,
    max_layers: int | None = None,
) -> dict:
    """Run both pipelines and return a comparison dict."""

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32
    )

    # ---- configure layer limit (apply to both paths) -----------------
    actual_layers = len(model.model.layers)
    if max_layers is not None:
        actual_layers = min(actual_layers, max_layers)

    # --------------------------------------------------- classic (HuggingFace)
    tracemalloc.start()
    t0 = time.perf_counter()
    classic_logits, token_ids = _classic_forward(model, tokenizer, text)
    classic_time = time.perf_counter() - t0
    classic_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    classic_pred_ids = classic_logits.argmax(axis=-1)
    classic_decoded = tokenizer.decode(classic_pred_ids, skip_special_tokens=True)

    model_params = sum(p.numel() for p in model.parameters())
    model_size_mb = (model_params * 4) / 1024 / 1024

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # --------------------------------------------------- kllm fabric
    engine = LosslessInferenceEngine(model_name, save_dir)
    if max_layers is not None:
        engine.num_layers = min(engine.num_layers, max_layers)

    tracemalloc.start()
    t0 = time.perf_counter()
    kllm_logits = engine.forward(token_ids)
    kllm_time = time.perf_counter() - t0
    kllm_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    kllm_pred_ids = kllm_logits.argmax(axis=-1)
    kllm_decoded = tokenizer.decode(kllm_pred_ids, skip_special_tokens=True)

    # Fabric disk size
    fabric_bytes = sum(
        os.path.getsize(os.path.join(save_dir, f))
        for f in os.listdir(save_dir)
        if f.endswith(".npz")
    )

    # Numerical comparison
    max_abs_diff = float(np.max(np.abs(classic_logits - kllm_logits)))
    mean_abs_diff = float(np.mean(np.abs(classic_logits - kllm_logits)))
    logits_match = np.allclose(classic_logits, kllm_logits, atol=1e-4)
    tokens_match = np.array_equal(classic_pred_ids, kllm_pred_ids)

    return {
        "text": text,
        "tokens": len(token_ids),
        "layers": actual_layers,
        "classic_time_s": classic_time,
        "kllm_time_s": kllm_time,
        "speedup": classic_time / kllm_time if kllm_time > 0 else float("inf"),
        "classic_peak_mb": classic_peak / 1024 / 1024,
        "kllm_peak_mb": kllm_peak / 1024 / 1024,
        "model_size_mb": model_size_mb,
        "fabric_size_mb": fabric_bytes / 1024 / 1024,
        "classic_output_shape": classic_logits.shape,
        "kllm_output_shape": kllm_logits.shape,
        "classic_decoded": classic_decoded,
        "kllm_decoded": kllm_decoded,
        "classic_sample": classic_logits[0, :5].tolist(),
        "kllm_sample": kllm_logits[0, :5].tolist(),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "logits_match": logits_match,
        "tokens_match": tokens_match,
    }


def compare_generate(
    model_name: str,
    save_dir: str,
    text: str,
    engine: str = "bitlogic",
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
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if engine == "bitlogic":
        from kllm.inference import BitLogicInferenceEngine
        eng = BitLogicInferenceEngine(model_name, save_dir)
    else:
        from kllm.inference import LosslessInferenceEngine
        eng = LosslessInferenceEngine(model_name, save_dir)

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
        "engine": engine,
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
    print(f"  kllm — HuggingFace Pipeline vs {stats['engine']} Engine")
    print("=" * w)
    print(f"  Prompt     : {stats['text']!r}")
    print(f"  Max tokens : {stats['max_tokens']}")
    print("-" * w)
    print(f"  HuggingFace ({stats['hf_time_s']:.2f}s):")
    print(f"    {stats['hf_output']}")
    print("-" * w)
    print(f"  kllm {stats['engine']} ({stats['kllm_time_s']:.2f}s):")
    print(f"    {stats['kllm_output']}")
    print("=" * w)


def print_report(stats: dict) -> None:
    """Pretty-print the comparison report."""
    w = 64
    print("\n" + "=" * w)
    print("  kllm — HuggingFace vs Lossless Fabric Inference")
    print("=" * w)
    print(f"  Prompt      : {stats['text']!r}")
    print(f"  Tokens      : {stats['tokens']}")
    print(f"  Layers      : {stats['layers']}")
    print("-" * w)
    print(f"  {'Metric':<30} {'HuggingFace':>16} {'kllm':>16}")
    print("-" * w)
    print(f"  {'Time (s)':<30} {stats['classic_time_s']:>16.4f} {stats['kllm_time_s']:>16.4f}")
    print(f"  {'Peak RAM (MB)':<30} {stats['classic_peak_mb']:>16.2f} {stats['kllm_peak_mb']:>16.2f}")
    print(f"  {'Size on disk (MB)':<30} {stats['model_size_mb']:>16.2f} {stats['fabric_size_mb']:>16.2f}")
    print(f"  {'Output shape':<30} {str(stats['classic_output_shape']):>16} {str(stats['kllm_output_shape']):>16}")
    print("-" * w)

    print(f"  {'Decoded output':<30}")
    print(f"    HF  : {stats['classic_decoded'][:60]!r}")
    print(f"    kllm: {stats['kllm_decoded'][:60]!r}")
    print("-" * w)

    print(f"  Logits max |diff| : {stats['max_abs_diff']:.6e}")
    print(f"  Logits mean|diff| : {stats['mean_abs_diff']:.6e}")
    ok = "\u2705" if stats["logits_match"] else "\u274c"
    print(f"  Logits match (atol=1e-4) : {ok}  {stats['logits_match']}")
    ok = "\u2705" if stats["tokens_match"] else "\u274c"
    print(f"  Predicted tokens match   : {ok}  {stats['tokens_match']}")
    print("=" * w)
