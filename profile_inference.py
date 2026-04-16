"""Profile kllm inference to find bottlenecks."""
import json
import time
import numpy as np

from kllm.fabric import Fabric
from kllm.processor import Processor
from kllm.native_runner import NativeRunner

fabric = Fabric("./lossless_logic")

# Read EOS token ID from tokenizer config
tok_dir = "./lossless_logic/tokenizer"
with open(f"{tok_dir}/tokenizer_config.json") as f:
    eos_token_id = json.load(f).get("eos_token", "</s>")
with open(f"{tok_dir}/tokenizer.json") as f:
    vocab = json.load(f)["model"]["vocab"]
eos_token_id = vocab.get(eos_token_id, 2)

prompt_text = "<|user|>\nHello<|assistant|>\n"

# --- Profile Processor + NativeRunner lifecycle ---
print("\n=== PROCESSOR ===")

t0 = time.perf_counter()
proc = Processor.build(fabric, eos_token_id, tokenizer_dir=tok_dir)
t_build = time.perf_counter() - t0
print(f"  build (compile + optimise): {t_build:.3f}s")

t0 = time.perf_counter()
runner = NativeRunner(proc)
t_init = time.perf_counter() - t0
print(f"  NativeRunner init: {t_init:.3f}s")

# Encode prompt via circuit tokenizer
prompt_bytes = prompt_text.encode("utf-8")
token_ids_arr, num_tokens = runner.encode_bytes(prompt_bytes)
prompt = token_ids_arr[:num_tokens].tolist()
print(f"Prompt tokens: {len(prompt)}")

t0 = time.perf_counter()
output = runner.infer(prompt, max_tokens=5)
t_infer = time.perf_counter() - t0
print(f"  infer (prefill+5 decode): {t_infer:.3f}s")
print(f"  generated tokens: {output}")
