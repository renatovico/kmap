"""Profile kllm inference to find bottlenecks."""
import time
import numpy as np

from kllm.fabric import Fabric
from kllm.processor import Processor, NativeRunner
from kllm.tokenizer import Tokenizer

fabric = Fabric("./lossless_logic")
tok = Tokenizer("./lossless_logic/tokenizer")

prompt = tok.encode("<|user|>\nHello<|assistant|>\n")
print(f"Prompt tokens: {len(prompt)}")

# --- Profile Processor + NativeRunner lifecycle ---
print("\n=== PROCESSOR ===")

t0 = time.perf_counter()
proc = Processor.build(fabric, tok.eos_token_id, tokenizer=tok)
t_build = time.perf_counter() - t0
print(f"  build (compile + optimise): {t_build:.3f}s")

t0 = time.perf_counter()
runner = NativeRunner(proc)
t_init = time.perf_counter() - t0
print(f"  NativeRunner init: {t_init:.3f}s")

t0 = time.perf_counter()
output = runner.infer(prompt, max_tokens=5)
t_infer = time.perf_counter() - t0
print(f"  infer (prefill+5 decode): {t_infer:.3f}s")
print(f"  generated tokens: {output}")
