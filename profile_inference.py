"""Profile kllm inference to find bottlenecks."""
import time
import numpy as np

from kllm.fabric import Fabric
from kllm.jit_optimizer import JitSession
from kllm.tokenizer import Tokenizer

fabric = Fabric("./lossless_logic")
tok = Tokenizer("./lossless_logic/tokenizer")

prompt = tok.encode("<|user|>\nHello<|assistant|>\n")
print(f"Prompt tokens: {len(prompt)}")

# --- Profile JitSession lifecycle ---
print("\n=== JITSESSION ===")

t0 = time.perf_counter()
session = JitSession(fabric)
t_init = time.perf_counter() - t0
print(f"  __init__ (build machine + cache consts): {t_init:.3f}s")

t0 = time.perf_counter()
logits = session.prefill(prompt)
t_prefill = time.perf_counter() - t0
print(f"  prefill:       {t_prefill:.3f}s")

for i in range(5):
    next_id = int(np.argmax(logits[-1]))
    t0 = time.perf_counter()
    logits = session.decode_step(next_id)
    t_step = time.perf_counter() - t0
    print(f"  decode_step {i+1}: {t_step:.3f}s  (token={next_id})")

print(f"\n  Total position: {session.position}")
print(f"  KV cache shape: {session.kv_cache[0][0].shape}")
