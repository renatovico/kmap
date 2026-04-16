"""Profile per-op-type time in evaluate_c."""
import json
import time
import numpy as np
from collections import defaultdict

from kllm.fabric import Fabric
from kllm.processor import Processor
from kllm.native_runner import NativeRunner
from kllm.circuit_executor import ExecutionPlan, precompute_consts, _get_lib, _LUT_FN_MAP, _to_c_float
from kllm.circuit_graph import Op

fabric = Fabric("./lossless_logic")

# Read EOS token ID from tokenizer config
tok_dir = "./lossless_logic/tokenizer"
with open(f"{tok_dir}/tokenizer_config.json") as f:
    eos_token_id = json.load(f).get("eos_token", "</s>")
with open(f"{tok_dir}/tokenizer.json") as f:
    vocab = json.load(f)["model"]["vocab"]
eos_token_id = vocab.get(eos_token_id, 2)

# Build processor and runner
proc = Processor.build(fabric, eos_token_id, tokenizer_dir=tok_dir)
runner = NativeRunner(proc)

# Encode prompt via circuit tokenizer
prompt_text = "<|user|>\nHello<|assistant|>\n"
prompt_bytes = prompt_text.encode("utf-8")
token_ids_arr, num_tokens = runner.encode_bytes(prompt_bytes)
prompt = token_ids_arr[:num_tokens].tolist()

# Warm up: prefill + 5 decode steps to reach steady state
output = runner._infer_python(prompt, max_tokens=5)

# Build inputs for profiling one decode step
p = proc
position = len(prompt) + len(output)
last_token = output[-1] if output else prompt[-1]
kv_cache = []
for _ in range(p.num_layers):
    kv_cache.append((
        np.zeros((p.num_kv_heads, position, p.head_dim), dtype=np.float32),
        np.zeros((p.num_kv_heads, position, p.head_dim), dtype=np.float32),
    ))
inputs = runner._build_inputs(last_token, position, kv_cache)

# Timed run through the instruction tape with per-op timing
plan = runner._plan
v = plan._base_values.copy()
for nid, arr in inputs.items():
    v[nid] = np.ascontiguousarray(arr)

op_times = defaultdict(float)
op_counts = defaultdict(int)

import kllm.circuit_executor as ce

TAG_NAMES = {
    ce._T_LUT: 'LUT', ce._T_BINOP: 'BINOP', ce._T_CMP_LE: 'CMP_LE',
    ce._T_MUX: 'MUX', ce._T_NEG: 'NEG', ce._T_ABS: 'ABS',
    ce._T_SQUARE: 'SQUARE', ce._T_MATMUL: 'MATMUL', ce._T_REDUCE: 'REDUCE',
    ce._T_ARGMAX: 'ARGMAX', ce._T_RESHAPE: 'RESHAPE',
    ce._T_TRANSPOSE: 'TRANSPOSE', ce._T_CONCAT: 'CONCAT',
    ce._T_REPEAT: 'REPEAT', ce._T_SLICE: 'SLICE', ce._T_CAST: 'CAST',
    ce._T_EXPAND_DIMS: 'EXPAND_DIMS',
}

for instr in plan._tape:
    tag = instr[0]
    t0 = time.perf_counter()

    # Execute (copy the dispatch logic from plan.run)
    if tag == ce._T_BINOP:
        v[instr[1]] = instr[4](np.asarray(v[instr[2]], dtype=np.float32),
                               np.asarray(v[instr[3]], dtype=np.float32))
    elif tag == ce._T_MATMUL:
        v[instr[1]] = np.matmul(np.ascontiguousarray(v[instr[2]], dtype=np.float32),
                                np.ascontiguousarray(v[instr[3]], dtype=np.float32))
    elif tag == ce._T_LUT:
        x = np.ascontiguousarray(v[instr[2]], dtype=np.float32)
        out = np.empty_like(x)
        instr[3](ce._to_c_float(out), ce._to_c_float(x), x.size)
        v[instr[1]] = out
    elif tag == ce._T_RESHAPE:
        v[instr[1]] = np.ascontiguousarray(v[instr[2]]).reshape(instr[3])
    elif tag == ce._T_TRANSPOSE:
        v[instr[1]] = np.ascontiguousarray(np.transpose(v[instr[2]], axes=instr[3]), dtype=np.float32)
    elif tag == ce._T_CONCAT:
        axis = instr[3]
        arrays = [np.ascontiguousarray(v[i], dtype=np.float32) for i in instr[2]]
        if axis < 0: axis = arrays[0].ndim + axis
        v[instr[1]] = np.concatenate(arrays, axis=axis)
    elif tag == ce._T_SLICE:
        v[instr[1]] = np.ascontiguousarray(np.asarray(v[instr[2]], dtype=np.float32)[instr[3]])
    elif tag == ce._T_REDUCE:
        v[instr[1]] = instr[3](np.asarray(v[instr[2]], dtype=np.float32), axis=instr[4], keepdims=instr[5])
    elif tag == ce._T_NEG:
        v[instr[1]] = -np.asarray(v[instr[2]], dtype=np.float32)
    elif tag == ce._T_SQUARE:
        x = np.asarray(v[instr[2]], dtype=np.float32)
        v[instr[1]] = x * x
    elif tag == ce._T_REPEAT:
        v[instr[1]] = np.repeat(np.asarray(v[instr[2]], dtype=np.float32), instr[3], axis=instr[4])
    elif tag == ce._T_EXPAND_DIMS:
        v[instr[1]] = np.expand_dims(np.ascontiguousarray(v[instr[2]]), axis=instr[3])
    elif tag == ce._T_MUX:
        cond = np.asarray(v[instr[2]], dtype=np.float32)
        v[instr[1]] = np.where(cond != 0.0, np.asarray(v[instr[4]], dtype=np.float32), np.asarray(v[instr[3]], dtype=np.float32))
    elif tag == ce._T_CMP_LE:
        v[instr[1]] = (np.asarray(v[instr[2]], dtype=np.float32) <= np.asarray(v[instr[3]], dtype=np.float32)).astype(np.uint8)
    elif tag == ce._T_ABS:
        v[instr[1]] = np.abs(np.asarray(v[instr[2]], dtype=np.float32))
    elif tag == ce._T_ARGMAX:
        v[instr[1]] = np.argmax(np.asarray(v[instr[2]], dtype=np.float32), axis=instr[3])
    elif tag == ce._T_CAST:
        v[instr[1]] = np.asarray(v[instr[2]]).astype(instr[3])

    dt = time.perf_counter() - t0
    name = TAG_NAMES.get(tag, f'TAG_{tag}')
    op_times[name] += dt
    op_counts[name] += 1

print(f"{'Op':<15} {'Count':>6} {'Total(ms)':>10} {'Avg(us)':>10}")
print("-" * 45)
total = 0
for op_name in sorted(op_times, key=op_times.get, reverse=True):
    t = op_times[op_name] * 1000
    c = op_counts[op_name]
    total += t
    print(f"{op_name:<15} {c:>6} {t:>10.1f} {t/c*1000:>10.0f}")
print(f"{'TOTAL':<15} {sum(op_counts.values()):>6} {total:>10.1f}")
