"""Profile per-op-type time in evaluate_c."""
import time
import numpy as np
from collections import defaultdict

from kllm.fabric import Fabric
from kllm.jit_optimizer import JitSession
from kllm.tokenizer import Tokenizer
from kllm.circuit_compiler import compile_decode_template
from kllm.circuit_executor import evaluate_c
from kllm.circuit_graph import Op

fabric = Fabric("./lossless_logic")
tok = Tokenizer("./lossless_logic/tokenizer")
prompt = tok.encode("<|user|>\nHello<|assistant|>\n")

# Use a real JitSession to get proper KV cache and inputs
session = JitSession(fabric)
logits = session.prefill(prompt)
next_id = int(np.argmax(logits[-1]))

# Warm-up decode
logits = session.decode_step(next_id)
next_id = int(np.argmax(logits[-1]))

# Now manually run one decode step with per-op timing
machine = session._machine
f = session.fabric

token_embed = f.embed_tokens[next_id:next_id + 1].astype(np.float32)
rope_cos = session._rope_cos[session.position:session.position + 1]
rope_sin = session._rope_sin[session.position:session.position + 1]

inputs = {
    machine.input_ids["token_embed"]: token_embed,
    machine.input_ids["rope_cos"]: rope_cos,
    machine.input_ids["rope_sin"]: rope_sin,
}
for li in range(session.num_layers):
    k_cache, v_cache = session.kv_cache[li]
    inputs[machine.input_ids[f"L{li}/cache_k"]] = k_cache
    inputs[machine.input_ids[f"L{li}/cache_v"]] = v_cache

graph = machine.graph

# Now profile per-op
from kllm.circuit_executor import _get_lib, _LUT_FN_MAP, _to_c_float

lib = _get_lib()
values = {}
order = graph.topological_order()

op_times = defaultdict(float)
op_counts = defaultdict(int)

for nid in order:
    node = graph.nodes[nid]
    t0 = time.perf_counter()
    
    inp = [values[i] for i in node.inputs]

    if node.op == Op.CONST:
        values[nid] = np.ascontiguousarray(node.params["value"])
    elif node.op == Op.INPUT:
        values[nid] = np.ascontiguousarray(inputs[nid])
    elif node.op == Op.LUT:
        fn_name = node.params["fn"]
        c_fn_name = _LUT_FN_MAP.get(fn_name)
        x = np.ascontiguousarray(inp[0], dtype=np.float32)
        out = np.empty_like(x)
        getattr(lib, c_fn_name)(_to_c_float(out), _to_c_float(x), x.size)
        values[nid] = out
    elif node.op in (Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.MAX):
        a = np.asarray(inp[0], dtype=np.float32)
        b = np.asarray(inp[1], dtype=np.float32)
        _np_binop = {Op.ADD: np.add, Op.SUB: np.subtract, Op.MUL: np.multiply, Op.DIV: np.divide, Op.MAX: np.maximum}
        values[nid] = _np_binop[node.op](a, b)
    elif node.op == Op.CMP_LE:
        a = np.asarray(inp[0], dtype=np.float32)
        b = np.asarray(inp[1], dtype=np.float32)
        values[nid] = (a <= b).astype(np.uint8)
    elif node.op == Op.MUX:
        cond = np.asarray(inp[0], dtype=np.float32)
        a = np.asarray(inp[1], dtype=np.float32)
        b = np.asarray(inp[2], dtype=np.float32)
        values[nid] = np.where(cond != 0.0, b, a)
    elif node.op == Op.NEG:
        values[nid] = -np.asarray(inp[0], dtype=np.float32)
    elif node.op == Op.ABS:
        values[nid] = np.abs(np.asarray(inp[0], dtype=np.float32))
    elif node.op == Op.SQUARE:
        x = np.asarray(inp[0], dtype=np.float32)
        values[nid] = x * x
    elif node.op == Op.MATMUL:
        a = np.ascontiguousarray(inp[0], dtype=np.float32)
        b = np.ascontiguousarray(inp[1], dtype=np.float32)
        values[nid] = np.matmul(a, b)
    elif node.op in (Op.SUM, Op.MAX_REDUCE, Op.MEAN):
        x = np.asarray(inp[0], dtype=np.float32)
        axis = node.params["axis"]
        keepdims = node.params.get("keepdims", False)
        _np_reduce = {Op.SUM: np.sum, Op.MAX_REDUCE: np.max, Op.MEAN: np.mean}
        values[nid] = _np_reduce[node.op](x, axis=axis, keepdims=keepdims)
    elif node.op == Op.ARGMAX:
        x = np.asarray(inp[0], dtype=np.float32)
        values[nid] = np.argmax(x, axis=node.params["axis"])
    elif node.op == Op.RESHAPE:
        values[nid] = np.ascontiguousarray(inp[0]).reshape(node.params["shape"])
    elif node.op == Op.TRANSPOSE:
        values[nid] = np.ascontiguousarray(np.transpose(inp[0], axes=node.params["axes"]), dtype=np.float32)
    elif node.op == Op.CONCAT:
        axis = node.params["axis"]
        arrays = [np.ascontiguousarray(a, dtype=np.float32) for a in inp]
        if axis < 0: axis = arrays[0].ndim + axis
        values[nid] = np.concatenate(arrays, axis=axis)
    elif node.op == Op.REPEAT:
        x = np.asarray(inp[0], dtype=np.float32)
        values[nid] = np.repeat(x, node.params["repeats"], axis=node.params["axis"])
    elif node.op == Op.SLICE:
        x = np.asarray(inp[0], dtype=np.float32)
        values[nid] = np.ascontiguousarray(x[node.params["slices"]])
    elif node.op == Op.CAST:
        values[nid] = np.asarray(inp[0]).astype(node.params["dtype"])
    elif node.op == Op.EXPAND_DIMS:
        values[nid] = np.expand_dims(np.ascontiguousarray(inp[0]), axis=node.params["axis"])
    else:
        raise ValueError(f"Unknown op: {node.op}")

    dt = time.perf_counter() - t0
    op_times[node.op.name] += dt
    op_counts[node.op.name] += 1

print(f"{'Op':<15} {'Count':>6} {'Total(ms)':>10} {'Avg(ms)':>10}")
print("-" * 45)
total = 0
for op_name in sorted(op_times, key=op_times.get, reverse=True):
    t = op_times[op_name] * 1000
    c = op_counts[op_name]
    total += t
    print(f"{op_name:<15} {c:>6} {t:>10.1f} {t/c:>10.3f}")
print(f"{'TOTAL':<15} {sum(op_counts.values()):>6} {total:>10.1f}")
