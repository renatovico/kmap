# kllm — Implementation Plan

Roadmap for turning the current hybrid system (gate weights + circuit
activations + NumPy arithmetic) into a **fully gate-based inference
engine** — zero NumPy at runtime — with JIT optimisation and FPGA export.

## Architecture overview

```
 ┌──────────────────────────────────────────────────────────────────┐
 │                       COMPILE TIME (offline)                     │
 └──────────────────────────────────────────────────────────────────┘

  HuggingFace model
        │
        ▼
  ┌─────────────┐     ┌──────────────┐     ┌──────────────────────┐
  │  compiler   │────▶│ circuit_graph│────▶│     optimizer        │
  │  (Z3 gates) │     │ (build DAG)  │     │ (constant fold,      │
  └─────────────┘     └──────────────┘     │  dead elim, merge,   │
                                           │  QMC minimise)       │
  ┌─────────────┐                          └──────────┬───────────┘
  │ops_compiler │                                     │
  │(full-domain │─── activation circuits ────────────▶│
  │  mmap LUTs) │                                     ▼
  └─────────────┘                          ┌──────────────────────┐
                                           │  Optimised circuit   │
                                           │  graph on disk       │
                                           └──────────────────────┘

 ┌──────────────────────────────────────────────────────────────────┐
 │              INFERENCE (C runtime, zero NumPy)                   │
 └──────────────────────────────────────────────────────────────────┘

  ┌──────────────────────┐
  │    jit_optimizer      │  KV cache values become constants
  │  (fold, propagate,   │  → specialise circuit per decode step
  │   re-minimise)       │  → cache optimised subgraphs
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │   C gate executor    │  Walk gate DAG (topological order)
  │  (_circuit_eval.c)   │  Pure integer bit ops: shift, XOR, LUT index
  │                      │  No float arithmetic, no NumPy
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │   hdl_export         │  (optional) Emit Verilog / VHDL
  │                      │  Gate graph → FPGA netlist
  └──────────────────────┘
```

---

## Complete operation audit

Every NumPy call during inference must map to a gate circuit or a DAG
structural primitive.  Nothing can remain as float arithmetic.

### Arithmetic operations → gate circuits

| Current NumPy op | Count per forward | Gate circuit replacement |
|---|---|---|
| `x @ W.T` (matmul) | 7 per layer + 1 lm_head | `circuit_mul` grid + `circuit_add` reduction tree |
| `np.matmul(q, k.T)` | 1 per layer | Same (attention scores) |
| `np.matmul(attn, v)` | 1 per layer | Same (context) |
| `x * scale` (element-wise mul) | ~6 per layer | `circuit_mul` gate |
| `x + y` (residual add) | 4 per layer | `circuit_add` (ripple-carry on byte planes) |
| `x - m` (softmax shift) | 1 per layer | `circuit_sub` (negate + add) |
| `e / sum` (softmax norm) | 1 per layer | `circuit_div` |
| `x.max(axis)` (softmax) | 1 per layer | `circuit_max` reduction tree |
| `e.sum(axis)` (softmax) | 1 per layer | `circuit_add` reduction tree |
| `mean(x²)` (RMSNorm var) | 2 per layer + 1 final | `circuit_square` + `circuit_add` tree + `circuit_div` |
| `silu(x)` | 1 per layer | ✅ Already a circuit (full-domain LUT) |
| `exp(x)` | 1 per layer (softmax) | ✅ Already a circuit (full-domain LUT) |
| `rsqrt(x)` | 2 per layer + 1 final | ✅ Already a circuit (full-domain LUT) |
| `np.cos(emb)`, `np.sin(emb)` | 1 per forward (RoPE) | **New**: full-domain LUT (same as SiLU/exp/rsqrt) |
| `theta ** (i/d)` | 1 per forward (RoPE) | Precompute as constant gates at compile time |
| `np.outer(t, freq)` | 1 per forward (RoPE) | `circuit_mul` grid |
| `x * cos + rot * sin` | 2 per layer (RoPE) | `circuit_mul` + `circuit_add` |
| `scores += causal` | 1 per layer | `circuit_add` gate |
| `argmax(logits)` | 1 per token | `circuit_argmax` (max reduction returning index) |
| `float64` widening (RMSNorm) | 2 per layer + 1 final | 8 byte-plane gates (float64 = 8 bytes) |

### Structural operations → DAG primitives

| Current NumPy op | Gate-world equivalent | Notes |
|---|---|---|
| `embed_tokens[token_ids]` | **MUX circuit**: token ID bits drive select into constant weight table | |
| `np.concatenate` (KV cache) | **Memory node**: append-only; values become constants after write | |
| `np.repeat` (GQA) | **Wire fanout**: same output feeds multiple inputs (zero gates) | |
| `np.where(cond, a, b)` | **Comparator + MUX**: `circuit_cmp_le` → `circuit_mux` | |
| `reshape` / `transpose` | **Wire routing**: reorder connections in DAG (zero gates) | |
| `.view(uint32)` / `.astype()` | **Identity wires**: bits are bits in gate world | |
| `np.arange(...)` | **Constant generator**: precomputed at compile time | |
| `np.empty` / `np.zeros` | **Register allocation**: mapped to output wires | |
| `np.unique` + inverse | **Eliminated**: C executor doesn't need dedup | |

### Executor runtime

The circuit executor itself replaces NumPy entirely:

- **C extension** (`_circuit_eval.c`) walks the DAG in topological order
- Each node: read input bits → LUT lookup → write output bits
- Pure integer operations: `<<`, `^`, array index
- Same pattern as existing `_codebook_gemv.c`
- Python wrapper only for I/O: load graph, feed token IDs, read logits

---
  └──────────────────────┘
```

## Phase 1 — Byte-plane gate infrastructure ✅ (done)

What exists today.

### Weight gates

- `compiler.py` decomposes float32 weights into 4 byte planes and
  Z3-proves a `(shift, mask)` constant gate for all 256 byte values.
- `fabric.py` reconstructs float32 weights at load time via
  `(0xFF << s1_lut[b]) ^ mask_lut[b]` — pure shift + XOR.
- Stored as `.npz` gate arrays per layer.

### Activation circuits

- `ops_compiler.py` evaluates SiLU, exp, rsqrt for all 2^32 float32
  bit patterns and writes 4 byte-plane mmap files per operation
  (~4 GB each, 12 files, ~48 GB total).
- `circuits.py` `ArithmeticUnit` loads mmap files and executes
  activations as array-index lookups — O(1) per value.
- Fast path: Z3-verified NumPy formulas (SIMD) are mathematically
  identical to the mmap tables.

### Boolean minimisation

- `optimizer.py` applies Quine-McCluskey to 8-variable gate LUTs.
- Materialises pre-computed float32 weight files (`optimized/`) for
  instant mmap loading, bypassing gate execution.

### Inference

- `circuit_model.py` runs the LLaMA transformer with gate-loaded
  weights and circuit activations.
- **All remaining NumPy operations listed in the audit above.**

---

## Phase 2 — Circuit graph + reference ops + missing activations ✅ (done)

**Goal**: define the circuit DAG representation, provide reference
implementations for every IEEE-754 operation, and compile missing
activation functions (cos, sin).

### Architecture correction

The original plan tried to implement IEEE-754 arithmetic in Python
using integer bit operations (shift, XOR, mask on uint32).  This was
wrong — it was a hypothetical "how to run" that:

1. Was fragile and incomplete (edge cases, rounding, denormals).
2. Was the **executor's** job, not the graph's.
3. Added no value — NumPy already IS the reference FPU.

The correct separation is:
- **`circuit_graph.py`** — DAG of gate nodes (WHAT to compute)
- **`binary_ops.py`** — reference implementation (NumPy golden model)
- **C gate executor** (Phase 4) — actual byte-plane execution (HOW to run)

### New module: `circuit_graph.py`

The core data structure.  Every operation is a node in the DAG:

| Node kind | Maps to | Gates? |
|---|---|---|
| `const` | Weight bytes, RoPE freqs, causal -inf | 0 |
| `input` | Token IDs, position indices | 0 |
| `lut` | SiLU, exp, rsqrt, cos, sin | 0 (LUT lookup) |
| `add/sub/mul/div` | IEEE-754 float32 arithmetic | Yes |
| `neg/abs` | Sign-bit manipulation | 1 gate |
| `max/cmp_le/mux` | Comparison and selection | Yes |
| `matmul` | Matrix multiply | Grid of mul+sum |
| `sum/max_reduce/argmax/mean` | Reductions | Tree of binary gates |
| `reshape/transpose/concat/repeat/slice` | Wire routing | 0 |
| `cast` | Type conversion | 0 |

Composite subgraphs: `softmax()`, `rms_norm()` decompose into
primitive nodes automatically.

Reference evaluator: `evaluate(graph, inputs)` walks the DAG in
topological order using NumPy — produces the golden output that the
C executor must match bit-for-bit.

### Module: `binary_ops.py` (reference)

| Function | What it does |
|---|---|
| `circuit_add(a, b)` | NumPy float32 addition (reference for executor) |
| `circuit_sub(a, b)` | NumPy float32 subtraction |
| `circuit_mul(a, b)` | NumPy float32 multiplication |
| `circuit_div(a, b)` | NumPy float32 division |
| `circuit_neg(a)` | XOR sign bit (real bit op) |
| `circuit_abs(a)` | AND clear sign bit (real bit op) |
| `circuit_max(a, b)` | `np.maximum` (reference) |
| `circuit_cmp_le(a, b)` | `a <= b` → uint8 (reference) |
| `circuit_mux(sel, a, b)` | `np.where` selection (reference) |
| `circuit_sum(arr)` | `np.sum` (reference) |
| `circuit_argmax(arr)` | `np.argmax` (reference) |
| `circuit_matmul(a, b)` | `a @ b` (reference) |
| `circuit_softmax(x)` | max → sub → exp → sum → div (reference) |

### Float64 support (8 byte planes)

RMSNorm variance requires float64 precision.  Float64 reference ops
provided (`circuit_add_f64`, etc.).  C executor will use 8 byte planes.

### New activation circuits

| Function | Strategy |
|---|---|
| `cos(x)` | Full-domain LUT: 4 byte-plane mmap files (same as SiLU/exp/rsqrt) |
| `sin(x)` | Full-domain LUT: 4 byte-plane mmap files |

Compiled by `ops_compiler.py`.  Adds 8 files (~32 GB) to circuits/.

### RoPE frequencies as constants

`theta ** (i/d)` and `np.outer(t, inv_freq)` depend only on model
config and sequence position.  Precompute at compile time and store
as constant gate arrays.

### Verification

```bash
pytest tests/test_binary_ops.py      # 24 tests — reference ops
pytest tests/test_circuit_graph.py   # 35 tests — DAG + evaluator
# binary_ops: bit-exact to NumPy for all operations
# circuit_graph: graph construction, topological order, gate counting
# evaluator: arithmetic, comparison, reduction, wiring, LUT, composite
```

---

## Phase 3 — Transformer → circuit graph compilation ✅ (done)

**Goal**: compile the full `circuit_model.py` transformer into a
`CircuitGraph` DAG.  The graph structure was built in Phase 2;
this phase populates it with the actual model.

> `circuit_graph.py` (Phase 2) already has:
> - `CircuitGraph` with all node types (const, add, mul, matmul, lut, etc.)
> - `evaluate()` reference evaluator (NumPy golden model)
> - Composite subgraphs: `softmax()`, `rms_norm()`
> - Topological ordering, gate counting
> - `keepdims` support on reductions, `expand_dims` node

### New module: `circuit_compiler.py`

`compile_model(fabric, token_ids, start_pos)` walks the model and
emits a complete `CircuitGraph`.  Every operation is a node:

- Weights → `const` nodes (transposed for matmul)
- RoPE cos/sin → `const` nodes (precomputed)
- Embedding → `const` node (selected rows for fixed token_ids)
- Each layer: `rms_norm` → `attention` → `residual` → `rms_norm` → `mlp` → `residual`
- Attention: QKV matmul → reshape → RoPE → GQA repeat → scores → causal mask → softmax → context
- MLP: gate/up matmul → SiLU LUT → mul → down matmul
- Final: `rms_norm` → `matmul(lm_head)` → logits

### Verification

```bash
pytest tests/test_circuit_compiler.py   # 11 tests
# Graph evaluation matches NumPy reference (single + multi token)
# Gate count grows with layers
# Argmax on logits produces valid token ID
```

### Transformer → circuit graph compilation

| Transformer op | Circuit subgraph |
|---|---|
| Embedding lookup | MUX tree: token ID bits select from constant weight rows |
| Matmul (`x @ W.T`) | Grid of `circuit_mul` + `circuit_sum` reduction per output |
| RMSNorm | `circuit_square` → `circuit_sum` → `circuit_div` → rsqrt LUT → `circuit_mul` |
| Softmax | `circuit_max` reduce → `circuit_sub` → exp LUT → `circuit_sum` → `circuit_div` |
| RoPE | Const cos/sin → `circuit_mul` → `circuit_add` |
| Causal mask | `circuit_cmp_le` → `circuit_mux` (select 0 or -inf) → `circuit_add` |
| GQA repeat | Wire fanout (same outputs feed multiple heads) |
| KV cache | Memory nodes (append; values become constants for JIT) |
| MLP | Matmul → SiLU LUT → `circuit_mul` (gate×up) → matmul |
| Token select | `circuit_argmax` over logits |

---

## Phase 4 — C gate executor  ✅ DONE

**Goal**: replace the Python/NumPy circuit evaluator with a compiled
C extension — zero NumPy at inference time.

### Implementation

| File | Purpose |
|---|---|
| `csrc/_circuit_eval.c` | C shared library — ~500 lines implementing all primitive tensor ops |
| `src/kllm/circuit_executor.py` | Python ctypes wrapper — `evaluate_c()` replaces `evaluate()` |
| `tests/test_c_executor.py` | 38 tests — every op + composites + full compiled model |

**C ops implemented** (all match NumPy reference bit-for-bit):
- **Arithmetic**: add, sub, mul, div (with full broadcasting)
- **Unary**: neg (sign XOR), abs (sign AND), square
- **LUT activations**: silu, exp, rsqrt, cos, sin (upcast to float64)
- **Comparison**: cmp_le, mux (conditional select)
- **Matmul**: arbitrary-dimension batched ((...,m,k) @ (...,k,n))
- **Reductions**: sum, max_reduce, mean, argmax (along any axis)
- **Wiring**: transpose, repeat, slice, copy (reshape/expand_dims)

**Architecture**: Python evaluation loop (trivial overhead) dispatches
each tensor op to C via ctypes.  NumPy arrays serve as containers;
all computation is in C.

```bash
# Compile
cc -O3 -shared -fPIC -march=native -o csrc/_circuit_eval.so csrc/_circuit_eval.c -lm

# Auto-compiles if .so is missing
from kllm.circuit_executor import evaluate_c
values = evaluate_c(graph)  # same API as evaluate()
```

### Graph serialization

CircuitGraph now supports `serialize(path)` / `deserialize(path)`:
- `nodes.bin`: packed node descriptors (op, inputs, params)
- `topo.bin`: topological order (uint32 array)
- `const_NNN.bin` + `.json`: raw tensor data + shape/dtype metadata

### Verification

```bash
pytest tests/test_c_executor.py  # 38 passed
# C executor output == Python CircuitGraph.evaluate()
```

---

## Phase 5 — Offline optimisation  ✅ DONE

**Goal**: minimise gate count at compile time.  Weights and model
structure are fully known.

### Implementation: `graph_optimizer.py`

Graph-level DAG optimizations (distinct from `optimizer.py` QMC byte-level):

| Pass | Description |
|---|---|
| **Identity elimination** | Remove trivial ops: add(x,0), mul(x,1), neg(neg(x)), sub(x,0), div(x,1) |
| **Constant folding** | Evaluate nodes with all-constant inputs, replace chains with CONST |
| **Dead node elimination** | Remove nodes not reachable from outputs, renumber contiguously |

**Key result**: A compiled model graph (all fixed token_ids, all weights
are constants) folds completely to a **single CONST node** — the
precomputed logits tensor.  84 nodes → 1 node for a 1-layer test model.

For graphs with INPUT nodes (dynamic token_ids), only the constant
subgraph folds — the input-dependent compute structure is preserved.

```python
from kllm.graph_optimizer import optimize_graph, optimization_stats
opt_graph, id_map = optimize_graph(graph, output_ids=[logits_id])
stats = optimization_stats(graph, opt_graph)
# stats["gate_reduction_pct"] == 100.0 for all-const graphs
```

### Verification

```bash
pytest tests/test_graph_optimizer.py  # 23 passed
# Optimised graph == unoptimised output, fewer gates
```

---

## Phase 6 — Online JIT optimisation (per-token)

**Goal**: after each decode, fold KV cache values as constants and
re-minimise — progressively smaller circuit.

### New module: `jit_optimizer.py`

| Step | What happens |
|---|---|
| Token 1 (cold) | Run full circuit graph.  Record KV values. |
| Token 2+ | Fold K, V as constants → constant-propagate through softmax → re-minimise → run smaller circuit. |
| Prefix cache | Hash prefix → reuse optimised circuit for same prefix. |

### Verification

```bash
pytest tests/test_jit_optimizer.py
# JIT tokens == full tokens, gate count decreases with seq length
```

---

## Phase 7 — FPGA export

**Goal**: emit optimised gate graph as synthesisable HDL.

### New module: `hdl_export.py`

| Feature | Description |
|---|---|
| **Verilog emitter** | `CircuitGraph` → `.v` with LUT instantiations |
| **VHDL emitter** | Xilinx/Altera alternative |
| **Testbench generator** | Z3 proofs → SystemVerilog assertions |
| **Pipelining** | Register stages at subgraph boundaries |

### Target platforms

- Xilinx Alveo / Zynq (LUT6)
- Intel/Altera Stratix / Cyclone (ALM)
- Lattice iCE40 via Yosys/nextpnr

---

## Phase 8 — Integration and verification

### End-to-end invariant

At every phase, bit-exact match with HuggingFace:

```bash
kllm --mode compare --text "Hello world" --max-tokens 10
```

### Test matrix

| Test | Phase | What it verifies |
|---|---|---|
| `test_binary_ops.py` | 2 | add/mul/div/argmax match IEEE-754; float64; cos/sin LUTs |
| `test_circuit_graph.py` | 3 | Single-layer DAG matches `circuit_model.py`; all node types |
| `test_c_executor.py` | 4 | C output == Python evaluation |
| `test_optimizer.py` | 5 | Optimised == unoptimised, fewer gates |
| `test_jit_optimizer.py` | 6 | JIT tokens == full tokens |
| `test_hdl_export.py` | 7 | Verilog sim matches C executor |
| `test_compare.py` | all | Full pipeline bit-exact vs HuggingFace |

---

## Key decisions

| Decision | Rationale |
|---|---|
| Zero NumPy at inference | C extension executor; Python only for I/O |
| Float64 kept for RMSNorm | 8 byte planes; precision preserved |
| cos/sin as full-domain LUTs | Same strategy as SiLU/exp/rsqrt; adds ~32 GB |
| RoPE precomputed as constants | Position-only → compile-time constant gates |
| Embedding = MUX circuit | Token ID bits drive select into constant table |
| KV cache = memory nodes | Write once, then become constants for JIT |
| Causal mask = comparator + MUX | Position compare → select 0 or -inf |
| Quine-McCluskey stays at 8 variables | Tractable; graph-level opts compose above |
| IEEE-754 byte-plane arithmetic | Preserves bit-exact semantics |
| JIT = offline bulk + online per-token | Offline: structure; online: adapts to data |
| FPGA is the end-goal target | Gate representation must be synthesisable |
| Bit-exact invariant throughout | Never trade correctness for optimisation |
