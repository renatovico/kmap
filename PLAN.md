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

## Phase 2 — Binary arithmetic circuits + missing activations

**Goal**: implement every IEEE-754 arithmetic operation as a gate
circuit on byte planes, plus compile the missing activation functions
(cos, sin), so no float operation remains.

### New module: `binary_ops.py`

| Function | Gate strategy |
|---|---|
| `circuit_add(a, b)` | Ripple-carry adder across 4 byte planes (uint8 carry chain) |
| `circuit_sub(a, b)` | Negate b (flip sign bit) + `circuit_add` |
| `circuit_mul(a, b)` | Byte-plane multiply with cross-plane carry propagation |
| `circuit_div(a, b)` | Reciprocal via rsqrt circuit + Newton-Raphson gate chain |
| `circuit_neg(a)` | XOR sign bit (bit 31) |
| `circuit_square(a)` | `circuit_mul(a, a)` (self-multiply) |
| `circuit_max(a, b)` | IEEE-754 sign-magnitude comparison gate (uint32 ordering trick) |
| `circuit_cmp_le(a, b)` | IEEE-754 magnitude compare → 0 or 1 output bit |
| `circuit_mux(sel, a, b)` | If sel=0 return a, else b (single LUT gate) |
| `circuit_sum(arr)` | Binary reduction tree of `circuit_add` |
| `circuit_argmax(arr)` | `circuit_max` reduction tree, tracking winning index |

### Float64 support (8 byte planes)

RMSNorm variance requires float64 precision.  Extend all binary ops
to 8-byte-plane gates (same shift+XOR strategy, double the planes).
Z3 proves correctness for uint8 LUTs on all 8 planes.

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
pytest tests/test_binary_ops.py
# circuit_add(a,b) == np.float32(a+b) for 10k random pairs
# circuit_mul(a,b) == np.float32(a*b) for 10k random pairs
# circuit_argmax matches np.argmax
# float64 ops match for 10k random pairs
# cos/sin LUTs match np.cos/np.sin for all 2^32 inputs
```

---

## Phase 3 — Circuit graph representation

**Goal**: represent the entire transformer forward pass as a single
composable DAG of gate nodes.  Every operation from the audit — both
arithmetic and structural — has a node type.

### New module: `circuit_graph.py`

```python
class GateNode:
    """Single node in the circuit DAG."""
    kind: str           # "const", "unary_lut", "binary", "mux",
                        # "wire", "memory", "reduce"
    inputs: list[int]   # indices of input nodes
    params: dict        # shift, mask, lut, etc.

class CircuitGraph:
    """DAG of gate nodes with topological ordering."""
    nodes: list[GateNode]
    inputs: list[int]   # graph input node indices
    outputs: list[int]  # graph output node indices

    def compose(self, other, op: str) -> CircuitGraph: ...
    def evaluate(self, *inputs) -> np.ndarray: ...
    def serialize(self, path: str) -> None: ...  # for C executor
    def to_dot(self) -> str: ...                 # Graphviz
```

### Node types for every operation

| Node kind | Maps to | Gates? |
|---|---|---|
| `"const"` | Weight bytes, RoPE freqs, causal -inf | 0 |
| `"unary_lut"` | SiLU, exp, rsqrt, cos, sin | 0 (LUT lookup) |
| `"binary"` | add, mul, div, sub, max, cmp | Yes |
| `"mux"` | embed lookup, causal mask select | 1 |
| `"wire"` | reshape, transpose, repeat, fanout | 0 |
| `"memory"` | KV cache append | 0 |
| `"reduce"` | sum, max, argmax over axis | Tree of binary gates |

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

## Phase 4 — C gate executor

**Goal**: replace the Python/NumPy circuit evaluator with a compiled
C extension — zero NumPy at inference time.

### New module: `_circuit_eval.c`

```c
// Walk DAG nodes in topological order
// Each node: read input bytes → LUT[input] → write output bytes
// Pure integer: <<, ^, array index
// Compile: cc -O3 -shared -fPIC -march=native -o _circuit_eval.so
```

- Python wrapper loads serialised `CircuitGraph`, calls into C
- Token IDs in, logit bytes out
- Memory-mapped LUT files (same mmap as current circuits)
- OpenMP parallelism across independent subgraphs (attention heads)

### Verification

```bash
pytest tests/test_c_executor.py
# C executor output == Python CircuitGraph.evaluate()
```

---

## Phase 5 — Offline optimisation

**Goal**: minimise gate count at compile time.  Weights and model
structure are fully known.

### Extensions to `optimizer.py`

| Optimisation | Description |
|---|---|
| **Constant folding** | Weight bytes are known → fold into gate nodes |
| **Dead gate elimination** | Remove nodes with no downstream consumers |
| **Gate merging** | Cascaded shift+XOR → single equivalent gate |
| **Subgraph dedup** | Identical attention heads share one subgraph |
| **RoPE const folding** | All trig values are position-only → fold into multiply gates |
| **QMC at boundaries** | Quine-McCluskey at subgraph I/O (8-variable blocks) |

### Verification

```bash
pytest tests/test_optimizer.py
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
