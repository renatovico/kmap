# kllm — Implementation Plan

Roadmap for turning the current hybrid system (gate weights + circuit
activations + NumPy arithmetic) into a **fully gate-based inference
engine** with JIT optimisation and FPGA export.

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
 │                     INFERENCE (per token)                        │
 └──────────────────────────────────────────────────────────────────┘

  ┌──────────────────────┐
  │    jit_optimizer      │  KV cache values become constants
  │  (fold, propagate,   │  → specialise circuit per decode step
  │   re-minimise)       │  → cache optimised subgraphs
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │   circuit_graph      │  Execute gate DAG (topological order)
  │   (evaluate)         │  All ops are gates: no float arithmetic
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │   hdl_export         │  (optional) Emit Verilog / VHDL
  │                      │  Gate graph → FPGA netlist
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
- **Remaining NumPy operations** (not yet circuits):
  - 7 matmuls per layer (Q, K, V, O, gate, up, down projections)
  - RMSNorm variance (`np.mean(x**2)`)
  - Softmax max / sum / divide
  - Residual additions
  - RoPE cos/sin multiplications
  - Attention score scaling

---

## Phase 2 — Binary arithmetic circuits

**Goal**: implement every IEEE-754 arithmetic operation as a gate
circuit on byte planes, so matmul, add, multiply, and divide can run
entirely through gates.

### New module: `binary_ops.py`

| Function | Gate strategy |
|---|---|
| `circuit_add(a, b)` | Ripple-carry adder across 4 byte planes (uint8 carry chain) |
| `circuit_mul(a, b)` | Byte-plane multiply with cross-plane carry propagation |
| `circuit_div(a, b)` | Reciprocal via existing rsqrt circuit + Newton-Raphson gate chain |
| `circuit_max(a, b)` | IEEE-754 sign-magnitude comparison gate (uint32 ordering trick) |
| `circuit_sum(arr)` | Binary reduction tree of `circuit_add` |

### Z3 correctness proofs

For each binary operation, Z3 proves that the byte-plane gate chain
matches the IEEE-754 result for all input pairs.  Proofs are stored
alongside the gate definitions and become FPGA testbenches later.

### Verification

```bash
pytest tests/test_binary_ops.py
# circuit_add(a,b) == np.float32(a+b) for 10k random pairs
# circuit_mul(a,b) == np.float32(a*b) for 10k random pairs
# etc.
```

---

## Phase 3 — Circuit graph representation

**Goal**: represent the entire transformer forward pass as a single
composable DAG of gate nodes.

### New module: `circuit_graph.py`

```python
class GateNode:
    """Single gate in the circuit."""
    kind: str           # "const", "unary", "binary"
    inputs: list[int]   # indices of input nodes
    shift: int          # s1 parameter
    mask: int           # XOR mask
    lut: np.ndarray     # optional: 256 or 256×256 entry table

class CircuitGraph:
    """DAG of gate nodes with topological ordering."""
    nodes: list[GateNode]
    inputs: list[int]   # graph input node indices
    outputs: list[int]  # graph output node indices

    def compose(self, other: CircuitGraph, op: str) -> CircuitGraph: ...
    def evaluate(self, *inputs: np.ndarray) -> np.ndarray: ...
    def to_dot(self) -> str: ...  # Graphviz visualisation
```

### Transformer → circuit graph compilation

Each operation maps to a subgraph:

| Transformer op | Circuit subgraph |
|---|---|
| Matmul (`x @ W.T`) | Grid of `circuit_mul` gates + `circuit_sum` reduction per output element |
| RMSNorm | Square gates → `circuit_sum` → rsqrt LUT → scale multiply gates |
| Softmax | `circuit_max` reduction → subtract → exp LUT → `circuit_sum` → `circuit_div` |
| RoPE | Constant cos/sin gates → `circuit_mul` → `circuit_add` |
| Residual | `circuit_add` gate |
| SiLU / exp / rsqrt | Existing unary LUT gate (from mmap or fast path) |

Each layer is a subgraph.  The full model composes 22 layer subgraphs
(for TinyLlama) + embedding + final norm + lm_head.

### Verification

```bash
pytest tests/test_circuit_graph.py
# Single-layer CircuitGraph.evaluate() matches circuit_model.py output
```

---

## Phase 4 — Offline optimisation

**Goal**: minimise gate count at compile time, before any tokens are
generated.  Weights and model structure are fully known.

### Extensions to `optimizer.py`

| Optimisation | Description |
|---|---|
| **Constant folding** | Weight bytes are known → fold into gate nodes, eliminate intermediates |
| **Dead gate elimination** | Remove nodes whose outputs are not consumed by any downstream node |
| **Gate merging** | Cascaded `(0xFF << s1) ^ mask` operations collapse into a single equivalent gate |
| **Subgraph dedup** | Identical attention heads (same weight gates) share a single subgraph instance |
| **QMC at boundaries** | Apply Quine-McCluskey minimisation at subgraph I/O boundaries (8-variable composable blocks) |

### Expected impact

- Constant folding alone should eliminate all weight-reconstruction
  gates (the shift+XOR is folded into the downstream multiply).
- Subgraph dedup reduces memory for models with grouped-query attention
  (e.g., TinyLlama: 4 KV heads shared across 32 query heads).

### Verification

```bash
pytest tests/test_optimizer.py
# Optimised graph produces identical output to unoptimised graph
# Gate count reduced (assert optimised.node_count < original.node_count)
```

---

## Phase 5 — Online JIT optimisation (per-token)

**Goal**: after each token decode, fold newly-known values (KV cache)
into the circuit and re-minimise — producing a progressively smaller
circuit as the sequence grows.

### New module: `jit_optimizer.py`

```python
class JITCircuitOptimizer:
    def __init__(self, base_graph: CircuitGraph): ...

    def specialize(self, kv_cache: dict) -> CircuitGraph:
        """Fold known KV values as constants, propagate, re-minimise."""
        ...

    def get_cached(self, prefix_hash: int) -> CircuitGraph | None:
        """Return previously optimised circuit for this prefix."""
        ...
```

### Strategy

| Step | What happens |
|---|---|
| Token 1 (cold) | Run full circuit graph.  Record KV cache values. |
| Token 2+ | Fold K, V for all past positions as constants in the attention subgraph.  Constant-propagate through softmax (known key magnitudes → max/sum simplify).  Re-minimise.  Run smaller circuit. |
| Prefix cache | Hash the token prefix → cache the optimised subgraph.  Same prefix (e.g., system prompt) reuses the same optimised circuit. |

### Amortisation constraint

Optimisation cost per token must be less than the gate-count savings
over subsequent tokens.  For long sequences, the savings compound:
each new token adds one KV entry but benefits from all previous
entries being folded.

### Verification

```bash
pytest tests/test_jit_optimizer.py
# JIT-optimised decode produces identical tokens to unoptimised
# Gate count decreases monotonically with sequence length
```

---

## Phase 6 — FPGA export

**Goal**: emit the optimised gate graph as synthesisable HDL.

### New module: `hdl_export.py`

| Feature | Description |
|---|---|
| **Verilog emitter** | `CircuitGraph` → `.v` module with LUT instantiations |
| **VHDL emitter** | Alternative for Xilinx/Altera targets |
| **Testbench generator** | Z3 proofs → SystemVerilog assertions |
| **Timing constraints** | Annotate critical paths from graph topology |
| **Pipelining** | Insert register stages at subgraph boundaries for clock frequency |

### Target platforms

- Xilinx Alveo / Zynq (LUT6 primitives)
- Intel/Altera Stratix / Cyclone (ALM primitives)
- Open-source: Lattice iCE40 via Yosys/nextpnr

### Verification

```bash
# Verilog simulation matches Python CircuitGraph.evaluate()
iverilog -o sim hdl/kllm_layer0.v hdl/tb_layer0.v && vvp sim
```

---

## Phase 7 — Integration and verification

### End-to-end invariant

At every phase, the system must produce **bit-exact** output compared
to HuggingFace:

```bash
kllm --mode compare --text "Hello world" --max-tokens 10
# ✅ Tokens match: [Hello, world, ...] (bit-exact)
```

### Test matrix

| Test | Phase | What it verifies |
|---|---|---|
| `test_binary_ops.py` | 2 | `circuit_add/mul/div` match IEEE-754 |
| `test_circuit_graph.py` | 3 | Single-layer DAG matches `circuit_model.py` |
| `test_optimizer.py` | 4 | Optimised graph = unoptimised output, fewer gates |
| `test_jit_optimizer.py` | 5 | JIT decode = full decode, gate count decreases |
| `test_hdl_export.py` | 6 | Verilog simulation matches Python evaluation |
| `test_compare.py` | all | Full pipeline bit-exact vs HuggingFace |

---

## Key decisions

| Decision | Rationale |
|---|---|
| Quine-McCluskey stays at 8 variables | Tractable (256 entries); graph-level opts compose above that |
| IEEE-754 byte-plane arithmetic (not integer) | Preserves bit-exact semantics; no conversion overhead |
| JIT = offline bulk + online per-token | Offline handles structure; online adapts to data |
| Circuit granularity: per-op AND per-layer | Everything is a circuit; layers are subgraphs of ops |
| FPGA is the end-goal hardware target | Gate representation must be directly synthesisable |
| Bit-exact invariant throughout | Never trade correctness for optimisation |
