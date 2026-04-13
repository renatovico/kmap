# kllm

**Compile a transformer into a gate circuit.  Optimise it.  Run it.**

kllm turns every operation in an LLM — weights, activations, matmul,
normalization, softmax, attention — into a **boolean gate circuit** that
the Z3 SMT solver proves correct.  The goal is a system where:

1. The entire inference pass is one large, composable gate graph.
2. An offline + online (JIT) optimiser **minimises gate count** before
   and during token generation.
3. The gate graph maps directly to FPGA LUT primitives — no FPU needed.

The result is **bit-exact** to HuggingFace output.  This is not
quantisation — no precision is lost.  Every bit of every original
float32 weight is preserved through a change of **representation**,
not a change of value.

## The core idea

### Byte-plane decomposition

A 32-bit float is 4 bytes.  Each byte takes one of 256 values.  Z3
proves, for every possible byte value, a constant gate:

$$\text{target} = (0\text{xFF} \ll s_1) \oplus \text{mask}$$

where $s_1 \in [0, 7]$ and $\text{mask} \in [0, 255]$.  Two 256-entry
LUTs (shifts + masks) reconstruct **any** byte via one shift and one
XOR.  Four planes reconstruct the full float32 — lossless.

### Everything is a circuit

The key insight is that this gate decomposition generalises beyond
weights and activation functions to **every arithmetic operation** in
the transformer:

| Operation | Gate circuit |
|---|---|
| **Weight storage** | 4 byte-plane constant gates per float32 |
| **Activation fns** (SiLU, exp, rsqrt) | Full-domain ($2^{32}$) byte-plane mmap LUTs |
| **Matrix multiply** | Grid of binary multiply gates + add-reduction tree |
| **RMSNorm** | Square gates → sum reduction → rsqrt circuit → scale |
| **Softmax** | Max reduction → subtract → exp circuit → sum → divide |
| **RoPE** | Precomputed cos/sin constant gates → multiply gates |
| **Residual connections** | Ripple-carry add gates on byte planes |
| **Attention scores** | Matmul circuit → scale gate → causal mask → softmax circuit |

Each layer is a **subgraph** of gate nodes.  The full model is a
composition of layer subgraphs into a single directed acyclic graph
(DAG).

### Two-phase optimisation

**Offline (compile time)** — weights and model structure are known:

- **Constant folding**: weight bytes are constants → fold into gates,
  eliminate intermediate nodes.
- **Dead gate elimination**: prune gates whose outputs are unused.
- **Gate merging**: cascaded shift+XOR collapses into a single gate.
- **Subgraph dedup**: identical attention heads share a single circuit.
- **Quine-McCluskey minimisation**: 8-variable boolean functions →
  minimal sum-of-products representation.

**Online (per-token JIT)** — KV cache values become known after each
decode step:

- Fold cached K/V values as constants into the attention circuit.
- Constant-propagate through softmax (known key magnitudes simplify
  max/sum reductions).
- Re-minimise the specialised circuit and cache it — same prefix
  reuses the same optimised circuit.
- Amortise: optimisation cost must be less than gate savings over
  subsequent tokens.

This is analogous to a JIT compiler: token 1 runs the full circuit;
token 2+ runs a progressively more specialised (smaller) circuit.

### FPGA as the target

Gate circuits map directly to FPGA lookup-table primitives.  The
optimised DAG can be exported to Verilog/VHDL for synthesis.  Every
gate in the graph is a proven-correct truth table — the Z3 proofs
become the testbench.

### Why full-domain?

Pre-computing every possible float32 input ($2^{32}$ values) means no
value is ever "unseen".  Every prompt, every temperature, every token
produces activation values that already exist in the compiled tables.
No hash table, no on-demand compilation, no fallback — a single array
index per value.

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python ≥ 3.13.  Only runtime dependency is **numpy** (≥ 1.26).
torch, transformers, and z3-solver are needed only at compile time.

## Usage

### 1. Compile weights into gate fabric

```bash
kllm --mode compile --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

### 2. Compile full-domain activation circuits

```bash
kllm --mode compile-circuits
```

This creates 12 byte-plane files (~48 GB) in `./lossless_logic/circuits/`.

### 3. Run inference (no torch, no HuggingFace)

```bash
kllm --mode inference --text "The capital of France is" --max-tokens 20
```

### 4. Stream tokens

```bash
kllm --mode stream --text "Hello" --max-tokens 50
```

### 5. Compare with HuggingFace (requires torch)

```bash
kllm --mode compare --text "Hello world" --max-tokens 10
```

### 6. Full pipeline (compile + compile-circuits + inference)

```bash
kllm --mode full --text "Hello world"
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--mode` | *(required)* | `compile`, `compile-circuits`, `inference`, `generate`, `stream`, `compare`, or `full` |
| `--model` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | HuggingFace model ID or local path |
| `--save-dir` | `./lossless_logic` | Where compiled fabric is stored |
| `--text` | *(interactive)* | Prompt text |
| `--solver-timeout` | `200` | Z3 timeout per pattern in ms |
| `--max-layers` | all | Limit layers (useful for quick tests) |
| `--max-tokens` | `50` | Max new tokens for generate/compare modes |

## Project structure

```
src/kllm/
├── bitops.py         # Lossless IEEE-754 byte-plane extract / repack
├── compiler.py       # Z3 weight gate synthesis (compile mode)
├── ops_compiler.py   # Full-domain activation circuit compilation
├── optimizer.py      # Quine-McCluskey boolean minimisation + weight materialisation
├── circuits.py       # Z3 gate primitives + ArithmeticUnit (mmap byte planes)
├── circuit_model.py  # LLaMA transformer with circuit execution + streaming
├── fabric.py         # Gate loader — shift+XOR → float32 weight matrices
├── model.py          # Reference NumPy LLaMA (used by compare)
├── inference.py      # Orchestrator: tokenizer + fabric + circuits → generate
├── tokenizer.py      # Pure-Python BPE tokenizer (no HuggingFace)
├── compare.py        # HuggingFace vs kllm generation comparison
├── device.py         # GPU / CPU abstraction (CuPy ↔ NumPy)
├── cli.py            # CLI entry point
├── binary_ops.py     # IEEE-754 reference implementations (golden model for executor)
├── circuit_graph.py  # DAG of gate nodes — circuit builder + reference evaluator
├── circuit_compiler.py # Compile LLaMA transformer into a CircuitGraph
├── circuit_executor.py # C-accelerated graph evaluator (ctypes wrapper)
├── graph_optimizer.py # DAG-level constant folding + dead elimination
├── jit_optimizer.py  # Per-token JIT optimization with KV cache folding
└── hdl_export.py     # Gate graph → Verilog / VHDL / testbench / resource estimation

csrc/
└── _circuit_eval.c   # C tensor ops library (all primitive ops for the graph)
```

## Running tests

```bash
pytest          # 251 tests
```

## Current status

The project implements **Phase 1** (weights + activations as circuits)
with a working inference pipeline.  See [PLAN.md](PLAN.md) for the
full roadmap toward the complete gate-circuit architecture.

| Component | Status |
|---|---|
| Weight gate compilation (Z3) | ✅ Complete |
| Activation circuits (SiLU, exp, rsqrt) | ✅ Full-domain mmap |
| Quine-McCluskey gate minimisation | ✅ 8-variable |
| Pure-NumPy inference (no torch) | ✅ Bit-exact |
| Binary arithmetic reference ops | ✅ Phase 2 |
| Circuit graph DAG + reference evaluator | ✅ Phase 2 |
| cos/sin activation LUTs + float64 support | 🔲 Phase 2 (cos/sin compile pending) |
| Transformer → circuit graph compilation | ✅ Phase 3 |
| C gate executor (zero NumPy runtime) | ✅ Phase 4 |
| Offline graph optimisation | ✅ Phase 5 |
| Online JIT per-token optimisation | ✅ Phase 6 |
| FPGA export (Verilog/VHDL) | ✅ Phase 7 |

## Compiled output layout

```
lossless_logic/
├── meta.npz              # model config (head counts, dims, theta)
├── globals.npz           # embed_tokens + lm_head gate arrays
├── layer_0.npz … 21.npz # per-layer weight gates (7 projections each)
├── circuits.npz          # constant LUT + op metadata
├── circuits/
│   ├── silu_p0.bin … p3.bin    # 4 × 4 GB (full float32 domain)
│   ├── exp_p0.bin  … p3.bin    # 4 × 4 GB
│   ├── rsqrt_p0.bin … p3.bin   # 4 × 4 GB
│   ├── cos_p0.bin … p3.bin     # 4 × 4 GB (RoPE)
│   └── sin_p0.bin … p3.bin     # 4 × 4 GB (RoPE)
└── tokenizer/
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── chat_template.jinja
```

## License

This project is dual-licensed:

- **AGPL-3.0-or-later** — free for open-source use under the terms of the
  [GNU Affero General Public License v3](LICENSE).
- **Commercial License** — for proprietary / closed-source use, contact
  Renato Augusto Viço Elias (renato@s2n.es).
