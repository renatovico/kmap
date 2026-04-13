# kllm

**Compile a transformer into a gate circuit.  Optimise it.  Run it.**

kllm turns every operation in an LLM — weights, activations, matmul,
normalization, softmax, attention — into a **circuit graph** (DAG of
gate nodes) that is:

1. **Bit-exact** to HuggingFace output — no quantisation, no precision loss.
2. **Optimised** offline (constant folding, dead elimination) and online
   (JIT per-token KV cache folding).
3. **Executable** via a C gate executor — zero NumPy at inference time.
4. **Exportable** to Verilog/VHDL for FPGA synthesis.

## The core idea

### Everything is a circuit

Every operation in the transformer maps to a node in a directed acyclic
graph (DAG):

| Operation | Circuit node |
|---|---|
| **Weights** | CONST nodes (float32 values baked into the graph) |
| **Activation fns** (SiLU, exp, rsqrt, cos, sin) | LUT nodes (computed in C) |
| **Matrix multiply** | MATMUL node |
| **RMSNorm** | Composite: square → mean → rsqrt LUT → mul |
| **Softmax** | Composite: max → sub → exp LUT → sum → div |
| **RoPE** | Precomputed cos/sin CONST → mul → add |
| **Residual connections** | ADD nodes |
| **Attention** | MATMUL → scale → causal mask → softmax |
| **Reshape / transpose** | Zero-cost wiring nodes |

The full model is one graph.  The compiler builds it, the optimizer
shrinks it, the C executor runs it.

### Two-phase optimisation

**Offline (compile time)** — weights and model structure are known:

- **Constant folding**: all-const subgraphs collapse to a single node.
- **Dead gate elimination**: unreachable nodes are pruned.
- **Identity elimination**: trivial ops like `add(x, 0)` removed.

**Online (per-token JIT)** — KV cache values become known after each
decode step:

- Fold cached K/V values as constants into the attention circuit.
- Re-optimise the specialised circuit — progressively smaller.
- Cache optimised circuits by prefix hash.

### FPGA as the target

The optimised DAG exports directly to Verilog/VHDL.  Every node maps
to synthesisable RTL — LUT nodes become BRAM ROMs, arithmetic nodes
become FPU instances (or vendor IP), wiring nodes are free.

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python ≥ 3.13.  Runtime dependency: **numpy** (≥ 1.26).
torch + transformers needed only at compile time (model download).

## Usage

### 1. Compile model to circuit graph

```bash
kllm --mode compile --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

Downloads model from HuggingFace, extracts weights, builds circuit
graph, optimises, and serialises to disk.

### 2. Run inference

```bash
kllm --mode infer --text "The capital of France is" --max-tokens 20
```

### 3. Compare with HuggingFace (requires torch)

```bash
kllm --mode compare --text "Hello world" --max-tokens 10
```

### 4. Export to HDL

```bash
kllm --mode export-hdl --output-format verilog
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--mode` | *(required)* | `compile`, `infer`, `compare`, `export-hdl` |
| `--model` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | HuggingFace model ID or local path |
| `--save-dir` | `./lossless_logic` | Where compiled circuit graph is stored |
| `--text` | *(interactive)* | Prompt text |
| `--max-tokens` | `50` | Max new tokens for infer/compare modes |
| `--output-format` | `verilog` | HDL format: `verilog` or `vhdl` |

## Project structure

```
src/kllm/
├── cli.py              # CLI entry point
├── fabric.py           # Model weight loader (HuggingFace → float32 arrays)
├── tokenizer.py        # Pure-Python BPE tokenizer
├── circuit_graph.py    # Core DAG — circuit builder + reference evaluator
├── circuit_compiler.py # Compile transformer into CircuitGraph
├── circuit_executor.py # C-accelerated graph evaluator (ctypes wrapper)
├── graph_optimizer.py  # Constant folding + dead elimination + identity elim
├── jit_optimizer.py    # Per-token JIT optimization with KV cache folding
├── hdl_export.py       # Verilog / VHDL / testbench / resource estimation
├── binary_ops.py       # IEEE-754 reference implementations (golden model)
├── compare.py          # HuggingFace vs kllm comparison
├── model.py            # Reference NumPy LLaMA (used by compare)
├── bitops.py           # IEEE-754 byte-plane extract / repack
├── optimizer.py        # Quine-McCluskey boolean minimisation
└── device.py           # GPU / CPU abstraction

csrc/
└── _circuit_eval.c     # C tensor ops library
```

## Running tests

```bash
pytest          # 265 tests
```

## Current status

| Component | Status |
|---|---|
| Circuit graph DAG + reference evaluator | ✅ |
| Transformer → circuit graph compilation | ✅ |
| C gate executor (zero NumPy runtime) | ✅ |
| Offline graph optimisation | ✅ |
| Online JIT per-token optimisation | ✅ |
| FPGA export (Verilog/VHDL) | ✅ |
| Integration tests (265 tests) | ✅ |
| Unified pipeline (no Z3) | 🔲 Phase 9 |

See [PLAN.md](PLAN.md) for the full roadmap.

## Compiled output layout

```
lossless_logic/
├── weights/
│   ├── config.json           # model config (dims, heads, theta)
│   ├── embed_tokens.npy      # (vocab_size, hidden_size) float32
│   ├── final_norm_weight.npy # (hidden_size,) float32
│   ├── lm_head.npy           # (vocab_size, hidden_size) or tied
│   └── layer_N/
│       ├── q_proj.npy … down_proj.npy   # projection weights
│       ├── input_layernorm_weight.npy
│       └── post_attention_layernorm_weight.npy
├── circuit_graph/            # serialised CircuitGraph
└── tokenizer/
    ├── tokenizer.json
    └── tokenizer_config.json
```

## License

This project is dual-licensed:

- **AGPL-3.0-or-later** — free for open-source use under the terms of the
  [GNU Affero General Public License v3](LICENSE).
- **Commercial License** — for proprietary / closed-source use, contact
  Renato Augusto Viço Elias (renato@s2n.es).
