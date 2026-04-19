# kllm

**Compile a transformer into a circuit.  Optimise it.  Run it on a virtual device.**

kllm turns every operation in an LLM — weights, activations, matmul,
normalization, softmax, attention — into a **circuit graph** (DAG of
gate nodes) and runs it on a virtual device emulated entirely in C.

1. **Bit-exact** to HuggingFace output — INT8 quantised weights,
   float32 activations.
2. **Compiled offline**: constant folding, dead elimination, identity
   elimination — the full model shrinks to a single datapath circuit.
3. **Executed in C**: one `processor_infer_bytes()` call does everything —
   BPE encode, prefill, decode, BPE decode, KV cache — zero Python in
   the inference loop.
4. **Exportable** to Verilog/VHDL for FPGA synthesis.

## The core idea

### Everything is a circuit

Every operation in the transformer maps to a node in a directed acyclic
graph (DAG):

| Operation | Circuit node |
|---|---|
| **Weights** | CONST nodes (INT8 quantised, per-column scales) |
| **Activation fns** (SiLU, exp, rsqrt, cos, sin) | LUT nodes (computed in C) |
| **Matrix multiply** | MATMUL / MATMUL_Q8 nodes |
| **RMSNorm** | Composite: square → mean → rsqrt LUT → mul |
| **Softmax** | Composite: max → sub → exp LUT → sum → div |
| **RoPE** | Precomputed cos/sin tables → mul → add |
| **Residual connections** | ADD nodes |
| **Attention** | MATMUL → scale → causal mask → softmax |
| **Reshape / transpose** | Zero-cost wiring nodes |

The full model is one circuit graph.  The compiler builds it, the
optimizer shrinks it, the virtual device runs it.

### The virtual device

The **Machine** is the compiled chip — a datapath circuit plus
configuration (embedding table, RoPE tables, KV cache layout).
The **VirtualDevice** is the CPU emulation of this machine,
implemented entirely in C:

```
Raw UTF-8 bytes
    │
    ▼
┌──────────────┐
│  BPE Encode  │  FNV-1a hash vocab lookup + merge loop (C)
└──────┬───────┘
       │ token IDs
       ▼
┌──────────────┐
│   Prefill    │  Feed each prompt token through the datapath tape (C)
└──────┬───────┘
       │ KV cache populated
       ▼
┌──────────────┐
│    Decode    │  Autoregressive: embed → RoPE → tape_run → argmax (C)
└──────┬───────┘
       │ token ID per step
       ▼
┌──────────────┐
│  BPE Decode  │  Token ID → UTF-8 bytes, streamed via callback (C)
└──────┬───────┘
       │
       ▼
Raw UTF-8 bytes (streamed)
```

One C function call.  Python is only the ctypes bridge.

### Graph optimisation

**Offline (compile time)** — weights and model structure are known:

- **Constant folding**: all-const subgraphs collapse to a single node.
- **Dead node elimination**: unreachable nodes are pruned.
- **Identity elimination**: trivial ops like `add(x, 0)` removed.

### Sub-vector decomposition: why this scales

A naïve truth table for a 32-bit float weight has $2^{32}$ entries — far
too large to optimise.  The **sub-vector** trick solves this by breaking
each 32-bit weight into four 8-bit slices:

| Scope | States |
|---|---|
| **Per sub-vector** | $2^{8} = 256$ |
| **Per neuron** (4 slices) | $4 \times 256 = 1{,}024$ |
| **Global model** (e.g. 7 B params) | Fully parallelisable — every neuron's K-map can be optimised simultaneously on a cluster |

This means the "compilation" (training → circuit) step is tractable for
billion-parameter models: each neuron's truth table is just 1 K of
states, and all neurons are independent.

### FPGA as the target

The optimised DAG exports directly to Verilog/VHDL.  Every node maps
to synthesisable RTL — LUT nodes become BRAM ROMs, arithmetic nodes
become FPU instances, wiring nodes are free.

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python ≥ 3.13.  Runtime dependency: **numpy** (≥ 1.26).
torch + transformers needed only at compile time (model download).

### Compile the C extensions

```bash
# macOS
cc -O3 -shared -fPIC -march=native -o csrc/_tape_runner.so \
   csrc/_tape_runner.c -lm -framework Accelerate

cc -O3 -shared -fPIC -march=native -o csrc/_circuit_eval.so \
   csrc/_circuit_eval.c -lm -framework Accelerate

# Linux
cc -O3 -shared -fPIC -march=native -o csrc/_tape_runner.so \
   csrc/_tape_runner.c -lm -lopenblas

cc -O3 -shared -fPIC -march=native -o csrc/_circuit_eval.so \
   csrc/_circuit_eval.c -lm -lopenblas
```

## Usage

### 1. Create a chip from a HuggingFace model

```bash
kllm create --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 ./mychip
```

Downloads the model, compiles the circuit graph, quantises weights to
INT8, serialises the BPE tokenizer as circuit ROMs, and writes the
complete chip to `./mychip/`.

### 2. Run inference

```bash
kllm infer ./mychip --max-tokens 100 "Hello world"
```

Everything runs in C — BPE encode, prefill, autoregressive decode,
BPE decode.  Output streams to stdout.

### 3. Compare with HuggingFace (requires torch)

```bash
kllm compare ./mychip --max-tokens 10 "Hello world"
```

### 4. Export to HDL

```bash
kllm export-hdl ./mychip --format verilog
```

### 5. Simulate inference in Verilog

```bash
kllm simulate-infer ./mychip --max-tokens 5 "Hello"
```

### Options

`--max-tokens` supports multiplier suffixes: `128k` → 131072, `1m` → 1048576.

## Project structure

```
src/kllm/
├── cli.py                # CLI entry point (create, infer, compare, export-hdl, simulate-infer)
├── compare.py            # HuggingFace vs kllm side-by-side comparison
├── graph/                # Core DAG, evaluation, optimization, execution
│   ├── circuit_graph.py  #   Circuit builder + reference NumPy evaluator
│   ├── evaluator.py      #   Reference NumPy evaluator (golden model)
│   ├── graph_optimizer.py#   Constant folding + dead elimination + identity elimination
│   └── circuit_executor.py#  C-accelerated evaluator + CTapeRunner (ctypes wrapper)
├── compiler/             # Model-to-circuit compilation
│   ├── fabric.py         #   Model weight loader (HuggingFace → float32 arrays)
│   ├── circuit_compiler.py#  Compile transformer into CircuitGraph
│   └── circuit_tokenizer.py# Compile BPE tokenizer into circuit graph ROMs
├── device/               # Assembled chip and runtime
│   ├── chip.py           #   Chip: compiled model artifact (bytes in → bytes out)
│   ├── machine.py        #   Machine: datapath + tokenizer + config (the compiled device)
│   └── virtual_device.py #   VirtualDevice: CPU emulation of the machine (ctypes bridge)
└── hdl/                  # Hardware synthesis and simulation
    ├── hdl_export.py     #   Verilog / VHDL export + resource estimation
    └── hdl_simulate.py   #   Verilog simulation (iverilog + vvp)

csrc/
├── _tape_runner.c        # C virtual device: tape engine + BPE encode/decode + full inference loop
└── _circuit_eval.c       # C tensor ops: add/sub/mul/div, matmul, LUT activations, reductions
```

## Chip layout

```
mychip/
├── chip.json             # chip metadata
├── processor.json        # machine config (dims, heads, layer count, slot maps)
├── circuit/              # compiled datapath circuit graph (binary nodes + const data)
├── q8/                   # INT8 quantised weight matrices
├── tables/               # embed_table.npy, rope_cos.npy, rope_sin.npy
├── tokenizer/            # raw tokenizer files
├── tokenizer_roms/       # BPE tokenizer as serialised circuit graph
└── weights/              # model config + layer weights
```

## Running tests

```bash
pytest          # 258 tests
```

## License

This project is dual-licensed:

- **AGPL-3.0-or-later** — free for open-source use under the terms of the
  [GNU Affero General Public License v3](LICENSE).
- **Commercial License** — for proprietary / closed-source use, contact
  Renato Augusto Viço Elias (renato@s2n.es).
