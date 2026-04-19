# kllm

**Compile a transformer into a truth-table circuit. Run it in C.**

kllm turns every weight matrix in an LLM into a **truth-table lookup**
(codebook + indices), compiles the entire model into a linear tape of
instructions, and executes it with a multithreaded C runtime.

1. **Bit-exact** to HuggingFace output — lossless truth-table compression,
   float32 activations.
2. **Compiled offline**: the full model compiles to a single binary tape —
   BPE tokenizer included as hash-table ROMs.
3. **Executed in pure C**: BPE encode, prefill, decode, BPE decode, KV cache —
   zero Python in the inference loop. 8-thread NEON-vectorized TT-GEMV.
4. **FPGA-targetable**: each codebook lookup maps directly to combinational
   logic. Built-in 6-LUT gate analysis.

## How it works

### Truth-table compression

Every weight matrix is decomposed into a **codebook** (the unique float32
values) plus **indices** (which codebook entry each element uses). Matrix-
vector products become truth-table lookups:

```
y[n] = Σ_k  codebook[ index[k][n] ] * x[k]
```

The codebook is small (typically 4,000–5,500 unique values per matrix),
so each lookup is a boolean function of ~13 input bits → 32 output bits.

### The tape

The compiler flattens the transformer DAG into a linear tape of
instructions: MATMUL_TT, ADD, MUL, RMSNORM, ROPE, SOFTMAX, SILU, etc.
The runtime walks the tape sequentially, executing each op on pre-
allocated slot buffers.

### Gate analysis

Each codebook lookup is 32 boolean functions of `ceil(log2(n_codes))`
inputs. The gate analyzer classifies each output bit:

- **Trivial** (constant across all codebook entries) → 0 LUTs
- **Non-trivial** → Shannon decomposition bound for FPGA 6-LUTs

Typical result: ~19 of 32 IEEE 754 bits are constant (sign + exponent),
giving **~59% LUT reduction** vs naive per-bit lookup.

## Build

```bash
make
```

Requires a C compiler with NEON support (macOS) or SSE (Linux).
macOS uses Accelerate framework, Linux uses OpenBLAS.

## Usage

### 1. Create a chip from a HuggingFace model

```bash
./kllm create --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 ./mychip
```

Downloads the model (requires Python + transformers), compiles the
truth-table circuit, serialises the BPE tokenizer as hash-table ROMs,
and runs FPGA gate analysis. Output goes to `./mychip/`.

### 2. Run inference

```bash
./kllm infer ./mychip --max-tokens 100 "Hello world"
```

Pure C — 8-thread TT-GEMV, NEON vectorized. ~2.5 tok/s on M1 Pro
for TinyLlama-1.1B.

### 3. Compare with HuggingFace

```bash
./kllm compare ./mychip --max-tokens 10
```

Runs 3 prompts through both HF and kllm, verifies outputs match.

### 4. Gate analysis

```bash
./kllm optimize ./mychip [-v]
```

FPGA 6-LUT analysis of all truth-table codebooks. Also runs
automatically during `create`.

## Project structure

```
csrc/
├── kllm.c            # CLI: create, infer, compare, optimize subcommands
├── kllm_compile.c    # C compiler: weights → truth tables → tape
├── kllm_gates.c      # FPGA 6-LUT gate analysis
├── _tape_runner.c    # Runtime: tape engine, 8-thread TT-GEMV, BPE, KV cache
└── tape_runner.h     # Shared types (Slot, TapeInstr, TapeCtx)

src/kllm/compiler/
└── fabric.py         # HuggingFace model downloader (only Python remaining)

Makefile              # Single binary build
```

## Chip layout

```
mychip/
├── chip.json          # metadata (model name, vocab size, dims)
├── processor.json     # runtime config (heads, layers, slot maps, KV layout)
├── circuit/
│   ├── tape.bin       # instruction tape
│   ├── slots.bin      # slot descriptors (type, shape)
│   ├── const_*.bin    # float32 constants (embeddings, norms, etc.)
│   ├── const_*_codebook.bin   # TT codebooks (unique float32 values)
│   └── const_*_indices.bin    # TT indices (uint16 per element)
├── tables/            # embed_table.npy, rope_cos.npy, rope_sin.npy
└── tokenizer/
    ├── vocab_hash_keys.bin    # BPE vocab hash table
    ├── vocab_hash_vals.bin    # token IDs
    ├── merges.bin             # BPE merge pairs
    └── ...
```

## License

This project is dual-licensed:

- **AGPL-3.0-or-later** — free for open-source use under the terms of the
  [GNU Affero General Public License v3](LICENSE).
- **Commercial License** — for proprietary / closed-source use, contact
  Renato Augusto Viço Elias (renato@s2n.es).
