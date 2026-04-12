# kllm

Z3 Universal Computing Engine for LLM Inference.

**kllm** treats the Z3 SMT solver as a **universal computing machine**.
Every operation in a transformer — weight storage, matrix multiply,
RMSNorm, RoPE, SiLU, softmax — is compiled into Z3-proved boolean
gates expressed as `(input << shift) ^ mask`.  At inference time the
entire pipeline runs through pure shift + XOR operations.  No floating-
point unit is used.  Zero precision loss.

## Key ideas

| Concept | Traditional (FP32) | kllm (Z3 Gate Fabric) |
|---|---|---|
| Weight storage | VRAM tensors | 4 × `(s1, mask)` constant gates |
| Activation fns | FPU (SiLU, exp, rsqrt) | Full-domain byte-plane maps (mmap) |
| Data path | float32 arithmetic | shift + XOR (boolean gates) |
| Precision | Lossy if quantised | **Lossless** (bit-exact to HuggingFace) |
| HF dependency | Yes (torch, transformers) | **None** at inference (numpy only) |

### Two compilation stages

1. **Weight compilation** (`kllm --mode compile`) — decomposes every
   float32 weight into 4 IEEE-754 byte planes.  Z3 proves
   `(0xFF << s1) ^ mask == target` for all 256 byte values.  Stored as
   `.npz` gate arrays per layer.

2. **Circuit compilation** (`kllm --mode compile-circuits`) — evaluates
   SiLU, exp, and rsqrt for **every possible float32 bit pattern**
   (all 2^32 values).  Each output is decomposed into 4 byte planes
   and written to disk as 4 GB binary files (12 files, ~48 GB total).
   At runtime these are memory-mapped — the OS pages in only what the
   model actually touches.  This is the same 4-plane strategy used for
   weight storage, now applied to activation functions.

### Why full-domain?

The Z3 solver proves a gate for each byte value.  By pre-computing
every possible float32 input, no value can ever be "unseen" — every
prompt, every temperature, every token produces activation values that
are already in the compiled tables.  Lookup is O(1) array indexing
into an mmap'd file.  No hash tables, no on-demand compilation, no
fallback.

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
├── circuits.py       # Z3 gate primitives + ArithmeticUnit (mmap byte planes)
├── circuit_model.py  # LLaMA transformer with Z3 circuit execution + streaming
├── fabric.py         # Gate loader — shift+XOR → float32 weight matrices
├── model.py          # Reference NumPy LLaMA (used by compare)
├── inference.py      # Orchestrator: tokenizer + fabric + circuits → generate
├── tokenizer.py      # Pure-Python BPE tokenizer (no HuggingFace)
├── compare.py        # HuggingFace vs kllm generation comparison
├── device.py         # GPU / CPU abstraction (CuPy ↔ NumPy)
└── cli.py            # CLI entry point
tests/
├── test_bitops.py
├── test_circuits.py
├── test_cli.py
├── test_compare.py
├── test_compiler.py
├── test_device.py
└── test_tokenizer.py
```

## Running tests

```bash
pytest          # 68 tests
```

## How it works

```
 ┌──────────────────────────────────────────────────────────┐
 │                    COMPILE TIME (one-time)                │
 └──────────────────────────────────────────────────────────┘

  HuggingFace          Z3 Solver            Disk
  ───────────          ─────────            ────
  float32 weights ──▶ .view(uint32) ──▶ 4 byte planes (m0-m3)
                       Z3: (0xFF << s1) ^ mask == target
                       ──▶ (s1, mask) per byte value
                       ──▶ layer_N.npz  (gate fabric)

  SiLU / exp / rsqrt   for all 2^32 float32 bit patterns:
                        fn(x) → 4 output bytes → index const LUT
                       ──▶ {op}_p{0-3}.bin  (4 GB each, 12 files)

 ┌──────────────────────────────────────────────────────────┐
 │               INFERENCE (numpy only, no torch)           │
 └──────────────────────────────────────────────────────────┘

  ┌─────────────┐
  │   Fabric    │  shift+XOR gates → float32 weight matrices
  │  (10s load) │  cached in RAM, gate arrays discarded
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  Circuits   │  mmap 12 byte-plane files (48 GB on disk)
  │  (mmap'd)   │  OS pages in only what's accessed (~MBs)
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │ Transformer │  RMSNorm → Q,K,V → RoPE → Attention → MLP
  │  (per layer)│  activations: idx = x.view(uint32)
  │             │  out_byte = plane_file[idx]  ← O(1) mmap lookup
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  KV-cached  │  prefill full prompt, then O(1) per decode token
  │  generation │  streaming: yields each token as produced
  └─────────────┘
```

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
│   └── rsqrt_p0.bin … p3.bin   # 4 × 4 GB
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
