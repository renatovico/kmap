# kllm

**Boolean Logic Synthesis for LLM Inference.**

kllm uses the Z3 SMT solver to **compile** a transformer model into
boolean circuits.  Every float32 weight and every activation function
(SiLU, exp, rsqrt) is decomposed into byte-level lookup tables that
Z3 proves correct.  At inference time, the entire pipeline runs on
**pure NumPy** — no torch, no FPU instructions — using only shift, XOR,
and array indexing.  The result is **bit-exact** to HuggingFace output.

This is not quantization.  No precision is lost.  Every bit of every
original float32 weight is preserved through a change of
**representation**, not a change of value.

## The core idea

A 32-bit float is 4 bytes.  Each byte can only take one of 256 values.
Z3 can prove, for every possible byte value, a constant gate of the form:

$$\text{target} = (0\text{xFF} \ll s_1) \oplus \text{mask}$$

where $s_1 \in [0, 7]$ and $\text{mask} \in [0, 255]$.  This yields two
256-entry lookup tables (one for shifts, one for masks) that reconstruct
**any** byte via a single shift and a single XOR.

**Weights** are stored as 4 byte-plane indices per float32.  At load time,
the gate LUTs execute `(0xFF << s1_lut[b]) ^ mask_lut[b]` for each byte
and reassemble the original float32 — lossless, no floating-point math.

**Activation functions** (SiLU, exp, rsqrt) use the same principle but
scaled to the entire float32 domain: for every one of the $2^{32}$
possible input bit patterns, the output is pre-computed and stored as
4 byte-plane files (~4 GB each).  At runtime, activations are a single
array index into a memory-mapped file — O(1) per value.

### Why this works

| Concept | Traditional approach | kllm |
|---|---|---|
| Weight storage | float32 tensors in VRAM | 4 byte-plane indices → gate LUT → float32 |
| Activation fns | FPU (SiLU, exp, rsqrt) | Full-domain byte-plane maps (mmap lookup) |
| Data path | float32 arithmetic | shift + XOR + array index |
| Precision | Lossy if quantised | **Lossless** — bit-exact to HuggingFace |
| Runtime deps | torch, transformers, CUDA | **numpy only** |

### Three compilation stages

1. **Weight compilation** (`kllm --mode compile`) — loads HuggingFace
   weights, decomposes every float32 into 4 IEEE-754 byte planes, and
   runs Z3 to prove a `(shift, mask)` gate for all 256 byte values.
   Result: `.npz` gate arrays per layer.

2. **Circuit compilation** (`kllm --mode compile-circuits`) — evaluates
   SiLU, exp, and rsqrt for **all $2^{32}$ float32 bit patterns**.
   Each output is split into 4 byte planes and written as ~4 GB binary
   files (12 files, ~48 GB total).  Memory-mapped at runtime — the OS
   pages in only what the model touches.

3. **Optimisation** (`kllm --mode optimize`) — applies Quine-McCluskey
   boolean minimisation to the gate LUTs (8 inputs → 8 outputs) and
   materialises pre-computed float32 weight files for instant mmap
   loading, bypassing gate execution entirely.

### Why full-domain?

Pre-computing every possible float32 input means no value is ever
"unseen".  Every prompt, every temperature, every token produces
activation values that already exist in the compiled tables.  There is
no hash table, no on-demand compilation, no fallback — just a single
array index per value.

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
├── test_optimizer.py
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

  HuggingFace weights        Z3 Solver              Disk
  ───────────────────        ─────────              ────
  float32 per weight  ──▶  .view(uint32)  ──▶  4 byte planes (m0-m3)
                           Z3: (0xFF << s1) ^ mask == target
                           ──▶ 256-entry (s1, mask) LUTs
                           ──▶ layer_N.npz  (gate fabric)

  SiLU / exp / rsqrt        NumPy eval              Disk
  ──────────────────        ──────────              ────
  for all 2^32 inputs ──▶  fn(x) → 4 bytes  ──▶  {op}_p{0-3}.bin
                            (one per byte plane)    (4 GB each, 12 files)

 ┌──────────────────────────────────────────────────────────┐
 │               INFERENCE (numpy only, no torch)           │
 └──────────────────────────────────────────────────────────┘

  ┌─────────────┐
  │   Fabric    │  Three load paths:
  │  (weights)  │  1. gate LUTs: shift+XOR → float32 (default)
  │             │  2. optimized/: pre-computed float32 mmap (fast)
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  Circuits   │  Activation functions:
  │ (SiLU, exp, │  Z3-verified NumPy formulas (SIMD, fast path)
  │   rsqrt)    │  — or mmap byte-plane files (full-domain fallback)
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │ Transformer │  RMSNorm → Q,K,V → RoPE → Attention → MLP
  │  (per layer)│  matmul on reconstructed float32 weights
  │             │  activations via circuit lookup
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
