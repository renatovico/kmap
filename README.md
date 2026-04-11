# kllm

Lossless Bit-Sliced Logic Engine for LLM inference.

Instead of running floating-point matrix multiplications at inference time,
**kllm** decomposes every model weight into its raw IEEE-754 bit planes,
uses a Z3 SMT solver to synthesise minimal boolean logic for each unique
8-bit pattern, and streams the resulting "logic fabric" through the GPU
using only bitwise operations (shift, XOR, AND).

## Key ideas

| Concept | Traditional (FP32) | kllm (Bit-Sliced Logic) |
|---|---|---|
| Data unit | 32-bit float | 4 × 8-bit sub-masks |
| Operation | Matrix multiply (FPU) | Boolean gates (XOR / shift) |
| Weight storage | VRAM tensors | Pre-solved gate instructions |
| Precision | Lossy if quantised | **Lossless** (raw IEEE-754) |

1. **Lossless sub-bit masking** — weights are viewed as raw `uint32` and
   split into four 8-bit planes. No scaling, no rounding.
2. **Z3 solver compilation** — each unique 8-bit pattern is reduced to
   a minimal `(shift, mask)` gate pair. Patterns are deduplicated across
   the entire model so the solver only runs once per unique value.
3. **Layer streaming** — compiled layers are saved to disk as `.npz` files
   and loaded one at a time during inference, keeping VRAM usage under 4 GB.
4. **GPU / CPU transparent** — uses CuPy when a CUDA GPU is available,
   falls back to NumPy automatically.

## Installation

```bash
pip install -e ".[dev]"

# Optional: GPU acceleration
pip install -e ".[gpu]"
```

Requires Python ≥ 3.13.

## Usage

### Compile a model into logic fabric (one-time)

```bash
kllm --mode compile --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

The compiled fabric is saved to `./lossless_logic/` by default.

### Run inference with customer text

```bash
kllm --mode inference --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --text "Hello world"

# Or interactively (prompts for input):
kllm --mode inference --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

### Compare classic FP32 vs bit-sliced logic

```bash
kllm --mode compare --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --text "Hello world"

# Limit layers for a quick test:
kllm --mode compare --text "Hello" --max-layers 2
```

### Compile + inference in one shot

```bash
kllm --mode full --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --text "Hello world"
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--mode` | *(required)* | `compile`, `inference`, `compare`, or `full` |
| `--model` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | HuggingFace model ID or local path |
| `--save-dir` | `./lossless_logic` | Where compiled fabric is stored |
| `--text` | *(interactive)* | Prompt text |
| `--solver-timeout` | `200` | Z3 timeout per pattern in ms |
| `--max-layers` | all | Limit layers (useful for quick tests) |

## Project structure

```
src/kllm/
├── bitops.py      # Lossless IEEE-754 sub-bit mask extract / repack
├── compiler.py    # Z3-based logic synthesis (compile mode)
├── inference.py   # Streaming bit-sliced inference engine
├── compare.py     # Side-by-side classic FP32 vs logic benchmark
├── device.py      # GPU / CPU abstraction (CuPy ↔ NumPy)
└── cli.py         # CLI entry point
tests/
├── test_bitops.py
├── test_compiler.py
└── test_compare.py
```

## Running tests

```bash
pytest
```

## How it works

```
              ┌─────────────┐
   FP32       │  .view(u32) │    4 × uint8 planes
  weights ──▶ │  sub-bit    │──▶  m0  m1  m2  m3
              │  masking    │
              └─────────────┘
                    │
                    ▼
              ┌─────────────┐
              │  Z3 solver  │    (shift, mask) per unique pattern
              │  per unique │──▶  deduplicated across all layers
              │  8-bit val  │
              └─────────────┘
                    │
                    ▼
              ┌─────────────┐
              │  .npz layer │    streamed to GPU one at a time
              │  fabric on  │──▶  (x << s1) ^ mask
              │  disk       │
              └─────────────┘
                    │
                    ▼
              ┌─────────────┐
              │  repack     │    lossless IEEE-754 floats out
              │  uint32 →   │──▶  identical to FP32 pipeline
              │  float32    │
              └─────────────┘
```

## License

This project is dual-licensed:

- **AGPL-3.0-or-later** — free for open-source use under the terms of the
  [GNU Affero General Public License v3](LICENSE).
- **Commercial License** — for proprietary / closed-source use, contact
  Renato Augusto Viço Elias (renato@s2n.es).
