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

### PyTorch engine — fast inference and generation with KV-cache

The `torch` engine loads model weights directly from HuggingFace and uses
PyTorch for compute. **No compiled fabric is required.** It supports
GPU acceleration and implements a sliding-window KV-cache so generation
stays fast even for very long outputs.

```bash
# CPU inference (fp32, default)
kllm --mode inference --engine torch \
     --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
     --text "Hello world"

# GPU inference with bfloat16
kllm --mode inference --engine torch \
     --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
     --device cuda --dtype bf16 --text "Hello world"

# Autoregressive generation with KV-cache (sliding window = 4096 tokens)
kllm --mode generate --engine torch \
     --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
     --device cuda --dtype bf16 \
     --window 4096 --max-tokens 200 \
     --text "Once upon a time"
```

#### Torch engine flags

| Flag | Default | Description |
|---|---|---|
| `--device` | `cpu` | PyTorch device string: `cpu`, `cuda`, `cuda:0`, … |
| `--dtype` | `fp32` | Floating-point format: `fp32`, `bf16`, `fp16` |
| `--window` | `131072` | Sliding KV-cache window in tokens (larger = more context, more memory) |

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
| `--mode` | *(required)* | `compile`, `inference`, `generate`, `compare`, or `full` |
| `--model` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | HuggingFace model ID or local path |
| `--save-dir` | `./lossless_logic` | Where compiled fabric is stored |
| `--text` | *(interactive)* | Prompt text |
| `--engine` | `standard` | `standard`, `bitlogic`, or `torch` |
| `--device` | `cpu` | PyTorch device (`torch` engine only) |
| `--dtype` | `fp32` | Float dtype: `fp32`, `bf16`, `fp16` (`torch` engine only) |
| `--window` | `131072` | KV-cache sliding window in tokens (`torch` engine only) |
| `--solver-timeout` | `200` | Z3 timeout per pattern in ms |
| `--max-layers` | all | Limit layers (useful for quick tests) |
| `--max-tokens` | `50` | Max new tokens for generate mode |

## Project structure

```
src/kllm/
├── bitops.py       # Lossless IEEE-754 sub-bit mask extract / repack
├── compiler.py     # Z3-based logic synthesis (compile mode)
├── inference.py    # Streaming bit-sliced inference engine (NumPy)
├── torch_engine.py # PyTorch engine with KV-cache and sliding window
├── compare.py      # Side-by-side classic FP32 vs logic benchmark
├── device.py       # GPU / CPU abstraction (CuPy ↔ NumPy)
└── cli.py          # CLI entry point
tests/
├── test_bitops.py
├── test_compiler.py
├── test_compare.py
├── test_device.py
├── test_cli.py
└── test_torch_engine.py
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
