# kllm

Z3 Gate Inference Engine for LLMs.

**kllm** decomposes every model weight into its raw IEEE-754 byte planes,
uses a Z3 SMT solver to prove a minimal boolean gate `(0xFF << s1) ^ mask == target`
for each byte value, and stores the gate parameters on disk. At inference time
the gates are executed (shift + XOR) to recover the exact original weights —
pure bit operations, zero precision loss.

## Key ideas

| Concept | Traditional (FP32) | kllm (Z3 Gate Fabric) |
|---|---|---|
| Data unit | 32-bit float | 4 × `(s1, mask)` gate pairs |
| Weight storage | VRAM tensors | Z3-proven gate instructions |
| Recovery | N/A | `uint8(0xFF << s1) ^ mask` → byte plane |
| Precision | Lossy if quantised | **Lossless** (raw IEEE-754) |
| KV cache | Framework-managed | Built-in, O(1) per decode token |

1. **Lossless byte-plane decomposition** — weights are viewed as raw `uint32` and
   split into four 8-bit planes. No scaling, no rounding.
2. **Z3 gate compilation** — the solver proves `(0xFF << s1) ^ mask == target`
   for all 256 possible byte values. A lookup table maps each byte to its
   `(s1, mask)` pair — no fallback, no failure.
3. **One-time gate execution** — at load time, gates are executed (shift + XOR)
   to recover float32 weights. The recovered weights are cached in RAM and the
   raw gate arrays are discarded.
4. **KV-cached generation** — prefill processes the full prompt, then each
   decode step only forwards the new token through all layers.

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python ≥ 3.13.

## Usage

### Compile a model into gate fabric (one-time)

```bash
kllm --mode compile --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

The compiled fabric is saved to `./lossless_logic/` by default.

### Run inference

```bash
kllm --mode inference --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --text "Hello world"
```

### Generate text (KV-cached)

```bash
kllm --mode generate --text "Who are you?" --max-tokens 50
```

### Compare with HuggingFace

```bash
kllm --mode compare --text "Who are you?" --max-tokens 10
```

### Compile + generate in one shot

```bash
kllm --mode full --text "Hello world"
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--mode` | *(required)* | `compile`, `inference`, `generate`, `compare`, or `full` |
| `--model` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | HuggingFace model ID or local path |
| `--save-dir` | `./lossless_logic` | Where compiled fabric is stored |
| `--text` | *(interactive)* | Prompt text |
| `--solver-timeout` | `200` | Z3 timeout per pattern in ms |
| `--max-layers` | all | Limit layers (useful for quick tests) |
| `--max-tokens` | `50` | Max new tokens for generate/compare modes |

## Project structure

```
src/kllm/
├── bitops.py      # Lossless IEEE-754 byte-plane extract / repack
├── compiler.py    # Z3-based gate synthesis (compile mode)
├── inference.py   # Z3 gate inference engine with KV cache
├── compare.py     # HuggingFace vs kllm generation comparison
├── device.py      # GPU / CPU abstraction (CuPy ↔ NumPy)
└── cli.py         # CLI entry point
tests/
├── test_bitops.py
├── test_cli.py
├── test_compare.py
├── test_compiler.py
└── test_device.py
```

## Running tests

```bash
pytest
```

## How it works

```
              ┌─────────────┐
   FP32       │  .view(u32) │    4 × uint8 planes
  weights ──▶ │  byte-plane │──▶  m0  m1  m2  m3
              │  decompose  │
              └─────────────┘
                    │
                    ▼
              ┌─────────────┐
              │  Z3 solver  │    (s1, mask) per byte value
              │  per unique │──▶  uint8(0xFF << s1) ^ mask
              │  8-bit val  │
              └─────────────┘
                    │
                    ▼
              ┌─────────────┐
              │  .npz gate  │    stored on disk as layer files
              │  fabric on  │──▶  s1[] + mask[] arrays
              │  disk       │
              └─────────────┘
                    │  (loaded once)
                    ▼
              ┌─────────────┐
              │  execute &  │    gates → byte planes → np.view
              │  cache      │──▶  float32 weights cached in RAM
              │  float32    │
              └─────────────┘
                    │
                    ▼
              ┌─────────────┐
              │  KV-cached  │    prefill + O(1) decode per token
              │  generation │──▶  identical output to HuggingFace
              └─────────────┘
```

## License

This project is dual-licensed:

- **AGPL-3.0-or-later** — free for open-source use under the terms of the
  [GNU Affero General Public License v3](LICENSE).
- **Commercial License** — for proprietary / closed-source use, contact
  Renato Augusto Viço Elias (renato@s2n.es).
