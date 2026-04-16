"""CLI entry-point for kllm.

Usage
-----
  kllm create  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 ./mychip
  kllm infer   ./mychip --max-tokens 128k Hello world
  kllm compare ./mychip
  kllm export-hdl ./mychip
  kllm simulate-infer ./mychip --max-tokens 5 Hello
"""

import argparse
import os
import re
import sys

import numpy as np


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _parse_token_count(s: str) -> int:
    """Parse a human-readable token count.

    Supports optional suffix: k/K (×1024), m/M (×1048576).
    Examples: "50" → 50, "128k" → 131072, "1M" → 1048576.
    """
    m = re.fullmatch(r"(\d+)\s*([kKmM])?", s.strip())
    if not m:
        raise argparse.ArgumentTypeError(
            f"Invalid token count: {s!r}. "
            "Use a number with optional k/m suffix (e.g. 50, 128k, 1m)."
        )
    value = int(m.group(1))
    suffix = (m.group(2) or "").lower()
    if suffix == "k":
        value *= 1024
    elif suffix == "m":
        value *= 1024 * 1024
    return value


def _get_prompt(args) -> str:
    """Get prompt text from positional args or interactive input."""
    if hasattr(args, "prompt") and args.prompt:
        return " ".join(args.prompt)
    return input("Enter your text prompt: ")


# ---------------------------------------------------------------
# Parser
# ---------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kllm",
        description="Compile a transformer into a custom chip. "
        "Run inference natively or through HDL simulation.",
    )
    sub = p.add_subparsers(dest="command")

    # ---- create ----
    create_p = sub.add_parser(
        "create",
        help="Download a HuggingFace model and compile it into a chip.",
    )
    create_p.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name (e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0)",
    )
    create_p.add_argument(
        "chip_path",
        type=str,
        help="Local directory for the compiled chip.",
    )

    # ---- infer ----
    infer_p = sub.add_parser(
        "infer",
        help="Run inference on a compiled chip.",
    )
    infer_p.add_argument(
        "chip_path",
        type=str,
        help="Path to a compiled chip directory.",
    )
    infer_p.add_argument(
        "--max-tokens",
        type=_parse_token_count,
        default=50,
        help="Max new tokens (supports k/m suffix: 128k, 1m). Default: 50.",
    )
    infer_p.add_argument(
        "prompt",
        nargs="*",
        help="Prompt text (remaining arguments joined with space). "
        "Omit to enter interactively.",
    )

    # ---- compare ----
    compare_p = sub.add_parser(
        "compare",
        help="Benchmark chip inference against HuggingFace.",
    )
    compare_p.add_argument(
        "chip_path",
        type=str,
        help="Path to a compiled chip directory.",
    )
    compare_p.add_argument(
        "--max-tokens",
        type=_parse_token_count,
        default=50,
        help="Max new tokens per benchmark prompt. Default: 50.",
    )
    compare_p.add_argument(
        "prompt",
        nargs="*",
        help="Optional custom prompt (omit for standard benchmark suite).",
    )

    # ---- export-hdl ----
    export_p = sub.add_parser(
        "export-hdl",
        help="Export the chip's processor to Verilog/VHDL.",
    )
    export_p.add_argument(
        "chip_path",
        type=str,
        help="Path to a compiled chip directory.",
    )
    export_p.add_argument(
        "--format",
        choices=["verilog", "vhdl"],
        default="verilog",
        dest="output_format",
        help="HDL output format (default: verilog).",
    )

    # ---- simulate-infer ----
    sim_p = sub.add_parser(
        "simulate-infer",
        help="Run inference through HDL simulation (iverilog).",
    )
    sim_p.add_argument(
        "chip_path",
        type=str,
        help="Path to a compiled chip directory.",
    )
    sim_p.add_argument(
        "--max-tokens",
        type=_parse_token_count,
        default=5,
        help="Max new tokens to simulate. Default: 5.",
    )
    sim_p.add_argument(
        "prompt",
        nargs="*",
        help="Prompt text. Omit to enter interactively.",
    )

    return p


# ---------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------

def _cmd_create(args) -> None:
    from kllm.device.chip import Chip
    Chip.create(model_name=args.model, chip_path=args.chip_path)


def _cmd_infer(args) -> None:
    from kllm.device.chip import Chip

    chip = Chip.load(args.chip_path)
    prompt = _get_prompt(args)

    print(f"[infer] Prompt: {prompt!r}")
    print(f"[infer] Max tokens: {args.max_tokens}")
    print()

    # Stream tokens as they're generated
    sys.stdout.write("--- kllm Generated Text ---\n")
    for chunk in chip.infer_streaming(prompt, args.max_tokens):
        sys.stdout.write(chunk)
        sys.stdout.flush()
    sys.stdout.write("\n")


def _cmd_compare(args) -> None:
    from kllm.compare import compare_chip, print_generate_report

    from kllm.device.chip import Chip
    chip = Chip.load(args.chip_path)

    prompt = _get_prompt(args) if args.prompt else None
    stats = compare_chip(chip, text=prompt, max_tokens=args.max_tokens)
    print_generate_report(stats)


def _cmd_export_hdl(args) -> None:
    from kllm.device.chip import Chip
    from kllm.hdl.hdl_export import (
        export_verilog, export_vhdl, estimate_resources,
    )

    chip = Chip.load(args.chip_path)
    proc = chip.processor

    hdl_dir = os.path.join(args.chip_path, "hdl")
    os.makedirs(hdl_dir, exist_ok=True)

    if args.output_format == "verilog":
        path = os.path.join(hdl_dir, "decode_datapath.v")
        export_verilog(proc.datapath, path)
        print(f"Verilog datapath exported to {path}")
    else:
        path = os.path.join(hdl_dir, "decode_datapath.vhd")
        export_vhdl(proc.datapath, path)
        print(f"VHDL datapath exported to {path}")

    res = estimate_resources(proc.datapath)
    print(f"Estimated resources: LUTs={res['luts']} FFs={res['ffs']} "
          f"BRAMs={res['brams']} DSPs={res['dsps']}")


def _cmd_simulate_infer(args) -> None:
    from kllm.device.chip import Chip

    chip = Chip.load(args.chip_path)
    prompt = _get_prompt(args)

    print(f"[simulate-infer] Prompt: {prompt!r}")
    print(f"[simulate-infer] Max tokens: {args.max_tokens}")
    print("[simulate-infer] Running native inference for golden reference …")

    # Native inference first (golden reference)
    native_output = chip.infer(prompt, args.max_tokens)
    print(f"[simulate-infer] Native output: {native_output!r}")

    # TODO: Phase 4 — full HDL simulation pipeline
    # 1. export_processor_verilog(chip.processor, work_dir)
    # 2. export_processor_testbench(chip.processor, prompt_tokens, max_tokens)
    # 3. iverilog compile + vvp simulate
    # 4. Compare simulation output vs golden reference
    print("[simulate-infer] HDL simulation not yet implemented — "
          "native reference generated successfully.")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

_COMMANDS = {
    "create": _cmd_create,
    "infer": _cmd_infer,
    "compare": _cmd_compare,
    "export-hdl": _cmd_export_hdl,
    "simulate-infer": _cmd_simulate_infer,
}


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        raise SystemExit(1)

    handler = _COMMANDS.get(args.command)
    if handler is None:
        parser.print_help()
        raise SystemExit(1)

    handler(args)


if __name__ == "__main__":
    main()
