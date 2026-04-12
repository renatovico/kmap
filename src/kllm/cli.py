"""CLI entry-point for kllm.

Usage
-----
  kllm --mode compile   --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
  kllm --mode inference  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
  kllm --mode compare    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --text "Hello"
  kllm --mode full       --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

import argparse


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kllm",
        description="Lossless Bit-Sliced Logic Engine for LLM inference",
    )
    p.add_argument(
        "--mode",
        choices=["compile", "inference", "generate", "compare", "full"],
        required=True,
        help="compile: bake model into logic fabric · "
        "inference: generate text from compiled fabric · "
        "generate: auto-regressive text generation · "
        "compare: HuggingFace vs kllm side-by-side · "
        "full: compile then generate",
    )
    p.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HuggingFace model name or local path (default: TinyLlama)",
    )
    p.add_argument(
        "--save-dir",
        type=str,
        default="./lossless_logic",
        help="Directory for compiled logic fabric (default: ./lossless_logic)",
    )
    p.add_argument(
        "--text",
        type=str,
        default=None,
        help="Prompt text for inference/compare (omit to enter interactively)",
    )
    p.add_argument(
        "--solver-timeout",
        type=int,
        default=200,
        help="Z3 solver timeout per pattern in ms (default: 200)",
    )
    p.add_argument(
        "--max-layers",
        type=int,
        default=None,
        help="Limit the number of layers to process (useful for quick tests)",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Max new tokens for generate mode (default: 50)",
    )
    return p


def _get_text(args) -> str:
    if args.text is not None:
        return args.text
    return input("Enter your text prompt: ")


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.mode in ("compile", "full"):
        from kllm.compiler import LosslessLogicCompiler

        compiler = LosslessLogicCompiler(
            model_name=args.model,
            save_dir=args.save_dir,
            solver_timeout=args.solver_timeout,
        )
        compiler.compile()

    if args.mode in ("inference", "full"):
        from kllm.inference import BitLogicInferenceEngine

        engine = BitLogicInferenceEngine(
            model_name=args.model,
            save_dir=args.save_dir,
        )

        text = _get_text(args)
        output = engine.generate(text, max_new_tokens=args.max_tokens)
        print(f"\n--- kllm Generated Text ---")
        print(output)

    if args.mode == "generate":
        from kllm.inference import BitLogicInferenceEngine

        engine = BitLogicInferenceEngine(
            model_name=args.model,
            save_dir=args.save_dir,
        )

        text = _get_text(args)
        output = engine.generate(text, max_new_tokens=args.max_tokens)
        print(f"\n--- kllm Generated Text ---")
        print(output)

    if args.mode == "compare":
        from kllm.compare import compare_generate, print_generate_report

        text = _get_text(args)
        stats = compare_generate(
            model_name=args.model,
            save_dir=args.save_dir,
            text=text,
            max_tokens=args.max_tokens,
        )
        print_generate_report(stats)


if __name__ == "__main__":
    main()
