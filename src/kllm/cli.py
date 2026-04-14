"""CLI entry-point for kllm.

Usage
-----
  kllm --mode compile   --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
  kllm --mode infer     --text "Hello"
  kllm --mode compare   --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --text "Hello"
  kllm --mode export-hdl
"""

import argparse
import os

import numpy as np


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kllm",
        description="Compile a transformer into a gate circuit. Optimise it. Run it.",
    )
    p.add_argument(
        "--mode",
        choices=["compile", "infer", "compare", "export-hdl", "simulate"],
        required=True,
        help="compile: download model + build circuit graph · "
        "infer: autoregressive text generation via circuit graph · "
        "compare: HuggingFace vs kllm · "
        "export-hdl: export circuit graph to Verilog/VHDL · "
        "simulate: verify HDL export with iverilog simulation",
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
        help="Directory for compiled model (default: ./lossless_logic)",
    )
    p.add_argument(
        "--text",
        type=str,
        default=None,
        help="Prompt text for infer/compare (omit to enter interactively)",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Max new tokens for infer/compare (default: 50)",
    )
    p.add_argument(
        "--output-format",
        choices=["verilog", "vhdl"],
        default="verilog",
        help="HDL output format for export-hdl (default: verilog)",
    )
    return p


def _get_text(args) -> str:
    if args.text is not None:
        return args.text
    return input("Enter your text prompt: ")


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.mode == "compile":
        from kllm.fabric import Fabric

        Fabric.from_pretrained(
            model_name=args.model,
            save_dir=args.save_dir,
        )

    if args.mode == "infer":
        from kllm.fabric import Fabric
        from kllm.jit_optimizer import JitSession
        from kllm.tokenizer import Tokenizer

        fabric = Fabric(args.save_dir)
        tok_dir = os.path.join(args.save_dir, "tokenizer")
        tokenizer = Tokenizer(tok_dir)

        text = _get_text(args)
        messages = [{"role": "user", "content": text}]
        prompt_str = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True,
        )
        token_ids = tokenizer.encode(prompt_str)

        session = JitSession(fabric)
        logits = session.prefill(token_ids)

        generated: list[int] = []
        for _ in range(args.max_tokens):
            next_id = int(np.argmax(logits[-1]))
            if next_id == tokenizer.eos_token_id:
                break
            generated.append(next_id)
            logits = session.decode_step(next_id)

        output = tokenizer.decode(generated, skip_special_tokens=True)
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

    if args.mode == "export-hdl":
        from kllm.circuit_compiler import compile_decode_template
        from kllm.circuit_graph import CircuitGraph
        from kllm.fabric import Fabric
        from kllm.graph_optimizer import optimize_graph
        from kllm.hdl_export import (
            export_verilog, export_vhdl, estimate_resources,
        )

        fabric = Fabric(args.save_dir)

        print("Compiling decode template to circuit graph (INPUT ports for prompt)...")
        machine = compile_decode_template(fabric)
        graph = machine.graph
        opt_graph, id_map = optimize_graph(graph, [machine.logits_id])

        print(f"  Graph: {len(graph)} nodes → {len(opt_graph)} optimised")
        print(f"  INPUT ports: {list(machine.input_ids.keys())}")

        hdl_dir = os.path.join(args.save_dir, "hdl")
        os.makedirs(hdl_dir, exist_ok=True)
        if args.output_format == "verilog":
            path = os.path.join(hdl_dir, "circuit_top.v")
            export_verilog(opt_graph, path)
            print(f"Verilog exported to {path}")
        else:
            path = os.path.join(hdl_dir, "circuit_top.vhd")
            export_vhdl(opt_graph, path)
            print(f"VHDL exported to {path}")
        res = estimate_resources(opt_graph)
        print(f"Estimated resources: LUTs={res['luts']} FFs={res['ffs']} "
              f"BRAMs={res['brams']} DSPs={res['dsps']}")

    if args.mode == "simulate":
        from kllm.circuit_graph import CircuitGraph, evaluate
        from kllm.hdl_simulate import simulate

        print("HDL simulation — building test circuit...")

        # Build a small self-contained test graph (scalar ops)
        g = CircuitGraph()
        a = g.const(np.float32(1.5), name="a")
        b = g.const(np.float32(2.0), name="b")
        c = g.add(a, b, name="sum")       # 3.5
        d = g.mul(a, b, name="product")   # 3.0
        e = g.neg(c, name="neg_sum")      # -3.5
        _f = g.sub(d, e, name="result")   # 3.0 - (-3.5) = 6.5

        values = evaluate(g)
        print(f"  Test graph: {len(g)} nodes, "
              f"{sum(1 for n in g.nodes if n.op.value not in ('const','input'))} gates")

        sim_dir = os.path.join(args.save_dir, "sim")
        result = simulate(g, values, work_dir=sim_dir, verbose=True)

        if result["passed"]:
            print("\nSimulation PASSED")
        else:
            print("\nSimulation FAILED")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
