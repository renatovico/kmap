"""CLI entry-point for kllm.

Usage
-----
  kllm --mode compile          --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
  kllm --mode compile-circuits
  kllm --mode inference        --text "Hello"
  kllm --mode stream           --text "Hello"
  kllm --mode compare          --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --text "Hello"
  kllm --mode full             --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
  kllm --mode circuit-compile  --save-dir ./lossless_logic --text "Hello"
  kllm --mode circuit-optimize --save-dir ./lossless_logic
  kllm --mode circuit-eval     --save-dir ./lossless_logic
  kllm --mode export-hdl       --save-dir ./lossless_logic
"""

import argparse
import os


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kllm",
        description="Lossless Bit-Sliced Logic Engine for LLM inference",
    )
    p.add_argument(
        "--mode",
        choices=[
            "compile", "compile-circuits", "optimize-circuits",
            "inference", "generate", "stream",
            "compare", "full",
            "circuit-compile", "circuit-optimize", "circuit-eval",
            "export-hdl",
        ],
        required=True,
        help="compile: bake model into logic fabric · "
        "compile-circuits: compile Z3 arithmetic circuits · "
        "optimize-circuits: Kmap/Espresso logic minimization · "
        "inference/generate: generate text · "
        "stream: streaming token generation · "
        "compare: HuggingFace vs kllm · "
        "full: compile + compile-circuits + generate · "
        "circuit-compile: compile model to circuit graph DAG · "
        "circuit-optimize: run graph optimizer on circuit graph · "
        "circuit-eval: evaluate circuit graph with C executor · "
        "export-hdl: export circuit graph to Verilog/VHDL",
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

    if args.mode in ("compile", "full"):
        from kllm.compiler import LosslessLogicCompiler

        compiler = LosslessLogicCompiler(
            model_name=args.model,
            save_dir=args.save_dir,
            solver_timeout=args.solver_timeout,
        )
        compiler.compile()

    if args.mode in ("compile-circuits", "full"):
        from kllm.ops_compiler import OpsCompiler

        ops = OpsCompiler(save_dir=args.save_dir)
        ops.compile()

    if args.mode == "optimize-circuits":
        from kllm.optimizer import CircuitOptimizer

        opt = CircuitOptimizer(save_dir=args.save_dir)
        opt.optimize()

    if args.mode in ("inference", "generate", "full"):
        from kllm.inference import BitLogicInferenceEngine

        engine = BitLogicInferenceEngine(
            save_dir=args.save_dir,
        )

        text = _get_text(args)
        messages = [{"role": "user", "content": text}]
        prompt = engine.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True,
        )
        output = engine.generate(prompt, max_new_tokens=args.max_tokens)
        print(f"\n--- kllm Generated Text ---")
        print(output)

    if args.mode == "stream":
        from kllm.inference import BitLogicInferenceEngine

        engine = BitLogicInferenceEngine(
            save_dir=args.save_dir,
        )

        text = _get_text(args)
        messages = [{"role": "user", "content": text}]
        prompt = engine.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True,
        )
        print(f"\n--- kllm Streaming ---")
        for tok in engine.stream(prompt, max_new_tokens=args.max_tokens):
            print(tok, end="", flush=True)
        print()

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

    if args.mode == "circuit-compile":
        from kllm.circuit_compiler import compile_model
        from kllm.fabric import Fabric
        from kllm.tokenizer import Tokenizer

        fabric = Fabric(args.save_dir)
        tok_dir = os.path.join(args.save_dir, "tokenizer")
        tokenizer = Tokenizer(tok_dir)
        text = _get_text(args)
        messages = [{"role": "user", "content": text}]
        token_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True,
        )
        print(f"Compiling {len(token_ids)} tokens to circuit graph...")
        graph, logits_id = compile_model(
            fabric, token_ids,
        )
        graph_dir = os.path.join(args.save_dir, "circuit_graph")
        graph.serialize(graph_dir)
        print(f"Circuit graph saved to {graph_dir}")
        print(f"  Nodes: {len(graph.nodes)}  Gates: {graph.gate_count()}")

    if args.mode == "circuit-optimize":
        from kllm.circuit_graph import CircuitGraph
        from kllm.graph_optimizer import optimize_graph, optimization_stats

        graph_dir = os.path.join(args.save_dir, "circuit_graph")
        print(f"Loading circuit graph from {graph_dir}...")
        graph = CircuitGraph.deserialize(graph_dir)
        output_ids = [n.id for n in graph.nodes if n.name and "logits" in n.name]
        if not output_ids:
            output_ids = [graph.nodes[-1].id]
        print(f"Optimizing ({len(graph.nodes)} nodes, {len(output_ids)} outputs)...")
        opt_graph, id_map = optimize_graph(graph, output_ids)
        stats = optimization_stats(graph, opt_graph)
        opt_dir = os.path.join(args.save_dir, "circuit_graph_opt")
        opt_graph.serialize(opt_dir)
        print(f"Optimized graph saved to {opt_dir}")
        print(f"  Before: {stats['original_nodes']} nodes")
        print(f"  After:  {stats['optimized_nodes']} nodes")
        print(f"  Reduction: {stats['gate_reduction_pct']:.1f}%")

    if args.mode == "circuit-eval":
        from kllm.circuit_executor import evaluate_c
        from kllm.circuit_graph import CircuitGraph

        opt_dir = os.path.join(args.save_dir, "circuit_graph_opt")
        graph_dir = os.path.join(args.save_dir, "circuit_graph")
        load_dir = opt_dir if os.path.isdir(opt_dir) else graph_dir
        print(f"Loading circuit graph from {load_dir}...")
        graph = CircuitGraph.deserialize(load_dir)
        print(f"Evaluating {len(graph.nodes)} nodes with C executor...")
        results = evaluate_c(graph)
        output_nodes = [n for n in graph.nodes if n.name and "logits" in n.name]
        if not output_nodes:
            output_nodes = [graph.nodes[-1]]
        for node in output_nodes:
            arr = results[node.id]
            print(f"  {node.name or f'node_{node.id}'}: shape={arr.shape} "
                  f"dtype={arr.dtype}")

    if args.mode == "export-hdl":
        from kllm.circuit_graph import CircuitGraph
        from kllm.hdl_export import (
            export_verilog, export_vhdl, estimate_resources,
        )

        opt_dir = os.path.join(args.save_dir, "circuit_graph_opt")
        graph_dir = os.path.join(args.save_dir, "circuit_graph")
        load_dir = opt_dir if os.path.isdir(opt_dir) else graph_dir
        print(f"Loading circuit graph from {load_dir}...")
        graph = CircuitGraph.deserialize(load_dir)
        hdl_dir = os.path.join(args.save_dir, "hdl")
        os.makedirs(hdl_dir, exist_ok=True)
        if args.output_format == "verilog":
            path = os.path.join(hdl_dir, "circuit_top.v")
            export_verilog(graph, path)
            print(f"Verilog exported to {path}")
        else:
            path = os.path.join(hdl_dir, "circuit_top.vhd")
            export_vhdl(graph, path)
            print(f"VHDL exported to {path}")
        res = estimate_resources(graph)
        print(f"Estimated resources: LUTs={res['luts']} FFs={res['ffs']} "
              f"BRAMs={res['brams']} DSPs={res['dsps']}")


if __name__ == "__main__":
    main()
