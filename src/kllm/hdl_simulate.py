"""Run HDL simulation to verify exported circuits.

Supports two simulation backends:

1. **Icarus Verilog** (``iverilog`` + ``vvp``) — full RTL simulation
2. **Python structural verifier** — always available, checks graph
   structure and constant hex encoding round-trip

Usage::

    from kllm.hdl_simulate import simulate
    result = simulate(graph, values, work_dir="./sim_out")
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import numpy as np

from kllm.circuit_graph import CircuitGraph, Op
from kllm.evaluator import evaluate
from kllm.hdl_export import (
    export_verilog,
    export_testbench,
    _find_output_nodes,
    _tensor_bits,
    _array_to_hex,
)


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

def simulate(
    graph: CircuitGraph,
    values: dict[int, np.ndarray] | None = None,
    work_dir: str | None = None,
    verbose: bool = True,
) -> dict:
    """Export, verify, and optionally simulate a circuit graph.

    Parameters
    ----------
    graph : CircuitGraph
        The circuit to simulate.
    values : dict, optional
        Pre-computed golden values.  If *None*, runs ``evaluate(graph)``.
    work_dir : str, optional
        Directory for simulation artifacts.  Uses a temp dir if *None*.
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    dict with keys:
        passed : bool — all checks passed
        structural : dict — structural verification results
        iverilog : dict | None — iverilog simulation results (if available)
    """
    if values is None:
        values = evaluate(graph)

    cleanup = False
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="kllm_sim_")
        cleanup = True

    os.makedirs(work_dir, exist_ok=True)

    result: dict = {"passed": False, "structural": {}, "iverilog": None}

    try:
        # 1. Structural verification (always runs)
        if verbose:
            print("  [1/3] Structural verification...")
        result["structural"] = _verify_structural(graph, values)
        s = result["structural"]
        if verbose:
            print(f"        Nodes: {s['total_nodes']}  "
                  f"Consts verified: {s['consts_ok']}/{s['consts_total']}  "
                  f"Outputs: {s['num_outputs']}")

        # 2. Export Verilog + testbench
        if verbose:
            print("  [2/3] Exporting Verilog + testbench...")
        v_path = os.path.join(work_dir, "circuit_top.v")
        tb_path = os.path.join(work_dir, "tb_circuit_top.sv")
        export_verilog(graph, v_path)
        export_testbench(graph, tb_path, values)
        if verbose:
            v_size = os.path.getsize(v_path)
            print(f"        Verilog: {v_path} ({v_size:,} bytes)")
            print(f"        Testbench: {tb_path}")

        # 3. Icarus Verilog simulation (if available)
        if verbose:
            print("  [3/3] Icarus Verilog simulation...")
        iverilog = shutil.which("iverilog")
        vvp = shutil.which("vvp")
        if iverilog and vvp:
            result["iverilog"] = _run_iverilog(
                v_path, tb_path, work_dir, iverilog, vvp, verbose)
        else:
            if verbose:
                print("        iverilog not found — skipping RTL sim")
                print("        Install: brew install icarus-verilog")
            result["iverilog"] = {"skipped": True}

        # Overall pass/fail
        struct_ok = s["consts_ok"] == s["consts_total"]
        iv = result["iverilog"] or {}
        iv_ok = iv.get("skipped", False) or iv.get("passed", False)
        result["passed"] = struct_ok and iv_ok

    finally:
        if cleanup and not verbose:
            shutil.rmtree(work_dir, ignore_errors=True)

    return result


# ---------------------------------------------------------------
# Structural verifier
# ---------------------------------------------------------------

def _verify_structural(
    graph: CircuitGraph,
    values: dict[int, np.ndarray],
) -> dict:
    """Verify graph structure: constants round-trip, connectivity."""
    consts_total = 0
    consts_ok = 0

    for node in graph.nodes:
        if node.op == Op.CONST:
            consts_total += 1
            val = node.params.get("value")
            if val is None:
                continue
            # Verify hex round-trip
            arr = np.ascontiguousarray(val, dtype=np.float32)
            hex_str = _array_to_hex(arr, 32)
            # Decode back
            raw = bytes.fromhex(hex_str)
            recovered = np.frombuffer(raw, dtype=np.uint8)[::-1].copy()
            recovered = recovered.view(np.float32).reshape(arr.shape)
            if np.array_equal(arr.view(np.uint32), recovered.view(np.uint32)):
                consts_ok += 1

    # Check all non-CONST/INPUT nodes have valid inputs
    dangling = 0
    for node in graph.nodes:
        if node.op in (Op.CONST, Op.INPUT):
            continue
        for inp in node.inputs:
            if inp >= len(graph.nodes):
                dangling += 1

    outputs = _find_output_nodes(graph)

    return {
        "total_nodes": len(graph),
        "consts_total": consts_total,
        "consts_ok": consts_ok,
        "dangling_inputs": dangling,
        "num_outputs": len(outputs),
    }


# ---------------------------------------------------------------
# Icarus Verilog runner
# ---------------------------------------------------------------

def _run_iverilog(
    v_path: str,
    tb_path: str,
    work_dir: str,
    iverilog_bin: str,
    vvp_bin: str,
    verbose: bool,
) -> dict:
    """Compile and run with Icarus Verilog."""
    vvp_out = os.path.join(work_dir, "sim.vvp")

    # Compile
    compile_cmd = [
        iverilog_bin, "-g2012",      # SystemVerilog
        "-o", vvp_out,
        tb_path, v_path,
    ]
    if verbose:
        print(f"        $ {' '.join(compile_cmd)}")

    cp = subprocess.run(
        compile_cmd,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if cp.returncode != 0:
        if verbose:
            print(f"        Compile FAILED (rc={cp.returncode})")
            for line in cp.stderr.splitlines()[:10]:
                print(f"          {line}")
        return {"passed": False, "phase": "compile",
                "stderr": cp.stderr[:2000]}

    # Simulate
    sim_cmd = [vvp_bin, vvp_out]
    if verbose:
        print(f"        $ {' '.join(sim_cmd)}")

    sp = subprocess.run(
        sim_cmd,
        capture_output=True,
        text=True,
        timeout=300,
    )

    stdout = sp.stdout
    if verbose:
        for line in stdout.splitlines():
            print(f"        {line}")

    passed = "MISMATCH" not in stdout and sp.returncode == 0
    passes = stdout.count("PASS:")
    mismatches = stdout.count("MISMATCH:")

    if verbose:
        tag = "PASS" if passed else "FAIL"
        print(f"        Simulation {tag}  "
              f"(passes={passes}, mismatches={mismatches})")

    return {
        "passed": passed,
        "phase": "simulate",
        "passes": passes,
        "mismatches": mismatches,
        "stdout": stdout[:4000],
    }
