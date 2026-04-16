"""Tests for HDL export — Verilog, VHDL, testbench, resource estimation.

Verifies that:
1. Exported Verilog/VHDL is syntactically valid (parseable structure)
2. All graph nodes are represented in the output
3. Testbench contains golden values
4. Resource estimates are reasonable
"""

import numpy as np
import pytest

from kllm.graph.circuit_graph import CircuitGraph, Op
from kllm.graph.evaluator import evaluate
from kllm.hdl.hdl_export import (
    export_verilog,
    export_vhdl,
    export_testbench,
    estimate_resources,
)


def _read(path: str) -> str:
    """Read a generated HDL file back for assertion checks."""
    with open(path) as f:
        return f.read()


@pytest.fixture
def simple_graph():
    """Graph: const → add → neg."""
    g = CircuitGraph()
    a = g.const(np.array([1.0, 2.0], dtype=np.float32))
    b = g.const(np.array([3.0, 4.0], dtype=np.float32))
    c = g.add(a, b)
    d = g.neg(c)
    return g


@pytest.fixture
def graph_with_lut():
    """Graph with LUT activation."""
    g = CircuitGraph()
    x = g.const(np.array([1.0, -1.0, 0.5], dtype=np.float32))
    y = g.lut(x, "silu")
    return g


@pytest.fixture
def graph_with_input():
    """Graph with an input node."""
    g = CircuitGraph()
    x = g.input((4,), name="token_embed")
    w = g.const(np.eye(4, dtype=np.float32))
    y = g.matmul(x, w)
    return g


# ---------------------------------------------------------------
# Verilog export
# ---------------------------------------------------------------

class TestVerilogExport:
    def test_generates_file(self, simple_graph, tmp_path):
        path = str(tmp_path / "test.v")
        export_verilog(simple_graph, path)
        source = _read(path)
        assert (tmp_path / "test.v").exists()
        assert len(source) > 0

    def test_contains_module(self, simple_graph, tmp_path):
        path = str(tmp_path / "test.v")
        export_verilog(simple_graph, path, module_name="test_top")
        source = _read(path)
        assert "module test_top" in source
        assert "endmodule" in source

    def test_contains_constants(self, simple_graph, tmp_path):
        path = str(tmp_path / "test.v")
        export_verilog(simple_graph, path)
        source = _read(path)
        assert "n_0" in source  # first const
        assert "n_1" in source  # second const

    def test_contains_add_node(self, simple_graph, tmp_path):
        path = str(tmp_path / "test.v")
        export_verilog(simple_graph, path)
        source = _read(path)
        assert "fp_add" in source

    def test_contains_neg_as_xor(self, simple_graph, tmp_path):
        path = str(tmp_path / "test.v")
        export_verilog(simple_graph, path)
        source = _read(path)
        assert "flip sign" in source or "80000000" in source

    def test_contains_lut_module(self, graph_with_lut, tmp_path):
        path = str(tmp_path / "test.v")
        export_verilog(graph_with_lut, path)
        source = _read(path)
        assert "lut_silu" in source

    def test_contains_input_ports(self, graph_with_input, tmp_path):
        path = str(tmp_path / "test.v")
        export_verilog(graph_with_input, path)
        source = _read(path)
        assert "input" in source
        assert "in_0" in source  # input node port

    def test_contains_matmul(self, graph_with_input, tmp_path):
        path = str(tmp_path / "test.v")
        export_verilog(graph_with_input, path)
        source = _read(path)
        assert "fp_matmul" in source

    def test_pipeline_registers(self, simple_graph, tmp_path):
        path = str(tmp_path / "test.v")
        export_verilog(simple_graph, path, pipeline_depth=2)
        source = _read(path)
        assert "Pipeline stage" in source

    def test_helper_modules_emitted(self, simple_graph, tmp_path):
        path = str(tmp_path / "test.v")
        export_verilog(simple_graph, path)
        source = _read(path)
        # Should have fp_add behavioral module
        assert "module fp_add" in source


# ---------------------------------------------------------------
# VHDL export
# ---------------------------------------------------------------

class TestVHDLExport:
    def test_generates_file(self, simple_graph, tmp_path):
        path = str(tmp_path / "test.vhd")
        export_vhdl(simple_graph, path)
        source = _read(path)
        assert (tmp_path / "test.vhd").exists()
        assert len(source) > 0

    def test_contains_entity(self, simple_graph, tmp_path):
        path = str(tmp_path / "test.vhd")
        export_vhdl(simple_graph, path, entity_name="test_top")
        source = _read(path)
        assert "entity test_top" in source
        assert "end entity" in source

    def test_contains_architecture(self, simple_graph, tmp_path):
        path = str(tmp_path / "test.vhd")
        export_vhdl(simple_graph, path)
        source = _read(path)
        assert "architecture rtl" in source
        assert "begin" in source

    def test_contains_ieee_libs(self, simple_graph, tmp_path):
        path = str(tmp_path / "test.vhd")
        export_vhdl(simple_graph, path)
        source = _read(path)
        assert "library IEEE" in source
        assert "std_logic_1164" in source

    def test_contains_neg_as_xor(self, simple_graph, tmp_path):
        path = str(tmp_path / "test.vhd")
        export_vhdl(simple_graph, path)
        source = _read(path)
        assert "xor" in source or "80000000" in source

    def test_contains_lut_call(self, graph_with_lut, tmp_path):
        path = str(tmp_path / "test.vhd")
        export_vhdl(graph_with_lut, path)
        source = _read(path)
        assert "lut_silu" in source


# ---------------------------------------------------------------
# Testbench
# ---------------------------------------------------------------

class TestTestbench:
    def test_generates_testbench(self, simple_graph, tmp_path):
        values = evaluate(simple_graph)
        path = str(tmp_path / "tb.sv")
        source = export_testbench(simple_graph, path, values)
        assert (tmp_path / "tb.sv").exists()
        assert "module tb_" in source

    def test_contains_dut_instance(self, simple_graph, tmp_path):
        values = evaluate(simple_graph)
        path = str(tmp_path / "tb.sv")
        source = export_testbench(simple_graph, path, values)
        assert "circuit_top dut" in source

    def test_contains_golden_check(self, simple_graph, tmp_path):
        values = evaluate(simple_graph)
        path = str(tmp_path / "tb.sv")
        source = export_testbench(simple_graph, path, values)
        assert "PASS" in source
        assert "MISMATCH" in source

    def test_contains_finish(self, simple_graph, tmp_path):
        values = evaluate(simple_graph)
        path = str(tmp_path / "tb.sv")
        source = export_testbench(simple_graph, path, values)
        assert "$finish" in source


# ---------------------------------------------------------------
# Resource estimation
# ---------------------------------------------------------------

class TestResourceEstimation:
    def test_returns_dict(self, simple_graph):
        est = estimate_resources(simple_graph)
        assert isinstance(est, dict)
        assert "luts" in est
        assert "dsps" in est
        assert "brams" in est

    def test_add_costs_luts(self):
        g = CircuitGraph()
        a = g.const(np.array([1.0], dtype=np.float32))
        b = g.const(np.array([2.0], dtype=np.float32))
        g.add(a, b)
        est = estimate_resources(g)
        assert est["luts"] > 0

    def test_lut_costs_bram(self):
        g = CircuitGraph()
        x = g.const(np.array([1.0], dtype=np.float32))
        g.lut(x, "silu")
        est = estimate_resources(g)
        assert est["brams"] > 0

    def test_mul_costs_dsps(self):
        g = CircuitGraph()
        a = g.const(np.array([1.0], dtype=np.float32))
        b = g.const(np.array([2.0], dtype=np.float32))
        g.mul(a, b)
        est = estimate_resources(g)
        assert est["dsps"] > 0

    def test_wire_ops_free(self):
        g = CircuitGraph()
        x = g.const(np.arange(6, dtype=np.float32))
        g.reshape(x, (2, 3))
        est = estimate_resources(g)
        assert est["luts"] == 0
        assert est["dsps"] == 0

    def test_compiled_model_estimation(self):
        """Resource estimate for a compiled model."""
        from kllm.compiler.circuit_compiler import compile_model
        from tests.test_circuit_compiler import MockFabric

        fabric = MockFabric(num_layers=1, hidden_size=8, num_heads=2,
                            num_kv_heads=2, intermediate_size=16,
                            vocab_size=32)
        graph, _, _kv = compile_model(fabric, [1, 5, 10])
        est = estimate_resources(graph)
        assert est["total_nodes"] > 0
        assert est["estimated_fmax_mhz"] > 0


# ---------------------------------------------------------------
# Round-trip: graph → HDL → check structure
# ---------------------------------------------------------------

class TestRoundTrip:
    def test_all_ops_represented(self, tmp_path):
        """Every graph op type should produce some Verilog output."""
        g = CircuitGraph()
        a = g.const(np.array([1.0, 2.0], dtype=np.float32))
        b = g.const(np.array([3.0, 4.0], dtype=np.float32))
        c = g.add(a, b)
        d = g.sub(a, b)
        e = g.mul(a, b)
        f = g.neg(c)
        h = g.abs(d)
        i = g.lut(e, "exp")
        j = g.sum(i, axis=-1)

        path = str(tmp_path / "all_ops.v")
        export_verilog(g, path)
        source = _read(path)

        # Each op should have a corresponding node
        assert "fp_add" in source
        assert "fp_sub" in source
        assert "fp_mul" in source
        assert "neg" in source
        assert "abs" in source
        assert "lut_exp" in source
        assert "reduce_sum" in source


# ---------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------

class TestSimulation:
    def test_structural_verification(self, tmp_path):
        """Structural verifier catches all constants and connectivity."""
        from kllm.hdl.hdl_simulate import simulate

        g = CircuitGraph()
        a = g.const(np.float32(2.5), name="a")
        b = g.const(np.float32(1.0), name="b")
        g.add(a, b, name="c")

        result = simulate(g, work_dir=str(tmp_path), verbose=False)
        s = result["structural"]
        assert s["consts_ok"] == s["consts_total"]
        assert s["dangling_inputs"] == 0

    def test_iverilog_simulation(self, tmp_path):
        """Full iverilog simulation with golden value check."""
        import shutil
        from kllm.hdl.hdl_simulate import simulate

        if not shutil.which("iverilog"):
            pytest.skip("iverilog not installed")

        g = CircuitGraph()
        a = g.const(np.float32(3.0), name="a")
        b = g.const(np.float32(4.0), name="b")
        g.add(a, b, name="sum")   # 7.0

        result = simulate(g, work_dir=str(tmp_path), verbose=False)
        assert result["passed"]
        iv = result["iverilog"]
        assert iv["passed"]
        assert iv["passes"] == 1
        assert iv["mismatches"] == 0
