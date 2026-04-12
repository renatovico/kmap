"""Tests for Z3 circuit primitives and arithmetic unit."""

import numpy as np
import pytest

from kllm.circuits import (
    ArithmeticUnit,
    _exp_fn,
    _rsqrt_fn,
    _silu_fn,
    bytes_to_float32,
    exec_constant,
    exec_unary,
    float32_to_bytes,
    solve_constant_gate,
    solve_unary_gate,
)


# ------------------------------------------------------------------
# Byte helpers
# ------------------------------------------------------------------

class TestByteHelpers:
    def test_float32_roundtrip(self):
        x = np.array([1.0, -2.5, 0.0, 3.14], dtype=np.float32)
        buf = float32_to_bytes(x)
        assert buf.shape == (4, 4)
        assert buf.dtype == np.uint8
        y = bytes_to_float32(buf)
        np.testing.assert_array_equal(x, y)

    def test_float32_roundtrip_2d(self):
        x = np.random.randn(3, 5).astype(np.float32)
        buf = float32_to_bytes(x)
        assert buf.shape == (3, 5, 4)
        y = bytes_to_float32(buf)
        np.testing.assert_array_equal(x, y)


# ------------------------------------------------------------------
# Constant gate synthesis
# ------------------------------------------------------------------

class TestConstantGate:
    def test_solve_all_256(self):
        s1, mask = solve_constant_gate(timeout=200)
        assert s1.shape == (256,)
        assert mask.shape == (256,)
        assert s1.dtype == np.uint8
        # Verify: (0xFF << s1[v]) ^ mask[v] == v for all v
        for v in range(256):
            result = (np.uint8(0xFF) << s1[v]).astype(np.uint8) ^ mask[v]
            assert int(result) == v, f"Failed for v={v}"


# ------------------------------------------------------------------
# Unary gate synthesis
# ------------------------------------------------------------------

class TestUnaryGate:
    def test_identity(self):
        """Identity LUT: output == input."""
        target = np.arange(256, dtype=np.uint8)
        s1, mask = solve_unary_gate(target, timeout=200)
        for i in range(256):
            result = (np.uint8(i) << s1[i]).astype(np.uint8) ^ mask[i]
            assert int(result) == i

    def test_bitwise_not(self):
        """Bitwise NOT: output == ~input."""
        target = (~np.arange(256, dtype=np.uint8)).astype(np.uint8)
        s1, mask = solve_unary_gate(target, timeout=200)
        for i in range(256):
            result = (np.uint8(i) << s1[i]).astype(np.uint8) ^ mask[i]
            assert int(result) == int(target[i])

    def test_exec_vectorized(self):
        """exec_unary produces correct output for arrays."""
        target = np.arange(256, dtype=np.uint8)[::-1].copy()  # reverse
        s1, mask = solve_unary_gate(target, timeout=200)
        x = np.array([0, 1, 127, 255], dtype=np.uint8)
        out = exec_unary(x, s1, mask)
        expected = target[x]
        np.testing.assert_array_equal(out, expected)


# ------------------------------------------------------------------
# ArithmeticUnit
# ------------------------------------------------------------------

class TestArithmeticUnit:
    def test_compile_constant_gates(self):
        unit = ArithmeticUnit()
        unit.compile_constant_gates(timeout=200)
        assert unit.const_s1 is not None
        assert unit.const_s1.shape == (256,)

    def test_compile_silu(self):
        unit = ArithmeticUnit()
        unit.compile_constant_gates(timeout=200)
        x_domain = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        unit.compile_unary_op("silu", _silu_fn, x_domain, timeout=200)
        assert "silu" in unit.ops

    def test_exec_silu_matches(self):
        """Z3-compiled SiLU matches NumPy for the compiled domain."""
        unit = ArithmeticUnit()
        unit.compile_constant_gates()
        x_domain = np.linspace(-5.0, 5.0, 1000, dtype=np.float32)
        unit.compile_unary_op("silu", _silu_fn, x_domain)

        # Execute on exact domain values (no interpolation error)
        idx = [100, 300, 500, 700, 900]
        x_test = x_domain[idx]
        result = unit.exec_unary_op("silu", x_test)
        expected = _silu_fn(x_test)

        # Exact domain values should match precisely
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

    def test_exec_exp_matches(self):
        """Z3-compiled exp matches NumPy on exact domain values."""
        unit = ArithmeticUnit()
        unit.compile_constant_gates()
        x_domain = np.linspace(-10.0, 5.0, 5000, dtype=np.float32)
        unit.compile_unary_op("exp", _exp_fn, x_domain)

        # Use exact domain values — exact hash lookup, no approximation
        idx = [500, 1500, 2500, 3500, 4500]
        x_test = x_domain[idx]
        result = unit.exec_unary_op("exp", x_test)
        expected = _exp_fn(x_test)

        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

    def test_exec_rsqrt_matches(self):
        """Z3-compiled rsqrt matches NumPy on exact domain values."""
        unit = ArithmeticUnit()
        unit.compile_constant_gates()
        x_domain = np.linspace(0.001, 10.0, 1000, dtype=np.float32)
        unit.compile_unary_op("rsqrt", _rsqrt_fn, x_domain)

        # Use exact domain values
        idx = [10, 100, 300, 500, 800]
        x_test = x_domain[idx]
        result = unit.exec_unary_op("rsqrt", x_test)
        expected = _rsqrt_fn(x_test)

        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

    def test_save_load_roundtrip(self, tmp_path):
        """Save and load preserves all gate LUTs."""
        unit = ArithmeticUnit()
        unit.compile_constant_gates()
        x_domain = np.linspace(-3.0, 3.0, 100, dtype=np.float32)
        unit.compile_unary_op("silu", _silu_fn, x_domain)

        path = str(tmp_path / "circuits.npz")
        unit.save(path)

        loaded = ArithmeticUnit.load(path)
        assert "silu" in loaded.ops
        np.testing.assert_array_equal(loaded.const_s1, unit.const_s1)
        np.testing.assert_array_equal(loaded.const_mask, unit.const_mask)

        # Execution should give same results (use exact domain values)
        x_test = x_domain[[10, 50, 90]]
        r1 = unit.exec_unary_op("silu", x_test)
        r2 = loaded.exec_unary_op("silu", x_test)
        np.testing.assert_array_equal(r1, r2)

    def test_full_domain_compile_and_mmap(self, tmp_path):
        """Full-domain byte planes cover ANY float32 value."""
        unit = ArithmeticUnit()
        unit.compile_constant_gates()

        circuits_dir = str(tmp_path / "circuits")
        # Compile with a tiny chunk size for speed in tests
        unit.compile_full_domain(
            "silu", _silu_fn, circuits_dir, chunk_size=1 << 20,
        )

        # Save metadata + reload with mmap
        npz_path = str(tmp_path / "circuits.npz")
        unit.save(npz_path)
        loaded = ArithmeticUnit.load(npz_path)

        # mmap planes should be detected
        assert "silu" in loaded._planes
        assert len(loaded._planes["silu"]) == 4

        # Test on random values — full domain means ANY float works
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100).astype(np.float32)
        result = loaded.exec_unary_op("silu", x)
        expected = _silu_fn(x)
        np.testing.assert_array_equal(result, expected)


# ------------------------------------------------------------------
# CircuitMath
# ------------------------------------------------------------------

class TestCircuitMath:
    def test_no_unit_raises(self):
        """CircuitMath without a unit raises on activation calls."""
        from kllm.circuit_model import CircuitMath
        cm = CircuitMath(unit=None)
        x = np.array([0.0, 1.0, -1.0], dtype=np.float32)
        with pytest.raises(RuntimeError, match="No Z3 circuits"):
            cm.silu(x)

    def test_softmax_with_unit(self):
        """CircuitMath softmax works with compiled circuits."""
        from kllm.circuit_model import CircuitMath
        unit = ArithmeticUnit()
        unit.compile_constant_gates()
        # softmax computes (x - max(x)) then exp; ensure those values
        # are in the compiled domain.
        x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        softmax_inputs = (x - x.max(axis=-1, keepdims=True)).ravel()
        unit.compile_unary_op("exp", _exp_fn, softmax_inputs)
        cm = CircuitMath(unit=unit)
        result = cm.softmax(x)
        assert result.shape == (1, 3)
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-5)

    def test_circuit_silu(self):
        """CircuitMath with compiled unit uses Z3 gates."""
        from kllm.circuit_model import CircuitMath
        unit = ArithmeticUnit()
        unit.compile_constant_gates()
        x_domain = np.linspace(-5.0, 5.0, 5000, dtype=np.float32)
        unit.compile_unary_op("silu", _silu_fn, x_domain)

        cm = CircuitMath(unit=unit)
        assert cm.has_circuits
        # Use exact domain values — solver proves every compiled float32
        x = x_domain[[500, 2500, 1000]]
        result = cm.silu(x)
        expected = _silu_fn(x)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)


# ------------------------------------------------------------------
# Streaming
# ------------------------------------------------------------------

class TestCircuitTransformerStreaming:
    """Test generator-based streaming (requires compiled fabric)."""

    @pytest.fixture
    def has_fabric(self):
        import os
        return os.path.exists("./lossless_logic/meta.npz")

    def test_forward_gen_yields_stages(self, has_fabric):
        if not has_fabric:
            pytest.skip("No compiled fabric")

        from kllm.circuits import ArithmeticUnit as AU
        from kllm.circuit_model import CircuitTransformer
        from kllm.fabric import Fabric
        fabric = Fabric("./lossless_logic")
        circuit_path = "./lossless_logic/circuits.npz"
        import os
        if not os.path.exists(circuit_path):
            pytest.skip("No compiled circuits")
        unit = AU.load(circuit_path)
        model = CircuitTransformer(fabric, unit)

        stages = []
        for stage, li, data in model.forward_gen([1, 15043], start_pos=0):
            stages.append(stage)

        assert stages[0] == "embed"
        assert stages[-1] == "logits"
        assert "layer" in stages
        assert "norm" in stages

    def test_stream_yields_tokens(self, has_fabric):
        if not has_fabric:
            pytest.skip("No compiled fabric")

        from kllm.inference import BitLogicInferenceEngine
        engine = BitLogicInferenceEngine("./lossless_logic")
        # Use the same prompt & token count the circuits were traced with
        tokens = list(engine.stream("Hello, how are you today?", max_new_tokens=2))
        assert len(tokens) >= 1
        assert all(isinstance(t, str) for t in tokens)
