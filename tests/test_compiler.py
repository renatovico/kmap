import numpy as np
import pytest

from kllm.compiler import LosslessLogicCompiler, _store_weight_gates


class TestSolvePattern:
    def test_target_zero(self):
        """Z3 solves target=0: uint8(0xFF << s1) ^ mask == 0."""
        s1, mask = LosslessLogicCompiler._solve_pattern(0)
        assert isinstance(s1, int)
        assert isinstance(mask, int)
        # Gate recovery must produce the target byte.
        recovered = ((0xFF << s1) & 0xFF) ^ mask
        assert recovered == 0

    def test_target_255(self):
        """Z3 solves target=255: uint8(0xFF << s1) ^ mask == 255."""
        s1, mask = LosslessLogicCompiler._solve_pattern(255)
        recovered = ((0xFF << s1) & 0xFF) ^ mask
        assert recovered == 255

    def test_returns_tuple_of_two_ints(self):
        for val in (0, 1, 42, 127, 200, 255):
            result = LosslessLogicCompiler._solve_pattern(val)
            assert len(result) == 2
            assert all(isinstance(v, int) for v in result)

    def test_gate_recovery_all_256(self):
        """Z3 proves ALL 256 byte values — no fallback needed."""
        for target in range(256):
            s1, mask = LosslessLogicCompiler._solve_pattern(target)
            recovered = ((0xFF << s1) & 0xFF) ^ mask
            assert recovered == target, (
                f"Gate recovery failed for target={target}: "
                f"s1={s1}, mask={mask}, got={recovered}"
            )

    def test_shift_within_range(self):
        """All solved shifts are in [0, 7]."""
        for target in (0, 1, 42, 170, 254, 255):
            s1, mask = LosslessLogicCompiler._solve_pattern(target)
            assert 0 <= s1 <= 7

    def test_custom_timeout(self):
        """Solver accepts a custom timeout without error."""
        s1, mask = LosslessLogicCompiler._solve_pattern(42, timeout=50)
        assert isinstance(s1, int)
        assert isinstance(mask, int)


class TestSolveUnique:
    def test_caches_results(self):
        import numpy as np

        compiler = LosslessLogicCompiler(model_name="dummy", save_dir="/tmp/kllm_test")
        unique = np.array([0, 1, 2, 1, 0], dtype=np.uint8)
        registry = compiler._solve_unique(unique)
        assert 0 in registry
        assert 1 in registry
        assert 2 in registry
        # Cache should have been populated.
        assert 0 in compiler._logic_cache
        assert 1 in compiler._logic_cache
        assert 2 in compiler._logic_cache

    def test_registry_values_are_tuples(self):
        import numpy as np

        compiler = LosslessLogicCompiler(model_name="dummy", save_dir="/tmp/kllm_test")
        unique = np.array([10, 20, 30], dtype=np.uint8)
        registry = compiler._solve_unique(unique)
        for val, (s1, mask) in registry.items():
            assert isinstance(s1, int)
            assert isinstance(mask, int)


class TestBuildGateLut:
    def test_lut_shape_and_dtype(self):
        compiler = LosslessLogicCompiler(model_name="dummy", save_dir="/tmp/kllm_test")
        s1_lut, mask_lut = compiler._build_gate_lut()
        assert s1_lut.shape == (256,)
        assert mask_lut.shape == (256,)
        assert s1_lut.dtype == np.uint8
        assert mask_lut.dtype == np.uint8

    def test_lut_shift_values_within_range(self):
        compiler = LosslessLogicCompiler(model_name="dummy", save_dir="/tmp/kllm_test")
        s1_lut, _ = compiler._build_gate_lut()
        assert np.all(s1_lut <= 7)


class TestStoreWeightGates:
    def test_stores_planes_and_gates(self):
        compiler = LosslessLogicCompiler(model_name="dummy", save_dir="/tmp/kllm_test")
        s1_lut, mask_lut = compiler._build_gate_lut()

        weight = np.array([[1.0, -2.0], [0.5, 3.14]], dtype=np.float32)
        dest: dict = {}
        _store_weight_gates(dest, "test", weight, s1_lut, mask_lut)

        # Should have byte planes AND gate arrays
        for i in range(4):
            assert f"test_m{i}" in dest
            assert f"test_m{i}_s1" in dest
            assert f"test_m{i}_mask" in dest
            assert dest[f"test_m{i}_s1"].shape == weight.shape
            assert dest[f"test_m{i}_mask"].shape == weight.shape
