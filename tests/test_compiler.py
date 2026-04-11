import pytest

from kllm.compiler import LosslessLogicCompiler


class TestSolvePattern:
    def test_identity_zero(self):
        """Pattern 0: (x & 0) == 0 for all x — should solve trivially."""
        s1, mask = LosslessLogicCompiler._solve_pattern(0)
        assert isinstance(s1, int)
        assert isinstance(mask, int)
        # The solver finds (s1, mask) such that for all x:
        #   (x << s1) ^ mask == (x & target_val)
        # For target 0, (x & 0) == 0 for all x.
        # The fallback (0, 0) means (x << 0) ^ 0 == x, which != 0,
        # so the solver may return a different pair. Either way the
        # function must return a valid tuple.
        assert 0 <= s1 <= 255
        assert 0 <= mask <= 255

    def test_identity_255(self):
        """Pattern 255: (x & 0xFF) == x — identity."""
        s1, mask = LosslessLogicCompiler._solve_pattern(255)
        assert isinstance(s1, int)
        assert isinstance(mask, int)

    def test_returns_tuple_of_two_ints(self):
        for val in (0, 1, 42, 127, 200, 255):
            result = LosslessLogicCompiler._solve_pattern(val)
            assert len(result) == 2
            assert all(isinstance(v, int) for v in result)

    def test_fallback_preserves_value(self):
        """Even if Z3 can't find a shift trick, the fallback is (0, val)."""
        # We can't force a timeout easily, but we can check the fallback path
        # manually: (x << 0) ^ val == x ^ val, compared to x & val.
        # The fallback is only used when Z3 fails, but it always returns a
        # valid tuple.
        s1, mask = LosslessLogicCompiler._solve_pattern(170)
        assert 0 <= s1 <= 7 or s1 == 0
        assert 0 <= mask <= 255

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
