from kllm.device import GPU_AVAILABLE, free_vram, to_device, to_numpy, xp
import numpy as np


class TestDeviceAbstraction:
    def test_xp_is_numpy_when_no_gpu(self):
        """On a machine without CuPy, xp should be numpy."""
        # We can't force the import to fail, but we can verify it's one of the two.
        assert xp is np or xp.__name__ == "cupy"

    def test_to_device_returns_array(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = to_device(arr)
        assert result.shape == (3,)

    def test_to_numpy_roundtrip(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        on_device = to_device(arr)
        back = to_numpy(on_device)
        assert isinstance(back, np.ndarray)
        assert np.array_equal(back, arr)

    def test_to_numpy_passthrough(self):
        arr = np.array([5, 6, 7])
        assert np.array_equal(to_numpy(arr), arr)

    def test_free_vram_does_not_raise(self):
        """free_vram should be safe to call regardless of GPU presence."""
        free_vram()

    def test_gpu_available_is_bool(self):
        assert isinstance(GPU_AVAILABLE, bool)
