"""GPU/CPU array abstraction.

Transparently uses CuPy when a CUDA GPU is available and falls back
to NumPy otherwise.  Every other module imports ``xp`` from here so
the decision is made in a single place.
"""

import numpy as np

try:
    import cupy as cp

    GPU_AVAILABLE = True
    xp = cp
except ImportError:
    GPU_AVAILABLE = False
    xp = np


def to_device(arr: np.ndarray):
    """Move a NumPy array to the active device (GPU or CPU)."""
    if GPU_AVAILABLE:
        return cp.array(arr)
    return arr


def to_numpy(arr) -> np.ndarray:
    """Bring an array back to CPU (NumPy)."""
    if GPU_AVAILABLE and hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)


def free_vram():
    """Release unused GPU memory (no-op on CPU)."""
    if GPU_AVAILABLE:
        cp.get_default_memory_pool().free_all_blocks()
