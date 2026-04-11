"""Lossless IEEE-754 sub-bit mask operations.

Every 32-bit float is decomposed into four 8-bit planes without any
scaling or quantization.  The raw memory is reinterpreted as uint32
and sliced with bitmasks, guaranteeing zero precision loss.
"""

import numpy as np


def extract_sub_masks(fp32_array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Losslessly split an FP32 array into four 8-bit planes.

    Parameters
    ----------
    fp32_array : np.ndarray
        Any-shape array of float32 values.

    Returns
    -------
    tuple of four np.ndarray (uint8)
        mask_0 (bits 0-7), mask_1 (bits 8-15),
        mask_2 (bits 16-23), mask_3 (bits 24-31, sign+exponent).
    """
    raw = fp32_array.astype(np.float32).view(np.uint32)
    mask_0 = (raw & 0x000000FF).astype(np.uint8)
    mask_1 = ((raw & 0x0000FF00) >> 8).astype(np.uint8)
    mask_2 = ((raw & 0x00FF0000) >> 16).astype(np.uint8)
    mask_3 = ((raw & 0xFF000000) >> 24).astype(np.uint8)
    return mask_0, mask_1, mask_2, mask_3


def repack_sub_masks(
    m0: np.ndarray, m1: np.ndarray, m2: np.ndarray, m3: np.ndarray
) -> np.ndarray:
    """Reconstruct FP32 values from four 8-bit planes (lossless).

    Parameters
    ----------
    m0, m1, m2, m3 : np.ndarray (uint8)
        The four sub-bit mask planes.

    Returns
    -------
    np.ndarray (float32)
    """
    packed = (
        m3.astype(np.uint32) << 24
        | m2.astype(np.uint32) << 16
        | m1.astype(np.uint32) << 8
        | m0.astype(np.uint32)
    )
    return packed.view(np.float32)
