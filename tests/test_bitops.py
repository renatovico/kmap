import numpy as np
import pytest

from kllm.bitops import extract_sub_masks, repack_sub_masks


class TestExtractSubMasks:
    def test_roundtrip_lossless(self):
        """Extract + repack must reproduce the exact same bytes."""
        original = np.array([1.2345678, -0.009876, 0.0, 3.14159265], dtype=np.float32)
        m0, m1, m2, m3 = extract_sub_masks(original)
        restored = repack_sub_masks(m0, m1, m2, m3)
        assert np.array_equal(original.view(np.uint32), restored.view(np.uint32))

    def test_roundtrip_special_values(self):
        """Edge cases: inf, -inf, nan, subnormals, max float."""
        original = np.array(
            [np.inf, -np.inf, 0.0, -0.0, np.finfo(np.float32).max, np.finfo(np.float32).tiny],
            dtype=np.float32,
        )
        masks = extract_sub_masks(original)
        restored = repack_sub_masks(*masks)
        assert np.array_equal(original.view(np.uint32), restored.view(np.uint32))

    def test_nan_roundtrip(self):
        original = np.array([np.nan], dtype=np.float32)
        masks = extract_sub_masks(original)
        restored = repack_sub_masks(*masks)
        assert np.array_equal(original.view(np.uint32), restored.view(np.uint32))

    def test_sub_masks_are_uint8(self):
        vals = np.array([42.0], dtype=np.float32)
        for mask in extract_sub_masks(vals):
            assert mask.dtype == np.uint8

    def test_shape_preserved_1d(self):
        arr = np.random.randn(16).astype(np.float32)
        for mask in extract_sub_masks(arr):
            assert mask.shape == (16,)

    def test_shape_preserved_2d(self):
        matrix = np.random.randn(4, 8).astype(np.float32)
        for mask in extract_sub_masks(matrix):
            assert mask.shape == (4, 8)

    def test_shape_preserved_3d(self):
        tensor = np.random.randn(2, 3, 4).astype(np.float32)
        for mask in extract_sub_masks(tensor):
            assert mask.shape == (2, 3, 4)

    def test_masks_cover_all_32_bits(self):
        """Repacking the four planes must reconstruct every bit."""
        vals = np.array([0xDEADBEEF], dtype=np.uint32).view(np.float32)
        m0, m1, m2, m3 = extract_sub_masks(vals)
        assert m0[0] == 0xEF
        assert m1[0] == 0xBE
        assert m2[0] == 0xAD
        assert m3[0] == 0xDE


class TestRepackSubMasks:
    def test_known_value(self):
        m0 = np.array([0xEF], dtype=np.uint8)
        m1 = np.array([0xBE], dtype=np.uint8)
        m2 = np.array([0xAD], dtype=np.uint8)
        m3 = np.array([0xDE], dtype=np.uint8)
        result = repack_sub_masks(m0, m1, m2, m3)
        assert result.view(np.uint32)[0] == 0xDEADBEEF

    def test_zeros(self):
        z = np.zeros(5, dtype=np.uint8)
        result = repack_sub_masks(z, z, z, z)
        assert np.all(result == 0.0)

    def test_large_batch(self):
        original = np.random.randn(10000).astype(np.float32)
        masks = extract_sub_masks(original)
        restored = repack_sub_masks(*masks)
        assert np.array_equal(original.view(np.uint32), restored.view(np.uint32))
