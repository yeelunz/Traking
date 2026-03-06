"""
tests/test_dataset_imread_guard.py

Regression test for the cv2.imread guard in SegmentationCropDataset._load_mask.

Ultralytics patches cv2.imread to use np.fromfile + cv2.imdecode.
If a mask file exists but is zero-byte (or corrupted), the patched
version raises:
    cv2.error: (-215:Assertion failed) !buf.empty() in function 'cv2::imdecode_'

The fix in dataset.py adds:
  1. os.path.isfile()     – non-existent path returns None silently
  2. os.path.getsize()==0 – zero-byte file returns None silently
  3. try/except           – corrupt file returns None silently
"""
from __future__ import annotations

import os
import numpy as np
import pytest

from tracking.segmentation.dataset import SegmentationCropDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dataset(tmp_path) -> SegmentationCropDataset:
    """Return a bare SegmentationCropDataset with no entries (just enough to call _load_mask)."""
    ds = SegmentationCropDataset.__new__(SegmentationCropDataset)
    ds.dataset_root = str(tmp_path)
    ds.entries = []
    ds.empty_masks = []
    ds._mask_valid_cache = {}
    return ds


def _make_valid_mask_file(tmp_path) -> str:
    """Write a tiny valid 10×10 grayscale PNG and return its path."""
    import cv2
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[3:7, 3:7] = 255  # filled square so keep_largest_component passes
    path = str(tmp_path / "valid_mask.png")
    cv2.imwrite(path, mask)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDatasetLoadMaskGuard:
    def test_returns_none_for_nonexistent_path(self, tmp_path):
        ds = _make_dataset(tmp_path)
        result = ds._load_mask("video.mp4", "does_not_exist.png")
        assert result is None

    def test_returns_none_for_zero_byte_file(self, tmp_path):
        zero_file = tmp_path / "empty.png"
        zero_file.write_bytes(b"")
        ds = _make_dataset(tmp_path)
        # Pass absolute path so the root-join branch is skipped
        result = ds._load_mask("video.mp4", str(zero_file))
        assert result is None

    def test_returns_none_for_corrupt_file(self, tmp_path):
        bad_file = tmp_path / "corrupt.png"
        bad_file.write_bytes(b"\x00\x01\x02\x03garbage")
        ds = _make_dataset(tmp_path)
        result = ds._load_mask("video.mp4", str(bad_file))
        assert result is None

    def test_returns_none_for_none_mask_path(self, tmp_path):
        ds = _make_dataset(tmp_path)
        result = ds._load_mask("video.mp4", None)
        assert result is None

    def test_returns_none_for_empty_mask_path(self, tmp_path):
        ds = _make_dataset(tmp_path)
        result = ds._load_mask("video.mp4", "")
        assert result is None

    def test_returns_array_for_valid_mask(self, tmp_path):
        path = _make_valid_mask_file(tmp_path)
        ds = _make_dataset(tmp_path)
        result = ds._load_mask("video.mp4", path)
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
