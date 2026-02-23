"""
tests/test_workflow_imread_guard.py

Regression test for the cv2.imread guard in SegmentationWorkflow.

Ultralytics patches cv2.imread to use np.fromfile + cv2.imdecode.
If a mask file exists but is zero-byte (or corrupted), the patched
version raises:
    cv2.error: (-215:Assertion failed) !buf.empty() in function 'cv2::imdecode_'

The fix adds:
  1. An os.path.getsize() == 0 guard before calling cv2.imread.
  2. A try/except around the imread call to catch any cv2.error.

Both fixes apply to:
  - SegmentationWorkflow._load_mask_from_annotation   (line ~1697)
  - The predicted-mask loader used for visualisation  (line ~1569)
"""
from __future__ import annotations

import os
import numpy as np
import pytest

from tracking.segmentation.workflow import SegmentationWorkflow
from tracking.core.interfaces import SegmentationData, MaskStats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_workflow(tmp_path) -> SegmentationWorkflow:
    """Minimal SegmentationWorkflow – no model needed for file-guard tests."""
    cfg = {
        "model": {"name": "unetpp", "params": {}},
        "train": False,
        "device": "cpu",
        "threshold": 0.5,
    }

    import torch.nn as nn

    class _DummyModel(nn.Module):
        def __init__(self, params=None):  # type: ignore[override]
            super().__init__()

        def forward(self, x):  # type: ignore[override]
            import torch
            b, _, h, w = x.shape
            return torch.zeros((b, 1, h, w))

    from tracking.segmentation.workflow import SEGMENTATION_MODEL_REGISTRY

    original = SEGMENTATION_MODEL_REGISTRY.get("unetpp")
    SEGMENTATION_MODEL_REGISTRY["unetpp"] = _DummyModel  # type: ignore[assignment]

    wf = SegmentationWorkflow(
        config=cfg,
        dataset_root=str(tmp_path),
        results_dir=str(tmp_path / "results"),
        logger=lambda msg: None,
    )
    # Restore
    if original is not None:
        SEGMENTATION_MODEL_REGISTRY["unetpp"] = original
    else:
        del SEGMENTATION_MODEL_REGISTRY["unetpp"]

    return wf


def _make_seg_data(mask_path: str) -> SegmentationData:
    stats = MaskStats(
        area_px=0.0,
        bbox=(0.0, 0.0, 1.0, 1.0),
        centroid=(0.0, 0.0),
        perimeter_px=0.0,
        equivalent_diameter_px=0.0,
    )
    return SegmentationData(mask_path=mask_path, stats=stats)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadMaskFromAnnotationGuard:
    """_load_mask_from_annotation must NOT raise on bad mask files."""

    def test_returns_none_for_nonexistent_file(self, tmp_path):
        wf = _make_workflow(tmp_path)
        seg = _make_seg_data(str(tmp_path / "ghost.png"))
        result = wf._load_mask_from_annotation("video.avi", seg)
        assert result is None

    def test_returns_none_for_zero_byte_file(self, tmp_path):
        """Core regression: zero-byte mask must return None without raising."""
        empty_mask = tmp_path / "empty_mask.png"
        empty_mask.write_bytes(b"")  # zero-byte file

        wf = _make_workflow(tmp_path)
        seg = _make_seg_data(str(empty_mask))
        # Must not raise cv2.error / assertion failure
        result = wf._load_mask_from_annotation("video.avi", seg)
        assert result is None

    def test_returns_none_for_corrupt_file(self, tmp_path):
        """Corrupt (non-image) bytes must return None without raising."""
        bad_mask = tmp_path / "corrupt_mask.png"
        bad_mask.write_bytes(b"not a valid image \xff\xfe\x00")

        wf = _make_workflow(tmp_path)
        seg = _make_seg_data(str(bad_mask))
        result = wf._load_mask_from_annotation("video.avi", seg)
        assert result is None

    def test_returns_mask_for_valid_file(self, tmp_path):
        """A real binary mask PNG must be loaded correctly."""
        import cv2
        mask_arr = np.zeros((64, 64), dtype=np.uint8)
        mask_arr[16:48, 16:48] = 255
        mask_path = str(tmp_path / "valid_mask.png")
        cv2.imwrite(mask_path, mask_arr)

        wf = _make_workflow(tmp_path)
        seg = _make_seg_data(mask_path)
        result = wf._load_mask_from_annotation("video.avi", seg)
        assert result is not None
        # Shape may be (H, W) or (H, W, 1) depending on post-processing chain
        assert result.shape[:2] == (64, 64)
        assert result.max() > 0

    def test_segmentation_data_none_returns_none(self, tmp_path):
        wf = _make_workflow(tmp_path)
        result = wf._load_mask_from_annotation("video.avi", None)  # type: ignore[arg-type]
        assert result is None
