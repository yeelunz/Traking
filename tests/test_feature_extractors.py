"""Tests for motion_only and motion_texture feature extractors."""
import os

import cv2
import numpy as np
import pytest

from tracking.core.interfaces import FramePrediction, SegmentationData, MaskStats
from tracking.core.registry import FEATURE_EXTRACTOR_REGISTRY
import tracking.classification.feature_extractors  # noqa: F401  - trigger registration


# ── helpers ──────────────────────────────────────────────────────────

def _make_temp_video(path: str, frame_count: int = 8) -> str:
    height, width = 72, 96
    video_path = os.path.join(path, "temp_video.avi")
    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"XVID"),
        15.0,
        (width, height),
        True,
    )
    if not writer.isOpened():
        raise RuntimeError("Unable to create temporary test video")
    for i in range(frame_count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.rectangle(
            frame,
            (10 + i, 15),
            (10 + i + 30, 15 + 20),
            (20 * i % 255, 120, 255 - 15 * i % 255),
            -1,
        )
        writer.write(frame)
    writer.release()
    return video_path


def _dummy_predictions(n: int = 5) -> list[FramePrediction]:
    preds = []
    for i in range(n):
        bbox = (10.0 + i * 2, 12.0 + i, 30.0, 22.0)
        seg_stats = MaskStats(
            area_px=float(200 + i * 10),
            bbox=bbox,
            centroid=(10.0 + i * 2 + 15.0, 12.0 + i + 11.0),
            perimeter_px=float(60 + i * 2),
            equivalent_diameter_px=float(16 + i),
        )
        seg = SegmentationData(mask_path=None, stats=seg_stats)
        preds.append(FramePrediction(frame_index=i * 3, bbox=bbox, segmentation=seg))
    return preds


def _build_extractor(name: str, params=None):
    cls = FEATURE_EXTRACTOR_REGISTRY.get(name)
    assert cls is not None, f"Feature extractor '{name}' not registered"
    return cls(params or {})


# ── motion_only ──────────────────────────────────────────────────────

class TestMotionOnly:
    def test_registered(self):
        assert "motion_only" in FEATURE_EXTRACTOR_REGISTRY

    def test_video_feature_keys(self):
        ext = _build_extractor("motion_only")
        samples = _dummy_predictions()
        feats = ext.extract_video(samples)
        video_keys = ext.feature_order("video")
        assert set(video_keys) == set(feats.keys()), "Keys mismatch"
        assert all(np.isfinite(v) for v in feats.values())

    def test_empty_samples(self):
        ext = _build_extractor("motion_only")
        feats = ext.extract_video([])
        assert all(v == 0.0 for v in feats.values())

    def test_single_sample(self):
        ext = _build_extractor("motion_only")
        feats = ext.extract_video(_dummy_predictions(1))
        assert feats["num_points"] == 1.0
        assert feats["path_length"] == 0.0

    def test_subject_aggregation(self):
        ext = _build_extractor("motion_only")
        v1 = ext.extract_video(_dummy_predictions(4))
        v2 = ext.extract_video(_dummy_predictions(6))
        agg = ext.aggregate_subject([v1, v2])
        subject_keys = ext.feature_order("subject")
        assert set(subject_keys) == set(agg.keys())
        assert agg["video_count"] == pytest.approx(2.0)
        assert all(np.isfinite(v) for v in agg.values())

    def test_kalman_smoother_preserves_endpoints(self):
        """Kalman smoothing should not wildly deviate from observations."""
        from tracking.classification.feature_extractors import _KalmanSmoother2D

        obs = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], dtype=np.float64)
        frames = np.array([0, 1, 2, 3, 4], dtype=np.float64)
        smoother = _KalmanSmoother2D()
        smoothed = smoother.smooth(obs, frames)
        # Smoothed should be close to the straight line
        assert np.allclose(smoothed, obs, atol=1.0)


# ── motion_texture ───────────────────────────────────────────────────

class TestMotionTexture:
    def test_registered(self):
        assert "motion_texture" in FEATURE_EXTRACTOR_REGISTRY

    def test_video_feature_keys_with_raw(self, tmp_path):
        ext = _build_extractor("motion_texture", {"texture_patch_size": 32})
        video_path = _make_temp_video(str(tmp_path))
        samples = _dummy_predictions()
        feats = ext.extract_video(samples, video_path=video_path)
        # Before finalize_batch, should contain _raw_texture_vec
        assert "_raw_texture_vec" in feats

    def test_finalize_batch_removes_raw(self, tmp_path):
        ext = _build_extractor("motion_texture", {"texture_patch_size": 32})
        video_path = _make_temp_video(str(tmp_path))
        s1 = _dummy_predictions(4)
        s2 = _dummy_predictions(6)
        f1 = ext.extract_video(s1, video_path=video_path)
        f2 = ext.extract_video(s2, video_path=video_path)
        finalized = ext.finalize_batch([f1, f2], fit=True)
        for feat in finalized:
            assert "_raw_texture_vec" not in feat
            video_keys = ext.feature_order("video")
            assert set(video_keys) == set(feat.keys())
            assert all(np.isfinite(v) for v in feat.values())

    def test_pca_reuse(self, tmp_path):
        """After fit=True, fit=False should use stored PCA."""
        ext = _build_extractor("motion_texture", {"texture_patch_size": 32})
        video_path = _make_temp_video(str(tmp_path))
        samples = _dummy_predictions(5)
        train_feats = [ext.extract_video(samples, video_path=video_path) for _ in range(3)]
        _ = ext.finalize_batch(train_feats, fit=True)
        # PCA should be stored
        assert ext._pca_mean is not None
        # Test set
        test_feats = [ext.extract_video(samples, video_path=video_path)]
        test_out = ext.finalize_batch(test_feats, fit=False)
        assert "_raw_texture_vec" not in test_out[0]

    def test_subject_aggregation(self, tmp_path):
        ext = _build_extractor("motion_texture", {"texture_patch_size": 32})
        video_path = _make_temp_video(str(tmp_path))
        f1 = ext.extract_video(_dummy_predictions(4), video_path=video_path)
        f2 = ext.extract_video(_dummy_predictions(5), video_path=video_path)
        finalized = ext.finalize_batch([f1, f2], fit=True)
        agg = ext.aggregate_subject(list(finalized))
        subject_keys = ext.feature_order("subject")
        assert set(subject_keys) == set(agg.keys())
        assert agg["video_count"] == pytest.approx(2.0)

    def test_no_video_path_texture_zeros(self):
        ext = _build_extractor("motion_texture")
        feats = ext.extract_video(_dummy_predictions(3), video_path=None)
        raw_vec = feats.get("_raw_texture_vec", [])
        assert all(v == 0.0 for v in raw_vec)

    def test_csa_features_present(self, tmp_path):
        ext = _build_extractor("motion_texture")
        video_path = _make_temp_video(str(tmp_path))
        feats = ext.extract_video(_dummy_predictions(5), video_path=video_path)
        assert feats.get("csa_first_area", 0.0) > 0
        assert feats.get("csa_last_area", 0.0) > 0
