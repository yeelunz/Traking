from __future__ import annotations

import numpy as np
import pytest

import tracking.classification.feature_extractors_v4_tsc  # noqa: F401
from tracking.classification.engine import _infer_texture_embedding_dim
from tracking.classification.feature_extractors.v6 import DepthScaleEstimate
from tracking.core.interfaces import FramePrediction, MaskStats, SegmentationData
from tracking.core.registry import FEATURE_EXTRACTOR_REGISTRY


def _build_extractor(params=None, name: str = "tsc_v4"):
    cls = FEATURE_EXTRACTOR_REGISTRY.get(name)
    assert cls is not None
    return cls(params or {})


def test_tsc_v4_registered_and_uses_distinct_video_keys():
    ext = _build_extractor({"texture_mode": "freeze"})
    video_keys = list(ext.feature_order("video"))
    assert video_keys
    assert video_keys[0].startswith("tsv4_")


def test_tsc_v4_extract_video_uses_scaled_samples(monkeypatch):
    ext = _build_extractor({"texture_mode": "freeze"})
    samples = [
        FramePrediction(
            frame_index=0,
            bbox=(10.0, 20.0, 30.0, 40.0),
            segmentation=SegmentationData(
                mask_path="mask.png",
                stats=MaskStats(
                    area_px=100.0,
                    bbox=(10.0, 20.0, 30.0, 40.0),
                    centroid=(25.0, 40.0),
                    perimeter_px=50.0,
                    equivalent_diameter_px=12.0,
                ),
            ),
        )
    ]
    seen = {}

    def fake_resolve(_video_path):
        return DepthScaleEstimate(px_per_cm=2.0, zero_depth_y_px=0.0, rule="unit_test")

    def fake_extract_ts_channels_v4(scaled_samples, video_path, frame_w, frame_h, interp_method, n_steps):
        del video_path, frame_w, frame_h, interp_method
        seen["bbox"] = scaled_samples[0].bbox
        seen["area"] = scaled_samples[0].segmentation.stats.area_px
        return np.zeros((15, n_steps), dtype=np.float32)

    def fake_texture_ts(_samples, _video_path, n_steps):
        return np.zeros((n_steps, ext._texture_dim), dtype=np.float32)

    monkeypatch.setattr(
        "tracking.classification.feature_extractors.v4_tsc.resolve_depth_scale_for_video",
        fake_resolve,
    )
    monkeypatch.setattr(
        "tracking.classification.feature_extractors.v4_tsc._extract_ts_channels_v4",
        fake_extract_ts_channels_v4,
    )
    monkeypatch.setattr(ext, "_extract_texture_ts", fake_texture_ts)

    out = ext.extract_video(samples, video_path="demo.avi")
    assert seen["bbox"] == (5.0, 10.0, 15.0, 20.0)
    assert seen["area"] == pytest.approx(25.0)
    assert out["_ts_n_vars"] == pytest.approx(15.0)
    assert "_raw_convnext_ts" in out


def test_engine_infers_texture_embedding_dim_for_tsc_v4():
    assert _infer_texture_embedding_dim("tsc_v4", {}, {}) == 5
    assert _infer_texture_embedding_dim("tsc_latest", {}, {}) == 5


def test_tsc_latest_alias_is_registered_to_v4():
    assert FEATURE_EXTRACTOR_REGISTRY.get("tsc_latest") is FEATURE_EXTRACTOR_REGISTRY.get("tsc_v4")
