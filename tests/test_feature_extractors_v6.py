from __future__ import annotations

import numpy as np
import pytest

import tracking.classification.feature_extractors_v6  # noqa: F401
from tracking.classification.engine import _infer_texture_embedding_dim
from tracking.classification.feature_extractors.v6 import (
    DepthScaleEstimate,
    normalise_video_path_key,
    precompute_depth_scale_lookup_for_videos,
    resolve_depth_scale_for_video,
    scale_predictions_to_cm,
)
from tracking.core.interfaces import FramePrediction, MaskStats, SegmentationData
from tracking.core.registry import FEATURE_EXTRACTOR_REGISTRY


def _build_extractor(params=None, name: str = "tab_v6"):
    cls = FEATURE_EXTRACTOR_REGISTRY.get(name)
    assert cls is not None
    return cls(params or {})


def test_resolve_depth_scale_for_japan_dataset_prefers_per_video_estimate(monkeypatch):
    resolve_depth_scale_for_video.cache_clear()

    monkeypatch.setattr(
        "tracking.classification.feature_extractors.v6._first_readable_frame",
        lambda _video_path: np.zeros((4, 4, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        "tracking.classification.feature_extractors.v6._estimate_depth_scale_without_rules_from_frame",
        lambda _frame: DepthScaleEstimate(px_per_cm=230.0, zero_depth_y_px=75.5, rule="merged_extend_c_wmv"),
    )

    scale = resolve_depth_scale_for_video(
        r"C:\demo\dataset\merged_extend_control_japan\mild42_right_59yo_femal\mild42_right_59yo_femal.wmv"
    )
    assert scale is not None
    assert scale.rule == "merged_extend_control_japan_auto"
    assert scale.px_per_cm == pytest.approx(230.0)
    assert scale.zero_depth_y_px == pytest.approx(75.5)


def test_resolve_depth_scale_for_japan_dataset_falls_back_when_estimation_fails(monkeypatch):
    resolve_depth_scale_for_video.cache_clear()

    monkeypatch.setattr(
        "tracking.classification.feature_extractors.v6._first_readable_frame",
        lambda _video_path: np.zeros((4, 4, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        "tracking.classification.feature_extractors.v6._estimate_depth_scale_without_rules_from_frame",
        lambda _frame: None,
    )

    scale = resolve_depth_scale_for_video(
        r"C:\demo\dataset\merged_extend_control_japan\control101_left_68yo_female\control101_left_68yo_female.wmv"
    )
    assert scale is not None
    assert scale.rule == "merged_extend_control_japan_fixed_fallback"
    assert scale.px_per_cm == pytest.approx(290.0)
    assert scale.zero_depth_y_px == pytest.approx(-0.5)


def test_scale_predictions_to_cm_scales_bbox_and_segmentation_stats():
    pred = FramePrediction(
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
    scale = DepthScaleEstimate(px_per_cm=2.0, zero_depth_y_px=0.0, rule="unit_test")
    scaled = scale_predictions_to_cm([pred], scale)[0]

    assert scaled.bbox == (5.0, 10.0, 15.0, 20.0)
    assert scaled.center == (12.5, 20.0)
    assert scaled.segmentation is not None
    assert scaled.segmentation.stats.area_px == 25.0
    assert scaled.segmentation.stats.perimeter_px == 25.0
    assert scaled.segmentation.stats.equivalent_diameter_px == 6.0
    assert scaled.segmentation.stats.centroid == (12.5, 20.0)


def test_tab_v6_extract_video_uses_scaled_samples(monkeypatch):
    ext = _build_extractor({"texture_mode": "freeze", "texture_pooling": "mean"}, name="tab_v6")
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

    def fake_motion(scaled_samples):
        seen["bbox"] = scaled_samples[0].bbox
        return {"motion": np.zeros((len(ext._motion_keys),), dtype=np.float32)}

    def fake_texture(_samples, _video_path):
        return {"tex": np.zeros((len(ext._texture_keys),), dtype=np.float32)}

    monkeypatch.setattr(
        "tracking.classification.feature_extractors.v6.resolve_depth_scale_for_video",
        fake_resolve,
    )
    monkeypatch.setattr(ext, "_extract_moment_motion_vector", fake_motion)
    monkeypatch.setattr(ext, "_extract_texture_vector", fake_texture)

    out = ext.extract_video(samples, video_path="demo.avi")
    assert seen["bbox"] == (5.0, 10.0, 15.0, 20.0)
    assert out["csa_mean"] == 25.0
    assert out["eq_diam_mean"] == 6.0
    assert out["aspect_ratio_mean"] == pytest.approx(0.75)


def test_tab_v6_lite_keeps_15_motion_15_texture_and_no_static_keys():
    ext = _build_extractor({"texture_mode": "freeze", "texture_pooling": "mean"}, name="tab_v6_lite")
    video_keys = list(ext.feature_order("video"))
    assert len(ext._motion_keys) == 15
    assert len(ext._texture_keys) == 15
    assert len(ext._static_keys) == 0
    assert len(video_keys) == 30


def test_engine_infers_texture_embedding_dim_for_tab_v6_variants():
    assert _infer_texture_embedding_dim("tab_v6", {}, {}) == 11
    assert _infer_texture_embedding_dim("tab_v6_lite", {}, {}) == 15


def test_precompute_depth_scale_lookup_for_videos_uses_no_rules_estimator(monkeypatch):
    seen = []

    def fake_estimator(video_path):
        seen.append(video_path)
        if "ok_video" in str(video_path):
            return DepthScaleEstimate(px_per_cm=250.0, zero_depth_y_px=12.5, rule="auto_test")
        return None

    monkeypatch.setattr(
        "tracking.classification.feature_extractors.v6.estimate_depth_scale_for_video_no_rules",
        fake_estimator,
    )

    lookup = precompute_depth_scale_lookup_for_videos(
        [
            r"C:\demo\ok_video.wmv",
            r"C:\demo\bad_video.wmv",
            r"C:\demo\ok_video.wmv",
        ]
    )
    assert len(seen) == 2

    ok_key = normalise_video_path_key(r"C:\demo\ok_video.wmv")
    bad_key = normalise_video_path_key(r"C:\demo\bad_video.wmv")

    assert ok_key in lookup
    assert bad_key not in lookup
    assert lookup[ok_key]["px_per_cm"] == pytest.approx(250.0)
    assert lookup[ok_key]["zero_depth_y_px"] == pytest.approx(12.5)
    assert lookup[ok_key]["rule"] == "auto_test"
