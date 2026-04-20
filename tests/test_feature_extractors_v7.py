from __future__ import annotations

import numpy as np
import pytest

import tracking.classification.feature_extractors_v4_tsc  # noqa: F401
import tracking.classification.feature_extractors_v7  # noqa: F401
from tracking.classification.engine import _infer_texture_embedding_dim
from tracking.classification.feature_extractors.v6 import DepthScaleEstimate, normalise_video_path_key
from tracking.classification.feature_extractors.v7 import (
    TAB_V7_GEOMETRY_KEYS,
    TAB_V7_MOTION_KEYS,
    TAB_V7_TEXTURE_KEYS,
)
from tracking.core.interfaces import FramePrediction, MaskStats, SegmentationData
from tracking.core.registry import FEATURE_EXTRACTOR_REGISTRY


def _build_extractor(params=None, name: str = "tab_v7"):
    cls = FEATURE_EXTRACTOR_REGISTRY.get(name)
    assert cls is not None
    return cls(params or {})


def test_tab_v7_has_24_dims_in_geometry_depth_motion_order():
    ext = _build_extractor({"texture_mode": "freeze", "texture_pooling": "mean"}, name="tab_v7")
    video_keys = list(ext.feature_order("video"))
    assert len(video_keys) == 24
    assert video_keys[:8] == TAB_V7_GEOMETRY_KEYS
    assert video_keys[8:16] == TAB_V7_TEXTURE_KEYS
    assert video_keys[16:] == TAB_V7_MOTION_KEYS


def test_tab_v7_extract_video_uses_eq_diam_std_and_no_v4_strain_fields(monkeypatch):
    ext = _build_extractor({"texture_mode": "freeze", "texture_pooling": "mean"}, name="tab_v7")
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

    def fake_resolve(_video_path):
        return DepthScaleEstimate(px_per_cm=2.0, zero_depth_y_px=0.0, rule="unit_test")

    def fake_motion(_samples, *, displacement_scale=1.0):
        assert displacement_scale == pytest.approx(0.5)
        return {"motion": np.zeros((len(ext._motion_keys),), dtype=np.float32)}

    def fake_texture(_samples, _video_path):
        return {"tex": np.zeros((len(ext._texture_keys),), dtype=np.float32)}

    monkeypatch.setattr(
        "tracking.classification.feature_extractors.v7.resolve_depth_scale_for_video",
        fake_resolve,
    )
    monkeypatch.setattr(ext, "_extract_moment_motion_vector", fake_motion)
    monkeypatch.setattr(ext, "_extract_texture_vector", fake_texture)

    out = ext.extract_video(samples, video_path="demo.avi")
    assert out["csa_mean"] == pytest.approx(25.0)
    assert out["eq_diam_mean"] == pytest.approx(6.0)
    assert out["eq_diam_std"] == pytest.approx(0.0)
    assert "eq_diam_strain" not in out
    assert "csa_strain" not in out


def test_tab_v7_variant_dimensions():
    expected = {
        "tab_v7_GD": 16,
        "tab_v7_GM": 16,
        "tab_v7_DM": 16,
        "tab_v7_G": 8,
        "tab_v7_D": 8,
        "tab_v7_M": 8,
    }
    for name, dim in expected.items():
        ext = _build_extractor({"texture_mode": "freeze", "texture_pooling": "mean"}, name=name)
        assert len(list(ext.feature_order("video"))) == dim


def test_tab_v7_motion_scale_applies_at_displacement_stage(monkeypatch):
    video_path = r"C:\demo\dataset\extendclen\c1\demo.wmv"
    ext = _build_extractor(
        {
            "texture_mode": "freeze",
            "texture_pooling": "mean",
            "depth_scale_lookup": {
                normalise_video_path_key(video_path): {
                    "px_per_cm": 4.0,
                    "zero_depth_y_px": 0.0,
                    "rule": "precomputed_test",
                }
            },
            "depth_scale_lookup_strict": True,
        },
        name="tab_v7",
    )
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

    def fake_motion(_samples, *, displacement_scale=1.0):
        seen["displacement_scale"] = float(displacement_scale)
        return {"motion": np.zeros((len(ext._motion_keys),), dtype=np.float32)}

    monkeypatch.setattr(ext, "_extract_moment_motion_vector", fake_motion)
    monkeypatch.setattr(
        ext,
        "_extract_texture_vector",
        lambda _samples, _video_path: {"tex": np.zeros((len(ext._texture_keys),), dtype=np.float32)},
    )

    ext.extract_video(samples, video_path=video_path)
    assert seen["displacement_scale"] == pytest.approx(0.25)


def test_tab_v7_uses_precomputed_scale_lookup_in_strict_mode(monkeypatch):
    video_path = r"C:\demo\dataset\extendclen\c1\demo.wmv"
    ext = _build_extractor(
        {
            "texture_mode": "freeze",
            "texture_pooling": "mean",
            "depth_scale_lookup": {
                normalise_video_path_key(video_path): {
                    "px_per_cm": 2.0,
                    "zero_depth_y_px": 0.0,
                    "rule": "precomputed_test",
                }
            },
            "depth_scale_lookup_strict": True,
        },
        name="tab_v7",
    )
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

    monkeypatch.setattr(
        "tracking.classification.feature_extractors.v7.resolve_depth_scale_for_video",
        lambda _video_path: (_ for _ in ()).throw(AssertionError("must not fallback to rule resolver")),
    )
    monkeypatch.setattr(
        ext,
        "_extract_moment_motion_vector",
        lambda _samples, **_kwargs: {"motion": np.zeros((len(ext._motion_keys),), dtype=np.float32)},
    )
    monkeypatch.setattr(
        ext,
        "_extract_texture_vector",
        lambda _samples, _video_path: {"tex": np.zeros((len(ext._texture_keys),), dtype=np.float32)},
    )

    out = ext.extract_video(samples, video_path=video_path)
    assert out["csa_mean"] == pytest.approx(25.0)
    assert out["eq_diam_mean"] == pytest.approx(6.0)


def test_tab_v7_strict_mode_keeps_unscaled_when_lookup_misses(monkeypatch):
    ext = _build_extractor(
        {
            "texture_mode": "freeze",
            "texture_pooling": "mean",
            "depth_scale_lookup": {},
            "depth_scale_lookup_strict": True,
        },
        name="tab_v7",
    )
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

    monkeypatch.setattr(
        "tracking.classification.feature_extractors.v7.resolve_depth_scale_for_video",
        lambda _video_path: (_ for _ in ()).throw(AssertionError("strict mode must not call fallback resolver")),
    )
    monkeypatch.setattr(
        ext,
        "_extract_moment_motion_vector",
        lambda _samples, **_kwargs: {"motion": np.zeros((len(ext._motion_keys),), dtype=np.float32)},
    )
    monkeypatch.setattr(
        ext,
        "_extract_texture_vector",
        lambda _samples, _video_path: {"tex": np.zeros((len(ext._texture_keys),), dtype=np.float32)},
    )

    out = ext.extract_video(samples, video_path=r"C:\missing\video.wmv")
    assert out["csa_mean"] == pytest.approx(100.0)
    assert out["eq_diam_mean"] == pytest.approx(12.0)


def test_latest_aliases_point_to_current_extractors_and_dims():
    assert FEATURE_EXTRACTOR_REGISTRY.get("tab_latest") is FEATURE_EXTRACTOR_REGISTRY.get("tab_v7")
    assert FEATURE_EXTRACTOR_REGISTRY.get("tsc_latest") is FEATURE_EXTRACTOR_REGISTRY.get("tsc_v4")

    assert _infer_texture_embedding_dim("tab_latest", {}, {}) == 8
    assert _infer_texture_embedding_dim("tsc_latest", {}, {}) == 5
    assert _infer_texture_embedding_dim("tab_v7_GM", {}, {}) == 0
