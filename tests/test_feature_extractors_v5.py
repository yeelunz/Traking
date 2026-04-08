from __future__ import annotations

import numpy as np

import tracking.classification.feature_extractors_v5  # noqa: F401
from tracking.core.interfaces import FramePrediction
from tracking.core.registry import FEATURE_EXTRACTOR_REGISTRY
from tracking.classification.feature_extractors.v4 import TAB_V4_STATIC_KEYS
from tracking.classification.feature_extractors.v5 import (
    _trajectory_to_moment_input_v5,
)


def _build_extractor(params=None, name: str = "tab_v5"):
    cls = FEATURE_EXTRACTOR_REGISTRY.get(name)
    assert cls is not None
    return cls(params or {})


def _zero_video_feature_dict(ext) -> dict:
    return {k: 0.0 for k in ext.feature_order("video")}


def test_moment_input_uses_step_displacements_with_zero_start():
    samples = [
        FramePrediction(frame_index=0, bbox=(100.0, 200.0, 10.0, 10.0)),
        FramePrediction(frame_index=1, bbox=(103.0, 201.0, 10.0, 10.0)),
        FramePrediction(frame_index=2, bbox=(104.0, 198.0, 10.0, 10.0)),
        FramePrediction(frame_index=3, bbox=(98.0, 196.0, 10.0, 10.0)),
    ]

    out = _trajectory_to_moment_input_v5(samples, target_steps=4)
    assert out.shape == (2, 4)
    assert np.allclose(out[0], np.array([0.0, 3.0, 1.0, -6.0], dtype=np.float32))
    assert np.allclose(out[1], np.array([0.0, 1.0, -3.0, -2.0], dtype=np.float32))


def test_tab_v5_finalize_batch_fits_motion_pca_and_reuses_it():
    ext = _build_extractor({"texture_mode": "freeze", "texture_pooling": "mean"})

    feat_a = _zero_video_feature_dict(ext)
    feat_a["_raw_moment_embedding"] = np.linspace(0.0, 1.0, 24, dtype=np.float64)
    feat_a["_raw_texture_backbone_batch"] = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    feat_a["_raw_texture_weights"] = np.array([1.0], dtype=np.float64)

    feat_b = _zero_video_feature_dict(ext)
    feat_b["_raw_moment_embedding"] = np.linspace(1.0, 2.0, 24, dtype=np.float64)
    feat_b["_raw_texture_backbone_batch"] = np.array([[3.0, 2.0, 1.0]], dtype=np.float64)
    feat_b["_raw_texture_weights"] = np.array([10.0], dtype=np.float64)

    fitted = ext.finalize_batch([feat_a, feat_b], fit=True)
    assert len(fitted) == 2
    assert ext._motion_pca_mean is not None
    assert ext._motion_pca_components is not None
    assert any(abs(fitted[0][k]) > 0.0 for k in ext._motion_keys)

    feat_c = _zero_video_feature_dict(ext)
    feat_c["_raw_moment_embedding"] = np.linspace(0.5, 1.5, 24, dtype=np.float64)
    feat_c["_raw_texture_backbone_batch"] = np.array([[2.0, 2.0, 2.0]], dtype=np.float64)
    feat_c["_raw_texture_weights"] = np.array([5.0], dtype=np.float64)

    transformed = ext.finalize_batch([feat_c], fit=False)
    assert len(transformed) == 1
    assert all(np.isfinite(transformed[0][k]) for k in ext.feature_order("video"))


def test_tab_v5_lite_keeps_15_motion_15_texture_and_no_static_keys():
    ext = _build_extractor({"texture_mode": "freeze", "texture_pooling": "mean"}, name="tab_v5_lite")

    video_keys = list(ext.feature_order("video"))
    assert len(ext._motion_keys) == 15
    assert len(ext._texture_keys) == 15
    assert len(ext._static_keys) == 0
    assert len(video_keys) == 30
    assert video_keys == ext._motion_keys + ext._texture_keys

    for k in TAB_V4_STATIC_KEYS:
        assert k not in video_keys
