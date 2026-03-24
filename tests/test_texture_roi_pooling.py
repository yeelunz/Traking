import numpy as np

from tracking.core.registry import FEATURE_EXTRACTOR_REGISTRY
import tracking.classification.feature_extractors  # noqa: F401
from tracking.core.interfaces import FramePrediction, SegmentationData, MaskStats
from tracking.classification.feature_extractors.v3pro import _resolve_texture_roi_bbox


def _build_extractor(name: str, params=None):
    cls = FEATURE_EXTRACTOR_REGISTRY.get(name)
    assert cls is not None, f"Feature extractor '{name}' not registered"
    return cls(params or {})


def _zero_video_feature_dict(ext) -> dict:
    return {k: 0.0 for k in ext.feature_order("video")}


def test_tab_v3_pro_freeze_uses_score_weighted_roi_pooling():
    ext = _build_extractor(
        "tab_v3_pro",
        {
            "texture_mode": "freeze",
            "texture_dim": 2,
            "texture_pooling": "score_weighted",
        },
    )
    feat = _zero_video_feature_dict(ext)
    feat["_raw_texture_backbone_batch"] = np.array(
        [
            [1.0, 0.0],
            [0.0, 2.0],
        ],
        dtype=np.float64,
    )
    feat["_raw_texture_weights"] = np.array([1.0, 3.0], dtype=np.float64)

    out = ext.finalize_batch([feat], fit=True)[0]
    assert out["v3pro_tex_00"] == np.float64(0.25)
    assert out["v3pro_tex_01"] == np.float64(1.5)


def test_tab_v4_freeze_uses_mean_roi_pooling():
    ext = _build_extractor(
        "tab_v4",
        {
            "texture_mode": "freeze",
            "texture_pooling": "mean",
        },
    )
    feat = _zero_video_feature_dict(ext)
    feat["_raw_texture_backbone_batch"] = np.array(
        [
            [2.0, 0.0, 1.0],
            [0.0, 4.0, 3.0],
        ],
        dtype=np.float64,
    )
    feat["_raw_texture_weights"] = np.array([1.0, 100.0], dtype=np.float64)

    out = ext.finalize_batch([feat], fit=True)[0]
    # mean pooling must ignore score weights under "mean" mode
    assert out["cnn_features_00"] == np.float64(1.0)
    assert out["cnn_features_01"] == np.float64(2.0)
    assert out["cnn_features_02"] == np.float64(2.0)


def test_tab_v3_pro_weight_source_confidence_controls_weighting():
    ext = _build_extractor(
        "tab_v3_pro",
        {
            "texture_mode": "freeze",
            "texture_dim": 2,
            "texture_pooling": "score_weighted",
            "texture_pooling_weight_source": "confidence",
        },
    )
    feat = _zero_video_feature_dict(ext)
    feat["_raw_texture_backbone_batch"] = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    # Explicitly provide confidence-based weights to mimic extracted per-frame confidence.
    feat["_raw_texture_weights"] = np.array([0.9, 0.1], dtype=np.float64)

    out = ext.finalize_batch([feat], fit=True)[0]
    assert out["v3pro_tex_00"] == np.float64(0.9)
    assert out["v3pro_tex_01"] == np.float64(0.1)


def test_texture_roi_prefers_segmentation_roi_bbox():
    seg = SegmentationData(
        mask_path=None,
        stats=MaskStats(
            area_px=1.0,
            bbox=(1.0, 2.0, 3.0, 4.0),
            centroid=(0.0, 0.0),
            perimeter_px=1.0,
            equivalent_diameter_px=1.0,
        ),
        roi_bbox=(10.0, 20.0, 30.0, 40.0),
    )
    sample = FramePrediction(
        frame_index=0,
        bbox=(1.0, 2.0, 3.0, 4.0),
        segmentation=seg,
    )

    bbox, pad_ratio = _resolve_texture_roi_bbox(sample, 0.15)
    assert bbox == (10.0, 20.0, 30.0, 40.0)
    assert pad_ratio == 0.0
