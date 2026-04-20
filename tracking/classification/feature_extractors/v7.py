from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ...core.interfaces import FramePrediction
from ...core.registry import register_feature_extractor
from .v4 import _circularity_v4, _safe_std_v4
from .v5 import MotionStaticV5FeatureExtractor
from .v6 import (
    lookup_depth_scale_from_table,
    resolve_depth_scale_for_video,
    scale_predictions_to_cm,
)


TAB_V7_GEOMETRY_KEYS: List[str] = [
    "csa_mean",
    "csa_std",
    "eq_diam_mean",
    "eq_diam_std",
    "circularity_mean",
    "circularity_std",
    "aspect_ratio_mean",
    "aspect_ratio_std",
]
TAB_V7_TEXTURE_KEYS: List[str] = [f"cnn_features_{i:02d}" for i in range(8)]
TAB_V7_MOTION_KEYS: List[str] = [f"moment_motion_{i:02d}" for i in range(8)]
TAB_V7_TOTAL_DIM: int = len(TAB_V7_GEOMETRY_KEYS) + len(TAB_V7_TEXTURE_KEYS) + len(TAB_V7_MOTION_KEYS)


def _compute_geometry_tab_v7(samples: Sequence[FramePrediction]) -> Dict[str, float]:
    zeros = OrderedDict((k, 0.0) for k in TAB_V7_GEOMETRY_KEYS)
    if not samples:
        return zeros

    usable = [
        s
        for s in samples
        if s.segmentation and s.segmentation.stats and getattr(s.segmentation.stats, "area_px", 0.0) > 0
    ]

    feat: Dict[str, float] = OrderedDict()
    if usable:
        areas = np.array([s.segmentation.stats.area_px for s in usable], dtype=np.float64)
        eq_diams = np.array(
            [s.segmentation.stats.equivalent_diameter_px for s in usable],
            dtype=np.float64,
        )
        circs = np.array(
            [
                _circularity_v4(
                    s.segmentation.stats.area_px,
                    s.segmentation.stats.perimeter_px,
                )
                for s in usable
            ],
            dtype=np.float64,
        )
    else:
        areas = np.zeros(0, dtype=np.float64)
        eq_diams = np.zeros(0, dtype=np.float64)
        circs = np.zeros(0, dtype=np.float64)

    if areas.size:
        feat["csa_mean"] = float(areas.mean())
        feat["csa_std"] = _safe_std_v4(areas)
    else:
        feat["csa_mean"] = 0.0
        feat["csa_std"] = 0.0

    if eq_diams.size:
        feat["eq_diam_mean"] = float(eq_diams.mean())
        feat["eq_diam_std"] = _safe_std_v4(eq_diams)
    else:
        feat["eq_diam_mean"] = 0.0
        feat["eq_diam_std"] = 0.0

    if circs.size:
        feat["circularity_mean"] = float(circs.mean())
        feat["circularity_std"] = _safe_std_v4(circs)
    else:
        feat["circularity_mean"] = 0.0
        feat["circularity_std"] = 0.0

    bboxes = np.array([s.bbox for s in samples], dtype=np.float64)
    if bboxes.size > 0:
        aspects = bboxes[:, 2] / (bboxes[:, 3] + 1e-9)
        feat["aspect_ratio_mean"] = float(aspects.mean())
        feat["aspect_ratio_std"] = _safe_std_v4(aspects)
    else:
        feat["aspect_ratio_mean"] = 0.0
        feat["aspect_ratio_std"] = 0.0

    for key in TAB_V7_GEOMETRY_KEYS:
        value = float(feat.get(key, 0.0))
        feat[key] = value if np.isfinite(value) else 0.0
    return feat


class _MotionStaticV7Base(MotionStaticV5FeatureExtractor):
    INCLUDE_GEOMETRY: bool = True
    INCLUDE_TEXTURE: bool = True
    INCLUDE_MOTION: bool = True

    DEFAULT_CONFIG: Dict[str, Any] = {
        **MotionStaticV5FeatureExtractor.DEFAULT_CONFIG,
        "moment_pca_dim": len(TAB_V7_MOTION_KEYS),
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params=params)

        raw_cfg = dict(params or {})
        raw_lookup = raw_cfg.get("depth_scale_lookup")
        self._depth_scale_lookup: Dict[str, Any] = dict(raw_lookup) if isinstance(raw_lookup, dict) else {}
        self._depth_scale_lookup_strict = bool(
            raw_cfg.get("depth_scale_lookup_strict", bool(self._depth_scale_lookup))
        )

        self._moment_pca_dim = len(TAB_V7_MOTION_KEYS)

        self._motion_keys = list(TAB_V7_MOTION_KEYS) if self.INCLUDE_MOTION else []
        self._static_keys = list(TAB_V7_GEOMETRY_KEYS) if self.INCLUDE_GEOMETRY else []

        self._texture_dim = len(TAB_V7_TEXTURE_KEYS) if self.INCLUDE_TEXTURE else 0
        self._texture_keys = list(TAB_V7_TEXTURE_KEYS[: self._texture_dim])

        self._non_deep_dim = len(self._motion_keys) + len(self._static_keys)
        self._video_keys = self._motion_keys + self._static_keys + self._texture_keys
        self._subject_keys = list(self._video_keys)

        self._public_video_keys = self._static_keys + self._texture_keys + self._motion_keys
        self._public_subject_keys = list(self._public_video_keys)

    def _scaled_samples_for_video(
        self,
        samples: Sequence[FramePrediction],
        video_path: Optional[str],
    ) -> List[FramePrediction]:
        depth_scale = self._resolve_depth_scale(video_path)
        return scale_predictions_to_cm(samples, depth_scale)

    def _resolve_depth_scale(self, video_path: Optional[str]) -> Any:
        depth_scale = lookup_depth_scale_from_table(video_path, self._depth_scale_lookup)
        if depth_scale is None and not self._depth_scale_lookup_strict:
            depth_scale = resolve_depth_scale_for_video(video_path)
        return depth_scale

    def feature_order(self, level: str = "video") -> Sequence[str]:
        return self._public_subject_keys if str(level).lower() == "subject" else self._public_video_keys

    def extract_video(
        self,
        samples: Sequence[FramePrediction],
        video_path: Optional[str] = None,
    ) -> Dict[str, float]:
        depth_scale = self._resolve_depth_scale(video_path)
        scaled_samples = scale_predictions_to_cm(samples, depth_scale)
        displacement_scale = float(depth_scale.cm_per_px) if depth_scale is not None else 1.0

        motion_info = (
            self._extract_moment_motion_vector(samples, displacement_scale=displacement_scale)
            if self._motion_keys
            else {"motion": np.zeros((0,), dtype=np.float32)}
        )
        geometry = _compute_geometry_tab_v7(scaled_samples) if self._static_keys else {}
        tex_info = (
            self._extract_texture_vector(samples, video_path)
            if self._texture_keys
            else {"tex": np.zeros((0,), dtype=np.float32)}
        )

        motion_vec = np.asarray(motion_info.get("motion"), dtype=np.float32)
        tex_vec = np.asarray(tex_info.get("tex"), dtype=np.float32)

        out: Dict[str, float] = OrderedDict()
        for key in self._static_keys:
            out[key] = float(geometry.get(key, 0.0))
        for i, key in enumerate(self._texture_keys):
            out[key] = float(tex_vec[i]) if i < tex_vec.size else 0.0
        for i, key in enumerate(self._motion_keys):
            out[key] = float(motion_vec[i]) if i < motion_vec.size else 0.0

        raw_motion = motion_info.get("_raw_moment_embedding")
        if raw_motion is not None:
            out["_raw_moment_embedding"] = np.asarray(raw_motion, dtype=np.float64)

        raw_batch = tex_info.get("_raw_texture_backbone_batch")
        if raw_batch is not None:
            out["_raw_texture_backbone_batch"] = np.asarray(raw_batch, dtype=np.float64)
            out["_raw_texture_weights"] = np.asarray(
                tex_info.get("_raw_texture_weights", np.zeros((0,), dtype=np.float64)),
                dtype=np.float64,
            )
        return out

    def finalize_batch(
        self,
        features_list: Sequence[Dict[str, float]],
        *,
        fit: bool = True,
    ) -> Sequence[Dict[str, float]]:
        standardized = super().finalize_batch(features_list, fit=fit)
        reordered: List[Dict[str, float]] = []
        for feat in standardized:
            out = OrderedDict()
            for key in self._public_video_keys:
                out[key] = float(feat.get(key, 0.0))
            reordered.append(out)
        return reordered

    def aggregate_subject(self, video_features: Sequence[Dict[str, float]]) -> Dict[str, float]:
        subject = super().aggregate_subject(video_features)
        out: Dict[str, float] = OrderedDict()
        for key in self._public_subject_keys:
            out[key] = float(subject.get(key, 0.0))
        return out


@register_feature_extractor("tab_latest")
@register_feature_extractor("tab_v7")
class MotionStaticV7FeatureExtractor(_MotionStaticV7Base):
    name = "MotionStaticV7Features"


@register_feature_extractor("tab_v7_GD")
@register_feature_extractor("tab_v7_gd")
class MotionStaticV7GDFeatureExtractor(_MotionStaticV7Base):
    name = "MotionStaticV7GDFeatures"
    INCLUDE_MOTION = False


@register_feature_extractor("tab_v7_GM")
@register_feature_extractor("tab_v7_gm")
class MotionStaticV7GMFeatureExtractor(_MotionStaticV7Base):
    name = "MotionStaticV7GMFeatures"
    INCLUDE_TEXTURE = False


@register_feature_extractor("tab_v7_DM")
@register_feature_extractor("tab_v7_dm")
class MotionStaticV7DMFeatureExtractor(_MotionStaticV7Base):
    name = "MotionStaticV7DMFeatures"
    INCLUDE_GEOMETRY = False


@register_feature_extractor("tab_v7_G")
@register_feature_extractor("tab_v7_g")
class MotionStaticV7GFeatureExtractor(_MotionStaticV7Base):
    name = "MotionStaticV7GFeatures"
    INCLUDE_TEXTURE = False
    INCLUDE_MOTION = False


@register_feature_extractor("tab_v7_D")
@register_feature_extractor("tab_v7_d")
class MotionStaticV7DFeatureExtractor(_MotionStaticV7Base):
    name = "MotionStaticV7DFeatures"
    INCLUDE_GEOMETRY = False
    INCLUDE_MOTION = False


@register_feature_extractor("tab_v7_M")
@register_feature_extractor("tab_v7_m")
class MotionStaticV7MFeatureExtractor(_MotionStaticV7Base):
    name = "MotionStaticV7MFeatures"
    INCLUDE_GEOMETRY = False
    INCLUDE_TEXTURE = False


__all__ = [
    "TAB_V7_GEOMETRY_KEYS",
    "TAB_V7_TEXTURE_KEYS",
    "TAB_V7_MOTION_KEYS",
    "TAB_V7_TOTAL_DIM",
    "MotionStaticV7FeatureExtractor",
    "MotionStaticV7GDFeatureExtractor",
    "MotionStaticV7GMFeatureExtractor",
    "MotionStaticV7DMFeatureExtractor",
    "MotionStaticV7GFeatureExtractor",
    "MotionStaticV7DFeatureExtractor",
    "MotionStaticV7MFeatureExtractor",
]
