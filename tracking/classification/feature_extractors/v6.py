from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import cv2
import numpy as np

from ...core.interfaces import FramePrediction, MaskStats, SegmentationData
from ...core.registry import register_feature_extractor
from .v4 import TAB_V4_STATIC_KEYS, _compute_static_tab_v4
from .v5 import (
    TAB_V5_LITE_MOTION_KEYS,
    TAB_V5_MOTION_KEYS,
    MotionStaticV5FeatureExtractor,
    MotionStaticV5LiteFeatureExtractor,
)


@dataclass(frozen=True)
class DepthScaleEstimate:
    px_per_cm: float
    zero_depth_y_px: float
    rule: str

    @property
    def cm_per_px(self) -> float:
        return float(1.0 / max(self.px_per_cm, 1e-9))


def _safe_path_token(path: str | None) -> str:
    return str(path or "").replace("\\", "/").lower()


def normalise_video_path_key(video_path: str | None) -> str:
    if not video_path:
        return ""
    try:
        raw = str(Path(video_path).resolve())
    except Exception:  # noqa: BLE001
        raw = str(video_path)
    return raw.replace("\\", "/").lower()


def _first_readable_frame(video_path: str) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    frame = None
    for _ in range(12):
        ok, current = cap.read()
        if ok and current is not None and current.size > 0:
            frame = current
            break
    cap.release()
    return frame


def _first_black_strip_column(gray_crop: np.ndarray, black_col_mean: float) -> int:
    means = gray_crop.mean(axis=0)
    consecutive = 0
    for idx, value in enumerate(means):
        consecutive = consecutive + 1 if float(value) < black_col_mean else 0
        if consecutive >= 5:
            return max(0, idx - 4)
    return 0


def _row_runs(binary_strip: np.ndarray) -> np.ndarray:
    height, width = binary_strip.shape
    x_start = max(0, width - 15)
    runs = np.zeros(height, dtype=int)
    for y in range(height):
        row = binary_strip[y]
        best = 0
        for x_right in range(x_start, width):
            if not row[x_right]:
                continue
            run = 0
            x = x_right
            while x >= 0 and row[x]:
                run += 1
                x -= 1
            if run > best:
                best = run
        runs[y] = best
    return runs


def _group_centers(runs: np.ndarray, run_min: int) -> np.ndarray:
    mask = runs >= int(run_min)
    centers: List[float] = []
    y = 0
    while y < len(runs):
        if mask[y]:
            y0 = y
            while y + 1 < len(runs) and mask[y + 1]:
                y += 1
            centers.append((y0 + y) / 2.0)
        y += 1
    return np.asarray(centers, dtype=float)


def _extract_tick_centers(
    frame: np.ndarray,
    *,
    crop_width: int,
    threshold: int,
    run_min: int,
    black_col_mean: float | None,
) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    crop = gray[:, gray.shape[1] - crop_width :]
    strip_start = _first_black_strip_column(crop, float(black_col_mean)) if black_col_mean is not None else 0
    strip = crop[:, strip_start:]
    binary_strip = strip >= int(threshold)
    runs = _row_runs(binary_strip)
    return _group_centers(runs, run_min=run_min)


def _dominant_spacing(
    diffs: Iterable[float],
    *,
    tolerance: float = 1.5,
) -> tuple[float, int, float] | None:
    arr = np.asarray(list(diffs), dtype=float)
    if arr.size == 0:
        return None
    best = None
    for value in arr:
        neighborhood = arr[np.abs(arr - value) <= tolerance]
        candidate = (float(np.median(neighborhood)), int(neighborhood.size), float(np.std(neighborhood)))
        if best is None or candidate[1] > best[1] or (candidate[1] == best[1] and candidate[2] < best[2]):
            best = candidate
    return best


def _search_spacing_candidate(
    frame: np.ndarray,
    *,
    crop_width: int,
    black_col_mean: float | None,
    thresholds: Sequence[int],
    run_mins: Sequence[int],
    diff_min: float,
    diff_max: float,
) -> tuple[np.ndarray, float, int, float] | None:
    best = None
    for threshold in thresholds:
        for run_min in run_mins:
            centers = _extract_tick_centers(
                frame,
                crop_width=crop_width,
                threshold=int(threshold),
                run_min=int(run_min),
                black_col_mean=black_col_mean,
            )
            if centers.size < 6:
                continue
            diffs = [float(b - a) for a, b in zip(centers[:-1], centers[1:]) if diff_min <= (b - a) <= diff_max]
            dominant = _dominant_spacing(diffs)
            if dominant is None:
                continue
            spacing, count, std = dominant
            candidate = (count, -std, centers, spacing, threshold, run_min)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    if best is None:
        return None
    _, _, centers, spacing, threshold, run_min = best
    return centers, float(spacing), int(threshold), int(run_min)


def _estimate_standard_merged_extend(frame: np.ndarray) -> DepthScaleEstimate | None:
    candidate = _search_spacing_candidate(
        frame,
        crop_width=100,
        black_col_mean=12.0,
        thresholds=(18, 20, 22, 25),
        run_mins=(4, 5, 6),
        diff_min=30.0,
        diff_max=50.0,
    )
    if candidate is None:
        return None
    centers, minor_spacing_px, _, _ = candidate
    search_offsets = np.arange(0.0, minor_spacing_px, 0.25)
    best_score = None
    best_offset_mod = 0.0
    for offset_mod in search_offsets:
        residuals = np.abs(((centers - offset_mod + 0.5 * minor_spacing_px) % minor_spacing_px) - 0.5 * minor_spacing_px)
        kept = np.sort(residuals)[: max(4, int(0.7 * residuals.size))]
        score = float(np.median(kept)) + 0.1 * float(np.mean(kept))
        if best_score is None or score < best_score:
            best_score = score
            best_offset_mod = float(offset_mod)

    offset_candidates = [best_offset_mod + k * minor_spacing_px for k in range(-4, 5)]
    zero_depth_y_px = float(min(offset_candidates, key=lambda value: abs(value)))
    return DepthScaleEstimate(
        px_per_cm=float(minor_spacing_px / 0.25),
        zero_depth_y_px=zero_depth_y_px,
        rule="merged_extend_standard",
    )


def _estimate_c_wmv(frame: np.ndarray) -> DepthScaleEstimate | None:
    candidate = _search_spacing_candidate(
        frame,
        crop_width=100,
        black_col_mean=None,
        thresholds=(8, 10, 12, 15, 18, 20),
        run_mins=(4, 5, 6),
        diff_min=18.0,
        diff_max=30.0,
    )
    if candidate is None:
        return None
    centers, minor_spacing_px, _, _ = candidate
    if centers.size == 0:
        return None
    return DepthScaleEstimate(
        px_per_cm=float(minor_spacing_px * 10.0),
        zero_depth_y_px=float(centers[0]),
        rule="merged_extend_c_wmv",
    )


def _estimate_depth_scale_without_rules_from_frame(frame: np.ndarray) -> DepthScaleEstimate | None:
    for estimator in (_estimate_c_wmv, _estimate_standard_merged_extend):
        estimate = estimator(frame)
        if estimate is not None:
            return estimate
    return None


def estimate_depth_scale_for_video_no_rules(video_path: str | None) -> DepthScaleEstimate | None:
    if not video_path:
        return None
    frame = _first_readable_frame(video_path)
    if frame is None:
        return None
    return _estimate_depth_scale_without_rules_from_frame(frame)


def depth_scale_to_dict(depth_scale: DepthScaleEstimate) -> Dict[str, float | str]:
    return {
        "px_per_cm": float(depth_scale.px_per_cm),
        "zero_depth_y_px": float(depth_scale.zero_depth_y_px),
        "rule": str(depth_scale.rule),
    }


def depth_scale_from_dict(payload: Any) -> DepthScaleEstimate | None:
    if isinstance(payload, DepthScaleEstimate):
        return payload
    if not isinstance(payload, dict):
        return None
    try:
        px_per_cm = float(payload.get("px_per_cm"))
        zero_depth_y_px = float(payload.get("zero_depth_y_px", 0.0))
    except Exception:  # noqa: BLE001
        return None
    if not np.isfinite(px_per_cm) or px_per_cm <= 0.0:
        return None
    if not np.isfinite(zero_depth_y_px):
        return None
    rule = str(payload.get("rule", "precomputed")).strip() or "precomputed"
    return DepthScaleEstimate(
        px_per_cm=px_per_cm,
        zero_depth_y_px=zero_depth_y_px,
        rule=rule,
    )


def lookup_depth_scale_from_table(
    video_path: str | None,
    depth_scale_lookup: Dict[str, Any] | None,
) -> DepthScaleEstimate | None:
    if not video_path or not depth_scale_lookup:
        return None

    key = normalise_video_path_key(video_path)
    payload = depth_scale_lookup.get(key)
    if payload is None:
        payload = depth_scale_lookup.get(str(video_path))
    return depth_scale_from_dict(payload)


def precompute_depth_scale_lookup_for_videos(
    video_paths: Sequence[str],
    *,
    logger: Optional[Callable[[str], None]] = None,
) -> Dict[str, Dict[str, float | str]]:
    unique_paths = sorted({str(path) for path in video_paths if path})
    if not unique_paths:
        return {}

    lookup: Dict[str, Dict[str, float | str]] = {}
    resolved = 0
    for video_path in unique_paths:
        estimate = estimate_depth_scale_for_video_no_rules(video_path)
        if estimate is None:
            continue
        lookup[normalise_video_path_key(video_path)] = depth_scale_to_dict(estimate)
        resolved += 1

    if logger is not None:
        logger(
            "[Classification] Depth scale precompute done: "
            f"resolved={resolved}/{len(unique_paths)} videos"
        )
    return lookup


@lru_cache(maxsize=512)
def resolve_depth_scale_for_video(video_path: str | None) -> DepthScaleEstimate | None:
    if not video_path:
        return None
    token = _safe_path_token(video_path)

    frame = _first_readable_frame(video_path)
    if frame is None:
        return None

    # The Japan merged-extend set can contain different probe-depth settings
    # across videos, so scale must be inferred per video whenever possible.
    if "merged_extend_control_japan" in token:
        estimate = _estimate_depth_scale_without_rules_from_frame(frame)
        if estimate is not None:
            return DepthScaleEstimate(
                px_per_cm=float(estimate.px_per_cm),
                zero_depth_y_px=float(estimate.zero_depth_y_px),
                rule="merged_extend_control_japan_auto",
            )
        return DepthScaleEstimate(px_per_cm=290.0, zero_depth_y_px=-0.5, rule="merged_extend_control_japan_fixed_fallback")

    path_obj = Path(video_path)
    subject = path_obj.parent.name.lower()
    suffix = path_obj.suffix.lower()
    if "merged_extend" in token and subject.startswith("c") and suffix == ".wmv":
        estimate = _estimate_c_wmv(frame)
        if estimate is not None:
            return estimate

    if "merged_extend" in token:
        estimate = _estimate_standard_merged_extend(frame)
        if estimate is not None:
            return estimate
    return None


def scale_predictions_to_cm(
    samples: Sequence[FramePrediction],
    depth_scale: DepthScaleEstimate | None,
) -> List[FramePrediction]:
    if not samples or depth_scale is None:
        return list(samples)

    linear = float(depth_scale.cm_per_px)
    area_scale = linear * linear
    scaled: List[FramePrediction] = []
    for sample in samples:
        x, y, w, h = sample.bbox
        new_seg = sample.segmentation
        if sample.segmentation is not None and sample.segmentation.stats is not None:
            stats = sample.segmentation.stats
            sx, sy, sw, sh = stats.bbox
            scx, scy = stats.centroid
            scaled_stats = MaskStats(
                area_px=float(stats.area_px) * area_scale,
                bbox=(sx * linear, sy * linear, sw * linear, sh * linear),
                centroid=(scx * linear, scy * linear),
                perimeter_px=float(stats.perimeter_px) * linear,
                equivalent_diameter_px=float(stats.equivalent_diameter_px) * linear,
            )
            new_seg = replace(sample.segmentation, stats=scaled_stats)

        scaled.append(
            replace(
                sample,
                bbox=(x * linear, y * linear, w * linear, h * linear),
                segmentation=new_seg,
            )
        )
    return scaled


@register_feature_extractor("tab_v6")
class MotionStaticV6FeatureExtractor(MotionStaticV5FeatureExtractor):
    name = "MotionStaticV6Features"
    DEFAULT_CONFIG: Dict[str, Any] = dict(MotionStaticV5FeatureExtractor.DEFAULT_CONFIG)

    def _scaled_samples_for_video(
        self,
        samples: Sequence[FramePrediction],
        video_path: Optional[str],
    ) -> List[FramePrediction]:
        depth_scale = resolve_depth_scale_for_video(video_path)
        return scale_predictions_to_cm(samples, depth_scale)

    def extract_video(
        self,
        samples: Sequence[FramePrediction],
        video_path: Optional[str] = None,
    ) -> Dict[str, float]:
        scaled_samples = self._scaled_samples_for_video(samples, video_path)
        motion_info = self._extract_moment_motion_vector(scaled_samples)
        static = _compute_static_tab_v4(scaled_samples)
        tex_info = self._extract_texture_vector(samples, video_path)
        motion_vec = np.asarray(motion_info.get("motion"), dtype=np.float32)
        tex_vec = np.asarray(tex_info.get("tex"), dtype=np.float32)

        out: Dict[str, float] = OrderedDict()
        for i, key in enumerate(self._motion_keys):
            out[key] = float(motion_vec[i]) if i < motion_vec.size else 0.0
        for key in self._static_keys:
            out[key] = float(static.get(key, 0.0))
        for i, key in enumerate(self._texture_keys):
            out[key] = float(tex_vec[i]) if i < tex_vec.size else 0.0

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


@register_feature_extractor("tab_v6_lite")
class MotionStaticV6LiteFeatureExtractor(MotionStaticV5LiteFeatureExtractor):
    name = "MotionStaticV6LiteFeatures"
    DEFAULT_CONFIG: Dict[str, Any] = {
        **MotionStaticV5LiteFeatureExtractor.DEFAULT_CONFIG,
        "moment_pca_dim": len(TAB_V5_LITE_MOTION_KEYS),
    }

    def _scaled_samples_for_video(
        self,
        samples: Sequence[FramePrediction],
        video_path: Optional[str],
    ) -> List[FramePrediction]:
        depth_scale = resolve_depth_scale_for_video(video_path)
        return scale_predictions_to_cm(samples, depth_scale)

    def extract_video(
        self,
        samples: Sequence[FramePrediction],
        video_path: Optional[str] = None,
    ) -> Dict[str, float]:
        scaled_samples = self._scaled_samples_for_video(samples, video_path)
        motion_info = self._extract_moment_motion_vector(scaled_samples)
        tex_info = self._extract_texture_vector(samples, video_path)
        motion_vec = np.asarray(motion_info.get("motion"), dtype=np.float32)
        tex_vec = np.asarray(tex_info.get("tex"), dtype=np.float32)

        out: Dict[str, float] = OrderedDict()
        for i, key in enumerate(self._motion_keys):
            out[key] = float(motion_vec[i]) if i < motion_vec.size else 0.0
        for i, key in enumerate(self._texture_keys):
            out[key] = float(tex_vec[i]) if i < tex_vec.size else 0.0

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


__all__ = [
    "DepthScaleEstimate",
    "MotionStaticV6FeatureExtractor",
    "MotionStaticV6LiteFeatureExtractor",
    "TAB_V4_STATIC_KEYS",
    "TAB_V5_MOTION_KEYS",
    "TAB_V5_LITE_MOTION_KEYS",
    "depth_scale_to_dict",
    "depth_scale_from_dict",
    "normalise_video_path_key",
    "lookup_depth_scale_from_table",
    "estimate_depth_scale_for_video_no_rules",
    "precompute_depth_scale_lookup_for_videos",
    "resolve_depth_scale_for_video",
    "scale_predictions_to_cm",
]
