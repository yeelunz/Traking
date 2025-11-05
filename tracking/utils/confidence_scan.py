from __future__ import annotations

import json
import math
import csv
import io
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .confidence import ConfidenceConfig, ConfidenceEstimator

BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class FrameConfidenceSnapshot:
    frame_index: int
    confidence: float
    raw_score: Optional[float]
    components: Dict[str, float]
    drift_pixels: float


@dataclass(frozen=True)
class SequenceConfidenceSummary:
    experiment: str
    tracker: str
    source_path: Path
    total_frames: int
    confidence_mean: float
    confidence_std: float
    confidence_p10: float
    confidence_p05: float
    confidence_min: float
    below_threshold: int
    below_threshold_ratio: float
    longest_low_streak: int
    score_component_mean: float
    token_component_mean: float
    distribution_component_mean: float
    attention_component_mean: float
    short_iou_component_mean: float
    drift_component_mean: float
    drift_pixels_mean: float
    drift_pixels_p95: float
    raw_score_mean: Optional[float]
    raw_score_p10: Optional[float]
    worst_frames: List[FrameConfidenceSnapshot] = field(default_factory=list)


def _bbox_center(bbox: BBox) -> Tuple[float, float]:
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0


def _frame_percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return math.nan
    if percentile <= 0.0:
        return float(values[0])
    if percentile >= 1.0:
        return float(values[-1])
    position = (len(values) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(values[lower])
    lower_value = values[lower]
    upper_value = values[upper]
    fraction = position - lower
    return float(lower_value + (upper_value - lower_value) * fraction)


def _compute_longest_low_streak(values: Iterable[float], threshold: float) -> int:
    current = 0
    longest = 0
    for value in values:
        if value < threshold:
            current += 1
            if current > longest:
                longest = current
        else:
            current = 0
    return longest


def _safe_mean(sequence: Sequence[float]) -> float:
    return float(mean(sequence)) if sequence else math.nan


def _safe_std(sequence: Sequence[float]) -> float:
    return float(pstdev(sequence)) if len(sequence) >= 2 else 0.0


def _safe_filtered_mean(sequence: Sequence[float]) -> float:
    filtered = [value for value in sequence if not math.isnan(value)]
    return float(mean(filtered)) if filtered else math.nan


def _safe_optional_mean(sequence: Sequence[Optional[float]]) -> Optional[float]:
    filtered = [value for value in sequence if value is not None]
    return float(mean(filtered)) if filtered else None


def _safe_optional_percentile(sequence: Sequence[Optional[float]], percentile: float) -> Optional[float]:
    filtered = sorted(value for value in sequence if value is not None)
    if not filtered:
        return None
    return _frame_percentile(filtered, percentile)


def _coerce_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def _load_predictions(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Prediction file {path} does not contain a list")
    return data


def analyse_prediction_file(
    prediction_path: Path,
    *,
    experiment: str,
    tracker: Optional[str] = None,
    config: Optional[ConfidenceConfig] = None,
    low_threshold: float = 0.5,
    top_k_frames: int = 8,
) -> Optional[SequenceConfidenceSummary]:
    records = _load_predictions(prediction_path)
    if not records:
        return None

    tracker_name = tracker or prediction_path.stem
    estimator = ConfidenceEstimator(config)
    estimator.reset()

    confidence_values: List[float] = []
    raw_scores: List[Optional[float]] = []
    score_components: List[float] = []
    token_components: List[float] = []
    distribution_components: List[float] = []
    attention_components: List[float] = []
    short_iou_components: List[float] = []
    drift_components: List[float] = []
    drift_pixels_list: List[float] = []
    snapshots: List[FrameConfidenceSnapshot] = []

    anchor_bbox: Optional[BBox] = None

    def _to_bbox(item: Dict[str, object]) -> BBox:
        bbox_list = item.get("bbox")
        if not isinstance(bbox_list, (list, tuple)) or len(bbox_list) != 4:
            raise ValueError(f"Invalid bbox in {prediction_path}: {bbox_list}")
        return tuple(float(value) for value in bbox_list)  # type: ignore[return-value]

    for entry in records:
        if not isinstance(entry, dict):
            continue
        try:
            frame_index = int(entry.get("frame_index"))
        except Exception:
            continue
        bbox = _to_bbox(entry)
        raw_score_obj = entry.get("score")
        raw_score = float(raw_score_obj) if raw_score_obj is not None else None

        if anchor_bbox is None:
            anchor_bbox = bbox

        state = estimator.update(frame_index=frame_index, bbox=bbox, raw_score=raw_score)

        components_entry = entry.get("components")
        components: Dict[str, float]
        if isinstance(components_entry, dict) and components_entry:
            comp_clean: Dict[str, float] = {}
            for key, value in components_entry.items():
                try:
                    comp_clean[key] = float(value)
                except (TypeError, ValueError):
                    continue
            components = comp_clean
            confidence_obj = entry.get("confidence")
            confidence = float(confidence_obj) if confidence_obj is not None else state.confidence
        else:
            components = state.raw_components
            confidence = state.confidence

        confidence_values.append(confidence)
        raw_scores.append(raw_score)

        score_value = _coerce_float(components.get("raw_score"))
        if score_value is None:
            score_value = _coerce_float(components.get("score"))
        if score_value is None and raw_score is not None:
            raw_score_coerced = _coerce_float(raw_score)
            if raw_score_coerced is not None:
                score_value = max(0.0, min(1.0, raw_score_coerced))
        score_component_value = score_value if score_value is not None else math.nan

        score_components.append(score_component_value)
        token_value = components.get("token")
        if token_value is not None:
            token_components.append(float(token_value))
        distribution_value = components.get("distribution")
        if distribution_value is not None:
            distribution_components.append(float(distribution_value))
        attention_value = components.get("attention")
        if attention_value is not None:
            attention_components.append(float(attention_value))
        short_iou_components.append(components.get("short_iou", math.nan))
        drift_components.append(components.get("drift", math.nan))

        if anchor_bbox is None:
            drift_pixels = 0.0
        else:
            anchor_cx, anchor_cy = _bbox_center(anchor_bbox)
            cur_cx, cur_cy = _bbox_center(bbox)
            drift_pixels = math.hypot(cur_cx - anchor_cx, cur_cy - anchor_cy)
        drift_pixels_list.append(drift_pixels)

        snapshots.append(
            FrameConfidenceSnapshot(
                frame_index=frame_index,
                confidence=confidence,
                raw_score=raw_score,
                components={
                    "raw_score": score_component_value,
                    "score": score_component_value,
                    "token": components.get("token", math.nan),
                    "distribution": components.get("distribution", math.nan),
                    "attention": components.get("attention", math.nan),
                    "short_iou": components.get("short_iou", math.nan),
                    "drift": components.get("drift", math.nan),
                    "blended": components.get("blended", math.nan),
                },
                drift_pixels=drift_pixels,
            )
        )

    if not confidence_values:
        return None

    paired = sorted(zip(confidence_values, snapshots), key=lambda item: item[0])
    worst_frames = [item[1] for item in paired[: max(1, top_k_frames)]]

    sorted_confidences = sorted(confidence_values)
    sorted_drift_pixels = sorted(drift_pixels_list)

    below_threshold = sum(1 for value in confidence_values if value < low_threshold)
    ratio = below_threshold / len(confidence_values) if confidence_values else 0.0
    longest_low = _compute_longest_low_streak(confidence_values, low_threshold)

    summary = SequenceConfidenceSummary(
        experiment=experiment,
        tracker=tracker_name,
        source_path=prediction_path,
        total_frames=len(confidence_values),
        confidence_mean=_safe_mean(confidence_values),
        confidence_std=_safe_std(confidence_values),
        confidence_p10=_frame_percentile(sorted_confidences, 0.10),
        confidence_p05=_frame_percentile(sorted_confidences, 0.05),
        confidence_min=sorted_confidences[0],
        below_threshold=below_threshold,
        below_threshold_ratio=ratio,
        longest_low_streak=longest_low,
    score_component_mean=_safe_filtered_mean(score_components),
    token_component_mean=_safe_mean(token_components),
    distribution_component_mean=_safe_mean(distribution_components),
    attention_component_mean=_safe_mean(attention_components),
    short_iou_component_mean=_safe_filtered_mean(short_iou_components),
    drift_component_mean=_safe_filtered_mean(drift_components),
        drift_pixels_mean=_safe_mean(drift_pixels_list),
        drift_pixels_p95=_frame_percentile(sorted_drift_pixels, 0.95) if sorted_drift_pixels else math.nan,
        raw_score_mean=_safe_optional_mean(raw_scores),
        raw_score_p10=_safe_optional_percentile(raw_scores, 0.10),
        worst_frames=worst_frames,
    )
    return summary


def scan_schedule_confidence(
    schedule_dir: Path,
    *,
    config: Optional[ConfidenceConfig] = None,
    low_threshold: float = 0.5,
    trackers: Optional[Sequence[str]] = None,
    top_k_frames: int = 8,
) -> List[SequenceConfidenceSummary]:
    schedule_path = Path(schedule_dir)
    if not schedule_path.is_dir():
        raise FileNotFoundError(f"Schedule directory not found: {schedule_dir}")

    summaries: List[SequenceConfidenceSummary] = []

    for experiment_dir in sorted(schedule_path.iterdir()):
        if not experiment_dir.is_dir():
            continue
        predictions_dir = experiment_dir / "test" / "predictions"
        if not predictions_dir.is_dir():
            continue
        for prediction_file in sorted(predictions_dir.glob("*.json")):
            tracker_name = prediction_file.stem
            if trackers and tracker_name not in trackers:
                continue
            summary = analyse_prediction_file(
                prediction_file,
                experiment=experiment_dir.name,
                tracker=tracker_name,
                config=config,
                low_threshold=low_threshold,
                top_k_frames=top_k_frames,
            )
            if summary:
                summaries.append(summary)

    summaries.sort(key=lambda item: (item.confidence_p05, item.confidence_mean))
    return summaries


def summaries_to_csv(
    summaries: Sequence[SequenceConfidenceSummary],
    *,
    include_header: bool = True,
    float_precision: int = 6,
) -> str:
    """Serialize summary metrics into a CSV string suitable for clipboard usage."""

    if not summaries:
        return ""

    columns: Sequence[Tuple[str, callable]] = [
        ("experiment", lambda s: s.experiment),
        ("tracker", lambda s: s.tracker),
        ("total_frames", lambda s: s.total_frames),
        ("confidence_mean", lambda s: s.confidence_mean),
        ("confidence_p10", lambda s: s.confidence_p10),
        ("confidence_p05", lambda s: s.confidence_p05),
        ("confidence_min", lambda s: s.confidence_min),
        ("below_threshold", lambda s: s.below_threshold),
        ("below_threshold_ratio", lambda s: s.below_threshold_ratio),
        ("longest_low_streak", lambda s: s.longest_low_streak),
        ("score_component_mean", lambda s: s.score_component_mean),
    ("token_component_mean", lambda s: s.token_component_mean),
    ("distribution_component_mean", lambda s: s.distribution_component_mean),
    ("attention_component_mean", lambda s: s.attention_component_mean),
        ("short_iou_component_mean", lambda s: s.short_iou_component_mean),
        ("drift_component_mean", lambda s: s.drift_component_mean),
        ("drift_pixels_mean", lambda s: s.drift_pixels_mean),
        ("drift_pixels_p95", lambda s: s.drift_pixels_p95),
        ("raw_score_mean", lambda s: s.raw_score_mean),
        ("raw_score_p10", lambda s: s.raw_score_p10),
        ("source_path", lambda s: str(s.source_path)),
    ]

    def _format_value(value: Optional[float]) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            if math.isnan(value):
                return ""
            return f"{value:.{float_precision}f}"
        return str(value)

    buffer = io.StringIO()
    writer = csv.writer(buffer)

    if include_header:
        writer.writerow([name for name, _ in columns])

    for summary in summaries:
        row = [_format_value(getter(summary)) for _, getter in columns]
        writer.writerow(row)

    return buffer.getvalue().strip()


__all__ = [
    "FrameConfidenceSnapshot",
    "SequenceConfidenceSummary",
    "analyse_prediction_file",
    "scan_schedule_confidence",
    "summaries_to_csv",
]
