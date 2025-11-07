from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np


COMPONENT_KEYS = [
    "token",
    "distribution",
    "drift",
    "blended",
    "logit",
]


@dataclass
class FrameRecord:
    frame_index: int
    confidence: float
    iou: float
    center_error: float
    components: Dict[str, Optional[float]]


def _load_predictions(prediction_file: Path) -> Dict[int, FrameRecord]:
    with prediction_file.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Prediction file does not contain a list: {prediction_file}")

    records: Dict[int, FrameRecord] = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        try:
            frame_index = int(entry.get("frame_index"))
        except Exception:
            continue
        confidence_obj = entry.get("confidence")
        if confidence_obj is None:
            continue
        try:
            confidence = float(confidence_obj)
        except Exception:
            continue
        components = entry.get("components")
        comp_dict: Dict[str, Optional[float]] = {}
        if isinstance(components, dict):
            for key in COMPONENT_KEYS:
                value = components.get(key)
                if value is None:
                    comp_dict[key] = None
                else:
                    try:
                        comp_dict[key] = float(value)
                    except Exception:
                        comp_dict[key] = None
        else:
            comp_dict = {key: None for key in COMPONENT_KEYS}
        records[frame_index] = FrameRecord(
            frame_index=frame_index,
            confidence=confidence,
            iou=math.nan,
            center_error=math.nan,
            components=comp_dict,
        )
    return records


def _attach_metrics(records: Dict[int, FrameRecord], metrics_file: Path) -> List[FrameRecord]:
    with metrics_file.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                frame_index = int(row["frame_index"])
                iou = float(row["iou"])
                center_error = float(row["center_error"])
            except Exception:
                continue
            record = records.get(frame_index)
            if record is None:
                continue
            record.iou = iou
            record.center_error = center_error
    return [record for record in records.values() if not math.isnan(record.iou) and not math.isnan(record.center_error)]


def _collect_experiment_records(experiment_dir: Path, model_name: str) -> List[FrameRecord]:
    detection_root = experiment_dir / "test" / "detection"
    predictions_root = detection_root / "predictions_by_video"
    metrics_root = detection_root / "metrics"
    aggregated_file = detection_root / "predictions" / f"{model_name}.json"
    if not predictions_root.is_dir() or not metrics_root.is_dir():
        return []

    aggregated_components: List[Dict[str, object]] = []
    if aggregated_file.is_file():
        with aggregated_file.open("r", encoding="utf-8") as handle:
            aggregated_components = json.load(handle)

    agg_index = 0
    collected: List[FrameRecord] = []
    for video_dir in sorted(predictions_root.iterdir()):
        if not video_dir.is_dir():
            continue
        pred_file = video_dir / f"{model_name}.json"
        if not pred_file.is_file():
            continue

        with pred_file.open("r", encoding="utf-8") as handle:
            video_predictions = json.load(handle)
        if not isinstance(video_predictions, list):
            continue

        # Attach component information from aggregated predictions if available
        component_chunk: List[Dict[str, object]] = []
        if aggregated_components:
            component_chunk = aggregated_components[agg_index: agg_index + len(video_predictions)]
            agg_index += len(video_predictions)
        records: Dict[int, FrameRecord] = {}
        for idx, entry in enumerate(video_predictions):
            if not isinstance(entry, dict):
                continue
            try:
                frame_index = int(entry.get("frame_index"))
                confidence = float(entry.get("confidence"))
            except Exception:
                continue

            comp_dict: Dict[str, Optional[float]] = {key: None for key in COMPONENT_KEYS}
            if component_chunk and idx < len(component_chunk):
                comp_entry = component_chunk[idx]
                if isinstance(comp_entry, dict):
                    comps = comp_entry.get("components")
                    if isinstance(comps, dict):
                        for key in COMPONENT_KEYS:
                            value = comps.get(key)
                            if value is None:
                                continue
                            try:
                                comp_dict[key] = float(value)
                            except Exception:
                                continue

            records[frame_index] = FrameRecord(
                frame_index=frame_index,
                confidence=confidence,
                iou=math.nan,
                center_error=math.nan,
                components=comp_dict,
            )

        metrics_file = metrics_root / video_dir.name / f"{model_name}_per_frame.csv"
        if not metrics_file.is_file():
            continue
        collected.extend(_attach_metrics(records, metrics_file))
    return collected


def _safe_corr(x_values: Iterable[float], y_values: Iterable[float]) -> float:
    x = np.asarray(list(x_values), dtype=np.float64)
    y = np.asarray(list(y_values), dtype=np.float64)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return math.nan
    if np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def _quantiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def _threshold_stats(values: List[float], targets: List[float], thresholds: List[float]) -> Dict[float, Dict[str, float]]:
    arr_values = np.asarray(values, dtype=np.float64)
    arr_targets = np.asarray(targets, dtype=np.float64)
    stats: Dict[float, Dict[str, float]] = {}
    for threshold in thresholds:
        mask = arr_values < threshold
        if not np.any(mask):
            stats[threshold] = {"count": 0}
            continue
        below = arr_targets[mask]
        above = arr_targets[~mask]
        stats[threshold] = {
            "count": int(below.size),
            "mean_below": float(np.mean(below)) if below.size else math.nan,
            "mean_above": float(np.mean(above)) if above.size else math.nan,
        }
    return stats


def summarise_records(name: str, records: List[FrameRecord], thresholds: List[float]) -> None:
    if not records:
        print(f"\n=== {name} ===\n  No matched frames")
        return

    confidences = [r.confidence for r in records]
    ious = [r.iou for r in records]
    center_errors = [r.center_error for r in records]

    print(f"\n=== {name} ===")
    print(f"  frames: {len(records)}  mean IoU: {np.mean(ious):.4f}  mean CLE: {np.mean(center_errors):.4f}")

    conf_quantiles = _quantiles(confidences)
    iou_quantiles = _quantiles(ious)
    cle_quantiles = _quantiles(center_errors)

    if conf_quantiles:
        print(
            "  confidence quantiles: "
            + ", ".join(f"{key}={value:.4f}" for key, value in conf_quantiles.items())
        )
    if iou_quantiles:
        print(
            "  IoU quantiles: "
            + ", ".join(f"{key}={value:.4f}" for key, value in iou_quantiles.items())
        )
    if cle_quantiles:
        print(
            "  CLE quantiles: "
            + ", ".join(f"{key}={value:.4f}" for key, value in cle_quantiles.items())
        )

    # Threshold analysis
    iou_stats = _threshold_stats(confidences, ious, thresholds)
    cle_stats = _threshold_stats(confidences, center_errors, thresholds)
    for threshold in thresholds:
        iou_entry = iou_stats.get(threshold, {})
        cle_entry = cle_stats.get(threshold, {})
        count = iou_entry.get("count", 0)
        print(
            f"  confidence < {threshold:.2f}: count={count:4d} "
            f"IoU(mean)={iou_entry.get('mean_below', float('nan')):.4f} vs {iou_entry.get('mean_above', float('nan')):.4f} "
            f"CLE(mean)={cle_entry.get('mean_below', float('nan')):.4f} vs {cle_entry.get('mean_above', float('nan')):.4f}"
        )

    # Correlations for confidence
    conf_iou_corr = _safe_corr(confidences, ious)
    conf_cle_corr = _safe_corr(confidences, [-c for c in center_errors])
    print(f"  Corr(confidence, IoU)={conf_iou_corr:.4f}  Corr(confidence, -CLE)={conf_cle_corr:.4f}")

    # Component correlations
    component_arrays: Dict[str, List[float]] = defaultdict(list)
    for record in records:
        for key in COMPONENT_KEYS:
            value = record.components.get(key)
            if value is None or math.isnan(value):
                continue
            component_arrays[key].append((value, record.iou, record.center_error))

    for key in COMPONENT_KEYS:
        entries = component_arrays.get(key)
        if not entries:
            print(f"  {key:12s}: no data")
            continue
        values, iou_vals, cle_vals = zip(*entries)
        corr_iou = _safe_corr(values, iou_vals)
        corr_cle = _safe_corr(values, [-c for c in cle_vals])
        print(f"  {key:12s}: Corr(v, IoU)={corr_iou:.4f}  Corr(v, -CLE)={corr_cle:.4f}")


def analyse_schedule(schedule_dir: Path, *, thresholds: List[float], model_name: str) -> None:
    schedule_dir = schedule_dir.resolve()
    if not schedule_dir.is_dir():
        raise FileNotFoundError(f"Schedule directory not found: {schedule_dir}")

    overall_records: List[FrameRecord] = []
    for experiment_dir in sorted(schedule_dir.iterdir()):
        if not experiment_dir.is_dir():
            continue
        experiment_records = _collect_experiment_records(experiment_dir, model_name)
        summarise_records(experiment_dir.name, experiment_records, thresholds)
        overall_records.extend(experiment_records)

    summarise_records("ALL", overall_records, thresholds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse confidence against ground truth IoU/CLE.")
    parser.add_argument("schedule", type=Path, help="Path to schedule results directory")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=[0.60, 0.65, 0.70],
        help="Confidence thresholds for low-confidence analysis",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="MixFormerV2",
        help="Model name whose predictions should be analysed",
    )
    args = parser.parse_args()

    analyse_schedule(args.schedule, thresholds=args.thresholds, model_name=args.model)


if __name__ == "__main__":
    main()
