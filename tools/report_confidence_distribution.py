from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def _percentile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        return math.nan
    if q <= 0:
        return sorted_values[0]
    if q >= 1:
        return sorted_values[-1]
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    fraction = position - lower
    return lower_value + (upper_value - lower_value) * fraction


def _describe(values: Iterable[Optional[float]]) -> Dict[str, float]:
    filtered = [float(v) for v in values if v is not None and not math.isnan(v)]
    if not filtered:
        return {"count": 0}
    filtered.sort()
    n = len(filtered)
    mean = sum(filtered) / n
    variance = sum((v - mean) ** 2 for v in filtered) / n
    return {
        "count": n,
        "mean": mean,
        "std": math.sqrt(variance),
        "min": filtered[0],
        "p05": _percentile(filtered, 0.05),
        "p10": _percentile(filtered, 0.10),
        "median": _percentile(filtered, 0.50),
        "p90": _percentile(filtered, 0.90),
        "p95": _percentile(filtered, 0.95),
        "max": filtered[-1],
    }


def _load_predictions(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Prediction file {path} does not contain a list")
    return data


def analyse_prediction(path: Path) -> Dict[str, Dict[str, float]]:
    records = _load_predictions(path)
    confidences: List[Optional[float]] = []
    components: Dict[str, List[Optional[float]]] = {
        "raw_score": [],
        "token": [],
        "distribution": [],
        "attention": [],
        "short_iou": [],
        "drift": [],
        "blended": [],
    }

    for entry in records:
        if not isinstance(entry, dict):
            continue
        confidence_obj = entry.get("confidence")
        confidences.append(float(confidence_obj) if confidence_obj is not None else None)
        comp = entry.get("components")
        if not isinstance(comp, dict):
            comp = {}
        for key in components.keys():
            value = comp.get(key)
            try:
                components[key].append(float(value) if value is not None else None)
            except (TypeError, ValueError):
                components[key].append(None)

    summary = {"confidence": _describe(confidences)}
    for key, values in components.items():
        summary[key] = _describe(values)
    return summary


def analyse_schedule(schedule_dir: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    result: Dict[str, Dict[str, Dict[str, float]]] = {}
    schedule_dir = schedule_dir.resolve()
    for experiment_dir in sorted(schedule_dir.iterdir()):
        if not experiment_dir.is_dir():
            continue
        prediction_dir = experiment_dir / "test" / "detection" / "predictions"
        if not prediction_dir.is_dir():
            continue
        for json_file in sorted(prediction_dir.glob("*.json")):
            key = f"{experiment_dir.name}/{json_file.stem}"
            result[key] = analyse_prediction(json_file)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse confidence distributions for a schedule directory.")
    parser.add_argument("schedule", type=Path, help="Path to the schedule results directory")
    args = parser.parse_args()

    if not args.schedule.exists():
        parser.error(f"Schedule directory not found: {args.schedule}")

    report = analyse_schedule(args.schedule)
    if not report:
        print("No prediction files found.")
        return

    for experiment_key, metrics in report.items():
        print(f"\n=== {experiment_key} ===")
        for component, stats in metrics.items():
            count = int(stats.get("count", 0))
            if count == 0:
                print(f"  {component:12s}: no data")
                continue
            print(
                f"  {component:12s}: count={count:4d} mean={stats['mean']:.4f} "
                f"std={stats['std']:.4f} min={stats['min']:.4f} p05={stats['p05']:.4f} "
                f"p10={stats['p10']:.4f} median={stats['median']:.4f} p90={stats['p90']:.4f} "
                f"p95={stats['p95']:.4f} max={stats['max']:.4f}"
            )


if __name__ == "__main__":
    main()
