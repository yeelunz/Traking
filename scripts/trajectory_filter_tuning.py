from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracking.classification.trajectory_filter import filter_detections


@dataclass
class EvalStats:
    count: int = 0
    sum_iou: float = 0.0
    sum_iou_sq: float = 0.0
    sum_ce: float = 0.0
    sum_ce_sq: float = 0.0
    tp50: int = 0
    fp50: int = 0
    tp75: int = 0
    fp75: int = 0

    def update(self, iou: float, ce: float) -> None:
        self.count += 1
        self.sum_iou += iou
        self.sum_iou_sq += iou * iou
        self.sum_ce += ce
        self.sum_ce_sq += ce * ce
        if iou >= 0.5:
            self.tp50 += 1
        else:
            self.fp50 += 1
        if iou >= 0.75:
            self.tp75 += 1
        else:
            self.fp75 += 1

    def to_metrics(self) -> Dict[str, float]:
        n = max(1, self.count)
        iou_mean = self.sum_iou / n
        ce_mean = self.sum_ce / n
        iou_std = max(0.0, self.sum_iou_sq / n - iou_mean * iou_mean) ** 0.5
        ce_std = max(0.0, self.sum_ce_sq / n - ce_mean * ce_mean) ** 0.5
        sr50 = self.tp50 / (self.tp50 + self.fp50) if (self.tp50 + self.fp50) else 0.0
        sr75 = self.tp75 / (self.tp75 + self.fp75) if (self.tp75 + self.fp75) else 0.0
        return {
            "count": float(self.count),
            "iou_mean": float(iou_mean),
            "iou_std": float(iou_std),
            "ce_mean": float(ce_mean),
            "ce_std": float(ce_std),
            "success_rate_50": float(sr50),
            "success_rate_75": float(sr75),
        }


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iou(px: float, py: float, pw: float, ph: float, gx: float, gy: float, gw: float, gh: float) -> float:
    ix1 = max(px, gx)
    iy1 = max(py, gy)
    ix2 = min(px + pw, gx + gw)
    iy2 = min(py + ph, gy + gh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = pw * ph + gw * gh - inter
    return inter / union if union > 0 else 0.0


def _split_predictions_by_video(preds: List[Dict]) -> List[List[Dict]]:
    chunks: List[List[Dict]] = []
    cur: List[Dict] = []
    prev_fi = None
    for p in preds:
        fi = int(p.get("frame_index", -1))
        if prev_fi is not None and fi < prev_fi and cur:
            chunks.append(cur)
            cur = []
        cur.append(p)
        prev_fi = fi
    if cur:
        chunks.append(cur)
    return chunks


def _build_pred_map(chunk: List[Dict], frame_offset: int = 0) -> Dict[int, Tuple[float, float, float, float]]:
    out: Dict[int, Tuple[float, float, float, float]] = {}
    for p in chunk:
        fi = int(p["frame_index"]) + frame_offset
        if fi in out:
            continue
        x, y, w, h = [float(v) for v in p["bbox"][:4]]
        out[fi] = (x, y, w, h)
    return out


def _extract_gt_map(gt_json: Path) -> Dict[int, Tuple[float, float, float, float]]:
    gt = _load_json(gt_json)
    out: Dict[int, Tuple[float, float, float, float]] = {}

    if isinstance(gt, dict) and "frames" in gt:
        frames = gt.get("frames", {})
        for k, boxes in frames.items():
            if not boxes:
                continue
            fi = int(k)
            b = boxes[0]
            out[fi] = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        return out

    if isinstance(gt, dict) and "images" in gt and "annotations" in gt:
        image_to_frame: Dict[int, int] = {}
        for img in gt.get("images", []):
            iid = int(img.get("id"))
            fi = int(img.get("frame_index", -1))
            if fi >= 0:
                image_to_frame[iid] = fi

        for ann in gt.get("annotations", []):
            iid = int(ann.get("image_id", -1))
            if iid not in image_to_frame:
                continue
            fi = image_to_frame[iid]
            if fi in out:
                continue
            b = ann.get("bbox") or []
            if len(b) < 4:
                continue
            out[fi] = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        return out

    return out


def _run_filter_on_chunk(chunk: List[Dict], strategy: str, bbox_params: Dict[str, Any], traj_params: Dict[str, Any]) -> List[Dict]:
    arr = chunk
    fi = np.asarray([int(p["frame_index"]) for p in arr], dtype=np.int64)
    x = np.asarray([float(p["bbox"][0]) for p in arr], dtype=np.float64)
    y = np.asarray([float(p["bbox"][1]) for p in arr], dtype=np.float64)
    w = np.asarray([float(p["bbox"][2]) for p in arr], dtype=np.float64)
    h = np.asarray([float(p["bbox"][3]) for p in arr], dtype=np.float64)
    s = np.asarray([float(p.get("score") or 0.0) for p in arr], dtype=np.float64)

    cx = x + w / 2.0
    cy = y + h / 2.0

    out = filter_detections(
        fi,
        cx,
        cy,
        w,
        h,
        s,
        bbox_strategy=strategy,
        bbox_params=bbox_params,
        traj_params=traj_params,
        skip_hampel=False,
    )

    cx2 = out["cx"]
    cy2 = out["cy"]
    w2 = out["widths"]
    h2 = out["heights"]
    x2 = cx2 - w2 / 2.0
    y2 = cy2 - h2 / 2.0

    filtered: List[Dict] = []
    for i in range(len(fi)):
        filtered.append(
            {
                "frame_index": int(out["frame_indices"][i]),
                "bbox": [float(x2[i]), float(y2[i]), float(w2[i]), float(h2[i])],
                "score": float(out["scores"][i]),
            }
        )
    return filtered


def _eval_video(pred_map: Dict[int, Tuple[float, float, float, float]], gt_map: Dict[int, Tuple[float, float, float, float]]) -> EvalStats:
    stats = EvalStats()
    frames = sorted(gt_map.keys())
    for fi in frames:
        if fi not in pred_map:
            continue
        px, py, pw, ph = pred_map[fi]
        gx, gy, gw, gh = gt_map[fi]
        iou = _iou(px, py, pw, ph, gx, gy, gw, gh)
        ce = float(np.hypot(px + pw / 2.0 - (gx + gw / 2.0), py + ph / 2.0 - (gy + gh / 2.0)))
        stats.update(iou, ce)
    return stats


def _merge_stats(a: EvalStats, b: EvalStats) -> EvalStats:
    out = EvalStats(
        count=a.count + b.count,
        sum_iou=a.sum_iou + b.sum_iou,
        sum_iou_sq=a.sum_iou_sq + b.sum_iou_sq,
        sum_ce=a.sum_ce + b.sum_ce,
        sum_ce_sq=a.sum_ce_sq + b.sum_ce_sq,
        tp50=a.tp50 + b.tp50,
        fp50=a.fp50 + b.fp50,
        tp75=a.tp75 + b.tp75,
        fp75=a.fp75 + b.fp75,
    )
    return out


def evaluate_experiment(exp_dir: Path, candidates: List[Tuple[str, Dict[str, Any], Dict[str, Any]]]) -> Dict[str, Any] | None:
    meta_path = exp_dir / "metadata.json"
    if not meta_path.exists():
        return None
    meta = _load_json(meta_path)
    dataset_root = Path(meta["dataset"]["root"])
    test_preview = list(meta["dataset"].get("test_preview", []))

    pred_dir = exp_dir / "test" / "detection" / "predictions"
    if not pred_dir.exists():
        return None
    pred_files = sorted(pred_dir.glob("*.json"))
    if not pred_files:
        return None

    # use first model file (single model in this pipeline)
    raw_preds = _load_json(pred_files[0])
    if not isinstance(raw_preds, list) or not raw_preds:
        return None
    chunks = _split_predictions_by_video(raw_preds)

    if len(chunks) != len(test_preview):
        return None

    gt_maps: List[Dict[int, Tuple[float, float, float, float]]] = []
    for rel in test_preview:
        gt_json = dataset_root / rel
        gt_json = gt_json.with_suffix(".json")
        if not gt_json.exists():
            return None
        gt_maps.append(_extract_gt_map(gt_json))

    # baseline from current filtered_detection_summary if exists
    baseline_metrics = None
    baseline_path = exp_dir / "test" / "trajectory_filter" / "filtered_detection_summary.json"
    if baseline_path.exists():
        bd = _load_json(baseline_path)
        if isinstance(bd, dict) and bd:
            first = next(iter(bd.values()))
            if isinstance(first, dict):
                baseline_metrics = first

    raw_baseline = None
    raw_path = exp_dir / "test" / "detection" / "metrics" / "summary.json"
    if raw_path.exists():
        rd = _load_json(raw_path)
        if isinstance(rd, dict) and rd:
            first = next(iter(rd.values()))
            if isinstance(first, dict):
                raw_baseline = first

    best_key = None
    best_metrics = None
    all_results: List[Dict[str, Any]] = []

    for strategy, bbox_params, traj_params in candidates:
        total = EvalStats()
        for chunk, gt_map in zip(chunks, gt_maps):
            filtered = _run_filter_on_chunk(chunk, strategy, bbox_params, traj_params)
            pm = _build_pred_map(filtered)
            vs = _eval_video(pm, gt_map)
            total = _merge_stats(total, vs)

        metrics = total.to_metrics()
        row = {
            "strategy": strategy,
            "bbox_params": bbox_params,
            "traj_params": traj_params,
            **metrics,
        }
        all_results.append(row)

        score = (metrics["iou_mean"], metrics["success_rate_50"], -metrics["ce_mean"])
        if best_key is None or score > best_key:
            best_key = score
            best_metrics = row

    return {
        "experiment": str(exp_dir),
        "raw_baseline": raw_baseline,
        "baseline": baseline_metrics,
        "best": best_metrics,
        "top5": sorted(all_results, key=lambda r: (r["iou_mean"], r["success_rate_50"], -r["ce_mean"]), reverse=True)[:5],
    }


def build_candidates() -> List[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    return [
        ("none", {}, {"sg_window": 5, "sg_polyorder": 2, "macro_ratio": 0.10, "micro_hw": 3}),
        ("none", {}, {"sg_window": 7, "sg_polyorder": 2, "macro_ratio": 0.10, "micro_hw": 3}),
        ("none", {}, {"sg_window": 7, "sg_polyorder": 3, "macro_ratio": 0.12, "micro_hw": 5}),
        ("hampel_only", {"macro_ratio": 0.10, "micro_hw": 3}, {"sg_window": 7, "sg_polyorder": 2, "macro_ratio": 0.10, "micro_hw": 3}),
        ("hampel_only", {"macro_ratio": 0.12, "micro_hw": 3}, {"sg_window": 7, "sg_polyorder": 2, "macro_ratio": 0.12, "micro_hw": 3}),
        ("hampel_only", {"macro_ratio": 0.20, "micro_hw": 5}, {"sg_window": 7, "sg_polyorder": 2, "macro_ratio": 0.20, "micro_hw": 5}),
        ("independent", {"sg_window": 5, "sg_polyorder": 2, "macro_ratio": 0.10, "micro_hw": 3}, {"sg_window": 7, "sg_polyorder": 2, "macro_ratio": 0.10, "micro_hw": 3}),
        ("independent", {"sg_window": 7, "sg_polyorder": 2, "macro_ratio": 0.12, "micro_hw": 3}, {"sg_window": 7, "sg_polyorder": 2, "macro_ratio": 0.12, "micro_hw": 3}),
        ("independent", {"sg_window": 7, "sg_polyorder": 3, "macro_ratio": 0.12, "micro_hw": 5}, {"sg_window": 7, "sg_polyorder": 3, "macro_ratio": 0.12, "micro_hw": 5}),
        ("area_constraint", {"max_area_change_ratio": 0.20, "sg_window": 7, "sg_polyorder": 2}, {"sg_window": 7, "sg_polyorder": 2, "macro_ratio": 0.12, "micro_hw": 3}),
        ("area_constraint", {"max_area_change_ratio": 0.10, "sg_window": 5, "sg_polyorder": 2}, {"sg_window": 5, "sg_polyorder": 2, "macro_ratio": 0.10, "micro_hw": 3}),
    ]


def iter_experiments(schedule_dirs: List[Path]) -> List[Path]:
    out: List[Path] = []
    for sdir in schedule_dirs:
        if not sdir.exists():
            continue
        for child in sorted(sdir.iterdir()):
            if child.is_dir() and (child / "metadata.json").exists() and (child / "test" / "trajectory_filter").exists():
                out.append(child)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedules", nargs="+", required=True)
    parser.add_argument("--out", default="results/trajectory_filter_tuning_report.json")
    parser.add_argument("--max-exp", type=int, default=0, help="0 means all")
    args = parser.parse_args()

    schedule_dirs = [Path(p) for p in args.schedules]
    experiments = iter_experiments(schedule_dirs)
    if args.max_exp > 0:
        experiments = experiments[: args.max_exp]

    candidates = build_candidates()
    report: Dict[str, Any] = {
        "n_experiments": len(experiments),
        "n_candidates": len(candidates),
        "results": [],
    }

    improved = 0
    for i, exp in enumerate(experiments, start=1):
        print(f"[{i}/{len(experiments)}] tuning {exp.name} ...")
        res = evaluate_experiment(exp, candidates)
        if not res:
            print("  -> skipped (missing/unsupported data layout)")
            continue
        base = res.get("baseline") or {}
        best = res.get("best") or {}
        biou = float(base.get("iou_mean", -1.0)) if base else -1.0
        siou = float(best.get("iou_mean", -1.0)) if best else -1.0
        if siou > biou + 1e-9:
            improved += 1
        print(f"  baseline iou={biou:.4f} -> best iou={siou:.4f}")
        report["results"].append(res)

    report["improved_experiments"] = improved
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nSaved report: {out_path}")
    print(f"Improved experiments: {improved}/{len(report['results'])}")


if __name__ == "__main__":
    main()
