from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracking.classification.trajectory_filter import filter_detections


@dataclass
class Agg:
    n: int = 0
    iou_sum: float = 0.0
    ce_sum: float = 0.0

    def add(self, iou: float, ce: float) -> None:
        self.n += 1
        self.iou_sum += iou
        self.ce_sum += ce

    def mean(self) -> Tuple[float, float]:
        if self.n == 0:
            return 0.0, 0.0
        return self.iou_sum / self.n, self.ce_sum / self.n


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _split_chunks(preds: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    chunks: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []
    prev: Optional[int] = None
    for p in preds:
        fi = int(p.get("frame_index", -1))
        if prev is not None and fi < prev and cur:
            chunks.append(cur)
            cur = []
        cur.append(p)
        prev = fi
    if cur:
        chunks.append(cur)
    return chunks


def _gt_map(coco_json: Path) -> Dict[int, Tuple[float, float, float, float]]:
    d = _load_json(coco_json)
    images = {int(x["id"]): int(x.get("frame_index", -1)) for x in d.get("images", [])}
    out: Dict[int, Tuple[float, float, float, float]] = {}
    for ann in d.get("annotations", []):
        iid = int(ann.get("image_id", -1))
        if iid not in images:
            continue
        fi = images[iid]
        if fi in out:
            continue
        b = ann.get("bbox") or []
        if len(b) < 4:
            continue
        out[fi] = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    return out


def _iou(pred: Tuple[float, float, float, float], gt: Tuple[float, float, float, float]) -> float:
    px, py, pw, ph = pred
    gx, gy, gw, gh = gt
    ix1 = max(px, gx)
    iy1 = max(py, gy)
    ix2 = min(px + pw, gx + gw)
    iy2 = min(py + ph, gy + gh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = pw * ph + gw * gh - inter
    return inter / union if union > 0 else 0.0


def _center_error(pred: Tuple[float, float, float, float], gt: Tuple[float, float, float, float]) -> float:
    px, py, pw, ph = pred
    gx, gy, gw, gh = gt
    pcx = px + pw / 2.0
    pcy = py + ph / 2.0
    gcx = gx + gw / 2.0
    gcy = gy + gh / 2.0
    return float(np.hypot(pcx - gcx, pcy - gcy))


def _eval_predictions(
    chunks: List[List[Dict[str, Any]]],
    gt_maps: List[Dict[int, Tuple[float, float, float, float]]],
    *,
    cfg: Dict[str, Any],
) -> Tuple[float, float]:
    agg = Agg()
    for chunk, gt in zip(chunks, gt_maps):
        fi = np.asarray([int(x["frame_index"]) for x in chunk], dtype=np.int64)
        bb = np.asarray([x["bbox"] for x in chunk], dtype=np.float64)
        x, y, w, h = bb[:, 0], bb[:, 1], bb[:, 2], bb[:, 3]
        cx = x + w / 2.0
        cy = y + h / 2.0
        s = np.asarray([float(x.get("score") or 0.0) for x in chunk], dtype=np.float64)

        if cfg["name"] == "raw":
            out_fi = fi
            out_cx = cx
            out_cy = cy
            out_w = w
            out_h = h
        else:
            out = filter_detections(
                fi,
                cx,
                cy,
                w,
                h,
                s,
                bbox_strategy=cfg["bbox_strategy"],
                bbox_params=cfg.get("bbox_params", {}),
                traj_params=cfg.get("traj_params", {}),
                skip_hampel=bool(cfg.get("skip_hampel", False)),
            )
            out_fi = out["frame_indices"]
            out_cx = out["cx"]
            out_cy = out["cy"]
            if bool(cfg.get("filter_bbox_size", True)):
                out_w = out["widths"]
                out_h = out["heights"]
            else:
                m = {int(f): i for i, f in enumerate(fi)}
                idx = np.asarray([m[int(f)] for f in out_fi], dtype=np.int64)
                out_w = w[idx]
                out_h = h[idx]

        pred_map: Dict[int, Tuple[float, float, float, float]] = {}
        for i in range(len(out_fi)):
            f = int(out_fi[i])
            pred_map[f] = (
                float(out_cx[i] - out_w[i] / 2.0),
                float(out_cy[i] - out_h[i] / 2.0),
                float(out_w[i]),
                float(out_h[i]),
            )

        for f, g in gt.items():
            if f not in pred_map:
                continue
            p = pred_map[f]
            agg.add(_iou(p, g), _center_error(p, g))

    return agg.mean()


def build_candidates() -> List[Dict[str, Any]]:
    return [
        {
            "name": "raw",
            "bbox_strategy": "none",
            "bbox_params": {},
            "traj_params": {},
            "skip_hampel": False,
            "filter_bbox_size": True,
        },
        {
            "name": "none_same_traj",
            "bbox_strategy": "none",
            "bbox_params": {},
            "traj_params": None,
            "skip_hampel": False,
            "filter_bbox_size": False,
        },
        {
            "name": "none_same_traj_skip_hampel",
            "bbox_strategy": "none",
            "bbox_params": {},
            "traj_params": None,
            "skip_hampel": True,
            "filter_bbox_size": False,
        },
        {
            "name": "none_mild_no_skip",
            "bbox_strategy": "none",
            "bbox_params": {},
            "traj_params": {"sg_window": 5, "sg_polyorder": 2, "macro_ratio": 0.10, "micro_hw": 3},
            "skip_hampel": False,
            "filter_bbox_size": False,
        },
        {
            "name": "none_mild_skip_hampel",
            "bbox_strategy": "none",
            "bbox_params": {},
            "traj_params": {"sg_window": 5, "sg_polyorder": 2, "macro_ratio": 0.10, "micro_hw": 3},
            "skip_hampel": True,
            "filter_bbox_size": False,
        },
        {
            "name": "none_light_skip_hampel",
            "bbox_strategy": "none",
            "bbox_params": {},
            "traj_params": {"sg_window": 3, "sg_polyorder": 1, "macro_ratio": 0.08, "micro_hw": 2},
            "skip_hampel": True,
            "filter_bbox_size": False,
        },
    ]


def iter_experiments(roots: List[Path]) -> List[Path]:
    exps: List[Path] = []
    for r in roots:
        if not r.exists():
            continue
        for d in sorted(r.iterdir()):
            if not d.is_dir():
                continue
            if (d / "metadata.json").exists() and (d / "test" / "trajectory_filter" / "filtered_detection_summary.json").exists():
                exps.append(d)
    return exps


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--schedules", nargs="+", required=True)
    p.add_argument("--out", default="results/trajectory_filter_joint_report.json")
    args = p.parse_args()

    roots = [Path(x) for x in args.schedules]
    exps = iter_experiments(roots)
    cands = build_candidates()

    summary: Dict[str, Dict[str, float]] = {}
    rows: List[Dict[str, Any]] = []

    by_name: Dict[str, List[Tuple[float, float]]] = {c["name"]: [] for c in cands}
    both_better: Dict[str, int] = {c["name"]: 0 for c in cands}

    for i, exp in enumerate(exps, start=1):
        meta = _load_json(exp / "metadata.json")
        ds_root = Path(meta["dataset"]["root"])
        test_preview = list(meta["dataset"].get("test_preview", []))

        pred_file = next((exp / "test" / "detection" / "predictions").glob("*.json"), None)
        if pred_file is None:
            continue
        preds = _load_json(pred_file)
        chunks = _split_chunks(preds)
        if len(chunks) != len(test_preview):
            continue

        gt_maps = []
        ok = True
        for rel in test_preview:
            g = (ds_root / rel).with_suffix(".json")
            if not g.exists():
                ok = False
                break
            gt_maps.append(_gt_map(g))
        if not ok:
            continue

        fd = _load_json(exp / "test" / "trajectory_filter" / "filtered_detection_summary.json")
        baseline = next(iter(fd.values()))
        b_iou = float(baseline.get("iou_mean", 0.0))
        b_ce = float(baseline.get("ce_mean", 0.0))

        exp_row: Dict[str, Any] = {
            "experiment": str(exp),
            "baseline": {"iou_mean": b_iou, "ce_mean": b_ce},
            "candidates": {},
        }

        for c in cands:
            cfg = dict(c)
            if cfg.get("traj_params") is None:
                run_cfg = _load_json(exp / "test" / "trajectory_filter" / "summary.json").get("config", {})
                cfg["traj_params"] = dict(run_cfg.get("traj_params") or {})

            iou_m, ce_m = _eval_predictions(chunks, gt_maps, cfg=cfg)
            d_iou = iou_m - b_iou
            d_ce = b_ce - ce_m
            exp_row["candidates"][c["name"]] = {
                "iou_mean": iou_m,
                "ce_mean": ce_m,
                "delta_iou": d_iou,
                "delta_ce_improve": d_ce,
                "both_improved": bool(d_iou > 0 and d_ce > 0),
            }
            by_name[c["name"]].append((d_iou, d_ce))
            if d_iou > 0 and d_ce > 0:
                both_better[c["name"]] += 1

        rows.append(exp_row)
        print(f"[{i}/{len(exps)}] {exp.name}")

    for c in cands:
        name = c["name"]
        arr = by_name[name]
        if not arr:
            continue
        d_i = np.asarray([x[0] for x in arr], dtype=np.float64)
        d_c = np.asarray([x[1] for x in arr], dtype=np.float64)
        summary[name] = {
            "n": float(len(arr)),
            "both_improved_count": float(both_better[name]),
            "both_improved_ratio": float(both_better[name] / len(arr)),
            "mean_delta_iou": float(d_i.mean()),
            "median_delta_iou": float(np.median(d_i)),
            "mean_delta_ce_improve": float(d_c.mean()),
            "median_delta_ce_improve": float(np.median(d_c)),
            "iou_improved_count": float((d_i > 0).sum()),
            "ce_improved_count": float((d_c > 0).sum()),
        }

    out = {
        "n_experiments": len(rows),
        "summary": summary,
        "per_experiment": rows,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("Saved:", out_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
