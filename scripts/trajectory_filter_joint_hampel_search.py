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

from tracking.classification.trajectory_filter import filter_detections, compute_trajectory_metrics


@dataclass
class DetAgg:
    n: int = 0
    iou_sum: float = 0.0
    ce_sum: float = 0.0

    def add(self, iou: float, ce: float) -> None:
        self.n += 1
        self.iou_sum += iou
        self.ce_sum += ce

    def means(self) -> Tuple[float, float]:
        if self.n <= 0:
            return 0.0, 0.0
        return self.iou_sum / self.n, self.ce_sum / self.n


@dataclass
class SmoothAgg:
    n_video: int = 0
    jitter_improve_sum: float = 0.0
    smooth_improve_sum: float = 0.0

    def add(self, jitter_improve: float, smooth_improve: float) -> None:
        self.n_video += 1
        self.jitter_improve_sum += jitter_improve
        self.smooth_improve_sum += smooth_improve

    def means(self) -> Tuple[float, float]:
        if self.n_video <= 0:
            return 0.0, 0.0
        return self.jitter_improve_sum / self.n_video, self.smooth_improve_sum / self.n_video


def _load_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _split_chunks(preds: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    out: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []
    prev: Optional[int] = None
    for rec in preds:
        fi = int(rec.get("frame_index", -1))
        if prev is not None and fi < prev and cur:
            out.append(cur)
            cur = []
        cur.append(rec)
        prev = fi
    if cur:
        out.append(cur)
    return out


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


def _iou(p: Tuple[float, float, float, float], g: Tuple[float, float, float, float]) -> float:
    px, py, pw, ph = p
    gx, gy, gw, gh = g
    ix1 = max(px, gx)
    iy1 = max(py, gy)
    ix2 = min(px + pw, gx + gw)
    iy2 = min(py + ph, gy + gh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    uni = pw * ph + gw * gh - inter
    return inter / uni if uni > 0 else 0.0


def _ce(p: Tuple[float, float, float, float], g: Tuple[float, float, float, float]) -> float:
    px, py, pw, ph = p
    gx, gy, gw, gh = g
    pcx = px + pw / 2.0
    pcy = py + ph / 2.0
    gcx = gx + gw / 2.0
    gcy = gy + gh / 2.0
    return float(np.hypot(pcx - gcx, pcy - gcy))


def _eval_candidate(
    chunks: List[List[Dict[str, Any]]],
    gt_maps: List[Dict[int, Tuple[float, float, float, float]]],
    cfg: Dict[str, Any],
) -> Dict[str, float]:
    det = DetAgg()
    sm = SmoothAgg()

    for chunk, gt in zip(chunks, gt_maps):
        fi = np.asarray([int(r["frame_index"]) for r in chunk], dtype=np.int64)
        bb = np.asarray([r["bbox"] for r in chunk], dtype=np.float64)
        x, y, w, h = bb[:, 0], bb[:, 1], bb[:, 2], bb[:, 3]
        cx = x + w / 2.0
        cy = y + h / 2.0
        sc = np.asarray([float(r.get("score") or 0.0) for r in chunk], dtype=np.float64)

        if cfg["name"] == "baseline_file":
            out_fi, out_cx, out_cy, out_w, out_h = fi, cx, cy, w, h
        else:
            out = filter_detections(
                fi,
                cx,
                cy,
                w,
                h,
                sc,
                bbox_strategy=cfg["bbox_strategy"],
                bbox_params=cfg.get("bbox_params", {}),
                traj_params=cfg.get("traj_params", {}),
                skip_hampel=False,
            )
            out_fi = out["frame_indices"]
            out_cx = out["cx"]
            out_cy = out["cy"]
            out_w = out["widths"]
            out_h = out["heights"]

        before = compute_trajectory_metrics(cx, cy, w, h, fi)
        after = compute_trajectory_metrics(out_cx, out_cy, out_w, out_h, out_fi)
        jitter_before = float(before.get("jitter_cx", 0.0) + before.get("jitter_cy", 0.0))
        jitter_after = float(after.get("jitter_cx", 0.0) + after.get("jitter_cy", 0.0))
        smooth_before = float(before.get("smoothness_cx", 0.0) + before.get("smoothness_cy", 0.0))
        smooth_after = float(after.get("smoothness_cx", 0.0) + after.get("smoothness_cy", 0.0))
        sm.add(jitter_before - jitter_after, smooth_before - smooth_after)

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
            det.add(_iou(p, g), _ce(p, g))

    iou_m, ce_m = det.means()
    j_imp, s_imp = sm.means()
    return {
        "iou_mean": float(iou_m),
        "ce_mean": float(ce_m),
        "jitter_improve": float(j_imp),
        "smoothness_improve": float(s_imp),
    }


def build_candidates() -> List[Dict[str, Any]]:
    return [
        {
            "name": "hampel_none_w5_p2_mr010_hw3",
            "bbox_strategy": "none",
            "bbox_params": {},
            "traj_params": {"sg_window": 5, "sg_polyorder": 2, "macro_ratio": 0.10, "micro_hw": 3},
        },
        {
            "name": "hampel_none_w3_p1_mr008_hw2",
            "bbox_strategy": "none",
            "bbox_params": {},
            "traj_params": {"sg_window": 3, "sg_polyorder": 1, "macro_ratio": 0.08, "micro_hw": 2},
        },
        {
            "name": "hampel_none_w5_p2_mr012_hw5",
            "bbox_strategy": "none",
            "bbox_params": {},
            "traj_params": {"sg_window": 5, "sg_polyorder": 2, "macro_ratio": 0.12, "micro_hw": 5},
        },
        {
            "name": "hampel_none_w7_p2_mr015_hw7",
            "bbox_strategy": "none",
            "bbox_params": {},
            "traj_params": {"sg_window": 7, "sg_polyorder": 2, "macro_ratio": 0.15, "micro_hw": 7},
        },
        {
            "name": "hampel_honly_w5_p2_mr010_hw3",
            "bbox_strategy": "hampel_only",
            "bbox_params": {"macro_ratio": 0.10, "micro_hw": 3},
            "traj_params": {"sg_window": 5, "sg_polyorder": 2, "macro_ratio": 0.10, "micro_hw": 3},
        },
        {
            "name": "hampel_honly_w3_p1_mr008_hw2",
            "bbox_strategy": "hampel_only",
            "bbox_params": {"macro_ratio": 0.08, "micro_hw": 2},
            "traj_params": {"sg_window": 3, "sg_polyorder": 1, "macro_ratio": 0.08, "micro_hw": 2},
        },
        {
            "name": "hampel_area_w5_p2_mr010_hw3",
            "bbox_strategy": "area_constraint",
            "bbox_params": {"max_area_change_ratio": 0.10, "sg_window": 5, "sg_polyorder": 2},
            "traj_params": {"sg_window": 5, "sg_polyorder": 2, "macro_ratio": 0.10, "micro_hw": 3},
        },
        {
            "name": "hampel_area_w3_p1_mr008_hw2",
            "bbox_strategy": "area_constraint",
            "bbox_params": {"max_area_change_ratio": 0.10, "sg_window": 3, "sg_polyorder": 1},
            "traj_params": {"sg_window": 3, "sg_polyorder": 1, "macro_ratio": 0.08, "micro_hw": 2},
        },
    ]


def iter_experiments(roots: List[Path]) -> List[Path]:
    out: List[Path] = []
    for r in roots:
        if not r.exists():
            continue
        for d in sorted(r.iterdir()):
            if d.is_dir() and (d / "metadata.json").exists() and (d / "test" / "trajectory_filter" / "filtered_detection_summary.json").exists():
                out.append(d)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedules", nargs="+", required=True)
    ap.add_argument("--out", default="results/trajectory_filter_joint_hampel_report.json")
    args = ap.parse_args()

    roots = [Path(x) for x in args.schedules]
    exps = iter_experiments(roots)
    cands = build_candidates()

    per_exp: List[Dict[str, Any]] = []
    summary_acc: Dict[str, List[Dict[str, float]]] = {c["name"]: [] for c in cands}
    both_counts: Dict[str, int] = {c["name"]: 0 for c in cands}

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
        valid = True
        for rel in test_preview:
            gp = (ds_root / rel).with_suffix(".json")
            if not gp.exists():
                valid = False
                break
            gt_maps.append(_gt_map(gp))
        if not valid:
            continue

        base_fd = _load_json(exp / "test" / "trajectory_filter" / "filtered_detection_summary.json")
        base_m = next(iter(base_fd.values()))
        b_iou = float(base_m.get("iou_mean", 0.0))
        b_ce = float(base_m.get("ce_mean", 0.0))

        exp_row = {
            "experiment": str(exp),
            "baseline": {"iou_mean": b_iou, "ce_mean": b_ce},
            "candidates": {},
        }

        for c in cands:
            m = _eval_candidate(chunks, gt_maps, c)
            di = float(m["iou_mean"] - b_iou)
            dc = float(b_ce - m["ce_mean"])
            dj = float(m["jitter_improve"])
            ds = float(m["smoothness_improve"])
            both = bool(di > 0 and dc > 0 and dj > 0 and ds > 0)
            rec = {
                **m,
                "delta_iou": di,
                "delta_ce_improve": dc,
                "joint_positive": both,
            }
            exp_row["candidates"][c["name"]] = rec
            summary_acc[c["name"]].append(rec)
            if both:
                both_counts[c["name"]] += 1

        per_exp.append(exp_row)
        print(f"[{i}/{len(exps)}] {exp.name}")

    summary: Dict[str, Dict[str, float]] = {}
    for c in cands:
        name = c["name"]
        vals = summary_acc[name]
        if not vals:
            continue
        di = np.asarray([x["delta_iou"] for x in vals], dtype=np.float64)
        dc = np.asarray([x["delta_ce_improve"] for x in vals], dtype=np.float64)
        dj = np.asarray([x["jitter_improve"] for x in vals], dtype=np.float64)
        ds = np.asarray([x["smoothness_improve"] for x in vals], dtype=np.float64)
        n = len(vals)
        summary[name] = {
            "n": float(n),
            "joint_positive_count": float(both_counts[name]),
            "joint_positive_ratio": float(both_counts[name] / n),
            "mean_delta_iou": float(di.mean()),
            "mean_delta_ce_improve": float(dc.mean()),
            "mean_jitter_improve": float(dj.mean()),
            "mean_smoothness_improve": float(ds.mean()),
            "iou_positive_count": float((di > 0).sum()),
            "ce_positive_count": float((dc > 0).sum()),
            "jitter_positive_count": float((dj > 0).sum()),
            "smoothness_positive_count": float((ds > 0).sum()),
        }

    out_obj = {
        "n_experiments": len(per_exp),
        "summary": summary,
        "per_experiment": per_exp,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    print("Saved:", out_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
