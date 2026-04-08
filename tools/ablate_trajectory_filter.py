from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tracking.classification.trajectory_filter import (  # noqa: E402
    _adaptive_savgol,
    filter_bbox_hampel_only,
    hampel_then_pchip_1d,
)


def _load_gt_bboxes(annotation_path: Path) -> Dict[int, list[float]]:
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    image_to_frame = {
        int(img["id"]): int(img["frame_index"])
        for img in (data.get("images") or [])
        if "id" in img and "frame_index" in img
    }
    gt: Dict[int, list[float]] = {}
    for ann in data.get("annotations") or []:
        try:
            fi = image_to_frame[int(ann["image_id"])]
            gt[fi] = list(map(float, ann["bbox"]))
        except Exception:
            continue
    return gt


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, aw, ah = map(float, a)
    bx1, by1, bw, bh = map(float, b)
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def _center_error(box: np.ndarray, gt: np.ndarray) -> float:
    cx = float(box[0] + box[2] / 2.0)
    cy = float(box[1] + box[3] / 2.0)
    gcx = float(gt[0] + gt[2] / 2.0)
    gcy = float(gt[1] + gt[3] / 2.0)
    return float(((cx - gcx) ** 2 + (cy - gcy) ** 2) ** 0.5)


def _build_observed_mask(rows: list[dict[str, Any]], bbox: np.ndarray) -> np.ndarray:
    n = len(rows)
    return np.array(
        [
            bool(np.isfinite(bbox[i]).all()) and float(bbox[i][2]) > 0.0 and float(bbox[i][3]) > 0.0
            for i in range(n)
        ],
        dtype=bool,
    )


def _box_from_center(cx: np.ndarray, cy: np.ndarray, w: np.ndarray, h: np.ndarray) -> np.ndarray:
    return np.column_stack([cx - w / 2.0, cy - h / 2.0, w, h]).astype(np.float64)


@dataclass
class StageEval:
    name: str
    iou_mean: float
    ce_mean: float
    delta_iou_vs_raw: float
    delta_ce_vs_raw: float
    worse_iou_count: int
    worse_ce_count: int


def _eval_stage(
    fi: np.ndarray,
    boxes: np.ndarray,
    raw_boxes: np.ndarray,
    gt_map: Dict[int, list[float]],
) -> Tuple[StageEval, Dict[int, Dict[str, float]]]:
    raw_iou, raw_ce, st_iou, st_ce = [], [], [], []
    frame_stats: Dict[int, Dict[str, float]] = {}
    worse_iou = 0
    worse_ce = 0

    for i, f in enumerate(fi.tolist()):
        gt = gt_map.get(f)
        if gt is None:
            continue
        gt_arr = np.array(gt, dtype=np.float64)
        rb = raw_boxes[i]
        sb = boxes[i]

        ri = _bbox_iou(rb, gt_arr)
        rc = _center_error(rb, gt_arr)
        si = _bbox_iou(sb, gt_arr)
        sc = _center_error(sb, gt_arr)

        di = si - ri
        dc = sc - rc

        if di < 0.0:
            worse_iou += 1
        if dc > 0.0:
            worse_ce += 1

        raw_iou.append(ri)
        raw_ce.append(rc)
        st_iou.append(si)
        st_ce.append(sc)
        frame_stats[int(f)] = {
            "raw_iou": float(ri),
            "raw_ce": float(rc),
            "stage_iou": float(si),
            "stage_ce": float(sc),
            "diou": float(di),
            "dce": float(dc),
        }

    if not st_iou:
        ev = StageEval(
            name="",
            iou_mean=0.0,
            ce_mean=0.0,
            delta_iou_vs_raw=0.0,
            delta_ce_vs_raw=0.0,
            worse_iou_count=0,
            worse_ce_count=0,
        )
        return ev, frame_stats

    ev = StageEval(
        name="",
        iou_mean=float(np.mean(st_iou)),
        ce_mean=float(np.mean(st_ce)),
        delta_iou_vs_raw=float(np.mean(st_iou) - np.mean(raw_iou)),
        delta_ce_vs_raw=float(np.mean(st_ce) - np.mean(raw_ce)),
        worse_iou_count=int(worse_iou),
        worse_ce_count=int(worse_ce),
    )
    return ev, frame_stats


def _run_pipeline(
    fi: np.ndarray,
    bbox: np.ndarray,
    observed_mask: np.ndarray,
    anchor_mask: np.ndarray,
    *,
    macro_ratio: float,
    macro_sigma: float,
    micro_hw: int,
    micro_sigma: float,
    max_outlier_run: int,
    sg_window: int,
    sg_polyorder: int,
    ablate_sg_window: int,
    ablate_sg_polyorder: int,
    anchor_keep_ratio: float,
) -> Dict[str, np.ndarray]:
    raw_cx = bbox[:, 0] + bbox[:, 2] / 2.0
    raw_cy = bbox[:, 1] + bbox[:, 3] / 2.0
    raw_w = bbox[:, 2]
    raw_h = bbox[:, 3]

    params = {
        "macro_ratio": float(macro_ratio),
        "macro_sigma": float(macro_sigma),
        "micro_hw": int(micro_hw),
        "micro_sigma": float(micro_sigma),
        "max_outlier_run": int(max_outlier_run),
    }

    cx_pchip, cx_marked, cx_mask = hampel_then_pchip_1d(raw_cx, fi, observed_mask=observed_mask, **params)
    cy_pchip, cy_marked, cy_mask = hampel_then_pchip_1d(raw_cy, fi, observed_mask=observed_mask, **params)

    # SG on Hampel+PCHIP output for ablation-only stage comparison.
    cx_sg = _adaptive_savgol(
        cx_pchip,
        fi,
        window_length=int(ablate_sg_window),
        polyorder=int(ablate_sg_polyorder),
    )
    cy_sg = _adaptive_savgol(
        cy_pchip,
        fi,
        window_length=int(ablate_sg_window),
        polyorder=int(ablate_sg_polyorder),
    )

    # SG-only ablation (skip Hampel, no interpolation effect because all observed)
    cx_sg_only = _adaptive_savgol(
        raw_cx,
        fi,
        window_length=int(ablate_sg_window),
        polyorder=int(ablate_sg_polyorder),
    )
    cy_sg_only = _adaptive_savgol(
        raw_cy,
        fi,
        window_length=int(ablate_sg_window),
        polyorder=int(ablate_sg_polyorder),
    )

    w_hampel, h_hampel = filter_bbox_hampel_only(
        raw_w,
        raw_h,
        frame_indices=fi,
        observed_mask=observed_mask,
        **params,
    )

    # Anchor blend on final full stage only
    keep = float(np.clip(anchor_keep_ratio, 0.0, 1.0))
    cx_full = _adaptive_savgol(cx_pchip, fi, window_length=int(sg_window), polyorder=int(sg_polyorder))
    cy_full = _adaptive_savgol(cy_pchip, fi, window_length=int(sg_window), polyorder=int(sg_polyorder))
    w_full = w_hampel.copy()
    h_full = h_hampel.copy()
    if keep > 0.0 and anchor_mask.any():
        cx_full[anchor_mask] = keep * raw_cx[anchor_mask] + (1.0 - keep) * cx_full[anchor_mask]
        cy_full[anchor_mask] = keep * raw_cy[anchor_mask] + (1.0 - keep) * cy_full[anchor_mask]
        w_full[anchor_mask] = keep * raw_w[anchor_mask] + (1.0 - keep) * w_full[anchor_mask]
        h_full[anchor_mask] = keep * raw_h[anchor_mask] + (1.0 - keep) * h_full[anchor_mask]

    return {
        "raw_cx": raw_cx,
        "raw_cy": raw_cy,
        "raw_w": raw_w,
        "raw_h": raw_h,
        "cx_marked": cx_marked,
        "cy_marked": cy_marked,
        "cx_mask": cx_mask,
        "cy_mask": cy_mask,
        "cx_pchip": cx_pchip,
        "cy_pchip": cy_pchip,
        "cx_sg": cx_sg,
        "cy_sg": cy_sg,
        "cx_sg_only": cx_sg_only,
        "cy_sg_only": cy_sg_only,
        "w_hampel": w_hampel,
        "h_hampel": h_hampel,
        "cx_full": cx_full,
        "cy_full": cy_full,
        "w_full": w_full,
        "h_full": h_full,
    }


def _export_raw_gt(
    out_csv: Path,
    fi: np.ndarray,
    rows: list[dict[str, Any]],
    raw_boxes: np.ndarray,
    gt_map: Dict[int, list[float]],
) -> None:
    idx_to_row = {int(r["frame_index"]): r for r in rows}
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "frame_index",
            "fallback",
            "score",
            "raw_x",
            "raw_y",
            "raw_w",
            "raw_h",
            "gt_x",
            "gt_y",
            "gt_w",
            "gt_h",
            "raw_iou",
            "raw_ce",
        ])
        for i, frame in enumerate(fi.tolist()):
            r = idx_to_row.get(int(frame), {})
            gt = gt_map.get(int(frame))
            if gt is None:
                continue
            rb = raw_boxes[i]
            ga = np.array(gt, dtype=np.float64)
            w.writerow([
                int(frame),
                int(bool(r.get("fallback", False) or r.get("is_fallback", False))),
                float(r.get("score", 0.0) or 0.0),
                float(rb[0]),
                float(rb[1]),
                float(rb[2]),
                float(rb[3]),
                float(ga[0]),
                float(ga[1]),
                float(ga[2]),
                float(ga[3]),
                float(_bbox_iou(rb, ga)),
                float(_center_error(rb, ga)),
            ])


def _stage_boxes(p: Dict[str, np.ndarray], name: str) -> np.ndarray:
    if name == "raw":
        return _box_from_center(p["raw_cx"], p["raw_cy"], p["raw_w"], p["raw_h"])
    if name == "hampel_pchip":
        return _box_from_center(p["cx_pchip"], p["cy_pchip"], p["w_hampel"], p["h_hampel"])
    if name == "hampel_pchip_sg":
        return _box_from_center(p["cx_sg"], p["cy_sg"], p["w_hampel"], p["h_hampel"])
    if name == "sg_only":
        return _box_from_center(p["cx_sg_only"], p["cy_sg_only"], p["raw_w"], p["raw_h"])
    if name == "full":
        return _box_from_center(p["cx_full"], p["cy_full"], p["w_full"], p["h_full"])
    raise ValueError(name)


def _search_best(
    fi: np.ndarray,
    bbox: np.ndarray,
    observed_mask: np.ndarray,
    anchor_mask: np.ndarray,
    gt_map: Dict[int, list[float]],
    key_frames: List[int],
) -> Dict[str, Any]:
    raw_boxes = bbox.copy().astype(np.float64)
    best: Optional[Dict[str, Any]] = None

    # Focused search grid: fast enough for interactive diagnosis while still
    # covering meaningful Hampel/PCHIP/SG/anchor combinations.
    for mr, ms, mh, mis, mor, sgw, akr in product(
        [0.04, 0.06, 0.08],
        [1.5, 2.0, 2.5],
        [1, 2],
        [1.5, 2.0],
        [1, 2],
        [5, 7],
        [0.6, 0.8, 1.0],
    ):
        p = _run_pipeline(
            fi,
            bbox,
            observed_mask,
            anchor_mask,
            macro_ratio=mr,
            macro_sigma=ms,
            micro_hw=mh,
            micro_sigma=mis,
            max_outlier_run=mor,
            sg_window=sgw,
            sg_polyorder=2,
            ablate_sg_window=5,
            ablate_sg_polyorder=2,
            anchor_keep_ratio=akr,
        )
        full_boxes = _stage_boxes(p, "full")
        ev, frame_stats = _eval_stage(fi, full_boxes, raw_boxes, gt_map)

        key_bad = False
        key_penalty = 0.0
        for f in key_frames:
            st = frame_stats.get(int(f))
            if st is None:
                continue
            di = float(st["diou"])
            dc = float(st["dce"])
            # hard reject catastrophic drift
            if di < -0.01 or dc > 0.5:
                key_bad = True
            key_penalty += max(0.0, -di) * 10.0 + max(0.0, dc)

        if key_bad:
            continue

        score = ev.delta_iou_vs_raw * 100.0 - max(0.0, ev.delta_ce_vs_raw) - key_penalty * 0.01

        cand = {
            "score": float(score),
            "params": {
                "macro_ratio": float(mr),
                "macro_sigma": float(ms),
                "micro_hw": int(mh),
                "micro_sigma": float(mis),
                "max_outlier_run": int(mor),
                "sg_window": int(sgw),
                "sg_polyorder": 2,
                "anchor_keep_ratio": float(akr),
            },
            "delta_iou": float(ev.delta_iou_vs_raw),
            "delta_ce": float(ev.delta_ce_vs_raw),
            "worse_iou": int(ev.worse_iou_count),
            "worse_ce": int(ev.worse_ce_count),
        }
        if best is None or cand["score"] > best["score"]:
            best = cand

    if best is None:
        # Fallback: keep safe no-regression setting
        best = {
            "score": 0.0,
            "params": {
                "macro_ratio": 0.06,
                "macro_sigma": 2.0,
                "micro_hw": 1,
                "micro_sigma": 2.0,
                "max_outlier_run": 1,
                "sg_window": 5,
                "sg_polyorder": 2,
                "anchor_keep_ratio": 1.0,
            },
            "delta_iou": 0.0,
            "delta_ce": 0.0,
            "worse_iou": 0,
            "worse_ce": 0,
            "fallback": True,
        }

    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--video", default="R2")
    ap.add_argument("--subject", default="n003")
    ap.add_argument("--model", default="YOLOv11")
    ap.add_argument("--dataset-root", default=r"C:\Users\User\Desktop\code\Traking\dataset\merged_extend")
    ap.add_argument("--macro-ratio", type=float, default=0.06)
    ap.add_argument("--macro-sigma", type=float, default=2.0)
    ap.add_argument("--micro-hw", type=int, default=1)
    ap.add_argument("--micro-sigma", type=float, default=2.0)
    ap.add_argument("--max-outlier-run", type=int, default=1)
    ap.add_argument("--sg-window", type=int, default=5)
    ap.add_argument("--sg-polyorder", type=int, default=2)
    ap.add_argument("--ablate-sg-window", type=int, default=5)
    ap.add_argument("--ablate-sg-polyorder", type=int, default=2)
    ap.add_argument("--anchor-keep-ratio", type=float, default=1.0)
    ap.add_argument("--key-frames", default="104,156,166")
    args = ap.parse_args()

    key_frames = [int(x.strip()) for x in str(args.key_frames).split(",") if x.strip()]

    base = Path(args.results_dir)
    pred_path = base / "test" / "detection" / "predictions_by_video" / args.subject / args.video / f"{args.model}.json"
    ann_path = Path(args.dataset_root) / args.subject / f"{args.video}.json"
    out_dir = base / "analysis" / "trajectory_ablation" / args.subject / args.video
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        rows = json.loads(pred_path.read_text(encoding="utf-8"))
        gt_map = _load_gt_bboxes(ann_path)

        fi = np.array([int(r["frame_index"]) for r in rows], dtype=np.int64)
        bbox = np.array([r["bbox"] for r in rows], dtype=np.float64)
        observed_mask = _build_observed_mask(rows, bbox)
        anchor_mask = np.array([not bool(r.get("fallback", False) or r.get("is_fallback", False)) for r in rows], dtype=bool)

        raw_boxes = bbox.copy().astype(np.float64)
        _export_raw_gt(out_dir / "raw_vs_gt.csv", fi, rows, raw_boxes, gt_map)

        # Baseline pipeline ablation with current args
        p = _run_pipeline(
            fi,
            bbox,
            observed_mask,
            anchor_mask,
            macro_ratio=args.macro_ratio,
            macro_sigma=args.macro_sigma,
            micro_hw=args.micro_hw,
            micro_sigma=args.micro_sigma,
            max_outlier_run=args.max_outlier_run,
            sg_window=args.sg_window,
            sg_polyorder=args.sg_polyorder,
            ablate_sg_window=args.ablate_sg_window,
            ablate_sg_polyorder=args.ablate_sg_polyorder,
            anchor_keep_ratio=args.anchor_keep_ratio,
        )

        stages = ["raw", "hampel_pchip", "hampel_pchip_sg", "sg_only", "full"]
        stage_summary: Dict[str, Any] = {}
        key_frame_summary: Dict[str, Any] = {}

        for st in stages:
            boxes = _stage_boxes(p, st)
            ev, frame_stats = _eval_stage(fi, boxes, raw_boxes, gt_map)
            ev.name = st
            stage_summary[st] = {
                "iou_mean": ev.iou_mean,
                "ce_mean": ev.ce_mean,
                "delta_iou_vs_raw": ev.delta_iou_vs_raw,
                "delta_ce_vs_raw": ev.delta_ce_vs_raw,
                "worse_iou_count": ev.worse_iou_count,
                "worse_ce_count": ev.worse_ce_count,
            }
            key_frame_summary[st] = {str(f): frame_stats.get(int(f), {}) for f in key_frames}

        # Search best full-pipeline process under key-frame anti-drift constraints
        best = _search_best(fi, bbox, observed_mask, anchor_mask, gt_map, key_frames)

        out = {
            "video": args.video,
            "subject": args.subject,
            "model": args.model,
            "key_frames": key_frames,
            "current_params": {
                "macro_ratio": args.macro_ratio,
                "macro_sigma": args.macro_sigma,
                "micro_hw": args.micro_hw,
                "micro_sigma": args.micro_sigma,
                "max_outlier_run": args.max_outlier_run,
                "sg_window": args.sg_window,
                "sg_polyorder": args.sg_polyorder,
                "ablate_sg_window": args.ablate_sg_window,
                "ablate_sg_polyorder": args.ablate_sg_polyorder,
                "anchor_keep_ratio": args.anchor_keep_ratio,
            },
            "stage_ablation": stage_summary,
            "key_frame_ablation": key_frame_summary,
            "best_pipeline": best,
            "raw_gt_csv": str((out_dir / "raw_vs_gt.csv").as_posix()),
        }

        (out_dir / "ablation_summary.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(json.dumps(out, ensure_ascii=False, indent=2))
    except Exception:
        (out_dir / "ablation_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
