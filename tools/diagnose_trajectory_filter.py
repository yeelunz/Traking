from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tracking.classification.trajectory_filter import (
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


def _parse_int_list(raw: str) -> list[int]:
    out: list[int] = []
    for token in str(raw).split(","):
        t = token.strip()
        if not t:
            continue
        out.append(int(t))
    return sorted(set(out))


def _parse_float_list(raw: str) -> list[float]:
    out: list[float] = []
    for token in str(raw).split(","):
        t = token.strip()
        if not t:
            continue
        out.append(float(t))
    return sorted(set(out))


def _build_observed_mask(rows: list[dict[str, Any]], bbox: np.ndarray, policy: str) -> np.ndarray:
    n = len(rows)
    if policy == "none":
        return np.ones(n, dtype=bool)
    if policy == "fallback_as_missing":
        return np.array(
            [not bool(r.get("fallback", False) or r.get("is_fallback", False)) for r in rows],
            dtype=bool,
        )
    # invalid_bbox: only explicitly invalid bbox rows are missing
    return np.array(
        [
            bool(np.isfinite(bbox[i]).all()) and float(bbox[i][2]) > 0.0 and float(bbox[i][3]) > 0.0
            for i in range(n)
        ],
        dtype=bool,
    )


def _evaluate_config(
    fi: np.ndarray,
    bbox: np.ndarray,
    observed_mask: np.ndarray,
    anchor_mask: np.ndarray,
    gt_map: Dict[int, list[float]],
    *,
    macro_ratio: float,
    macro_sigma: float,
    micro_hw: int,
    micro_sigma: float,
    max_outlier_run: int,
    anchor_keep_ratio: float,
    sg_window: int,
    sg_polyorder: int,
) -> Dict[str, Any]:
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
    cx_final = _adaptive_savgol(cx_pchip, fi, window_length=int(sg_window), polyorder=int(sg_polyorder))
    cy_final = _adaptive_savgol(cy_pchip, fi, window_length=int(sg_window), polyorder=int(sg_polyorder))
    w_final, h_final = filter_bbox_hampel_only(
        raw_w,
        raw_h,
        frame_indices=fi,
        observed_mask=observed_mask,
        **params,
    )

    keep = float(np.clip(anchor_keep_ratio, 0.0, 1.0))
    if keep > 0.0 and anchor_mask.any():
        cx_final[anchor_mask] = keep * raw_cx[anchor_mask] + (1.0 - keep) * cx_final[anchor_mask]
        cy_final[anchor_mask] = keep * raw_cy[anchor_mask] + (1.0 - keep) * cy_final[anchor_mask]
        w_final[anchor_mask] = keep * raw_w[anchor_mask] + (1.0 - keep) * w_final[anchor_mask]
        h_final[anchor_mask] = keep * raw_h[anchor_mask] + (1.0 - keep) * h_final[anchor_mask]

    raw_ious, raw_ces, final_ious, final_ces = [], [], [], []
    for i, frame in enumerate(fi.tolist()):
        gt = gt_map.get(frame)
        if gt is None:
            continue
        gt_arr = np.array(gt, dtype=np.float64)
        raw_box = bbox[i]
        final_box = np.array(
            [
                cx_final[i] - w_final[i] / 2.0,
                cy_final[i] - h_final[i] / 2.0,
                w_final[i],
                h_final[i],
            ],
            dtype=np.float64,
        )
        raw_ious.append(_bbox_iou(raw_box, gt_arr))
        raw_ces.append(_center_error(raw_box, gt_arr))
        final_ious.append(_bbox_iou(final_box, gt_arr))
        final_ces.append(_center_error(final_box, gt_arr))

    return {
        "macro_ratio": float(macro_ratio),
        "macro_sigma": float(macro_sigma),
        "micro_hw": int(micro_hw),
        "micro_sigma": float(micro_sigma),
        "max_outlier_run": int(max_outlier_run),
        "anchor_keep_ratio": float(anchor_keep_ratio),
        "sg_window": int(sg_window),
        "sg_polyorder": int(sg_polyorder),
        "cx_marked": cx_marked,
        "cy_marked": cy_marked,
        "cx_pchip": cx_pchip,
        "cy_pchip": cy_pchip,
        "cx_final": cx_final,
        "cy_final": cy_final,
        "w_final": w_final,
        "h_final": h_final,
        "cx_mask": cx_mask,
        "cy_mask": cy_mask,
        "raw_iou_mean": float(np.mean(raw_ious)) if raw_ious else None,
        "raw_ce_mean": float(np.mean(raw_ces)) if raw_ces else None,
        "final_iou_mean": float(np.mean(final_ious)) if final_ious else None,
        "final_ce_mean": float(np.mean(final_ces)) if final_ces else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--video", required=True, help="video stem, e.g. R1")
    ap.add_argument("--subject", default="n003")
    ap.add_argument("--model", default="YOLOv11")
    ap.add_argument("--dataset-root", default=r"C:\Users\User\Desktop\code\Traking\dataset\merged_extend")
    ap.add_argument("--macro-ratio", type=float, default=0.08)
    ap.add_argument("--macro-sigma", type=float, default=1.5)
    ap.add_argument("--micro-hw", type=int, default=7)
    ap.add_argument("--micro-sigma", type=float, default=1.5)
    ap.add_argument("--max-outlier-run", type=int, default=2)
    ap.add_argument("--anchor-keep-ratio", type=float, default=0.0)
    ap.add_argument("--sg-window", type=int, default=1)
    ap.add_argument("--sg-polyorder", type=int, default=1)
    ap.add_argument(
        "--missing-policy",
        choices=["invalid_bbox", "none", "fallback_as_missing"],
        default="invalid_bbox",
        help="如何決定缺失值: invalid_bbox(預設) / none / fallback_as_missing(舊流程)",
    )
    ap.add_argument("--sweep", action="store_true", help="對 Hampel+SG 參數做網格搜尋")
    ap.add_argument("--sweep-macro-ratio", default="0.02,0.04,0.06,0.08,0.1")
    ap.add_argument("--sweep-macro-sigma", default="1.0,1.5,2.0,2.5,3.0")
    ap.add_argument("--sweep-micro-hw", default="1,2,3,5,7,9")
    ap.add_argument("--sweep-micro-sigma", default="1.0,1.5,2.0,2.5,3.0")
    ap.add_argument("--sweep-sg-window", default="5,7,9,11,13")
    ap.add_argument("--sweep-sg-polyorder", default="2")
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()

    base = Path(args.results_dir)
    pred_path = base / "test" / "detection" / "predictions_by_video" / args.subject / args.video / f"{args.model}.json"
    ann_path = Path(args.dataset_root) / args.subject / f"{args.video}.json"
    out_dir = base / "analysis" / "trajectory_diagnostics" / args.subject / args.video
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = json.loads(pred_path.read_text(encoding="utf-8"))
    gt_map = _load_gt_bboxes(ann_path)

    fi = np.array([int(r["frame_index"]) for r in rows], dtype=np.int64)
    bbox = np.array([r["bbox"] for r in rows], dtype=np.float64)
    observed_mask = _build_observed_mask(rows, bbox, args.missing_policy)
    anchor_mask = np.array([not bool(r.get("fallback", False) or r.get("is_fallback", False)) for r in rows], dtype=bool)

    best_eval: Optional[Dict[str, Any]] = None
    if args.sweep:
        macro_ratio_list = _parse_float_list(args.sweep_macro_ratio)
        macro_sigma_list = _parse_float_list(args.sweep_macro_sigma)
        micro_hw_list = _parse_int_list(args.sweep_micro_hw)
        micro_sigma_list = _parse_float_list(args.sweep_micro_sigma)
        sg_window_list = [w for w in _parse_int_list(args.sweep_sg_window) if w >= 3 and (w % 2 == 1)]
        sg_polyorder_list = _parse_int_list(args.sweep_sg_polyorder)

        sweep_rows: list[dict[str, Any]] = []
        for m_ratio, m_sigma, mi_hw, mi_sigma, sg_w, sg_po in product(
            macro_ratio_list,
            macro_sigma_list,
            micro_hw_list,
            micro_sigma_list,
            sg_window_list,
            sg_polyorder_list,
        ):
            if sg_po >= sg_w:
                continue
            ev = _evaluate_config(
                fi,
                bbox,
                observed_mask,
                anchor_mask,
                gt_map,
                macro_ratio=m_ratio,
                macro_sigma=m_sigma,
                micro_hw=mi_hw,
                micro_sigma=mi_sigma,
                max_outlier_run=args.max_outlier_run,
                anchor_keep_ratio=args.anchor_keep_ratio,
                sg_window=sg_w,
                sg_polyorder=sg_po,
            )
            row = {
                "macro_ratio": ev["macro_ratio"],
                "macro_sigma": ev["macro_sigma"],
                "micro_hw": ev["micro_hw"],
                "micro_sigma": ev["micro_sigma"],
                "max_outlier_run": ev["max_outlier_run"],
                "anchor_keep_ratio": ev["anchor_keep_ratio"],
                "sg_window": ev["sg_window"],
                "sg_polyorder": ev["sg_polyorder"],
                "raw_iou_mean": ev["raw_iou_mean"],
                "raw_ce_mean": ev["raw_ce_mean"],
                "final_iou_mean": ev["final_iou_mean"],
                "final_ce_mean": ev["final_ce_mean"],
                "delta_iou": None if ev["raw_iou_mean"] is None else float(ev["final_iou_mean"] - ev["raw_iou_mean"]),
                "delta_ce": None if ev["raw_ce_mean"] is None else float(ev["final_ce_mean"] - ev["raw_ce_mean"]),
            }
            sweep_rows.append(row)
            if best_eval is None:
                best_eval = ev
            else:
                cur_ce = float(ev["final_ce_mean"] if ev["final_ce_mean"] is not None else 1e18)
                best_ce = float(best_eval["final_ce_mean"] if best_eval["final_ce_mean"] is not None else 1e18)
                cur_iou = float(ev["final_iou_mean"] if ev["final_iou_mean"] is not None else -1e18)
                best_iou = float(best_eval["final_iou_mean"] if best_eval["final_iou_mean"] is not None else -1e18)
                if (cur_ce < best_ce) or (abs(cur_ce - best_ce) < 1e-9 and cur_iou > best_iou):
                    best_eval = ev

        sweep_rows_sorted = sorted(
            sweep_rows,
            key=lambda r: (
                float(r["final_ce_mean"] if r["final_ce_mean"] is not None else 1e18),
                -float(r["final_iou_mean"] if r["final_iou_mean"] is not None else -1e18),
            ),
        )

        sweep_rows_both = [
            r for r in sweep_rows_sorted
            if (r["delta_ce"] is not None and r["delta_iou"] is not None and float(r["delta_ce"]) < 0.0 and float(r["delta_iou"]) > 0.0)
        ]
        if sweep_rows_both:
            best = sweep_rows_both[0]
            best_eval = _evaluate_config(
                fi,
                bbox,
                observed_mask,
                anchor_mask,
                gt_map,
                macro_ratio=best["macro_ratio"],
                macro_sigma=best["macro_sigma"],
                micro_hw=best["micro_hw"],
                micro_sigma=best["micro_sigma"],
                max_outlier_run=best["max_outlier_run"],
                anchor_keep_ratio=best["anchor_keep_ratio"],
                sg_window=best["sg_window"],
                sg_polyorder=best["sg_polyorder"],
            )
        else:
            best = sweep_rows_sorted[0]
            best_eval = _evaluate_config(
                fi,
                bbox,
                observed_mask,
                anchor_mask,
                gt_map,
                macro_ratio=best["macro_ratio"],
                macro_sigma=best["macro_sigma"],
                micro_hw=best["micro_hw"],
                micro_sigma=best["micro_sigma"],
                max_outlier_run=best["max_outlier_run"],
                anchor_keep_ratio=best["anchor_keep_ratio"],
                sg_window=best["sg_window"],
                sg_polyorder=best["sg_polyorder"],
            )

        sweep_csv = out_dir / "sweep_results.csv"
        with sweep_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "macro_ratio",
                    "macro_sigma",
                    "micro_hw",
                    "micro_sigma",
                    "max_outlier_run",
                    "anchor_keep_ratio",
                    "sg_window",
                    "sg_polyorder",
                    "raw_iou_mean",
                    "raw_ce_mean",
                    "final_iou_mean",
                    "final_ce_mean",
                    "delta_iou",
                    "delta_ce",
                ],
            )
            writer.writeheader()
            for row in sweep_rows_sorted:
                writer.writerow(row)

        topk = max(1, int(args.top_k))
        (out_dir / "sweep_topk.json").write_text(
            json.dumps(sweep_rows_sorted[:topk], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (out_dir / "sweep_topk_both_improved.json").write_text(
            json.dumps(sweep_rows_both[:topk], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        best_eval = _evaluate_config(
            fi,
            bbox,
            observed_mask,
            anchor_mask,
            gt_map,
            macro_ratio=args.macro_ratio,
            macro_sigma=args.macro_sigma,
            micro_hw=args.micro_hw,
            micro_sigma=args.micro_sigma,
            max_outlier_run=args.max_outlier_run,
            anchor_keep_ratio=args.anchor_keep_ratio,
            sg_window=args.sg_window,
            sg_polyorder=args.sg_polyorder,
        )

    if best_eval is None:
        raise RuntimeError("No valid evaluation result was generated.")

    cx_marked = best_eval["cx_marked"]
    cy_marked = best_eval["cy_marked"]
    cx_pchip = best_eval["cx_pchip"]
    cy_pchip = best_eval["cy_pchip"]
    cx_final = best_eval["cx_final"]
    cy_final = best_eval["cy_final"]
    w_final = best_eval["w_final"]
    h_final = best_eval["h_final"]
    cx_mask = best_eval["cx_mask"]
    cy_mask = best_eval["cy_mask"]

    gt_frames = [idx for idx in fi.tolist() if idx in gt_map]
    raw_ious, raw_ces, final_ious, final_ces = [], [], [], []

    csv_path = out_dir / "stages.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_index", "is_observed", "gt_x", "gt_y", "gt_w", "gt_h",
            "raw_x", "raw_y", "raw_w", "raw_h",
            "cx_marked", "cy_marked", "cx_pchip", "cy_pchip", "cx_final", "cy_final",
            "w_final", "h_final", "cx_outlier", "cy_outlier",
            "raw_iou", "raw_ce", "final_iou", "final_ce",
        ])
        for i, frame in enumerate(fi.tolist()):
            gt = gt_map.get(frame)
            gt_arr = np.array(gt, dtype=np.float64) if gt is not None else None
            raw_box = bbox[i]
            final_box = np.array([
                cx_final[i] - w_final[i] / 2.0,
                cy_final[i] - h_final[i] / 2.0,
                w_final[i],
                h_final[i],
            ], dtype=np.float64)
            raw_iou = raw_ce = final_iou = final_ce = ""
            if gt_arr is not None:
                raw_iou = _bbox_iou(raw_box, gt_arr)
                raw_ce = _center_error(raw_box, gt_arr)
                final_iou = _bbox_iou(final_box, gt_arr)
                final_ce = _center_error(final_box, gt_arr)
                raw_ious.append(float(raw_iou))
                raw_ces.append(float(raw_ce))
                final_ious.append(float(final_iou))
                final_ces.append(float(final_ce))
            writer.writerow([
                frame,
                int(observed_mask[i]),
                *(gt if gt is not None else ["", "", "", ""]),
                *raw_box.tolist(),
                float(cx_marked[i]) if np.isfinite(cx_marked[i]) else "",
                float(cy_marked[i]) if np.isfinite(cy_marked[i]) else "",
                float(cx_pchip[i]),
                float(cy_pchip[i]),
                float(cx_final[i]),
                float(cy_final[i]),
                float(w_final[i]),
                float(h_final[i]),
                int(cx_mask[i]),
                int(cy_mask[i]),
                raw_iou,
                raw_ce,
                final_iou,
                final_ce,
            ])

    summary = {
        "video": args.video,
        "subject": args.subject,
        "model": args.model,
        "frames_total": int(len(fi)),
        "frames_observed": int(observed_mask.sum()),
        "frames_missing": int((~observed_mask).sum()),
        "missing_policy": args.missing_policy,
        "cx_outliers": int(cx_mask.sum()),
        "cy_outliers": int(cy_mask.sum()),
        "gt_eval_frames": int(len(gt_frames)),
        "raw_iou_mean": best_eval["raw_iou_mean"],
        "raw_ce_mean": best_eval["raw_ce_mean"],
        "final_iou_mean": best_eval["final_iou_mean"],
        "final_ce_mean": best_eval["final_ce_mean"],
        "delta_iou": None if best_eval["raw_iou_mean"] is None else float(best_eval["final_iou_mean"] - best_eval["raw_iou_mean"]),
        "delta_ce": None if best_eval["raw_ce_mean"] is None else float(best_eval["final_ce_mean"] - best_eval["raw_ce_mean"]),
        "selected_params": {
            "macro_ratio": best_eval["macro_ratio"],
            "macro_sigma": best_eval["macro_sigma"],
            "micro_hw": best_eval["micro_hw"],
            "micro_sigma": best_eval["micro_sigma"],
            "max_outlier_run": best_eval["max_outlier_run"],
            "anchor_keep_ratio": best_eval["anchor_keep_ratio"],
            "sg_window": best_eval["sg_window"],
            "sg_polyorder": best_eval["sg_polyorder"],
        },
        "sweep_enabled": bool(args.sweep),
        "csv_path": str(csv_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
