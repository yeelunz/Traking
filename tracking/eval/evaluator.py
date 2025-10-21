from __future__ import annotations
from typing import Dict, Any, List, Tuple
import math
import os
import json
import csv
from collections import defaultdict

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None

from ..core.interfaces import FramePrediction
from ..core.registry import register_evaluator


def bbox_iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = aw * ah + bw * bh - inter
    return inter / ua if ua > 0 else 0.0


def center_error(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ac = (ax + aw / 2.0, ay + ah / 2.0)
    bc = (bx + bw / 2.0, by + bh / 2.0)
    return math.hypot(ac[0] - bc[0], ac[1] - bc[1])


@register_evaluator("BasicEvaluator")
class BasicEvaluator:
    name = "BasicEvaluator"

    def evaluate(self, predictions: Dict[str, List[FramePrediction]], gt: Dict[str, Any], out_dir: str) -> Dict[str, Any]:
        os.makedirs(out_dir, exist_ok=True)
        # Assume gt: {"frames": {fi: [bbox,...]}, ...}
        frames_gt = gt.get("frames", {})
        results = {}

        def _det_counts_ap(preds: List[FramePrediction], frames_gt: Dict[int, List[Tuple[float, float, float, float]]], thr: float):
            # Single-object per frame. Match first GT box if present.
            pred_map = {}
            for p in preds:
                # if multiple preds at same frame, keep the first
                pred_map.setdefault(p.frame_index, p.bbox)
            gt_frames = {fi for fi, boxes in frames_gt.items() if boxes}
            all_frames = gt_frames | set(pred_map.keys())
            tp = fp = fn = 0
            for fi in all_frames:
                has_gt = fi in gt_frames
                has_pred = fi in pred_map
                if has_gt and has_pred:
                    gtb = frames_gt.get(fi, [None])[0]
                    if gtb is None:
                        fp += 1
                    else:
                        iou = bbox_iou(pred_map[fi], gtb)
                        if iou >= thr:
                            tp += 1
                        else:
                            fp += 1
                elif has_gt and not has_pred:
                    fn += 1
                elif has_pred and not has_gt:
                    fp += 1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            # With single operating point, we report AP as precision at this point.
            ap = precision
            return {
                "tp": tp, "fp": fp, "fn": fn,
                "precision": precision, "recall": recall, "ap": ap,
            }
        for model_name, preds in predictions.items():
            # --- Debug coverage stats (before metric loops) ---
            try:
                gt_frames_nonempty = sorted([int(fi) for fi, boxes in frames_gt.items() if boxes])
            except Exception:
                gt_frames_nonempty = []
            total_gt_frames = len(gt_frames_nonempty)
            pred_frame_indices_all = [int(getattr(p, 'frame_index', -1)) for p in preds]
            unique_pred_frames = sorted(set(pred_frame_indices_all))
            total_pred_frames = len(unique_pred_frames)
            # direct intersection (frames with at least one GT box and at least one pred)
            matched_frames = len(set(gt_frames_nonempty) & set(unique_pred_frames))
            coverage_ratio = (matched_frames / total_gt_frames) if total_gt_frames > 0 else 0.0
            gt_frame_range = [min(gt_frames_nonempty), max(gt_frames_nonempty)] if gt_frames_nonempty else [None, None]
            pred_frame_range = [min(unique_pred_frames), max(unique_pred_frames)] if unique_pred_frames else [None, None]
            # heuristic: suggested constant offset to maximize overlap using first frame indices
            suggested_offset = None
            if gt_frames_nonempty and unique_pred_frames:
                suggested_offset = int(gt_frames_nonempty[0]) - int(unique_pred_frames[0])
            ious: List[float] = []
            ces: List[float] = []
            per_frame_rows: List[List[Any]] = [["frame_index", "iou", "center_error"]]
            # For displacement plot (center magnitude vs. time)
            centers_mag: List[Tuple[int, float]] = []  # (frame_index, |center_pred|)
            # Prepare GT center magnitude per frame (first GT box only)
            gt_center_mag_map: Dict[int, float] = {}
            for _fi, _boxes in frames_gt.items():
                if not _boxes:
                    continue
                try:
                    gx, gy, gw, gh = _boxes[0]
                    gcx, gcy = float(gx) + float(gw) / 2.0, float(gy) + float(gh) / 2.0
                    gt_center_mag_map[int(_fi)] = float((gcx ** 2 + gcy ** 2) ** 0.5)
                except Exception:
                    continue
            # Build a quick lookup of predictions by frame (first prediction wins)
            pred_map: Dict[int, Tuple[float, float, float, float]] = {}
            for p in preds:
                pred_map.setdefault(int(p.frame_index), p.bbox)
            for p in preds:
                gt_boxes = frames_gt.get(p.frame_index, [])
                if not gt_boxes:
                    # no GT for this frame -> skip metrics row but still record NA (optional)
                    per_frame_rows.append([p.frame_index, None, None])
                    # still record center displacement magnitude
                    cx, cy = p.center
                    centers_mag.append((p.frame_index, float((cx ** 2 + cy ** 2) ** 0.5)))
                    continue
                # use first GT box for now
                gtb = gt_boxes[0]
                i = bbox_iou(p.bbox, gtb)
                c = center_error(p.bbox, gtb)
                ious.append(i)
                ces.append(c)
                per_frame_rows.append([p.frame_index, i, c])
                # center displacement relative to origin (0,0) for prediction
                cx, cy = p.center
                centers_mag.append((p.frame_index, float((cx ** 2 + cy ** 2) ** 0.5)))
            # --- Success curve & AUC ---
            success_curve: List[Tuple[float, float]] = []  # (IoU threshold, success rate)
            success_auc = 0.0
            if ious:
                thresholds = [i / 100.0 for i in range(0, 101)]
                total_frames = float(len(ious))
                success_rates: List[float] = []
                for thr in thresholds:
                    success_rate = sum(1 for val in ious if val >= thr) / total_frames
                    success_curve.append((thr, success_rate))
                    success_rates.append(success_rate)
                success_auc = sum(success_rates) / len(success_rates)
            # detection-style metrics (single object) at IoU thresholds
            det50 = _det_counts_ap(preds, frames_gt, 0.5)
            det75 = _det_counts_ap(preds, frames_gt, 0.75)
            summary = {
                "iou_mean": sum(ious) / len(ious) if ious else 0.0,
                "iou_std": (sum((x - (sum(ious) / len(ious))) ** 2 for x in ious) / len(ious)) ** 0.5 if ious else 0.0,
                "ce_mean": sum(ces) / len(ces) if ces else 0.0,
                "ce_std": (sum((x - (sum(ces) / len(ces))) ** 2 for x in ces) / len(ces)) ** 0.5 if ces else 0.0,
                # aggregation helpers for cross-video/global stats (over frames)
                "count": len(ious),
                "sum_iou": float(sum(ious)),
                "sum_iou_sq": float(sum(x * x for x in ious)),
                "sum_ce": float(sum(ces)),
                "sum_ce_sq": float(sum(x * x for x in ces)),
                # mAP-like metrics (single-object precision at IoU threshold)
                "mAP_50": det50["ap"],
                "mAP_75": det75["ap"],
                # raw counts for aggregation
                "tp_50": det50["tp"], "fp_50": det50["fp"], "fn_50": det50["fn"],
                "tp_75": det75["tp"], "fp_75": det75["fp"], "fn_75": det75["fn"],
                # Success curve metric (per-sequence)
                "success_auc": float(success_auc),
                # --- Debug coverage fields ---
                "debug_total_gt_frames": total_gt_frames,
                "debug_total_pred_frames": total_pred_frames,
                "debug_matched_frames": matched_frames,
                "debug_coverage_ratio": coverage_ratio,
                "debug_gt_frame_range": gt_frame_range,
                "debug_pred_frame_range": pred_frame_range,
                "debug_suggested_constant_offset": suggested_offset,
            }
            results[model_name] = summary
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, f"{model_name}_metrics.json"), "w", encoding="utf-8") as f:
                json.dump({"per_model": summary}, f, ensure_ascii=False, indent=2)
            # CSV for convenience
            with open(os.path.join(out_dir, f"{model_name}_metrics.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["metric", "value"])
                for k, v in summary.items():
                    w.writerow([k, v])
            # per-frame csv
            with open(os.path.join(out_dir, f"{model_name}_per_frame.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerows(per_frame_rows)
            # Success curve csv
            if success_curve:
                with open(os.path.join(out_dir, f"{model_name}_success_curve.csv"), "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["iou_threshold", "success_rate"])
                    for thr, val in success_curve:
                        w.writerow([thr, val])
            # plots
            if plt is not None:
                try:
                    # IoU histogram
                    if ious:
                        plt.figure()
                        plt.hist(ious, bins=20, alpha=0.7)
                        plt.title(f"IoU Histogram - {model_name}")
                        plt.xlabel("IoU"); plt.ylabel("Count")
                        plt.tight_layout()
                        plt.savefig(os.path.join(out_dir, f"{model_name}_iou_hist.png"))
                        plt.close()
                    # Center error histogram
                    if ces:
                        plt.figure()
                        plt.hist(ces, bins=20, alpha=0.7)
                        plt.title(f"Center Error Histogram - {model_name}")
                        plt.xlabel("Pixels"); plt.ylabel("Count")
                        plt.tight_layout()
                        plt.savefig(os.path.join(out_dir, f"{model_name}_ce_hist.png"))
                        plt.close()
                    # Center displacement (|center|) over time curve: plot GT and Pred lines aligned on GT frames
                    # If GT exists, align x-axis to GT frames and draw both lines; otherwise fallback to Pred-only
                    has_gt_centers = bool(gt_center_mag_map)
                    if has_gt_centers or centers_mag:
                        plt.figure()
                        if has_gt_centers:
                            # x-axis: sorted GT frames
                            xs_gt = sorted(gt_center_mag_map.keys())
                            ys_gt = [gt_center_mag_map[fi] for fi in xs_gt]
                            # Prediction magnitudes mapped by frame
                            pred_center_mag_map = {}
                            for fi, bb in pred_map.items():
                                try:
                                    px, py, pw, ph = bb
                                    pcx, pcy = float(px) + float(pw) / 2.0, float(py) + float(ph) / 2.0
                                    pred_center_mag_map[int(fi)] = float((pcx ** 2 + pcy ** 2) ** 0.5)
                                except Exception:
                                    continue
                            import math as _m
                            ys_pred = [pred_center_mag_map.get(fi, float('nan')) for fi in xs_gt]
                            plt.plot(xs_gt, ys_gt, label='GT', linewidth=1.8)
                            plt.plot(xs_gt, ys_pred, label='Pred', linewidth=1.2)
                            plt.legend()
                            xs_final = xs_gt
                        else:
                            centers_mag.sort(key=lambda t: t[0])
                            xs_final = [fi for fi, _ in centers_mag]
                            ys = [mag for _, mag in centers_mag]
                            plt.plot(xs_final, ys, linewidth=1.5, label='Pred')
                            plt.legend()
                        plt.title(f"Center displacement vs Time - {model_name}")
                        plt.xlabel("Frame index")
                        plt.ylabel("|center| (pixels)")
                        plt.tight_layout()
                        plt.savefig(os.path.join(out_dir, f"{model_name}_center_displacement.png"))
                        plt.close()
                    # Success curve plot
                    if success_curve:
                        plt.figure()
                        xs = [thr for thr, _ in success_curve]
                        ys = [v for _, v in success_curve]
                        plt.plot(xs, ys, linewidth=1.8)
                        plt.title(f"Success Curve - {model_name}")
                        plt.xlabel("IoU threshold")
                        plt.ylabel("Success rate")
                        plt.ylim(0.0, 1.0)
                        plt.tight_layout()
                        plt.savefig(os.path.join(out_dir, f"{model_name}_success_curve.png"))
                        plt.close()
                except Exception:
                    pass
        # write combined
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        return results
