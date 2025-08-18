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
        for model_name, preds in predictions.items():
            ious: List[float] = []
            ces: List[float] = []
            per_frame_rows: List[List[Any]] = [["frame_index", "iou", "center_error"]]
            for p in preds:
                gt_boxes = frames_gt.get(p.frame_index, [])
                if not gt_boxes:
                    # no GT for this frame -> skip metrics row but still record NA (optional)
                    per_frame_rows.append([p.frame_index, None, None])
                    continue
                # use first GT box for now
                gtb = gt_boxes[0]
                i = bbox_iou(p.bbox, gtb)
                c = center_error(p.bbox, gtb)
                ious.append(i)
                ces.append(c)
                per_frame_rows.append([p.frame_index, i, c])
            summary = {
                "iou_mean": sum(ious) / len(ious) if ious else 0.0,
                "iou_std": (sum((x - (sum(ious) / len(ious))) ** 2 for x in ious) / len(ious)) ** 0.5 if ious else 0.0,
                "ce_mean": sum(ces) / len(ces) if ces else 0.0,
                "ce_std": (sum((x - (sum(ces) / len(ces))) ** 2 for x in ces) / len(ces)) ** 0.5 if ces else 0.0,
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
                except Exception:
                    pass
        # write combined
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        return results
