from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tracking.classification.trajectory_filter import filter_detections


RESULT_BASE = Path(
    r"c:/Users/User/Desktop/code/Traking/results/replay_n003_full_rf_concat_debug_v2/2026-03-24_20-09-17_mednext_clahe_global_tabv4_rf_concat_loso_n003"
)
PRED_BASE = RESULT_BASE / "test" / "detection" / "predictions_by_video" / "n003"
ANN_BASE = Path(r"c:/Users/User/Desktop/code/Traking/dataset/merged_extend/n003")
OUT_DIR = RESULT_BASE / "analysis" / "generic_postprocess_search" / "n003"
OUT_DIR.mkdir(parents=True, exist_ok=True)


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
    return inter / union if union > 0 else 0.0


def _center_error(a: np.ndarray, b: np.ndarray) -> float:
    acx, acy = float(a[0] + a[2] / 2.0), float(a[1] + a[3] / 2.0)
    bcx, bcy = float(b[0] + b[2] / 2.0), float(b[1] + b[3] / 2.0)
    return float(((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5)


def _load_gt_map(video: str) -> dict[int, list[float]]:
    ann = json.loads((ANN_BASE / f"{video}.json").read_text(encoding="utf-8"))
    id2f = {int(i["id"]): int(i["frame_index"]) for i in ann.get("images") or []}
    gt: dict[int, list[float]] = {}
    for a in ann.get("annotations") or []:
        imid = int(a["image_id"])
        if imid in id2f:
            gt[id2f[imid]] = list(map(float, a["bbox"]))
    return gt


def _eval_video(video: str, cfg: dict[str, Any]) -> dict[str, Any]:
    rows = json.loads((PRED_BASE / video / "YOLOv11.json").read_text(encoding="utf-8"))
    gt_map = _load_gt_map(video)

    fi = np.array([int(r["frame_index"]) for r in rows], dtype=np.int64)
    bbox = np.array([r["bbox"] for r in rows], dtype=np.float64)
    cx = bbox[:, 0] + bbox[:, 2] / 2.0
    cy = bbox[:, 1] + bbox[:, 3] / 2.0
    w = bbox[:, 2]
    h = bbox[:, 3]
    s = np.array([float(r.get("score", 0.0) or 0.0) for r in rows], dtype=np.float64)
    observed = np.isfinite(bbox).all(axis=1) & (bbox[:, 2] > 0.0) & (bbox[:, 3] > 0.0)

    if cfg["bbox_strategy"] == "none":
        bbox_params: dict[str, Any] = {}
    elif cfg["bbox_strategy"] == "hampel_only":
        bbox_params = {
            "macro_ratio": cfg["macro_ratio"],
            "macro_sigma": cfg["macro_sigma"],
            "micro_hw": cfg["micro_hw"],
            "micro_sigma": cfg["micro_sigma"],
        }
    elif cfg["bbox_strategy"] == "independent":
        bbox_params = {
            "macro_ratio": cfg["macro_ratio"],
            "micro_hw": cfg["micro_hw"],
            "sg_window": cfg["sg_window"],
            "sg_polyorder": cfg["sg_polyorder"],
        }
    elif cfg["bbox_strategy"] == "area_constraint":
        bbox_params = {
            "max_area_change_ratio": 0.20,
            "sg_window": cfg["sg_window"],
            "sg_polyorder": cfg["sg_polyorder"],
        }
    else:
        raise ValueError(f"Unsupported bbox_strategy: {cfg['bbox_strategy']}")

    out = filter_detections(
        fi,
        cx,
        cy,
        w,
        h,
        s,
        bbox_strategy=cfg["bbox_strategy"],
        bbox_params=bbox_params,
        traj_params={
            "macro_ratio": cfg["macro_ratio"],
            "macro_sigma": cfg["macro_sigma"],
            "micro_hw": cfg["micro_hw"],
            "micro_sigma": cfg["micro_sigma"],
            "sg_window": cfg["sg_window"],
            "sg_polyorder": cfg["sg_polyorder"],
        },
        anchor_keep_ratio=0.0,
    )

    idx = np.argsort(fi)
    fi_s = fi[idx]
    raw_s = bbox[idx]
    filt_b = np.column_stack([
        out["cx"] - out["widths"] / 2.0,
        out["cy"] - out["heights"] / 2.0,
        out["widths"],
        out["heights"],
    ])

    raw_ious: list[float] = []
    raw_ces: list[float] = []
    filt_ious: list[float] = []
    filt_ces: list[float] = []
    for i, f in enumerate(fi_s.tolist()):
        if f not in gt_map:
            continue
        gt = np.array(gt_map[f], dtype=np.float64)
        raw_ious.append(_bbox_iou(raw_s[i], gt))
        raw_ces.append(_center_error(raw_s[i], gt))
        filt_ious.append(_bbox_iou(filt_b[i], gt))
        filt_ces.append(_center_error(filt_b[i], gt))

    return {
        "video": video,
        "frames": len(filt_ious),
        "raw_iou": float(np.mean(raw_ious)) if raw_ious else 0.0,
        "raw_ce": float(np.mean(raw_ces)) if raw_ces else 0.0,
        "iou": float(np.mean(filt_ious)) if filt_ious else 0.0,
        "ce": float(np.mean(filt_ces)) if filt_ces else 0.0,
    }


def main() -> None:
    videos = sorted([p.name for p in PRED_BASE.iterdir() if p.is_dir()])

    macro_ratio_list = [0.02, 0.06, 0.10]
    macro_sigma_list = [1.5, 2.5]
    micro_hw_list = [1, 3]
    micro_sigma_list = [1.5, 2.0]
    sg_window_list = [1, 5, 9]
    bbox_strategy_list = ["none", "hampel_only", "independent", "area_constraint"]

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for idx, (mr, ms, mh, mis, sgw, bs) in enumerate(product(
        macro_ratio_list,
        macro_sigma_list,
        micro_hw_list,
        micro_sigma_list,
        sg_window_list,
        bbox_strategy_list,
    )):
        sg_polyorder = 1 if sgw == 1 else 2
        cfg = {
            "macro_ratio": mr,
            "macro_sigma": ms,
            "micro_hw": mh,
            "micro_sigma": mis,
            "sg_window": sgw,
            "sg_polyorder": sg_polyorder,
            "bbox_strategy": bs,
        }

        per_video = [_eval_video(v, cfg) for v in videos]
        iou = float(np.mean([x["iou"] for x in per_video]))
        ce = float(np.mean([x["ce"] for x in per_video]))
        r2 = next(x for x in per_video if x["video"] == "R2")
        r1 = next(x for x in per_video if x["video"] == "R1")

        rec = {
            **cfg,
            "mean_iou": iou,
            "mean_ce": ce,
            "r2_iou": float(r2["iou"]),
            "r2_ce": float(r2["ce"]),
            "r1_iou": float(r1["iou"]),
            "r1_ce": float(r1["ce"]),
            "per_video": per_video,
        }
        rows.append(rec)

        if (idx + 1) % 20 == 0:
            print(f"progress {idx + 1}")

        if best is None:
            best = rec
        else:
            # prioritize lower CE then higher IoU globally
            if (rec["mean_ce"] < best["mean_ce"]) or (
                abs(rec["mean_ce"] - best["mean_ce"]) < 1e-9 and rec["mean_iou"] > best["mean_iou"]
            ):
                best = rec

    rows_sorted = sorted(rows, key=lambda r: (r["mean_ce"], -r["mean_iou"]))
    target_rows = [r for r in rows_sorted if r["r2_iou"] >= 0.65 and r["r2_ce"] <= 20.0]

    summary = {
        "videos": videos,
        "search_size": len(rows_sorted),
        "best_global": best,
        "best_r2": max(rows_sorted, key=lambda r: (r["r2_iou"], -r["r2_ce"])),
        "best_r2_ce": min(rows_sorted, key=lambda r: r["r2_ce"]),
        "target_hits_r2": len(target_rows),
        "target_examples_r2": target_rows[:10],
    }

    (OUT_DIR / "search_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "top20_by_global.json").write_text(json.dumps(rows_sorted[:20], ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "top20_by_r2_iou.json").write_text(
        json.dumps(sorted(rows, key=lambda r: (-r["r2_iou"], r["r2_ce"]))[:20], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
