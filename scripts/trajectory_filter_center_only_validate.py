from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracking.classification.trajectory_filter import filter_detections


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


def _ce(pred: Tuple[float, float, float, float], gt: Tuple[float, float, float, float]) -> float:
    px, py, pw, ph = pred
    gx, gy, gw, gh = gt
    pcx = px + pw / 2.0
    pcy = py + ph / 2.0
    gcx = gx + gw / 2.0
    gcy = gy + gh / 2.0
    return float(np.hypot(pcx - gcx, pcy - gcy))


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
    ap.add_argument("--sg-window", type=int, required=True)
    ap.add_argument("--sg-polyorder", type=int, required=True)
    ap.add_argument("--macro-ratio", type=float, required=True)
    ap.add_argument("--micro-hw", type=int, required=True)
    ap.add_argument("--macro-sigma", type=float, default=3.0)
    ap.add_argument("--micro-sigma", type=float, default=3.0)
    ap.add_argument("--out", default="results/trajectory_filter_center_only_validation.json")
    args = ap.parse_args()

    traj_params = {
        "sg_window": int(args.sg_window),
        "sg_polyorder": int(args.sg_polyorder),
        "macro_ratio": float(args.macro_ratio),
        "micro_hw": int(args.micro_hw),
        "macro_sigma": float(args.macro_sigma),
        "micro_sigma": float(args.micro_sigma),
    }

    roots = [Path(x) for x in args.schedules]
    exps = iter_experiments(roots)

    deltas_iou: List[float] = []
    deltas_ce: List[float] = []
    both_count = 0

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

        gt_maps: List[Dict[int, Tuple[float, float, float, float]]] = []
        valid = True
        for rel in test_preview:
            p = (ds_root / rel).with_suffix(".json")
            if not p.exists():
                valid = False
                break
            gt_maps.append(_gt_map(p))
        if not valid:
            continue

        fd = _load_json(exp / "test" / "trajectory_filter" / "filtered_detection_summary.json")
        base = next(iter(fd.values()))
        b_iou = float(base.get("iou_mean", 0.0))
        b_ce = float(base.get("ce_mean", 0.0))

        iou_sum = 0.0
        ce_sum = 0.0
        n = 0
        for chunk, gt in zip(chunks, gt_maps):
            fi = np.asarray([int(x["frame_index"]) for x in chunk], dtype=np.int64)
            bb = np.asarray([x["bbox"] for x in chunk], dtype=np.float64)
            x, y, w, h = bb[:, 0], bb[:, 1], bb[:, 2], bb[:, 3]
            cx = x + w / 2.0
            cy = y + h / 2.0
            s = np.asarray([float(x.get("score") or 0.0) for x in chunk], dtype=np.float64)

            out = filter_detections(
                fi, cx, cy, w, h, s,
                bbox_strategy="hampel_only",
                bbox_params={"macro_ratio": traj_params["macro_ratio"], "micro_hw": traj_params["micro_hw"]},
                traj_params=traj_params,
                skip_hampel=False,
            )

            fi2 = out["frame_indices"]
            cx2 = out["cx"]
            cy2 = out["cy"]
            fi_idx = {int(f): j for j, f in enumerate(fi.tolist())}
            pred_map: Dict[int, Tuple[float, float, float, float]] = {}
            for j in range(len(fi2)):
                f = int(fi2[j])
                src = fi_idx.get(f)
                if src is None:
                    continue
                ww = float(w[src])
                hh = float(h[src])
                pred_map[f] = (float(cx2[j] - ww / 2.0), float(cy2[j] - hh / 2.0), ww, hh)

            for f, g in gt.items():
                p = pred_map.get(f)
                if p is None:
                    continue
                iou_sum += _iou(p, g)
                ce_sum += _ce(p, g)
                n += 1

        if n == 0:
            continue
        iou_m = iou_sum / n
        ce_m = ce_sum / n
        di = iou_m - b_iou
        dc = b_ce - ce_m
        deltas_iou.append(di)
        deltas_ce.append(dc)
        if di > 0 and dc > 0:
            both_count += 1
        print(f"[{i}/{len(exps)}] {exp.name}")

    result = {
        "traj_params": traj_params,
        "n_experiments": len(deltas_iou),
        "both_positive_count": both_count,
        "both_positive_ratio": (both_count / len(deltas_iou)) if deltas_iou else 0.0,
        "mean_delta_iou": float(np.mean(deltas_iou)) if deltas_iou else 0.0,
        "mean_delta_ce_improve": float(np.mean(deltas_ce)) if deltas_ce else 0.0,
        "iou_positive_count": int(np.sum(np.asarray(deltas_iou) > 0)) if deltas_iou else 0,
        "ce_positive_count": int(np.sum(np.asarray(deltas_ce) > 0)) if deltas_ce else 0,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved:", out_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
