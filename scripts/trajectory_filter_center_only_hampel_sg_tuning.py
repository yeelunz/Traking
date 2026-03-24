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
        if self.n <= 0:
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


def _ce(pred: Tuple[float, float, float, float], gt: Tuple[float, float, float, float]) -> float:
    px, py, pw, ph = pred
    gx, gy, gw, gh = gt
    pcx = px + pw / 2.0
    pcy = py + ph / 2.0
    gcx = gx + gw / 2.0
    gcy = gy + gh / 2.0
    return float(np.hypot(pcx - gcx, pcy - gcy))


def _candidate_grid(round_idx: int) -> List[Dict[str, Any]]:
    if round_idx == 1:
        sg_windows = [3, 5]
        sg_orders = [1, 2]
        macro_ratios = [0.06, 0.08, 0.10]
        micro_hws = [1, 2]
        sigma_pairs = [(2.5, 2.5), (3.0, 3.0)]
    else:
        sg_windows = [3, 5, 7]
        sg_orders = [1, 2]
        macro_ratios = [0.05, 0.07, 0.09, 0.11]
        micro_hws = [1, 2]
        sigma_pairs = [(2.5, 2.5), (3.0, 3.0)]

    out: List[Dict[str, Any]] = []
    for w in sg_windows:
        for p in sg_orders:
            if p >= w:
                continue
            for mr in macro_ratios:
                for hw in micro_hws:
                    for ms, ns in sigma_pairs:
                        params = {
                            "sg_window": int(w),
                            "sg_polyorder": int(p),
                            "macro_ratio": float(mr),
                            "micro_hw": int(hw),
                            "macro_sigma": float(ms),
                            "micro_sigma": float(ns),
                        }
                        name = (
                            f"w{w}_p{p}_mr{int(mr*100):02d}_hw{hw}"
                            f"_ms{str(ms).replace('.', '')}_ns{str(ns).replace('.', '')}"
                        )
                        out.append({"name": name, "traj_params": params})
    return out


def _eval_cfg(
    chunks: List[List[Dict[str, Any]]],
    gt_maps: List[Dict[int, Tuple[float, float, float, float]]],
    traj_params: Dict[str, Any],
) -> Tuple[float, float]:
    agg = Agg()
    for chunk, gt in zip(chunks, gt_maps):
        fi = np.asarray([int(x["frame_index"]) for x in chunk], dtype=np.int64)
        bb = np.asarray([x["bbox"] for x in chunk], dtype=np.float64)
        x, y, w, h = bb[:, 0], bb[:, 1], bb[:, 2], bb[:, 3]
        cx = x + w / 2.0
        cy = y + h / 2.0
        s = np.asarray([float(x.get("score") or 0.0) for x in chunk], dtype=np.float64)

        out = filter_detections(
            fi,
            cx,
            cy,
            w,
            h,
            s,
            bbox_strategy="hampel_only",
            bbox_params={"macro_ratio": traj_params["macro_ratio"], "micro_hw": traj_params["micro_hw"]},
            traj_params=traj_params,
            skip_hampel=False,
        )

        fi2 = out["frame_indices"]
        cx2 = out["cx"]
        cy2 = out["cy"]

        fi_idx = {int(f): i for i, f in enumerate(fi.tolist())}
        pred_map: Dict[int, Tuple[float, float, float, float]] = {}
        for i in range(len(fi2)):
            f = int(fi2[i])
            src_i = fi_idx.get(f)
            if src_i is None:
                continue
            ww = float(w[src_i])
            hh = float(h[src_i])
            px = float(cx2[i] - ww / 2.0)
            py = float(cy2[i] - hh / 2.0)
            pred_map[f] = (px, py, ww, hh)

        for f, g in gt.items():
            p = pred_map.get(f)
            if p is None:
                continue
            agg.add(_iou(p, g), _ce(p, g))

    return agg.mean()


def iter_experiments(roots: List[Path]) -> List[Path]:
    out: List[Path] = []
    for r in roots:
        if not r.exists():
            continue
        for d in sorted(r.iterdir()):
            if not d.is_dir():
                continue
            if (d / "metadata.json").exists() and (d / "test" / "trajectory_filter" / "filtered_detection_summary.json").exists():
                out.append(d)
    return out


def _rank_summary(rows: List[Dict[str, Any]]) -> List[Tuple[str, Dict[str, float]]]:
    by_name: Dict[str, List[Dict[str, float]]] = {}
    for r in rows:
        for k, v in r["candidates"].items():
            by_name.setdefault(k, []).append(v)

    out: List[Tuple[str, Dict[str, float]]] = []
    for name, arr in by_name.items():
        di = np.asarray([x["delta_iou"] for x in arr], dtype=np.float64)
        dc = np.asarray([x["delta_ce_improve"] for x in arr], dtype=np.float64)
        both = (di > 0) & (dc > 0)
        n = len(arr)
        out.append((name, {
            "n": float(n),
            "both_count": float(both.sum()),
            "both_ratio": float(both.sum() / n),
            "mean_delta_iou": float(di.mean()),
            "mean_delta_ce_improve": float(dc.mean()),
            "iou_pos_count": float((di > 0).sum()),
            "ce_pos_count": float((dc > 0).sum()),
        }))

    out.sort(key=lambda kv: (kv[1]["both_ratio"], kv[1]["mean_delta_iou"], kv[1]["mean_delta_ce_improve"]), reverse=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedules", nargs="+", required=True)
    ap.add_argument("--max-rounds", type=int, default=2)
    ap.add_argument("--max-exp", type=int, default=0)
    ap.add_argument("--out", default="results/trajectory_filter_center_only_hampel_sg_report.json")
    args = ap.parse_args()

    roots = [Path(x) for x in args.schedules]
    exps = iter_experiments(roots)
    if int(args.max_exp) > 0:
        exps = exps[: int(args.max_exp)]

    report: Dict[str, Any] = {
        "n_experiments": 0,
        "rounds": [],
        "selected": None,
    }

    for round_idx in range(1, max(1, int(args.max_rounds)) + 1):
        cands = _candidate_grid(round_idx)
        per_exp: List[Dict[str, Any]] = []

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

            row: Dict[str, Any] = {
                "experiment": str(exp),
                "baseline": {"iou_mean": b_iou, "ce_mean": b_ce},
                "candidates": {},
            }

            for cand in cands:
                iou_m, ce_m = _eval_cfg(chunks, gt_maps, cand["traj_params"])
                di = float(iou_m - b_iou)
                dc = float(b_ce - ce_m)
                row["candidates"][cand["name"]] = {
                    "iou_mean": float(iou_m),
                    "ce_mean": float(ce_m),
                    "delta_iou": di,
                    "delta_ce_improve": dc,
                    "both_improved": bool(di > 0 and dc > 0),
                    "traj_params": cand["traj_params"],
                }

            per_exp.append(row)
            print(f"[round {round_idx}] [{i}/{len(exps)}] {exp.name}")

        ranked = _rank_summary(per_exp)
        top_name, top_summary = ranked[0]

        best_params = None
        for r in per_exp:
            if top_name in r["candidates"]:
                best_params = r["candidates"][top_name]["traj_params"]
                break

        round_result = {
            "round": round_idx,
            "n_candidates": len(cands),
            "n_experiments": len(per_exp),
            "top_name": top_name,
            "top_summary": top_summary,
            "top10": [{"name": name, **stats} for name, stats in ranked[:10]],
            "per_experiment": per_exp,
            "selected_traj_params": best_params,
        }
        report["rounds"].append(round_result)
        report["n_experiments"] = len(per_exp)

        if top_summary["mean_delta_iou"] > 0 and top_summary["mean_delta_ce_improve"] > 0:
            report["selected"] = {
                "round": round_idx,
                "name": top_name,
                "traj_params": best_params,
                "summary": top_summary,
            }
            break

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("Saved:", out_path)
    print(json.dumps(report.get("selected") or {}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
