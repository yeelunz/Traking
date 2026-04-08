from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracking.classification.trajectory_filter import filter_detections
from tracking.core.interfaces import FramePrediction
from tracking.core.registry import PREPROC_REGISTRY, MODEL_REGISTRY
from tracking.utils.annotations import load_coco_vid

# Populate registries.
from tracking.preproc import clahe  # noqa: F401
from tracking.preproc import augment  # noqa: F401
from tracking.models import yolov11  # noqa: F401


def _iou(box_a, box_b) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _center_error(box_a, box_b) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    acx = ax + aw / 2.0
    acy = ay + ah / 2.0
    bcx = bx + bw / 2.0
    bcy = by + bh / 2.0
    return float(np.sqrt((acx - bcx) ** 2 + (acy - bcy) ** 2))


def _summarise(preds_by_video: Dict[str, List[FramePrediction]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for video_path, preds in preds_by_video.items():
        fallback_frames = sum(1 for p in preds if bool(getattr(p, "is_fallback", False)))
        out[video_path] = {
            "frames": float(len(preds)),
            "fallback_frames": float(fallback_frames),
            "fallback_rate": (float(fallback_frames) / float(len(preds))) if preds else 0.0,
        }
    return out


def _evaluate(dataset_root: Path, preds_by_video: Dict[str, List[FramePrediction]]) -> Dict[str, Any]:
    per_video: Dict[str, Dict[str, float]] = {}
    ious: List[float] = []
    ces: List[float] = []
    tp50 = 0
    fp50 = 0
    for video_path, preds in preds_by_video.items():
        ann_path = Path(video_path).with_suffix(".json")
        gt = load_coco_vid(str(ann_path))
        gt_frames = gt.get("frames", {}) or {}
        pred_map = {int(p.frame_index): tuple(map(float, p.bbox)) for p in preds}
        v_ious: List[float] = []
        v_ces: List[float] = []
        for fi_raw, boxes in gt_frames.items():
            if not boxes:
                continue
            fi = int(fi_raw)
            pred_box = pred_map.get(fi)
            if pred_box is None:
                continue
            gt_box = tuple(map(float, boxes[0][:4]))
            iou = _iou(pred_box, gt_box)
            ce = _center_error(pred_box, gt_box)
            v_ious.append(iou)
            v_ces.append(ce)
            ious.append(iou)
            ces.append(ce)
            if iou >= 0.5:
                tp50 += 1
            else:
                fp50 += 1
        per_video[video_path] = {
            "frames_count": float(len(v_ious)),
            "iou_mean": float(np.mean(v_ious)) if v_ious else 0.0,
            "ce_mean": float(np.mean(v_ces)) if v_ces else 0.0,
        }
    return {
        "per_video": per_video,
        "summary": {
            "frames_count": float(len(ious)),
            "iou_mean": float(np.mean(ious)) if ious else 0.0,
            "iou_std": float(np.std(ious)) if ious else 0.0,
            "ce_mean": float(np.mean(ces)) if ces else 0.0,
            "ce_std": float(np.std(ces)) if ces else 0.0,
            "success_rate_50": (float(tp50) / float(tp50 + fp50)) if (tp50 + fp50) > 0 else 0.0,
        },
    }


def _apply_filter(preds: List[FramePrediction]) -> List[FramePrediction]:
    if len(preds) < 2:
        return list(preds)
    frame_indices = np.asarray([int(p.frame_index) for p in preds], dtype=np.int64)
    bboxes = np.asarray([list(map(float, p.bbox)) for p in preds], dtype=np.float64)
    cx = bboxes[:, 0] + bboxes[:, 2] / 2.0
    cy = bboxes[:, 1] + bboxes[:, 3] / 2.0
    widths = bboxes[:, 2].copy()
    heights = bboxes[:, 3].copy()
    scores = np.asarray([float(p.score) if p.score is not None else 0.0 for p in preds], dtype=np.float64)
    filtered = filter_detections(
        frame_indices,
        cx,
        cy,
        widths,
        heights,
        scores,
        bbox_strategy="hampel_only",
        bbox_params={"macro_ratio": 0.06, "micro_hw": 2},
        traj_params={
            "sg_window": 5,
            "sg_polyorder": 2,
            "macro_ratio": 0.06,
            "micro_hw": 2,
            "macro_sigma": 3.0,
            "micro_sigma": 3.0,
        },
        skip_hampel=False,
    )
    idx_map = {int(fi): i for i, fi in enumerate(filtered["frame_indices"])}
    out: List[FramePrediction] = []
    for pred in preds:
        k = idx_map[int(pred.frame_index)]
        nw = float(filtered["widths"][k])
        nh = float(filtered["heights"][k])
        ncx = float(filtered["cx"][k])
        ncy = float(filtered["cy"][k])
        out.append(
            FramePrediction(
                frame_index=int(pred.frame_index),
                bbox=(ncx - nw / 2.0, ncy - nh / 2.0, nw, nh),
                score=pred.score,
                is_fallback=bool(getattr(pred, "is_fallback", False)),
                bbox_source=str(getattr(pred, "bbox_source", "detector")),
            )
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay detector inference and trajectory filtering from an existing experiment.")
    ap.add_argument("experiment_dir", help="Path to an existing experiment directory under results/")
    args = ap.parse_args()

    exp_dir = Path(args.experiment_dir)
    metadata = json.loads((exp_dir / "metadata.json").read_text(encoding="utf-8"))
    dataset_root = Path(metadata["dataset"]["root"])
    subject = str((metadata.get("dataset", {}).get("split", {}) or {}).get("subject") or "")
    detector_ckpt = None
    for stage in metadata.get("stages", []):
        if stage.get("name") == "detector_train_full":
            models = stage.get("models") or []
            if models:
                detector_ckpt = models[0].get("checkpoint")
                break
    if not detector_ckpt:
        raise SystemExit("Detector checkpoint not found in metadata.")

    pipeline = metadata.get("experiment", {}).get("pipeline") or []
    model_step = next((x for x in pipeline if x.get("type") == "model"), None)
    if not model_step:
        raise SystemExit("Detector pipeline step not found in metadata.")
    model_name = str(model_step.get("name"))
    model_cfg = dict(model_step.get("params") or {})
    model_cfg["train_enabled"] = False

    model_cls = MODEL_REGISTRY.get(model_name)
    if model_cls is None:
        raise SystemExit(f"Model not registered: {model_name}")
    model = model_cls(model_cfg)
    model.load_checkpoint(str(detector_ckpt))

    preprocs = []
    for step in pipeline:
        if step.get("type") != "preproc":
            continue
        pp_cls = PREPROC_REGISTRY.get(str(step.get("name")))
        if pp_cls is None:
            continue
        preprocs.append(pp_cls(step.get("params") or {}))
    try:
        model.preprocs = preprocs
    except Exception:
        pass

    test_videos = [str(p) for p in sorted((dataset_root / subject).glob("*.avi"))]
    if not test_videos:
        raise SystemExit(f"No test videos found for subject {subject!r} under {dataset_root}")

    raw_preds: Dict[str, List[FramePrediction]] = {}
    filt_preds: Dict[str, List[FramePrediction]] = {}
    for video_path in test_videos:
        preds = model.predict(video_path)
        raw_preds[video_path] = preds
        filt_preds[video_path] = _apply_filter(preds)

    report = {
        "experiment_dir": str(exp_dir),
        "subject": subject,
        "checkpoint": str(detector_ckpt),
        "raw_fallback": _summarise(raw_preds),
        "raw_metrics": _evaluate(dataset_root, raw_preds),
        "filtered_metrics": _evaluate(dataset_root, filt_preds),
    }
    out_path = exp_dir / "test" / "trajectory_filter" / "replay_pchip_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report["raw_metrics"]["summary"], ensure_ascii=False, indent=2))
    print(json.dumps(report["filtered_metrics"]["summary"], ensure_ascii=False, indent=2))
    print(f"saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
