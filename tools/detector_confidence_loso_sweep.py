from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from tracking.data.dataset_manager import COCOJsonDatasetManager, SimpleDataset
from tracking.core.interfaces import FramePrediction
from tracking.models.rtdetrv2 import RTDETRv2Model
from tracking.models.yolov11 import YOLOv11Model
from tracking.preproc.clahe import CLAHE
from tracking.utils.annotations import load_coco_vid
from tracking.utils.prediction_interpolation import cubic_clip_interpolate_predictions

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None


BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class DetectorSpec:
    key: str
    label: str
    model_type: str
    weights: str


@dataclass
class FrameRawPrediction:
    frame_index: int
    bbox: BBox
    score: float
    is_fallback: bool


def _bbox_iou(a: BBox, b: BBox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def _center_error(a: BBox, b: BBox) -> float:
    acx = float(a[0] + a[2] / 2.0)
    acy = float(a[1] + a[3] / 2.0)
    bcx = float(b[0] + b[2] / 2.0)
    bcy = float(b[1] + b[3] / 2.0)
    return float(math.hypot(acx - bcx, acy - bcy))


def _is_valid_bbox(bbox: BBox) -> bool:
    x, y, w, h = bbox
    return bool(np.all(np.isfinite([x, y, w, h])) and w > 1.0 and h > 1.0)


def _safe_mean(values: Sequence[float]) -> float:
    valid = [float(v) for v in values if math.isfinite(float(v))]
    return float(mean(valid)) if valid else float("nan")


def _safe_std(values: Sequence[float]) -> float:
    valid = [float(v) for v in values if math.isfinite(float(v))]
    if len(valid) < 2:
        return float("nan")
    return float(pstdev(valid))


def _finite_count(values: Sequence[float]) -> int:
    return sum(1 for v in values if math.isfinite(float(v)))


def _safe_tag(text: str) -> str:
    out = []
    for ch in str(text):
        if ch.isalnum() or ch in {"_", "-"}:
            out.append(ch)
        else:
            out.append("_")
    cleaned = "".join(out).strip("_")
    return cleaned or "unknown"


def _to_key(path: str, dataset_root: Path) -> str:
    return Path(path).resolve().relative_to(dataset_root.resolve()).as_posix()


def _load_gt_for_video(video_path: str) -> Dict[int, BBox]:
    ann = load_coco_vid(str(Path(video_path).with_suffix(".json")))
    frames = ann.get("frames", {})
    out: Dict[int, BBox] = {}
    for fi_raw, boxes in frames.items():
        if not boxes:
            continue
        fi = int(fi_raw)
        bb = boxes[0]
        if not isinstance(bb, (list, tuple)) or len(bb) != 4:
            continue
        out[fi] = (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))
    return out


def _estimate_video_diagonal(video_path: str, gt_frames: Dict[int, BBox]) -> float:
    width = 0.0
    height = 0.0
    if cv2 is not None:
        cap = cv2.VideoCapture(video_path)
        try:
            if cap.isOpened():
                width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0)
                height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)
        finally:
            cap.release()

    if width <= 1.0 or height <= 1.0:
        max_x = 0.0
        max_y = 0.0
        for bb in gt_frames.values():
            x, y, w, h = bb
            max_x = max(max_x, float(x + w))
            max_y = max(max_y, float(y + h))
        width = max(width, max_x)
        height = max(height, max_y)

    if width <= 1.0 or height <= 1.0:
        return 1000.0
    return float(math.hypot(width, height))


def _bbox_diagonal(bbox: BBox) -> float:
    return float(math.hypot(float(bbox[2]), float(bbox[3])))


def _build_thresholds(start: float, stop: float, step: float) -> List[float]:
    if step <= 0.0:
        raise ValueError("threshold step must be > 0")
    out: List[float] = []
    cur = start
    while cur <= stop + 1e-12:
        out.append(round(cur, 10))
        cur += step
    return out


def _split_train_val_subject_level(
    train_videos: Sequence[str],
    video_subjects: Dict[str, str],
    *,
    val_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    videos = list(train_videos)
    if len(videos) <= 1:
        return videos, []

    groups: Dict[str, List[str]] = {}
    for vp in videos:
        subject = str(video_subjects.get(vp, "unknown"))
        groups.setdefault(subject, []).append(vp)

    subjects = sorted(groups.keys())
    if len(subjects) <= 1:
        return videos, []

    rng = random.Random(seed)
    rng.shuffle(subjects)
    n_val_subjects = int(round(len(subjects) * max(0.0, min(0.9, val_ratio))))
    n_val_subjects = max(1, min(len(subjects) - 1, n_val_subjects))

    val_subjects = set(subjects[:n_val_subjects])
    train_split: List[str] = []
    val_split: List[str] = []
    for s in subjects:
        if s in val_subjects:
            val_split.extend(groups[s])
        else:
            train_split.extend(groups[s])

    if not train_split:
        return videos, []
    return sorted(train_split), sorted(val_split)


def _count_subjects(videos: Sequence[str], video_subjects: Dict[str, str]) -> int:
    return len({str(video_subjects.get(v, "unknown")) for v in videos})


def _make_model_config(
    spec: DetectorSpec,
    *,
    device: str,
    imgsz: int,
    iou: float,
    max_det: int,
    conf: float,
    min_confidence: float,
    train_enabled: bool,
    epochs: int,
    batch: int,
    lr0: float,
    patience: int,
    workers: int,
    use_clahe: bool,
    clahe_clip_limit: float,
    clahe_tile_grid_x: int,
    clahe_tile_grid_y: int,
) -> Dict[str, Any]:
    return {
        "weights": spec.weights,
        "conf": float(conf),
        "iou": float(iou),
        "imgsz": int(imgsz),
        "device": str(device),
        "max_det": int(max_det),
        "min_confidence": float(min_confidence),
        "fallback_last_prediction": False,
        "fallback_missing_interpolation": False,
        "train_enabled": bool(train_enabled),
        "epochs": int(epochs),
        "batch": int(batch),
        "lr0": float(lr0),
        "patience": int(patience),
        "workers": int(workers),
        "use_clahe": bool(use_clahe),
        "clahe_clip_limit": float(clahe_clip_limit),
        "clahe_tile_grid_x": int(clahe_tile_grid_x),
        "clahe_tile_grid_y": int(clahe_tile_grid_y),
    }


def _init_model(spec: DetectorSpec, config: Dict[str, Any]) -> Any:
    model: Any
    if spec.model_type == "YOLOv11":
        model = YOLOv11Model(config)
    elif spec.model_type == "RTDETRv2":
        model = RTDETRv2Model(config)
    else:
        raise ValueError(f"Unsupported model type: {spec.model_type}")

    # Align detector preprocessing with pipeline behavior when requested.
    if bool(config.get("use_clahe", False)):
        clip_limit = float(config.get("clahe_clip_limit", float(CLAHE.DEFAULT_CONFIG.get("clipLimit", 2.0))))
        tile_x = int(config.get("clahe_tile_grid_x", int(CLAHE.DEFAULT_CONFIG.get("tileGridSize", [8, 8])[0])))
        tile_y = int(config.get("clahe_tile_grid_y", int(CLAHE.DEFAULT_CONFIG.get("tileGridSize", [8, 8])[1])))
        tile_x = max(1, tile_x)
        tile_y = max(1, tile_y)
        if hasattr(model, "preprocs"):
            setattr(
                model,
                "preprocs",
                [CLAHE({"clipLimit": clip_limit, "tileGridSize": [tile_x, tile_y]})],
            )

    return model


def _serialise_raw_predictions(predictions: Iterable[Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in predictions:
        try:
            frame_index = int(getattr(p, "frame_index"))
            bbox_any = getattr(p, "bbox")
            if not isinstance(bbox_any, (list, tuple)) or len(bbox_any) != 4:
                continue
            bbox = [float(bbox_any[0]), float(bbox_any[1]), float(bbox_any[2]), float(bbox_any[3])]
            score_obj = getattr(p, "score", None)
            score = 0.0 if score_obj is None else float(score_obj)
            if not math.isfinite(score):
                score = 0.0
            rows.append(
                {
                    "frame_index": frame_index,
                    "bbox": bbox,
                    "score": score,
                    "is_fallback": bool(getattr(p, "is_fallback", False)),
                    "bbox_source": str(getattr(p, "bbox_source", "detector")),
                }
            )
        except Exception:
            continue
    return rows


def _load_cached_raw_predictions(path: Path) -> Dict[int, FrameRawPrediction]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[int, FrameRawPrediction] = {}
    if not isinstance(data, list):
        return out
    for row in data:
        if not isinstance(row, dict):
            continue
        try:
            fi = int(row.get("frame_index"))
            bb = row.get("bbox")
            if not isinstance(bb, (list, tuple)) or len(bb) != 4:
                continue
            bbox = (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))
            sc = float(row.get("score", 0.0))
            if not math.isfinite(sc):
                sc = 0.0
            is_fallback = bool(row.get("is_fallback", False))
            bbox_source = str(row.get("bbox_source", "detector")).lower()
            if bbox_source.startswith("missing"):
                is_fallback = True
            out[fi] = FrameRawPrediction(frame_index=fi, bbox=bbox, score=sc, is_fallback=is_fallback)
        except Exception:
            continue
    return out


def _predict_or_load_video(*, model: Any, video_path: str, cache_path: Path, reuse_predictions: bool) -> Dict[int, FrameRawPrediction]:
    if reuse_predictions and cache_path.is_file():
        return _load_cached_raw_predictions(cache_path)

    preds = model.predict(video_path)
    rows = _serialise_raw_predictions(preds)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return _load_cached_raw_predictions(cache_path)


def _compute_auroc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    if len(y_true) < 2:
        return float("nan")
    if len(set(int(v) for v in y_true)) < 2:
        return float("nan")

    if roc_auc_score is not None:
        try:
            return float(roc_auc_score(y_true, y_score))
        except Exception:
            return float("nan")

    # Fallback AUROC by rank-sum when sklearn is not available.
    pairs = sorted([(float(s), int(t)) for s, t in zip(y_score, y_true)], key=lambda x: x[0])
    n_pos = sum(t for _, t in pairs)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    rank_sum_pos = 0.0
    rank = 1
    idx = 0
    while idx < len(pairs):
        j = idx + 1
        while j < len(pairs) and pairs[j][0] == pairs[idx][0]:
            j += 1
        avg_rank = (rank + (rank + (j - idx) - 1)) / 2.0
        pos_in_block = sum(pairs[k][1] for k in range(idx, j))
        rank_sum_pos += avg_rank * pos_in_block
        rank += j - idx
        idx = j

    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _interpolate_missing_queries_with_backoff(
    *,
    known_preds: Sequence[FramePrediction],
    query_frame_indices: Sequence[int],
) -> List[FramePrediction]:
    frame_map: Dict[int, FramePrediction] = {}
    for pred in known_preds:
        fi = int(pred.frame_index)
        frame_map[fi] = pred

    if not frame_map:
        return []

    known_sorted = [frame_map[fi] for fi in sorted(frame_map.keys())]
    known_set = set(frame_map.keys())
    query_unique = sorted({int(fi) for fi in query_frame_indices})
    query_missing = [fi for fi in query_unique if fi not in known_set]
    if not query_missing:
        return []

    if len(known_sorted) >= 3:
        return cubic_clip_interpolate_predictions(
            known_sorted,
            query_frame_indices=query_unique,
            fill_all_queries=True,
            interpolated_bbox_source="interpolated_pchip",
        )

    known_frames = np.asarray([int(p.frame_index) for p in known_sorted], dtype=np.float64)
    known_boxes = np.asarray([list(p.bbox) for p in known_sorted], dtype=np.float64)
    dim_min = np.min(known_boxes, axis=0)
    dim_max = np.max(known_boxes, axis=0)
    query_array = np.asarray(query_missing, dtype=np.float64)

    if len(known_sorted) == 2:
        interp_dims: List[np.ndarray] = []
        for dim in range(4):
            values = np.interp(query_array, known_frames, known_boxes[:, dim])
            np.clip(values, float(dim_min[dim]), float(dim_max[dim]), out=values)
            interp_dims.append(values)

        out: List[FramePrediction] = []
        for i, frame_value in enumerate(query_array.tolist()):
            out.append(
                FramePrediction(
                    frame_index=int(frame_value),
                    bbox=(
                        float(interp_dims[0][i]),
                        float(interp_dims[1][i]),
                        float(interp_dims[2][i]),
                        float(interp_dims[3][i]),
                    ),
                    score=None,
                    is_fallback=True,
                    bbox_source="interpolated_linear",
                )
            )
        return out

    bbox_single = known_sorted[0].bbox
    return [
        FramePrediction(
            frame_index=int(frame_idx),
            bbox=(float(bbox_single[0]), float(bbox_single[1]), float(bbox_single[2]), float(bbox_single[3])),
            score=None,
            is_fallback=True,
            bbox_source="interpolated_hold",
        )
        for frame_idx in query_missing
    ]


def _build_eval_predictions_for_threshold(
    *,
    gt_frames: Dict[int, BBox],
    pred_frames: Dict[int, FrameRawPrediction],
    threshold: float,
    min_interp_points: int,
) -> Tuple[Dict[int, BBox], Dict[int, float], set[int], int]:
    frame_indices = sorted(int(fi) for fi in gt_frames.keys())
    known_preds: List[FramePrediction] = []
    score_after_thr: Dict[int, float] = {}

    for raw_pred in pred_frames.values():
        score = float(raw_pred.score) if math.isfinite(float(raw_pred.score)) else 0.0
        keep = (not bool(raw_pred.is_fallback)) and _is_valid_bbox(raw_pred.bbox) and score >= float(threshold)
        if not keep:
            continue
        known_preds.append(
            FramePrediction(
                frame_index=int(raw_pred.frame_index),
                bbox=raw_pred.bbox,
                score=score,
                confidence=score,
                is_fallback=False,
                bbox_source="detector",
            )
        )

    for fi in frame_indices:
        raw_pred = pred_frames.get(fi)
        if raw_pred is None:
            score_after_thr[fi] = 0.0
            continue

        score = float(raw_pred.score) if math.isfinite(float(raw_pred.score)) else 0.0
        keep = (not bool(raw_pred.is_fallback)) and _is_valid_bbox(raw_pred.bbox) and score >= float(threshold)
        if keep:
            score_after_thr[fi] = score
        else:
            score_after_thr[fi] = 0.0

    known_set = {int(p.frame_index) for p in known_preds}
    non_interpolable_missing = {fi for fi in frame_indices if fi not in known_set}

    pred_bbox_map: Dict[int, BBox] = {int(p.frame_index): p.bbox for p in known_preds}

    effective_min_interp_points = max(1, int(min_interp_points))
    if len(known_preds) >= effective_min_interp_points:
        interp = _interpolate_missing_queries_with_backoff(
            known_preds=known_preds,
            query_frame_indices=frame_indices,
        )
        for p in interp:
            pred_bbox_map[int(p.frame_index)] = p.bbox
        non_interpolable_missing = {fi for fi in frame_indices if fi not in pred_bbox_map}

    return pred_bbox_map, score_after_thr, non_interpolable_missing, len(known_preds)


def _evaluate_fold_metrics(
    *,
    test_videos: Sequence[str],
    gt_map_by_video: Dict[str, Dict[int, BBox]],
    pred_map_by_video: Dict[str, Dict[int, FrameRawPrediction]],
    thresholds: Sequence[float],
    penalty_ce_mode: str,
    penalty_ce_by_video: Dict[str, float],
    min_interp_points: int,
) -> Dict[float, Dict[str, float]]:
    out: Dict[float, Dict[str, float]] = {}

    for thr in thresholds:
        ious: List[float] = []
        ces: List[float] = []
        y_true_auc: List[int] = []
        y_score_auc: List[float] = []
        non_interpolable_frames = 0.0
        known_points_total = 0.0

        for video_path in test_videos:
            gt_frames = gt_map_by_video.get(video_path, {})
            pred_frames = pred_map_by_video.get(video_path, {})
            penalty_ce = float(penalty_ce_by_video.get(video_path, 1000.0))

            pred_bbox_map, score_after_thr, non_interp_missing, known_points = _build_eval_predictions_for_threshold(
                gt_frames=gt_frames,
                pred_frames=pred_frames,
                threshold=float(thr),
                min_interp_points=int(min_interp_points),
            )
            known_points_total += float(known_points)
            non_interpolable_frames += float(len(non_interp_missing))

            for fi, gt_bbox in gt_frames.items():
                fi_int = int(fi)
                pred_bbox = pred_bbox_map.get(fi_int)
                if pred_bbox is not None and _is_valid_bbox(pred_bbox):
                    iou_thr = _bbox_iou(pred_bbox, gt_bbox)
                    ce_thr = _center_error(pred_bbox, gt_bbox)
                else:
                    iou_thr = 0.0
                    if str(penalty_ce_mode) == "gt-bbox-diagonal":
                        ce_thr = _bbox_diagonal(gt_bbox)
                    else:
                        ce_thr = penalty_ce

                ious.append(float(iou_thr))
                ces.append(float(ce_thr))
                y_true_auc.append(1 if iou_thr >= 0.5 else 0)
                y_score_auc.append(float(score_after_thr.get(fi_int, 0.0)))

        out[thr] = {
            "iou": float(mean(ious)) if ious else float("nan"),
            "ce": float(mean(ces)) if ces else float("nan"),
            "auroc": _compute_auroc(y_true_auc, y_score_auc),
            "n_eval_frames": float(len(ious)),
            "n_gt_frames": float(len(y_true_auc)),
            "n_non_interpolable_frames": float(non_interpolable_frames),
            "known_points_mean": float(known_points_total / max(1, len(test_videos))),
        }

    return out


def _count_completed_epochs(results_csv: Path) -> int:
    if not results_csv.is_file():
        return 0
    try:
        with results_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception:
        return 0

    if not rows:
        return 0

    if "epoch" in rows[0]:
        epochs: List[int] = []
        for row in rows:
            try:
                epochs.append(int(float(row.get("epoch", "0"))))
            except Exception:
                continue
        if epochs:
            return int(len(set(epochs)))

    return int(len(rows))


def _train_fold_model(
    *,
    spec: DetectorSpec,
    fold_subject: str,
    train_videos: Sequence[str],
    val_videos: Sequence[str],
    ann_by_video: Dict[str, Any],
    output_dir: Path,
    model_cfg: Dict[str, Any],
    seed: int,
    reuse_checkpoints: bool,
    expected_epochs: int,
) -> str:
    fold_tag = _safe_tag(fold_subject)
    train_dir = output_dir / "train" / spec.key / fold_tag
    marker_path = train_dir / "train_result.json"

    def _resolve_run_dir_from_best(best_ckpt: Path) -> Optional[Path]:
        if not best_ckpt.is_file():
            return None
        if best_ckpt.parent.name != "weights":
            return None
        return best_ckpt.parent.parent

    def _validate_reuse_marker(marker_payload: Dict[str, Any]) -> Optional[str]:
        best_ckpt_raw = str(marker_payload.get("best_ckpt", "")).strip()
        if not best_ckpt_raw:
            return None
        best_ckpt = Path(best_ckpt_raw)
        if not best_ckpt.is_file():
            return None

        train_result = marker_payload.get("train_result")
        if not isinstance(train_result, dict):
            return None
        status = str(train_result.get("status", "")).lower()
        if status != "ok":
            return None

        marker_hparams = marker_payload.get("train_hparams")
        if isinstance(marker_hparams, dict) and marker_hparams.get("epochs") is not None:
            try:
                marker_epochs = int(marker_hparams.get("epochs"))
            except Exception:
                return None
            if marker_epochs != int(expected_epochs):
                return None

        expected_use_clahe = bool(model_cfg.get("use_clahe", False))
        expected_clahe_clip = float(model_cfg.get("clahe_clip_limit", float(CLAHE.DEFAULT_CONFIG.get("clipLimit", 2.0))))
        expected_clahe_grid = [
            int(model_cfg.get("clahe_tile_grid_x", int(CLAHE.DEFAULT_CONFIG.get("tileGridSize", [8, 8])[0]))),
            int(model_cfg.get("clahe_tile_grid_y", int(CLAHE.DEFAULT_CONFIG.get("tileGridSize", [8, 8])[1]))),
        ]
        expected_clahe_grid = [max(1, expected_clahe_grid[0]), max(1, expected_clahe_grid[1])]

        if expected_use_clahe:
            if not isinstance(marker_hparams, dict):
                return None
            if bool(marker_hparams.get("use_clahe", False)) is not True:
                return None
            try:
                marker_clip = float(marker_hparams.get("clahe_clip_limit"))
                marker_grid_raw = marker_hparams.get("clahe_tile_grid")
                marker_grid = [int(marker_grid_raw[0]), int(marker_grid_raw[1])]
            except Exception:
                return None
            if abs(marker_clip - expected_clahe_clip) > 1e-12:
                return None
            if marker_grid != expected_clahe_grid:
                return None
        else:
            if isinstance(marker_hparams, dict) and "use_clahe" in marker_hparams and bool(marker_hparams.get("use_clahe", False)):
                return None

        completed_epochs = 0
        try:
            completed_epochs = int(marker_payload.get("completed_epochs", 0) or 0)
        except Exception:
            completed_epochs = 0

        if completed_epochs <= 0:
            run_dir = _resolve_run_dir_from_best(best_ckpt)
            if run_dir is not None:
                completed_epochs = _count_completed_epochs(run_dir / "results.csv")

        if completed_epochs < int(expected_epochs):
            return None

        return str(best_ckpt.resolve())

    if reuse_checkpoints and marker_path.is_file():
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if isinstance(marker, dict):
                reusable_ckpt = _validate_reuse_marker(marker)
                if reusable_ckpt:
                    print(f"[Reuse] completed fold checkpoint: detector={spec.key} subject={fold_subject}")
                    return reusable_ckpt
                print(
                    f"[Reuse] checkpoint rejected (possibly interrupted or mismatched epochs), retraining: "
                    f"detector={spec.key} subject={fold_subject}"
                )
        except Exception:
            pass

    model = _init_model(spec, model_cfg)
    train_ds = SimpleDataset(list(train_videos), ann_by_video)
    if len(train_ds) <= 0:
        raise RuntimeError(f"No annotated training videos for fold {fold_subject}")

    val_ds = None
    if val_videos:
        val_ds = SimpleDataset(list(val_videos), ann_by_video)
        if len(val_ds) <= 0:
            val_ds = None

    train_ret = model.train(train_ds, val_ds, seed=int(seed), output_dir=str(train_dir))
    if not isinstance(train_ret, dict):
        raise RuntimeError(f"Unexpected train return for {spec.key} fold={fold_subject}: {type(train_ret)}")

    status = str(train_ret.get("status", "")).lower()
    if status != "ok":
        raise RuntimeError(
            f"Training failed for {spec.key} fold={fold_subject}: "
            f"status={train_ret.get('status')} error={train_ret.get('error')}"
        )

    best_ckpt_raw = str(train_ret.get("best_ckpt", "")).strip()
    best_ckpt = Path(best_ckpt_raw) if best_ckpt_raw else Path()
    if not best_ckpt_raw or not best_ckpt.is_file():
        fallback = train_dir / model.name / "weights" / "best.pt"
        if fallback.is_file():
            best_ckpt = fallback
        else:
            raise RuntimeError(f"best checkpoint not found for {spec.key} fold={fold_subject}")

    run_dir = best_ckpt.parent.parent if best_ckpt.parent.name == "weights" else None
    completed_epochs = _count_completed_epochs((run_dir / "results.csv") if run_dir is not None else Path())

    train_dir.mkdir(parents=True, exist_ok=True)
    marker_payload = {
        "detector": spec.key,
        "fold_subject": fold_subject,
        "best_ckpt": str(best_ckpt.resolve()),
        "completed_epochs": int(completed_epochs),
        "train_hparams": {
            "epochs": int(expected_epochs),
            "batch": int(model_cfg.get("batch", 0)),
            "lr0": float(model_cfg.get("lr0", 0.0)),
            "patience": int(model_cfg.get("patience", 0)),
            "workers": int(model_cfg.get("workers", 0)),
            "use_clahe": bool(model_cfg.get("use_clahe", False)),
            "clahe_clip_limit": float(model_cfg.get("clahe_clip_limit", float(CLAHE.DEFAULT_CONFIG.get("clipLimit", 2.0)))),
            "clahe_tile_grid": [
                int(model_cfg.get("clahe_tile_grid_x", int(CLAHE.DEFAULT_CONFIG.get("tileGridSize", [8, 8])[0]))),
                int(model_cfg.get("clahe_tile_grid_y", int(CLAHE.DEFAULT_CONFIG.get("tileGridSize", [8, 8])[1]))),
            ],
        },
        "artifacts": {
            "run_dir": str(run_dir.resolve()) if run_dir is not None and run_dir.exists() else None,
            "results_csv": str((run_dir / "results.csv").resolve())
            if run_dir is not None and (run_dir / "results.csv").is_file()
            else None,
            "last_ckpt": str((run_dir / "weights" / "last.pt").resolve())
            if run_dir is not None and (run_dir / "weights" / "last.pt").is_file()
            else None,
        },
        "train_videos": len(train_videos),
        "val_videos": len(val_videos),
        "train_result": train_ret,
    }
    marker_path.write_text(json.dumps(marker_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(best_ckpt.resolve())


def _predict_fold_videos(
    *,
    spec: DetectorSpec,
    fold_subject: str,
    checkpoint_path: str,
    test_videos: Sequence[str],
    dataset_root: Path,
    output_dir: Path,
    model_cfg: Dict[str, Any],
    reuse_predictions: bool,
) -> Dict[str, Dict[int, FrameRawPrediction]]:
    infer_model = _init_model(spec, model_cfg)
    infer_model.load_checkpoint(checkpoint_path)

    fold_tag = _safe_tag(fold_subject)
    pred_map_by_video: Dict[str, Dict[int, FrameRawPrediction]] = {}
    for video_path in test_videos:
        rel = _to_key(video_path, dataset_root)
        cache_path = output_dir / "raw_predictions" / spec.key / fold_tag / f"{rel}.json"
        pred_map_by_video[video_path] = _predict_or_load_video(
            model=infer_model,
            video_path=video_path,
            cache_path=cache_path,
            reuse_predictions=reuse_predictions,
        )
    return pred_map_by_video


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path

    def _dump(to_path: Path) -> None:
        with to_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in rows:
                w.writerow(row)

    try:
        _dump(target)
    except PermissionError:
        target = path.with_name(f"{path.stem}_recomputed{path.suffix}")
        _dump(target)
        print(f"[Warn] File locked, wrote recomputed CSV instead: {target}")

    return target


def _format_metric(mu: float, sd: float) -> str:
    if math.isnan(mu):
        return "nan"
    if math.isnan(sd):
        return f"{mu:.6f}"
    return f"{mu:.6f} +/- {sd:.6f}"


def _build_markdown_table(label: str, rows: Sequence[Dict[str, Any]]) -> str:
    lines = [f"## {label}", "", "| confidence | IoU mean+/-std | CE mean+/-std | AUROC mean+/-std |", "|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(
            "| "
            f"{float(row['confidence']):.2f} | "
            f"{_format_metric(float(row['iou_mean']), float(row['iou_std']))} | "
            f"{_format_metric(float(row['ce_mean']), float(row['ce_std']))} | "
            f"{_format_metric(float(row['auroc_mean']), float(row['auroc_std']))} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="LOSO retrain + confidence-threshold sweep for three detectors on extendclen.")
    ap.add_argument("--dataset-root", default="dataset/extendclen")
    ap.add_argument("--output-dir", default="")
    ap.add_argument("--split-mode", choices=["loso", "subject_level"], default="loso")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr0", type=float, default=0.01)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--holdout-train-ratio", type=float, default=0.7)
    ap.add_argument("--holdout-val-ratio", type=float, default=0.15)
    ap.add_argument("--holdout-test-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--threshold-start", type=float, default=0.0)
    ap.add_argument("--threshold-stop", type=float, default=0.95)
    ap.add_argument("--threshold-step", type=float, default=0.05)
    ap.add_argument("--min-interp-points", type=int, default=1)
    ap.add_argument("--penalty-ce-mode", choices=["gt-bbox-diagonal", "image-diagonal", "fixed"], default="gt-bbox-diagonal")
    ap.add_argument("--penalty-ce-value", type=float, default=1000.0)
    ap.add_argument("--use-clahe", action="store_true")
    ap.add_argument("--clahe-clip-limit", type=float, default=float(CLAHE.DEFAULT_CONFIG.get("clipLimit", 2.0)))
    ap.add_argument("--clahe-tile-grid-x", type=int, default=int(CLAHE.DEFAULT_CONFIG.get("tileGridSize", [8, 8])[0]))
    ap.add_argument("--clahe-tile-grid-y", type=int, default=int(CLAHE.DEFAULT_CONFIG.get("tileGridSize", [8, 8])[1]))
    ap.add_argument("--no-reuse-checkpoints", action="store_true")
    ap.add_argument("--no-reuse-predictions", action="store_true")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else Path("results")
        / f"extendclen_detector_conf_{args.split_mode}_retrain{int(args.epochs)}_{ts}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = _build_thresholds(args.threshold_start, args.threshold_stop, args.threshold_step)

    detector_specs = [
        DetectorSpec(
            key="yolov11_large",
            label="YOLOv11 large",
            model_type="YOLOv11",
            weights="models/detection/yolo11l.pt",
        ),
        DetectorSpec(
            key="yolo26_large",
            label="YOLO26 large",
            model_type="YOLOv11",
            weights="models/detection/yolo26l.pt",
        ),
        DetectorSpec(
            key="rtdetrv2_large",
            label="RTDETRv2 large",
            model_type="RTDETRv2",
            weights="rtdetr-l.pt",
        ),
    ]

    dm = COCOJsonDatasetManager(str(dataset_root))
    folds: List[Dict[str, Any]]
    if args.split_mode == "loso":
        folds = list(dm.loso())
    else:
        holdout_train = float(args.holdout_train_ratio)
        holdout_val = float(args.holdout_val_ratio)
        holdout_test = float(args.holdout_test_ratio)
        ratio_sum = holdout_train + holdout_val + holdout_test
        if ratio_sum <= 0.0:
            raise ValueError("holdout ratios must sum to > 0")
        holdout_train /= ratio_sum
        holdout_val /= ratio_sum
        holdout_test /= ratio_sum
        split_ds = dm.split(
            method="subject_level",
            seed=int(args.seed),
            ratios=(holdout_train, holdout_val, holdout_test),
        )
        holdout_train_videos = [str(vp) for vp, _ in split_ds["train"].items]
        holdout_val_videos = [str(vp) for vp, _ in split_ds["val"].items]
        holdout_test_videos = [str(vp) for vp, _ in split_ds["test"].items]
        print(
            "[Split] subject_level holdout: "
            f"total_subjects={_count_subjects(dm.videos, dm.video_subjects)} total_videos={len(dm.videos)}; "
            f"train_subjects={_count_subjects(holdout_train_videos, dm.video_subjects)} "
            f"train_videos={len(holdout_train_videos)}; "
            f"val_subjects={_count_subjects(holdout_val_videos, dm.video_subjects)} "
            f"val_videos={len(holdout_val_videos)}; "
            f"test_subjects={_count_subjects(holdout_test_videos, dm.video_subjects)} "
            f"test_videos={len(holdout_test_videos)}"
        )
        folds = [
            {
                "subject": "subject_level_holdout",
                "train": holdout_train_videos,
                "val": holdout_val_videos,
                "test": holdout_test_videos,
            }
        ]
    if not folds:
        raise RuntimeError(f"No folds found for split mode: {args.split_mode}")

    run_cfg = {
        "dataset_root": str(dataset_root),
        "split": str(args.split_mode),
        "n_folds": len(folds),
        "split_config": {
            "val_ratio": float(args.val_ratio),
            "holdout_train_ratio": float(args.holdout_train_ratio),
            "holdout_val_ratio": float(args.holdout_val_ratio),
            "holdout_test_ratio": float(args.holdout_test_ratio),
        },
        "detectors": [d.__dict__ for d in detector_specs],
        "train_hparams": {
            "epochs": int(args.epochs),
            "batch": int(args.batch),
            "lr0": float(args.lr0),
            "patience": int(args.patience),
            "workers": int(args.workers),
            "val_ratio": float(args.val_ratio),
            "seed": int(args.seed),
        },
        "infer_hparams": {
            "device": str(args.device),
            "imgsz": int(args.imgsz),
            "iou": float(args.iou),
            "max_det": int(args.max_det),
            "detector_conf": 0.0,
            "min_confidence": 0.0,
        },
        "thresholds": thresholds,
        "interpolation": {
            "method": "pchip_linear_hold_backoff",
            "min_interp_points": int(args.min_interp_points),
            "known_points_scope": "all_predicted_frames",
            "backoff": {
                "points_ge_3": "pchip",
                "points_eq_2": "linear",
                "points_eq_1": "hold",
            },
        },
        "penalty": {
            "ce_mode": str(args.penalty_ce_mode),
            "ce_value": float(args.penalty_ce_value),
        },
        "preprocessing": {
            "detector_preproc_scheme": "A",
            "use_clahe": bool(args.use_clahe),
            "clahe_params": {
                "clipLimit": float(args.clahe_clip_limit),
                "tileGridSize": [int(args.clahe_tile_grid_x), int(args.clahe_tile_grid_y)],
            },
        },
        "reuse_checkpoints": not bool(args.no_reuse_checkpoints),
        "reuse_predictions": not bool(args.no_reuse_predictions),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    # Prepare GT once.
    gt_map_by_video: Dict[str, Dict[int, BBox]] = {}
    penalty_ce_by_video: Dict[str, float] = {}
    for video_path in dm.videos:
        gt_frames = _load_gt_for_video(video_path)
        gt_map_by_video[video_path] = gt_frames
        if str(args.penalty_ce_mode) == "fixed":
            penalty_ce_by_video[video_path] = float(args.penalty_ce_value)
        elif str(args.penalty_ce_mode) == "image-diagonal":
            penalty_ce_by_video[video_path] = _estimate_video_diagonal(video_path, gt_frames)
        else:
            # gt-bbox-diagonal is frame-dependent and computed during evaluation.
            penalty_ce_by_video[video_path] = 0.0

    aggregate_rows_for_json: Dict[str, List[Dict[str, Any]]] = {}
    train_fold_reports: Dict[str, List[Dict[str, Any]]] = {}

    for spec in detector_specs:
        print(f"[Run] Detector: {spec.label} ({spec.weights})")
        fold_rows: List[Dict[str, Any]] = []
        per_fold_by_thr: Dict[float, List[Dict[str, Any]]] = {thr: [] for thr in thresholds}
        train_fold_reports[spec.key] = []

        for fold_idx, fold in enumerate(folds, start=1):
            subject = str(fold.get("subject") or f"fold{fold_idx}")
            train_videos = list(fold.get("train") or [])
            test_videos = list(fold.get("test") or [])
            if not train_videos or not test_videos:
                continue

            preset_val_videos = list(fold.get("val") or [])
            if preset_val_videos:
                fit_train_videos = sorted(train_videos)
                fit_val_videos = sorted(preset_val_videos)
            else:
                fit_train_videos, fit_val_videos = _split_train_val_subject_level(
                    train_videos,
                    dm.video_subjects,
                    val_ratio=float(args.val_ratio),
                    seed=int(args.seed) + fold_idx,
                )

            print(
                f"  [Fold {fold_idx}/{len(folds)}] subject={subject} "
                f"train_videos={len(fit_train_videos)} "
                f"val_videos={len(fit_val_videos)} "
                f"test_videos={len(test_videos)}"
            )

            train_model_cfg = _make_model_config(
                spec,
                device=str(args.device),
                imgsz=int(args.imgsz),
                iou=float(args.iou),
                max_det=int(args.max_det),
                conf=0.0,
                min_confidence=0.0,
                train_enabled=True,
                epochs=int(args.epochs),
                batch=int(args.batch),
                lr0=float(args.lr0),
                patience=int(args.patience),
                workers=int(args.workers),
                use_clahe=bool(args.use_clahe),
                clahe_clip_limit=float(args.clahe_clip_limit),
                clahe_tile_grid_x=int(args.clahe_tile_grid_x),
                clahe_tile_grid_y=int(args.clahe_tile_grid_y),
            )

            best_ckpt = _train_fold_model(
                spec=spec,
                fold_subject=subject,
                train_videos=fit_train_videos,
                val_videos=fit_val_videos,
                ann_by_video=dm.ann_by_video,
                output_dir=output_dir,
                model_cfg=train_model_cfg,
                seed=int(args.seed),
                reuse_checkpoints=(not bool(args.no_reuse_checkpoints)),
                expected_epochs=int(args.epochs),
            )
            train_fold_reports[spec.key].append(
                {
                    "fold": fold_idx,
                    "subject": subject,
                    "train_videos": len(fit_train_videos),
                    "val_videos": len(fit_val_videos),
                    "test_videos": len(test_videos),
                    "best_ckpt": best_ckpt,
                }
            )

            infer_model_cfg = _make_model_config(
                spec,
                device=str(args.device),
                imgsz=int(args.imgsz),
                iou=float(args.iou),
                max_det=int(args.max_det),
                conf=0.0,
                min_confidence=0.0,
                train_enabled=False,
                epochs=int(args.epochs),
                batch=int(args.batch),
                lr0=float(args.lr0),
                patience=int(args.patience),
                workers=int(args.workers),
                use_clahe=bool(args.use_clahe),
                clahe_clip_limit=float(args.clahe_clip_limit),
                clahe_tile_grid_x=int(args.clahe_tile_grid_x),
                clahe_tile_grid_y=int(args.clahe_tile_grid_y),
            )
            pred_map_by_video = _predict_fold_videos(
                spec=spec,
                fold_subject=subject,
                checkpoint_path=best_ckpt,
                test_videos=test_videos,
                dataset_root=dataset_root,
                output_dir=output_dir,
                model_cfg=infer_model_cfg,
                reuse_predictions=(not bool(args.no_reuse_predictions)),
            )

            fold_metrics = _evaluate_fold_metrics(
                test_videos=test_videos,
                gt_map_by_video=gt_map_by_video,
                pred_map_by_video=pred_map_by_video,
                thresholds=thresholds,
                penalty_ce_mode=str(args.penalty_ce_mode),
                penalty_ce_by_video=penalty_ce_by_video,
                min_interp_points=max(1, int(args.min_interp_points)),
            )

            for thr in thresholds:
                metric = fold_metrics[thr]
                row = {
                    "detector": spec.key,
                    "fold": fold_idx,
                    "subject": subject,
                    "confidence": thr,
                    "iou": metric["iou"],
                    "ce": metric["ce"],
                    "auroc": metric["auroc"],
                    "n_eval_frames": int(metric["n_eval_frames"]),
                    "n_gt_frames": int(metric["n_gt_frames"]),
                    "n_non_interpolable_frames": int(metric["n_non_interpolable_frames"]),
                    "known_points_mean": float(metric["known_points_mean"]),
                }
                fold_rows.append(row)
                per_fold_by_thr[thr].append(row)

        fold_csv = output_dir / f"{spec.key}_fold_metrics.csv"
        _write_csv(
            fold_csv,
            fold_rows,
            [
                "detector",
                "fold",
                "subject",
                "confidence",
                "iou",
                "ce",
                "auroc",
                "n_eval_frames",
                "n_gt_frames",
                "n_non_interpolable_frames",
                "known_points_mean",
            ],
        )

        summary_rows: List[Dict[str, Any]] = []
        for thr in thresholds:
            rows = per_fold_by_thr[thr]
            iou_vals = [float(r["iou"]) for r in rows]
            ce_vals = [float(r["ce"]) for r in rows]
            auroc_vals = [float(r["auroc"]) for r in rows if not math.isnan(float(r["auroc"]))]
            eval_frames_vals = [float(r["n_eval_frames"]) for r in rows]
            gt_frames_vals = [float(r["n_gt_frames"]) for r in rows]
            non_interp_vals = [float(r["n_non_interpolable_frames"]) for r in rows]
            known_points_vals = [float(r["known_points_mean"]) for r in rows]

            summary_rows.append(
                {
                    "detector": spec.key,
                    "confidence": thr,
                    "n_folds": len(rows),
                    "n_valid_iou_folds": _finite_count(iou_vals),
                    "n_valid_ce_folds": _finite_count(ce_vals),
                    "eval_frames_mean": _safe_mean(eval_frames_vals),
                    "gt_frames_mean": _safe_mean(gt_frames_vals),
                    "non_interpolable_frames_mean": _safe_mean(non_interp_vals),
                    "known_points_mean": _safe_mean(known_points_vals),
                    "iou_mean": _safe_mean(iou_vals),
                    "iou_std": _safe_std(iou_vals),
                    "ce_mean": _safe_mean(ce_vals),
                    "ce_std": _safe_std(ce_vals),
                    "auroc_mean": _safe_mean(auroc_vals),
                    "auroc_std": _safe_std(auroc_vals),
                }
            )

        summary_csv = output_dir / f"{spec.key}_summary_table.csv"
        _write_csv(
            summary_csv,
            summary_rows,
            [
                "detector",
                "confidence",
                "n_folds",
                "n_valid_iou_folds",
                "n_valid_ce_folds",
                "eval_frames_mean",
                "gt_frames_mean",
                "non_interpolable_frames_mean",
                "known_points_mean",
                "iou_mean",
                "iou_std",
                "ce_mean",
                "ce_std",
                "auroc_mean",
                "auroc_std",
            ],
        )
        aggregate_rows_for_json[spec.key] = summary_rows

    report_lines = [
        "# Extendclen Detector Confidence Sweep (Retrain)",
        "",
        "- Dataset: dataset/extendclen",
        (
            "- Split: LOSO (subject as test fold)"
            if args.split_mode == "loso"
            else "- Split: Subject-level holdout (subject buckets for train/val/test)"
        ),
        f"- Thresholds: {thresholds[0]:.2f} .. {thresholds[-1]:.2f} (step {args.threshold_step:.2f})",
        (
            "- Train hyperparameters: "
            f"epochs={int(args.epochs)}, batch={int(args.batch)}, lr0={float(args.lr0)}, "
            f"patience={int(args.patience)}, workers={int(args.workers)}, val_ratio={float(args.val_ratio):.2f}"
        ),
        (
            "- Infer hyperparameters: "
            f"device={args.device}, imgsz={int(args.imgsz)}, iou={float(args.iou)}, "
            f"max_det={int(args.max_det)}, detector_conf=0.0"
        ),
        (
            "- Missing handling: low-confidence detections become missing, then interpolation backoff is applied "
            "(PCHIP for >=3 known points, linear for 2 points, hold for 1 point); "
            f"interpolation is enabled when known points >= {max(1, int(args.min_interp_points))}."
        ),
        (
            "- Known-points counting: threshold-passed non-fallback detections are counted across all predicted frames "
            "(not limited to GT-annotated frames)."
        ),
        (
            "- Penalty: non-interpolable missing samples receive CE penalty by selected mode "
            f"(mode={args.penalty_ce_mode}; fixed value={float(args.penalty_ce_value):.3f} when mode=fixed)."
        ),
        (
            "- Detector preprocessing: "
            f"CLAHE={'on' if bool(args.use_clahe) else 'off'} "
            f"(clipLimit={float(args.clahe_clip_limit):.3f}, "
            f"tileGridSize=[{int(args.clahe_tile_grid_x)},{int(args.clahe_tile_grid_y)}], scheme=A/global)."
        ),
        "- AUROC is computed from frame-level success label (IoU>=0.5) vs thresholded detector confidence score.",
        "",
    ]

    label_map = {
        "yolov11_large": "YOLOv11 large",
        "yolo26_large": "YOLO26 large",
        "rtdetrv2_large": "RTDETRv2 large",
    }
    for key in ["yolov11_large", "yolo26_large", "rtdetrv2_large"]:
        report_lines.append(_build_markdown_table(label_map[key], aggregate_rows_for_json.get(key, [])))

    report_path = output_dir / "summary_tables.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    summary_json = {
        "dataset_root": str(dataset_root),
        "split": str(args.split_mode),
        "split_config": {
            "val_ratio": float(args.val_ratio),
            "holdout_train_ratio": float(args.holdout_train_ratio),
            "holdout_val_ratio": float(args.holdout_val_ratio),
            "holdout_test_ratio": float(args.holdout_test_ratio),
        },
        "thresholds": thresholds,
        "train_hyperparameters": {
            "epochs": int(args.epochs),
            "batch": int(args.batch),
            "lr0": float(args.lr0),
            "patience": int(args.patience),
            "workers": int(args.workers),
            "val_ratio": float(args.val_ratio),
        },
        "infer_hyperparameters": {
            "device": str(args.device),
            "imgsz": int(args.imgsz),
            "iou": float(args.iou),
            "max_det": int(args.max_det),
            "detector_conf": 0.0,
        },
        "interpolation": {
            "method": "pchip_linear_hold_backoff",
            "min_interp_points": int(args.min_interp_points),
            "known_points_scope": "all_predicted_frames",
            "backoff": {
                "points_ge_3": "pchip",
                "points_eq_2": "linear",
                "points_eq_1": "hold",
            },
        },
        "penalty": {
            "ce_mode": str(args.penalty_ce_mode),
            "ce_value": float(args.penalty_ce_value),
        },
        "preprocessing": {
            "detector_preproc_scheme": "A",
            "use_clahe": bool(args.use_clahe),
            "clahe_params": {
                "clipLimit": float(args.clahe_clip_limit),
                "tileGridSize": [int(args.clahe_tile_grid_x), int(args.clahe_tile_grid_y)],
            },
        },
        "train_folds": train_fold_reports,
        "tables": aggregate_rows_for_json,
    }
    summary_json_path = output_dir / "summary_tables.json"
    summary_json_path.write_text(json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_dir": str(output_dir.resolve()),
                "report": str(report_path.resolve()),
                "summary_json": str(summary_json_path.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
