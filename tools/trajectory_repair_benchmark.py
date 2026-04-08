from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tracking.classification.trajectory_filter import (  # noqa: E402
    _adaptive_savgol,
    filter_bbox_hampel_only,
    hampel_then_pchip_1d,
    pchip_interpolate_1d,
)

try:
    from scipy.interpolate import BSpline, UnivariateSpline
except Exception:  # pragma: no cover
    BSpline = None
    UnivariateSpline = None

try:
    from tabpfn import TabPFNRegressor
except Exception:  # pragma: no cover
    TabPFNRegressor = None


DEFAULT_DATASET_ROOT = Path("dataset/merged_extend")
DEFAULT_OUTPUT_DIR = Path("results/trajectory_repair_benchmark")
DEFAULT_SEED = 3407
PIPE_HAMPEL_MACRO_HW = 25
PIPE_HAMPEL_MICRO_HW = 10


@dataclass(frozen=True)
class TrajectoryRecord:
    trajectory_id: str
    source_json: str
    video_name: str
    frame_index: np.ndarray
    gt_bbox_dense: np.ndarray
    gt_bbox_observed: np.ndarray
    observed_mask: np.ndarray
    video_path: str | None = None
    raw_bbox_dense: np.ndarray | None = None
    raw_observed_mask: np.ndarray | None = None
    raw_corrupt_mask: np.ndarray | None = None
    case_window: tuple[int, int] | None = None


@dataclass(frozen=True)
class BenchmarkConfig:
    dataset_root: str = str(DEFAULT_DATASET_ROOT)
    train_dataset_root: str = ""
    test_dataset_root: str = ""
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    seed: int = DEFAULT_SEED
    train_ratio: float = 0.8
    target_test_cases: int = 50
    cases_per_trajectory: int = 3
    max_corruption_ratio: float = 0.30
    max_trajectories: int = 0
    raw_source: str = "synthetic"
    detector_checkpoint: str = ""
    detector_experiment_dir: str = ""
    detector_cache_root: str = ""
    detector_video_extensions: str = ".avi,.wmv,.mp4,.mov,.mkv"
    real_case_window: int = 64
    corrupt_iou_threshold: float = 0.5
    tabpfn_device: str = "auto"
    tabpfn_max_train_rows: int = 4096
    mild_sg_window: int = 5
    mild_sg_polyorder: int = 2
    baseline_sg_window: int = 7
    baseline_sg_polyorder: int = 2
    bspline_degree: int = 3
    bspline_knots: int = 8
    anomaly_residual_quantile: float = 0.80
    robust_iterations: int = 3
    raw_noise_xy_ratio: float = 0.03
    raw_noise_wh_ratio: float = 0.02
    lowess_frac: float = 0.5
    lmeds_trials: int = 500


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _normalise_raw_source(raw_source: str) -> str:
    src = str(raw_source or "synthetic").strip().lower()
    return "detector" if src in {"detector", "real", "detector_raw", "detector-real"} else "synthetic"


def _iter_video_extensions(csv_text: str) -> tuple[str, ...]:
    items = [x.strip().lower() for x in str(csv_text or "").split(",")]
    exts = [x if x.startswith(".") else f".{x}" for x in items if x]
    return tuple(exts or [".avi", ".wmv", ".mp4", ".mov", ".mkv"])


def _bbox_to_center_xywh(bbox: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(bbox[:, 0], dtype=np.float64)
    y = np.asarray(bbox[:, 1], dtype=np.float64)
    w = np.asarray(bbox[:, 2], dtype=np.float64)
    h = np.asarray(bbox[:, 3], dtype=np.float64)
    cx = x + w / 2.0
    cy = y + h / 2.0
    return cx, cy, w, h


def _center_xywh_to_bbox(cx: np.ndarray, cy: np.ndarray, w: np.ndarray, h: np.ndarray) -> np.ndarray:
    x = np.asarray(cx, dtype=np.float64) - np.asarray(w, dtype=np.float64) / 2.0
    y = np.asarray(cy, dtype=np.float64) - np.asarray(h, dtype=np.float64) / 2.0
    w = np.maximum(np.asarray(w, dtype=np.float64), 1.0)
    h = np.maximum(np.asarray(h, dtype=np.float64), 1.0)
    return np.column_stack([x, y, w, h]).astype(np.float64)


def _fill_non_finite(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    finite = np.isfinite(out)
    if finite.all():
        return out
    if not finite.any():
        return np.zeros_like(out, dtype=np.float64)
    first = int(np.where(finite)[0][0])
    out[:first] = out[first]
    for idx in range(first + 1, len(out)):
        if not np.isfinite(out[idx]):
            out[idx] = out[idx - 1]
    last = int(np.where(np.isfinite(out))[0][-1])
    out[last + 1 :] = out[last]
    out[~np.isfinite(out)] = 0.0
    return out


def _make_tab_context_bbox(raw_bbox: np.ndarray) -> np.ndarray:
    bbox = np.asarray(raw_bbox, dtype=np.float64)
    observed = np.isfinite(bbox).all(axis=1)
    if observed.any():
        cx, cy, w, h = _bbox_to_center_xywh(bbox)
        cx_ctx = _fill_non_finite(cx)
        cy_ctx = _fill_non_finite(cy)
        w_ctx = np.maximum(_fill_non_finite(w), 1.0)
        h_ctx = np.maximum(_fill_non_finite(h), 1.0)
        return _center_xywh_to_bbox(cx_ctx, cy_ctx, w_ctx, h_ctx)
    med = np.nanmedian(bbox, axis=0)
    fill = np.where(np.isfinite(med), med, np.array([0.0, 0.0, 1.0, 1.0]))
    return np.tile(fill[None, :], (len(bbox), 1))


def _flag_from_anomaly_score(
    anomaly_score: np.ndarray,
    observed_mask: np.ndarray,
    quantile: float,
) -> np.ndarray:
    scores = np.asarray(anomaly_score, dtype=np.float64)
    observed = np.asarray(observed_mask, dtype=bool)
    obs_scores = scores[observed]
    if obs_scores.size > 0 and float(np.nanmax(obs_scores)) > 0.0:
        threshold = float(np.quantile(obs_scores, float(quantile)))
        flag_mask = observed & (scores >= threshold)
    else:
        flag_mask = np.zeros(len(scores), dtype=bool)
    flag_mask |= ~observed
    return flag_mask


def _safe_mean(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else float("nan")


def _safe_std(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(vals.std(ddof=0)) if vals.size else float("nan")


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax1 = a[:, 0]
    ay1 = a[:, 1]
    ax2 = a[:, 0] + a[:, 2]
    ay2 = a[:, 1] + a[:, 3]
    bx1 = b[:, 0]
    by1 = b[:, 1]
    bx2 = b[:, 0] + b[:, 2]
    by2 = b[:, 1] + b[:, 3]
    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    union = a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter
    out = np.zeros_like(inter, dtype=np.float64)
    valid = union > 0
    out[valid] = inter[valid] / union[valid]
    return out


def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = len(a)
    m = len(b)
    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = float(np.linalg.norm(a[i - 1] - b[j - 1]))
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m] / max(n, m, 1))


def _contiguous_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    segments: list[tuple[int, int]] = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < n and mask[j]:
            j += 1
        segments.append((i, j))
        i = j
    return segments


def _dense_frame_axis(payload: dict[str, Any]) -> np.ndarray:
    images = payload.get("images") or []
    if not images:
        return np.arange(1, dtype=np.int64)
    frames = [int(img.get("frame_index", img.get("id", 1) - 1)) for img in images]
    frame_min = int(min(frames))
    video_total = None
    videos = payload.get("videos") or []
    if videos and isinstance(videos[0], dict) and videos[0].get("total_frames") is not None:
        video_total = int(videos[0]["total_frames"])
    frame_max = int(max(frames))
    if video_total is not None and video_total > frame_max:
        frame_max = video_total - 1
    return np.arange(frame_min, frame_max + 1, dtype=np.int64)


def _prepare_gt_from_json(json_path: Path, smooth_window: int, smooth_polyorder: int) -> TrajectoryRecord | None:
    payload = _load_json(json_path)
    images = payload.get("images") or []
    annotations = payload.get("annotations") or []
    if not images or not annotations:
        return None

    image_to_frame = {
        int(img["id"]): int(img.get("frame_index", int(img["id"]) - 1))
        for img in images
        if "id" in img
    }
    ann_by_frame: dict[int, list[float]] = {}
    for ann in annotations:
        img_id = ann.get("image_id")
        bbox = ann.get("bbox")
        if img_id is None or bbox is None or len(bbox) < 4 or int(img_id) not in image_to_frame:
            continue
        frame = image_to_frame[int(img_id)]
        ann_by_frame.setdefault(frame, list(map(float, bbox[:4])))
    if not ann_by_frame:
        return None

    dense_frames = _dense_frame_axis(payload)
    n = len(dense_frames)
    bbox_observed = np.full((n, 4), np.nan, dtype=np.float64)
    observed_mask = np.zeros(n, dtype=bool)
    frame_to_dense = {int(f): i for i, f in enumerate(dense_frames)}
    for frame, bbox in ann_by_frame.items():
        idx = frame_to_dense.get(int(frame))
        if idx is None:
            continue
        bbox_observed[idx] = np.asarray(bbox, dtype=np.float64)
        observed_mask[idx] = True
    if observed_mask.sum() < 4:
        return None

    cx_obs, cy_obs, w_obs, h_obs = _bbox_to_center_xywh(bbox_observed)
    cx_fill = pchip_interpolate_1d(dense_frames[observed_mask].astype(np.float64), cx_obs[observed_mask], dense_frames.astype(np.float64))
    cy_fill = pchip_interpolate_1d(dense_frames[observed_mask].astype(np.float64), cy_obs[observed_mask], dense_frames.astype(np.float64))
    w_fill = pchip_interpolate_1d(dense_frames[observed_mask].astype(np.float64), w_obs[observed_mask], dense_frames.astype(np.float64))
    h_fill = pchip_interpolate_1d(dense_frames[observed_mask].astype(np.float64), h_obs[observed_mask], dense_frames.astype(np.float64))

    cx_gt = _adaptive_savgol(cx_fill, dense_frames, window_length=smooth_window, polyorder=smooth_polyorder)
    cy_gt = _adaptive_savgol(cy_fill, dense_frames, window_length=smooth_window, polyorder=smooth_polyorder)
    w_gt = _adaptive_savgol(np.maximum(w_fill, 1.0), dense_frames, window_length=smooth_window, polyorder=smooth_polyorder)
    h_gt = _adaptive_savgol(np.maximum(h_fill, 1.0), dense_frames, window_length=smooth_window, polyorder=smooth_polyorder)
    gt_bbox_dense = _center_xywh_to_bbox(cx_gt, cy_gt, np.maximum(w_gt, 1.0), np.maximum(h_gt, 1.0))

    video_name = str((payload.get("videos") or [{"name": json_path.stem}])[0].get("name", json_path.stem))
    rel = json_path.as_posix()
    return TrajectoryRecord(
        trajectory_id=rel,
        source_json=str(json_path),
        video_name=video_name,
        frame_index=dense_frames,
        gt_bbox_dense=gt_bbox_dense,
        gt_bbox_observed=bbox_observed,
        observed_mask=observed_mask,
    )


def _find_video_for_annotation(json_path: Path, video_extensions: tuple[str, ...]) -> Path | None:
    stem = json_path.with_suffix("")
    for ext in video_extensions:
        cand = Path(f"{stem}{ext}")
        if cand.exists():
            return cand
    return None


def _auto_discover_detector_checkpoint() -> tuple[Path, Path | None]:
    candidates = sorted(
        (
            p
            for p in (REPO_ROOT / "results").rglob("best.pt")
            if "detection" + os.sep + "YOLOv11" + os.sep + "weights" + os.sep + "best.pt" in str(p)
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError("Could not auto-discover any previous detector checkpoint under results/.")
    ckpt = candidates[0]
    exp_dir = ckpt.parents[4] if len(ckpt.parents) >= 5 else None
    return ckpt, exp_dir


def _resolve_detector_checkpoint_and_metadata(cfg: BenchmarkConfig) -> tuple[Path, dict[str, Any], Path | None]:
    exp_dir = Path(str(cfg.detector_experiment_dir)).expanduser().resolve() if str(cfg.detector_experiment_dir).strip() else None
    if exp_dir is not None:
        metadata_path = exp_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Detector experiment metadata not found: {metadata_path}")
        metadata = _load_json(metadata_path)
        ckpt_path = None
        if str(cfg.detector_checkpoint).strip():
            ckpt_path = Path(str(cfg.detector_checkpoint)).expanduser().resolve()
        else:
            for stage in metadata.get("stages", []):
                if stage.get("name") == "detector_train_full":
                    models = stage.get("models") or []
                    if models:
                        ckpt_raw = models[0].get("checkpoint")
                        if ckpt_raw:
                            ckpt_path = Path(str(ckpt_raw)).expanduser().resolve()
                            break
        if ckpt_path is None:
            raise RuntimeError(f"Could not resolve detector checkpoint from experiment metadata: {metadata_path}")
        return ckpt_path, metadata, exp_dir

    if str(cfg.detector_checkpoint).strip():
        ckpt_path = Path(str(cfg.detector_checkpoint)).expanduser().resolve()
        metadata_path = ckpt_path.parents[4] / "metadata.json" if len(ckpt_path.parents) >= 5 else None
        metadata = _load_json(metadata_path) if metadata_path is not None and metadata_path.exists() else {}
        exp_dir = metadata_path.parent if metadata_path is not None and metadata_path.exists() else None
        return ckpt_path, metadata, exp_dir

    ckpt_path, exp_dir = _auto_discover_detector_checkpoint()
    metadata = _load_json(exp_dir / "metadata.json") if exp_dir is not None and (exp_dir / "metadata.json").exists() else {}
    return ckpt_path.resolve(), metadata, exp_dir


def _build_detector_runner(cfg: BenchmarkConfig) -> dict[str, Any]:
    from tracking.core.registry import MODEL_REGISTRY, PREPROC_REGISTRY
    from tracking.preproc import augment as _load_preproc_augment  # noqa: F401
    from tracking.preproc import clahe as _load_preproc_clahe  # noqa: F401
    from tracking.models import yolov11 as _load_model_yolov11  # noqa: F401

    ckpt_path, metadata, exp_dir = _resolve_detector_checkpoint_and_metadata(cfg)
    pipeline = (
        (metadata.get("experiment") or {}).get("pipeline")
        or (metadata.get("config") or {}).get("pipeline")
        or []
    )
    model_step = next((x for x in pipeline if x.get("type") == "model"), None)
    if model_step is None:
        model_step = {"name": "YOLOv11", "params": {}}
    model_name = str(model_step.get("name") or "YOLOv11")
    model_cfg = dict(model_step.get("params") or {})
    model_cfg["train_enabled"] = False

    model_cls = MODEL_REGISTRY.get(model_name)
    if model_cls is None:
        raise RuntimeError(f"Detector model is not registered: {model_name}")
    model = model_cls(model_cfg)
    model.load_checkpoint(str(ckpt_path))

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

    return {
        "checkpoint": str(ckpt_path),
        "experiment_dir": str(exp_dir) if exp_dir is not None else "",
        "metadata": metadata,
        "model_name": model_name,
        "model": model,
    }


def _default_detector_cache_root(cfg: BenchmarkConfig, checkpoint_path: str) -> Path:
    if str(cfg.detector_cache_root).strip():
        return Path(str(cfg.detector_cache_root))
    stem = Path(str(checkpoint_path)).parent.parent.parent.parent.name
    return REPO_ROOT / "results" / "trajectory_repair_detector_cache" / stem


def _prediction_rows_to_dense_bbox(rows: list[dict[str, Any]], dense_frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame_to_bbox: dict[int, np.ndarray] = {}
    for row in rows:
        try:
            fi = int(row.get("frame_index"))
            bbox = row.get("bbox")
            if bbox is None or len(bbox) < 4:
                continue
            frame_to_bbox[fi] = np.asarray(bbox[:4], dtype=np.float64)
        except Exception:
            continue

    raw_bbox = np.full((len(dense_frames), 4), np.nan, dtype=np.float64)
    observed = np.zeros(len(dense_frames), dtype=bool)
    for i, frame in enumerate(np.asarray(dense_frames, dtype=np.int64)):
        bbox = frame_to_bbox.get(int(frame))
        if bbox is None:
            continue
        raw_bbox[i] = bbox
        observed[i] = np.isfinite(bbox).all()
    return raw_bbox, observed


def _serialize_prediction_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serializable: list[dict[str, Any]] = []
    for row in rows:
        serializable.append(
            {
                "frame_index": int(row.get("frame_index", 0)),
                "bbox": [float(x) for x in list(row.get("bbox") or [0.0, 0.0, 1.0, 1.0])[:4]],
                "score": None if row.get("score") is None else float(row.get("score")),
                "confidence": None if row.get("confidence") is None else float(row.get("confidence")),
                "is_fallback": bool(row.get("is_fallback", False)),
                "bbox_source": str(row.get("bbox_source", "detector")),
            }
        )
    return serializable


def _predict_detector_rows(video_path: Path, runner: dict[str, Any]) -> list[dict[str, Any]]:
    preds = runner["model"].predict(str(video_path))
    rows: list[dict[str, Any]] = []
    for pred in preds:
        rows.append(
            {
                "frame_index": int(getattr(pred, "frame_index", 0)),
                "bbox": [float(x) for x in tuple(getattr(pred, "bbox", (0.0, 0.0, 1.0, 1.0)))[:4]],
                "score": None if getattr(pred, "score", None) is None else float(pred.score),
                "confidence": None if getattr(pred, "confidence", None) is None else float(pred.confidence),
                "is_fallback": bool(getattr(pred, "is_fallback", False)),
                "bbox_source": str(getattr(pred, "bbox_source", "detector")),
            }
        )
    return rows


def _load_or_run_detector_rows(
    video_path: Path,
    dataset_root: Path,
    cache_root: Path,
    runner: dict[str, Any],
) -> list[dict[str, Any]]:
    rel = video_path.relative_to(dataset_root).with_suffix(".json")
    cache_path = cache_root / rel
    if cache_path.exists():
        payload = _load_json(cache_path)
        if isinstance(payload, list):
            return payload

    rows = _serialize_prediction_rows(_predict_detector_rows(video_path, runner))
    _write_json(cache_path, rows)
    return rows


def _compute_real_corrupt_mask(
    raw_bbox: np.ndarray,
    gt_bbox_dense: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    raw = np.asarray(raw_bbox, dtype=np.float64)
    gt = np.asarray(gt_bbox_dense, dtype=np.float64)
    observed = np.isfinite(raw).all(axis=1)
    corrupt = np.ones(len(raw), dtype=bool)
    if observed.any():
        iou = _bbox_iou(raw[observed], gt[observed])
        corrupt[observed] = iou < float(iou_threshold)
    return corrupt


def _prepare_detector_record(
    json_path: Path,
    video_path: Path,
    raw_rows: list[dict[str, Any]],
    *,
    smooth_window: int,
    smooth_polyorder: int,
    iou_threshold: float,
) -> TrajectoryRecord | None:
    gt_record = _prepare_gt_from_json(json_path, smooth_window=smooth_window, smooth_polyorder=smooth_polyorder)
    if gt_record is None:
        return None

    raw_bbox_dense, raw_observed = _prediction_rows_to_dense_bbox(raw_rows, gt_record.frame_index)
    raw_corrupt = _compute_real_corrupt_mask(raw_bbox_dense, gt_record.gt_bbox_dense, iou_threshold=iou_threshold)
    return TrajectoryRecord(
        trajectory_id=gt_record.trajectory_id,
        source_json=gt_record.source_json,
        video_name=gt_record.video_name,
        frame_index=gt_record.frame_index,
        gt_bbox_dense=gt_record.gt_bbox_dense,
        gt_bbox_observed=gt_record.gt_bbox_observed,
        observed_mask=gt_record.observed_mask,
        video_path=str(video_path),
        raw_bbox_dense=raw_bbox_dense,
        raw_observed_mask=raw_observed,
        raw_corrupt_mask=raw_corrupt,
    )


def load_detector_trajectories(
    dataset_root: Path,
    cfg: BenchmarkConfig,
    *,
    cache_tag: str,
) -> tuple[list[TrajectoryRecord], dict[str, Any]]:
    video_extensions = _iter_video_extensions(cfg.detector_video_extensions)
    runner = _build_detector_runner(cfg)
    cache_root = _default_detector_cache_root(cfg, runner["checkpoint"]) / cache_tag
    cache_root.mkdir(parents=True, exist_ok=True)

    json_paths = sorted(dataset_root.rglob("*.json"))
    records: list[TrajectoryRecord] = []
    used = 0
    for json_path in json_paths:
        video_path = _find_video_for_annotation(json_path, video_extensions)
        if video_path is None:
            continue
        rows = _load_or_run_detector_rows(video_path, dataset_root, cache_root, runner)
        rec = _prepare_detector_record(
            json_path,
            video_path,
            rows,
            smooth_window=cfg.mild_sg_window,
            smooth_polyorder=cfg.mild_sg_polyorder,
            iou_threshold=cfg.corrupt_iou_threshold,
        )
        if rec is None:
            continue
        records.append(rec)
        used += 1
        if cfg.max_trajectories > 0 and used >= cfg.max_trajectories:
            break

    return records, {
        "checkpoint": runner["checkpoint"],
        "experiment_dir": runner["experiment_dir"],
        "cache_root": str(cache_root),
        "model_name": runner["model_name"],
    }


def _slice_record(rec: TrajectoryRecord, start: int, stop: int) -> TrajectoryRecord:
    raw_bbox = None if rec.raw_bbox_dense is None else np.asarray(rec.raw_bbox_dense[start:stop], dtype=np.float64)
    raw_observed = None if rec.raw_observed_mask is None else np.asarray(rec.raw_observed_mask[start:stop], dtype=bool)
    raw_corrupt = None if rec.raw_corrupt_mask is None else np.asarray(rec.raw_corrupt_mask[start:stop], dtype=bool)
    suffix = f"__win_{int(start):03d}_{int(stop):03d}"
    return TrajectoryRecord(
        trajectory_id=f"{rec.trajectory_id}{suffix}",
        source_json=rec.source_json,
        video_name=rec.video_name,
        frame_index=np.asarray(rec.frame_index[start:stop], dtype=np.int64),
        gt_bbox_dense=np.asarray(rec.gt_bbox_dense[start:stop], dtype=np.float64),
        gt_bbox_observed=np.asarray(rec.gt_bbox_observed[start:stop], dtype=np.float64),
        observed_mask=np.asarray(rec.observed_mask[start:stop], dtype=bool),
        video_path=rec.video_path,
        raw_bbox_dense=raw_bbox,
        raw_observed_mask=raw_observed,
        raw_corrupt_mask=raw_corrupt,
        case_window=(int(start), int(stop)),
    )


def _select_real_detector_cases(
    records: list[TrajectoryRecord],
    *,
    target_ratio: float,
    target_cases: int,
    case_window: int,
    seed: int,
) -> list[TrajectoryRecord]:
    if not records or target_cases <= 0:
        return []
    case_counts = allocate_test_cases(len(records), target_cases, cases_per_trajectory=1)
    rng = np.random.default_rng(seed)
    picked: list[TrajectoryRecord] = []
    for rec, want in zip(records, case_counts):
        raw_corrupt = np.asarray(rec.raw_corrupt_mask, dtype=bool) if rec.raw_corrupt_mask is not None else np.zeros(len(rec.frame_index), dtype=bool)
        n = len(rec.frame_index)
        if n <= 0:
            continue
        win = int(max(8, min(case_window, n)))
        starts = np.arange(max(1, n - win + 1), dtype=np.int64)
        rng.shuffle(starts)
        scored = []
        for start in starts.tolist():
            stop = min(n, start + win)
            ratio = float(raw_corrupt[start:stop].mean()) if stop > start else 0.0
            scored.append((abs(ratio - float(target_ratio)), float(ratio), int(start), int(stop)))
        scored.sort(key=lambda item: (item[0], abs(item[1] - float(target_ratio)), item[2]))
        if not scored:
            scored = [(0.0, float(raw_corrupt.mean()) if raw_corrupt.size else 0.0, 0, n)]
        chosen = scored[: max(1, int(want))]
        while len(chosen) < int(want):
            chosen.append(scored[len(chosen) % len(scored)])
        for _, _, start, stop in chosen:
            picked.append(_slice_record(rec, start, stop))
    return picked[:target_cases]


def load_trajectories(dataset_root: Path, smooth_window: int, smooth_polyorder: int, max_trajectories: int = 0) -> list[TrajectoryRecord]:
    json_paths = sorted(dataset_root.rglob("*.json"))
    out: list[TrajectoryRecord] = []
    for path in json_paths:
        rec = _prepare_gt_from_json(path, smooth_window=smooth_window, smooth_polyorder=smooth_polyorder)
        if rec is None:
            continue
        out.append(rec)
        if max_trajectories > 0 and len(out) >= max_trajectories:
            break
    return out


def split_trajectories(records: list[TrajectoryRecord], train_ratio: float, seed: int) -> tuple[list[TrajectoryRecord], list[TrajectoryRecord]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(records))
    rng.shuffle(idx)
    n_train = max(1, int(round(len(idx) * train_ratio)))
    n_train = min(n_train, len(idx) - 1) if len(idx) > 1 else len(idx)
    train_ids = set(idx[:n_train].tolist())
    train = [records[i] for i in range(len(records)) if i in train_ids]
    test = [records[i] for i in range(len(records)) if i not in train_ids]
    return train, test


def allocate_test_cases(n_test: int, target_test_cases: int, cases_per_trajectory: int) -> list[int]:
    if n_test <= 0:
        return []
    if int(target_test_cases) > 0:
        target = int(target_test_cases)
        base = target // n_test
        rem = target % n_test
        return [max(1, base + (1 if i < rem else 0)) for i in range(n_test)]
    fallback = max(1, int(cases_per_trajectory))
    return [fallback] * n_test


def _sample_non_overlapping_segment(
    n: int,
    used: np.ndarray,
    desired_len: int,
    rng: np.random.Generator,
    start_lo: int = 0,
    start_hi: int | None = None,
) -> np.ndarray:
    desired_len = int(max(1, min(desired_len, n)))
    if start_hi is None:
        start_hi = n - desired_len
    start_hi = max(start_lo, start_hi)
    choices = np.arange(start_lo, start_hi + 1)
    rng.shuffle(choices)
    for start in choices:
        sl = slice(int(start), int(start + desired_len))
        if not used[sl].any():
            mask = np.zeros(n, dtype=bool)
            mask[sl] = True
            return mask
    return np.zeros(n, dtype=bool)


def _apply_drift(bbox: np.ndarray, mask: np.ndarray, offset_xy: np.ndarray) -> None:
    bbox[mask, 0] += float(offset_xy[0])
    bbox[mask, 1] += float(offset_xy[1])


def _sample_available_points(available_mask: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.where(np.asarray(available_mask, dtype=bool))[0]
    if idx.size == 0 or count <= 0:
        return np.zeros(len(available_mask), dtype=bool)
    chosen = rng.choice(idx, size=min(int(count), idx.size), replace=False)
    mask = np.zeros(len(available_mask), dtype=bool)
    mask[chosen] = True
    return mask


def _apply_detection_noise(
    bbox: np.ndarray,
    *,
    rng: np.random.Generator,
    xy_sigma: float,
) -> np.ndarray:
    noisy = np.asarray(bbox, dtype=np.float64).copy()
    cx, cy, w, h = _bbox_to_center_xywh(noisy)
    cx += rng.normal(0.0, xy_sigma, size=len(cx))
    cy += rng.normal(0.0, xy_sigma, size=len(cy))
    return _center_xywh_to_bbox(cx, cy, np.maximum(w, 1.0), np.maximum(h, 1.0))


def _corrupt_bbox(
    gt_bbox: np.ndarray,
    frame_index: np.ndarray,
    *,
    rng: np.random.Generator,
    max_ratio: float,
    cfg: BenchmarkConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or BenchmarkConfig()
    n = len(gt_bbox)
    target_points = max(1, min(n, int(round(n * max_ratio))))
    used = np.zeros(n, dtype=bool)
    raw_bbox = gt_bbox.copy()
    pattern_masks: dict[str, np.ndarray] = {}

    _, _, w, h = _bbox_to_center_xywh(gt_bbox)
    scale = float(np.nanmedian(np.sqrt(np.maximum(w * h, 1.0))))
    raw_bbox = _apply_detection_noise(
        raw_bbox,
        rng=rng,
        xy_sigma=max(0.5, cfg.raw_noise_xy_ratio * scale),
    )
    drift_mag = max(8.0, 0.40 * scale)
    spike_mag = max(12.0, 0.60 * scale)
    missing_budget = int(round(target_points * rng.uniform(0.08, 0.18)))

    missing_mask = np.zeros(n, dtype=bool)
    attempts = 0
    while int(used.sum()) < target_points and attempts < (6 * n + 20):
        attempts += 1
        remaining = target_points - int(used.sum())
        available = ~used
        if not available.any():
            break

        choices = ["spike", "short_drift", "two_way_drift", "initial_drift"]
        if remaining >= 1 and missing_budget > 0:
            choices.append("random_missing")
        pattern = str(rng.choice(choices))

        if pattern == "initial_drift":
            seg_len = int(min(remaining, max(2, rng.integers(2, max(4, min(max(5, n // 6), remaining) + 1)))))
            mask = _sample_non_overlapping_segment(n, used, seg_len, rng, start_lo=0, start_hi=max(0, min(n - seg_len, max(0, n // 7))))
            if not mask.any():
                continue
            ang = rng.uniform(-0.35 * np.pi, 0.35 * np.pi)
            mag = drift_mag * rng.uniform(0.9, 1.4)
            _apply_drift(raw_bbox, mask, np.array([np.cos(ang), np.sin(ang)]) * mag)
        elif pattern == "short_drift":
            seg_len = int(min(remaining, max(3, rng.integers(3, max(5, min(max(6, n // 5), remaining) + 1)))))
            mask = _sample_non_overlapping_segment(n, used, seg_len, rng, start_lo=max(1, n // 10), start_hi=max(1, n - seg_len))
            if not mask.any():
                continue
            ang = rng.uniform(0.0, 2.0 * np.pi)
            mag = drift_mag * rng.uniform(0.8, 1.5)
            _apply_drift(raw_bbox, mask, np.array([np.cos(ang), np.sin(ang)]) * mag)
        elif pattern == "two_way_drift":
            if remaining < 4:
                continue
            total_len = int(min(remaining, max(4, rng.integers(4, max(6, min(max(8, n // 4), remaining) + 1)))))
            len1 = max(2, total_len // 2)
            len2 = max(2, total_len - len1)
            mask1 = _sample_non_overlapping_segment(n, used, len1, rng, start_lo=max(1, n // 10), start_hi=max(1, n - len1))
            if not mask1.any():
                continue
            mask2 = _sample_non_overlapping_segment(n, used | mask1, len2, rng, start_lo=max(1, n // 10), start_hi=max(1, n - len2))
            if not mask2.any():
                continue
            ang = rng.uniform(0.0, 2.0 * np.pi)
            vec = np.array([np.cos(ang), np.sin(ang)]) * drift_mag * rng.uniform(0.9, 1.5)
            _apply_drift(raw_bbox, mask1, vec)
            _apply_drift(raw_bbox, mask2, -vec)
            mask = mask1 | mask2
        elif pattern == "random_missing":
            miss_count = int(min(remaining, max(1, min(missing_budget, rng.integers(1, max(2, min(remaining, 8) + 1))))))
            mask = _sample_available_points(~used, miss_count, rng)
            if not mask.any():
                continue
            missing_mask |= mask
            raw_bbox[mask] = np.nan
            missing_budget = max(0, missing_budget - int(mask.sum()))
        else:
            count = int(min(remaining, max(1, rng.integers(1, max(2, min(remaining, 10) + 1)))))
            mask = _sample_available_points(~used, count, rng)
            if not mask.any():
                continue
            angles = rng.uniform(0.0, 2.0 * np.pi, size=mask.sum())
            mags = spike_mag * rng.uniform(0.8, 1.6, size=mask.sum())
            offsets = np.column_stack([np.cos(angles), np.sin(angles)]) * mags[:, None]
            raw_bbox[mask, 0] += offsets[:, 0]
            raw_bbox[mask, 1] += offsets[:, 1]

        if mask.any():
            pattern_masks[pattern] = pattern_masks.get(pattern, np.zeros(n, dtype=bool)) | mask
            used |= mask

    remaining = target_points - int(used.sum())
    if remaining > 0:
        mask = _sample_available_points(~used, remaining, rng)
        if mask.any():
            angles = rng.uniform(0.0, 2.0 * np.pi, size=mask.sum())
            mags = spike_mag * rng.uniform(0.9, 1.5, size=mask.sum())
            offsets = np.column_stack([np.cos(angles), np.sin(angles)]) * mags[:, None]
            raw_bbox[mask, 0] += offsets[:, 0]
            raw_bbox[mask, 1] += offsets[:, 1]
            pattern_masks["spike"] = pattern_masks.get("spike", np.zeros(n, dtype=bool)) | mask
            used |= mask

    corrupt_mask = used.copy()
    return {
        "raw_bbox": raw_bbox,
        "corrupt_mask": corrupt_mask,
        "missing_mask": missing_mask,
        "pattern_masks": pattern_masks,
        "pattern_names": sorted(pattern_masks.keys()) + (["random_missing"] if missing_mask.any() else []),
        "frame_index": frame_index.copy(),
    }


def _baseline_method(raw_bbox: np.ndarray, frame_index: np.ndarray, cfg: BenchmarkConfig) -> dict[str, Any]:
    cx, cy, w, h = _bbox_to_center_xywh(raw_bbox)
    observed = np.isfinite(raw_bbox).all(axis=1)
    n = max(1, int(len(frame_index)))
    macro_ratio = (2.0 * float(PIPE_HAMPEL_MACRO_HW) + 1e-9) / float(n)
    cx_fill, _, cx_mask = hampel_then_pchip_1d(
        cx,
        frame_index,
        observed_mask=observed,
        macro_ratio=macro_ratio,
        macro_sigma=2.0,
        micro_hw=PIPE_HAMPEL_MICRO_HW,
        micro_sigma=2.0,
        max_outlier_run=2,
    )
    cy_fill, _, cy_mask = hampel_then_pchip_1d(
        cy,
        frame_index,
        observed_mask=observed,
        macro_ratio=macro_ratio,
        macro_sigma=2.0,
        micro_hw=PIPE_HAMPEL_MICRO_HW,
        micro_sigma=2.0,
        max_outlier_run=2,
    )
    cx_smooth = _adaptive_savgol(cx_fill, frame_index, window_length=cfg.baseline_sg_window, polyorder=cfg.baseline_sg_polyorder)
    cy_smooth = _adaptive_savgol(cy_fill, frame_index, window_length=cfg.baseline_sg_window, polyorder=cfg.baseline_sg_polyorder)
    w_out, h_out = filter_bbox_hampel_only(
        w,
        h,
        macro_ratio=macro_ratio,
        macro_sigma=2.0,
        micro_hw=PIPE_HAMPEL_MICRO_HW,
        micro_sigma=2.0,
        frame_indices=frame_index,
        observed_mask=observed,
    )
    bbox = _center_xywh_to_bbox(cx_smooth, cy_smooth, w_out, h_out)
    anomaly_score = np.maximum(cx_mask.astype(np.float64), cy_mask.astype(np.float64))
    anomaly_score[~observed] = 1.0
    return {
        "variant": "direct",
        "bbox": bbox,
        "center": np.column_stack([cx_smooth, cy_smooth]),
        "anomaly_score": anomaly_score,
        "flag_mask": anomaly_score > 0.0,
    }


def _make_bspline_design(t: np.ndarray, knots: int, degree: int) -> np.ndarray:
    if BSpline is None or len(t) < (degree + 2):
        x = (t - t.min()) / max(float(t.max() - t.min()), 1.0)
        return np.column_stack([np.ones_like(x), x, x**2, x**3])
    t = np.asarray(t, dtype=np.float64)
    t_norm = (t - t.min()) / max(float(t.max() - t.min()), 1.0)
    interior_count = int(max(0, min(knots, len(t_norm) - degree - 1)))
    if interior_count <= 0:
        return np.column_stack([np.ones_like(t_norm), t_norm, t_norm**2, t_norm**3])
    interior = np.quantile(t_norm, np.linspace(0.1, 0.9, interior_count))
    interior = np.unique(np.clip(interior, 1e-6, 1.0 - 1e-6))
    knot_vector = np.concatenate([np.zeros(degree + 1, dtype=np.float64), interior, np.ones(degree + 1, dtype=np.float64)])
    n_basis = len(knot_vector) - degree - 1
    design = np.zeros((len(t_norm), n_basis), dtype=np.float64)
    for i in range(n_basis):
        coeff = np.zeros(n_basis, dtype=np.float64)
        coeff[i] = 1.0
        basis = BSpline(knot_vector, coeff, degree, extrapolate=True)
        design[:, i] = basis(t_norm)
    return design


def _mad_threshold(values: np.ndarray, floor: float = 2.5, scale: float = 3.0) -> float:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float(floor)
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) + 1e-6
    return float(max(floor, scale * 1.4826 * mad))


def _robust_refine_linear_model(
    design_obs: np.ndarray,
    yy: np.ndarray,
    inlier_obs: np.ndarray,
    *,
    min_samples: int,
    robust_iterations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inlier = np.asarray(inlier_obs, dtype=bool).copy()
    if inlier.sum() < min_samples:
        inlier[:] = True
    coef, *_ = np.linalg.lstsq(design_obs[inlier], yy[inlier], rcond=None)
    pred_obs = np.asarray(design_obs @ coef, dtype=np.float64)
    residual_obs = np.abs(yy - pred_obs)

    for _ in range(max(0, int(robust_iterations))):
        threshold = _mad_threshold(residual_obs[inlier] if inlier.any() else residual_obs)
        new_inlier = residual_obs <= threshold
        if new_inlier.sum() < min_samples:
            break
        coef, *_ = np.linalg.lstsq(design_obs[new_inlier], yy[new_inlier], rcond=None)
        pred_obs = np.asarray(design_obs @ coef, dtype=np.float64)
        residual_obs = np.abs(yy - pred_obs)
        if np.array_equal(new_inlier, inlier):
            inlier = new_inlier
            break
        inlier = new_inlier
    return np.asarray(coef, dtype=np.float64), inlier, residual_obs


def _fit_ransac_spline_1d(
    t: np.ndarray,
    y: np.ndarray,
    knots: int,
    degree: int,
    residual_threshold: float | None = None,
    robust_iterations: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(y)
    x = np.asarray(t[finite], dtype=np.float64)
    yy = np.asarray(y[finite], dtype=np.float64)
    if finite.sum() < 4:
        fill = pchip_interpolate_1d(x, yy, t) if finite.sum() >= 2 else np.full_like(t, float(np.nanmean(y) if finite.any() else 0.0), dtype=np.float64)
        return fill, np.zeros(len(t), dtype=bool), np.zeros(len(t), dtype=np.float64)
    design = _make_bspline_design(x, knots=knots, degree=degree)
    threshold = residual_threshold
    if threshold is None:
        threshold = _mad_threshold(yy)
    min_samples = min(max(4, design.shape[1] + 1), len(x))
    ransac = RANSACRegressor(
        estimator=LinearRegression(),
        min_samples=min_samples,
        residual_threshold=float(threshold),
        max_trials=max(1, int(robust_iterations)),
        random_state=0,
    )
    ransac.fit(design, yy)
    coef, inlier_obs, residual_obs = _robust_refine_linear_model(
        design,
        yy,
        np.asarray(ransac.inlier_mask_, dtype=bool),
        min_samples=min_samples,
        robust_iterations=robust_iterations,
    )
    design_all = _make_bspline_design(np.asarray(t, dtype=np.float64), knots=knots, degree=degree)
    pred_all = np.asarray(design_all @ coef, dtype=np.float64)
    inlier_mask = np.zeros(len(t), dtype=bool)
    inlier_mask[np.where(finite)[0]] = inlier_obs
    residual_all = np.zeros(len(t), dtype=np.float64)
    residual_all[np.where(finite)[0]] = residual_obs
    return pred_all, inlier_mask, residual_all


def _ransac_bspline_method(raw_bbox: np.ndarray, frame_index: np.ndarray, cfg: BenchmarkConfig) -> dict[str, Any]:
    cx, cy, w, h = _bbox_to_center_xywh(raw_bbox)
    cx_fit, cx_inlier, cx_res = _fit_ransac_spline_1d(frame_index, cx, cfg.bspline_knots, cfg.bspline_degree, robust_iterations=cfg.robust_iterations)
    cy_fit, cy_inlier, cy_res = _fit_ransac_spline_1d(frame_index, cy, cfg.bspline_knots, cfg.bspline_degree, robust_iterations=cfg.robust_iterations)
    w_fit, w_inlier, w_res = _fit_ransac_spline_1d(frame_index, w, cfg.bspline_knots, cfg.bspline_degree, robust_iterations=cfg.robust_iterations)
    h_fit, h_inlier, h_res = _fit_ransac_spline_1d(frame_index, h, cfg.bspline_knots, cfg.bspline_degree, robust_iterations=cfg.robust_iterations)
    cx_fit = _adaptive_savgol(cx_fit, frame_index, window_length=cfg.mild_sg_window, polyorder=cfg.mild_sg_polyorder)
    cy_fit = _adaptive_savgol(cy_fit, frame_index, window_length=cfg.mild_sg_window, polyorder=cfg.mild_sg_polyorder)
    w_fit = _adaptive_savgol(np.maximum(w_fit, 1.0), frame_index, window_length=cfg.mild_sg_window, polyorder=cfg.mild_sg_polyorder)
    h_fit = _adaptive_savgol(np.maximum(h_fit, 1.0), frame_index, window_length=cfg.mild_sg_window, polyorder=cfg.mild_sg_polyorder)
    bbox = _center_xywh_to_bbox(cx_fit, cy_fit, np.maximum(w_fit, 1.0), np.maximum(h_fit, 1.0))
    anomaly_score = np.maximum.reduce([cx_res, cy_res, w_res, h_res])
    flag_mask = (~cx_inlier) | (~cy_inlier) | (~w_inlier) | (~h_inlier)
    observed_mask = np.isfinite(raw_bbox).all(axis=1)
    flag_mask |= ~observed_mask
    if np.nanmax(anomaly_score) > 0:
        anomaly_score = anomaly_score / float(np.nanmax(anomaly_score))
    anomaly_score[~observed_mask] = 1.0
    return {
        "variant": "direct",
        "bbox": bbox,
        "center": np.column_stack([cx_fit, cy_fit]),
        "anomaly_score": anomaly_score,
        "flag_mask": flag_mask,
    }


def _fit_detector_curve_msac(
    t: np.ndarray,
    y: np.ndarray,
    knots: int,
    degree: int,
    rng: np.random.Generator,
    n_iter: int = 500,
    robust_iterations: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(y)
    x = np.asarray(t[finite], dtype=np.float64)
    yy = np.asarray(y[finite], dtype=np.float64)
    if finite.sum() < 4:
        pred = pchip_interpolate_1d(x, yy, t) if finite.sum() >= 2 else np.full_like(t, float(np.nanmean(y) if finite.any() else 0.0))
        return np.asarray(pred, dtype=np.float64), np.zeros(len(t), dtype=np.float64)
    design_obs = _make_bspline_design(x, knots=knots, degree=degree)
    design_all = _make_bspline_design(np.asarray(t, dtype=np.float64), knots=knots, degree=degree)
    p = design_obs.shape[1]
    sample_size = min(max(p + 1, 4), len(x))
    threshold = _mad_threshold(yy)

    best_cost = float("inf")
    best_coef = None
    for _ in range(n_iter):
        ids = rng.choice(len(x), size=sample_size, replace=False)
        try:
            coef, *_ = np.linalg.lstsq(design_obs[ids], yy[ids], rcond=None)
        except np.linalg.LinAlgError:
            continue
        pred = design_obs @ coef
        residual = np.abs(yy - pred)
        cost = float(np.minimum(residual**2, threshold**2).sum())
        if cost < best_cost:
            best_cost = cost
            best_coef = coef
    if best_coef is None:
        coef, *_ = np.linalg.lstsq(design_obs, yy, rcond=None)
        best_coef = coef
    pred_obs = np.asarray(design_obs @ best_coef, dtype=np.float64)
    residual_obs = np.abs(yy - pred_obs)
    inlier_obs = residual_obs <= threshold
    coef, _, residual_obs = _robust_refine_linear_model(
        design_obs,
        yy,
        inlier_obs,
        min_samples=sample_size,
        robust_iterations=robust_iterations,
    )
    pred_all = np.asarray(design_all @ coef, dtype=np.float64)
    residual_all = np.zeros(len(t), dtype=np.float64)
    residual_all[np.where(finite)[0]] = residual_obs
    return pred_all, residual_all


def _fit_detector_curve_ransac(
    t: np.ndarray,
    y: np.ndarray,
    knots: int,
    degree: int,
    robust_iterations: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    pred, _, residual = _fit_ransac_spline_1d(t, y, knots=knots, degree=degree, robust_iterations=robust_iterations)
    return pred, residual


def _fit_detector_curve_irls(
    t: np.ndarray,
    y: np.ndarray,
    knots: int,
    degree: int,
    robust_iterations: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(y)
    x = np.asarray(t[finite], dtype=np.float64)
    yy = np.asarray(y[finite], dtype=np.float64)
    if finite.sum() < 4:
        pred = pchip_interpolate_1d(x, yy, t) if finite.sum() >= 2 else np.full_like(t, float(np.nanmean(y) if finite.any() else 0.0))
        return np.asarray(pred, dtype=np.float64), np.zeros(len(t), dtype=np.float64)

    design_obs = _make_bspline_design(x, knots=knots, degree=degree)
    design_all = _make_bspline_design(np.asarray(t, dtype=np.float64), knots=knots, degree=degree)
    coef, *_ = np.linalg.lstsq(design_obs, yy, rcond=None)
    coef = np.asarray(coef, dtype=np.float64)

    for _ in range(max(1, int(robust_iterations))):
        pred_obs = np.asarray(design_obs @ coef, dtype=np.float64)
        residual_obs = yy - pred_obs
        scale = 1.4826 * np.median(np.abs(residual_obs - np.median(residual_obs))) + 1e-6
        cutoff = max(1e-6, 1.345 * scale)
        abs_res = np.abs(residual_obs)
        weights = np.ones_like(abs_res, dtype=np.float64)
        large = abs_res > cutoff
        weights[large] = cutoff / abs_res[large]
        sqrt_w = np.sqrt(weights)
        coef_new, *_ = np.linalg.lstsq(design_obs * sqrt_w[:, None], yy * sqrt_w, rcond=None)
        coef_new = np.asarray(coef_new, dtype=np.float64)
        if np.allclose(coef_new, coef, atol=1e-6, rtol=1e-6):
            coef = coef_new
            break
        coef = coef_new

    pred_obs = np.asarray(design_obs @ coef, dtype=np.float64)
    residual_obs = np.abs(yy - pred_obs)
    pred_all = np.asarray(design_all @ coef, dtype=np.float64)
    residual_all = np.zeros(len(t), dtype=np.float64)
    residual_all[np.where(finite)[0]] = residual_obs
    return pred_all, residual_all


def _lowess_fit_1d(x: np.ndarray, y: np.ndarray, frac: float, robust_iterations: int) -> np.ndarray:
    n = len(x)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    r = max(4, min(n, int(np.ceil(max(0.05, min(0.95, frac)) * n))))
    robust = np.ones(n, dtype=np.float64)
    pred = np.asarray(y, dtype=np.float64).copy()
    for _ in range(max(1, int(robust_iterations))):
        for i in range(n):
            dist = np.abs(x - x[i])
            h = float(np.partition(dist, min(r - 1, n - 1))[min(r - 1, n - 1)])
            h = h if h > 1e-12 else max(float(np.max(dist)), 1.0)
            u = np.clip(dist / h, 0.0, 1.0)
            weights = (1.0 - u**3) ** 3
            weights *= robust
            if np.count_nonzero(weights > 1e-12) < 2:
                pred[i] = float(np.average(y))
                continue
            x0 = x - x[i]
            design = np.column_stack([np.ones(n, dtype=np.float64), x0])
            sw = np.sqrt(weights)
            coef, *_ = np.linalg.lstsq(design * sw[:, None], y * sw, rcond=None)
            pred[i] = float(coef[0])
        resid = y - pred
        scale = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-6
        u = resid / (6.0 * scale)
        robust = np.where(np.abs(u) < 1.0, (1.0 - u**2) ** 2, 0.0)
    return np.asarray(pred, dtype=np.float64)


def _fit_detector_curve_lowess(
    t: np.ndarray,
    y: np.ndarray,
    frac: float,
    robust_iterations: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(y)
    x = np.asarray(t[finite], dtype=np.float64)
    yy = np.asarray(y[finite], dtype=np.float64)
    if finite.sum() < 4:
        pred = np.full(len(t), np.nan, dtype=np.float64)
        if finite.sum() >= 1:
            pred[np.where(finite)[0]] = yy
            pred = _fill_non_finite(pred)
        else:
            pred.fill(float(np.nanmean(y) if finite.any() else 0.0))
        return np.asarray(pred, dtype=np.float64), np.zeros(len(t), dtype=np.float64)
    pred_fwd = _lowess_fit_1d(x, yy, frac=frac, robust_iterations=robust_iterations)
    x_rev = (x.max() - x[::-1]).astype(np.float64)
    pred_rev = _lowess_fit_1d(x_rev, yy[::-1], frac=frac, robust_iterations=robust_iterations)[::-1]
    pred_obs = 0.5 * (pred_fwd + pred_rev)
    residual_obs = np.abs(yy - pred_obs)
    pred_all = np.full(len(t), np.nan, dtype=np.float64)
    pred_all[np.where(finite)[0]] = pred_obs
    pred_all = _fill_non_finite(pred_all)
    residual_all = np.zeros(len(t), dtype=np.float64)
    residual_all[np.where(finite)[0]] = residual_obs
    return np.asarray(pred_all, dtype=np.float64), residual_all


def _fit_detector_curve_lmeds(
    t: np.ndarray,
    y: np.ndarray,
    knots: int,
    degree: int,
    rng: np.random.Generator,
    n_iter: int = 500,
    robust_iterations: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(y)
    x = np.asarray(t[finite], dtype=np.float64)
    yy = np.asarray(y[finite], dtype=np.float64)
    if finite.sum() < 4:
        pred = pchip_interpolate_1d(x, yy, t) if finite.sum() >= 2 else np.full_like(t, float(np.nanmean(y) if finite.any() else 0.0))
        return np.asarray(pred, dtype=np.float64), np.zeros(len(t), dtype=np.float64)
    design_obs = _make_bspline_design(x, knots=knots, degree=degree)
    design_all = _make_bspline_design(np.asarray(t, dtype=np.float64), knots=knots, degree=degree)
    p = design_obs.shape[1]
    sample_size = min(max(p + 1, 4), len(x))
    best_cost = float("inf")
    best_coef = None
    for _ in range(max(1, int(n_iter))):
        ids = rng.choice(len(x), size=sample_size, replace=False)
        try:
            coef, *_ = np.linalg.lstsq(design_obs[ids], yy[ids], rcond=None)
        except np.linalg.LinAlgError:
            continue
        residual = yy - (design_obs @ coef)
        cost = float(np.median(residual**2))
        if cost < best_cost:
            best_cost = cost
            best_coef = coef
    if best_coef is None:
        best_coef, *_ = np.linalg.lstsq(design_obs, yy, rcond=None)
    pred_obs = np.asarray(design_obs @ best_coef, dtype=np.float64)
    residual_obs = np.abs(yy - pred_obs)
    threshold = _mad_threshold(residual_obs)
    inlier_obs = residual_obs <= threshold
    coef, _, residual_obs = _robust_refine_linear_model(
        design_obs,
        yy,
        inlier_obs,
        min_samples=sample_size,
        robust_iterations=robust_iterations,
    )
    pred_all = np.asarray(design_all @ coef, dtype=np.float64)
    residual_all = np.zeros(len(t), dtype=np.float64)
    residual_all[np.where(finite)[0]] = residual_obs
    return pred_all, residual_all


def _detect_outliers(
    raw_bbox: np.ndarray,
    frame_index: np.ndarray,
    cfg: BenchmarkConfig,
    *,
    detector: str,
    seed: int,
) -> dict[str, Any]:
    cx_raw, cy_raw, w_raw, h_raw = _bbox_to_center_xywh(raw_bbox)
    observed_mask = np.isfinite(raw_bbox).all(axis=1)
    rng = np.random.default_rng(seed)
    if detector == "msac":
        _, cx_res = _fit_detector_curve_msac(frame_index, cx_raw, cfg.bspline_knots, cfg.bspline_degree, rng=rng, robust_iterations=cfg.robust_iterations)
        _, cy_res = _fit_detector_curve_msac(frame_index, cy_raw, cfg.bspline_knots, cfg.bspline_degree, rng=rng, robust_iterations=cfg.robust_iterations)
        _, w_res = _fit_detector_curve_msac(frame_index, w_raw, cfg.bspline_knots, cfg.bspline_degree, rng=rng, robust_iterations=cfg.robust_iterations)
        _, h_res = _fit_detector_curve_msac(frame_index, h_raw, cfg.bspline_knots, cfg.bspline_degree, rng=rng, robust_iterations=cfg.robust_iterations)
    elif detector == "ransac":
        _, cx_res = _fit_detector_curve_ransac(frame_index, cx_raw, cfg.bspline_knots, cfg.bspline_degree, robust_iterations=cfg.robust_iterations)
        _, cy_res = _fit_detector_curve_ransac(frame_index, cy_raw, cfg.bspline_knots, cfg.bspline_degree, robust_iterations=cfg.robust_iterations)
        _, w_res = _fit_detector_curve_ransac(frame_index, w_raw, cfg.bspline_knots, cfg.bspline_degree, robust_iterations=cfg.robust_iterations)
        _, h_res = _fit_detector_curve_ransac(frame_index, h_raw, cfg.bspline_knots, cfg.bspline_degree, robust_iterations=cfg.robust_iterations)
    elif detector == "irls":
        _, cx_res = _fit_detector_curve_irls(frame_index, cx_raw, cfg.bspline_knots, cfg.bspline_degree, robust_iterations=cfg.robust_iterations)
        _, cy_res = _fit_detector_curve_irls(frame_index, cy_raw, cfg.bspline_knots, cfg.bspline_degree, robust_iterations=cfg.robust_iterations)
        _, w_res = _fit_detector_curve_irls(frame_index, w_raw, cfg.bspline_knots, cfg.bspline_degree, robust_iterations=cfg.robust_iterations)
        _, h_res = _fit_detector_curve_irls(frame_index, h_raw, cfg.bspline_knots, cfg.bspline_degree, robust_iterations=cfg.robust_iterations)
    elif detector == "lowess":
        _, cx_res = _fit_detector_curve_lowess(frame_index, cx_raw, frac=cfg.lowess_frac, robust_iterations=cfg.robust_iterations)
        _, cy_res = _fit_detector_curve_lowess(frame_index, cy_raw, frac=cfg.lowess_frac, robust_iterations=cfg.robust_iterations)
        _, w_res = _fit_detector_curve_lowess(frame_index, w_raw, frac=cfg.lowess_frac, robust_iterations=cfg.robust_iterations)
        _, h_res = _fit_detector_curve_lowess(frame_index, h_raw, frac=cfg.lowess_frac, robust_iterations=cfg.robust_iterations)
    elif detector == "lmeds":
        _, cx_res = _fit_detector_curve_lmeds(frame_index, cx_raw, cfg.bspline_knots, cfg.bspline_degree, rng=rng, n_iter=cfg.lmeds_trials, robust_iterations=cfg.robust_iterations)
        _, cy_res = _fit_detector_curve_lmeds(frame_index, cy_raw, cfg.bspline_knots, cfg.bspline_degree, rng=rng, n_iter=cfg.lmeds_trials, robust_iterations=cfg.robust_iterations)
        _, w_res = _fit_detector_curve_lmeds(frame_index, w_raw, cfg.bspline_knots, cfg.bspline_degree, rng=rng, n_iter=cfg.lmeds_trials, robust_iterations=cfg.robust_iterations)
        _, h_res = _fit_detector_curve_lmeds(frame_index, h_raw, cfg.bspline_knots, cfg.bspline_degree, rng=rng, n_iter=cfg.lmeds_trials, robust_iterations=cfg.robust_iterations)
    else:
        raise ValueError(f"Unknown detector: {detector}")
    anomaly_score = np.maximum.reduce([cx_res, cy_res, w_res, h_res])
    if np.nanmax(anomaly_score) > 0:
        anomaly_score = anomaly_score / float(np.nanmax(anomaly_score))
    flag_mask = _flag_from_anomaly_score(anomaly_score, observed_mask, cfg.anomaly_residual_quantile)
    return {
        "proxy_bbox": _make_tab_context_bbox(raw_bbox),
        "anomaly_score": anomaly_score,
        "flag_mask": flag_mask,
        "observed_mask": observed_mask,
    }


def _make_proxy_bbox(raw_bbox: np.ndarray, frame_index: np.ndarray) -> np.ndarray:
    observed = np.isfinite(raw_bbox).all(axis=1)
    cx, cy, w, h = _bbox_to_center_xywh(raw_bbox)
    if observed.sum() >= 2:
        cx_proxy = pchip_interpolate_1d(frame_index[observed].astype(np.float64), cx[observed], frame_index.astype(np.float64))
        cy_proxy = pchip_interpolate_1d(frame_index[observed].astype(np.float64), cy[observed], frame_index.astype(np.float64))
        w_proxy = pchip_interpolate_1d(frame_index[observed].astype(np.float64), w[observed], frame_index.astype(np.float64))
        h_proxy = pchip_interpolate_1d(frame_index[observed].astype(np.float64), h[observed], frame_index.astype(np.float64))
    else:
        med = np.nanmedian(raw_bbox, axis=0)
        fill = np.where(np.isfinite(med), med, np.array([0.0, 0.0, 1.0, 1.0]))
        return np.tile(fill[None, :], (len(raw_bbox), 1))
    return _center_xywh_to_bbox(cx_proxy, cy_proxy, np.maximum(w_proxy, 1.0), np.maximum(h_proxy, 1.0))


def _delete_flagged_bbox(raw_bbox: np.ndarray, flag_mask: np.ndarray) -> np.ndarray:
    deleted = np.asarray(raw_bbox, dtype=np.float64).copy()
    mask = np.asarray(flag_mask, dtype=bool)
    deleted[mask] = np.nan
    return deleted


def _build_regression_features(bbox: np.ndarray, frame_index: np.ndarray) -> pd.DataFrame:
    cx, cy, w, h = _bbox_to_center_xywh(bbox)
    t = np.asarray(frame_index, dtype=np.float64)
    dt_prev = np.diff(t, prepend=t[0])
    dt_prev[0] = 1.0
    dx_prev = np.diff(cx, prepend=cx[0])
    dy_prev = np.diff(cy, prepend=cy[0])
    vx = dx_prev / np.maximum(dt_prev, 1.0)
    vy = dy_prev / np.maximum(dt_prev, 1.0)
    ax = np.diff(vx, prepend=vx[0]) / np.maximum(dt_prev, 1.0)
    ay = np.diff(vy, prepend=vy[0]) / np.maximum(dt_prev, 1.0)
    heading = np.arctan2(vy, vx + 1e-12)
    features = pd.DataFrame(
        {
            "dx_prev": dx_prev,
            "dy_prev": dy_prev,
            "vx": vx,
            "vy": vy,
            "ax": ax,
            "ay": ay,
            "heading_cos": np.cos(heading),
            "heading_sin": np.sin(heading),
            "delta_frame_prev": dt_prev,
            "bbox_w": w,
            "bbox_h": h,
        }
    )
    features = features.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    return features


def _build_training_table(records: Iterable[TrajectoryRecord]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    feature_rows: list[pd.DataFrame] = []
    targets: dict[str, list[np.ndarray]] = {"dx": [], "dy": [], "w": [], "h": []}
    for rec in records:
        bbox = np.asarray(rec.gt_bbox_dense, dtype=np.float64)
        source_bbox = rec.raw_bbox_dense if rec.raw_bbox_dense is not None else bbox
        features = _build_regression_features(_make_tab_context_bbox(np.asarray(source_bbox, dtype=np.float64)), rec.frame_index)
        cx, cy, w, h = _bbox_to_center_xywh(bbox)
        dx = np.diff(cx, prepend=cx[0])
        dy = np.diff(cy, prepend=cy[0])
        feature_rows.append(features)
        targets["dx"].append(dx)
        targets["dy"].append(dy)
        targets["w"].append(w)
        targets["h"].append(h)
    X = pd.concat(feature_rows, axis=0, ignore_index=True) if feature_rows else pd.DataFrame()
    y = {k: np.concatenate(v, axis=0) if v else np.zeros(0, dtype=np.float64) for k, v in targets.items()}
    return X, y


def _fit_tabpfn_regressors(train_X: pd.DataFrame, train_y: dict[str, np.ndarray], cfg: BenchmarkConfig) -> dict[str, Any]:
    if TabPFNRegressor is None:
        raise RuntimeError("tabpfn is not installed; cannot run method c/d")
    if train_X.empty:
        raise RuntimeError("Empty training table for TabPFN regression")

    rng = np.random.default_rng(cfg.seed)
    max_rows = int(max(64, cfg.tabpfn_max_train_rows))
    if len(train_X) > max_rows:
        keep = rng.choice(np.arange(len(train_X)), size=max_rows, replace=False)
        keep.sort()
        X_fit = train_X.iloc[keep].reset_index(drop=True)
        y_fit = {k: v[keep] for k, v in train_y.items()}
    else:
        X_fit = train_X.reset_index(drop=True)
        y_fit = train_y

    models: dict[str, Any] = {}
    for target_name in ("dx", "dy", "w", "h"):
        model = TabPFNRegressor(device=cfg.tabpfn_device, random_state=cfg.seed)
        model.fit(X_fit, y_fit[target_name])
        models[target_name] = model
    return models


def _reconstruct_with_poisson(
    predicted_dx: np.ndarray,
    trusted_abs: np.ndarray,
    trusted_mask: np.ndarray,
) -> np.ndarray:
    n = len(predicted_dx)
    A_rows: list[np.ndarray] = []
    b_rows: list[float] = []
    trusted_mask = np.asarray(trusted_mask, dtype=bool)
    trusted_abs = np.asarray(trusted_abs, dtype=np.float64)
    anchor_candidates = np.where(trusted_mask & np.isfinite(trusted_abs))[0]
    if anchor_candidates.size > 0:
        anchor_idx = int(anchor_candidates[0])
        anchor_value = float(trusted_abs[anchor_idx])
    else:
        finite_idx = np.where(np.isfinite(trusted_abs))[0]
        anchor_idx = int(finite_idx[0]) if finite_idx.size > 0 else 0
        anchor_value = float(trusted_abs[anchor_idx]) if finite_idx.size > 0 else 0.0
    row0 = np.zeros(n, dtype=np.float64)
    row0[anchor_idx] = 1.0
    A_rows.append(row0)
    b_rows.append(anchor_value)
    for i in range(1, n):
        row = np.zeros(n, dtype=np.float64)
        row[i] = 1.0
        row[i - 1] = -1.0
        A_rows.append(row)
        b_rows.append(float(predicted_dx[i]))
    for i in np.where(trusted_mask & np.isfinite(trusted_abs))[0]:
        row = np.zeros(n, dtype=np.float64)
        row[i] = 2.0
        A_rows.append(row)
        b_rows.append(float(2.0 * trusted_abs[i]))
    A = np.vstack(A_rows)
    b = np.asarray(b_rows, dtype=np.float64)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.asarray(x, dtype=np.float64)


def _reconstruct_with_bspline(
    predicted_dx: np.ndarray,
    trusted_abs: np.ndarray,
    trusted_mask: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    provisional = np.cumsum(np.asarray(predicted_dx, dtype=np.float64))
    provisional -= provisional[0]
    provisional += float(trusted_abs[0])
    target = provisional.copy()
    target[trusted_mask] = trusted_abs[trusted_mask]
    weights = np.where(trusted_mask, 8.0, 1.5).astype(np.float64)
    if UnivariateSpline is None or len(t) < 5:
        return target
    tt = np.asarray(t, dtype=np.float64)
    spline = UnivariateSpline(tt, target, w=weights, s=len(tt) * 0.5, k=min(3, len(tt) - 1))
    return np.asarray(spline(tt), dtype=np.float64)


def _tabpfn_method(
    raw_bbox: np.ndarray,
    frame_index: np.ndarray,
    cfg: BenchmarkConfig,
    models: dict[str, Any],
    detector: str,
    seed: int,
) -> dict[str, dict[str, Any]]:
    detection = _detect_outliers(raw_bbox, frame_index, cfg, detector=detector, seed=seed)
    anomaly_score = np.asarray(detection["anomaly_score"], dtype=np.float64)
    flag_mask = np.asarray(detection["flag_mask"], dtype=bool)
    observed_mask = np.asarray(detection["observed_mask"], dtype=bool)
    deleted_bbox = _delete_flagged_bbox(raw_bbox, flag_mask)
    context_bbox = _make_tab_context_bbox(deleted_bbox)
    features = _build_regression_features(context_bbox, frame_index)
    pred_dx = np.asarray(models["dx"].predict(features), dtype=np.float64)
    pred_dy = np.asarray(models["dy"].predict(features), dtype=np.float64)
    pred_w = np.asarray(models["w"].predict(features), dtype=np.float64)
    pred_h = np.asarray(models["h"].predict(features), dtype=np.float64)
    trusted_mask = (~flag_mask) & observed_mask
    deleted_cx, deleted_cy, deleted_w, deleted_h = _bbox_to_center_xywh(deleted_bbox)
    obs_dx = np.diff(deleted_cx, prepend=deleted_cx[0])
    obs_dy = np.diff(deleted_cy, prepend=deleted_cy[0])
    obs_dx[0] = 0.0
    obs_dy[0] = 0.0
    mixed_dx = pred_dx.copy()
    mixed_dy = pred_dy.copy()
    valid_dx = trusted_mask.copy()
    valid_dx[1:] &= trusted_mask[:-1]
    valid_dx[0] &= np.isfinite(deleted_cx[0]) and np.isfinite(deleted_cy[0])
    mixed_dx[valid_dx] = obs_dx[valid_dx]
    mixed_dy[valid_dx] = obs_dy[valid_dx]

    cx_poisson = _reconstruct_with_poisson(mixed_dx, deleted_cx, trusted_mask)
    cy_poisson = _reconstruct_with_poisson(mixed_dy, deleted_cy, trusted_mask)
    cx_bspline = _reconstruct_with_bspline(mixed_dx, deleted_cx, trusted_mask, frame_index)
    cy_bspline = _reconstruct_with_bspline(mixed_dy, deleted_cy, trusted_mask, frame_index)

    w_final = pred_w.copy()
    h_final = pred_h.copy()
    w_final[trusted_mask] = deleted_w[trusted_mask]
    h_final[trusted_mask] = deleted_h[trusted_mask]
    w_final = _adaptive_savgol(np.maximum(w_final, 1.0), frame_index, window_length=cfg.mild_sg_window, polyorder=cfg.mild_sg_polyorder)
    h_final = _adaptive_savgol(np.maximum(h_final, 1.0), frame_index, window_length=cfg.mild_sg_window, polyorder=cfg.mild_sg_polyorder)
    cx_poisson = _adaptive_savgol(cx_poisson, frame_index, window_length=cfg.mild_sg_window, polyorder=cfg.mild_sg_polyorder)
    cy_poisson = _adaptive_savgol(cy_poisson, frame_index, window_length=cfg.mild_sg_window, polyorder=cfg.mild_sg_polyorder)
    cx_bspline = _adaptive_savgol(cx_bspline, frame_index, window_length=cfg.mild_sg_window, polyorder=cfg.mild_sg_polyorder)
    cy_bspline = _adaptive_savgol(cy_bspline, frame_index, window_length=cfg.mild_sg_window, polyorder=cfg.mild_sg_polyorder)

    return {
        "poisson": {
            "variant": "poisson",
            "bbox": _center_xywh_to_bbox(cx_poisson, cy_poisson, w_final, h_final),
            "center": np.column_stack([cx_poisson, cy_poisson]),
            "anomaly_score": anomaly_score,
            "flag_mask": flag_mask,
            "deleted_bbox": deleted_bbox,
        },
        "bspline": {
            "variant": "bspline",
            "bbox": _center_xywh_to_bbox(cx_bspline, cy_bspline, w_final, h_final),
            "center": np.column_stack([cx_bspline, cy_bspline]),
            "anomaly_score": anomaly_score,
            "flag_mask": flag_mask,
            "deleted_bbox": deleted_bbox,
        },
    }


def _msac_poisson_method(
    raw_bbox: np.ndarray,
    frame_index: np.ndarray,
    cfg: BenchmarkConfig,
    seed: int,
) -> dict[str, Any]:
    proxy_bbox = _make_proxy_bbox(raw_bbox, frame_index)
    cx_proxy, cy_proxy, w_proxy, h_proxy = _bbox_to_center_xywh(proxy_bbox)
    rng = np.random.default_rng(seed)

    _, cx_res = _fit_detector_curve_msac(frame_index, cx_proxy, cfg.bspline_knots, cfg.bspline_degree, rng=rng, robust_iterations=cfg.robust_iterations)
    _, cy_res = _fit_detector_curve_msac(frame_index, cy_proxy, cfg.bspline_knots, cfg.bspline_degree, rng=rng, robust_iterations=cfg.robust_iterations)
    _, w_res = _fit_detector_curve_msac(frame_index, w_proxy, cfg.bspline_knots, cfg.bspline_degree, rng=rng, robust_iterations=cfg.robust_iterations)
    _, h_res = _fit_detector_curve_msac(frame_index, h_proxy, cfg.bspline_knots, cfg.bspline_degree, rng=rng, robust_iterations=cfg.robust_iterations)

    anomaly_score = np.maximum.reduce([cx_res, cy_res, w_res, h_res])
    if np.nanmax(anomaly_score) > 0:
        anomaly_score = anomaly_score / float(np.nanmax(anomaly_score))
    observed_mask = np.isfinite(raw_bbox).all(axis=1)
    threshold = float(np.quantile(anomaly_score[observed_mask], cfg.anomaly_residual_quantile)) if observed_mask.any() else 1.0
    flag_mask = anomaly_score >= threshold
    flag_mask |= ~observed_mask

    deleted_bbox = _delete_flagged_bbox(raw_bbox, flag_mask)
    deleted_proxy_bbox = _make_proxy_bbox(deleted_bbox, frame_index)
    del_cx, del_cy, del_w, del_h = _bbox_to_center_xywh(deleted_proxy_bbox)

    fit_cx, _ = _fit_detector_curve_msac(frame_index, del_cx, cfg.bspline_knots, cfg.bspline_degree, rng=rng, robust_iterations=cfg.robust_iterations)
    fit_cy, _ = _fit_detector_curve_msac(frame_index, del_cy, cfg.bspline_knots, cfg.bspline_degree, rng=rng, robust_iterations=cfg.robust_iterations)
    fit_w, _ = _fit_detector_curve_msac(frame_index, del_w, cfg.bspline_knots, cfg.bspline_degree, rng=rng, robust_iterations=cfg.robust_iterations)
    fit_h, _ = _fit_detector_curve_msac(frame_index, del_h, cfg.bspline_knots, cfg.bspline_degree, rng=rng, robust_iterations=cfg.robust_iterations)

    pred_dx = np.diff(fit_cx, prepend=fit_cx[0])
    pred_dy = np.diff(fit_cy, prepend=fit_cy[0])
    pred_dx[0] = 0.0
    pred_dy[0] = 0.0
    trusted_mask = (~flag_mask) & observed_mask
    cx_poisson = _reconstruct_with_poisson(pred_dx, del_cx, trusted_mask)
    cy_poisson = _reconstruct_with_poisson(pred_dy, del_cy, trusted_mask)
    cx_poisson = _adaptive_savgol(cx_poisson, frame_index, window_length=cfg.mild_sg_window, polyorder=cfg.mild_sg_polyorder)
    cy_poisson = _adaptive_savgol(cy_poisson, frame_index, window_length=cfg.mild_sg_window, polyorder=cfg.mild_sg_polyorder)
    fit_w = _adaptive_savgol(np.maximum(fit_w, 1.0), frame_index, window_length=cfg.mild_sg_window, polyorder=cfg.mild_sg_polyorder)
    fit_h = _adaptive_savgol(np.maximum(fit_h, 1.0), frame_index, window_length=cfg.mild_sg_window, polyorder=cfg.mild_sg_polyorder)

    return {
        "variant": "poisson",
        "bbox": _center_xywh_to_bbox(cx_poisson, cy_poisson, fit_w, fit_h),
        "center": np.column_stack([cx_poisson, cy_poisson]),
        "anomaly_score": anomaly_score,
        "flag_mask": flag_mask,
        "deleted_bbox": deleted_bbox,
    }


def _evaluate_prediction(
    name: str,
    bbox_pred: np.ndarray,
    center_pred: np.ndarray,
    anomaly_score: np.ndarray,
    flag_mask: np.ndarray,
    gt_bbox: np.ndarray,
    corrupt_mask: np.ndarray,
) -> dict[str, Any]:
    gt_center = np.column_stack(_bbox_to_center_xywh(gt_bbox)[:2])
    y_true = np.asarray(corrupt_mask, dtype=bool)
    y_pred = np.asarray(flag_mask, dtype=bool)
    center_error = np.linalg.norm(center_pred - gt_center, axis=1)
    abs_err_xy = np.abs(center_pred - gt_center)
    mse = float(np.mean((center_pred - gt_center) ** 2))
    mae = float(np.mean(abs_err_xy))
    dtw = _dtw_distance(center_pred, gt_center)
    iou = _bbox_iou(bbox_pred, gt_bbox)
    bbox_l1 = np.mean(np.abs(bbox_pred - gt_bbox), axis=1)
    try:
        auroc = float(roc_auc_score(corrupt_mask.astype(np.int32), anomaly_score.astype(np.float64)))
    except Exception:
        auroc = float("nan")
    tp = int(np.sum(y_true & y_pred))
    tn = int(np.sum((~y_true) & (~y_pred)))
    fp = int(np.sum((~y_true) & y_pred))
    fn = int(np.sum(y_true & (~y_pred)))
    total = max(1, len(y_true))
    accuracy = float((tp + tn) / total)
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    return {
        "method": name,
        "iou_mean": float(np.mean(iou)),
        "iou_std_within_case": float(np.std(iou)),
        "error_mean": float(np.mean(center_error)),
        "error_std_within_case": float(np.std(center_error)),
        "auroc": auroc,
        "roi_l1_mean": float(np.mean(bbox_l1)),
        "mse": mse,
        "mae": mae,
        "center_error": float(np.mean(center_error)),
        "dtw": float(dtw),
        "n_flagged": int(np.sum(flag_mask)),
        "det_accuracy": accuracy,
        "det_precision": precision,
        "det_recall": recall,
        "det_f1": f1,
        "det_specificity": specificity,
        "det_tp": tp,
        "det_tn": tn,
        "det_fp": fp,
        "det_fn": fn,
    }


def _evaluate_detection(
    name: str,
    anomaly_score: np.ndarray,
    flag_mask: np.ndarray,
    corrupt_mask: np.ndarray,
) -> dict[str, Any]:
    y_true = np.asarray(corrupt_mask, dtype=bool)
    y_pred = np.asarray(flag_mask, dtype=bool)
    try:
        auroc = float(roc_auc_score(y_true.astype(np.int32), np.asarray(anomaly_score, dtype=np.float64)))
    except Exception:
        auroc = float("nan")
    tp = int(np.sum(y_true & y_pred))
    tn = int(np.sum((~y_true) & (~y_pred)))
    fp = int(np.sum((~y_true) & y_pred))
    fn = int(np.sum(y_true & (~y_pred)))
    total = max(1, len(y_true))
    accuracy = float((tp + tn) / total)
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    return {
        "method": name,
        "auroc": auroc,
        "det_accuracy": accuracy,
        "det_precision": precision,
        "det_recall": recall,
        "det_f1": f1,
        "det_specificity": specificity,
        "det_tp": tp,
        "det_tn": tn,
        "det_fp": fp,
        "det_fn": fn,
        "n_flagged": int(np.sum(y_pred)),
    }


def _plot_bbox_timeseries(
    output_path: Path,
    title: str,
    frame_index: np.ndarray,
    series_map: dict[str, np.ndarray],
    raw_missing_mask: np.ndarray | None = None,
) -> None:
    palette = {
        "GT": "black",
        "raw": "#999999",
        "pipe": "#1f77b4",
        "r_spline": "#ff7f0e",
        "ms_poi": "#8c564b",
        "ms_tab_p": "#2ca02c",
        "ms_tab_b": "#17becf",
        "ra_tab_p": "#d62728",
        "ra_tab_b": "#9467bd",
        "ir_tab_p": "#bcbd22",
        "ir_tab_b": "#e377c2",
        "lw_tab_p": "#ff9896",
        "lw_tab_b": "#98df8a",
        "lm_tab_p": "#c5b0d5",
        "lm_tab_b": "#c49c94",
    }
    components = [
        ("center_x", 0, "Center X"),
        ("center_y", 1, "Center Y"),
        ("bbox_w", 2, "BBox W"),
        ("bbox_h", 3, "BBox H"),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(10.5, 10.0), dpi=130, sharex=True)
    for ax, (key, _, label) in zip(axes, components):
        for name, arr in series_map.items():
            ax.plot(frame_index, arr[key], lw=1.8 if name != "raw" else 1.4, alpha=1.0 if name != "raw" else 0.9, color=palette.get(name), label=name)
        if raw_missing_mask is not None and "raw" in series_map:
            miss = np.asarray(raw_missing_mask, dtype=bool)
            if miss.any():
                ax.scatter(
                    frame_index[miss],
                    series_map["raw"][key][miss],
                    s=28,
                    facecolors="none",
                    edgecolors="#444444",
                    linewidths=1.1,
                    marker="o",
                    label="raw missing",
                )
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[0].set_title(title)
    axes[-1].set_xlabel("Frame")
    handles, labels = axes[0].get_legend_handles_labels()
    uniq: dict[str, Any] = {}
    for h, l in zip(handles, labels):
        uniq.setdefault(l, h)
    axes[0].legend(uniq.values(), uniq.keys(), loc="best", frameon=True, ncol=min(4, len(uniq)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _bbox_components(bbox: np.ndarray) -> dict[str, np.ndarray]:
    cx, cy, w, h = _bbox_to_center_xywh(bbox)
    return {
        "center_x": np.asarray(cx, dtype=np.float64),
        "center_y": np.asarray(cy, dtype=np.float64),
        "bbox_w": np.asarray(w, dtype=np.float64),
        "bbox_h": np.asarray(h, dtype=np.float64),
    }


def _plot_detection_case(
    output_path: Path,
    title: str,
    frame_index: np.ndarray,
    gt_bbox: np.ndarray,
    raw_bbox: np.ndarray,
    true_corrupt_mask: np.ndarray,
    detector_flags: dict[str, np.ndarray],
) -> None:
    gt_center = np.column_stack(_bbox_to_center_xywh(gt_bbox)[:2])
    proxy_center = np.column_stack(_bbox_to_center_xywh(_make_proxy_bbox(raw_bbox, frame_index))[:2])
    detector_style = {
        "PIPE": ("#8c564b", "P"),
        "MSAC": ("#2ca02c", "o"),
        "RANSAC": ("#d62728", "x"),
        "IRLS": ("#1f77b4", "^"),
        "BiLOWESS": ("#ff7f0e", "s"),
        "LMedS": ("#9467bd", "D"),
    }
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 6.6), dpi=130, sharex=True)
    true_segments = _contiguous_segments(true_corrupt_mask)
    for ax, dim, label in zip(axes, range(2), ["Center X", "Center Y"]):
        ax.plot(frame_index, gt_center[:, dim], color="black", lw=2.0, label="GT")
        ax.plot(frame_index, proxy_center[:, dim], color="#777777", lw=1.2, alpha=0.8, label="raw/proxy")
        for idx, (s, e) in enumerate(true_segments):
            ax.axvspan(frame_index[s], frame_index[e - 1], color="#ffcccc", alpha=0.45, label="synthetic corrupt GT" if idx == 0 else None)
        for det_name, flag_mask in detector_flags.items():
            color, marker = detector_style.get(det_name, ("#444444", "o"))
            flag_mask = np.asarray(flag_mask, dtype=bool)
            ax.scatter(frame_index[flag_mask], proxy_center[flag_mask, dim], c=color, s=24, marker=marker, label=f"{det_name} flagged")
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
    axes[0].set_title(title)
    axes[-1].set_xlabel("Frame")
    handles, labels = axes[0].get_legend_handles_labels()
    uniq: dict[str, Any] = {}
    for h, l in zip(handles, labels):
        uniq.setdefault(l, h)
    axes[0].legend(uniq.values(), uniq.keys(), loc="best", frameon=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_reconstruction_case(
    output_path: Path,
    title: str,
    frame_index: np.ndarray,
    gt_bbox: np.ndarray,
    ms_poi: np.ndarray,
    c_poisson: np.ndarray,
    c_bspline: np.ndarray,
    d_poisson: np.ndarray,
    d_bspline: np.ndarray,
    i_poisson: np.ndarray,
    i_bspline: np.ndarray,
    lw_poisson: np.ndarray,
    lw_bspline: np.ndarray,
    lm_poisson: np.ndarray,
    lm_bspline: np.ndarray,
) -> None:
    _plot_bbox_timeseries(
        output_path,
        title,
        frame_index,
        {
            "GT": _bbox_components(gt_bbox),
            "ms_poi": _bbox_components(ms_poi),
            "ms_tab_p": _bbox_components(c_poisson),
            "ms_tab_b": _bbox_components(c_bspline),
            "ra_tab_p": _bbox_components(d_poisson),
            "ra_tab_b": _bbox_components(d_bspline),
            "ir_tab_p": _bbox_components(i_poisson),
            "ir_tab_b": _bbox_components(i_bspline),
            "lw_tab_p": _bbox_components(lw_poisson),
            "lw_tab_b": _bbox_components(lw_bspline),
            "lm_tab_p": _bbox_components(lm_poisson),
            "lm_tab_b": _bbox_components(lm_bspline),
        },
    )


def _aggregate_summary(rows: list[dict[str, Any]], method_order: list[str]) -> list[dict[str, Any]]:
    df = pd.DataFrame(rows)
    summary: list[dict[str, Any]] = []
    for method in method_order:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        entry = {"method": method, "n_cases": int(len(sub))}
        for metric in [
            "iou_mean",
            "error_mean",
            "auroc",
            "roi_l1_mean",
            "mse",
            "mae",
            "center_error",
            "dtw",
            "det_accuracy",
            "det_precision",
            "det_recall",
            "det_f1",
            "det_specificity",
        ]:
            vals = sub[metric].to_numpy(dtype=np.float64)
            entry[f"{metric}_mean"] = _safe_mean(vals)
            entry[f"{metric}_std"] = _safe_std(vals)
        summary.append(entry)
    return summary


def _aggregate_detection_summary(rows: list[dict[str, Any]], method_order: list[str]) -> list[dict[str, Any]]:
    df = pd.DataFrame(rows)
    summary: list[dict[str, Any]] = []
    for method in method_order:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        entry = {"method": method, "n_cases": int(len(sub))}
        for metric in ["auroc", "det_accuracy", "det_precision", "det_recall", "det_f1", "det_specificity"]:
            vals = sub[metric].to_numpy(dtype=np.float64)
            entry[f"{metric}_mean"] = _safe_mean(vals)
            entry[f"{metric}_std"] = _safe_std(vals)
        summary.append(entry)
    return summary


def run_benchmark(cfg: BenchmarkConfig) -> dict[str, Any]:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_source = _normalise_raw_source(cfg.raw_source)

    detector_info: dict[str, Any] = {}
    if raw_source == "detector":
        train_root = Path(cfg.train_dataset_root or cfg.dataset_root)
        test_root = Path(cfg.test_dataset_root or cfg.dataset_root)
        train_records, detector_info_train = load_detector_trajectories(train_root, cfg, cache_tag="train")
        test_records_all, detector_info_test = load_detector_trajectories(test_root, cfg, cache_tag="test")
        detector_info = {
            "checkpoint": detector_info_train.get("checkpoint") or detector_info_test.get("checkpoint"),
            "experiment_dir": detector_info_train.get("experiment_dir") or detector_info_test.get("experiment_dir"),
            "train_cache_root": detector_info_train.get("cache_root"),
            "test_cache_root": detector_info_test.get("cache_root"),
            "model_name": detector_info_train.get("model_name") or detector_info_test.get("model_name"),
        }
        if len(train_records) < 1:
            raise RuntimeError(f"Need at least 1 usable detector-backed training trajectory under {train_root}")
        if len(test_records_all) < 1:
            raise RuntimeError(f"Need at least 1 usable detector-backed test trajectory under {test_root}")
        test_records = test_records_all
        dataset_root = test_root
        records = train_records + test_records_all
        test_case_counts = allocate_test_cases(len(test_records), cfg.target_test_cases, cfg.cases_per_trajectory)
    else:
        dataset_root = Path(cfg.dataset_root)
        records = load_trajectories(
            dataset_root,
            smooth_window=cfg.mild_sg_window,
            smooth_polyorder=cfg.mild_sg_polyorder,
            max_trajectories=cfg.max_trajectories,
        )
        if len(records) < 2:
            raise RuntimeError(f"Need at least 2 usable trajectories under {dataset_root} to build 8:2 train/test split")
        train_records, test_records = split_trajectories(records, cfg.train_ratio, cfg.seed)
        test_case_counts = allocate_test_cases(len(test_records), cfg.target_test_cases, cfg.cases_per_trajectory)

    train_X, train_y = _build_training_table(train_records)
    tabpfn_models = _fit_tabpfn_regressors(train_X, train_y, cfg)

    case_rows: list[dict[str, Any]] = []
    case_debug: list[dict[str, Any]] = []

    test_iter = []
    for test_i, rec in enumerate(test_records):
        for case_idx in range(test_case_counts[test_i]):
            test_iter.append((test_i, rec, case_idx))

    for test_i, rec, case_idx in test_iter:
        seed_case = int(cfg.seed + test_i * 1000 + case_idx * 17)
        case_rng = np.random.default_rng(seed_case)
        corruption_input = (
            np.asarray(rec.raw_bbox_dense, dtype=np.float64)
            if raw_source == "detector" and rec.raw_bbox_dense is not None
            else np.asarray(rec.gt_bbox_dense, dtype=np.float64)
        )
        corruption = _corrupt_bbox(corruption_input, rec.frame_index, rng=case_rng, max_ratio=cfg.max_corruption_ratio, cfg=cfg)
        raw_bbox = np.asarray(corruption["raw_bbox"], dtype=np.float64)
        corrupt_mask = np.asarray(corruption["corrupt_mask"], dtype=bool)
        missing_mask = np.asarray(corruption["missing_mask"], dtype=bool)
        observed_mask = np.isfinite(raw_bbox).all(axis=1)

        case_prefix = Path(rec.trajectory_id).with_suffix("").as_posix().replace("/", "__").replace("\\", "__").replace(" ", "_")
        case_id = f"{case_prefix}__case{case_idx:02d}"

        raw_proxy_bbox = _make_tab_context_bbox(raw_bbox)
        raw_center = np.column_stack(_bbox_to_center_xywh(raw_proxy_bbox)[:2])
        raw_features = _build_regression_features(raw_proxy_bbox, rec.frame_index)
        raw_score = np.hypot(
            raw_features["vx"].to_numpy(dtype=np.float64),
            raw_features["vy"].to_numpy(dtype=np.float64),
        )
        if np.nanmax(raw_score) > 0:
            raw_score = raw_score / float(np.nanmax(raw_score))
        raw_flag = _flag_from_anomaly_score(raw_score, observed_mask, cfg.anomaly_residual_quantile)
        raw_eval = _evaluate_prediction(
            "raw",
            raw_proxy_bbox,
            raw_center,
            raw_score,
            raw_flag,
            rec.gt_bbox_dense,
            corrupt_mask,
        )
        for key in ["auroc", "det_accuracy", "det_precision", "det_recall", "det_f1", "det_specificity"]:
            raw_eval[key] = 0.0
        for key in ["det_tp", "det_tn", "det_fp", "det_fn", "n_flagged"]:
            raw_eval[key] = 0

        method_a = _baseline_method(raw_bbox, rec.frame_index, cfg)
        eval_a = _evaluate_prediction("pipe", method_a["bbox"], method_a["center"], method_a["anomaly_score"], method_a["flag_mask"], rec.gt_bbox_dense, corrupt_mask)

        method_b = _ransac_bspline_method(raw_bbox, rec.frame_index, cfg)
        eval_b = _evaluate_prediction("r_spline", method_b["bbox"], method_b["center"], method_b["anomaly_score"], method_b["flag_mask"], rec.gt_bbox_dense, corrupt_mask)

        method_ms_po = _msac_poisson_method(raw_bbox, rec.frame_index, cfg, seed=seed_case + 101)
        eval_ms_po = _evaluate_prediction("ms_poi", method_ms_po["bbox"], method_ms_po["center"], method_ms_po["anomaly_score"], method_ms_po["flag_mask"], rec.gt_bbox_dense, corrupt_mask)

        method_c = _tabpfn_method(raw_bbox, rec.frame_index, cfg, tabpfn_models, detector="msac", seed=seed_case)
        eval_c_poisson = _evaluate_prediction("ms_tab_p", method_c["poisson"]["bbox"], method_c["poisson"]["center"], method_c["poisson"]["anomaly_score"], method_c["poisson"]["flag_mask"], rec.gt_bbox_dense, corrupt_mask)
        eval_c_bspline = _evaluate_prediction("ms_tab_b", method_c["bspline"]["bbox"], method_c["bspline"]["center"], method_c["bspline"]["anomaly_score"], method_c["bspline"]["flag_mask"], rec.gt_bbox_dense, corrupt_mask)

        method_d = _tabpfn_method(raw_bbox, rec.frame_index, cfg, tabpfn_models, detector="ransac", seed=seed_case + 1)
        eval_d_poisson = _evaluate_prediction("ra_tab_p", method_d["poisson"]["bbox"], method_d["poisson"]["center"], method_d["poisson"]["anomaly_score"], method_d["poisson"]["flag_mask"], rec.gt_bbox_dense, corrupt_mask)
        eval_d_bspline = _evaluate_prediction("ra_tab_b", method_d["bspline"]["bbox"], method_d["bspline"]["center"], method_d["bspline"]["anomaly_score"], method_d["bspline"]["flag_mask"], rec.gt_bbox_dense, corrupt_mask)

        method_i = _tabpfn_method(raw_bbox, rec.frame_index, cfg, tabpfn_models, detector="irls", seed=seed_case + 2)
        eval_i_poisson = _evaluate_prediction("ir_tab_p", method_i["poisson"]["bbox"], method_i["poisson"]["center"], method_i["poisson"]["anomaly_score"], method_i["poisson"]["flag_mask"], rec.gt_bbox_dense, corrupt_mask)
        eval_i_bspline = _evaluate_prediction("ir_tab_b", method_i["bspline"]["bbox"], method_i["bspline"]["center"], method_i["bspline"]["anomaly_score"], method_i["bspline"]["flag_mask"], rec.gt_bbox_dense, corrupt_mask)

        method_lw = _tabpfn_method(raw_bbox, rec.frame_index, cfg, tabpfn_models, detector="lowess", seed=seed_case + 3)
        eval_lw_poisson = _evaluate_prediction("lw_tab_p", method_lw["poisson"]["bbox"], method_lw["poisson"]["center"], method_lw["poisson"]["anomaly_score"], method_lw["poisson"]["flag_mask"], rec.gt_bbox_dense, corrupt_mask)
        eval_lw_bspline = _evaluate_prediction("lw_tab_b", method_lw["bspline"]["bbox"], method_lw["bspline"]["center"], method_lw["bspline"]["anomaly_score"], method_lw["bspline"]["flag_mask"], rec.gt_bbox_dense, corrupt_mask)

        method_lm = _tabpfn_method(raw_bbox, rec.frame_index, cfg, tabpfn_models, detector="lmeds", seed=seed_case + 4)
        eval_lm_poisson = _evaluate_prediction("lm_tab_p", method_lm["poisson"]["bbox"], method_lm["poisson"]["center"], method_lm["poisson"]["anomaly_score"], method_lm["poisson"]["flag_mask"], rec.gt_bbox_dense, corrupt_mask)
        eval_lm_bspline = _evaluate_prediction("lm_tab_b", method_lm["bspline"]["bbox"], method_lm["bspline"]["center"], method_lm["bspline"]["anomaly_score"], method_lm["bspline"]["flag_mask"], rec.gt_bbox_dense, corrupt_mask)

        irls_det = _detect_outliers(raw_bbox, rec.frame_index, cfg, detector="irls", seed=seed_case + 200)
        lowess_det = _detect_outliers(raw_bbox, rec.frame_index, cfg, detector="lowess", seed=seed_case + 201)
        lmeds_det = _detect_outliers(raw_bbox, rec.frame_index, cfg, detector="lmeds", seed=seed_case + 202)
        detector_flags = {
            "PIPE": method_a["flag_mask"],
            "MSAC": method_c["poisson"]["flag_mask"],
            "RANSAC": method_d["poisson"]["flag_mask"],
            "IRLS": irls_det["flag_mask"],
            "BiLOWESS": lowess_det["flag_mask"],
            "LMedS": lmeds_det["flag_mask"],
        }

        shared = {
            "case_id": case_id,
            "trajectory_id": rec.trajectory_id,
            "source_json": rec.source_json,
            "patterns": ",".join(corruption["pattern_names"]),
            "seed": seed_case,
            "n_frames": int(len(rec.frame_index)),
            "corruption_ratio": float(corrupt_mask.mean()),
            "target_corruption_ratio": float(cfg.max_corruption_ratio),
            "raw_source": raw_source,
        }
        for metrics in [raw_eval, eval_a, eval_b, eval_ms_po, eval_c_poisson, eval_c_bspline, eval_d_poisson, eval_d_bspline, eval_i_poisson, eval_i_bspline, eval_lw_poisson, eval_lw_bspline, eval_lm_poisson, eval_lm_bspline]:
            row = dict(shared)
            row.update(metrics)
            case_rows.append(row)

        gt_series = _bbox_components(rec.gt_bbox_dense)
        raw_series = _bbox_components(raw_proxy_bbox)
        single_method_boxes = {
            "pipe": method_a["bbox"],
            "r_spline": method_b["bbox"],
            "ms_poi": method_ms_po["bbox"],
            "ms_tab_p": method_c["poisson"]["bbox"],
            "ms_tab_b": method_c["bspline"]["bbox"],
            "ra_tab_p": method_d["poisson"]["bbox"],
            "ra_tab_b": method_d["bspline"]["bbox"],
            "ir_tab_p": method_i["poisson"]["bbox"],
            "ir_tab_b": method_i["bspline"]["bbox"],
            "lw_tab_p": method_lw["poisson"]["bbox"],
            "lw_tab_b": method_lw["bspline"]["bbox"],
            "lm_tab_p": method_lm["poisson"]["bbox"],
            "lm_tab_b": method_lm["bspline"]["bbox"],
        }
        for method_name, method_bbox in single_method_boxes.items():
            _plot_bbox_timeseries(
                output_dir / "plots" / "method_single" / f"{case_id}__{method_name}.png",
                f"{case_id} | {method_name} vs raw vs GT | patterns={','.join(corruption['pattern_names'])}",
                rec.frame_index,
                {
                    "GT": gt_series,
                    "raw": raw_series,
                    method_name: _bbox_components(method_bbox),
                },
                raw_missing_mask=missing_mask,
            )
        _plot_bbox_timeseries(
            output_dir / "plots" / "method_all" / f"{case_id}.png",
            f"{case_id} | all methods vs raw vs GT | patterns={','.join(corruption['pattern_names'])}",
            rec.frame_index,
            {
                "GT": gt_series,
                "raw": raw_series,
                "pipe": _bbox_components(method_a["bbox"]),
                "r_spline": _bbox_components(method_b["bbox"]),
                "ms_poi": _bbox_components(method_ms_po["bbox"]),
                "ms_tab_p": _bbox_components(method_c["poisson"]["bbox"]),
                "ms_tab_b": _bbox_components(method_c["bspline"]["bbox"]),
                "ra_tab_p": _bbox_components(method_d["poisson"]["bbox"]),
                "ra_tab_b": _bbox_components(method_d["bspline"]["bbox"]),
                "ir_tab_p": _bbox_components(method_i["poisson"]["bbox"]),
                "ir_tab_b": _bbox_components(method_i["bspline"]["bbox"]),
                "lw_tab_p": _bbox_components(method_lw["poisson"]["bbox"]),
                "lw_tab_b": _bbox_components(method_lw["bspline"]["bbox"]),
                "lm_tab_p": _bbox_components(method_lm["poisson"]["bbox"]),
                "lm_tab_b": _bbox_components(method_lm["bspline"]["bbox"]),
            },
            raw_missing_mask=missing_mask,
        )
        _plot_detection_case(
            output_dir / "plots" / "detection" / f"{case_id}.png",
            f"{case_id} | detector anomaly pickup",
            rec.frame_index,
            rec.gt_bbox_dense,
            raw_bbox,
            corrupt_mask,
            detector_flags,
        )
        for det_name, det_flag in detector_flags.items():
            _plot_detection_case(
                output_dir / "plots" / "detection_single" / f"{case_id}__{det_name}.png",
                f"{case_id} | {det_name} anomaly pickup",
                rec.frame_index,
                rec.gt_bbox_dense,
                raw_bbox,
                corrupt_mask,
                {det_name: det_flag},
            )
        _plot_reconstruction_case(
            output_dir / "plots" / "reconstruction" / f"{case_id}.png",
            f"{case_id} | Poisson vs B-spline glue-back",
            rec.frame_index,
            rec.gt_bbox_dense,
            method_ms_po["bbox"],
            method_c["poisson"]["bbox"],
            method_c["bspline"]["bbox"],
            method_d["poisson"]["bbox"],
            method_d["bspline"]["bbox"],
            method_i["poisson"]["bbox"],
            method_i["bspline"]["bbox"],
            method_lw["poisson"]["bbox"],
            method_lw["bspline"]["bbox"],
            method_lm["poisson"]["bbox"],
            method_lm["bspline"]["bbox"],
        )

        case_debug.append(
            {
                "case_id": case_id,
                "trajectory_id": rec.trajectory_id,
                "patterns": corruption["pattern_names"],
                "frame_index": rec.frame_index,
                "corrupt_mask": corrupt_mask,
                "missing_mask": missing_mask,
                "raw_bbox": raw_bbox,
                "gt_bbox": rec.gt_bbox_dense,
                "pipe_flag_mask": method_a["flag_mask"],
                "r_spline_flag_mask": method_b["flag_mask"],
                "ms_poi_flag_mask": method_ms_po["flag_mask"],
                "ms_tab_flag_mask": method_c["poisson"]["flag_mask"],
                "ra_tab_flag_mask": method_d["poisson"]["flag_mask"],
                "ir_tab_flag_mask": method_i["poisson"]["flag_mask"],
                "lw_tab_flag_mask": method_lw["poisson"]["flag_mask"],
                "lm_tab_flag_mask": method_lm["poisson"]["flag_mask"],
                "irls_flag_mask": irls_det["flag_mask"],
                "lowess_flag_mask": lowess_det["flag_mask"],
                "lmeds_flag_mask": lmeds_det["flag_mask"],
                "case_window": rec.case_window,
                "raw_source": raw_source,
            }
        )

    method_order = ["raw", "pipe", "r_spline", "ms_poi", "ms_tab_p", "ms_tab_b", "ra_tab_p", "ra_tab_b", "ir_tab_p", "ir_tab_b", "lw_tab_p", "lw_tab_b", "lm_tab_p", "lm_tab_b"]
    summary_rows = _aggregate_summary(case_rows, method_order=method_order)
    pd.DataFrame(case_rows).to_csv(output_dir / "per_case_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_rows).to_csv(output_dir / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    _write_json(output_dir / "case_debug.json", case_debug)

    summary = {
        "config": asdict(cfg),
        "dataset_root": str(dataset_root),
        "n_total_trajectories": len(records),
        "n_train_trajectories": len(train_records),
        "n_test_trajectories": len(test_records),
        "target_test_cases": int(cfg.target_test_cases),
        "n_cases": int(len(case_debug)),
        "test_cases_per_trajectory": test_case_counts,
        "train_trajectory_ids": [r.trajectory_id for r in train_records],
        "test_trajectory_ids": [r.trajectory_id for r in test_records],
        "summary_metrics": summary_rows,
        "raw_source": raw_source,
        "detector": detector_info if raw_source == "detector" else None,
        "notes": {
            "pipe": "existing pipeline baseline: double Hampel + PCHIP + bidirectional Savitzky-Golay; bbox w/h uses existing ROI size strategy",
            "r_spline": "RANSAC B-spline with robust iteration",
            "ms_poi": "MSAC + Poisson with robust iteration",
            "ms_tab_p": "MSAC -> delete outliers -> TabPFN fill -> Poisson glue-back",
            "ms_tab_b": "MSAC -> delete outliers -> TabPFN fill -> B-spline glue-back",
            "ra_tab_p": "RANSAC -> delete outliers -> TabPFN fill -> Poisson glue-back",
            "ra_tab_b": "RANSAC -> delete outliers -> TabPFN fill -> B-spline glue-back",
            "ir_tab_p": "IRLS -> delete outliers -> TabPFN fill -> Poisson glue-back",
            "ir_tab_b": "IRLS -> delete outliers -> TabPFN fill -> B-spline glue-back",
            "lw_tab_p": "BiLOWESS -> delete outliers -> TabPFN fill -> Poisson glue-back",
            "lw_tab_b": "BiLOWESS -> delete outliers -> TabPFN fill -> B-spline glue-back",
            "lm_tab_p": "LMedS -> delete outliers -> TabPFN fill -> Poisson glue-back",
            "lm_tab_b": "LMedS -> delete outliers -> TabPFN fill -> B-spline glue-back",
        },
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def run_detection_benchmark(cfg: BenchmarkConfig) -> dict[str, Any]:
    dataset_root = Path(cfg.dataset_root)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_trajectories(
        dataset_root,
        smooth_window=cfg.mild_sg_window,
        smooth_polyorder=cfg.mild_sg_polyorder,
        max_trajectories=cfg.max_trajectories,
    )
    if len(records) < 2:
        raise RuntimeError(f"Need at least 2 usable trajectories under {dataset_root} to build 8:2 train/test split")

    train_records, test_records = split_trajectories(records, cfg.train_ratio, cfg.seed)
    test_case_counts = allocate_test_cases(len(test_records), cfg.target_test_cases, cfg.cases_per_trajectory)
    detector_specs = [
        ("ransac", "RANSAC"),
        ("msac", "MSAC"),
        ("irls", "IRLS"),
        ("lowess", "BiLOWESS"),
        ("lmeds", "LMedS"),
    ]

    detector_rows: list[dict[str, Any]] = []
    case_debug: list[dict[str, Any]] = []

    for test_i, rec in enumerate(test_records):
        for case_idx in range(test_case_counts[test_i]):
            seed_case = int(cfg.seed + test_i * 1000 + case_idx * 17)
            case_rng = np.random.default_rng(seed_case)
            corruption = _corrupt_bbox(rec.gt_bbox_dense, rec.frame_index, rng=case_rng, max_ratio=cfg.max_corruption_ratio, cfg=cfg)
            raw_bbox = np.asarray(corruption["raw_bbox"], dtype=np.float64)
            corrupt_mask = np.asarray(corruption["corrupt_mask"], dtype=bool)
            missing_mask = np.asarray(corruption["missing_mask"], dtype=bool)
            case_prefix = Path(rec.trajectory_id).with_suffix("").as_posix().replace("/", "__").replace("\\", "__").replace(" ", "_")
            case_id = f"{case_prefix}__case{case_idx:02d}"

            det_outputs: dict[str, dict[str, Any]] = {}
            for det_offset, (det_key, det_label) in enumerate(detector_specs):
                det = _detect_outliers(raw_bbox, rec.frame_index, cfg, detector=det_key, seed=seed_case + 100 + det_offset)
                det_outputs[det_label] = det
                row = {
                    "case_id": case_id,
                    "trajectory_id": rec.trajectory_id,
                    "source_json": rec.source_json,
                    "patterns": ",".join(corruption["pattern_names"]),
                    "seed": seed_case,
                    "n_frames": int(len(rec.frame_index)),
                    "corruption_ratio": float(corrupt_mask.mean()),
                }
                row.update(_evaluate_detection(det_key, det["anomaly_score"], det["flag_mask"], corrupt_mask))
                detector_rows.append(row)

                _plot_detection_case(
                    output_dir / "plots" / "detection_single" / f"{case_id}__{det_label}.png",
                    f"{case_id} | {det_label} anomaly pickup",
                    rec.frame_index,
                    rec.gt_bbox_dense,
                    raw_bbox,
                    corrupt_mask,
                    {det_label: det["flag_mask"]},
                )

            _plot_detection_case(
                output_dir / "plots" / "detection" / f"{case_id}.png",
                f"{case_id} | detector anomaly pickup",
                rec.frame_index,
                rec.gt_bbox_dense,
                raw_bbox,
                corrupt_mask,
                {k: v["flag_mask"] for k, v in det_outputs.items()},
            )

            case_debug.append(
                {
                    "case_id": case_id,
                    "trajectory_id": rec.trajectory_id,
                    "patterns": corruption["pattern_names"],
                    "frame_index": rec.frame_index,
                    "corrupt_mask": corrupt_mask,
                    "missing_mask": missing_mask,
                    "raw_bbox": raw_bbox,
                    "gt_bbox": rec.gt_bbox_dense,
                    "detector_flags": {k: v["flag_mask"] for k, v in det_outputs.items()},
                }
            )

    detector_order = [x[0] for x in detector_specs]
    summary_rows = _aggregate_detection_summary(detector_rows, method_order=detector_order)
    pd.DataFrame(detector_rows).to_csv(output_dir / "detector_per_case_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_rows).to_csv(output_dir / "detector_summary_metrics.csv", index=False, encoding="utf-8-sig")
    _write_json(output_dir / "case_debug.json", case_debug)

    summary = {
        "config": asdict(cfg),
        "dataset_root": str(dataset_root),
        "n_total_trajectories": len(records),
        "n_train_trajectories": len(train_records),
        "n_test_trajectories": len(test_records),
        "target_test_cases": int(cfg.target_test_cases),
        "n_cases": int(len(case_debug)),
        "test_cases_per_trajectory": test_case_counts,
        "train_trajectory_ids": [r.trajectory_id for r in train_records],
        "test_trajectory_ids": [r.trajectory_id for r in test_records],
        "detector_summary_metrics": summary_rows,
        "notes": {
            "ransac": "RANSAC detector on bbox center/size residuals",
            "msac": "MSAC detector on bbox center/size residuals",
            "irls": "IRLS detector on bbox center/size residuals",
            "lowess": "bidirectional LOWESS detector on bbox center/size residuals",
            "lmeds": "LMedS detector on bbox center/size residuals",
        },
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Benchmark trajectory outlier correction / repair methods on bbox trajectories")
    ap.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    ap.add_argument("--train-dataset-root", default="")
    ap.add_argument("--test-dataset-root", default="")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--target-test-cases", type=int, default=50)
    ap.add_argument("--cases-per-trajectory", type=int, default=3)
    ap.add_argument("--max-corruption-ratio", type=float, default=0.30)
    ap.add_argument("--max-trajectories", type=int, default=0)
    ap.add_argument("--raw-source", default="synthetic", choices=["synthetic", "detector"])
    ap.add_argument("--detector-checkpoint", default="")
    ap.add_argument("--detector-experiment-dir", default="")
    ap.add_argument("--detector-cache-root", default="")
    ap.add_argument("--tabpfn-device", default="auto")
    ap.add_argument("--tabpfn-max-train-rows", type=int, default=4096)
    ap.add_argument("--detection-only", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BenchmarkConfig(
        dataset_root=args.dataset_root,
        train_dataset_root=args.train_dataset_root,
        test_dataset_root=args.test_dataset_root,
        output_dir=args.output_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        target_test_cases=args.target_test_cases,
        cases_per_trajectory=args.cases_per_trajectory,
        max_corruption_ratio=args.max_corruption_ratio,
        max_trajectories=args.max_trajectories,
        raw_source=args.raw_source,
        detector_checkpoint=args.detector_checkpoint,
        detector_experiment_dir=args.detector_experiment_dir,
        detector_cache_root=args.detector_cache_root,
        tabpfn_device=args.tabpfn_device,
        tabpfn_max_train_rows=args.tabpfn_max_train_rows,
    )
    summary = run_detection_benchmark(cfg) if args.detection_only else run_benchmark(cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
