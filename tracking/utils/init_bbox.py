from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .annotations import load_coco_vid

try:  # pragma: no cover - OpenCV optional at import time
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

_YOLO_MODEL_CACHE: Dict[str, Any] = {}


def _resolve_weights_path(weights: str) -> str:
    if not weights:
        return weights
    candidates = []
    candidates.append(weights)
    candidates.append(os.path.abspath(weights))
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidates.append(os.path.join(repo_root, weights))
    # Deduplicate while preserving order
    seen = set()
    ordered = []
    for path in candidates:
        if path and path not in seen:
            ordered.append(path)
            seen.add(path)
    for path in ordered:
        if os.path.exists(path):
            return path
    return weights


def resolve_weights_path(weights: str) -> str:
    """Public helper that resolves model weight paths relative to the repository root.

    Returns the original string when the file does not exist, so third-party
    model aliases (e.g. ultralytics hub identifiers) continue to function.
    """

    return _resolve_weights_path(weights)


def _first_frame_from_gt(video_path: str) -> Optional[Tuple[float, float, float, float]]:
    json_path = os.path.splitext(video_path)[0] + ".json"
    if not os.path.exists(json_path):
        return None
    try:
        gt = load_coco_vid(json_path)
        frames = gt.get("frames", {})
        if not frames:
            return None
        first_idx = sorted(int(k) for k in frames.keys() if frames.get(k))
        if not first_idx:
            return None
        bboxes = frames.get(first_idx[0]) or []
        if not bboxes:
            return None
        x, y, w, h = bboxes[0]
        return float(x), float(y), float(w), float(h)
    except Exception:
        return None


def _get_yolo_model(weights: str):
    resolved = _resolve_weights_path(weights)
    cache_key = os.path.normpath(resolved)
    model = _YOLO_MODEL_CACHE.get(cache_key)
    if model is not None:
        return model
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:  # pragma: no cover - informative error path
        raise RuntimeError(
            "Ultralytics 'ultralytics' package is required for YOLO-based first-frame detection."
        ) from exc
    if resolved and not os.path.exists(resolved):
        raise FileNotFoundError(f"YOLO weights file not found: {resolved}")
    try:
        model = YOLO(resolved)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to load YOLO weights '{resolved}': {exc}") from exc
    _YOLO_MODEL_CACHE[cache_key] = model
    return model


def _infer_best_bbox_from_frame(
    frame: Any,
    detector: Dict[str, Any],
) -> Tuple[Optional[Tuple[float, float, float, float]], Optional[float]]:
    if frame is None:
        return None, None
    weights = str(detector.get("weights") or "best.pt")
    model = _get_yolo_model(weights)

    device = str(detector.get("device", "auto") or "auto").lower()
    if device == "auto":
        try:
            import torch  # type: ignore

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # pragma: no cover - fallback path
            device = "cpu"

    conf = float(detector.get("conf", 0.25) or 0.25)
    iou = float(detector.get("iou", 0.5) or 0.5)
    imgsz = int(detector.get("imgsz", 640) or 640)
    max_det = int(detector.get("max_det", 50) or 50)
    classes = detector.get("classes")

    try:
        results = model.predict(
            source=[frame],
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            classes=classes,
            max_det=max_det,
            verbose=False,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"YOLO inference failed: {exc}") from exc

    if not results:
        return None, None
    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None, None

    try:
        confs = boxes.conf.cpu().numpy().astype(float)
        xyxy = boxes.xyxy.cpu().numpy().astype(float)
    except Exception:
        return None, None
    if confs.size == 0 or xyxy.size == 0:
        return None, None
    best_idx = int(np.argmax(confs))
    x1, y1, x2, y2 = xyxy[best_idx]
    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))

    height, width = frame.shape[:2]
    x = float(max(0.0, min(width - 1.0, x1)))
    y = float(max(0.0, min(height - 1.0, y1)))
    w = float(min(w, max(1.0, width - x)))
    h = float(min(h, max(1.0, height - y)))
    return (x, y, w, h), float(confs[best_idx])


def _first_frame_from_yolo(
    video_path: str,
    detector: Dict[str, Any],
) -> Optional[Tuple[float, float, float, float]]:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for YOLO-based initialization.")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        return None

    bbox, _ = _infer_best_bbox_from_frame(frame, detector)
    return bbox


def detect_bbox_on_frame(
    frame_bgr: Any,
    detector: Optional[Dict[str, Any]] = None,
    min_confidence: Optional[float] = None,
) -> Tuple[Optional[Tuple[float, float, float, float]], Optional[float]]:
    """Run YOLO detection on a single BGR frame and return the best bbox and confidence."""

    if frame_bgr is None:
        return None, None
    detector_cfg = detector or {}
    bbox, conf = _infer_best_bbox_from_frame(frame_bgr, detector_cfg)
    if conf is None:
        return None, None
    if min_confidence is not None and conf < float(min_confidence):
        return None, None
    return bbox, conf


def resolve_first_frame_bbox(
    video_path: str,
    mode: str = "gt",
    detector: Optional[Dict[str, Any]] = None,
    fallback: Optional[str] = None,
) -> Optional[Tuple[float, float, float, float]]:
    """Resolve the first-frame bounding box according to the requested source.

    Parameters
    ----------
    video_path: str
        Path to the input video.
    mode: str
        "gt" to load annotations, "yolov11" (or "yolo") to run detection,
        "auto" to try GT first then YOLO.
    detector: Dict[str, Any]
        Additional parameters for YOLO inference (weights, conf, device, ...).
    fallback: Optional[str]
        Alternate mode to try when the primary mode yields no bbox.
    """

    detector = detector or {}
    mode_norm = (mode or "gt").strip().lower()

    if mode_norm == "auto":
        bbox = _first_frame_from_gt(video_path)
        if bbox is not None:
            return bbox
        bbox = _first_frame_from_yolo(video_path, detector)
        return bbox

    if mode_norm in ("gt", "groundtruth", "ground_truth"):
        bbox = _first_frame_from_gt(video_path)
    elif mode_norm in ("yolo", "yolov11", "yolo_v11", "detector"):
        bbox = _first_frame_from_yolo(video_path, detector)
    else:
        raise ValueError(f"Unsupported first-frame source: {mode}")

    if bbox is not None or not fallback:
        return bbox

    fallback_norm = fallback.strip().lower()
    if fallback_norm == mode_norm:
        return bbox

    return resolve_first_frame_bbox(
        video_path,
        mode=fallback_norm,
        detector=detector,
        fallback=None,
    )