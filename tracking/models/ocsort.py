from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:  # Ultralytics YOLO for detections
    from ultralytics import YOLO  # type: ignore
except Exception as e:  # pragma: no cover - optional dependency at runtime
    _YOLO_IMPORT_ERROR = e
    YOLO = None  # type: ignore

try:  # OC-SORT tracker core
    from ocsort import OCSort  # type: ignore
except Exception as e:  # pragma: no cover - optional dependency at runtime
    _OCSORT_IMPORT_ERROR = e
    OCSort = None  # type: ignore

try:  # PyTorch tensors required by OC-SORT
    import torch
except Exception as e:  # pragma: no cover
    torch = None  # type: ignore
    _TORCH_IMPORT_ERROR = e

from ..core.interfaces import FramePrediction, PreprocessingModule, TrackingModel
from ..core.registry import register_model
from ..utils.init_bbox import resolve_weights_path


def _bbox_iou_xyxy(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


@dataclass
class _TrackCandidate:
    track_id: int
    bbox_xyxy: Tuple[float, float, float, float]
    score: float
    cls: Optional[int]


@register_model("OCSort")
class OCSortTracker(TrackingModel):
    """Tracker that fuses Ultralytics YOLO detections with OC-SORT multi-object tracking.

    For the single-object benchmarking pipeline this wrapper extracts one representative
    trajectory from the multi-object tracker (by default the track that best matches the
    previous prediction, otherwise the highest-confidence track).
    """

    name = "OCSort"

    DEFAULT_CONFIG: Dict[str, Any] = {
        # Detector (Ultralytics YOLO)
        "weights": "yolo11n.pt",
        "conf": 0.25,
        "iou": 0.5,
        "imgsz": 640,
        "device": "cuda",
        "classes": None,
        "max_det": 300,
        # Tracker (OC-SORT)
        "det_thresh": 0.05,
        "max_age": 30,
        "min_hits": 3,
        "tracker_iou_threshold": 0.3,
        "delta_t": 3,
        "asso_func": "iou",
        "inertia": 0.2,
        "use_byte": False,
        # Post-processing / selection
        "target_track_id": None,
        "target_strategy": "auto",  # auto -> stick to id if available, else IoU, fallback to score
        "fallback_last_prediction": True,
        "min_confidence": 0.0,
    }

    def __init__(self, config: Dict[str, Any]):
        if OCSort is None:
            detail = f" (import error: {_OCSORT_IMPORT_ERROR!r})" if '_OCSORT_IMPORT_ERROR' in globals() else ""
            raise RuntimeError("The 'ocsort' package is required for OCSort tracker." + detail)
        if YOLO is None:
            detail = f" (import error: {_YOLO_IMPORT_ERROR!r})" if '_YOLO_IMPORT_ERROR' in globals() else ""
            raise RuntimeError("Ultralytics 'ultralytics' package is required for OCSort tracker." + detail)
        if torch is None:
            detail = f" (import error: {_TORCH_IMPORT_ERROR!r})" if '_TORCH_IMPORT_ERROR' in globals() else ""
            raise RuntimeError("PyTorch is required for OCSort tracker." + detail)

        # --- Detector configuration ---
        self.weights = str(config.get("weights", self.DEFAULT_CONFIG["weights"]))
        self._weights_path = resolve_weights_path(self.weights)
        self.det_conf = float(config.get("conf", self.DEFAULT_CONFIG["conf"]))
        self.det_iou = float(config.get("iou", self.DEFAULT_CONFIG["iou"]))
        self.imgsz = int(config.get("imgsz", self.DEFAULT_CONFIG["imgsz"]))
        self.device = str(config.get("device", self.DEFAULT_CONFIG["device"]))
        self.det_classes = config.get("classes", self.DEFAULT_CONFIG["classes"])  # type: ignore
        self.max_det = int(config.get("max_det", self.DEFAULT_CONFIG["max_det"]))

        # --- Tracker configuration ---
        self.det_thresh = float(config.get("det_thresh", self.DEFAULT_CONFIG["det_thresh"]))
        self.max_age = int(config.get("max_age", self.DEFAULT_CONFIG["max_age"]))
        self.min_hits = int(config.get("min_hits", self.DEFAULT_CONFIG["min_hits"]))
        self.tracker_iou_threshold = float(config.get("tracker_iou_threshold", self.DEFAULT_CONFIG["tracker_iou_threshold"]))
        self.delta_t = int(config.get("delta_t", self.DEFAULT_CONFIG["delta_t"]))
        self.asso_func = str(config.get("asso_func", self.DEFAULT_CONFIG["asso_func"]))
        self.inertia = float(config.get("inertia", self.DEFAULT_CONFIG["inertia"]))
        self.use_byte = bool(config.get("use_byte", self.DEFAULT_CONFIG["use_byte"]))

        # --- Output selection ---
        self.target_track_id = config.get("target_track_id", self.DEFAULT_CONFIG["target_track_id"])  # type: ignore
        self.target_strategy = str(config.get("target_strategy", self.DEFAULT_CONFIG["target_strategy"])).lower()
        self.fallback_last_prediction = bool(config.get("fallback_last_prediction", self.DEFAULT_CONFIG["fallback_last_prediction"]))
        self.min_confidence = float(config.get("min_confidence", self.DEFAULT_CONFIG["min_confidence"]))

        # Pre-processing chain may be injected by runner
        self.preprocs: List[PreprocessingModule] = []

        # Instantiate detector lazily with a corruption-safe fallback similar to YOLO wrapper
        try:
            self.detector = YOLO(self._weights_path)
        except Exception as e:
            try:
                import os
                if os.path.isfile(self._weights_path) and os.path.getsize(self._weights_path) < 2048:
                    os.remove(self._weights_path)
                    self.detector = YOLO(self._weights_path)
                else:
                    raise
            except Exception:
                raise RuntimeError(f"Failed to load YOLO weights '{self.weights}': {e}")

        try:
            import torch as _torch
            if self.device != "cpu" and not _torch.cuda.is_available():
                self._device_str = "cpu"
            else:
                self._device_str = self.device
        except Exception:
            self._device_str = "cpu"

        self._torch = torch

    # ------------------------------------------------------------------
    # TrackingModel API
    # ------------------------------------------------------------------
    def train(self, train_dataset, val_dataset=None, seed: int = 0, output_dir: Optional[str] = None):
        # OCSort has no trainable parameters; emit a single progress step for UI feedback.
        cb = getattr(self, "progress_callback", None)
        if callable(cb):
            try:
                cb("train_epoch_start", 1, 1)
            except Exception:
                pass
            try:
                cb("train_epoch_end", 1, 1)
            except Exception:
                pass
        return {"status": "no_training"}

    def load_checkpoint(self, ckpt_path: str):
        # Allow swapping detector weights at runtime.
        self.weights = ckpt_path
        self._weights_path = resolve_weights_path(self.weights)
        self.detector = YOLO(self._weights_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _apply_preprocs(self, frame_bgr: np.ndarray) -> np.ndarray:
        import cv2
        if frame_bgr is not None and frame_bgr.ndim == 3:
            _g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            frame_bgr = cv2.cvtColor(_g, cv2.COLOR_GRAY2BGR)
        if not self.preprocs:
            return frame_bgr
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        for proc in self.preprocs:
            rgb = proc.apply_to_frame(rgb)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _build_tracker(self):
        return OCSort(
            det_thresh=self.det_thresh,
            max_age=self.max_age,
            min_hits=self.min_hits,
            iou_threshold=self.tracker_iou_threshold,
            delta_t=self.delta_t,
            asso_func=self.asso_func,
            inertia=self.inertia,
            use_byte=self.use_byte,
        )

    def _tensor_from_boxes(self, result):
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return self._torch.empty((0, 6), dtype=self._torch.float32)
        xyxy = boxes.xyxy.detach().cpu().to(self._torch.float32)
        conf = boxes.conf.detach().cpu().to(self._torch.float32)
        cls = boxes.cls.detach().cpu().to(self._torch.float32)
        if xyxy.ndim == 1:
            xyxy = xyxy.view(1, -1)
        dets = self._torch.stack([xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3], conf, cls], dim=1)
        return dets

    def _select_candidate(
        self,
        tracks: np.ndarray,
        last_xyxy: Optional[Tuple[float, float, float, float]],
        current_track_id: Optional[int],
    ) -> Tuple[Optional[_TrackCandidate], Optional[int]]:
        if tracks is None or len(tracks) == 0:
            return None, current_track_id

        arr = np.asarray(tracks)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        candidates: List[_TrackCandidate] = []
        for row in arr:
            x1, y1, x2, y2 = map(float, row[:4])
            track_id = int(round(row[4])) if arr.shape[1] >= 5 else 0
            cls_val = int(round(row[5])) if arr.shape[1] >= 6 else None
            score = float(row[6]) if arr.shape[1] >= 7 else (float(row[5]) if arr.shape[1] >= 6 else 0.0)
            if score < self.min_confidence:
                continue
            candidates.append(_TrackCandidate(track_id=track_id, bbox_xyxy=(x1, y1, x2, y2), score=score, cls=cls_val))

        if not candidates:
            return None, current_track_id

        # Prefer a user-specified target track id if present
        if self.target_track_id is not None:
            chosen = next((c for c in candidates if c.track_id == int(self.target_track_id)), None)
            if chosen is not None:
                return chosen, chosen.track_id

        # Stick to the previously selected id when available
        if current_track_id is not None:
            chosen = next((c for c in candidates if c.track_id == current_track_id), None)
            if chosen is not None:
                return chosen, chosen.track_id

        # Strategy-based selection
        chosen: Optional[_TrackCandidate] = None
        if self.target_strategy in ("auto", "closest") and last_xyxy is not None:
            best_iou = -1.0
            for cand in candidates:
                iou = _bbox_iou_xyxy(cand.bbox_xyxy, last_xyxy)
                if iou > best_iou:
                    best_iou = iou
                    chosen = cand
            if chosen is not None and best_iou > 0.0:
                return chosen, chosen.track_id

        if self.target_strategy in ("auto", "best_score", "score") or chosen is None:
            chosen = max(candidates, key=lambda c: c.score)
            return chosen, chosen.track_id

        # Fallback to first candidate
        return candidates[0], candidates[0].track_id

    # ------------------------------------------------------------------
    def predict(self, video_path: str) -> List[FramePrediction]:
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        tracker = self._build_tracker()
        preds: List[FramePrediction] = []
        frame_idx = 0
        last_bbox_xyxy: Optional[Tuple[float, float, float, float]] = None
        current_track_id: Optional[int] = self.target_track_id if isinstance(self.target_track_id, int) else None

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_proc = self._apply_preprocs(frame)

            # Run detector per-frame to keep tracker state consistent
            try:
                results = self.detector.predict(
                    source=frame_proc,
                    conf=self.det_conf,
                    iou=self.det_iou,
                    imgsz=self.imgsz,
                    device=self._device_str,
                    classes=self.det_classes,
                    verbose=False,
                    max_det=self.max_det,
                )
                if isinstance(results, list):
                    det_res = results[0]
                else:  # generator/iterator
                    det_res = next(iter(results))
            except StopIteration:
                det_res = None
            except Exception as e:
                cap.release()
                raise RuntimeError(f"YOLO detection failed on frame {frame_idx}: {e}")

            det_tensor = self._tensor_from_boxes(det_res) if det_res is not None else self._torch.empty((0, 6), dtype=self._torch.float32)
            # OC-SORT expects a tensor with class column included
            if det_tensor.numel() == 0:
                det_tensor = self._torch.empty((0, 6), dtype=self._torch.float32)

            tracks = tracker.update(det_tensor, frame_proc)
            candidate, current_track_id = self._select_candidate(tracks, last_bbox_xyxy, current_track_id)

            if candidate is not None:
                x1, y1, x2, y2 = candidate.bbox_xyxy
                w = max(1.0, x2 - x1)
                h = max(1.0, y2 - y1)
                bbox = (float(x1), float(y1), float(w), float(h))
                preds.append(FramePrediction(frame_idx, bbox, candidate.score))
                last_bbox_xyxy = (float(x1), float(y1), float(x2), float(y2))
            elif self.fallback_last_prediction and last_bbox_xyxy is not None:
                x1, y1, x2, y2 = last_bbox_xyxy
                bbox = (float(x1), float(y1), float(max(1.0, x2 - x1)), float(max(1.0, y2 - y1)))
                preds.append(
                    FramePrediction(
                        frame_index=frame_idx,
                        bbox=bbox,
                        score=None,
                        is_fallback=True,
                    )
                )

            frame_idx += 1

        cap.release()
        return preds
