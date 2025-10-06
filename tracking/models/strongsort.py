from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception as ex:  # pragma: no cover - ensure graceful failure in minimal environments
    cv2 = None  # type: ignore
    _CV2_IMPORT_ERROR = ex

try:  # Ultralytics YOLO detector (for per-frame detections)
    from ultralytics import YOLO  # type: ignore
except Exception as ex:  # pragma: no cover - optional dependency
    YOLO = None  # type: ignore
    _YOLO_IMPORT_ERROR = ex

try:  # PyTorch is required by Ultralytics
    import torch
except Exception as ex:  # pragma: no cover
    torch = None  # type: ignore
    _TORCH_IMPORT_ERROR = ex

try:  # StrongSORT core components cloned under libs/
    from libs.StrongSORT.deep_sort import nn_matching  # type: ignore
    from libs.StrongSORT.deep_sort.detection import Detection  # type: ignore
    from libs.StrongSORT.deep_sort.tracker import Tracker as StrongSortCore  # type: ignore
except Exception as ex:  # pragma: no cover - make failure explicit at runtime
    nn_matching = None  # type: ignore
    Detection = None  # type: ignore
    StrongSortCore = None  # type: ignore
    _STRONGSORT_IMPORT_ERROR = ex

from ..core.interfaces import FramePrediction, PreprocessingModule, TrackingModel
from ..core.registry import register_model


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


class _ColorHistogramExtractor:
    """
    Lightweight, dependency-free appearance descriptor used to supply StrongSORT
    with per-detection features. Produces an L2-normalised concatenation of
    per-channel histograms, which is sufficient for simple appearance cues
    without pulling in the original FastReID backbone.
    """

    def __init__(self, bins: int = 16):
        if cv2 is None:  # pragma: no cover - handled earlier
            detail = f" (import error: {_CV2_IMPORT_ERROR!r})" if '_CV2_IMPORT_ERROR' in globals() else ""
            raise RuntimeError("OpenCV is required for StrongSORT histogram features" + detail)
        self.bins = max(1, int(bins))
        self.dim = self.bins * 3
        base = np.ones(self.dim, dtype=np.float32)
        self._fallback_vec = base / np.linalg.norm(base)

    def __call__(self, patch: np.ndarray) -> np.ndarray:
        if patch is None or patch.size == 0:
            return self._fallback_vec.copy()
        hist_parts = []
        for channel in range(3):
            h = cv2.calcHist([patch], [channel], None, [self.bins], [0, 256])
            hist_parts.append(h.flatten())
        feat = np.concatenate(hist_parts).astype(np.float32)
        norm = float(np.linalg.norm(feat))
        if norm > 1e-6:
            feat /= norm
        else:
            feat = self._fallback_vec
        return feat


@register_model("StrongSORT")
class StrongSortTracker(TrackingModel):
    """Wraps cloned StrongSORT implementation for single-object tracking.

    We run Ultralytics YOLO per-frame, feed detections (with lightweight colour
    histogram embeddings) into the StrongSORT tracker, then collapse the
    multi-object output down to a single representative trajectory using the
    same selection strategy as the OC-SORT wrapper.
    """

    name = "StrongSORT"

    DEFAULT_CONFIG: Dict[str, Any] = {
        # Detector (Ultralytics YOLO)
        "weights": "yolo11n.pt",
        "conf": 0.25,
        "iou": 0.5,
        "imgsz": 640,
        "device": "cuda",
        "classes": None,
        "max_det": 300,
        # StrongSORT tracker hyper-parameters
        "max_cosine_distance": 0.3,
        "nn_budget": 100,
        "max_iou_distance": 0.7,
        "max_age": 30,
        "n_init": 3,
        # Appearance descriptor
        "feature_bins": 16,
        # Post-processing / selection
        "target_track_id": None,
        "target_strategy": "auto",
        "fallback_last_prediction": True,
        "use_yolo_fallback": True,
        "min_confidence": 0.05,
    }

    def __init__(self, config: Dict[str, Any]):
        if StrongSortCore is None or nn_matching is None or Detection is None:
            detail = f" (import error: {_STRONGSORT_IMPORT_ERROR!r})" if '_STRONGSORT_IMPORT_ERROR' in globals() else ""
            raise RuntimeError("libs/StrongSORT components are required for StrongSORT tracker." + detail)
        if YOLO is None:
            detail = f" (import error: {_YOLO_IMPORT_ERROR!r})" if '_YOLO_IMPORT_ERROR' in globals() else ""
            raise RuntimeError("Ultralytics 'ultralytics' package is required for StrongSORT tracker." + detail)
        if torch is None:
            detail = f" (import error: {_TORCH_IMPORT_ERROR!r})" if '_TORCH_IMPORT_ERROR' in globals() else ""
            raise RuntimeError("PyTorch is required for StrongSORT tracker." + detail)
        if cv2 is None:
            detail = f" (import error: {_CV2_IMPORT_ERROR!r})" if '_CV2_IMPORT_ERROR' in globals() else ""
            raise RuntimeError("OpenCV is required for StrongSORT tracker." + detail)

        # --- Detector configuration ---
        self.weights = str(config.get("weights", self.DEFAULT_CONFIG["weights"]))
        self.det_conf = float(config.get("conf", self.DEFAULT_CONFIG["conf"]))
        self.det_iou = float(config.get("iou", self.DEFAULT_CONFIG["iou"]))
        self.imgsz = int(config.get("imgsz", self.DEFAULT_CONFIG["imgsz"]))
        self.device = str(config.get("device", self.DEFAULT_CONFIG["device"]))
        self.det_classes = config.get("classes", self.DEFAULT_CONFIG["classes"])  # type: ignore
        self.max_det = int(config.get("max_det", self.DEFAULT_CONFIG["max_det"]))
        self.min_confidence = float(config.get("min_confidence", self.DEFAULT_CONFIG["min_confidence"]))

        # --- StrongSORT configuration ---
        self.max_cosine_distance = float(config.get("max_cosine_distance", self.DEFAULT_CONFIG["max_cosine_distance"]))
        self.nn_budget = int(config.get("nn_budget", self.DEFAULT_CONFIG["nn_budget"]))
        self.max_iou_distance = float(config.get("max_iou_distance", self.DEFAULT_CONFIG["max_iou_distance"]))
        self.max_age = int(config.get("max_age", self.DEFAULT_CONFIG["max_age"]))
        self.n_init = int(config.get("n_init", self.DEFAULT_CONFIG["n_init"]))
        self.feature_bins = int(config.get("feature_bins", self.DEFAULT_CONFIG["feature_bins"]))

        # --- Output selection ---
        self.target_track_id = config.get("target_track_id", self.DEFAULT_CONFIG["target_track_id"])  # type: ignore
        self.target_strategy = str(config.get("target_strategy", self.DEFAULT_CONFIG["target_strategy"])).lower()
        self.fallback_last_prediction = bool(config.get("fallback_last_prediction", self.DEFAULT_CONFIG["fallback_last_prediction"]))
        self.use_yolo_fallback = bool(config.get("use_yolo_fallback", self.DEFAULT_CONFIG["use_yolo_fallback"]))

        # Pre-processing chain will be injected by orchestrator if present
        self.preprocs: List[PreprocessingModule] = []

        # Instantiate detector with corruption-safe retry (same approach as YOLO/OC-SORT wrappers)
        try:
            self.detector = YOLO(self.weights)
        except Exception as e:
            try:
                import os
                if os.path.isfile(self.weights) and os.path.getsize(self.weights) < 2048:
                    os.remove(self.weights)
                    self.detector = YOLO(self.weights)
                else:
                    raise
            except Exception:
                raise RuntimeError(f"Failed to load YOLO weights '{self.weights}': {e}")

        try:
            if self.device != "cpu" and not torch.cuda.is_available():
                self._device_str = "cpu"
            else:
                self._device_str = self.device
        except Exception:
            self._device_str = "cpu"

        self._torch = torch
        self._feature_extractor = _ColorHistogramExtractor(self.feature_bins)

    # ------------------------------------------------------------------
    # TrackingModel API
    # ------------------------------------------------------------------
    def train(self, train_dataset, val_dataset=None, seed: int = 0, output_dir: Optional[str] = None):
        # StrongSORT has no trainable parameters in this setup.
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
        self.detector = YOLO(ckpt_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _apply_preprocs(self, frame_bgr: np.ndarray) -> np.ndarray:
        if not self.preprocs:
            return frame_bgr
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        for proc in self.preprocs:
            rgb = proc.apply_to_frame(rgb)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _build_tracker(self) -> Any:
        budget = self.nn_budget if self.nn_budget > 0 else None
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine",
            matching_threshold=self.max_cosine_distance,
            budget=budget,
        )
        return StrongSortCore(
            metric,
            max_iou_distance=self.max_iou_distance,
            max_age=self.max_age,
            n_init=self.n_init,
        )

    def _run_detector(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        try:
            results = self.detector.predict(
                source=frame_bgr,
                conf=self.det_conf,
                iou=self.det_iou,
                imgsz=self.imgsz,
                device=self._device_str,
                classes=self.det_classes,
                verbose=False,
                max_det=self.max_det,
            )
            det_res = results[0] if isinstance(results, list) else next(iter(results))
        except StopIteration:
            det_res = None
        except Exception as e:
            raise RuntimeError(f"YOLO detection failed: {e}")

        if det_res is None:
            return []

        boxes = getattr(det_res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        scores = boxes.conf.detach().cpu().numpy().astype(np.float32)
        cls_vals = None
        try:
            cls_raw = boxes.cls
            if cls_raw is not None:
                cls_vals = cls_raw.detach().cpu().numpy().astype(np.int64)
        except Exception:
            cls_vals = None

        detections: List[Dict[str, Any]] = []
        for idx, (box, score) in enumerate(zip(xyxy, scores)):
            if float(score) < self.min_confidence:
                continue
            x1, y1, x2, y2 = map(float, box.tolist())
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            detections.append(
                {
                    "xyxy": (x1, y1, x2, y2),
                    "tlwh": (x1, y1, w, h),
                    "score": float(score),
                    "cls": int(cls_vals[idx]) if cls_vals is not None and idx < len(cls_vals) else None,
                }
            )
        return detections

    def _extract_features(self, frame_bgr: np.ndarray, tlwh_list: Sequence[Tuple[float, float, float, float]]) -> List[np.ndarray]:
        h_frame, w_frame = frame_bgr.shape[:2]
        feats: List[np.ndarray] = []
        for (x, y, w, h) in tlwh_list:
            x1 = int(max(0, np.floor(x)))
            y1 = int(max(0, np.floor(y)))
            x2 = int(min(w_frame - 1, np.floor(x + w)))
            y2 = int(min(h_frame - 1, np.floor(y + h)))
            if x2 <= x1 or y2 <= y1:
                feats.append(self._feature_extractor(np.zeros((1, 1, 3), dtype=np.uint8)))
                continue
            patch = frame_bgr[y1:y2, x1:x2]
            # Resize to a modest reference size to stabilise histograms on tiny patches
            if patch.size > 0:
                patch = cv2.resize(patch, (64, 128), interpolation=cv2.INTER_LINEAR)
            feats.append(self._feature_extractor(patch))
        return feats

    def _select_candidate(
        self,
        candidates: Sequence[_TrackCandidate],
        last_xyxy: Optional[Tuple[float, float, float, float]],
        current_track_id: Optional[int],
    ) -> Tuple[Optional[_TrackCandidate], Optional[int]]:
        if not candidates:
            return None, current_track_id

        # Prefer user-specified track id
        if self.target_track_id is not None:
            chosen = next((c for c in candidates if c.track_id == int(self.target_track_id)), None)
            if chosen is not None:
                return chosen, chosen.track_id

        # Stick to previously selected id when available
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

        return candidates[0], candidates[0].track_id

    # ------------------------------------------------------------------
    def predict(self, video_path: str) -> List[FramePrediction]:
        if cv2 is None:
            detail = f" (import error: {_CV2_IMPORT_ERROR!r})" if '_CV2_IMPORT_ERROR' in globals() else ""
            raise RuntimeError("OpenCV is required for StrongSORT tracker." + detail)

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
            detections = self._run_detector(frame_proc)
            best_yolo_det: Optional[Dict[str, Any]] = None
            if detections:
                best_yolo_det = max(detections, key=lambda d: d.get("score", 0.0))

            tracker.predict()
            det_objs: List[Any] = []
            if detections:
                features = self._extract_features(frame_proc, [d["tlwh"] for d in detections])
                for det, feat in zip(detections, features):
                    det_objs.append(Detection(det["tlwh"], det["score"], feat))

            tracker.update(det_objs)

            candidates: List[_TrackCandidate] = []
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                tlwh = track.to_tlwh()
                x, y, w, h = map(float, tlwh.tolist())
                xyxy = (x, y, x + w, y + h)
                score_list = getattr(track, "scores", None)
                score = float(score_list[-1]) if score_list else 1.0
                candidates.append(_TrackCandidate(track.track_id, xyxy, score, None))

            candidate, current_track_id = self._select_candidate(candidates, last_bbox_xyxy, current_track_id)

            if candidate is not None:
                x1, y1, x2, y2 = candidate.bbox_xyxy
                w = max(1.0, x2 - x1)
                h = max(1.0, y2 - y1)
                bbox = (float(x1), float(y1), float(w), float(h))
                preds.append(FramePrediction(frame_idx, bbox, candidate.score))
                last_bbox_xyxy = (float(x1), float(y1), float(x2), float(y2))
            elif self.use_yolo_fallback and best_yolo_det is not None:
                bx1, by1, bx2, by2 = best_yolo_det["xyxy"]
                bw = max(1.0, bx2 - bx1)
                bh = max(1.0, by2 - by1)
                bbox = (float(bx1), float(by1), float(bw), float(bh))
                preds.append(
                    FramePrediction(frame_idx, bbox, float(best_yolo_det.get("score", 1.0)))
                )
                last_bbox_xyxy = (float(bx1), float(by1), float(bx2), float(by2))
            elif self.fallback_last_prediction and last_bbox_xyxy is not None:
                x1, y1, x2, y2 = last_bbox_xyxy
                bbox = (float(x1), float(y1), float(max(1.0, x2 - x1)), float(max(1.0, y2 - y1)))
                preds.append(FramePrediction(frame_idx, bbox, None))

            frame_idx += 1

        cap.release()
        return preds
