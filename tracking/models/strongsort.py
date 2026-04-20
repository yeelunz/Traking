from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import math

try:
    import cv2  # type: ignore
except Exception as ex:  # pragma: no cover - ensure graceful failure in minimal environments
    raise ImportError(
        "Failed to import OpenCV for tracking.models.strongsort. Install opencv-python."
    ) from ex

try:  # Ultralytics YOLO detector (for per-frame detections)
    from ultralytics import YOLO  # type: ignore
except Exception as ex:  # pragma: no cover - optional dependency
    raise ImportError(
        "Failed to import ultralytics for tracking.models.strongsort. Install ultralytics."
    ) from ex

try:  # PyTorch is required by Ultralytics
    import torch
except Exception as ex:  # pragma: no cover
    raise ImportError(
        "Failed to import torch for tracking.models.strongsort. Install torch."
    ) from ex

try:  # StrongSORT core components cloned under libs/
    from libs.StrongSORT.deep_sort import nn_matching  # type: ignore
    from libs.StrongSORT.deep_sort.detection import Detection  # type: ignore
    from libs.StrongSORT.deep_sort.tracker import Tracker as StrongSortCore  # type: ignore
except Exception as ex:  # pragma: no cover - make failure explicit at runtime
    raise ImportError(
        "Failed to import StrongSORT dependencies for tracking.models.strongsort. "
        "Ensure libs/StrongSORT is available and importable."
    ) from ex

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
        self._weights_path = resolve_weights_path(self.weights)
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
        if frame_bgr is not None and frame_bgr.ndim == 3:
            _g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            frame_bgr = cv2.cvtColor(_g, cv2.COLOR_GRAY2BGR)
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


@register_model("StrongSORT++")
class StrongSortPPTracker(StrongSortTracker):
    """Enhanced StrongSORT variant with temporal interpolation and smoothing."""

    name = "StrongSORT++"

    DEFAULT_CONFIG: Dict[str, Any] = {
        **StrongSortTracker.DEFAULT_CONFIG,
        "enable_gsi": True,
        "gsi_interval": 20,
        "enable_gaussian_smoothing": True,
        "smoothing_sigma": 1.25,
        "score_fill_value": 0.6,
        "clip_scores_between": [0.0, 1.0],
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        merged = {**self.DEFAULT_CONFIG, **(config or {})}
        self._enable_interp = bool(merged.get("enable_gsi", True))
        self._interp_max_gap = max(2, int(merged.get("gsi_interval", 20)))
        self._enable_smoothing = bool(merged.get("enable_gaussian_smoothing", True))
        self._smoothing_sigma = float(merged.get("smoothing_sigma", 1.25))
        self._score_fill = float(merged.get("score_fill_value", 0.6))
        clip = merged.get("clip_scores_between", None)
        if isinstance(clip, (list, tuple)) and len(clip) == 2:
            self._score_clip: Optional[Tuple[float, float]] = (float(clip[0]), float(clip[1]))
        else:
            self._score_clip = None

    def predict(self, video_path: str) -> List[FramePrediction]:
        preds = super().predict(video_path)
        if not preds:
            return preds
        processed = self._deduplicate(preds)
        if self._enable_interp:
            processed = self._interpolate_gaps(processed, self._interp_max_gap)
        if self._enable_smoothing:
            processed = self._smooth_sequence(processed, self._smoothing_sigma)
        return processed

    @staticmethod
    def _score_value(score: Optional[float]) -> float:
        return float(score) if score is not None else float("-inf")

    def _score_float(self, score: Optional[float]) -> float:
        if score is None:
            return self._score_fill
        return float(score)

    def _clip_score(self, value: float) -> float:
        if self._score_clip is None:
            return float(value)
        lo, hi = self._score_clip
        return float(max(lo, min(hi, value)))

    def _deduplicate(self, preds: Sequence[FramePrediction]) -> List[FramePrediction]:
        frame_map: Dict[int, FramePrediction] = {}
        for p in preds:
            idx = int(getattr(p, "frame_index", 0))
            bbox = tuple(float(x) for x in p.bbox)
            score = p.score
            candidate = FramePrediction(idx, bbox, score)
            stored = frame_map.get(idx)
            if stored is None or self._score_value(candidate.score) > self._score_value(stored.score):
                frame_map[idx] = candidate
        return [frame_map[i] for i in sorted(frame_map.keys())]

    def _interpolate_gaps(self, preds: Sequence[FramePrediction], max_gap: int) -> List[FramePrediction]:
        """Cubic-spline gap interpolation (GSI) for detection gaps.

        Uses cubic spline interpolation to fill gaps in the detection
        sequence, ensuring C2 continuity (velocity and acceleration) at
        interpolated points.  Falls back to linear interpolation when the
        number of known keypoints is < 4.
        """
        if len(preds) < 2 or max_gap < 2:
            return list(preds)
        preds_sorted = sorted(preds, key=lambda p: int(p.frame_index))

        # Collect all known keypoints for global spline fitting
        known_frames = np.array([int(p.frame_index) for p in preds_sorted], dtype=np.float64)
        known_boxes = np.array([list(p.bbox) for p in preds_sorted], dtype=np.float64)
        known_scores = np.array(
            [self._score_float(p.score) for p in preds_sorted], dtype=np.float64,
        )
        has_score = any(p.score is not None for p in preds_sorted)

        # Build cubic spline for each bbox dimension + score
        _spline_ok = False
        try:
            from scipy.interpolate import CubicSpline
            if len(known_frames) >= 4:
                box_splines = [
                    CubicSpline(known_frames, known_boxes[:, d], extrapolate=True)
                    for d in range(4)
                ]
                score_spline = CubicSpline(known_frames, known_scores, extrapolate=True) if has_score else None
                _spline_ok = True
        except (ImportError, Exception):
            _spline_ok = False

        output: List[FramePrediction] = []
        for cur, nxt in zip(preds_sorted[:-1], preds_sorted[1:]):
            output.append(cur)
            gap = int(nxt.frame_index) - int(cur.frame_index)
            if 1 < gap <= max_gap:
                for step in range(1, gap):
                    frame_idx = int(cur.frame_index) + step
                    t = np.float64(frame_idx)
                    if _spline_ok:
                        bbox = tuple(float(box_splines[d](t)) for d in range(4))
                        score = None
                        if has_score and score_spline is not None:
                            score = self._clip_score(float(score_spline(t)))
                    else:
                        # Fallback: linear interpolation between two endpoints
                        alpha = float(step) / float(gap)
                        cur_box = np.asarray(cur.bbox, dtype=np.float64)
                        nxt_box = np.asarray(nxt.bbox, dtype=np.float64)
                        interp_box = cur_box + (nxt_box - cur_box) * alpha
                        bbox = tuple(float(v) for v in interp_box)
                        score = None
                        if has_score:
                            cur_s = self._score_float(cur.score)
                            nxt_s = self._score_float(nxt.score)
                            score = self._clip_score(cur_s + (nxt_s - cur_s) * alpha)
                    output.append(
                        FramePrediction(frame_idx, bbox, score)
                    )
        output.append(preds_sorted[-1])
        return self._deduplicate(output)

    def _smooth_sequence(self, preds: Sequence[FramePrediction], sigma: float) -> List[FramePrediction]:
        if sigma <= 0 or len(preds) < 3:
            return list(preds)
        arr = np.asarray([p.bbox for p in preds], dtype=np.float32)
        kernel = self._gaussian_kernel(sigma)
        smoothed_boxes = self._apply_kernel(arr, kernel)

        scores_available = any(p.score is not None for p in preds)
        score_mask: List[bool] = [p.score is None for p in preds]
        smoothed_scores: Optional[np.ndarray] = None
        if scores_available:
            score_vals = np.asarray([self._score_float(p.score) for p in preds], dtype=np.float32)
            smoothed_scores = self._apply_kernel(score_vals.reshape(-1, 1), kernel).reshape(-1)

        smoothed: List[FramePrediction] = []
        for idx, p in enumerate(preds):
            bbox = tuple(float(x) for x in smoothed_boxes[idx])
            score: Optional[float]
            if smoothed_scores is not None and not score_mask[idx]:
                score = self._clip_score(float(smoothed_scores[idx]))
            else:
                score = p.score
            smoothed.append(FramePrediction(int(p.frame_index), bbox, score))
        return smoothed

    @staticmethod
    def _gaussian_kernel(sigma: float) -> np.ndarray:
        sigma = max(1e-3, float(sigma))
        radius = max(1, int(math.ceil(3.0 * sigma)))
        offsets = np.arange(-radius, radius + 1, dtype=np.float32)
        kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
        kernel_sum = float(np.sum(kernel))
        if kernel_sum <= 0:
            return np.array([1.0], dtype=np.float32)
        return kernel / kernel_sum

    @staticmethod
    def _apply_kernel(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        radius = (len(kernel) - 1) // 2
        padded = np.pad(arr, ((radius, radius), (0, 0)), mode="edge")
        filtered = np.empty_like(arr, dtype=np.float32)
        for col in range(arr.shape[1]):
            filtered[:, col] = np.convolve(padded[:, col], kernel, mode="valid")
        return filtered if filtered.shape == arr.shape else filtered[:arr.shape[0]]
