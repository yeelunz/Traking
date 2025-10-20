from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import os

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore
import numpy as np

from ..core.interfaces import TrackingModel, FramePrediction, PreprocessingModule
from ..core.registry import register_model
from ..utils.init_bbox import resolve_first_frame_bbox


@register_model("CSRT")
class CSRTTracker(TrackingModel):
    name = "CSRT"
    DEFAULT_CONFIG = {
        "init_box": [64, 64],  # initial box size (w,h) if no GT provided
        "first_frame_source": "gt",
        "first_frame_fallback": "gt",
        "init_detector_weights": "best.pt",
        "init_detector_conf": 0.25,
        "init_detector_iou": 0.5,
        "init_detector_imgsz": 640,
        "init_detector_device": "auto",
        "init_detector_classes": None,
        "init_detector_max_det": 50,
    }

    def __init__(self, config: Dict[str, Any]):
        if cv2 is None:
            raise RuntimeError("OpenCV is required for CSRT tracker.")
        # CSRT requires contrib tracker; check presence of any known factory variant
        has_any = (
            (hasattr(cv2, "legacy") and (hasattr(cv2.legacy, "TrackerCSRT_create") or hasattr(cv2.legacy, "TrackerCSRT")))
            or hasattr(cv2, "TrackerCSRT_create")
            or hasattr(cv2, "TrackerCSRT")
        )
        if not has_any:
            ver = getattr(cv2, "__version__", "?")
            raise RuntimeError(f"CSRT requires opencv-contrib-python (CSRT factory not found). cv2={ver}")
        self.init_w, self.init_h = tuple(config.get("init_box", [64, 64]))
        self.preprocs: List[PreprocessingModule] = []

        self.first_frame_source = str(config.get("first_frame_source", self.DEFAULT_CONFIG["first_frame_source"]) or "gt").lower()
        raw_fallback = config.get("first_frame_fallback", self.DEFAULT_CONFIG["first_frame_fallback"])
        if raw_fallback is None:
            self.first_frame_fallback: Optional[str] = None
        else:
            fb = str(raw_fallback).strip().lower()
            self.first_frame_fallback = fb if fb not in ("", "none", "null") else None
        self.init_detector_params = {
            "weights": str(config.get("init_detector_weights", self.DEFAULT_CONFIG["init_detector_weights"])),
            "conf": float(config.get("init_detector_conf", self.DEFAULT_CONFIG["init_detector_conf"])),
            "iou": float(config.get("init_detector_iou", self.DEFAULT_CONFIG["init_detector_iou"])),
            "imgsz": int(config.get("init_detector_imgsz", self.DEFAULT_CONFIG["init_detector_imgsz"])),
            "device": str(config.get("init_detector_device", self.DEFAULT_CONFIG["init_detector_device"])),
            "classes": config.get("init_detector_classes", self.DEFAULT_CONFIG["init_detector_classes"]),
            "max_det": int(config.get("init_detector_max_det", self.DEFAULT_CONFIG["init_detector_max_det"])),
        }

    def train(self, train_dataset, val_dataset=None, seed: int = 0, output_dir: str | None = None):
        cb = getattr(self, 'progress_callback', None)
        try:
            if callable(cb):
                cb('train_epoch_start', 1, 1)
                cb('train_epoch_end', 1, 1)
        except Exception:
            pass
        return {"status": "no_training"}

    def load_checkpoint(self, ckpt_path: str):
        pass

    def _apply_preprocs(self, frame_bgr):
        if not self.preprocs:
            return frame_bgr
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        for p in self.preprocs:
            rgb = p.apply_to_frame(rgb)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _resolve_first_bbox(self, video_path: str) -> Optional[Tuple[float, float, float, float]]:
        fallback = self.first_frame_fallback
        return resolve_first_frame_bbox(
            video_path,
            mode=self.first_frame_source,
            detector=self.init_detector_params,
            fallback=fallback,
        )

    def predict(self, video_path: str) -> List[FramePrediction]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        preds: List[FramePrediction] = []
        # Create CSRT tracker with robust fallbacks across OpenCV variants
        tracker = None
        try_order = []
        if hasattr(cv2, "legacy"):
            if hasattr(cv2.legacy, "TrackerCSRT_create"):
                try_order.append(lambda: cv2.legacy.TrackerCSRT_create())
            if hasattr(cv2.legacy, "TrackerCSRT"):
                try_order.append(lambda: cv2.legacy.TrackerCSRT.create())
        if hasattr(cv2, "TrackerCSRT_create"):
            try_order.append(lambda: cv2.TrackerCSRT_create())
        if hasattr(cv2, "TrackerCSRT"):
            try_order.append(lambda: cv2.TrackerCSRT.create())
        last_err = None
        for fn in try_order:
            try:
                tracker = fn()
                if tracker is not None:
                    break
            except Exception as e:
                last_err = e
        if tracker is None:
            ver = getattr(cv2, "__version__", "?")
            raise RuntimeError(f"CSRT tracker API not found (cv2={ver}). Last error: {last_err}")
        initbb = None
        for i in range(total):
            ok, frame = cap.read()
            if not ok:
                break
            frame = self._apply_preprocs(frame)
            if i == 0:
                h, w = frame.shape[:2]
                gtbb = self._resolve_first_bbox(video_path)
                if gtbb is None:
                    cap.release()
                    raise RuntimeError(
                        f"Failed to obtain first-frame bbox for video: {os.path.basename(video_path)} "
                        f"(mode={self.first_frame_source})."
                    )
                gx, gy, gw, gh = gtbb
                x = max(0.0, min(float(w - 1), float(gx)))
                y = max(0.0, min(float(h - 1), float(gy)))
                initbb = (x, y, float(min(gw, w - x)), float(min(gh, h - y)))
                tracker.init(frame, initbb)
                preds.append(FramePrediction(i, initbb, 1.0))
                continue
            ok, box = tracker.update(frame)
            if not ok:
                # fallback to last box
                box = initbb
            else:
                initbb = box
            x, y, w, h = box
            preds.append(FramePrediction(i, (float(x), float(y), float(w), float(h)), 1.0))
        cap.release()
        return preds
