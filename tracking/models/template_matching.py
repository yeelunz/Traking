from __future__ import annotations
import os
from typing import Dict, Any, List, Tuple, Optional

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore
import numpy as np

from ..core.interfaces import TrackingModel, FramePrediction, PreprocessingModule
from ..core.registry import register_model
from ..utils.init_bbox import resolve_first_frame_bbox


@register_model("TemplateMatching")
class TemplateMatching(TrackingModel):
    name = "TemplateMatching"
    DEFAULT_CONFIG = {
        "method": "TM_CCOEFF_NORMED",
        "template_size": [64, 64],
        "search_margin": 32,
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
            raise RuntimeError("OpenCV (opencv-python) is required for TemplateMatching model.")
        self.method = getattr(cv2, config.get("method", "TM_CCOEFF_NORMED"))
        self.template_size = tuple(config.get("template_size", [64, 64]))
        self.search_margin = int(config.get("search_margin", 32))
        self.template = None  # initialized per video from first frame bbox
        self.preprocs: List[PreprocessingModule] = []  # optional, injected by runner

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

    def train(self, train_dataset, val_dataset=None, seed: int = 0, output_dir: str | None = None) -> Dict[str, Any]:
        # Template matching has no training. Return stub.
        cb = getattr(self, 'progress_callback', None)
        try:
            if callable(cb):
                cb('train_epoch_start', 1, 1)
                cb('train_epoch_end', 1, 1)
        except Exception:
            pass
        return {"status": "no_training"}

    def load_checkpoint(self, ckpt_path: str):
        # Not applicable for template matching
        pass

    def _apply_preprocs(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Apply preproc chain on an RGB frame then return BGR for downstream."""
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
        first_frame = True
        templ = None
        tlx, tly, tw, th = 0, 0, 0, 0
        for i in range(total):
            ok, frame = cap.read()
            if not ok:
                break
            # apply preproc chain if any
            frame = self._apply_preprocs(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if first_frame:
                # Resolve first-frame bbox for initialization
                h, w = frame.shape
                gtbb = self._resolve_first_bbox(video_path)
                if gtbb is None:
                    cap.release()
                    raise RuntimeError(
                        f"Failed to obtain first-frame bbox for video: {os.path.basename(video_path)} "
                        f"(mode={self.first_frame_source})."
                    )
                gx, gy, gw, gh = gtbb
                tlx = int(max(0, min(w - 1, gx)))
                tly = int(max(0, min(h - 1, gy)))
                tw = int(max(1, min(w - tlx, gw)))
                th = int(max(1, min(h - tly, gh)))
                templ = frame[tly:tly + th, tlx:tlx + tw].copy()
                first_frame = False
                preds.append(FramePrediction(i, (float(tlx), float(tly), float(tw), float(th)), 1.0))
                continue
            # define search region
            x0 = max(0, tlx - self.search_margin)
            y0 = max(0, tly - self.search_margin)
            x1 = min(frame.shape[1], tlx + tw + self.search_margin)
            y1 = min(frame.shape[0], tly + th + self.search_margin)
            roi = frame[y0:y1, x0:x1]
            res = cv2.matchTemplate(roi, templ, self.method)
            _, _, _, max_loc = cv2.minMaxLoc(res)
            dx, dy = max_loc
            tlx = x0 + dx
            tly = y0 + dy
            preds.append(FramePrediction(i, (float(tlx), float(tly), float(tw), float(th)), 1.0))
        cap.release()
        return preds
