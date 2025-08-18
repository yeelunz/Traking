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
from ..utils.annotations import load_coco_vid


@register_model("CSRT")
class CSRTTracker(TrackingModel):
    name = "CSRT"
    DEFAULT_CONFIG = {"init_box": [64, 64]}  # initial box size (w,h) if no GT provided

    def __init__(self, config: Dict[str, Any]):
        if cv2 is None:
            raise RuntimeError("OpenCV is required for CSRT tracker.")
        # CSRT requires opencv-contrib-python
        if not hasattr(cv2, "legacy") or not hasattr(cv2.legacy, "TrackerCSRT_create"):
            raise RuntimeError("CSRT requires opencv-contrib-python (cv2.legacy.TrackerCSRT_create not found).")
        self.init_w, self.init_h = tuple(config.get("init_box", [64, 64]))
        self.preprocs: List[PreprocessingModule] = []

    def train(self, train_dataset, val_dataset=None, seed: int = 0, output_dir: str | None = None):
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

    def _first_gt_bbox(self, video_path: str) -> Optional[Tuple[float, float, float, float]]:
        import os
        j = os.path.splitext(video_path)[0] + ".json"
        if not os.path.exists(j):
            return None
        try:
            gt = load_coco_vid(j)
            frames = gt.get("frames", {})
            if not frames:
                return None
            first_idx = sorted(int(k) for k in frames.keys() if frames.get(k))[0]
            bboxes = frames.get(first_idx) or []
            if not bboxes:
                return None
            x, y, w, h = bboxes[0]
            return float(x), float(y), float(w), float(h)
        except Exception:
            return None

    def predict(self, video_path: str) -> List[FramePrediction]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        preds: List[FramePrediction] = []
        tracker = cv2.legacy.TrackerCSRT_create()
        initbb = None
        for i in range(total):
            ok, frame = cap.read()
            if not ok:
                break
            frame = self._apply_preprocs(frame)
            if i == 0:
                h, w = frame.shape[:2]
                gtbb = self._first_gt_bbox(video_path)
                if gtbb is None:
                    cap.release()
                    raise RuntimeError(f"Missing GT first-frame bbox for video: {os.path.basename(video_path)}. Provide GT JSON.")
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
