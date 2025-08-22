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
from ..utils.annotations import load_coco_vid


@register_model("TemplateMatching")
class TemplateMatching(TrackingModel):
    name = "TemplateMatching"
    DEFAULT_CONFIG = {"method": "TM_CCOEFF_NORMED", "template_size": [64, 64], "search_margin": 32}

    def __init__(self, config: Dict[str, Any]):
        if cv2 is None:
            raise RuntimeError("OpenCV (opencv-python) is required for TemplateMatching model.")
        self.method = getattr(cv2, config.get("method", "TM_CCOEFF_NORMED"))
        self.template_size = tuple(config.get("template_size", [64, 64]))
        self.search_margin = int(config.get("search_margin", 32))
        self.template = None  # initialized per video from first frame & GT bbox
        self.preprocs: List[PreprocessingModule] = []  # optional, injected by runner

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

    def _first_gt_bbox(self, video_path: str) -> Optional[Tuple[float, float, float, float]]:
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
                # Require GT first-frame bbox for initialization
                h, w = frame.shape
                gtbb = self._first_gt_bbox(video_path)
                if gtbb is None:
                    cap.release()
                    raise RuntimeError(f"Missing GT first-frame bbox for video: {os.path.basename(video_path)}. Provide GT JSON.")
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
