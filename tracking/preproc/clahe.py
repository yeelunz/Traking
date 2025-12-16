from __future__ import annotations
try:
    import cv2  # type: ignore
except Exception:  # allow import of package without OpenCV installed
    cv2 = None  # type: ignore
import numpy as np
from typing import Dict

from ..core.interfaces import PreprocessingModule
from ..core.registry import register_preproc


@register_preproc("CLAHE")
class CLAHE(PreprocessingModule):
    name = "CLAHE"
    DEFAULT_CONFIG = {"clipLimit": 2.0, "tileGridSize": [8, 8]}

    def __init__(self, config: Dict):
        if cv2 is None:
            raise RuntimeError("OpenCV (opencv-python) is required for CLAHE preprocessor.")
        # Accept both OpenCV-style keys and schedule-friendly aliases.
        clip = float(config.get("clipLimit", config.get("clip", 2.0)))
        grid = config.get("tileGridSize", config.get("grid", [8, 8]))
        self.clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tuple(grid))

    def apply_to_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3 and frame.shape[2] == 3:
            lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l2 = self.clahe.apply(l)
            lab2 = cv2.merge((l2, a, b))
            return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
        else:
            return self.clahe.apply(frame)

    def apply_to_video(self, video_path: str) -> str:
        # Minimal: return original path for now (on-the-fly per frame is recommended)
        return video_path
