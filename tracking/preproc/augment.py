from __future__ import annotations
import numpy as np
try:
    import cv2  # type: ignore
except Exception:  # allow package import without OpenCV installed
    cv2 = None  # type: ignore
from typing import Dict

from ..core.interfaces import PreprocessingModule
from ..core.registry import register_preproc


@register_preproc("AUGMENT")
class Augment(PreprocessingModule):
    name = "AUGMENT"
    DEFAULT_CONFIG = {
        "hflip_prob": 0.5,
        "vflip_prob": 0.0,
        "rotate_max_deg": 8.0,
        "brightness": 0.08,
        "contrast": 0.08,
        "noise_std": 0.0,
        "seed": 1337,
    }

    def __init__(self, config: Dict):
        if cv2 is None:
            raise RuntimeError("OpenCV (opencv-python) is required for AUGMENT preprocessor.")
        self.hflip_prob = float(config.get("hflip_prob", 0.5))
        self.vflip_prob = float(config.get("vflip_prob", 0.0))
        self.rotate_max_deg = float(config.get("rotate_max_deg", 8.0))
        self.brightness = float(config.get("brightness", 0.08))
        self.contrast = float(config.get("contrast", 0.08))
        self.noise_std = float(config.get("noise_std", 0.0))
        seed = config.get("seed", None)
        self.rng = np.random.default_rng(int(seed)) if seed is not None else np.random.default_rng()

    def apply_to_frame(self, frame: np.ndarray) -> np.ndarray:
        img = frame
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        # Flips
        if self.rng.random() < self.hflip_prob:
            img = np.ascontiguousarray(np.fliplr(img))
        if self.rng.random() < self.vflip_prob:
            img = np.ascontiguousarray(np.flipud(img))

        # Rotate small angles to preserve structure
        if self.rotate_max_deg > 0:
            angle = float(self.rng.uniform(-self.rotate_max_deg, self.rotate_max_deg))
            h, w = img.shape[:2]
            m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
            img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # Brightness/contrast jitter
        if any(v > 0 for v in (self.brightness, self.contrast)):
            delta = float(self.rng.uniform(-self.brightness, self.brightness))
            alpha = 1.0 + float(self.rng.uniform(-self.contrast, self.contrast))
            img = img.astype(np.float32) / 255.0
            img = np.clip(alpha * img + delta, 0.0, 1.0)
            img = (img * 255.0).astype(np.uint8)

        # Gaussian noise
        if self.noise_std > 0:
            noise = self.rng.normal(0.0, self.noise_std, size=img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)

        return img

    def apply_to_video(self, video_path: str) -> str:
        # On-the-fly frame augmentation is preferred; no video rewrite.
        return video_path
