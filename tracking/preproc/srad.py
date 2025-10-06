from __future__ import annotations
"""SRAD (Speckle Reducing Anisotropic Diffusion)
A lightweight implementation suitable for ultrasound / speckle noise reduction.
Reference idea: Yu & Acton 2002 (not an exact reproduction, simplified for speed).

We implement a fixed number of iterations of anisotropic diffusion where the
conduction coefficient is modulated by local coefficient of variation (CV).
This keeps edges (structural boundaries) while smoothing speckle in homogeneous regions.

Config parameters:
- iterations: int (default 10)
- lambda: float (time step, stability ~ (0,0.25]) default 0.15
- eps: small number to avoid division by zero (default 1e-6)
- convert_gray: if True and input is RGB, process luminance then reconstruct (default True)

NOTE: For performance we work in float32 and do minimal Python loops.
"""
from typing import Dict
import numpy as np
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

from ..core.registry import register_preproc
from ..core.interfaces import PreprocessingModule


@register_preproc("SRAD")
class SRAD(PreprocessingModule):
    name = "SRAD"
    DEFAULT_CONFIG = {"iterations": 10, "lambda": 0.15, "eps": 1e-6, "convert_gray": True}

    def __init__(self, config: Dict):
        self.iters = int(config.get("iterations", 10))
        self.dt = float(config.get("lambda", 0.15))
        self.eps = float(config.get("eps", 1e-6))
        self.conv_gray = bool(config.get("convert_gray", True))
        if self.dt <= 0 or self.dt > 0.25:
            raise ValueError("lambda (time step) should be in (0, 0.25] for stability")

    def _ensure_gray(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        if frame.ndim == 3 and frame.shape[2] == 3 and self.conv_gray:
            if cv2 is None:
                raise RuntimeError("OpenCV required for color conversion in SRAD")
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            return gray, frame
        return frame, None

    def _restore_color(self, processed_gray: np.ndarray, original_color: np.ndarray | None) -> np.ndarray:
        if original_color is None:
            return processed_gray
        if cv2 is None:
            return processed_gray
        # simple replace luminance by processed gray using LAB space
        lab = cv2.cvtColor(original_color, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        # scale processed_gray to l range if needed
        if processed_gray.dtype != l.dtype:
            proc = processed_gray.astype(l.dtype)
        else:
            proc = processed_gray
        lab2 = cv2.merge((proc, a, b))
        return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)

    def apply_to_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame.dtype != np.uint8:
            # assume already float-ish; scale to 0-255 for diffusion stability then back
            norm = frame.astype(np.float32)
            mx = norm.max() if norm.size else 1.0
            if mx > 0:
                norm = norm / mx
            norm = (norm * 255.0).clip(0,255).astype(np.uint8)
        else:
            norm = frame
        gray, color_src = self._ensure_gray(norm)
        I = gray.astype(np.float32)
        for _ in range(self.iters):
            # 4-neighborhood differences
            north = np.zeros_like(I); north[1:] = I[:-1]; dN = north - I
            south = np.zeros_like(I); south[:-1] = I[1:]; dS = south - I
            west = np.zeros_like(I);  west[:,1:] = I[:, :-1]; dW = west - I
            east = np.zeros_like(I);  east[:, :-1] = I[:,1:]; dE = east - I
            # gradient magnitude squared
            grad_sq = (dN**2 + dS**2 + dW**2 + dE**2)
            # local mean & variance estimates (simple)
            meanI = (north + south + west + east + 4*I) / 8.0
            varI = ((north - meanI)**2 + (south - meanI)**2 + (west - meanI)**2 + (east - meanI)**2 + 4*(I-meanI)**2) / 8.0
            # coefficient of variation squared
            q_sq = varI / (meanI**2 + self.eps)
            # conduction coefficient (inverse relation)
            c = 1.0 / (1.0 + q_sq)
            # update
            I = I + self.dt * (c * dN + c * dS + c * dW + c * dE)
        I = np.clip(I, 0, 255).astype(np.uint8)
        if color_src is not None:
            return self._restore_color(I, color_src)
        return I

    def apply_to_video(self, video_path: str) -> str:
        return video_path
