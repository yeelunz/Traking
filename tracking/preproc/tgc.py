from __future__ import annotations
"""Time Gain Compensation (TGC) / Depth Attenuation Compensation

In ultrasound, deeper echoes are weaker due to attenuation. We simulate a depth gain
profile that amplifies lower (deeper) rows of the image. For generic grayscale
or RGB frames, we treat vertical axis (y) as 'depth'.

Config parameters:
- mode: 'linear' | 'exp' | 'custom' (default 'linear')
- gain_start: starting multiplier at top (default 1.0)
- gain_end: ending multiplier at bottom (default 2.0)
- exp_k: exponent factor when mode='exp' (default 1.0)
- custom_points: optional list of (y_norm, gain) points for piecewise-linear curve when mode='custom'
- per_channel: apply per-channel if RGB (default True)
- clip: clip output to [0,255] after applying gain (default True)

Implementation details:
We precompute a 1D gain vector of shape (H,) then broadcast over width (W) and channel (C).
"""
from typing import Dict, List, Tuple
import numpy as np

from ..core.registry import register_preproc
from ..core.interfaces import PreprocessingModule

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


@register_preproc("TGC")
class TGC(PreprocessingModule):
    name = "TGC"
    DEFAULT_CONFIG = {
        "mode": "linear",
        "gain_start": 1.0,
        "gain_end": 2.0,
        "exp_k": 1.0,
        "custom_points": None,
        "per_channel": True,
        "clip": True,
    }

    def __init__(self, config: Dict):
        self.mode = str(config.get("mode", "linear")).lower()
        self.g0 = float(config.get("gain_start", 1.0))
        self.g1 = float(config.get("gain_end", 2.0))
        self.exp_k = float(config.get("exp_k", 1.0))
        self.custom = config.get("custom_points")
        self.per_channel = bool(config.get("per_channel", True))
        self.clip = bool(config.get("clip", True))
        if self.mode not in {"linear", "exp", "custom"}:
            raise ValueError("mode must be one of 'linear','exp','custom'")
        if self.mode == 'custom' and not self.custom:
            raise ValueError("custom mode requires 'custom_points' list of (y_norm,gain)")

    def _build_gain(self, H: int) -> np.ndarray:
        y = np.linspace(0.0, 1.0, H, dtype=np.float32)
        if self.mode == 'linear':
            g = self.g0 + (self.g1 - self.g0) * y
        elif self.mode == 'exp':
            # exponential growth from g0 to g1
            # g(y) = g0 * ( (g1/g0) ^ (y^k) )
            ratio = (self.g1 / max(1e-6, self.g0))
            g = self.g0 * np.power(ratio, np.power(y, self.exp_k))
        else:  # custom piecewise linear
            pts: List[Tuple[float,float]] = sorted([(float(a), float(b)) for a,b in self.custom])  # type: ignore
            xs = [p[0] for p in pts]; vs = [p[1] for p in pts]
            g = np.interp(y, xs, vs).astype(np.float32)
        return g

    def apply_to_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            H, W = frame.shape
            g = self._build_gain(H)[:, None]
            out = frame.astype(np.float32) * g
            if self.clip:
                out = np.clip(out, 0, 255)
            return out.astype(frame.dtype)
        if frame.ndim == 3 and frame.shape[2] == 3:
            H, W, C = frame.shape
            g = self._build_gain(H)[:, None, None]
            if self.per_channel:
                out = frame.astype(np.float32) * g
            else:
                # operate on luminance only via LAB
                if cv2 is None:
                    gray = frame.mean(axis=-1, keepdims=True).astype(np.float32)
                    gray = gray * g
                    if self.clip:
                        gray = np.clip(gray, 0, 255)
                    return np.repeat(gray.astype(frame.dtype), 3, axis=-1)
                lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
                l,a,b = cv2.split(lab)
                l2 = l.astype(np.float32) * g.squeeze(-1)
                if self.clip:
                    l2 = np.clip(l2, 0, 255)
                l2 = l2.astype(l.dtype)
                lab2 = cv2.merge((l2,a,b))
                return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
            if self.clip:
                out = np.clip(out, 0, 255)
            return out.astype(frame.dtype)
        return frame

    def apply_to_video(self, video_path: str) -> str:
        return video_path
