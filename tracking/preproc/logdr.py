from __future__ import annotations
"""Dynamic Range Compression + Log Enhancement
Common in ultrasound / radar imaging: apply optional normalization, then log or gamma
curve to compress bright regions and reveal low-intensity structures.

Config parameters:
- method: 'log' | 'gamma' (default 'log')
- gamma: float (used when method='gamma', default 0.5 -> brighten darker areas)
- clip_percentile: optional float in (0,100) to clip high values before scaling (default 99.5)
- eps: small value to avoid log(0)
- per_channel: if True apply independently per channel (for RGB), else convert to gray and re-expand (default True)
"""
from typing import Dict
import numpy as np
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

from ..core.registry import register_preproc
from ..core.interfaces import PreprocessingModule


@register_preproc("LOG_DR")
class LogDynamicRange(PreprocessingModule):
    name = "LOG_DR"
    DEFAULT_CONFIG = {
        "method": "log",
        "gamma": 0.5,
        "clip_percentile": 99.5,
        "eps": 1e-6,
        "per_channel": True,
    }

    def __init__(self, config: Dict):
        self.method = str(config.get("method", "log")).lower()
        self.gamma = float(config.get("gamma", 0.5))
        self.clip_p = float(config.get("clip_percentile", 99.5))
        self.eps = float(config.get("eps", 1e-6))
        self.per_channel = bool(config.get("per_channel", True))
        if self.method not in {"log", "gamma"}:
            raise ValueError("method must be 'log' or 'gamma'")

    def _proc_channel(self, ch: np.ndarray) -> np.ndarray:
        f = ch.astype(np.float32)
        if self.clip_p > 0 and self.clip_p < 100:
            hi = float(np.percentile(f, self.clip_p))
            if hi > 0:
                f = np.clip(f, 0, hi)
        # normalize to [0, 1]; use eps as zero-guard to avoid division-by-zero
        mx = float(f.max())
        if mx <= 0.0:
            mx = self.eps
        f = f / mx
        if self.method == 'log':
            f = np.log1p(f * (np.e - 1.0))  # log compression in [0,1]
        else:  # gamma
            g = max(1e-3, self.gamma)
            f = np.power(f, g)
        f = (f * 255.0).clip(0,255).astype(np.uint8)
        return f

    def apply_to_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return self._proc_channel(frame)
        if frame.ndim == 3 and frame.shape[2] == 3:
            if self.per_channel:
                return np.stack([self._proc_channel(frame[...,i]) for i in range(3)], axis=-1)
            else:
                if cv2 is None:
                    # fallback simple average
                    gray = frame.mean(axis=-1).astype(frame.dtype)
                else:
                    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                proc = self._proc_channel(gray)
                if cv2 is None:
                    return np.repeat(proc[...,None], 3, axis=-1)
                # merge via LAB to preserve chroma roughly
                lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
                l,a,b = cv2.split(lab)
                lab2 = cv2.merge((proc, a, b))
                return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
        return frame

    def apply_to_video(self, video_path: str) -> str:
        return video_path
