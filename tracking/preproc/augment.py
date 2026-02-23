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
        "translate_frac": 0.0,
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
        self.translate_frac = float(config.get("translate_frac", 0.0))
        self.brightness = float(config.get("brightness", 0.08))
        self.contrast = float(config.get("contrast", 0.08))
        self.noise_std = float(config.get("noise_std", 0.0))
        self.train_only = bool(config.get("train_only", True))
        seed = config.get("seed", None)
        self.rng = np.random.default_rng(int(seed)) if seed is not None else np.random.default_rng()

    def apply_to_frame(self, frame: np.ndarray) -> np.ndarray:
        img = frame
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        params = self._sample_params(img.shape[:2])
        img = self._apply_geometry(img, params, interp=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT)
        img = self._apply_photometric(img, params)
        return img

    def apply_to_frame_and_mask(self, frame: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        img = frame
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8)

        params = self._sample_params(img.shape[:2])
        img = self._apply_geometry(img, params, interp=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT)
        mask = self._apply_geometry(mask, params, interp=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, border_value=0)
        img = self._apply_photometric(img, params)
        return img, mask

    def apply_to_frame_mask_bbox(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
        img = frame
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8)

        params = self._sample_params(img.shape[:2])
        img = self._apply_geometry(img, params, interp=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT)
        mask = self._apply_geometry(mask, params, interp=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, border_value=0)
        bbox_out = self._apply_bbox_geometry(bbox, params, img.shape[:2])
        img = self._apply_photometric(img, params)
        return img, mask, bbox_out

    def apply_to_frame_and_bboxes(
        self,
        frame: np.ndarray,
        bboxes: list[tuple[float, float, float, float]],
    ) -> tuple[np.ndarray, list[tuple[float, float, float, float]]]:
        img = frame
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        params = self._sample_params(img.shape[:2])
        img = self._apply_geometry(img, params, interp=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT)
        out_bboxes = [self._apply_bbox_geometry(b, params, img.shape[:2]) for b in (bboxes or [])]
        img = self._apply_photometric(img, params)
        return img, out_bboxes

    def _sample_params(self, image_shape: tuple[int, int]) -> Dict:
        img_h, img_w = image_shape
        max_dx = abs(self.translate_frac) * float(img_w)
        max_dy = abs(self.translate_frac) * float(img_h)
        return {
            "hflip": self.rng.random() < self.hflip_prob,
            "vflip": self.rng.random() < self.vflip_prob,
            "angle": float(self.rng.uniform(-self.rotate_max_deg, self.rotate_max_deg)) if self.rotate_max_deg > 0 else 0.0,
            "shift_x": float(self.rng.uniform(-max_dx, max_dx)) if max_dx > 0 else 0.0,
            "shift_y": float(self.rng.uniform(-max_dy, max_dy)) if max_dy > 0 else 0.0,
            "delta": float(self.rng.uniform(-self.brightness, self.brightness)) if self.brightness > 0 else 0.0,
            "alpha": 1.0 + float(self.rng.uniform(-self.contrast, self.contrast)) if self.contrast > 0 else 1.0,
        }

    def _apply_geometry(
        self,
        img: np.ndarray,
        params: Dict,
        interp: int,
        border_mode: int,
        border_value: int = 0,
    ) -> np.ndarray:
        out = img
        if params.get("hflip"):
            out = np.ascontiguousarray(np.fliplr(out))
        if params.get("vflip"):
            out = np.ascontiguousarray(np.flipud(out))
        angle = float(params.get("angle", 0.0))
        shift_x = float(params.get("shift_x", 0.0))
        shift_y = float(params.get("shift_y", 0.0))
        if angle != 0.0 or shift_x != 0.0 or shift_y != 0.0:
            h, w = out.shape[:2]
            m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
            m[0, 2] += shift_x
            m[1, 2] += shift_y
            out = cv2.warpAffine(
                out,
                m,
                (w, h),
                flags=interp,
                borderMode=border_mode,
                borderValue=border_value,
            )
        return out

    def _apply_photometric(self, img: np.ndarray, params: Dict) -> np.ndarray:
        out = img
        if any(v > 0 for v in (self.brightness, self.contrast)):
            delta = float(params.get("delta", 0.0))
            alpha = float(params.get("alpha", 1.0))
            out = out.astype(np.float32) / 255.0
            out = np.clip(alpha * out + delta, 0.0, 1.0)
            out = (out * 255.0).astype(np.uint8)
        if self.noise_std > 0:
            noise = self.rng.normal(0.0, self.noise_std, size=out.shape).astype(np.float32)
            out = np.clip(out.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)
        return out

    def _apply_bbox_geometry(
        self,
        bbox: tuple[float, float, float, float],
        params: Dict,
        image_shape: tuple[int, int],
    ) -> tuple[float, float, float, float]:
        x, y, w, h = bbox
        img_h, img_w = image_shape
        if params.get("hflip"):
            x = float(img_w - x - w)
        if params.get("vflip"):
            y = float(img_h - y - h)
        angle = float(params.get("angle", 0.0))
        shift_x = float(params.get("shift_x", 0.0))
        shift_y = float(params.get("shift_y", 0.0))
        if angle != 0.0:
            cx, cy = img_w / 2.0, img_h / 2.0
            m = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            pts = np.array(
                [
                    [x, y],
                    [x + w, y],
                    [x, y + h],
                    [x + w, y + h],
                ],
                dtype=np.float32,
            )
            ones = np.ones((pts.shape[0], 1), dtype=np.float32)
            pts_h = np.hstack([pts, ones])
            rot = pts_h @ m.T
            xs = rot[:, 0]
            ys = rot[:, 1]
            x0 = float(np.clip(xs.min(), 0.0, img_w))
            y0 = float(np.clip(ys.min(), 0.0, img_h))
            x1 = float(np.clip(xs.max(), 0.0, img_w))
            y1 = float(np.clip(ys.max(), 0.0, img_h))
            w = max(1.0, x1 - x0)
            h = max(1.0, y1 - y0)
            x, y = x0, y0
        if shift_x != 0.0 or shift_y != 0.0:
            x += shift_x
            y += shift_y
        x0 = float(np.clip(x, 0.0, img_w))
        y0 = float(np.clip(y, 0.0, img_h))
        x1 = float(np.clip(x + w, 0.0, img_w))
        y1 = float(np.clip(y + h, 0.0, img_h))
        w = max(1.0, x1 - x0)
        h = max(1.0, y1 - y0)
        x, y = x0, y0
        return (float(x), float(y), float(w), float(h))

    def apply_to_video(self, video_path: str) -> str:
        # On-the-fly frame augmentation is preferred; no video rewrite.
        return video_path
