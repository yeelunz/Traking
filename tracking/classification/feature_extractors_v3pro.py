from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
import logging

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None  # type: ignore

from ..core.interfaces import FramePrediction
from ..core.registry import register_feature_extractor
from .interfaces import TrajectoryFeatureExtractor
from .feature_extractors_v3lite import (
    LITE_MOTION_KEYS,
    LITE_STATIC_KEYS,
    _compute_motion_lite,
    _compute_static_lite,
)
from .texture_backbone import TextureBackboneWrapper


logger = logging.getLogger(__name__)


def _evenly_spaced_indices(n: int, k: int) -> List[int]:
    if n <= 0:
        return []
    if k <= 0 or k >= n:
        return list(range(n))
    idx = np.linspace(0, n - 1, k, dtype=int)
    return sorted(set(int(i) for i in idx))


def _crop_roi_with_padding(
    frame_bgr: np.ndarray,
    bbox: Sequence[float],
    pad_ratio: float,
) -> Optional[np.ndarray]:
    if frame_bgr is None or frame_bgr.size == 0 or len(bbox) != 4:
        return None
    h_img, w_img = frame_bgr.shape[:2]
    x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    if w <= 1.0 or h <= 1.0:
        return None

    pad_ratio = max(0.0, float(pad_ratio))
    pad_w = w * pad_ratio
    pad_h = h * pad_ratio

    x1 = int(max(0, min(w_img - 1, round(x - pad_w))))
    y1 = int(max(0, min(h_img - 1, round(y - pad_h))))
    x2 = int(max(x1 + 1, min(w_img, round(x + w + pad_w))))
    y2 = int(max(y1 + 1, min(h_img, round(y + h + pad_h))))

    roi = frame_bgr[y1:y2, x1:x2]
    if roi is None or roi.size == 0:
        return None
    return roi


def _extract_mean_roi_rgb(
    samples: Sequence[FramePrediction],
    video_path: Optional[str],
    image_size: int,
    n_frames: int,
    pad_ratio: float,
) -> np.ndarray:
    image_size = max(32, int(image_size))
    zeros = np.zeros((3, image_size, image_size), dtype=np.float32)
    if not samples or not video_path:
        return zeros

    n = len(samples)
    k = max(1, int(n_frames))
    if n <= k:
        pick = list(range(n))
    else:
        scored = sorted(
            range(n),
            key=lambda i: getattr(samples[i], "score", 0.0) or 0.0,
            reverse=True
        )
        pick = sorted(scored[:k])
    if not pick:
        return zeros

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return zeros

    patches: List[np.ndarray] = []
    try:
        for i in pick:
            s = samples[i]
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(s.frame_index))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            roi = _crop_roi_with_padding(frame, s.bbox, pad_ratio)
            if roi is None:
                continue
            resized = cv2.resize(roi, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            patches.append(rgb)
    finally:
        cap.release()

    if not patches:
        return zeros

    avg = np.mean(np.stack(patches, axis=0), axis=0)  # (H, W, C)
    chw = np.transpose(avg, (2, 0, 1)).astype(np.float32)  # (C, H, W)
    return chw


@register_feature_extractor("tab_v3_pro")
class MotionStaticV3ProFeatureExtractor(TrajectoryFeatureExtractor):
    name = "MotionStaticV3ProFeatures"
    DEFAULT_CONFIG: Dict[str, Any] = {
        "n_texture_frames": 5,
        "texture_image_size": 96,
        "roi_pad_ratio": 0.15,
        "texture_mode": "freeze",  # freeze | learnable | pretrain
        "texture_backbone": "convnext_tiny",
        "texture_dim": 10,
        "texture_pretrain_ckpt": None,
        "pretrained_backbone": True,
        "texture_device": "auto",
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        cfg = {**self.DEFAULT_CONFIG, **(params or {})}
        self._n_texture_frames = int(cfg.get("n_texture_frames", 5))
        self._texture_image_size = int(cfg.get("texture_image_size", 96))
        self._roi_pad_ratio = float(cfg.get("roi_pad_ratio", 0.15))
        self._texture_mode = str(cfg.get("texture_mode", "freeze")).lower()
        self._texture_backbone = str(cfg.get("texture_backbone", "convnext_tiny"))
        self._texture_dim = int(cfg.get("texture_dim", 10))
        self._texture_pretrain_ckpt = cfg.get("texture_pretrain_ckpt")
        self._pretrained_backbone = bool(cfg.get("pretrained_backbone", True))
        self._texture_device = str(cfg.get("texture_device", "auto"))
        self._texture_wrapper: Optional[TextureBackboneWrapper] = None

        self._pca_mean: Optional[np.ndarray] = None
        self._pca_components: Optional[np.ndarray] = None
        self._feat_mean: Optional[np.ndarray] = None
        self._feat_std: Optional[np.ndarray] = None

        self._motion_keys = list(LITE_MOTION_KEYS)
        self._static_keys = list(LITE_STATIC_KEYS)

        self._texture_keys = [f"v3pro_tex_{i:02d}" for i in range(self._texture_dim)]

        self._video_keys = self._motion_keys + self._static_keys + self._texture_keys
        self._subject_keys = self._video_keys

    def _ensure_texture_wrapper(self) -> TextureBackboneWrapper:
        if self._texture_wrapper is not None:
            return self._texture_wrapper
        wrapper = TextureBackboneWrapper(
            mode=self._texture_mode,
            backbone_name=self._texture_backbone,
            texture_dim=self._texture_dim,
            image_size=self._texture_image_size,
            pretrain_ckpt=self._texture_pretrain_ckpt,
            pretrained_imagenet=self._pretrained_backbone,
        )
        if torch is not None:
            if self._texture_device == "auto":
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                dev = self._texture_device
            wrapper = wrapper.to(dev)
            wrapper.eval()
        self._texture_wrapper = wrapper
        return wrapper

    @staticmethod
    def _pca_fit_transform(X: np.ndarray, target_dim: int):
        n, d = X.shape
        if d <= target_dim:
            pad_w = target_dim - d
            reduced = np.hstack([X, np.zeros((n, pad_w), dtype=X.dtype)]) if pad_w > 0 else X.copy()
            mean = np.zeros((d,), dtype=np.float64)
            comps = np.eye(target_dim, d, dtype=np.float64)
            return reduced, mean, comps
        mean = X.mean(axis=0)
        Xc = X - mean
        actual = min(target_dim, n, d)
        try:
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
            comps = Vt[:actual]
            reduced = Xc @ comps.T
        except np.linalg.LinAlgError:
            comps = np.eye(target_dim, d, dtype=np.float64)
            reduced = Xc[:, :target_dim]
        if reduced.shape[1] < target_dim:
            reduced = np.hstack([reduced, np.zeros((n, target_dim - reduced.shape[1]), dtype=reduced.dtype)])
        return reduced, mean, comps

    @staticmethod
    def _pca_transform(X: np.ndarray, mean: np.ndarray, components: np.ndarray, target_dim: int):
        Xc = X - mean
        try:
            reduced = Xc @ components.T
        except Exception:  # noqa: BLE001
            reduced = Xc[:, :target_dim]
        if reduced.shape[1] < target_dim:
            reduced = np.hstack([reduced, np.zeros((reduced.shape[0], target_dim - reduced.shape[1]), dtype=reduced.dtype)])
        return reduced

    def _extract_texture_vector(
        self,
        samples: Sequence[FramePrediction],
        video_path: Optional[str],
    ) -> Dict[str, Any]:
        zeros = np.zeros((self._texture_dim,), dtype=np.float32)
        tex_img = _extract_mean_roi_rgb(
            samples=samples,
            video_path=video_path,
            image_size=self._texture_image_size,
            n_frames=self._n_texture_frames,
            pad_ratio=self._roi_pad_ratio,
        )
        if torch is None:
            return {"tex": zeros}
        wrapper = self._ensure_texture_wrapper()
        try:
            dev = next(wrapper.parameters()).device
        except Exception:  # noqa: BLE001
            dev = torch.device("cpu")
        xb = torch.tensor(tex_img[None, ...], dtype=torch.float32, device=dev)

        # Automatic routing:
        # - freeze: use raw backbone feature then PCA in finalize_batch
        # - learnable/pretrain: use wrapper projection output directly
        if self._texture_mode == "freeze":
            with torch.no_grad():
                x_norm = (xb - wrapper.img_mean) / (wrapper.img_std + 1e-8)
                raw = wrapper._forward_backbone(x_norm).detach().cpu().numpy().astype(np.float64)
            return {"tex": zeros, "_raw_texture_backbone": raw.reshape(-1)}

        with torch.no_grad():
            feat = wrapper(xb).detach().cpu().numpy().reshape(-1).astype(np.float32)
        out = np.zeros((self._texture_dim,), dtype=np.float32)
        out[: min(self._texture_dim, feat.shape[0])] = feat[: min(self._texture_dim, feat.shape[0])]
        return {"tex": out}

    def feature_order(self, level: str = "video") -> Sequence[str]:
        return self._subject_keys if str(level).lower() == "subject" else self._video_keys

    def extract_video(
        self,
        samples: Sequence[FramePrediction],
        video_path: Optional[str] = None,
    ) -> Dict[str, float]:
        motion = _compute_motion_lite(samples)
        static = _compute_static_lite(samples)
        tex_info = self._extract_texture_vector(samples, video_path)
        tex_vec = np.asarray(tex_info.get("tex"), dtype=np.float32)

        out: Dict[str, float] = OrderedDict()
        for k in self._motion_keys:
            out[k] = float(motion.get(k, 0.0))
        for k in self._static_keys:
            out[k] = float(static.get(k, 0.0))
        for i, k in enumerate(self._texture_keys):
            out[k] = float(tex_vec[i]) if i < tex_vec.size else 0.0

        raw = tex_info.get("_raw_texture_backbone")
        if raw is not None:
            out["_raw_texture_backbone"] = np.asarray(raw, dtype=np.float64)
        return out

    def finalize_batch(
        self,
        features_list: Sequence[Dict[str, float]],
        *,
        fit: bool = True,
    ) -> Sequence[Dict[str, float]]:
        if not features_list:
            return []

        result: List[Dict[str, float]] = []
        if self._texture_mode != "freeze":
            for feat in features_list:
                out = OrderedDict((k, float(feat.get(k, 0.0))) for k in self._video_keys)
                result.append(out)
        else:
            raw_rows: List[np.ndarray] = []
            indices: List[int] = []
            for i, feat in enumerate(features_list):
                raw = feat.get("_raw_texture_backbone")
                if raw is None:
                    continue
                arr = np.asarray(raw, dtype=np.float64).reshape(-1)
                raw_rows.append(arr)
                indices.append(i)

            reduced = None
            if raw_rows:
                X = np.vstack(raw_rows)
                if fit:
                    reduced, self._pca_mean, self._pca_components = self._pca_fit_transform(X, self._texture_dim)
                else:
                    if self._pca_mean is None or self._pca_components is None:
                        raise RuntimeError(
                            "tab_v3_pro freeze mode requires fitted PCA state; run finalize_batch(..., fit=True) first."
                        )
                    reduced = self._pca_transform(X, self._pca_mean, self._pca_components, self._texture_dim)

            ridx = 0
            for i, feat in enumerate(features_list):
                out = OrderedDict()
                for k in self._motion_keys:
                    out[k] = float(feat.get(k, 0.0))
                for k in self._static_keys:
                    out[k] = float(feat.get(k, 0.0))
                if reduced is not None and i in indices:
                    tex_vec = reduced[ridx]
                    ridx += 1
                else:
                    tex_vec = np.zeros((self._texture_dim,), dtype=np.float64)
                for j, k in enumerate(self._texture_keys):
                    out[k] = float(tex_vec[j]) if j < tex_vec.shape[0] else 0.0
                result.append(out)

        mats = np.array(
            [[feat.get(k, 0.0) for k in self._video_keys] for feat in result],
            dtype=np.float64,
        )
        if fit:
            self._feat_mean = mats.mean(axis=0)
            self._feat_std = mats.std(axis=0)
            self._feat_std = np.where(self._feat_std < 1e-9, 1.0, self._feat_std)
            # Bypass z-score for deep network texture features
            n_base = len(self._motion_keys) + len(self._static_keys)
            self._feat_mean[n_base:] = 0.0
            self._feat_std[n_base:] = 1.0
        else:
            if self._feat_mean is None or self._feat_std is None:
                raise RuntimeError(
                    "tab_v3_pro global z-score is not fitted. finalize_batch(fit=True) must run first."
                )

        mats = (mats - self._feat_mean) / self._feat_std
        np.nan_to_num(mats, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        standardized: List[Dict[str, float]] = []
        for i in range(len(result)):
            row = OrderedDict()
            for j, k in enumerate(self._video_keys):
                row[k] = float(mats[i, j])
            standardized.append(row)
        return standardized

    def aggregate_subject(self, video_features: Sequence[Dict[str, float]]) -> Dict[str, float]:
        if not video_features:
            return OrderedDict((k, 0.0) for k in self._subject_keys)

        mat = np.array(
            [[vf.get(k, 0.0) for k in self._video_keys] for vf in video_features],
            dtype=np.float64,
        )

        mean_vec = mat.mean(axis=0)
        out: Dict[str, float] = OrderedDict(
            (k, float(v)) for k, v in zip(self._video_keys, mean_vec)
        )
        return out

    def get_state(self) -> Dict[str, Any]:
        return {
            "texture_mode": self._texture_mode,
            "pca_mean": self._pca_mean,
            "pca_components": self._pca_components,
            "feat_mean": self._feat_mean,
            "feat_std": self._feat_std,
            "texture_dim": self._texture_dim,
            "video_keys": list(self._video_keys),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise ValueError("tab_v3_pro state must be a dict")
        self._pca_mean = state.get("pca_mean")
        self._pca_components = state.get("pca_components")
        self._feat_mean = state.get("feat_mean")
        self._feat_std = state.get("feat_std")
