from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ...core.interfaces import FramePrediction
from ...core.registry import register_feature_extractor
from ..interfaces import TrajectoryFeatureExtractor
from .v3lite import N_TS_STEPS_LITE, _extract_ts_channels_lite
from ..texture_backbone import TextureBackboneWrapper

logger = logging.getLogger(__name__)

N_TS_CHANNELS_V3PRO = 12
N_TEX_CHANNELS_V3PRO = 3


def _pca_fit_transform(
    X: np.ndarray, target_dim: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, d = X.shape
    if d <= target_dim:
        pad_w = target_dim - d
        reduced = np.hstack([X, np.zeros((n, pad_w))]) if pad_w > 0 else X.copy()
        return reduced, np.zeros(d), np.eye(target_dim, d)

    mean = X.mean(axis=0)
    Xc = X - mean
    actual = min(target_dim, n, d)
    try:
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        comps = Vt[:actual]
        reduced = Xc @ comps.T
    except np.linalg.LinAlgError:
        comps = np.eye(target_dim, d)
        reduced = Xc[:, :target_dim]

    if reduced.shape[1] < target_dim:
        pad = np.zeros((n, target_dim - reduced.shape[1]))
        reduced = np.hstack([reduced, pad])
    return reduced, mean, comps


def _pca_transform(
    X: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
    target_dim: int,
) -> np.ndarray:
    n = X.shape[0]
    Xc = X - mean
    try:
        reduced = Xc @ components.T
    except (ValueError, np.linalg.LinAlgError):
        reduced = Xc[:, :target_dim]
    if reduced.shape[1] < target_dim:
        pad = np.zeros((n, target_dim - reduced.shape[1]))
        reduced = np.hstack([reduced, pad])
    return reduced


def _crop_roi_rgb(
    frame_bgr: np.ndarray,
    bbox: Sequence[float],
    image_size: int,
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
    if roi.size == 0:
        return None

    resized = cv2.resize(roi, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(rgb, (2, 0, 1)).astype(np.float32)


@register_feature_extractor("tsc_v3_pro")
class TimeSeriesV3ProFeatureExtractor(TrajectoryFeatureExtractor):
    """TSC v3-pro extractor.

    Keeps v3-lite motion/static channels (0-8) and replaces texture channels
    with ConvNeXt pretrained features (two-stage pretrain) projected to 3 channels.
    """

    name = "TimeSeriesV3ProFeatures"
    DEFAULT_CONFIG: Dict[str, Any] = {
        "frame_w": 640.0,
        "frame_h": 480.0,
        "interp_method": "cubic",
        "n_steps": N_TS_STEPS_LITE,
        "texture_mode": "pretrain",  # pretrain | freeze | learnable(fallback)
        "texture_backbone": "convnext_tiny",
        "texture_dim": 32,
        "texture_pretrain_ckpt": None,
        "texture_image_size": 96,
        "roi_pad_ratio": 0.15,
        "texture_batch_size": 64,
        "texture_device": "auto",
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        cfg = {**self.DEFAULT_CONFIG, **(params or {})}
        self._frame_w = float(cfg.get("frame_w", 640.0))
        self._frame_h = float(cfg.get("frame_h", 480.0))
        self._interp_method = str(cfg.get("interp_method", "cubic"))
        self._n_steps = int(cfg.get("n_steps", N_TS_STEPS_LITE))

        self._texture_mode = str(cfg.get("texture_mode", "pretrain")).lower()
        if self._texture_mode not in {"freeze", "learnable", "pretrain"}:
            logger.warning("Unknown texture_mode=%s, fallback to freeze", self._texture_mode)
            self._texture_mode = "freeze"

        self._texture_backbone = str(cfg.get("texture_backbone", "convnext_tiny"))
        self._texture_dim = int(cfg.get("texture_dim", 32))
        if self._texture_mode != "freeze" and self._texture_dim != N_TEX_CHANNELS_V3PRO:
            logger.warning(
                "tsc_v3_pro %s mode uses projected texture channels=%d; overriding texture_dim from %d to %d.",
                self._texture_mode,
                N_TEX_CHANNELS_V3PRO,
                self._texture_dim,
                N_TEX_CHANNELS_V3PRO,
            )
            self._texture_dim = N_TEX_CHANNELS_V3PRO
        self._texture_pretrain_ckpt = cfg.get("texture_pretrain_ckpt")
        self._texture_image_size = int(cfg.get("texture_image_size", 96))
        self._roi_pad_ratio = float(cfg.get("roi_pad_ratio", 0.15))
        self._texture_batch_size = int(cfg.get("texture_batch_size", 64))
        self._texture_device = str(cfg.get("texture_device", "auto"))

        self._n_channels = N_TS_CHANNELS_V3PRO
        self._flat_len = self._n_channels * self._n_steps
        self._video_keys = [f"tsv3p_{i:04d}" for i in range(self._flat_len)]
        self._subject_keys = self._video_keys

        self._pca_mean: Optional[np.ndarray] = None
        self._pca_components: Optional[np.ndarray] = None
        self._channel_mean: Optional[np.ndarray] = None
        self._channel_std: Optional[np.ndarray] = None
        self._texture_wrapper = None

    def feature_order(self, level: str = "video") -> Sequence[str]:
        return self._video_keys

    def _ensure_texture_wrapper(self):
        if self._texture_wrapper is not None:
            return self._texture_wrapper
        self._texture_wrapper = TextureBackboneWrapper(
            mode=self._texture_mode,
            backbone_name=self._texture_backbone,
            texture_dim=self._texture_dim,
            image_size=self._texture_image_size,
            pretrain_ckpt=self._texture_pretrain_ckpt,
            pretrained_imagenet=True,
        )
        if self._texture_device == "auto":
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            dev = self._texture_device
        self._texture_wrapper.to(dev)
        self._texture_wrapper.eval()
        return self._texture_wrapper

    def _extract_texture_ts(
        self,
        samples: Sequence[FramePrediction],
        video_path: Optional[str],
        n_steps: int,
    ) -> np.ndarray:
        out = np.zeros((n_steps, self._texture_dim), dtype=np.float32)
        if not samples or not video_path:
            return out

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return out

        known = np.array([int(s.frame_index) for s in samples], dtype=np.int64)
        order = np.argsort(known)
        known = known[order]
        samples_sorted = [samples[i] for i in order]

        T = int(known[-1]) + 1 if len(known) > 0 else 1
        target_frames = np.round(np.linspace(0, max(0, T - 1), n_steps)).astype(int)

        rois: List[np.ndarray] = []
        valid_indices: List[int] = []
        try:
            for i, target in enumerate(target_frames):
                pos = int(np.searchsorted(known, target))
                pos = min(pos, len(known) - 1)
                if pos > 0:
                    left_dist = abs(int(known[pos - 1]) - int(target))
                    right_dist = abs(int(known[pos]) - int(target))
                    ref = samples_sorted[pos - 1] if left_dist <= right_dist else samples_sorted[pos]
                else:
                    ref = samples_sorted[0]

                cap.set(cv2.CAP_PROP_POS_FRAMES, int(target))
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue

                roi = _crop_roi_rgb(
                    frame,
                    ref.bbox,
                    image_size=self._texture_image_size,
                    pad_ratio=self._roi_pad_ratio,
                )
                if roi is None:
                    continue
                rois.append(roi)
                valid_indices.append(i)
        finally:
            cap.release()

        if not rois:
            return out

        wrapper = self._ensure_texture_wrapper()
        import torch

        device = next(wrapper.parameters()).device
        bs = max(1, self._texture_batch_size)
        for start in range(0, len(rois), bs):
            end = min(start + bs, len(rois))
            batch = np.stack(rois[start:end], axis=0)
            xb = torch.tensor(batch, dtype=torch.float32, device=device)
            with torch.no_grad():
                if self._texture_mode == "freeze":
                    x_norm = (xb - wrapper.img_mean) / (wrapper.img_std + 1e-8)
                    emb = wrapper._forward_backbone(x_norm).detach().cpu().numpy().astype(np.float32)
                else:
                    emb = wrapper(xb).detach().cpu().numpy().astype(np.float32)
            for j, row in enumerate(range(start, end)):
                out[valid_indices[row]] = emb[j]
        return out

    def extract_video(
        self,
        samples: Sequence[FramePrediction],
        video_path: Optional[str] = None,
    ) -> Dict[str, float]:
        # base 12-ch from v3lite without texture input (texture channels become zero)
        ts_base = _extract_ts_channels_lite(
            samples,
            video_path=None,
            frame_w=self._frame_w,
            frame_h=self._frame_h,
            interp_method=self._interp_method,
            n_steps=self._n_steps,
        )

        # texture embedding timeline to be reduced in finalize_batch
        tex_raw = self._extract_texture_ts(samples, video_path, self._n_steps)

        flat = ts_base.flatten(order="C")
        feat: Dict[str, float] = OrderedDict()
        for i, v in enumerate(flat):
            feat[f"tsv3p_{i:04d}"] = float(v)
        feat["_ts_n_vars"] = float(self._n_channels)
        feat["_ts_n_timesteps"] = float(self._n_steps)
        if self._texture_mode == "freeze":
            feat["_raw_convnext_ts"] = tex_raw
        else:
            feat["_proj_convnext_ts"] = tex_raw
        return feat

    def aggregate_subject(
        self, video_features: Sequence[Dict[str, float]]
    ) -> Dict[str, float]:
        clean = [{k: v for k, v in vf.items() if not k.startswith("_")} for vf in video_features]
        if not clean:
            return OrderedDict((k, 0.0) for k in self._subject_keys)
        mat = np.array([[vf.get(k, 0.0) for k in self._video_keys] for vf in clean], dtype=np.float64)
        mean_vec = mat.mean(axis=0)
        return OrderedDict((k, float(v)) for k, v in zip(self._video_keys, mean_vec))

    def finalize_batch(
        self,
        features_list: Sequence[Dict[str, float]],
        *,
        fit: bool = True,
    ) -> Sequence[Dict[str, float]]:
        if not features_list:
            return []

        result = list(features_list)
        if self._texture_mode == "freeze":
            raw_mats: List[np.ndarray] = []
            indices: List[int] = []
            for i, feat in enumerate(features_list):
                rm = feat.get("_raw_convnext_ts")
                if rm is not None:
                    raw_mats.append(np.asarray(rm, dtype=np.float64))
                    indices.append(i)

            if raw_mats:
                all_frames = np.vstack(raw_mats)  # (sum_steps, raw_backbone_dim)
                target = N_TEX_CHANNELS_V3PRO

                if fit:
                    reduced_all, self._pca_mean, self._pca_components = _pca_fit_transform(all_frames, target)
                else:
                    if self._pca_mean is None or self._pca_components is None:
                        raise RuntimeError(
                            "PCA state is not available. finalize_batch(fit=True) must run on training set first."
                        )
                    reduced_all = _pca_transform(all_frames, self._pca_mean, self._pca_components, target)

                offset = 0
                for idx, row_idx in enumerate(indices):
                    n_frames = raw_mats[idx].shape[0]
                    reduced_video = reduced_all[offset : offset + n_frames]
                    offset += n_frames

                    feat = dict(result[row_idx])
                    vec = np.array([feat.get(k, 0.0) for k in self._video_keys], dtype=np.float32)
                    ts = vec.reshape(self._n_channels, self._n_steps)

                    n_fill = min(self._n_steps, reduced_video.shape[0])
                    ts[9, :n_fill] = reduced_video[:n_fill, 0].astype(np.float32)
                    ts[10, :n_fill] = reduced_video[:n_fill, 1].astype(np.float32)
                    ts[11, :n_fill] = reduced_video[:n_fill, 2].astype(np.float32)

                    flat = ts.flatten(order="C")
                    cleaned: Dict[str, float] = OrderedDict()
                    for j, k in enumerate(self._video_keys):
                        cleaned[k] = float(flat[j])
                    result[row_idx] = cleaned
        else:
            for i, feat in enumerate(features_list):
                proj = feat.get("_proj_convnext_ts")
                if proj is None:
                    continue
                proj_arr = np.asarray(proj, dtype=np.float32)
                if proj_arr.ndim != 2:
                    continue
                n_fill = min(self._n_steps, proj_arr.shape[0])
                vec = np.array([feat.get(k, 0.0) for k in self._video_keys], dtype=np.float32)
                ts = vec.reshape(self._n_channels, self._n_steps)
                ts[9, :n_fill] = proj_arr[:n_fill, 0].astype(np.float32)
                ts[10, :n_fill] = proj_arr[:n_fill, 1].astype(np.float32)
                ts[11, :n_fill] = proj_arr[:n_fill, 2].astype(np.float32)
                flat = ts.flatten(order="C")
                cleaned: Dict[str, float] = OrderedDict()
                for j, k in enumerate(self._video_keys):
                    cleaned[k] = float(flat[j])
                result[i] = cleaned

        mats = np.zeros((len(result), self._n_channels, self._n_steps), dtype=np.float64)
        for i, feat in enumerate(result):
            vec = np.array([feat.get(k, 0.0) for k in self._video_keys], dtype=np.float64)
            mats[i] = vec.reshape(self._n_channels, self._n_steps)

        if fit:
            self._channel_mean = mats.mean(axis=(0, 2))
            self._channel_std = mats.std(axis=(0, 2))
            self._channel_std = np.where(self._channel_std < 1e-9, 1.0, self._channel_std)
            # Bypass z-score for deep network texture features
            self._channel_mean[9:12] = 0.0
            self._channel_std[9:12] = 1.0
        else:
            if self._channel_mean is None or self._channel_std is None:
                raise RuntimeError(
                    "Global channel scaler is not fitted. finalize_batch(fit=True) must run on training first."
                )

        mean = self._channel_mean.reshape(1, self._n_channels, 1)
        std = self._channel_std.reshape(1, self._n_channels, 1)
        mats = (mats - mean) / std
        np.nan_to_num(mats, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        out: List[Dict[str, float]] = []
        for i in range(len(result)):
            flat = mats[i].reshape(-1)
            cleaned: Dict[str, float] = OrderedDict()
            for j, k in enumerate(self._video_keys):
                cleaned[k] = float(flat[j])
            out.append(cleaned)
        return out

    def get_state(self) -> Dict[str, Any]:
        return {
            "texture_mode": self._texture_mode,
            "pca_mean": self._pca_mean,
            "pca_components": self._pca_components,
            "channel_mean": self._channel_mean,
            "channel_std": self._channel_std,
            "n_channels": self._n_channels,
            "n_steps": self._n_steps,
            "video_keys": list(self._video_keys),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise ValueError("tsc_v3_pro state must be a dict")
        self._pca_mean = state.get("pca_mean")
        self._pca_components = state.get("pca_components")
        self._channel_mean = state.get("channel_mean")
        self._channel_std = state.get("channel_std")
