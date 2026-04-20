from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np

from ...core.interfaces import FramePrediction
from ...core.registry import register_feature_extractor
from .v3lite import _condition_dense, _condition_sparse, _interp_channel
from .v3pro_tsc import TimeSeriesV3ProFeatureExtractor, _pca_fit_transform, _pca_transform
from .v6 import scale_predictions_to_cm, resolve_depth_scale_for_video


logger = logging.getLogger(__name__)

N_TS_CHANNELS_V4 = 15
N_TEX_CHANNELS_V4 = 5
N_STATIC_CHANNELS_V4 = 4
N_MOTION_CHANNELS_V4 = 6
N_DEEP_START_V4 = N_STATIC_CHANNELS_V4 + N_MOTION_CHANNELS_V4


def _circularity(area: float, perimeter: float) -> float:
    if perimeter <= 0:
        return 0.0
    return float(4.0 * np.pi * area / (perimeter ** 2))


def _extract_ts_channels_v4(
    samples: Sequence[FramePrediction],
    video_path: Optional[str],
    frame_w: float = 640.0,
    frame_h: float = 480.0,
    interp_method: str = "pchip",
    n_steps: int = 256,
) -> np.ndarray:
    """Build ``(15, n_steps)`` channels: 4 static + 6 motion + 5 deep slots.

    Channel layout:
    0  csa_t
    1  eq_diam_t
    2  circularity_t
    3  aspect_ratio_t
    4  speed
    5  accel
    6  displacement_x
    7  displacement_y
    8  heading_sin
    9  heading_cos
    10..14 deep_features_1..5 (filled in finalize_batch)
    """
    del video_path, frame_w, frame_h

    if not samples:
        return np.zeros((N_TS_CHANNELS_V4, int(n_steps)), dtype=np.float32)

    ordered = sorted(samples, key=lambda s: int(s.frame_index))
    frame_indices = np.array([int(s.frame_index) for s in ordered], dtype=np.int64)
    bboxes = np.array([s.bbox for s in ordered], dtype=np.float64).reshape(-1, 4)

    n_raw = len(ordered)
    csa_sparse = np.zeros((n_raw,), dtype=np.float64)
    eq_diam_sparse = np.zeros((n_raw,), dtype=np.float64)
    circularity_sparse = np.zeros((n_raw,), dtype=np.float64)

    for i, sample in enumerate(ordered):
        bw = max(float(sample.bbox[2]), 0.0)
        bh = max(float(sample.bbox[3]), 0.0)

        area = 0.0
        perimeter = 0.0
        eq_diam = 0.0
        if sample.segmentation and sample.segmentation.stats:
            stats = sample.segmentation.stats
            area = float(getattr(stats, "area_px", 0.0) or 0.0)
            perimeter = float(getattr(stats, "perimeter_px", 0.0) or 0.0)
            eq_diam = float(getattr(stats, "equivalent_diameter_px", 0.0) or 0.0)

        if area <= 0.0:
            area = float(bw * bh)
        if perimeter <= 0.0:
            perimeter = float(2.0 * (bw + bh))
        if eq_diam <= 0.0:
            eq_diam = float(np.sqrt(max(4.0 * area / np.pi, 0.0)))

        csa_sparse[i] = area
        eq_diam_sparse[i] = eq_diam
        circularity_sparse[i] = _circularity(area, perimeter)

    cx_sparse = bboxes[:, 0] + bboxes[:, 2] / 2.0
    cy_sparse = bboxes[:, 1] + bboxes[:, 3] / 2.0
    aspect_sparse = bboxes[:, 2] / (bboxes[:, 3] + 1e-9)

    cx_sparse = _condition_sparse(cx_sparse)
    cy_sparse = _condition_sparse(cy_sparse)
    csa_sparse = _condition_sparse(csa_sparse)
    eq_diam_sparse = _condition_sparse(eq_diam_sparse)
    circularity_sparse = _condition_sparse(circularity_sparse)
    aspect_sparse = _condition_sparse(aspect_sparse)

    T = int(max(1, int(frame_indices[-1]) + 1))
    t_all = np.arange(T, dtype=np.float64)
    t_known = frame_indices.astype(np.float64)

    cx_full = _interp_channel(t_known, cx_sparse, t_all, interp_method)
    cy_full = _interp_channel(t_known, cy_sparse, t_all, interp_method)
    csa_full = _interp_channel(t_known, csa_sparse, t_all, interp_method)
    eq_diam_full = _interp_channel(t_known, eq_diam_sparse, t_all, interp_method)
    circularity_full = _interp_channel(t_known, circularity_sparse, t_all, interp_method)
    aspect_full = _interp_channel(t_known, aspect_sparse, t_all, interp_method)

    cx_full = _condition_dense(cx_full)
    cy_full = _condition_dense(cy_full)
    csa_full = _condition_dense(csa_full)
    eq_diam_full = _condition_dense(eq_diam_full)
    circularity_full = _condition_dense(circularity_full)
    aspect_full = _condition_dense(aspect_full)

    vx = np.zeros((T,), dtype=np.float64)
    vy = np.zeros((T,), dtype=np.float64)
    if T > 1:
        vx[1:] = np.diff(cx_full)
        vy[1:] = np.diff(cy_full)

    speed = np.sqrt(vx ** 2 + vy ** 2)
    accel = np.zeros((T,), dtype=np.float64)
    if T > 2:
        accel[1:] = np.abs(np.diff(speed))

    displacement_x = cx_full - float(np.median(cx_full))
    displacement_y = cy_full - float(np.median(cy_full))

    vmag = np.sqrt(vx ** 2 + vy ** 2) + 1e-12
    heading_sin = vy / vmag
    heading_cos = vx / vmag

    timeline = np.zeros((T, N_TS_CHANNELS_V4), dtype=np.float64)
    timeline[:, 0] = csa_full
    timeline[:, 1] = eq_diam_full
    timeline[:, 2] = circularity_full
    timeline[:, 3] = aspect_full
    timeline[:, 4] = speed
    timeline[:, 5] = accel
    timeline[:, 6] = displacement_x
    timeline[:, 7] = displacement_y
    timeline[:, 8] = heading_sin
    timeline[:, 9] = heading_cos

    np.nan_to_num(timeline, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    resample_idx = np.round(np.linspace(0, T - 1, int(n_steps))).astype(int)
    resample_idx = np.clip(resample_idx, 0, T - 1)
    return timeline[resample_idx, :].T.astype(np.float32)


@register_feature_extractor("tsc_latest")
@register_feature_extractor("tsc_v4")
class TimeSeriesV4FeatureExtractor(TimeSeriesV3ProFeatureExtractor):
    """TSC v4 extractor.

    Layout:
    - 4 static channels: csa, eq_diam, circularity, aspect_ratio
    - 6 motion channels: speed, accel, displacement_x, displacement_y, heading_sin, heading_cos
    - 5 deep channels: deep_features_1..5 (from ConvNeXt timeline)

    Pixel-derived channels are converted to centimetre units before building
    the base timeline. Deep texture extraction still reads from raw video.
    """

    name = "TimeSeriesV4Features"

    def __init__(self, params: Optional[Dict[str, object]] = None):
        super().__init__(params=params)

        if self._texture_mode != "freeze" and self._texture_dim != N_TEX_CHANNELS_V4:
            logger.warning(
                "tsc_v4 %s mode uses projected texture channels=%d; overriding texture_dim from %d to %d.",
                self._texture_mode,
                N_TEX_CHANNELS_V4,
                self._texture_dim,
                N_TEX_CHANNELS_V4,
            )
            self._texture_dim = N_TEX_CHANNELS_V4

        self._n_channels = N_TS_CHANNELS_V4
        self._flat_len = self._n_channels * self._n_steps
        self._video_keys = [f"tsv4_{i:04d}" for i in range(self._flat_len)]
        self._subject_keys = list(self._video_keys)

    def _scaled_samples_for_video(
        self,
        samples: Sequence[FramePrediction],
        video_path: Optional[str],
    ) -> List[FramePrediction]:
        depth_scale = resolve_depth_scale_for_video(video_path)
        return scale_predictions_to_cm(samples, depth_scale)

    def extract_video(
        self,
        samples: Sequence[FramePrediction],
        video_path: Optional[str] = None,
    ) -> Dict[str, float]:
        scaled_samples = self._scaled_samples_for_video(samples, video_path)

        ts_base = _extract_ts_channels_v4(
            scaled_samples,
            video_path=None,
            frame_w=self._frame_w,
            frame_h=self._frame_h,
            interp_method=self._interp_method,
            n_steps=self._n_steps,
        )

        tex_raw = self._extract_texture_ts(samples, video_path, self._n_steps)

        flat = ts_base.flatten(order="C")
        feat: Dict[str, float] = OrderedDict()
        for i, value in enumerate(flat):
            feat[f"tsv4_{i:04d}"] = float(value)
        feat["_ts_n_vars"] = float(self._n_channels)
        feat["_ts_n_timesteps"] = float(self._n_steps)
        if self._texture_mode == "freeze":
            feat["_raw_convnext_ts"] = tex_raw
        else:
            feat["_proj_convnext_ts"] = tex_raw
        return feat

    def finalize_batch(
        self,
        features_list: Sequence[Dict[str, float]],
        *,
        fit: bool = True,
    ) -> Sequence[Dict[str, float]]:
        if not features_list:
            return []

        result = list(features_list)
        deep_slice = slice(N_DEEP_START_V4, N_DEEP_START_V4 + N_TEX_CHANNELS_V4)

        if self._texture_mode == "freeze":
            raw_mats: List[np.ndarray] = []
            indices: List[int] = []
            for i, feat in enumerate(features_list):
                rm = feat.get("_raw_convnext_ts")
                if rm is not None:
                    raw_mats.append(np.asarray(rm, dtype=np.float64))
                    indices.append(i)

            if raw_mats:
                all_frames = np.vstack(raw_mats)
                target = N_TEX_CHANNELS_V4

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
                    for ch in range(N_TEX_CHANNELS_V4):
                        ts[N_DEEP_START_V4 + ch, :n_fill] = reduced_video[:n_fill, ch].astype(np.float32)

                    flat = ts.flatten(order="C")
                    cleaned: Dict[str, float] = OrderedDict()
                    for j, key in enumerate(self._video_keys):
                        cleaned[key] = float(flat[j])
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
                n_tex = min(N_TEX_CHANNELS_V4, proj_arr.shape[1])
                vec = np.array([feat.get(k, 0.0) for k in self._video_keys], dtype=np.float32)
                ts = vec.reshape(self._n_channels, self._n_steps)
                for ch in range(n_tex):
                    ts[N_DEEP_START_V4 + ch, :n_fill] = proj_arr[:n_fill, ch].astype(np.float32)

                flat = ts.flatten(order="C")
                cleaned: Dict[str, float] = OrderedDict()
                for j, key in enumerate(self._video_keys):
                    cleaned[key] = float(flat[j])
                result[i] = cleaned

        mats = np.zeros((len(result), self._n_channels, self._n_steps), dtype=np.float64)
        for i, feat in enumerate(result):
            vec = np.array([feat.get(k, 0.0) for k in self._video_keys], dtype=np.float64)
            mats[i] = vec.reshape(self._n_channels, self._n_steps)

        if fit:
            self._channel_mean = mats.mean(axis=(0, 2))
            self._channel_std = mats.std(axis=(0, 2))
            self._channel_std = np.where(self._channel_std < 1e-9, 1.0, self._channel_std)
            self._channel_mean[deep_slice] = 0.0
            self._channel_std[deep_slice] = 1.0
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
            for j, key in enumerate(self._video_keys):
                cleaned[key] = float(flat[j])
            out.append(cleaned)
        return out


__all__ = [
    "N_TS_CHANNELS_V4",
    "N_TEX_CHANNELS_V4",
    "TimeSeriesV4FeatureExtractor",
]
