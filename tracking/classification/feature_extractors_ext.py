"""Extended feature extractors for CTS ultrasound classification.

Provides two feature extractors that use **ResNet-18 deep texture features**
extracted from segmentation-masked ROIs:

1. ``motion_texture_static``
   Motion (31 D) + CTS-Static (35 D) + ResNet Texture PCA (66 D) = **132 D**.
   Suited for tabular classifiers (RF / XGBoost / LightGBM / TabPFN v2).

2. ``time_series``
   Per-frame multivariate time series — **18 channels × 128 steps = 2 304 D**.
   Channels 0–9: geometric / kinematic (cx, cy, bw, bh, area, seg_area,
   circularity, eq_diam, speed, flatness).
   Channels 10–17: ResNet texture PCA (8 D per frame).
   Suited for TSC classifiers (MultiRocket / PatchTST / TimeMachine).

Texture Pipeline (shared by both extractors)
---------------------------------------------
1. Crop bounding-box ROI from video frame.
2. If segmentation mask available → apply mask (zero-out background).
3. Forward through truncated ResNet-18 (pretrained) → 512-D embedding.
4. PCA reduction for dimensionality control (fitted in ``finalize_batch``).

CTS 常用靜態診斷指標參考:
  - CSA (Cross-Sectional Area，橫截面積) > 9–10 mm² 視為異常
  - Flattening Index (扁平指數) = 長軸 / 短軸
  - Swelling Ratio (腫脹率) = CSA_grasp / CSA_rest
  - Circularity (圓度)
  - Echo Intensity (回音強度) within the nerve mask
  - Nerve displacement (神經移位量)
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..core.interfaces import FramePrediction
from ..core.registry import register_feature_extractor
from .interfaces import TrajectoryFeatureExtractor
from .feature_extractors import (
    _compute_motion_features,
    _aggregate_video_features,
    _build_subject_keys,
    _KalmanSmoother2D,  # kept for backward compatibility only
    _safe_mean,
    _safe_std,
    MOTION_FEATURE_KEYS,
)
from .trajectory_filter import (
    multiscale_hampel as _multiscale_hampel,
    bidirectional_savgol as _bidirectional_savgol,
)
from .texture_resnet import MaskedROIResNetExtractor, RESNET_FEAT_DIM

logger = logging.getLogger(__name__)


def _global_standardize_feature_dicts(
    features_list: Sequence[Dict[str, float]],
    keys: Sequence[str],
    *,
    fit: bool,
    mean: Optional[np.ndarray],
    std: Optional[np.ndarray],
) -> Tuple[Sequence[Dict[str, float]], np.ndarray, np.ndarray]:
    """Global Z-score standardization for tabular feature dicts."""
    if not features_list:
        if mean is None:
            mean = np.zeros(len(keys), dtype=np.float64)
        if std is None:
            std = np.ones(len(keys), dtype=np.float64)
        return [], mean, std

    X = np.array(
        [[feat.get(k, 0.0) for k in keys] for feat in features_list],
        dtype=np.float64,
    )

    if fit:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std = np.where(std < 1e-9, 1.0, std)
    else:
        if mean is None or std is None:
            raise RuntimeError(
                "Global feature scaler is not available. finalize_batch(fit=True) "
                "must be called on the training set before finalize_batch(fit=False)."
            )

    X = (X - mean) / std
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    out: List[Dict[str, float]] = []
    for i, feat in enumerate(features_list):
        new_feat = dict(feat)
        for j, k in enumerate(keys):
            new_feat[k] = float(X[i, j])
        out.append(new_feat)
    return out, mean, std

# ═════════════════════════════════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════════════════════════════════

N_GEO_VARS: int = 10           # Geometric time-series channels
N_TEX_PCA_TS: int = 8          # ResNet PCA texture channels for TSC
N_TS_VARS: int = N_GEO_VARS + N_TEX_PCA_TS   # 18 total TS channels
N_TS_STEPS: int = 256          # Fixed time-series length (pad / resample)

# ═════════════════════════════════════════════════════════════════════════════
# CTS Static Feature Keys (35 D)
# ═════════════════════════════════════════════════════════════════════════════

CTS_STATIC_FEATURE_KEYS: List[str] = [
    # CSA 統計 (from seg mask area_px)
    "cts_csa_min",
    "cts_csa_max",
    "cts_csa_mean",
    "cts_csa_std",
    "cts_csa_range",
    "cts_csa_cv",
    # Swelling proxy
    "cts_csa_first",
    "cts_csa_last",
    "cts_swelling_ratio",
    # Flattening Index (bbox w / h)
    "cts_flat_mean",
    "cts_flat_std",
    "cts_flat_max",
    "cts_flat_min",
    # Circularity
    "cts_circularity_first",
    "cts_circularity_last",
    "cts_circularity_mean",
    "cts_circularity_std",
    # Equivalent diameter
    "cts_eq_diam_mean",
    "cts_eq_diam_std",
    "cts_eq_diam_first",
    "cts_eq_diam_last",
    # Compactness
    "cts_compact_mean",
    "cts_compact_std",
    # Echo Intensity (bbox patch)
    "cts_echo_mean_first",
    "cts_echo_std_first",
    "cts_echo_mean_last",
    "cts_echo_std_last",
    "cts_echo_delta",
    # Nerve displacement
    "cts_nerve_net_disp",
    "cts_nerve_path_len",
    "cts_nerve_stability_x",
    "cts_nerve_stability_y",
    # Aspect ratio
    "cts_aspect_first",
    "cts_aspect_last",
]

# ═════════════════════════════════════════════════════════════════════════════
# Time-Series Channel Names
# ═════════════════════════════════════════════════════════════════════════════

TS_GEO_CHANNEL_KEYS: List[str] = [
    "ts_cx",            # centroid x (raw px)
    "ts_cy",            # centroid y (raw px)
    "ts_bw",            # bbox width  (raw px)
    "ts_bh",            # bbox height (raw px)
    "ts_area",          # bbox area   (raw px^2)
    "ts_seg_area",      # seg mask area (raw px^2)
    "ts_circularity",   # seg circularity
    "ts_eq_diam",       # equivalent diameter (raw px)
    "ts_speed",         # inter-frame centroid speed (raw px/frame)
    "ts_flat",          # flattening index w / h
]

TS_TEX_CHANNEL_KEYS: List[str] = [
    f"ts_tex_pca_{i}" for i in range(N_TEX_PCA_TS)
]

TS_CHANNEL_KEYS: List[str] = TS_GEO_CHANNEL_KEYS + TS_TEX_CHANNEL_KEYS

# ═════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═════════════════════════════════════════════════════════════════════════════


def _circularity(area: float, perimeter: float) -> float:
    if perimeter <= 0:
        return 0.0
    return float(4.0 * np.pi * area / (perimeter ** 2))


def _compactness(area: float, perimeter: float) -> float:
    if perimeter <= 0:
        return 0.0
    return float(4.0 * np.pi * area / (perimeter ** 2))


def _read_gray_patch(
    video_path: str,
    sample: FramePrediction,
    target_size: int = 64,
) -> Optional[np.ndarray]:
    """Read bbox patch from video frame as grayscale."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        frame_idx = max(0, int(round(sample.frame_index)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        h, w = frame.shape[:2]
        x, y, bw, bh = sample.bbox
        x0 = max(0, int(np.floor(x)))
        y0 = max(0, int(np.floor(y)))
        x1 = min(w, int(np.ceil(x + bw)))
        y1 = min(h, int(np.ceil(y + bh)))
        if x1 <= x0 or y1 <= y0:
            return None
        patch = frame[y0:y1, x0:x1]
        if patch.ndim == 3:
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        if min(patch.shape[:2]) > 0:
            patch = cv2.resize(
                patch, (target_size, target_size), interpolation=cv2.INTER_AREA
            )
        return patch
    finally:
        cap.release()


# ═════════════════════════════════════════════════════════════════════════════
# CTS Static Features (35 D)
# ═════════════════════════════════════════════════════════════════════════════


def _compute_cts_static_features(
    samples: Sequence[FramePrediction],
    video_path: Optional[str] = None,
) -> Dict[str, float]:
    """Extract CTS-specific static diagnostic features."""
    zeros = OrderedDict((k, 0.0) for k in CTS_STATIC_FEATURE_KEYS)
    if not samples:
        return zeros

    usable = [s for s in samples if s.segmentation and s.segmentation.stats
              and getattr(s.segmentation.stats, 'area_px', 0.0) > 0]

    # ── CSA stats ──
    if usable:
        areas = np.array(
            [s.segmentation.stats.area_px for s in usable], dtype=np.float64
        )
        perims = np.array(
            [s.segmentation.stats.perimeter_px for s in usable], dtype=np.float64
        )
        eq_diam = np.array(
            [s.segmentation.stats.equivalent_diameter_px for s in usable],
            dtype=np.float64,
        )
        circs = np.array(
            [
                _circularity(
                    s.segmentation.stats.area_px,
                    s.segmentation.stats.perimeter_px,
                )
                for s in usable
            ],
            dtype=np.float64,
        )
        compacts = np.array(
            [
                _compactness(
                    s.segmentation.stats.area_px,
                    s.segmentation.stats.perimeter_px,
                )
                for s in usable
            ],
            dtype=np.float64,
        )
    else:
        areas = perims = eq_diam = circs = compacts = np.zeros(0)

    feat = OrderedDict()

    # CSA
    if areas.size:
        feat["cts_csa_min"] = float(areas.min())
        feat["cts_csa_max"] = float(areas.max())
        feat["cts_csa_mean"] = float(areas.mean())
        feat["cts_csa_std"] = _safe_std(areas)
        feat["cts_csa_range"] = float(areas.max() - areas.min())
        feat["cts_csa_cv"] = _safe_std(areas) / (float(areas.mean()) + 1e-9)
        feat["cts_csa_first"] = float(areas[0])
        feat["cts_csa_last"] = float(areas[-1])
        feat["cts_swelling_ratio"] = float(areas.max() / (areas.min() + 1e-9))
    else:
        for k in [
            "cts_csa_min", "cts_csa_max", "cts_csa_mean", "cts_csa_std",
            "cts_csa_range", "cts_csa_cv", "cts_csa_first", "cts_csa_last",
            "cts_swelling_ratio",
        ]:
            feat[k] = 0.0

    # ── Flattening Index ──
    bboxes = np.array([s.bbox for s in samples], dtype=np.float64)
    flat = bboxes[:, 2] / (bboxes[:, 3] + 1e-9)
    feat["cts_flat_mean"] = float(flat.mean())
    feat["cts_flat_std"] = _safe_std(flat)
    feat["cts_flat_max"] = float(flat.max())
    feat["cts_flat_min"] = float(flat.min())

    # ── Circularity ──
    if circs.size:
        feat["cts_circularity_first"] = float(circs[0])
        feat["cts_circularity_last"] = float(circs[-1])
        feat["cts_circularity_mean"] = float(circs.mean())
        feat["cts_circularity_std"] = _safe_std(circs)
    else:
        feat.update({
            "cts_circularity_first": 0.0,
            "cts_circularity_last": 0.0,
            "cts_circularity_mean": 0.0,
            "cts_circularity_std": 0.0,
        })

    # ── Equivalent diameter ──
    if eq_diam.size:
        feat["cts_eq_diam_mean"] = float(eq_diam.mean())
        feat["cts_eq_diam_std"] = _safe_std(eq_diam)
        feat["cts_eq_diam_first"] = float(eq_diam[0])
        feat["cts_eq_diam_last"] = float(eq_diam[-1])
    else:
        feat.update({
            "cts_eq_diam_mean": 0.0,
            "cts_eq_diam_std": 0.0,
            "cts_eq_diam_first": 0.0,
            "cts_eq_diam_last": 0.0,
        })

    # ── Compactness ──
    if compacts.size:
        feat["cts_compact_mean"] = float(compacts.mean())
        feat["cts_compact_std"] = _safe_std(compacts)
    else:
        feat.update({"cts_compact_mean": 0.0, "cts_compact_std": 0.0})

    # ── Echo Intensity ──
    echo_first_mean = echo_first_std = echo_last_mean = echo_last_std = 0.0
    if video_path and samples:
        patch_first = _read_gray_patch(video_path, samples[0])
        if patch_first is not None:
            echo_first_mean = float(patch_first.mean())
            echo_first_std = float(patch_first.std())
        patch_last = _read_gray_patch(video_path, samples[-1])
        if patch_last is not None:
            echo_last_mean = float(patch_last.mean())
            echo_last_std = float(patch_last.std())
    feat["cts_echo_mean_first"] = echo_first_mean
    feat["cts_echo_std_first"] = echo_first_std
    feat["cts_echo_mean_last"] = echo_last_mean
    feat["cts_echo_std_last"] = echo_last_std
    feat["cts_echo_delta"] = echo_last_mean - echo_first_mean

    # ── Nerve displacement ──
    centers = np.array(
        [
            (s.bbox[0] + s.bbox[2] / 2.0, s.bbox[1] + s.bbox[3] / 2.0)
            for s in samples
        ],
        dtype=np.float64,
    )
    if len(centers) > 1:
        disp_vec = centers[-1] - centers[0]
        net_disp = float(np.linalg.norm(disp_vec))
        step_dists = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        path_len = float(step_dists.sum())
    else:
        net_disp = path_len = 0.0
    feat["cts_nerve_net_disp"] = net_disp
    feat["cts_nerve_path_len"] = path_len
    feat["cts_nerve_stability_x"] = _safe_std(centers[:, 0])
    feat["cts_nerve_stability_y"] = _safe_std(centers[:, 1])

    # ── Aspect ratio ──
    feat["cts_aspect_first"] = float(bboxes[0, 2] / (bboxes[0, 3] + 1e-9))
    feat["cts_aspect_last"] = float(bboxes[-1, 2] / (bboxes[-1, 3] + 1e-9))

    return feat


# ═════════════════════════════════════════════════════════════════════════════
# Geometric time-series channels (10 D per frame)
# ═════════════════════════════════════════════════════════════════════════════


_SCIPY_INTERP_OK: bool
try:
    from scipy.interpolate import CubicSpline as _CubicSpline  # noqa: F401
    _SCIPY_INTERP_OK = True
except ImportError:
    _SCIPY_INTERP_OK = False

_SCIPY_SIGNAL_OK: bool
try:
    from scipy.signal import savgol_filter as _savgol_filter  # noqa: F401
    _SCIPY_SIGNAL_OK = True
except ImportError:
    _SCIPY_SIGNAL_OK = False


# ═════════════════════════════════════════════════════════════════════════════
# Hampel + Savitzky-Golay signal conditioning helpers
# ═════════════════════════════════════════════════════════════════════════════


def _hampel_filter_1d(
    values: np.ndarray,
    half_window: int = 3,
    n_sigma: float = 3.0,
) -> np.ndarray:
    """Multi-scale Hampel outlier filter for a 1-D array.

    Uses two-stage filtering (macro + micro) with mirror padding via
    ``trajectory_filter.multiscale_hampel``.  Falls back to a simple
    single-pass for very short sequences.
    """
    n = len(values)
    if n < 3:
        return values.copy()
    filtered, _ = _multiscale_hampel(
        values,
        macro_ratio=0.15,
        macro_sigma=n_sigma,
        micro_hw=half_window,
        micro_sigma=n_sigma,
        use_mirror_pad=True,
    )
    return filtered


def _savgol_smooth_1d(
    values: np.ndarray,
    window_length: int = 7,
    polyorder: int = 2,
) -> np.ndarray:
    """Apply bidirectional Savitzky-Golay smoothing filter.

    Forward + backward pass averaged to eliminate phase shift.
    Falls back to a copy when scipy is unavailable or the sequence is too short.
    """
    return _bidirectional_savgol(values, window_length=window_length, polyorder=polyorder)


def _condition_sparse_channel(
    values: np.ndarray,
    *,
    hampel_hw: int = 3,
    hampel_sigma: float = 3.0,
) -> np.ndarray:
    """Pre-interpolation conditioning: multi-scale Hampel outlier removal on sparse data."""
    return _hampel_filter_1d(values, half_window=hampel_hw, n_sigma=hampel_sigma)


def _condition_dense_channel(
    values: np.ndarray,
    *,
    sg_window: int = 7,
    sg_polyorder: int = 2,
) -> np.ndarray:
    """Post-interpolation conditioning: bidirectional S-G smoothing on dense timeline."""
    return _savgol_smooth_1d(values, window_length=sg_window, polyorder=sg_polyorder)


def _interp_channel(
    t_known: np.ndarray,
    v_known: np.ndarray,
    t_all: np.ndarray,
    method: str,
) -> np.ndarray:
    """Interpolate a single channel over the full timeline.

    Parameters
    ----------
    t_known : 1-D float array of known time points (frame indices).
    v_known : 1-D float array of channel values at *t_known*.
    t_all   : 1-D float array of all time points to evaluate (0..T-1).
    method  : ``"linear"`` or ``"cubic"``.
    """
    if len(t_known) == 0:
        return np.zeros(len(t_all), dtype=np.float64)
    if len(t_known) == 1:
        return np.full(len(t_all), v_known[0], dtype=np.float64)
    if method == "cubic" and _SCIPY_INTERP_OK and len(t_known) >= 4:
        try:
            from scipy.interpolate import CubicSpline
            cs = CubicSpline(t_known, v_known, extrapolate=True)
            result = cs(t_all)
            # Clamp extrapolated boundary values to the range of known data
            # to prevent cubic polynomial divergence outside the knot span.
            v_min, v_max = float(v_known.min()), float(v_known.max())
            np.clip(result, v_min, v_max, out=result)
            return result
        except Exception:
            pass
    # Linear interpolation (numpy; extrapolates flat at boundaries)
    return np.interp(t_all, t_known, v_known)


def _extract_ts_geo_channels(
    samples: Sequence[FramePrediction],
    frame_w: float = 640.0,
    frame_h: float = 480.0,
    interp_method: str = "cubic",
) -> Tuple[np.ndarray, np.ndarray]:
    """Build ``(N_GEO_VARS, N_TS_STEPS)`` geometric channel matrix.

    Unlike the previous zero-padding approach, this function preserves the
    **true temporal structure** of sparse annotations:

    1. Each known sample is placed at its real frame index in a full-length
       timeline of length ``T = max_frame_index + 1``.
    2. Gaps are filled by per-channel cubic spline interpolation (C2 continuity).
    3. Speed (channel 8) is *re-derived* from the interpolated center
       positions rather than interpolated directly, giving physically
       correct velocity estimates.
    4. The complete T-step trajectory is uniformly resampled to N_TS_STEPS.

    Returns
    -------
    geo : ndarray, shape ``(N_GEO_VARS, N_TS_STEPS)``
    resample_frame_idx : ndarray of int, shape ``(N_TS_STEPS,)``
        Actual **video frame indices** corresponding to each of the 128
        resampled steps.  Used by ``_extract_resnet_for_ts`` to seek the
        right frames in the video file.
    """
    n_raw = len(samples)
    if n_raw == 0:
        return (
            np.zeros((N_GEO_VARS, N_TS_STEPS), dtype=np.float32),
            np.zeros(N_TS_STEPS, dtype=int),
        )

    # ── Sort samples by their actual frame index ──────────────────────────
    samples = sorted(samples, key=lambda s: s.frame_index)
    frame_indices = np.array([s.frame_index for s in samples], dtype=np.int64)

    # ── Raw channel sources from known samples ─────────────────────────────
    bboxes = np.array([s.bbox for s in samples], dtype=np.float64).reshape(-1, 4)

    seg_areas = np.array(
        [
            s.segmentation.stats.area_px
            if (s.segmentation and s.segmentation.stats)
            else 0.0
            for s in samples
        ],
        dtype=np.float64,
    )

    eq_diams = np.array(
        [
            s.segmentation.stats.equivalent_diameter_px
            if (s.segmentation and s.segmentation.stats)
            else 0.0
            for s in samples
        ],
        dtype=np.float64,
    )

    # ── Build sparse (n_raw,) channel vectors at known frame positions ─────
    # Channels 0-7, 9 are filled here; channel 8 (speed) recomputed later.
    SPEED_CH = 8
    sparse: np.ndarray = np.zeros((n_raw, N_GEO_VARS), dtype=np.float64)
    for i, s in enumerate(samples):
        x, y, bw, bh = s.bbox
        sparse[i, 0] = float(x + bw / 2.0)
        sparse[i, 1] = float(y + bh / 2.0)
        sparse[i, 2] = float(bw)
        sparse[i, 3] = float(bh)
        sparse[i, 4] = float(bw * bh)
        sparse[i, 5] = float(seg_areas[i])
        sparse[i, 6] = float(
            _circularity(
                s.segmentation.stats.area_px,
                s.segmentation.stats.perimeter_px,
            )
            if (s.segmentation and s.segmentation.stats)
            else 0.0
        )
        sparse[i, 7] = float(eq_diams[i])
        # ch 8 intentionally skipped – computed after interpolation
        sparse[i, 9] = float(bw / (bh + 1e-9))

    # ── Build full-length timeline and interpolate ─────────────────────────
    T = int(frame_indices[-1]) + 1
    t_all = np.arange(T, dtype=np.float64)
    t_known = frame_indices.astype(np.float64)

    # ── Pre-interpolation: Hampel filter on sparse known values ──────────
    # Channels 0-4, 9 (cx, cy, bw, bh, area, flatness) are already smoothed
    # by upstream trajectory_filter (Hampel+S-G for test, S-G for GT).
    # Re-applying Hampel would cause double-smoothing and over-attenuation
    # of legitimate high-frequency dynamics.  Only seg-derived channels
    # (circularity, eq_diam) genuinely benefit from outlier removal here.
    _HAMPEL_CHANNELS = {6, 7}  # circularity, eq_diam only (seg-derived, not pre-smoothed)
    for ch in _HAMPEL_CHANNELS:
        if ch < N_GEO_VARS and ch != SPEED_CH:
            sparse[:, ch] = _condition_sparse_channel(sparse[:, ch])

    timeline = np.zeros((T, N_GEO_VARS), dtype=np.float64)
    for ch in range(N_GEO_VARS):
        if ch == SPEED_CH:
            continue  # derived later
        timeline[:, ch] = _interp_channel(
            t_known, sparse[:, ch], t_all, interp_method
        )

    # ── Post-interpolation: Savitzky-Golay smoothing on dense timeline ───
    # Only seg-derived channels need post-interpolation smoothing.
    # Positional/size channels (0-4, 9) are already smoothed upstream.
    _SG_CHANNELS = {5, 6, 7}  # seg_area, circularity, eq_diam
    for ch in _SG_CHANNELS:
        if ch < N_GEO_VARS and ch != SPEED_CH:
            timeline[:, ch] = _condition_dense_channel(timeline[:, ch])

    # ── Re-derive speed from interpolated centre positions ─────────────────
    cx_full = timeline[:, 0]
    cy_full = timeline[:, 1]
    speeds_full = np.zeros(T, dtype=np.float64)
    if T > 1:
        step_dist = np.sqrt(np.diff(cx_full) ** 2 + np.diff(cy_full) ** 2)
        speeds_full[1:] = step_dist
    timeline[:, SPEED_CH] = speeds_full

    # ── Guard against NaN / Inf from interpolation or arithmetic ──────────
    np.nan_to_num(timeline, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Uniform resample from T steps → N_TS_STEPS ────────────────────────
    resample_frame_idx = np.round(
        np.linspace(0, T - 1, N_TS_STEPS)
    ).astype(int)
    resample_frame_idx = np.clip(resample_frame_idx, 0, T - 1)

    geo = timeline[resample_frame_idx, :].T.astype(np.float32)  # (N_GEO_VARS, N_TS_STEPS)

    return geo, resample_frame_idx


# ═════════════════════════════════════════════════════════════════════════════
# Shared PCA helpers
# ═════════════════════════════════════════════════════════════════════════════


def _pca_fit_transform(
    X: np.ndarray, target_dim: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit PCA and transform *X*.

    Returns ``(reduced, mean, components)``.
    """
    n, d = X.shape
    if d <= target_dim:
        pad_w = target_dim - d
        reduced = (
            np.hstack([X, np.zeros((n, pad_w))]) if pad_w > 0 else X.copy()
        )
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
    X: np.ndarray, mean: np.ndarray, components: np.ndarray, target_dim: int
) -> np.ndarray:
    """Apply pre-fitted PCA."""
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


# ═════════════════════════════════════════════════════════════════════════════
# Singleton ResNet extractor (lazy, shared between both FE classes)
# ═════════════════════════════════════════════════════════════════════════════

_resnet_extractor: Optional[MaskedROIResNetExtractor] = None


def _get_resnet_extractor(device: str = "auto") -> MaskedROIResNetExtractor:
    """Return (and lazily create) a module-level ResNet extractor singleton."""
    global _resnet_extractor
    if _resnet_extractor is None:
        _resnet_extractor = MaskedROIResNetExtractor(device=device)
    return _resnet_extractor


# ═════════════════════════════════════════════════════════════════════════════
# Extractor 1: MotionTextureStatic (Non-TSC / Tabular)
# ═════════════════════════════════════════════════════════════════════════════


@register_feature_extractor("motion_texture_static")
class MotionTextureStaticFeatureExtractor(TrajectoryFeatureExtractor):
    """Motion + CTS-Static + ResNet Texture (PCA) for tabular classifiers.

    Feature structure (video level)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - **Motion**  : 31 D — Hampel+S-G smoothed trajectory features
    - **Static**  : 35 D — CTS diagnostic features (CSA / circularity /
      flattening / echo / nerve displacement)
    - **Texture** : 66 D — ResNet-18 embeddings (first + last frame masked
      ROI, 2 × 512 = 1 024 raw → PCA to 66 D = dim(motion) + dim(static))

    Total = 31 + 35 + 66 = **132 D** per video.

    ``finalize_batch`` fits PCA on the training set and applies it on the
    test set (called by the engine).
    """

    name = "MotionTextureStaticFeatures"
    DEFAULT_CONFIG: Dict[str, Any] = {
        "resnet_device": "auto",
        "aggregate_stats": ["mean", "std", "min", "max"],
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        cfg = params or {}
        # Legacy Kalman params accepted but ignored
        self._resnet_device = str(cfg.get("resnet_device", "auto"))
        self._subject_stats = list(
            cfg.get("aggregate_stats", ["mean", "std", "min", "max"])
        )

        self._motion_keys = list(MOTION_FEATURE_KEYS)          # 31
        self._static_keys = list(CTS_STATIC_FEATURE_KEYS)      # 35
        self._tex_target = (
            len(self._motion_keys) + len(self._static_keys)
        )  # 66
        self._texture_keys = [
            f"mts_tex_pca_{i:03d}" for i in range(self._tex_target)
        ]

        self._video_keys = (
            self._motion_keys + self._static_keys + self._texture_keys
        )
        self._subject_keys = _build_subject_keys(
            self._video_keys, self._subject_stats
        )

        # PCA state (fitted in finalize_batch)
        self._pca_mean: Optional[np.ndarray] = None
        self._pca_components: Optional[np.ndarray] = None
        self._feat_mean: Optional[np.ndarray] = None
        self._feat_std: Optional[np.ndarray] = None

    def feature_order(self, level: str = "video") -> Sequence[str]:
        return (
            self._subject_keys
            if str(level).lower() == "subject"
            else self._video_keys
        )

    def extract_video(
        self,
        samples: Sequence[FramePrediction],
        video_path: Optional[str] = None,
    ) -> Dict[str, float]:
        motion = _compute_motion_features(samples)
        static = _compute_cts_static_features(samples, video_path)

        # Raw ResNet texture: first + last frame → 2 × 512 = 1024
        raw_tex = self._extract_resnet_texture(samples, video_path)

        combined = OrderedDict()
        for k in self._motion_keys:
            combined[k] = motion.get(k, 0.0)
        for k in self._static_keys:
            combined[k] = static.get(k, 0.0)
        for k in self._texture_keys:
            combined[k] = 0.0  # placeholder; filled by finalize_batch
        combined["_raw_texture_vec"] = raw_tex  # metadata
        return combined

    def aggregate_subject(
        self, video_features: Sequence[Dict[str, float]]
    ) -> Dict[str, float]:
        clean = [
            {k: v for k, v in vf.items() if not k.startswith("_")}
            for vf in video_features
        ]
        return _aggregate_video_features(
            clean, self._video_keys, self._subject_stats
        )

    def get_pca_state(self) -> Optional[Dict[str, np.ndarray]]:
        """Return PCA state for serialisation (global pre-fitting)."""
        if self._pca_mean is None or self._pca_components is None:
            return None
        return {"mean": self._pca_mean, "components": self._pca_components}

    def set_pca_state(self, state: Dict[str, np.ndarray]) -> None:
        """Restore PCA state from a previously saved global fit."""
        self._pca_mean = state["mean"]
        self._pca_components = state["components"]

    def finalize_batch(
        self,
        features_list: Sequence[Dict[str, float]],
        *,
        fit: bool = True,
    ) -> Sequence[Dict[str, float]]:
        """Batch PCA on ResNet texture features.

        When ``fit=True`` (training set), PCA is fitted and the state is
        stored.  When ``fit=False`` (test/validation), only
        ``pca.transform()`` is used — **PCA parameters are never re-fitted
        on non-training data**.  If PCA state is missing at transform time
        a ``RuntimeError`` is raised instead of silently re-fitting.
        """
        raw_vecs, indices = [], []
        for i, feat in enumerate(features_list):
            rv = feat.get("_raw_texture_vec")
            if rv is not None:
                raw_vecs.append(rv)
                indices.append(i)
        if not raw_vecs:
            return features_list

        X_raw = np.array(raw_vecs, dtype=np.float64)
        target = self._tex_target

        if fit:
            reduced, self._pca_mean, self._pca_components = _pca_fit_transform(
                X_raw, target
            )
        else:
            if self._pca_mean is None or self._pca_components is None:
                raise RuntimeError(
                    "PCA state is not available.  finalize_batch(fit=True) must "
                    "be called on the training set before finalize_batch(fit=False) "
                    "is called on the test/validation set."
                )
            reduced = _pca_transform(
                X_raw, self._pca_mean, self._pca_components, target
            )

        result = list(features_list)
        for idx, row_idx in enumerate(indices):
            feat = dict(result[row_idx])
            for j, k in enumerate(self._texture_keys):
                feat[k] = float(reduced[idx, j]) if j < reduced.shape[1] else 0.0
            feat.pop("_raw_texture_vec", None)
            result[row_idx] = feat
        result, self._feat_mean, self._feat_std = _global_standardize_feature_dicts(
            result,
            self._video_keys,
            fit=fit,
            mean=self._feat_mean,
            std=self._feat_std,
        )
        return result

    # ── Internal helpers ─────────────────────────────────────────────────

    def _extract_resnet_texture(
        self,
        samples: Sequence[FramePrediction],
        video_path: Optional[str],
    ) -> List[float]:
        """Extract ResNet features for first + last frame → flat list (1024)."""
        raw_dim = RESNET_FEAT_DIM * 2  # 512 × 2
        zeros = [0.0] * raw_dim
        if not samples or not video_path:
            return zeros

        resnet = _get_resnet_extractor(self._resnet_device)
        if not resnet.available:
            logger.warning(
                "ResNet unavailable; returning zero texture for %s", video_path
            )
            return zeros

        key_samples = [samples[0]]
        if len(samples) > 1:
            key_samples.append(samples[-1])
        else:
            key_samples.append(samples[0])

        feats = resnet.extract_from_video(video_path, key_samples)  # (2, 512)
        return feats.flatten().tolist()


# ═════════════════════════════════════════════════════════════════════════════
# Extractor 2: TimeSeries (TSC — MultiRocket / PatchTST / TimeMachine)
# ═════════════════════════════════════════════════════════════════════════════


@register_feature_extractor("time_series")
class TimeSeriesFeatureExtractor(TrajectoryFeatureExtractor):
    """Per-frame multivariate time-series features with ResNet texture channels.

    Channel layout (18 total)
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    ===========  ========  ==========================================
    Index range  Count     Description
    ===========  ========  ==========================================
    0 – 9        10        Geometric / kinematic (cx, cy, bw, bh,
                           area, seg_area, circularity, eq_diam,
                           speed, flatness)
    10 – 17      8         ResNet-18 texture PCA (per-frame masked
                           ROI → 512 D → PCA to 8 D)
    ===========  ========  ==========================================

    Time-series length = 128 (padded / resampled).
    Flat feature vector = 18 × 128 = **2 304 D**.

    ``finalize_batch`` fits PCA across all *training-set* frames to reduce
    ResNet 512 D → 8 D per timestep, then fills channels 10–17.

    Subject aggregation: element-wise mean across the subject's videos.
    """

    name = "TimeSeriesFeatures"
    DEFAULT_CONFIG: Dict[str, Any] = {
        "frame_w": 640.0,
        "frame_h": 480.0,
        "resnet_device": "auto",
        "tex_pca_dim": N_TEX_PCA_TS,
        # Temporal interpolation method for sparse annotations.
        # "cubic" (default, C2 continuity) or "linear" (fallback).
        "interp_method": "cubic",
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        cfg = params or {}
        self._frame_w = float(cfg.get("frame_w", 640.0))
        self._frame_h = float(cfg.get("frame_h", 480.0))
        self._resnet_device = str(cfg.get("resnet_device", "auto"))
        self._tex_pca_dim = int(cfg.get("tex_pca_dim", N_TEX_PCA_TS))
        self._interp_method: str = str(cfg.get("interp_method", "cubic"))

        self._flat_len = N_TS_VARS * N_TS_STEPS
        self._video_keys = [f"ts_{i:04d}" for i in range(self._flat_len)]
        self._subject_keys = self._video_keys  # same after mean

        # PCA state for ResNet texture (fitted in finalize_batch)
        self._pca_mean: Optional[np.ndarray] = None
        self._pca_components: Optional[np.ndarray] = None
        # Global channel-wise scaler state (fitted on training set only)
        self._channel_mean: Optional[np.ndarray] = None
        self._channel_std: Optional[np.ndarray] = None

    def feature_order(self, level: str = "video") -> Sequence[str]:
        return self._video_keys

    def extract_video(
        self,
        samples: Sequence[FramePrediction],
        video_path: Optional[str] = None,
    ) -> Dict[str, float]:
        # 1. Geometric channels → (N_GEO_VARS, N_TS_STEPS)
        geo, resample_idx = _extract_ts_geo_channels(
            samples, self._frame_w, self._frame_h, self._interp_method
        )

        # 2. Raw ResNet texture for the resampled frames
        raw_resnet = self._extract_resnet_for_ts(
            samples, resample_idx, video_path
        )  # (N_TS_STEPS, RESNET_FEAT_DIM)

        # 3. Assemble full time series with placeholder texture channels
        ts = np.zeros((N_TS_VARS, N_TS_STEPS), dtype=np.float32)
        ts[:N_GEO_VARS, :] = geo
        # Channels N_GEO_VARS .. N_TS_VARS-1 left as zeros → finalize_batch

        flat = ts.flatten(order="C")  # (N_TS_VARS × N_TS_STEPS,)
        feat: Dict[str, float] = OrderedDict()
        for i, v in enumerate(flat):
            feat[f"ts_{i:04d}"] = float(v)

        # Metadata (removed by finalize_batch)
        feat["_ts_n_vars"] = float(N_TS_VARS)
        feat["_ts_n_timesteps"] = float(N_TS_STEPS)
        feat["_raw_resnet_ts"] = raw_resnet  # (N_TS_STEPS, 512) ndarray
        return feat

    def aggregate_subject(
        self, video_features: Sequence[Dict[str, float]]
    ) -> Dict[str, float]:
        """Element-wise mean across a subject's videos."""
        clean = [
            {k: v for k, v in vf.items() if not k.startswith("_")}
            for vf in video_features
        ]
        if not clean:
            return OrderedDict((k, 0.0) for k in self._subject_keys)
        mat = np.array(
            [[vf.get(k, 0.0) for k in self._video_keys] for vf in clean],
            dtype=np.float64,
        )
        mean_vec = mat.mean(axis=0)
        return OrderedDict(
            (k, float(v)) for k, v in zip(self._video_keys, mean_vec)
        )

    def get_pca_state(self) -> Optional[Dict[str, np.ndarray]]:
        """Return PCA state for serialisation (global pre-fitting)."""
        if self._pca_mean is None or self._pca_components is None:
            return None
        return {"mean": self._pca_mean, "components": self._pca_components}

    def set_pca_state(self, state: Dict[str, np.ndarray]) -> None:
        """Restore PCA state from a previously saved global fit."""
        self._pca_mean = state["mean"]
        self._pca_components = state["components"]

    def finalize_batch(
        self,
        features_list: Sequence[Dict[str, float]],
        *,
        fit: bool = True,
    ) -> Sequence[Dict[str, float]]:
        """Batch PCA on per-frame ResNet texture → fill channels 10–17.

        When ``fit=True`` (training set), PCA is fitted globally across all
        training frames and the state is stored.  When ``fit=False``
        (test/validation), only ``pca.transform()`` is used — **PCA
        parameters are never re-fitted on non-training data**.
        """
        raw_mats, indices = [], []
        for i, feat in enumerate(features_list):
            rm = feat.get("_raw_resnet_ts")
            if rm is not None:
                raw_mats.append(np.asarray(rm, dtype=np.float64))
                indices.append(i)
        if not raw_mats:
            # Still apply (or fit+apply) channel-wise global scaling.
            return self._apply_global_channel_zscore(features_list, fit=fit)

        # Stack all frames across all videos
        all_frames = np.vstack(raw_mats)  # (total_frames, 512)
        target = self._tex_pca_dim

        if fit:
            reduced_all, self._pca_mean, self._pca_components = (
                _pca_fit_transform(all_frames, target)
            )
        else:
            if self._pca_mean is None or self._pca_components is None:
                raise RuntimeError(
                    "PCA state is not available.  finalize_batch(fit=True) must "
                    "be called on the training set before finalize_batch(fit=False) "
                    "is called on the test/validation set."
                )
            reduced_all = _pca_transform(
                all_frames, self._pca_mean, self._pca_components, target
            )

        # Split back per video and inject into flat TS dict
        result = list(features_list)
        offset = 0
        for idx, row_idx in enumerate(indices):
            n_frames = raw_mats[idx].shape[0]
            reduced_video = reduced_all[offset : offset + n_frames]
            offset += n_frames

            feat = dict(result[row_idx])

            # Reconstruct time-series array
            ts = np.array(
                [feat.get(f"ts_{i:04d}", 0.0) for i in range(self._flat_len)],
                dtype=np.float32,
            ).reshape(N_TS_VARS, N_TS_STEPS)

            # Write PCA texture into channels N_GEO_VARS .. N_GEO_VARS+target-1
            n_fill = min(reduced_video.shape[0], N_TS_STEPS)
            for ch in range(target):
                ch_idx = N_GEO_VARS + ch
                if ch_idx < N_TS_VARS:
                    ts[ch_idx, :n_fill] = reduced_video[:n_fill, ch].astype(
                        np.float32
                    )

            # Re-flatten
            flat = ts.flatten(order="C")
            for i, v in enumerate(flat):
                feat[f"ts_{i:04d}"] = float(v)

            feat.pop("_raw_resnet_ts", None)
            result[row_idx] = feat

        # Channel-wise global scaling on full TS matrix (geo + texture channels)
        return self._apply_global_channel_zscore(result, fit=fit)

    def _apply_global_channel_zscore(
        self,
        features_list: Sequence[Dict[str, float]],
        *,
        fit: bool,
    ) -> Sequence[Dict[str, float]]:
        if not features_list:
            return []

        mats = np.zeros((len(features_list), N_TS_VARS, N_TS_STEPS), dtype=np.float64)
        for i, feat in enumerate(features_list):
            vec = np.array(
                [feat.get(f"ts_{j:04d}", 0.0) for j in range(self._flat_len)],
                dtype=np.float64,
            )
            mats[i] = vec.reshape(N_TS_VARS, N_TS_STEPS)

        if fit:
            self._channel_mean = mats.mean(axis=(0, 2))
            self._channel_std = mats.std(axis=(0, 2))
            self._channel_std = np.where(self._channel_std < 1e-9, 1.0, self._channel_std)
        else:
            if self._channel_mean is None or self._channel_std is None:
                raise RuntimeError(
                    "Global channel scaler is not available. finalize_batch(fit=True) "
                    "must be called on the training set before finalize_batch(fit=False)."
                )

        mean = self._channel_mean.reshape(1, N_TS_VARS, 1)
        std = self._channel_std.reshape(1, N_TS_VARS, 1)
        mats = (mats - mean) / std
        np.nan_to_num(mats, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        out = []
        for i, feat in enumerate(features_list):
            new_feat = dict(feat)
            flat = mats[i].reshape(-1)
            for j, v in enumerate(flat):
                new_feat[f"ts_{j:04d}"] = float(v)
            out.append(new_feat)
        return out

    # ── Internal helpers ─────────────────────────────────────────────────

    def _extract_resnet_for_ts(
        self,
        samples: Sequence[FramePrediction],
        resample_frame_idx: np.ndarray,
        video_path: Optional[str],
    ) -> np.ndarray:
        """Extract ResNet features for the resampled frame indices.

        ``resample_frame_idx`` now contains **actual video frame indices**
        (not positions into the *samples* list).  For each resampled frame we
        locate the nearest annotated sample to obtain a valid crop bbox.

        Returns ``(N_TS_STEPS, RESNET_FEAT_DIM)`` float32.
        """
        result = np.zeros((N_TS_STEPS, RESNET_FEAT_DIM), dtype=np.float32)
        if not video_path or not samples or len(resample_frame_idx) == 0:
            return result

        resnet = _get_resnet_extractor(self._resnet_device)
        if not resnet.available:
            return result

        # Build a sorted array of known frame indices for nearest-neighbour lookup
        known_frame_indices = np.array(
            [s.frame_index for s in samples], dtype=np.int64
        )
        sorted_order = np.argsort(known_frame_indices)
        known_sorted = known_frame_indices[sorted_order]
        samples_sorted = [samples[i] for i in sorted_order]

        # For each resampled frame, find the nearest annotated sample
        # (closest by frame index) to use as the crop/mask reference.
        selected: List[FramePrediction] = []
        for target_fi in resample_frame_idx:
            pos = int(np.searchsorted(known_sorted, target_fi))
            # Clamp to valid range
            pos = min(pos, len(known_sorted) - 1)
            if pos > 0:
                # Pick whichever neighbour (left or right) is closer
                left_dist = abs(int(known_sorted[pos - 1]) - int(target_fi))
                right_dist = abs(int(known_sorted[pos]) - int(target_fi))
                nearest_sample = samples_sorted[pos - 1] if left_dist <= right_dist else samples_sorted[pos]
            else:
                nearest_sample = samples_sorted[0]

            # Override frame_index so the video capture seeks the correct frame
            from ..core.interfaces import FramePrediction as _FP
            proxy = _FP(
                frame_index=int(target_fi),
                bbox=nearest_sample.bbox,
                score=nearest_sample.score,
                segmentation=nearest_sample.segmentation,
            )
            selected.append(proxy)

        feats = resnet.extract_from_video(video_path, selected)  # (N_TS_STEPS, 512)

        n_actual = min(feats.shape[0], N_TS_STEPS)
        result[:n_actual] = feats[:n_actual]
        return result


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                          V2 FEATURE EXTRACTORS                             ║
# ║                                                                            ║
# ║  Improvements over V1:                                                     ║
# ║  • TSC: positional channels store *displacement relative to median*        ║
# ║    (after Hampel + S-G filtering) instead of absolute normalised coords.   ║
# ║  • TSC: 6 additional kinematic channels — acceleration, heading (sin/cos), ║
# ║    rate-of-area-change, radial distance, curvature.                        ║
# ║  • Non-TSC: 24 additional statistical/gradient/displacement features       ║
# ║    (percentiles, IQR, skewness, kurtosis, gradient stats, displacement     ║
# ║    from median, path efficiency).                                          ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

# ═════════════════════════════════════════════════════════════════════════════
# V2 Constants
# ═════════════════════════════════════════════════════════════════════════════

N_GEO_VARS_V2: int = 16         # 10 original + 6 derivative channels
N_TEX_PCA_TS_V2: int = 8        # Same ResNet PCA dimension
N_TS_VARS_V2: int = N_GEO_VARS_V2 + N_TEX_PCA_TS_V2   # 24 total
N_TS_STEPS_V2: int = N_TS_STEPS  # Same 128 steps

# ═════════════════════════════════════════════════════════════════════════════
# V2 Time-Series Channel Names (24 channels)
# ═════════════════════════════════════════════════════════════════════════════

TS_GEO_CHANNEL_KEYS_V2: List[str] = [
    # ── Displacement from median (instead of absolute coords) ──
    "ts_dx",            # centroid x displacement from median (normalised)
    "ts_dy",            # centroid y displacement from median (normalised)
    # ── Size / shape (same as V1) ──
    "ts_bw",            # bbox width  (normalised)
    "ts_bh",            # bbox height (normalised)
    "ts_area",          # bbox area   (normalised)
    "ts_seg_area",      # seg mask area (normalised)
    "ts_circularity",   # seg circularity
    "ts_eq_diam",       # equivalent diameter (normalised)
    # ── Kinematics (V1) ──
    "ts_speed",         # inter-frame centroid speed (normalised)
    "ts_flat",          # flattening index w / h
    # ── NEW derivative channels (V2) ──
    "ts_accel",         # acceleration magnitude  |d(speed)/dt|
    "ts_heading_sin",   # sin(heading angle) of velocity vector
    "ts_heading_cos",   # cos(heading angle) of velocity vector
    "ts_d_area",        # rate of bbox area change  d(area)/dt
    "ts_radial_dist",   # distance from median position sqrt(dx²+dy²)
    "ts_curvature",     # trajectory curvature (cross-product proxy)
]

TS_TEX_CHANNEL_KEYS_V2: List[str] = [
    f"ts_tex_pca_{i}" for i in range(N_TEX_PCA_TS_V2)
]

TS_CHANNEL_KEYS_V2: List[str] = TS_GEO_CHANNEL_KEYS_V2 + TS_TEX_CHANNEL_KEYS_V2

# ═════════════════════════════════════════════════════════════════════════════
# V2 CTS Static Feature Keys (59 D = 35 original + 24 new)
# ═════════════════════════════════════════════════════════════════════════════

CTS_STATIC_V2_EXTRA_KEYS: List[str] = [
    # ── CSA percentiles / higher-order moments ──
    "cts_csa_q25",
    "cts_csa_q75",
    "cts_csa_iqr",
    "cts_csa_skew",
    "cts_csa_kurt",
    # ── Circularity percentiles ──
    "cts_circ_q25",
    "cts_circ_q75",
    "cts_circ_iqr",
    # ── Equivalent diameter percentiles ──
    "cts_eq_diam_q25",
    "cts_eq_diam_q75",
    "cts_eq_diam_iqr",
    # ── Flattening percentiles ──
    "cts_flat_q25",
    "cts_flat_q75",
    "cts_flat_iqr",
    # ── Temporal gradient statistics (first differences) ──
    "cts_csa_grad_mean",
    "cts_csa_grad_std",
    "cts_flat_grad_mean",
    "cts_flat_grad_std",
    # ── Displacement from median position ──
    "cts_disp_median_mean",
    "cts_disp_median_max",
    "cts_temporal_energy",
    "cts_radial_range",
    # ── Heading / efficiency ──
    "cts_mean_angular_change",
    "cts_path_efficiency",
]

CTS_STATIC_V2_FEATURE_KEYS: List[str] = (
    CTS_STATIC_FEATURE_KEYS + CTS_STATIC_V2_EXTRA_KEYS
)


# ═════════════════════════════════════════════════════════════════════════════
# V2 helpers
# ═════════════════════════════════════════════════════════════════════════════


def _safe_skew(arr: np.ndarray) -> float:
    """Compute skewness; return 0 for degenerate arrays."""
    if arr.size < 3 or arr.std() < 1e-12:
        return 0.0
    m = arr.mean()
    s = arr.std()
    return float(np.mean(((arr - m) / s) ** 3))


def _safe_kurtosis(arr: np.ndarray) -> float:
    """Compute excess kurtosis; return 0 for degenerate arrays."""
    if arr.size < 4 or arr.std() < 1e-12:
        return 0.0
    m = arr.mean()
    s = arr.std()
    return float(np.mean(((arr - m) / s) ** 4) - 3.0)


def _compute_cts_static_features_v2(
    samples: Sequence[FramePrediction],
    video_path: Optional[str] = None,
) -> Dict[str, float]:
    """Compute V2 CTS static features (59 D).

    Includes all 35 V1 features plus 24 additional features:
    percentiles, IQR, skewness/kurtosis, gradient stats,
    displacement-from-median, and path efficiency.
    """
    # Start with all V1 features
    feat = _compute_cts_static_features(samples, video_path)

    zeros_extra = OrderedDict((k, 0.0) for k in CTS_STATIC_V2_EXTRA_KEYS)
    if not samples:
        feat.update(zeros_extra)
        return feat

    # ── Recompute raw arrays for percentile/gradient analysis ─────────
    usable = [s for s in samples if s.segmentation and s.segmentation.stats
              and getattr(s.segmentation.stats, 'area_px', 0.0) > 0]

    if usable:
        areas = np.array(
            [s.segmentation.stats.area_px for s in usable], dtype=np.float64
        )
        circs = np.array(
            [
                _circularity(
                    s.segmentation.stats.area_px,
                    s.segmentation.stats.perimeter_px,
                )
                for s in usable
            ],
            dtype=np.float64,
        )
        eq_diam = np.array(
            [s.segmentation.stats.equivalent_diameter_px for s in usable],
            dtype=np.float64,
        )
    else:
        areas = circs = eq_diam = np.zeros(0)

    bboxes = np.array([s.bbox for s in samples], dtype=np.float64)
    flat = bboxes[:, 2] / (bboxes[:, 3] + 1e-9)

    # ── CSA percentiles / higher-order ──
    if areas.size >= 2:
        feat["cts_csa_q25"] = float(np.percentile(areas, 25))
        feat["cts_csa_q75"] = float(np.percentile(areas, 75))
        feat["cts_csa_iqr"] = feat["cts_csa_q75"] - feat["cts_csa_q25"]
        feat["cts_csa_skew"] = _safe_skew(areas)
        feat["cts_csa_kurt"] = _safe_kurtosis(areas)
    else:
        for k in ["cts_csa_q25", "cts_csa_q75", "cts_csa_iqr",
                   "cts_csa_skew", "cts_csa_kurt"]:
            feat[k] = 0.0

    # ── Circularity percentiles ──
    if circs.size >= 2:
        feat["cts_circ_q25"] = float(np.percentile(circs, 25))
        feat["cts_circ_q75"] = float(np.percentile(circs, 75))
        feat["cts_circ_iqr"] = feat["cts_circ_q75"] - feat["cts_circ_q25"]
    else:
        feat.update({"cts_circ_q25": 0.0, "cts_circ_q75": 0.0, "cts_circ_iqr": 0.0})

    # ── Eq diameter percentiles ──
    if eq_diam.size >= 2:
        feat["cts_eq_diam_q25"] = float(np.percentile(eq_diam, 25))
        feat["cts_eq_diam_q75"] = float(np.percentile(eq_diam, 75))
        feat["cts_eq_diam_iqr"] = feat["cts_eq_diam_q75"] - feat["cts_eq_diam_q25"]
    else:
        feat.update({"cts_eq_diam_q25": 0.0, "cts_eq_diam_q75": 0.0,
                      "cts_eq_diam_iqr": 0.0})

    # ── Flattening percentiles ──
    if flat.size >= 2:
        feat["cts_flat_q25"] = float(np.percentile(flat, 25))
        feat["cts_flat_q75"] = float(np.percentile(flat, 75))
        feat["cts_flat_iqr"] = feat["cts_flat_q75"] - feat["cts_flat_q25"]
    else:
        feat.update({"cts_flat_q25": 0.0, "cts_flat_q75": 0.0, "cts_flat_iqr": 0.0})

    # ── Temporal gradient statistics ──
    if areas.size >= 3:
        csa_grad = np.diff(areas)
        feat["cts_csa_grad_mean"] = float(csa_grad.mean())
        feat["cts_csa_grad_std"] = _safe_std(csa_grad)
    else:
        feat["cts_csa_grad_mean"] = 0.0
        feat["cts_csa_grad_std"] = 0.0

    if flat.size >= 3:
        flat_grad = np.diff(flat)
        feat["cts_flat_grad_mean"] = float(flat_grad.mean())
        feat["cts_flat_grad_std"] = _safe_std(flat_grad)
    else:
        feat["cts_flat_grad_mean"] = 0.0
        feat["cts_flat_grad_std"] = 0.0

    # ── Displacement from median position ──
    centers = np.array(
        [
            (s.bbox[0] + s.bbox[2] / 2.0, s.bbox[1] + s.bbox[3] / 2.0)
            for s in samples
        ],
        dtype=np.float64,
    )
    median_pos = np.median(centers, axis=0)
    displacements = np.linalg.norm(centers - median_pos, axis=1)
    feat["cts_disp_median_mean"] = float(displacements.mean())
    feat["cts_disp_median_max"] = float(displacements.max())
    feat["cts_temporal_energy"] = float(np.sum(displacements ** 2))
    feat["cts_radial_range"] = float(displacements.max() - displacements.min())

    # ── Heading / angular change ──
    if len(centers) >= 3:
        diffs = np.diff(centers, axis=0)
        angles = np.arctan2(diffs[:, 1], diffs[:, 0])
        angular_changes = np.abs(np.diff(angles))
        # Wrap to [0, π]
        angular_changes = np.minimum(angular_changes, 2.0 * np.pi - angular_changes)
        feat["cts_mean_angular_change"] = float(angular_changes.mean())
    else:
        feat["cts_mean_angular_change"] = 0.0

    path_len = feat.get("cts_nerve_path_len", 0.0)
    net_disp = feat.get("cts_nerve_net_disp", 0.0)
    feat["cts_path_efficiency"] = float(
        net_disp / (path_len + 1e-9)
    ) if path_len > 1e-9 else 0.0

    return feat


def _extract_ts_geo_channels_v2(
    samples: Sequence[FramePrediction],
    frame_w: float = 640.0,
    frame_h: float = 480.0,
    interp_method: str = "cubic",
) -> Tuple[np.ndarray, np.ndarray]:
    """Build ``(N_GEO_VARS_V2, N_TS_STEPS)`` V2 geometric channel matrix.

    **Key difference from V1**: positional channels 0–1 store *displacement
    from the trajectory median* (after Hampel + S-G conditioning) rather
    than absolute normalised coordinates.  Six additional derivative
    channels (acceleration, heading sin/cos, d_area/dt, radial distance,
    curvature) increase temporal information density.

    Returns
    -------
    geo : ndarray, shape ``(N_GEO_VARS_V2, N_TS_STEPS)``
    resample_frame_idx : ndarray of int, shape ``(N_TS_STEPS,)``
    """
    n_raw = len(samples)
    if n_raw == 0:
        return (
            np.zeros((N_GEO_VARS_V2, N_TS_STEPS_V2), dtype=np.float32),
            np.zeros(N_TS_STEPS_V2, dtype=int),
        )

    # ── Sort samples by their actual frame index ──
    samples = sorted(samples, key=lambda s: s.frame_index)
    frame_indices = np.array([s.frame_index for s in samples], dtype=np.int64)

    # ── Raw channel sources ──
    bboxes = np.array([s.bbox for s in samples], dtype=np.float64).reshape(-1, 4)

    seg_areas = np.array(
        [
            s.segmentation.stats.area_px
            if (s.segmentation and s.segmentation.stats)
            else 0.0
            for s in samples
        ],
        dtype=np.float64,
    )

    eq_diams = np.array(
        [
            s.segmentation.stats.equivalent_diameter_px
            if (s.segmentation and s.segmentation.stats)
            else 0.0
            for s in samples
        ],
        dtype=np.float64,
    )

    # ── Build sparse channels (same 10 base channels as V1) ──
    SPEED_CH = 8
    sparse: np.ndarray = np.zeros((n_raw, N_GEO_VARS), dtype=np.float64)
    for i, s in enumerate(samples):
        x, y, bw, bh = s.bbox
        sparse[i, 0] = float(x + bw / 2.0)    # cx raw px
        sparse[i, 1] = float(y + bh / 2.0)    # cy raw px
        sparse[i, 2] = float(bw)
        sparse[i, 3] = float(bh)
        sparse[i, 4] = float(bw * bh)
        sparse[i, 5] = float(seg_areas[i])
        sparse[i, 6] = float(
            _circularity(
                s.segmentation.stats.area_px,
                s.segmentation.stats.perimeter_px,
            )
            if (s.segmentation and s.segmentation.stats)
            else 0.0
        )
        sparse[i, 7] = float(eq_diams[i])
        sparse[i, 9] = float(bw / (bh + 1e-9))

    # ── Hampel on sparse data ──
    # Only seg-derived channels (not pre-smoothed by upstream trajectory_filter)
    _HAMPEL_CHS = {6, 7}  # circularity, eq_diam
    for ch in _HAMPEL_CHS:
        if ch < N_GEO_VARS and ch != SPEED_CH:
            sparse[:, ch] = _condition_sparse_channel(sparse[:, ch])

    # ── Interpolate to full timeline T ──
    T = int(frame_indices[-1]) + 1
    t_all = np.arange(T, dtype=np.float64)
    t_known = frame_indices.astype(np.float64)

    timeline = np.zeros((T, N_GEO_VARS), dtype=np.float64)
    for ch in range(N_GEO_VARS):
        if ch == SPEED_CH:
            continue
        timeline[:, ch] = _interp_channel(
            t_known, sparse[:, ch], t_all, interp_method
        )

    # ── Savitzky-Golay on dense timeline ──
    # Only seg-derived channels that weren't pre-smoothed upstream
    _SG_CHS = {5, 6, 7}  # seg_area, circularity, eq_diam
    for ch in _SG_CHS:
        if ch < N_GEO_VARS and ch != SPEED_CH:
            timeline[:, ch] = _condition_dense_channel(timeline[:, ch])

    # ── Compute speed from smoothed positions ──
    cx_full = timeline[:, 0]
    cy_full = timeline[:, 1]
    speeds_full = np.zeros(T, dtype=np.float64)
    if T > 1:
        dx = np.diff(cx_full)
        dy = np.diff(cy_full)
        step_dist = np.sqrt(dx ** 2 + dy ** 2)
        speeds_full[1:] = step_dist
    timeline[:, SPEED_CH] = speeds_full

    # ══════════════════════════════════════════════════════════════════════
    # V2: Convert position channels to displacement from median
    # ══════════════════════════════════════════════════════════════════════
    median_cx = float(np.median(timeline[:, 0]))
    median_cy = float(np.median(timeline[:, 1]))
    dx_from_median = timeline[:, 0] - median_cx  # displacement x
    dy_from_median = timeline[:, 1] - median_cy  # displacement y

    # ══════════════════════════════════════════════════════════════════════
    # V2: Derive additional kinematic channels
    # ══════════════════════════════════════════════════════════════════════

    # Acceleration = |d(speed)/dt|
    accel = np.zeros(T, dtype=np.float64)
    if T > 2:
        accel[1:] = np.abs(np.diff(speeds_full))

    # Heading (velocity direction) — sin and cos components
    heading_sin = np.zeros(T, dtype=np.float64)
    heading_cos = np.zeros(T, dtype=np.float64)
    if T > 1:
        vx = np.zeros(T, dtype=np.float64)
        vy = np.zeros(T, dtype=np.float64)
        vx[1:] = np.diff(cx_full)
        vy[1:] = np.diff(cy_full)
        vmag = np.sqrt(vx ** 2 + vy ** 2) + 1e-12
        heading_sin = vy / vmag
        heading_cos = vx / vmag

    # Rate of area change
    area_full = timeline[:, 4]  # raw area px^2
    d_area = np.zeros(T, dtype=np.float64)
    if T > 1:
        d_area[1:] = np.diff(area_full)

    # Radial distance from median
    radial_dist = np.sqrt(dx_from_median ** 2 + dy_from_median ** 2)

    # Curvature proxy: |cross(v_{t-1}, v_t)| / (|v_{t-1}| * |v_t|)
    curvature = np.zeros(T, dtype=np.float64)
    if T > 2:
        vx = np.zeros(T, dtype=np.float64)
        vy = np.zeros(T, dtype=np.float64)
        vx[1:] = np.diff(cx_full)
        vy[1:] = np.diff(cy_full)
        for t_idx in range(2, T):
            cross = vx[t_idx - 1] * vy[t_idx] - vy[t_idx - 1] * vx[t_idx]
            mag_prev = np.sqrt(vx[t_idx - 1] ** 2 + vy[t_idx - 1] ** 2)
            mag_curr = np.sqrt(vx[t_idx] ** 2 + vy[t_idx] ** 2)
            denom = mag_prev * mag_curr + 1e-12
            curvature[t_idx] = abs(cross) / denom

    # ══════════════════════════════════════════════════════════════════════
    # Assemble V2 geo matrix (16 channels × T)
    # ══════════════════════════════════════════════════════════════════════
    timeline_v2 = np.zeros((T, N_GEO_VARS_V2), dtype=np.float64)

    # Channels 0-1: displacement from median (NOT absolute position)
    timeline_v2[:, 0] = dx_from_median
    timeline_v2[:, 1] = dy_from_median
    # Channels 2-7: size/shape (same as V1)
    timeline_v2[:, 2] = timeline[:, 2]   # bw
    timeline_v2[:, 3] = timeline[:, 3]   # bh
    timeline_v2[:, 4] = timeline[:, 4]   # area
    timeline_v2[:, 5] = timeline[:, 5]   # seg_area
    timeline_v2[:, 6] = timeline[:, 6]   # circularity
    timeline_v2[:, 7] = timeline[:, 7]   # eq_diam
    # Channels 8-9: V1 kinematics
    timeline_v2[:, 8] = speeds_full      # speed
    timeline_v2[:, 9] = timeline[:, 9]   # flatness
    # Channels 10-15: V2 derivatives
    timeline_v2[:, 10] = accel
    timeline_v2[:, 11] = heading_sin
    timeline_v2[:, 12] = heading_cos
    timeline_v2[:, 13] = d_area
    timeline_v2[:, 14] = radial_dist
    timeline_v2[:, 15] = curvature

    # ── Guard against NaN / Inf from interpolation or arithmetic ──────────
    np.nan_to_num(timeline_v2, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Uniform resample T → N_TS_STEPS_V2 ──
    resample_frame_idx = np.round(
        np.linspace(0, T - 1, N_TS_STEPS_V2)
    ).astype(int)
    resample_frame_idx = np.clip(resample_frame_idx, 0, T - 1)

    geo = timeline_v2[resample_frame_idx, :].T.astype(np.float32)

    return geo, resample_frame_idx


# ═════════════════════════════════════════════════════════════════════════════
# Extractor 3 (V2): MotionTextureStaticV2 (Non-TSC / Tabular)
# ═════════════════════════════════════════════════════════════════════════════


@register_feature_extractor("motion_texture_static_v2")
class MotionTextureStaticV2FeatureExtractor(TrajectoryFeatureExtractor):
    """V2 motion + CTS-static + ResNet texture for tabular classifiers.

    Feature structure (video level)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - **Motion**  : 31 D — Hampel+S-G smoothed trajectory
    - **Static**  : 59 D — V1 (35 D) + 24 D (percentiles, IQR, skewness,
      kurtosis, gradient stats, displacement-from-median, path efficiency)
    - **Texture** : 90 D — ResNet-18 PCA (= dim(motion) + dim(static))

    Total = 31 + 59 + 90 = **180 D** per video.
    """

    name = "MotionTextureStaticFeaturesV2"
    DEFAULT_CONFIG: Dict[str, Any] = {
        "resnet_device": "auto",
        "aggregate_stats": ["mean", "std", "min", "max"],
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        cfg = params or {}
        # Legacy Kalman params accepted but ignored
        self._resnet_device = str(cfg.get("resnet_device", "auto"))
        self._subject_stats = list(
            cfg.get("aggregate_stats", ["mean", "std", "min", "max"])
        )

        self._motion_keys = list(MOTION_FEATURE_KEYS)               # 31
        self._static_keys = list(CTS_STATIC_V2_FEATURE_KEYS)        # 59
        self._tex_target = (
            len(self._motion_keys) + len(self._static_keys)
        )  # 90
        self._texture_keys = [
            f"mts_tex_pca_{i:03d}" for i in range(self._tex_target)
        ]

        self._video_keys = (
            self._motion_keys + self._static_keys + self._texture_keys
        )
        self._subject_keys = _build_subject_keys(
            self._video_keys, self._subject_stats
        )

        # PCA state
        self._pca_mean: Optional[np.ndarray] = None
        self._pca_components: Optional[np.ndarray] = None
        self._feat_mean: Optional[np.ndarray] = None
        self._feat_std: Optional[np.ndarray] = None

    def feature_order(self, level: str = "video") -> Sequence[str]:
        return (
            self._subject_keys
            if str(level).lower() == "subject"
            else self._video_keys
        )

    def extract_video(
        self,
        samples: Sequence[FramePrediction],
        video_path: Optional[str] = None,
    ) -> Dict[str, float]:
        motion = _compute_motion_features(samples)
        static = _compute_cts_static_features_v2(samples, video_path)

        raw_tex = self._extract_resnet_texture(samples, video_path)

        combined = OrderedDict()
        for k in self._motion_keys:
            combined[k] = motion.get(k, 0.0)
        for k in self._static_keys:
            combined[k] = static.get(k, 0.0)
        for k in self._texture_keys:
            combined[k] = 0.0
        combined["_raw_texture_vec"] = raw_tex
        return combined

    def aggregate_subject(
        self, video_features: Sequence[Dict[str, float]]
    ) -> Dict[str, float]:
        clean = [
            {k: v for k, v in vf.items() if not k.startswith("_")}
            for vf in video_features
        ]
        return _aggregate_video_features(
            clean, self._video_keys, self._subject_stats
        )

    def get_pca_state(self) -> Optional[Dict[str, np.ndarray]]:
        if self._pca_mean is None or self._pca_components is None:
            return None
        return {"mean": self._pca_mean, "components": self._pca_components}

    def set_pca_state(self, state: Dict[str, np.ndarray]) -> None:
        self._pca_mean = state["mean"]
        self._pca_components = state["components"]

    def finalize_batch(
        self,
        features_list: Sequence[Dict[str, float]],
        *,
        fit: bool = True,
    ) -> Sequence[Dict[str, float]]:
        raw_vecs, indices = [], []
        for i, feat in enumerate(features_list):
            rv = feat.get("_raw_texture_vec")
            if rv is not None:
                raw_vecs.append(rv)
                indices.append(i)
        if not raw_vecs:
            return features_list

        X_raw = np.array(raw_vecs, dtype=np.float64)
        target = self._tex_target

        if fit:
            reduced, self._pca_mean, self._pca_components = _pca_fit_transform(
                X_raw, target
            )
        else:
            if self._pca_mean is None or self._pca_components is None:
                raise RuntimeError(
                    "PCA state is not available.  finalize_batch(fit=True) must "
                    "be called on the training set before finalize_batch(fit=False) "
                    "is called on the test/validation set."
                )
            reduced = _pca_transform(
                X_raw, self._pca_mean, self._pca_components, target
            )

        result = list(features_list)
        for idx, row_idx in enumerate(indices):
            feat = dict(result[row_idx])
            for j, k in enumerate(self._texture_keys):
                feat[k] = float(reduced[idx, j]) if j < reduced.shape[1] else 0.0
            feat.pop("_raw_texture_vec", None)
            result[row_idx] = feat
        result, self._feat_mean, self._feat_std = _global_standardize_feature_dicts(
            result,
            self._video_keys,
            fit=fit,
            mean=self._feat_mean,
            std=self._feat_std,
        )
        return result

    # ── Internal ─────────────────────────────────────────────────────────

    def _extract_resnet_texture(
        self,
        samples: Sequence[FramePrediction],
        video_path: Optional[str],
    ) -> List[float]:
        raw_dim = RESNET_FEAT_DIM * 2
        zeros = [0.0] * raw_dim
        if not samples or not video_path:
            return zeros

        resnet = _get_resnet_extractor(self._resnet_device)
        if not resnet.available:
            logger.warning(
                "ResNet unavailable; returning zero texture for %s", video_path
            )
            return zeros

        key_samples = [samples[0]]
        if len(samples) > 1:
            key_samples.append(samples[-1])
        else:
            key_samples.append(samples[0])

        feats = resnet.extract_from_video(video_path, key_samples)
        return feats.flatten().tolist()


# ═════════════════════════════════════════════════════════════════════════════
# Extractor 4 (V2): TimeSeriesV2 (TSC)
# ═════════════════════════════════════════════════════════════════════════════


@register_feature_extractor("time_series_v2")
class TimeSeriesV2FeatureExtractor(TrajectoryFeatureExtractor):
    """V2 per-frame multivariate time-series features.

    Key differences from V1
    ~~~~~~~~~~~~~~~~~~~~~~~
    1. **Displacement from median** — channels 0–1 now record positional
       displacement relative to the trajectory median (after Hampel + S-G
       filtering) instead of absolute normalised coordinates.  This removes
       inter-subject positional bias and focuses TSC on *movement patterns*.
    2. **6 extra kinematic channels** — acceleration, heading (sin/cos),
       rate of area change, radial distance from median, curvature.

    Channel layout (24 total)
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    ===========  ========  ============================================
    Index range  Count     Description
    ===========  ========  ============================================
    0 – 1        2         Displacement from median (dx, dy)
    2 – 7        6         Size/shape (bw, bh, area, seg_area,
                           circularity, eq_diam)
    8 – 9        2         Kinematics V1 (speed, flatness)
    10 – 15      6         Kinematics V2 (accel, heading_sin,
                           heading_cos, d_area, radial_dist, curvature)
    16 – 23      8         ResNet-18 texture PCA
    ===========  ========  ============================================

    Time-series length = 128.
    Flat feature vector = 24 × 128 = **3 072 D**.
    """

    name = "TimeSeriesFeaturesV2"
    DEFAULT_CONFIG: Dict[str, Any] = {
        "frame_w": 640.0,
        "frame_h": 480.0,
        "resnet_device": "auto",
        "tex_pca_dim": N_TEX_PCA_TS_V2,
        "interp_method": "cubic",
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        cfg = params or {}
        self._frame_w = float(cfg.get("frame_w", 640.0))
        self._frame_h = float(cfg.get("frame_h", 480.0))
        self._resnet_device = str(cfg.get("resnet_device", "auto"))
        self._tex_pca_dim = int(cfg.get("tex_pca_dim", N_TEX_PCA_TS_V2))
        self._interp_method: str = str(cfg.get("interp_method", "cubic"))

        self._flat_len = N_TS_VARS_V2 * N_TS_STEPS_V2
        self._video_keys = [f"ts2_{i:04d}" for i in range(self._flat_len)]
        self._subject_keys = self._video_keys

        self._pca_mean: Optional[np.ndarray] = None
        self._pca_components: Optional[np.ndarray] = None

    def feature_order(self, level: str = "video") -> Sequence[str]:
        return self._video_keys

    def extract_video(
        self,
        samples: Sequence[FramePrediction],
        video_path: Optional[str] = None,
    ) -> Dict[str, float]:
        # 1. V2 geometric channels → (N_GEO_VARS_V2, N_TS_STEPS_V2)
        geo, resample_idx = _extract_ts_geo_channels_v2(
            samples, self._frame_w, self._frame_h, self._interp_method
        )

        # 2. Raw ResNet texture for the resampled frames (reuse V1 helper)
        raw_resnet = self._extract_resnet_for_ts(
            samples, resample_idx, video_path
        )

        # 3. Assemble full time series (24 channels)
        ts = np.zeros((N_TS_VARS_V2, N_TS_STEPS_V2), dtype=np.float32)
        ts[:N_GEO_VARS_V2, :] = geo
        # Channels N_GEO_VARS_V2 .. N_TS_VARS_V2-1 → finalize_batch fills

        flat = ts.flatten(order="C")
        feat: Dict[str, float] = OrderedDict()
        for i, v in enumerate(flat):
            feat[f"ts2_{i:04d}"] = float(v)

        feat["_ts_n_vars"] = float(N_TS_VARS_V2)
        feat["_ts_n_timesteps"] = float(N_TS_STEPS_V2)
        feat["_raw_resnet_ts"] = raw_resnet
        return feat

    def aggregate_subject(
        self, video_features: Sequence[Dict[str, float]]
    ) -> Dict[str, float]:
        clean = [
            {k: v for k, v in vf.items() if not k.startswith("_")}
            for vf in video_features
        ]
        if not clean:
            return OrderedDict((k, 0.0) for k in self._subject_keys)
        mat = np.array(
            [[vf.get(k, 0.0) for k in self._video_keys] for vf in clean],
            dtype=np.float64,
        )
        mean_vec = mat.mean(axis=0)
        return OrderedDict(
            (k, float(v)) for k, v in zip(self._video_keys, mean_vec)
        )

    def get_pca_state(self) -> Optional[Dict[str, np.ndarray]]:
        if self._pca_mean is None or self._pca_components is None:
            return None
        return {"mean": self._pca_mean, "components": self._pca_components}

    def set_pca_state(self, state: Dict[str, np.ndarray]) -> None:
        self._pca_mean = state["mean"]
        self._pca_components = state["components"]

    def finalize_batch(
        self,
        features_list: Sequence[Dict[str, float]],
        *,
        fit: bool = True,
    ) -> Sequence[Dict[str, float]]:
        """Batch PCA on per-frame ResNet texture → fill channels 16–23."""
        raw_mats, indices = [], []
        for i, feat in enumerate(features_list):
            rm = feat.get("_raw_resnet_ts")
            if rm is not None:
                raw_mats.append(np.asarray(rm, dtype=np.float64))
                indices.append(i)
        if not raw_mats:
            return self._apply_global_channel_zscore(features_list, fit=fit)

        all_frames = np.vstack(raw_mats)
        target = self._tex_pca_dim

        if fit:
            reduced_all, self._pca_mean, self._pca_components = (
                _pca_fit_transform(all_frames, target)
            )
        else:
            if self._pca_mean is None or self._pca_components is None:
                raise RuntimeError(
                    "PCA state is not available.  finalize_batch(fit=True) must "
                    "be called on the training set before finalize_batch(fit=False) "
                    "is called on the test/validation set."
                )
            reduced_all = _pca_transform(
                all_frames, self._pca_mean, self._pca_components, target
            )

        result = list(features_list)
        offset = 0
        for idx, row_idx in enumerate(indices):
            n_frames = raw_mats[idx].shape[0]
            reduced_video = reduced_all[offset : offset + n_frames]
            offset += n_frames

            feat = dict(result[row_idx])

            ts = np.array(
                [feat.get(f"ts2_{i:04d}", 0.0) for i in range(self._flat_len)],
                dtype=np.float32,
            ).reshape(N_TS_VARS_V2, N_TS_STEPS_V2)

            n_fill = min(reduced_video.shape[0], N_TS_STEPS_V2)
            for ch in range(target):
                ch_idx = N_GEO_VARS_V2 + ch
                if ch_idx < N_TS_VARS_V2:
                    ts[ch_idx, :n_fill] = reduced_video[:n_fill, ch].astype(
                        np.float32
                    )

            flat = ts.flatten(order="C")
            for i, v in enumerate(flat):
                feat[f"ts2_{i:04d}"] = float(v)

            feat.pop("_raw_resnet_ts", None)
            result[row_idx] = feat

        return self._apply_global_channel_zscore(result, fit=fit)

    def _apply_global_channel_zscore(
        self,
        features_list: Sequence[Dict[str, float]],
        *,
        fit: bool,
    ) -> Sequence[Dict[str, float]]:
        if not features_list:
            return []

        mats = np.zeros((len(features_list), N_TS_VARS_V2, N_TS_STEPS_V2), dtype=np.float64)
        for i, feat in enumerate(features_list):
            vec = np.array(
                [feat.get(f"ts2_{j:04d}", 0.0) for j in range(self._flat_len)],
                dtype=np.float64,
            )
            mats[i] = vec.reshape(N_TS_VARS_V2, N_TS_STEPS_V2)

        if fit:
            self._channel_mean = mats.mean(axis=(0, 2))
            self._channel_std = mats.std(axis=(0, 2))
            self._channel_std = np.where(self._channel_std < 1e-9, 1.0, self._channel_std)
        else:
            if self._channel_mean is None or self._channel_std is None:
                raise RuntimeError(
                    "Global channel scaler is not available. finalize_batch(fit=True) "
                    "must be called on the training set before finalize_batch(fit=False)."
                )

        mean = self._channel_mean.reshape(1, N_TS_VARS_V2, 1)
        std = self._channel_std.reshape(1, N_TS_VARS_V2, 1)
        mats = (mats - mean) / std
        np.nan_to_num(mats, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        out = []
        for i, feat in enumerate(features_list):
            new_feat = dict(feat)
            flat = mats[i].reshape(-1)
            for j, v in enumerate(flat):
                new_feat[f"ts2_{j:04d}"] = float(v)
            out.append(new_feat)
        return out

    # ── Internal helpers ─────────────────────────────────────────────────

    def _extract_resnet_for_ts(
        self,
        samples: Sequence[FramePrediction],
        resample_frame_idx: np.ndarray,
        video_path: Optional[str],
    ) -> np.ndarray:
        """Extract ResNet features for the resampled frame indices.

        Reuses the same nearest-neighbour lookup logic as V1.
        """
        result = np.zeros((N_TS_STEPS_V2, RESNET_FEAT_DIM), dtype=np.float32)
        if not video_path or not samples or len(resample_frame_idx) == 0:
            return result

        resnet = _get_resnet_extractor(self._resnet_device)
        if not resnet.available:
            return result

        known_frame_indices = np.array(
            [s.frame_index for s in samples], dtype=np.int64
        )
        sorted_order = np.argsort(known_frame_indices)
        known_sorted = known_frame_indices[sorted_order]
        samples_sorted = [samples[i] for i in sorted_order]

        selected: List[FramePrediction] = []
        for target_fi in resample_frame_idx:
            pos = int(np.searchsorted(known_sorted, target_fi))
            pos = min(pos, len(known_sorted) - 1)
            if pos > 0:
                left_dist = abs(int(known_sorted[pos - 1]) - int(target_fi))
                right_dist = abs(int(known_sorted[pos]) - int(target_fi))
                nearest_sample = (
                    samples_sorted[pos - 1]
                    if left_dist <= right_dist
                    else samples_sorted[pos]
                )
            else:
                nearest_sample = samples_sorted[0]

            from ..core.interfaces import FramePrediction as _FP
            proxy = _FP(
                frame_index=int(target_fi),
                bbox=nearest_sample.bbox,
                score=nearest_sample.score,
                segmentation=nearest_sample.segmentation,
            )
            selected.append(proxy)

        feats = resnet.extract_from_video(video_path, selected)

        n_actual = min(feats.shape[0], N_TS_STEPS_V2)
        result[:n_actual] = feats[:n_actual]
        return result
