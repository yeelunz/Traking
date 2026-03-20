"""V3-Lite feature extractors — lightweight, no deep learning.

Design philosophy
-----------------
* **No absolute positions** — motion features use displacement-from-median,
  relative speeds, and heading change instead of cx/cy.
* **No ResNet / deep learning** — texture relies on GLCM + gray-level stats
  extracted directly from video patches (no GPU needed).
* **Compact** — Non-TSC ~33 D, TSC 12 ch × 256 = 3 072 D (default).
* **TSC vs Non-TSC 分別設計** — different channel/feature sets tailored
  for each classification paradigm.

Registered names
~~~~~~~~~~~~~~~~
- ``tab_v3_lite``    : Non-TSC tabular  (33 D per video)
- ``tsc_v3_lite``    : TSC time-series  (12 ch × 256 = 3 072 D default)
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
    _safe_mean,
    _safe_std,
    _compute_glcm_features,
    _aggregate_video_features,
    _build_subject_keys,
)
from .trajectory_filter import (
    multiscale_hampel as _multiscale_hampel,
    bidirectional_savgol as _bidirectional_savgol,
    smooth_trajectory_2d as _smooth_trajectory_2d,
)

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════════════════════════════════

N_TS_STEPS_LITE: int = 256        # Default time-series length (configurable)
N_TS_CHANNELS_LITE: int = 12      # 6 motion + 3 static + 3 texture
_TEXTURE_PATCH_SIZE: int = 64     # Resize patch before GLCM

# ═════════════════════════════════════════════════════════════════════════════
# Non-TSC Feature Keys
# ═════════════════════════════════════════════════════════════════════════════

# --- Motion (13 D) --- no absolute position
LITE_MOTION_KEYS: List[str] = [
    # Basic context
    "num_points",
    "duration_frames",
    # Trajectory shape (position-invariant)
    "path_length",
    "straightness_ratio",
    # 速度 (velocity)
    "mean_speed",
    "std_speed",
    "median_speed",
    # 加速度 (acceleration)
    "mean_acc",
    "std_acc",
    # 相對中位數的變化 (displacement from median)
    "disp_median_mean",
    "disp_median_std",
    "disp_median_max",
    # 移動方向 (heading change)
    "mean_heading_change",
]

# --- Static (10 D) --- CSA, diameter, aspect ratio
LITE_STATIC_KEYS: List[str] = [
    "csa_mean",
    "csa_std",
    "csa_strain_rate",
    "swelling_ratio",
    "eq_diam_mean",
    "eq_diam_strain_rate",
    "circularity_mean",
    "circularity_std",
    "aspect_ratio_mean",
    "aspect_ratio_std",
]

# --- Texture (10 D) --- GLCM-based, aggregated across sampled frames
LITE_TEXTURE_KEYS: List[str] = [
    "tex_glcm_contrast_mean",
    "tex_glcm_contrast_std",
    "tex_glcm_homogeneity_mean",
    "tex_glcm_homogeneity_std",
    "tex_glcm_energy_mean",
    "tex_glcm_correlation_mean",
    "tex_gray_mean",
    "tex_gray_std",
    "tex_grad_mean",
    "tex_grad_std",
]

LITE_VIDEO_KEYS: List[str] = LITE_MOTION_KEYS + LITE_STATIC_KEYS + LITE_TEXTURE_KEYS

# ═════════════════════════════════════════════════════════════════════════════
# TSC Channel Names (12 channels)
# ═════════════════════════════════════════════════════════════════════════════

TS_LITE_CHANNEL_KEYS: List[str] = [
    # ── Motion (6 ch) ──
    "ts_speed",             # inter-frame speed (px/frame)
    "ts_accel",             # acceleration magnitude (px/frame^2)
    "ts_dx_median",         # x displacement from median (px)
    "ts_dy_median",         # y displacement from median (px)
    "ts_heading_sin",       # sin(heading angle)
    "ts_heading_cos",       # cos(heading angle)
    # ── Static / shape (3 ch) ──
    "ts_csa_norm",          # segmentation CSA (raw px^2; name kept for compatibility)
    "ts_eq_diam_norm",      # equivalent diameter (raw px; name kept for compatibility)
    "ts_aspect_ratio",      # w / h
    # ── Texture (3 ch) ──
    "ts_glcm_contrast",     # per-frame GLCM contrast (raw)
    "ts_glcm_homogeneity",  # per-frame GLCM homogeneity
    "ts_gray_mean",         # per-frame gray mean (raw)
]


# ═════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═════════════════════════════════════════════════════════════════════════════


def _circularity(area: float, perimeter: float) -> float:
    if perimeter <= 0:
        return 0.0
    return float(4.0 * np.pi * area / (perimeter ** 2))


def _read_gray_patch(
    cap: cv2.VideoCapture,
    frame_idx: int,
    bbox: Tuple[float, float, float, float],
    target_size: int = _TEXTURE_PATCH_SIZE,
) -> Optional[np.ndarray]:
    """Read a grayscale bbox patch from an already-opened VideoCapture."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    h, w = frame.shape[:2]
    x, y, bw, bh = bbox
    x0, y0 = max(0, int(np.floor(x))), max(0, int(np.floor(y)))
    x1, y1 = min(w, int(np.ceil(x + bw))), min(h, int(np.ceil(y + bh)))
    if x1 <= x0 or y1 <= y0:
        return None
    patch = frame[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    if patch.ndim == 3:
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    if target_size > 0 and min(patch.shape[:2]) > 0:
        patch = cv2.resize(patch, (target_size, target_size),
                           interpolation=cv2.INTER_AREA)
    return patch


def _compute_patch_texture(gray: np.ndarray) -> Dict[str, float]:
    """Compute lightweight texture features from a grayscale patch.

    Returns 4 values: glcm_contrast, glcm_homogeneity, gray_mean, grad_mean.
    (Used for TSC per-frame channels.)
    """
    glcm = _compute_glcm_features(gray)
    grad_x = cv2.Sobel(gray.astype(np.float64), cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray.astype(np.float64), cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x.astype(np.float32), grad_y.astype(np.float32))
    return {
        "glcm_contrast": glcm["tex_glcm_contrast"],
        "glcm_homogeneity": glcm["tex_glcm_homogeneity"],
        "glcm_energy": glcm["tex_glcm_energy"],
        "glcm_correlation": glcm["tex_glcm_correlation"],
        "gray_mean": float(np.mean(gray)),
        "gray_std": float(np.std(gray)),
        "grad_mean": float(np.mean(grad_mag)),
        "grad_std": float(np.std(grad_mag)),
    }


def _pca_fit_transform(
    X: np.ndarray,
    target_dim: int,
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


# ═════════════════════════════════════════════════════════════════════════════
# Interpolation helper (shared with ext)
# ═════════════════════════════════════════════════════════════════════════════

_SCIPY_INTERP_OK: bool
try:
    from scipy.interpolate import CubicSpline as _CubicSpline  # noqa: F401
    _SCIPY_INTERP_OK = True
except ImportError:
    _SCIPY_INTERP_OK = False


def _interp_channel(
    t_known: np.ndarray,
    v_known: np.ndarray,
    t_all: np.ndarray,
    method: str = "cubic",
) -> np.ndarray:
    """Interpolate a 1-D channel over a dense timeline."""
    if len(t_known) == 0:
        return np.zeros(len(t_all), dtype=np.float64)
    if len(t_known) == 1:
        return np.full(len(t_all), v_known[0], dtype=np.float64)
    if method == "cubic" and _SCIPY_INTERP_OK and len(t_known) >= 4:
        try:
            from scipy.interpolate import CubicSpline
            cs = CubicSpline(t_known, v_known, extrapolate=True)
            result = cs(t_all)
            v_min, v_max = float(v_known.min()), float(v_known.max())
            np.clip(result, v_min, v_max, out=result)
            return result
        except Exception:
            pass
    return np.interp(t_all, t_known, v_known)


def _condition_sparse(values: np.ndarray) -> np.ndarray:
    """Hampel outlier removal on sparse data."""
    if len(values) < 3:
        return values.copy()
    filtered, _ = _multiscale_hampel(
        values, macro_ratio=0.15, macro_sigma=3.0,
        micro_hw=3, micro_sigma=3.0, use_mirror_pad=True,
    )
    return filtered


def _condition_dense(values: np.ndarray) -> np.ndarray:
    """Bidirectional S-G smoothing on dense data."""
    return _bidirectional_savgol(values, window_length=7, polyorder=2)


# ═════════════════════════════════════════════════════════════════════════════
# Non-TSC: Motion features (13 D)
# ═════════════════════════════════════════════════════════════════════════════


def _compute_motion_lite(
    samples: Sequence[FramePrediction],
) -> Dict[str, float]:
    """Compute lite motion features — no absolute positions."""
    zeros = OrderedDict((k, 0.0) for k in LITE_MOTION_KEYS)
    if not samples:
        return zeros

    frames = np.asarray([float(s.frame_index) for s in samples], dtype=np.float64)
    raw_centers = np.asarray([s.center for s in samples], dtype=np.float64)
    try:
        centers = _smooth_trajectory_2d(raw_centers, frames)
    except Exception:
        centers = raw_centers
    n = len(samples)

    feat = OrderedDict()
    feat["num_points"] = float(n)
    feat["duration_frames"] = float(frames[-1] - frames[0]) if n > 1 else 0.0

    # ── Path length & straightness ──
    if n > 1:
        diffs = np.diff(centers, axis=0)
        dt = np.diff(frames)
        dt[dt == 0] = 1.0
        step_dist = np.linalg.norm(diffs, axis=1)
        path_length = float(np.sum(step_dist))
        speeds = step_dist / dt
    else:
        path_length = 0.0
        speeds = np.zeros(0)

    disp_vec = centers[-1] - centers[0] if n > 1 else np.zeros(2)
    net_disp = float(np.linalg.norm(disp_vec))
    feat["path_length"] = path_length
    feat["straightness_ratio"] = float(net_disp / max(path_length, 1e-9))

    # ── 速度 ──
    feat["mean_speed"] = _safe_mean(speeds)
    feat["std_speed"] = _safe_std(speeds)
    feat["median_speed"] = float(np.median(speeds)) if speeds.size else 0.0

    # ── 加速度 ──
    if speeds.size > 1:
        accels = np.abs(np.diff(speeds))
    else:
        accels = np.zeros(0)
    feat["mean_acc"] = _safe_mean(accels)
    feat["std_acc"] = _safe_std(accels)

    # ── 相對中位數的變化（以初始等效直徑正規化，無量綱） ──
    init_eq_diam = 0.0
    for s in samples:
        if s.segmentation and s.segmentation.stats:
            init_eq_diam = float(getattr(s.segmentation.stats, "equivalent_diameter_px", 0.0) or 0.0)
            if init_eq_diam > 0:
                break
    if init_eq_diam <= 0:
        # fallback: use initial bbox geometric mean length scale
        bw0, bh0 = float(samples[0].bbox[2]), float(samples[0].bbox[3])
        init_eq_diam = float(np.sqrt(max(bw0 * bh0, 1e-9)))
    scale_ref = max(init_eq_diam, 1e-9)

    median_pos = np.median(centers, axis=0)
    displacements = np.linalg.norm(centers - median_pos, axis=1)
    rel_displacements = displacements / scale_ref
    feat["disp_median_mean"] = float(rel_displacements.mean())
    feat["disp_median_std"] = _safe_std(rel_displacements)
    feat["disp_median_max"] = float(rel_displacements.max())

    # ── 移動方向變化 ──
    if n > 2:
        diffs_all = np.diff(centers, axis=0)
        angles = np.arctan2(diffs_all[:, 1], diffs_all[:, 0])
        ang_changes = np.abs(np.diff(angles))
        ang_changes = np.minimum(ang_changes, 2.0 * np.pi - ang_changes)
        feat["mean_heading_change"] = float(ang_changes.mean())
    else:
        feat["mean_heading_change"] = 0.0

    return feat


# ═════════════════════════════════════════════════════════════════════════════
# Non-TSC: Static features (10 D)
# ═════════════════════════════════════════════════════════════════════════════


def _compute_static_lite(
    samples: Sequence[FramePrediction],
) -> Dict[str, float]:
    """CSA, equivalent diameter, circularity, aspect ratio statistics."""
    zeros = OrderedDict((k, 0.0) for k in LITE_STATIC_KEYS)
    if not samples:
        return zeros

    usable = [s for s in samples
              if s.segmentation and s.segmentation.stats
              and getattr(s.segmentation.stats, "area_px", 0.0) > 0]

    feat = OrderedDict()

    # CSA
    if usable:
        areas = np.array([s.segmentation.stats.area_px for s in usable],
                         dtype=np.float64)
        eq_diams = np.array([s.segmentation.stats.equivalent_diameter_px
                             for s in usable], dtype=np.float64)
        circs = np.array([
            _circularity(s.segmentation.stats.area_px,
                         s.segmentation.stats.perimeter_px)
            for s in usable
        ], dtype=np.float64)
    else:
        areas = eq_diams = circs = np.zeros(0)

    if areas.size:
        init_area = float(areas[0])
        init_area = max(init_area, 1e-9)
        feat["csa_mean"] = float(areas.mean())
        feat["csa_std"] = _safe_std(areas)
        feat["csa_strain_rate"] = float((areas.max() - areas.min()) / init_area)
        feat["swelling_ratio"] = float(areas.max() / init_area)
    else:
        feat["csa_mean"] = feat["csa_std"] = feat["csa_strain_rate"] = 0.0
        feat["swelling_ratio"] = 0.0

    if eq_diams.size:
        init_eq_d = max(float(eq_diams[0]), 1e-9)
        feat["eq_diam_mean"] = float(eq_diams.mean())
        feat["eq_diam_strain_rate"] = float((eq_diams.max() - eq_diams.min()) / init_eq_d)
    else:
        feat["eq_diam_mean"] = feat["eq_diam_strain_rate"] = 0.0

    if circs.size:
        feat["circularity_mean"] = float(circs.mean())
        feat["circularity_std"] = _safe_std(circs)
    else:
        feat["circularity_mean"] = feat["circularity_std"] = 0.0

    # Aspect ratio from bbox
    bboxes = np.array([s.bbox for s in samples], dtype=np.float64)
    aspects = bboxes[:, 2] / (bboxes[:, 3] + 1e-9)
    feat["aspect_ratio_mean"] = float(aspects.mean())
    feat["aspect_ratio_std"] = _safe_std(aspects)

    return feat


# ═════════════════════════════════════════════════════════════════════════════
# Non-TSC: Texture features (10 D) — GLCM-based, multi-frame aggregation
# ═════════════════════════════════════════════════════════════════════════════


def _compute_texture_lite(
    samples: Sequence[FramePrediction],
    video_path: Optional[str],
    n_sample_frames: int = 5,
) -> Dict[str, float]:
    """Sample up to *n_sample_frames* evenly-spaced frames, compute GLCM
    + gray stats per frame, then aggregate with mean/std → 10 D.
    """
    zeros = OrderedDict((k, 0.0) for k in LITE_TEXTURE_KEYS)
    if not samples or not video_path:
        return zeros

    # Select top N highest confidence frames (fallback to time-sorted if no scores)
    n = len(samples)
    if n <= n_sample_frames:
        indices = list(range(n))
    else:
        scored_indices = sorted(
            range(n),
            key=lambda i: getattr(samples[i], "score", 0.0) or 0.0,
            reverse=True
        )
        indices = sorted(scored_indices[:n_sample_frames])

    # Read patches and compute per-frame texture
    per_frame: List[Dict[str, float]] = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return zeros
    try:
        for idx in indices:
            s = samples[idx]
            patch = _read_gray_patch(cap, int(round(s.frame_index)), s.bbox)
            if patch is not None:
                per_frame.append(_compute_patch_texture(patch))
    finally:
        cap.release()

    if not per_frame:
        return zeros

    # Aggregate across sampled frames
    def _agg(key: str) -> Tuple[float, float]:
        vals = np.array([pf[key] for pf in per_frame], dtype=np.float64)
        return _safe_mean(vals), _safe_std(vals)

    feat = OrderedDict()
    m, s = _agg("glcm_contrast")
    feat["tex_glcm_contrast_mean"] = m
    feat["tex_glcm_contrast_std"] = s

    m, s = _agg("glcm_homogeneity")
    feat["tex_glcm_homogeneity_mean"] = m
    feat["tex_glcm_homogeneity_std"] = s

    feat["tex_glcm_energy_mean"] = _agg("glcm_energy")[0]
    feat["tex_glcm_correlation_mean"] = _agg("glcm_correlation")[0]

    feat["tex_gray_mean"] = _agg("gray_mean")[0]
    feat["tex_gray_std"] = _agg("gray_std")[0]

    feat["tex_grad_mean"] = _agg("grad_mean")[0]
    feat["tex_grad_std"] = _agg("grad_std")[0]

    return feat


# ═════════════════════════════════════════════════════════════════════════════
# TSC: Build (12, 128) channel matrix
# ═════════════════════════════════════════════════════════════════════════════


def _extract_ts_channels_lite(
    samples: Sequence[FramePrediction],
    video_path: Optional[str],
    frame_w: float = 640.0,
    frame_h: float = 480.0,
    interp_method: str = "cubic",
    n_steps: int = N_TS_STEPS_LITE,
) -> np.ndarray:
    """Build ``(N_TS_CHANNELS_LITE, n_steps)`` v3-lite channel matrix.

    Channel layout (12 total)
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    0  speed           — inter-frame speed (px/frame)
    1  accel           — acceleration magnitude (px/frame^2)
    2  dx_median       — x displacement from trajectory median (px)
    3  dy_median       — y displacement from trajectory median (px)
    4  heading_sin     — sin(heading angle)
    5  heading_cos     — cos(heading angle)
    6  csa_norm        — segmentation CSA (raw px^2; key name kept for compat)
    7  eq_diam_norm    — equivalent diameter (raw px; key name kept for compat)
    8  aspect_ratio    — w / h
    9  glcm_contrast   — per-frame GLCM contrast (raw)
    10 glcm_homogeneity— per-frame GLCM homogeneity
    11 gray_mean       — per-frame gray mean (raw)

    Returns shape ``(N_TS_CHANNELS_LITE, n_steps)`` float32.
    """
    N_CH = N_TS_CHANNELS_LITE
    N_T = n_steps

    if not samples:
        return np.zeros((N_CH, N_T), dtype=np.float32)

    # ── Sort by frame index ──
    samples = sorted(samples, key=lambda s: s.frame_index)
    n_raw = len(samples)
    frame_indices = np.array([s.frame_index for s in samples], dtype=np.int64)

    # ── Build base arrays (raw physical units) ──
    bboxes = np.array([s.bbox for s in samples], dtype=np.float64).reshape(-1, 4)

    seg_areas = np.array([
        s.segmentation.stats.area_px
        if (s.segmentation and s.segmentation.stats) else 0.0
        for s in samples
    ], dtype=np.float64)

    eq_diams = np.array([
        s.segmentation.stats.equivalent_diameter_px
        if (s.segmentation and s.segmentation.stats) else 0.0
        for s in samples
    ], dtype=np.float64)

    # ── Build sparse geometric channels at known frames ──
    # We'll build: cx_px, cy_px, csa_px2, eq_diam_px, aspect
    cx_sparse = np.array([(s.bbox[0] + s.bbox[2] / 2.0) for s in samples])
    cy_sparse = np.array([(s.bbox[1] + s.bbox[3] / 2.0) for s in samples])
    csa_sparse = seg_areas
    eq_diam_sparse = eq_diams
    aspect_sparse = bboxes[:, 2] / (bboxes[:, 3] + 1e-9)

    # ── Hampel on sparse channels (include centroid for classification path) ──
    cx_sparse = _condition_sparse(cx_sparse)
    cy_sparse = _condition_sparse(cy_sparse)
    csa_sparse = _condition_sparse(csa_sparse)
    eq_diam_sparse = _condition_sparse(eq_diam_sparse)

    # ── Extract per-frame GLCM texture at sparse positions ──
    tex_contrast = np.zeros(n_raw, dtype=np.float64)
    tex_homogeneity = np.zeros(n_raw, dtype=np.float64)
    tex_gray = np.zeros(n_raw, dtype=np.float64)

    if video_path:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            try:
                for i, s in enumerate(samples):
                    patch = _read_gray_patch(
                        cap, int(round(s.frame_index)), s.bbox)
                    if patch is not None:
                        tf = _compute_patch_texture(patch)
                        tex_contrast[i] = tf["glcm_contrast"]
                        tex_homogeneity[i] = tf["glcm_homogeneity"]
                        tex_gray[i] = tf["gray_mean"]
            finally:
                cap.release()

    # ── Hampel on texture channels ──
    tex_contrast = _condition_sparse(tex_contrast)
    tex_homogeneity = _condition_sparse(tex_homogeneity)
    tex_gray = _condition_sparse(tex_gray)

    # ── Build full-length timeline T and interpolate ──
    T = int(frame_indices[-1]) + 1
    t_all = np.arange(T, dtype=np.float64)
    t_known = frame_indices.astype(np.float64)

    cx_full = _interp_channel(t_known, cx_sparse, t_all, interp_method)
    cy_full = _interp_channel(t_known, cy_sparse, t_all, interp_method)
    csa_full = _interp_channel(t_known, csa_sparse, t_all, interp_method)
    eq_diam_full = _interp_channel(t_known, eq_diam_sparse, t_all, interp_method)
    aspect_full = _interp_channel(t_known, aspect_sparse, t_all, interp_method)
    contrast_full = _interp_channel(t_known, tex_contrast, t_all, interp_method)
    homogeneity_full = _interp_channel(t_known, tex_homogeneity, t_all, interp_method)
    gray_full = _interp_channel(t_known, tex_gray, t_all, interp_method)

    # ── Post-interpolation S-G smoothing (include centroid channels) ──
    cx_full = _condition_dense(cx_full)
    cy_full = _condition_dense(cy_full)
    csa_full = _condition_dense(csa_full)
    eq_diam_full = _condition_dense(eq_diam_full)
    contrast_full = _condition_dense(contrast_full)
    homogeneity_full = _condition_dense(homogeneity_full)
    gray_full = _condition_dense(gray_full)

    # ── Derived motion channels from interpolated positions ──
    cx_px = cx_full
    cy_px = cy_full

    # Speed
    speeds_full = np.zeros(T, dtype=np.float64)
    if T > 1:
        dx = np.diff(cx_px)
        dy = np.diff(cy_px)
        step_dist = np.sqrt(dx ** 2 + dy ** 2)
        speeds_full[1:] = step_dist

    # Acceleration
    accel_full = np.zeros(T, dtype=np.float64)
    if T > 2:
        accel_full[1:] = np.abs(np.diff(speeds_full))

    # Displacement from median
    median_cx = float(np.median(cx_full))
    median_cy = float(np.median(cy_full))
    dx_median = cx_full - median_cx
    dy_median = cy_full - median_cy

    # Heading (sin/cos)
    heading_sin = np.zeros(T, dtype=np.float64)
    heading_cos = np.zeros(T, dtype=np.float64)
    if T > 1:
        vx = np.zeros(T, dtype=np.float64)
        vy = np.zeros(T, dtype=np.float64)
        vx[1:] = np.diff(cx_px)
        vy[1:] = np.diff(cy_px)
        vmag = np.sqrt(vx ** 2 + vy ** 2) + 1e-12
        heading_sin = vy / vmag
        heading_cos = vx / vmag

    # ── Assemble (N_CH, T) ──
    timeline = np.zeros((T, N_CH), dtype=np.float64)
    timeline[:, 0] = speeds_full
    timeline[:, 1] = accel_full
    timeline[:, 2] = dx_median
    timeline[:, 3] = dy_median
    timeline[:, 4] = heading_sin
    timeline[:, 5] = heading_cos
    timeline[:, 6] = csa_full
    timeline[:, 7] = eq_diam_full
    timeline[:, 8] = aspect_full
    timeline[:, 9] = contrast_full
    timeline[:, 10] = homogeneity_full   # already in [0, 1]
    timeline[:, 11] = gray_full

    # ── Guard NaN / Inf ──
    np.nan_to_num(timeline, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Uniform resample T → 128 ──
    resample_idx = np.round(np.linspace(0, T - 1, N_T)).astype(int)
    resample_idx = np.clip(resample_idx, 0, T - 1)
    result = timeline[resample_idx, :].T.astype(np.float32)  # (N_CH, N_T)

    return result


# ═════════════════════════════════════════════════════════════════════════════
# EXTRACTOR 1 (Non-TSC): tab_v3_lite  — 33 D per video
# ═════════════════════════════════════════════════════════════════════════════


@register_feature_extractor("tab_v3_lite")
class MotionStaticLiteFeatureExtractor(TrajectoryFeatureExtractor):
    """Lightweight non-TSC feature extractor (33 D, no deep learning).

    Feature structure (video level)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - **Motion**  : 13 D — speed, acceleration, displacement-from-median,
      heading change (no absolute positions)
    - **Static**  : 10 D — CSA, equiv diameter, circularity, aspect ratio
    - **Texture** : 10 D — GLCM + gray stats aggregated from sampled frames

    Total = 13 + 10 + 10 = **33 D** per video.

    No PCA, no ResNet, no GPU required.
    """

    name = "MotionStaticLiteFeatures"
    DEFAULT_CONFIG: Dict[str, Any] = {
        "aggregate_stats": ["mean", "std", "min", "max"],
        "n_texture_frames": 5,
        "texture_mode": "freeze",
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        cfg = params or {}
        self._subject_stats = list(
            cfg.get("aggregate_stats", ["mean", "std", "min", "max"])
        )
        self._n_tex_frames = int(cfg.get("n_texture_frames", 5))
        self._texture_mode = str(cfg.get("texture_mode", "freeze")).lower()
        if self._texture_mode != "freeze":
            logger.warning(
                "tab_v3_lite uses GLCM handcrafted texture; forcing texture_mode=freeze (non-learnable)."
            )
            self._texture_mode = "freeze"
        self._video_keys = list(LITE_VIDEO_KEYS)
        self._subject_keys = _build_subject_keys(
            self._video_keys, self._subject_stats
        )
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
        motion = _compute_motion_lite(samples)
        static = _compute_static_lite(samples)
        texture = _compute_texture_lite(samples, video_path, self._n_tex_frames)

        combined = OrderedDict()
        for k in LITE_MOTION_KEYS:
            combined[k] = motion.get(k, 0.0)
        for k in LITE_STATIC_KEYS:
            combined[k] = static.get(k, 0.0)
        for k in LITE_TEXTURE_KEYS:
            combined[k] = texture.get(k, 0.0)
        return combined

    def aggregate_subject(
        self, video_features: Sequence[Dict[str, float]]
    ) -> Dict[str, float]:
        return _aggregate_video_features(
            video_features, self._video_keys, self._subject_stats
        )

    def finalize_batch(
        self,
        features_list: Sequence[Dict[str, float]],
        *,
        fit: bool = True,
    ) -> Sequence[Dict[str, float]]:
        if not features_list:
            return []

        X_tex = np.array(
            [[feat.get(k, 0.0) for k in LITE_TEXTURE_KEYS] for feat in features_list],
            dtype=np.float64,
        )

        target = len(LITE_TEXTURE_KEYS)
        if fit:
            reduced, self._pca_mean, self._pca_components = _pca_fit_transform(X_tex, target)
        else:
            if self._pca_mean is None or self._pca_components is None:
                raise RuntimeError(
                    "GLCM texture PCA state is not fitted. finalize_batch(fit=True) must run first."
                )
            reduced = _pca_transform(X_tex, self._pca_mean, self._pca_components, target)

        out: List[Dict[str, float]] = []
        for i, feat in enumerate(features_list):
            new_feat = OrderedDict((k, float(feat.get(k, 0.0))) for k in self._video_keys)
            for j, key in enumerate(LITE_TEXTURE_KEYS):
                new_feat[key] = float(reduced[i, j])
            out.append(new_feat)

        mats = np.array(
            [[feat.get(k, 0.0) for k in self._video_keys] for feat in out],
            dtype=np.float64,
        )
        if fit:
            self._feat_mean = mats.mean(axis=0)
            self._feat_std = mats.std(axis=0)
            self._feat_std = np.where(self._feat_std < 1e-9, 1.0, self._feat_std)
        else:
            if self._feat_mean is None or self._feat_std is None:
                raise RuntimeError(
                    "tab_v3_lite global z-score is not fitted. finalize_batch(fit=True) must run first."
                )

        mats = (mats - self._feat_mean) / self._feat_std
        np.nan_to_num(mats, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        standardized: List[Dict[str, float]] = []
        for i in range(len(out)):
            row = OrderedDict()
            for j, k in enumerate(self._video_keys):
                row[k] = float(mats[i, j])
            standardized.append(row)
        return standardized

    def get_state(self) -> Dict[str, Any]:
        return {
            "pca_mean": self._pca_mean,
            "pca_components": self._pca_components,
            "feat_mean": self._feat_mean,
            "feat_std": self._feat_std,
            "video_keys": list(self._video_keys),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise ValueError("tab_v3_lite state must be a dict")
        self._pca_mean = state.get("pca_mean")
        self._pca_components = state.get("pca_components")
        self._feat_mean = state.get("feat_mean")
        self._feat_std = state.get("feat_std")


# ═════════════════════════════════════════════════════════════════════════════
# EXTRACTOR 2 (TSC): tsc_v3_lite  — 12 ch × 256 = 3,072 D (default)
# ═════════════════════════════════════════════════════════════════════════════


@register_feature_extractor("tsc_v3_lite")
class TimeSeriesV3LiteFeatureExtractor(TrajectoryFeatureExtractor):
    """Lightweight TSC time-series feature extractor (no deep learning).

    Channel layout (12 total)
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    ===========  ========  ============================================
    Index range  Count     Description
    ===========  ========  ============================================
    0 – 5        6         Motion: speed, accel, dx_median, dy_median,
                           heading_sin, heading_cos
    6 – 8        3         Static: csa_norm, eq_diam_norm, aspect_ratio
    9 – 11       3         Texture: glcm_contrast, glcm_homogeneity,
                           gray_mean (per-frame, interpolated)
    ===========  ========  ============================================

    Time-series length = **256** by default (configurable via ``n_steps``).
    Flat feature vector = 12 × 256 = **3,072 D** (default).

    **No PCA, no ResNet, no GPU required.**  Texture channels are computed
    directly from GLCM on video patches at sparse frames, then interpolated
    to the full timeline.

    ``finalize_batch`` performs channel-wise global standardization:
    fit on training set only, then transform validation/test.

    Downstream classifiers may apply their own dimensionality reduction:
    - PatchTST / TimeMachine: end-to-end learned temporal projection
    - MultiRocket: Autoencoder pre-reduction
    - Tabular classifiers: UMAP / LDA
    """

    name = "TimeSeriesV3LiteFeatures"
    DEFAULT_CONFIG: Dict[str, Any] = {
        "frame_w": 640.0,
        "frame_h": 480.0,
        "interp_method": "cubic",
        "n_steps": N_TS_STEPS_LITE,        # default 256; set to 128 for compat
        "texture_mode": "freeze",
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        cfg = params or {}
        self._frame_w = float(cfg.get("frame_w", 640.0))
        self._frame_h = float(cfg.get("frame_h", 480.0))
        self._interp_method = str(cfg.get("interp_method", "cubic"))
        self._n_steps = int(cfg.get("n_steps", N_TS_STEPS_LITE))
        self._texture_mode = str(cfg.get("texture_mode", "freeze")).lower()
        if self._texture_mode != "freeze":
            logger.warning(
                "tsc_v3_lite uses GLCM handcrafted texture; forcing texture_mode=freeze (non-learnable)."
            )
            self._texture_mode = "freeze"

        self._n_channels = N_TS_CHANNELS_LITE
        self._flat_len = self._n_channels * self._n_steps
        self._video_keys = [f"tsl_{i:04d}" for i in range(self._flat_len)]
        self._subject_keys = self._video_keys
        self._channel_mean: Optional[np.ndarray] = None
        self._channel_std: Optional[np.ndarray] = None
        self._tex_pca_mean: Optional[np.ndarray] = None
        self._tex_pca_components: Optional[np.ndarray] = None

    def feature_order(self, level: str = "video") -> Sequence[str]:
        return self._video_keys

    def extract_video(
        self,
        samples: Sequence[FramePrediction],
        video_path: Optional[str] = None,
    ) -> Dict[str, float]:
        ts = _extract_ts_channels_lite(
            samples, video_path,
            self._frame_w, self._frame_h, self._interp_method,
            n_steps=self._n_steps,
        )  # (12, n_steps)

        flat = ts.flatten(order="C")
        feat: Dict[str, float] = OrderedDict()
        for i, v in enumerate(flat):
            feat[f"tsl_{i:04d}"] = float(v)

        # Store metadata for TSC classifiers to reshape
        feat["_ts_n_vars"] = float(self._n_channels)
        feat["_ts_n_timesteps"] = float(self._n_steps)
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

    def finalize_batch(
        self,
        features_list: Sequence[Dict[str, float]],
        *,
        fit: bool = True,
    ) -> Sequence[Dict[str, float]]:
        """Channel-wise global Z-score for TSC v3lite features.

        - ``fit=True``: estimate mean/std from training videos only.
        - ``fit=False``: apply stored training statistics to val/test.
        """
        if not features_list:
            return []

        mats = np.zeros((len(features_list), self._n_channels, self._n_steps), dtype=np.float64)
        for i, feat in enumerate(features_list):
            vec = np.array([feat.get(k, 0.0) for k in self._video_keys], dtype=np.float64)
            mats[i] = vec.reshape(self._n_channels, self._n_steps)

        # GLCM texture channels are non-learnable -> PCA route.
        tex = mats[:, 9:12, :].transpose(0, 2, 1).reshape(-1, 3)
        if fit:
            tex_red, self._tex_pca_mean, self._tex_pca_components = _pca_fit_transform(tex, 3)
        else:
            if self._tex_pca_mean is None or self._tex_pca_components is None:
                raise RuntimeError(
                    "GLCM texture PCA state is not fitted. finalize_batch(fit=True) must run first."
                )
            tex_red = _pca_transform(tex, self._tex_pca_mean, self._tex_pca_components, 3)
        mats[:, 9:12, :] = tex_red.reshape(len(features_list), self._n_steps, 3).transpose(0, 2, 1)

        if fit:
            # global stats per channel across all training videos and timesteps
            self._channel_mean = mats.mean(axis=(0, 2))
            self._channel_std = mats.std(axis=(0, 2))
            self._channel_std = np.where(self._channel_std < 1e-9, 1.0, self._channel_std)
        else:
            if self._channel_mean is None or self._channel_std is None:
                raise RuntimeError(
                    "Global channel scaler is not fitted. finalize_batch(fit=True) "
                    "must be called on training data before finalize_batch(fit=False)."
                )

        mean = self._channel_mean.reshape(1, self._n_channels, 1)
        std = self._channel_std.reshape(1, self._n_channels, 1)
        mats = (mats - mean) / std
        np.nan_to_num(mats, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        result = []
        for i, feat in enumerate(features_list):
            cleaned: Dict[str, float] = OrderedDict()
            flat = mats[i].reshape(-1)
            for j, k in enumerate(self._video_keys):
                cleaned[k] = float(flat[j])
            result.append(cleaned)
        return result

    def get_state(self) -> Dict[str, Any]:
        return {
            "channel_mean": self._channel_mean,
            "channel_std": self._channel_std,
            "tex_pca_mean": self._tex_pca_mean,
            "tex_pca_components": self._tex_pca_components,
            "n_channels": self._n_channels,
            "n_steps": self._n_steps,
            "video_keys": list(self._video_keys),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise ValueError("tsc_v3_lite state must be a dict")
        self._channel_mean = state.get("channel_mean")
        self._channel_std = state.get("channel_std")
        self._tex_pca_mean = state.get("tex_pca_mean")
        self._tex_pca_components = state.get("tex_pca_components")
