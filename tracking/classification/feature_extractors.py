"""Feature extractors for the classification pipeline.

提供兩種特徵提取方法：
1. ``motion_only``  — 僅運動特徵 (多尺度 Hampel + 雙向 S-G 平滑)
2. ``motion_texture`` — 運動特徵 + 紋理特徵 + CSA 靜態特徵 (紋理過多時以 PCA 降維)

原有的 basic / texture_hybrid / backbone_texture / segmentation_motion 已移除。
卡爾曼濾波已完全移除，改用多尺度雙階段 Hampel + 雙向 Savitzky-Golay 濾波。
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..core.interfaces import FramePrediction, SegmentationData
from ..core.registry import register_feature_extractor
from .interfaces import TrajectoryFeatureExtractor

# ═══════════════════════════════════════════════════════════════════════
# Utility helpers
# ═══════════════════════════════════════════════════════════════════════

def _safe_std(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.std(arr, ddof=0))


def _safe_mean(arr: np.ndarray) -> float:
    return float(np.mean(arr)) if arr.size else 0.0


def _global_standardize_features(
    features_list: Sequence[Dict[str, float]],
    keys: Sequence[str],
    *,
    fit: bool,
    mean: Optional[np.ndarray],
    std: Optional[np.ndarray],
) -> Tuple[Sequence[Dict[str, float]], np.ndarray, np.ndarray]:
    """Global Z-score standardization over a list of feature dicts.

    Parameters
    ----------
    fit : bool
        True => fit mean/std on this batch (training set) and apply.
        False => apply provided mean/std (validation/test set).
    """
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
                "Global feature scaler is not fitted. finalize_batch(fit=True) "
                "must be called on training data before finalize_batch(fit=False)."
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


# ───────────────────── Trajectory Smoothing (Hampel + S-G) ─────────────────────

from .trajectory_filter import smooth_trajectory_2d as _smooth_trajectory_2d

# Legacy stub: _KalmanSmoother2D is DEPRECATED — retained only for backward
# compatibility of imports; all smoothing now uses multi-scale Hampel + S-G.

class _KalmanSmoother2D:
    """DEPRECATED — retained for import compatibility only.

    All smoothing now uses multi-scale Hampel + bidirectional Savitzky-Golay
    via ``trajectory_filter.smooth_trajectory_2d``.
    """

    def __init__(
        self,
        process_noise: float = 1e-2,
        measurement_noise: float = 1e-1,
    ) -> None:
        import warnings
        warnings.warn(
            "_KalmanSmoother2D is deprecated; use trajectory_filter.smooth_trajectory_2d instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    def smooth(
        self,
        observations: np.ndarray,
        frame_indices: np.ndarray,
    ) -> np.ndarray:
        """Forward to multi-scale Hampel + bidirectional S-G."""
        return _smooth_trajectory_2d(observations, frame_indices)


# ═══════════════════════════════════════════════════════════════════════
# Motion feature computation (shared logic)
# ═══════════════════════════════════════════════════════════════════════

# 運動特徵列表
MOTION_FEATURE_KEYS: List[str] = [
    # 基本
    "num_points",
    "duration_frames",
    # 位移
    "displacement_x",
    "displacement_y",
    "net_displacement",
    "span_x",
    "span_y",
    "path_length",
    "straightness_ratio",
    # 速度
    "mean_speed",
    "max_speed",
    "std_speed",
    "median_speed",
    "p95_speed",
    "p5_speed",
    # 加速度
    "mean_acc",
    "max_acc",
    "std_acc",
    # 急動度 (Jerk)
    "mean_jerk",
    "max_jerk",
    "std_jerk",
    # 曲率
    "curvature_mean",
    "curvature_std",
    # 角度變化
    "angular_change_mean",
    "angular_change_std",
    # 面積動態
    "area_mean",
    "area_std",
    "area_range",
    "area_change_mean",
    "area_change_std",
]


def _compute_motion_features(
    samples: Sequence[FramePrediction],
    kalman_process_noise: float = 1e-2,
    kalman_measurement_noise: float = 1e-1,
) -> Dict[str, float]:
    """從追蹤樣本計算運動特徵。

    Parameters ``kalman_process_noise`` and ``kalman_measurement_noise``
    are kept in the signature for backward compatibility but are **ignored**.

    NOTE (2025-06): Upstream trajectory filtering (multi-scale Hampel + bidi
    S-G via ``trajectory_filter.filter_detections``) is now applied to **both**
    test predictions and ground-truth training trajectories *before* they are
    passed to the feature extractor.  Re-smoothing here would cause
    double-smoothing (over-attenuation of legitimate high-frequency dynamics)
    and would also re-introduce Hampel on GT trajectories that were explicitly
    designed to bypass outlier removal.  Therefore the centroids are used
    as-is from the input FramePrediction objects.
    """
    zeros = OrderedDict((k, 0.0) for k in MOTION_FEATURE_KEYS)
    if not samples:
        return zeros

    frames = np.asarray([float(s.frame_index) for s in samples], dtype=np.float64)
    raw_centers = np.asarray([s.center for s in samples], dtype=np.float64)
    bboxes = np.asarray([s.bbox for s in samples], dtype=np.float64)
    areas = bboxes[:, 2] * bboxes[:, 3]

    # Upstream trajectory_filter already applied appropriate smoothing;
    # use pre-smoothed centroids directly (no double-smoothing).
    centers = raw_centers

    n = len(samples)
    duration = float(frames[-1] - frames[0]) if n > 1 else 0.0

    # 位移和路徑
    disp_x = float(centers[-1, 0] - centers[0, 0])
    disp_y = float(centers[-1, 1] - centers[0, 1])
    net_disp = float(np.sqrt(disp_x ** 2 + disp_y ** 2))
    span_x = float(np.ptp(centers[:, 0]))
    span_y = float(np.ptp(centers[:, 1]))

    # 速度、加速度、急動度
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

    straightness = float(net_disp / max(path_length, 1e-9))

    if speeds.size > 1:
        accels = np.abs(np.diff(speeds))
    else:
        accels = np.zeros(0)

    if accels.size > 1:
        jerks = np.abs(np.diff(accels))
    else:
        jerks = np.zeros(0)

    # 曲率 (離散近似: |cross(v1,v2)| / |v1|^3)
    curvatures = np.zeros(0)
    if n > 2:
        diffs_all = np.diff(centers, axis=0)
        curv_list = []
        _MIN_SPEED_FOR_CURVATURE = 1.0  # at least 1 px/frame to avoid blow-up
        for i in range(len(diffs_all) - 1):
            v1 = diffs_all[i]
            v2 = diffs_all[i + 1]
            cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
            norm_v1 = np.linalg.norm(v1)
            if norm_v1 > _MIN_SPEED_FOR_CURVATURE:
                curv_list.append(cross / (norm_v1 ** 3 + 1e-9))
            else:
                curv_list.append(0.0)
        curvatures = np.array(curv_list)

    # 角度變化
    angular_changes = np.zeros(0)
    if n > 2:
        diffs_all = np.diff(centers, axis=0)
        angles = np.arctan2(diffs_all[:, 1], diffs_all[:, 0])
        angular_changes = np.abs(np.diff(angles))
        # 處理角度 wrap-around
        angular_changes = np.minimum(angular_changes, 2 * np.pi - angular_changes)

    # 面積動態
    if areas.size > 1:
        area_changes = np.abs(np.diff(areas))
    else:
        area_changes = np.zeros(0)

    # ─── 組裝特徵 ───
    feat = OrderedDict()
    feat["num_points"] = float(n)
    feat["duration_frames"] = duration
    feat["displacement_x"] = disp_x
    feat["displacement_y"] = disp_y
    feat["net_displacement"] = net_disp
    feat["span_x"] = span_x
    feat["span_y"] = span_y
    feat["path_length"] = path_length
    feat["straightness_ratio"] = straightness

    feat["mean_speed"] = _safe_mean(speeds)
    feat["max_speed"] = float(np.max(speeds)) if speeds.size else 0.0
    feat["std_speed"] = _safe_std(speeds)
    feat["median_speed"] = float(np.median(speeds)) if speeds.size else 0.0
    feat["p95_speed"] = float(np.percentile(speeds, 95)) if speeds.size else 0.0
    feat["p5_speed"] = float(np.percentile(speeds, 5)) if speeds.size else 0.0

    feat["mean_acc"] = _safe_mean(accels)
    feat["max_acc"] = float(np.max(accels)) if accels.size else 0.0
    feat["std_acc"] = _safe_std(accels)

    feat["mean_jerk"] = _safe_mean(jerks)
    feat["max_jerk"] = float(np.max(jerks)) if jerks.size else 0.0
    feat["std_jerk"] = _safe_std(jerks)

    feat["curvature_mean"] = _safe_mean(curvatures)
    feat["curvature_std"] = _safe_std(curvatures)

    feat["angular_change_mean"] = _safe_mean(angular_changes)
    feat["angular_change_std"] = _safe_std(angular_changes)

    feat["area_mean"] = _safe_mean(areas)
    feat["area_std"] = _safe_std(areas)
    feat["area_range"] = float(np.ptp(areas)) if areas.size else 0.0
    feat["area_change_mean"] = _safe_mean(area_changes)
    feat["area_change_std"] = _safe_std(area_changes)

    return feat


# ═══════════════════════════════════════════════════════════════════════
# CSA Static features
# ═══════════════════════════════════════════════════════════════════════

CSA_FEATURE_KEYS: List[str] = [
    "csa_first_area",
    "csa_last_area",
    "csa_first_perimeter",
    "csa_last_perimeter",
    "csa_first_eq_diameter",
    "csa_last_eq_diameter",
    "csa_first_circularity",
    "csa_last_circularity",
]


def _compute_csa_features(
    samples: Sequence[FramePrediction],
) -> Dict[str, float]:
    """從第一幀和最後一幀的分割結果提取 Cross-Sectional Area 靜態特徵。"""
    zeros = OrderedDict((k, 0.0) for k in CSA_FEATURE_KEYS)

    usable = [s for s in samples if s.segmentation and s.segmentation.stats]
    if not usable:
        return zeros

    first_seg = usable[0].segmentation.stats
    last_seg = usable[-1].segmentation.stats

    def _circularity(area: float, perimeter: float) -> float:
        if perimeter <= 0:
            return 0.0
        return (4.0 * np.pi * area) / (perimeter ** 2)

    feat = OrderedDict()
    feat["csa_first_area"] = float(first_seg.area_px)
    feat["csa_last_area"] = float(last_seg.area_px)
    feat["csa_first_perimeter"] = float(first_seg.perimeter_px)
    feat["csa_last_perimeter"] = float(last_seg.perimeter_px)
    feat["csa_first_eq_diameter"] = float(first_seg.equivalent_diameter_px)
    feat["csa_last_eq_diameter"] = float(last_seg.equivalent_diameter_px)
    feat["csa_first_circularity"] = _circularity(first_seg.area_px, first_seg.perimeter_px)
    feat["csa_last_circularity"] = _circularity(last_seg.area_px, last_seg.perimeter_px)
    return feat


# ═══════════════════════════════════════════════════════════════════════
# Texture features
# ═══════════════════════════════════════════════════════════════════════

_RAW_TEXTURE_KEYS_PER_FRAME: List[str] = [
    # Gray-level stats
    "tex_mean",
    "tex_std",
    "tex_skewness",
    "tex_kurtosis",
    # Gradient
    "tex_grad_mean",
    "tex_grad_std",
    # GLCM
    "tex_glcm_contrast",
    "tex_glcm_dissimilarity",
    "tex_glcm_homogeneity",
    "tex_glcm_energy",
    "tex_glcm_correlation",
    # LBP histogram (10 bins from uniform LBP)
    "tex_lbp_00",
    "tex_lbp_01",
    "tex_lbp_02",
    "tex_lbp_03",
    "tex_lbp_04",
    "tex_lbp_05",
    "tex_lbp_06",
    "tex_lbp_07",
    "tex_lbp_08",
    "tex_lbp_09",
]


def _compute_glcm_features(gray: np.ndarray) -> Dict[str, float]:
    """簡單的 GLCM 特徵計算 (不依賴 skimage)。使用向量化加速。"""
    # 量化到 16 級
    levels = 16
    g = (gray.astype(np.float32) / 255.0 * (levels - 1)).astype(np.int32)
    g = np.clip(g, 0, levels - 1)

    # 水平方向 co-occurrence (向量化)
    left = g[:, :-1].ravel()
    right = g[:, 1:].ravel()
    glcm = np.zeros((levels, levels), dtype=np.float64)
    np.add.at(glcm, (left, right), 1)
    np.add.at(glcm, (right, left), 1)

    total = glcm.sum()
    if total > 0:
        glcm /= total

    # 使用向量化計算 GLCM 特性
    ii, jj = np.meshgrid(np.arange(levels), np.arange(levels), indexing="ij")
    contrast = float(np.sum((ii - jj) ** 2 * glcm))
    diss = float(np.sum(np.abs(ii - jj) * glcm))
    homo = float(np.sum(glcm / (1.0 + (ii - jj) ** 2)))
    energy = float(np.sum(glcm ** 2))

    mu_i = float(np.sum(ii * glcm))
    mu_j = float(np.sum(jj * glcm))
    var_i = float(np.sum((ii - mu_i) ** 2 * glcm))
    var_j = float(np.sum((jj - mu_j) ** 2 * glcm))
    std_i = np.sqrt(var_i) + 1e-9
    std_j = np.sqrt(var_j) + 1e-9
    corr = float(np.sum((ii - mu_i) * (jj - mu_j) * glcm / (std_i * std_j)))

    return {
        "tex_glcm_contrast": contrast,
        "tex_glcm_dissimilarity": diss,
        "tex_glcm_homogeneity": homo,
        "tex_glcm_energy": energy,
        "tex_glcm_correlation": corr,
    }


def _compute_lbp_histogram(gray: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """計算 LBP (Local Binary Pattern) 直方圖。

    使用 8 鄰居 circular LBP, radius=1。向量化實作。
    """
    h, w = gray.shape
    if h < 3 or w < 3:
        return np.zeros(n_bins, dtype=np.float64)

    center = gray[1: h - 1, 1: w - 1].astype(np.int16)
    lbp = np.zeros_like(center, dtype=np.uint8)

    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for bit, (dy, dx) in enumerate(offsets):
        neighbor = gray[1 + dy: h - 1 + dy, 1 + dx: w - 1 + dx].astype(np.int16)
        lbp |= ((neighbor >= center).astype(np.uint8) << bit)

    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, 256))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def _extract_texture_from_patch(gray_patch: np.ndarray) -> Dict[str, float]:
    """從灰階 patch 提取紋理特徵。"""
    patch_f = gray_patch.astype(np.float64)
    mean_val = float(np.mean(patch_f))
    std_val = float(np.std(patch_f))

    # Skewness / Kurtosis
    centered = patch_f - mean_val
    if std_val > 1e-9 and patch_f.size > 0:
        skewness = float(np.mean(centered ** 3) / (std_val ** 3))
        kurtosis = float(np.mean(centered ** 4) / (std_val ** 4) - 3.0)
    else:
        skewness = 0.0
        kurtosis = 0.0

    # Gradient (Sobel)
    grad_x = cv2.Sobel(patch_f, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(patch_f, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x.astype(np.float32), grad_y.astype(np.float32))
    grad_mean = float(np.mean(grad_mag))
    grad_std = float(np.std(grad_mag))

    # GLCM features
    glcm_feats = _compute_glcm_features(gray_patch)

    # LBP histogram
    lbp_hist = _compute_lbp_histogram(gray_patch, n_bins=10)

    result = {
        "tex_mean": mean_val,
        "tex_std": std_val,
        "tex_skewness": skewness,
        "tex_kurtosis": kurtosis,
        "tex_grad_mean": grad_mean,
        "tex_grad_std": grad_std,
    }
    result.update(glcm_feats)
    for i, val in enumerate(lbp_hist):
        result[f"tex_lbp_{i:02d}"] = float(val)

    return result


def _read_segmented_patch(
    video_path: str,
    sample: FramePrediction,
    target_size: int = 96,
) -> Optional[np.ndarray]:
    """從影片中讀取該幀的 bbox patch 並轉灰階。"""
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
        if patch.size == 0:
            return None
        if patch.ndim == 3:
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        if target_size > 0:
            patch = cv2.resize(patch, (target_size, target_size), interpolation=cv2.INTER_AREA)
        return patch
    finally:
        cap.release()


# ═══════════════════════════════════════════════════════════════════════
# Subject-level aggregation helper
# ═══════════════════════════════════════════════════════════════════════

def _aggregate_video_features(
    video_features: Sequence[Dict[str, float]],
    video_keys: List[str],
    subject_stats: List[str],
) -> Dict[str, float]:
    """將多個 video-level 特徵聚合為 subject-level 特徵。"""
    subject_keys = ["video_count"]
    for stat in subject_stats:
        for key in video_keys:
            subject_keys.append(f"{stat}__{key}")

    if not video_features:
        return OrderedDict((k, 0.0) for k in subject_keys)

    agg: Dict[str, float] = OrderedDict()
    agg["video_count"] = float(len(video_features))

    matrix = {
        key: np.asarray([vf.get(key, 0.0) for vf in video_features], dtype=np.float64)
        for key in video_keys
    }

    stat_funcs = {
        "mean": lambda a: _safe_mean(a),
        "std": lambda a: _safe_std(a),
        "min": lambda a: float(np.min(a)) if a.size else 0.0,
        "max": lambda a: float(np.max(a)) if a.size else 0.0,
        "median": lambda a: float(np.median(a)) if a.size else 0.0,
    }

    for stat in subject_stats:
        func = stat_funcs.get(stat, stat_funcs["mean"])
        for key, values in matrix.items():
            agg[f"{stat}__{key}"] = func(values)

    for k in subject_keys:
        agg.setdefault(k, 0.0)
    return agg


def _build_subject_keys(video_keys: List[str], subject_stats: List[str]) -> List[str]:
    keys = ["video_count"]
    for stat in subject_stats:
        for key in video_keys:
            keys.append(f"{stat}__{key}")
    return keys



