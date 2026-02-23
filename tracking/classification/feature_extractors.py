"""Feature extractors for the classification pipeline.

提供兩種特徵提取方法：
1. ``motion_only``  — 僅運動特徵 (含卡爾曼濾波平滑)
2. ``motion_texture`` — 運動特徵 + 紋理特徵 + CSA 靜態特徵 (紋理過多時以 PCA 降維)

原有的 basic / texture_hybrid / backbone_texture / segmentation_motion 已移除。
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


# ───────────────────── Kalman Filter ─────────────────────

class _KalmanSmoother2D:
    """簡易二維卡爾曼濾波器，用於平滑 centroid 軌跡。

    狀態: [x, y, vx, vy]
    觀測: [x, y]
    """

    def __init__(
        self,
        process_noise: float = 1e-2,
        measurement_noise: float = 1e-1,
    ) -> None:
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    def smooth(
        self,
        observations: np.ndarray,
        frame_indices: np.ndarray,
    ) -> np.ndarray:
        """Forward-backward Kalman smoothing (RTS smoother).

        Parameters
        ----------
        observations : (N, 2) centroid positions
        frame_indices : (N,) frame numbers (used to compute dt)

        Returns
        -------
        smoothed : (N, 2)
        """
        n = observations.shape[0]
        if n <= 2:
            return observations.copy()

        dt_arr = np.diff(frame_indices).astype(np.float64)
        dt_arr[dt_arr == 0] = 1.0

        # Forward pass
        dim_x, dim_z = 4, 2
        x = np.zeros(dim_x)
        x[:2] = observations[0]
        P = np.eye(dim_x) * 1.0
        Q_base = np.eye(dim_x) * self.process_noise
        R = np.eye(dim_z) * self.measurement_noise
        H = np.zeros((dim_z, dim_x))
        H[0, 0] = H[1, 1] = 1.0

        xs_fwd = np.zeros((n, dim_x))
        Ps_fwd = np.zeros((n, dim_x, dim_x))
        xs_fwd[0] = x
        Ps_fwd[0] = P

        for k in range(1, n):
            dt = dt_arr[k - 1]
            F = np.eye(dim_x)
            F[0, 2] = dt
            F[1, 3] = dt
            Q = Q_base * (dt ** 2)

            # Predict
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q

            # Update
            z = observations[k]
            y_innov = z - H @ x_pred
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ np.linalg.inv(S)
            x = x_pred + K @ y_innov
            P = (np.eye(dim_x) - K @ H) @ P_pred

            xs_fwd[k] = x
            Ps_fwd[k] = P

        # Backward (RTS) smoothing
        xs_smooth = xs_fwd.copy()
        for k in range(n - 2, -1, -1):
            dt = dt_arr[k]
            F = np.eye(dim_x)
            F[0, 2] = dt
            F[1, 3] = dt
            Q = Q_base * (dt ** 2)

            P_pred = F @ Ps_fwd[k] @ F.T + Q
            try:
                G = Ps_fwd[k] @ F.T @ np.linalg.inv(P_pred)
            except np.linalg.LinAlgError:
                continue
            xs_smooth[k] = xs_fwd[k] + G @ (xs_smooth[k + 1] - F @ xs_fwd[k])

        return xs_smooth[:, :2]


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
    """從追蹤樣本計算運動特徵 (含卡爾曼平滑)。"""
    zeros = OrderedDict((k, 0.0) for k in MOTION_FEATURE_KEYS)
    if not samples:
        return zeros

    frames = np.asarray([float(s.frame_index) for s in samples], dtype=np.float64)
    raw_centers = np.asarray([s.center for s in samples], dtype=np.float64)
    bboxes = np.asarray([s.bbox for s in samples], dtype=np.float64)
    areas = bboxes[:, 2] * bboxes[:, 3]

    # ─── 卡爾曼平滑 ───
    smoother = _KalmanSmoother2D(kalman_process_noise, kalman_measurement_noise)
    centers = smoother.smooth(raw_centers, frames)

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
        for i in range(len(diffs_all) - 1):
            v1 = diffs_all[i]
            v2 = diffs_all[i + 1]
            cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
            norm_v1 = np.linalg.norm(v1)
            if norm_v1 > 1e-9:
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


# ═══════════════════════════════════════════════════════════════════════
# Extractor 1: Motion Only (with Kalman filtering)
# ═══════════════════════════════════════════════════════════════════════


@register_feature_extractor("motion_only")
class MotionOnlyFeatureExtractor(TrajectoryFeatureExtractor):
    """僅提取運動特徵，使用卡爾曼濾波平滑 centroid 軌跡。

    運動特徵 (31 維):
    位移/路徑長度/直線度比/速度統計/加速度統計/急動度統計/曲率/角度變化/面積動態
    """

    name = "MotionOnlyFeatures"
    DEFAULT_CONFIG = {
        "kalman_process_noise": 1e-2,
        "kalman_measurement_noise": 1e-1,
    }

    def __init__(self, params: Dict[str, Any] | None = None):
        cfg = params or {}
        self._kalman_process_noise = float(cfg.get("kalman_process_noise", 1e-2))
        self._kalman_measurement_noise = float(cfg.get("kalman_measurement_noise", 1e-1))
        self._video_keys = list(MOTION_FEATURE_KEYS)
        self._subject_stats = [str(s) for s in cfg.get("aggregate_stats", ["mean", "std", "min", "max"])]
        self._subject_keys = _build_subject_keys(self._video_keys, self._subject_stats)

    def feature_order(self, level: str = "video") -> Sequence[str]:
        if str(level).lower() == "subject":
            return self._subject_keys
        return self._video_keys

    def extract_video(
        self,
        samples: Sequence[FramePrediction],
        video_path: Optional[str] = None,
    ) -> Dict[str, float]:
        return _compute_motion_features(
            samples,
            self._kalman_process_noise,
            self._kalman_measurement_noise,
        )

    def aggregate_subject(self, video_features: Sequence[Dict[str, float]]) -> Dict[str, float]:
        return _aggregate_video_features(video_features, self._video_keys, self._subject_stats)


# ═══════════════════════════════════════════════════════════════════════
# Extractor 2: Motion + Texture + CSA (with PCA reduction for texture)
# ═══════════════════════════════════════════════════════════════════════


@register_feature_extractor("motion_texture")
class MotionTextureFeatureExtractor(TrajectoryFeatureExtractor):
    """運動特徵 + 紋理特徵 + CSA 靜態特徵。

    特徵結構:
    - 運動特徵 (31 維): 含卡爾曼平滑
    - CSA 靜態特徵 (8 維): 第一幀 / 最後一幀的面積/周長/等效直徑/圓度
    - 紋理特徵: 第一幀和最後一幀的灰階紋理描述子 (GLCM + LBP + 統計量)
      → 降維至 dim(motion) + dim(CSA) = 39 維

    紋理降維策略:
    - 原始紋理特徵 = 2 frames × 21 features = 42 維
    - 目標紋理維度 = len(motion_keys) + len(csa_keys) = 39 維
    - 使用 PCA 將 42 維降至 39 維
    """

    name = "MotionTextureFeatures"
    DEFAULT_CONFIG = {
        "kalman_process_noise": 1e-2,
        "kalman_measurement_noise": 1e-1,
        "texture_patch_size": 96,
    }

    def __init__(self, params: Dict[str, Any] | None = None):
        cfg = params or {}
        self._kalman_process_noise = float(cfg.get("kalman_process_noise", 1e-2))
        self._kalman_measurement_noise = float(cfg.get("kalman_measurement_noise", 1e-1))
        self._patch_size = int(cfg.get("texture_patch_size", 96))
        self._subject_stats = [str(s) for s in cfg.get("aggregate_stats", ["mean", "std", "min", "max"])]

        # 運動特徵 keys
        self._motion_keys = list(MOTION_FEATURE_KEYS)
        # CSA 靜態特徵 keys
        self._csa_keys = list(CSA_FEATURE_KEYS)
        # 紋理目標維度 = motion + CSA
        self._texture_target_dim = len(self._motion_keys) + len(self._csa_keys)

        # 原始紋理特徵 per frame (21) × 2 frames = 42
        self._raw_tex_per_frame = len(_RAW_TEXTURE_KEYS_PER_FRAME)
        self._raw_tex_total = self._raw_tex_per_frame * 2  # first + last frame

        # PCA 狀態
        self._pca_mean: Optional[np.ndarray] = None
        self._pca_components: Optional[np.ndarray] = None

        # 紋理 key names
        self._texture_keys = [f"tex_pca_{i:03d}" for i in range(self._texture_target_dim)]

        # 完整 video keys
        self._video_keys = self._motion_keys + self._csa_keys + self._texture_keys
        self._subject_keys = _build_subject_keys(self._video_keys, self._subject_stats)

    def feature_order(self, level: str = "video") -> Sequence[str]:
        if str(level).lower() == "subject":
            return self._subject_keys
        return self._video_keys

    def extract_video(
        self,
        samples: Sequence[FramePrediction],
        video_path: Optional[str] = None,
    ) -> Dict[str, float]:
        # 1. Motion features
        motion = _compute_motion_features(
            samples,
            self._kalman_process_noise,
            self._kalman_measurement_noise,
        )

        # 2. CSA features
        csa = _compute_csa_features(samples)

        # 3. Texture features (first + last frame)
        raw_texture = self._extract_raw_texture(samples, video_path)

        combined = OrderedDict()
        for k in self._motion_keys:
            combined[k] = motion.get(k, 0.0)
        for k in self._csa_keys:
            combined[k] = csa.get(k, 0.0)

        # 暫存原始紋理向量，後續批量做 PCA
        combined["_raw_texture_vec"] = raw_texture
        # 佔位
        for k in self._texture_keys:
            combined[k] = 0.0

        return combined

    def aggregate_subject(self, video_features: Sequence[Dict[str, float]]) -> Dict[str, float]:
        # 先移除暫存 key
        clean_features = []
        for vf in video_features:
            cleaned = {k: v for k, v in vf.items() if not k.startswith("_")}
            clean_features.append(cleaned)
        return _aggregate_video_features(clean_features, self._video_keys, self._subject_stats)

    def finalize_batch(
        self,
        features_list: Sequence[Dict[str, float]],
        *,
        fit: bool = True,
    ) -> Sequence[Dict[str, float]]:
        """批量對紋理特徵做 PCA 降維。由 engine 在 vectorization 前呼叫。

        Parameters
        ----------
        fit : bool
            True → 擬合 PCA 然後轉換 (訓練集)。
            False → 使用已擬合的 PCA 轉換 (測試集)。
        """
        # 收集原始紋理向量
        raw_vecs = []
        indices = []
        for i, feat in enumerate(features_list):
            rv = feat.get("_raw_texture_vec")
            if rv is not None:
                raw_vecs.append(rv)
                indices.append(i)

        if not raw_vecs:
            return features_list

        X_raw = np.array(raw_vecs, dtype=np.float64)

        # PCA 降維
        if fit:
            reduced = self._reduce_texture(X_raw)
        else:
            reduced = self._apply_pca(X_raw)

        # 回寫
        result = list(features_list)
        for idx, row_idx in enumerate(indices):
            feat = dict(result[row_idx])
            for j, k in enumerate(self._texture_keys):
                feat[k] = float(reduced[idx, j]) if j < reduced.shape[1] else 0.0
            # 移除暫存 key
            feat.pop("_raw_texture_vec", None)
            result[row_idx] = feat

        return result

    def _reduce_texture(self, X: np.ndarray) -> np.ndarray:
        """PCA 降維紋理特徵。"""
        target_dim = self._texture_target_dim
        n_samples, n_features = X.shape

        if n_features <= target_dim:
            # 不需降維，直接 pad
            if n_features < target_dim:
                pad = np.zeros((n_samples, target_dim - n_features))
                return np.hstack([X, pad])
            return X

        # 減均值
        mean = X.mean(axis=0)
        X_centered = X - mean

        # SVD-based PCA
        actual_components = min(target_dim, n_samples, n_features)
        try:
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            components = Vt[:actual_components]
            reduced = X_centered @ components.T

            # 儲存 PCA 參數供後續使用
            self._pca_mean = mean
            self._pca_components = components
        except np.linalg.LinAlgError:
            # fallback: truncate
            reduced = X_centered[:, :target_dim]

        # 確保輸出維度
        if reduced.shape[1] < target_dim:
            pad = np.zeros((n_samples, target_dim - reduced.shape[1]))
            reduced = np.hstack([reduced, pad])

        return reduced

    def _apply_pca(self, X: np.ndarray) -> np.ndarray:
        """使用已擬合的 PCA 參數進行轉換 (測試集)。"""
        target_dim = self._texture_target_dim
        if self._pca_mean is None or self._pca_components is None:
            # 尚未擬合 → fallback 到擬合模式
            return self._reduce_texture(X)

        n_samples = X.shape[0]
        X_centered = X - self._pca_mean
        try:
            reduced = X_centered @ self._pca_components.T
        except (ValueError, np.linalg.LinAlgError):
            reduced = X_centered[:, :target_dim]

        if reduced.shape[1] < target_dim:
            pad = np.zeros((n_samples, target_dim - reduced.shape[1]))
            reduced = np.hstack([reduced, pad])
        return reduced

    def _extract_raw_texture(
        self,
        samples: Sequence[FramePrediction],
        video_path: Optional[str],
    ) -> List[float]:
        """提取第一幀和最後一幀的原始紋理特徵向量。"""
        zeros = [0.0] * self._raw_tex_total

        if not samples or not video_path:
            return zeros

        # 取第一幀和最後一幀
        first_sample = samples[0]
        last_sample = samples[-1] if len(samples) > 1 else samples[0]

        textures: List[float] = []
        for sample in [first_sample, last_sample]:
            patch = _read_segmented_patch(video_path, sample, self._patch_size)
            if patch is None:
                textures.extend([0.0] * self._raw_tex_per_frame)
            else:
                tex_dict = _extract_texture_from_patch(patch)
                for key in _RAW_TEXTURE_KEYS_PER_FRAME:
                    textures.append(tex_dict.get(key, 0.0))

        return textures
