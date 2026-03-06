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
    _KalmanSmoother2D,
    _safe_mean,
    _safe_std,
    MOTION_FEATURE_KEYS,
)
from .texture_resnet import MaskedROIResNetExtractor, RESNET_FEAT_DIM

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════════════════════════════════

N_GEO_VARS: int = 10           # Geometric time-series channels
N_TEX_PCA_TS: int = 8          # ResNet PCA texture channels for TSC
N_TS_VARS: int = N_GEO_VARS + N_TEX_PCA_TS   # 18 total TS channels
N_TS_STEPS: int = 128          # Fixed time-series length (pad / resample)

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
    "ts_cx",            # centroid x (normalised)
    "ts_cy",            # centroid y (normalised)
    "ts_bw",            # bbox width  (normalised)
    "ts_bh",            # bbox height (normalised)
    "ts_area",          # bbox area   (normalised)
    "ts_seg_area",      # seg mask area (normalised)
    "ts_circularity",   # seg circularity
    "ts_eq_diam",       # equivalent diameter (normalised)
    "ts_speed",         # inter-frame centroid speed (normalised)
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

    usable = [s for s in samples if s.segmentation and s.segmentation.stats]

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


def _extract_ts_geo_channels(
    samples: Sequence[FramePrediction],
    frame_w: float = 640.0,
    frame_h: float = 480.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build ``(N_GEO_VARS, N_TS_STEPS)`` geometric channel matrix.

    Returns
    -------
    geo : ndarray, shape ``(N_GEO_VARS, N_TS_STEPS)``
    resample_idx : ndarray of int
        Indices into the original *samples* list that were selected.
        Length ``N_TS_STEPS`` if ``len(samples) >= N_TS_STEPS``,
        otherwise ``len(samples)`` (short videos are zero-padded).
    """
    n_raw = len(samples)
    raw = np.zeros((N_GEO_VARS, n_raw), dtype=np.float32)

    bboxes = np.array([s.bbox for s in samples], dtype=np.float32)
    max_w = float(bboxes[:, 2].max() + 1e-9)
    max_h = float(bboxes[:, 3].max() + 1e-9)
    max_area = float((bboxes[:, 2] * bboxes[:, 3]).max() + 1e-9)

    seg_areas = np.array(
        [
            s.segmentation.stats.area_px
            if (s.segmentation and s.segmentation.stats)
            else 0.0
            for s in samples
        ],
        dtype=np.float32,
    )
    max_seg_area = float(seg_areas.max() + 1e-9)
    eq_diams = np.array(
        [
            s.segmentation.stats.equivalent_diameter_px
            if (s.segmentation and s.segmentation.stats)
            else 0.0
            for s in samples
        ],
        dtype=np.float32,
    )
    max_eq_diam = float(eq_diams.max() + 1e-9)

    centers = bboxes[:, :2] + bboxes[:, 2:] / 2.0
    speeds = np.zeros(n_raw, dtype=np.float32)
    if n_raw > 1:
        step_dist = np.linalg.norm(
            np.diff(centers, axis=0), axis=1
        ).astype(np.float32)
        speeds[1:] = step_dist
        max_speed = float(step_dist.max() + 1e-9)
        speeds /= max_speed

    for i, s in enumerate(samples):
        x, y, bw, bh = s.bbox
        raw[0, i] = float((x + bw / 2.0) / frame_w)
        raw[1, i] = float((y + bh / 2.0) / frame_h)
        raw[2, i] = float(bw / max_w)
        raw[3, i] = float(bh / max_h)
        raw[4, i] = float((bw * bh) / max_area)
        raw[5, i] = float(seg_areas[i] / max_seg_area)
        raw[6, i] = float(
            _circularity(
                s.segmentation.stats.area_px,
                s.segmentation.stats.perimeter_px,
            )
            if (s.segmentation and s.segmentation.stats)
            else 0.0
        )
        raw[7, i] = float(eq_diams[i] / max_eq_diam)
        raw[8, i] = speeds[i]
        raw[9, i] = float(bw / (bh + 1e-9))

    # Resample / pad to N_TS_STEPS
    geo = np.zeros((N_GEO_VARS, N_TS_STEPS), dtype=np.float32)
    if n_raw >= N_TS_STEPS:
        resample_idx = np.round(
            np.linspace(0, n_raw - 1, N_TS_STEPS)
        ).astype(int)
        geo = raw[:, resample_idx]
    else:
        resample_idx = np.arange(n_raw)
        geo[:, :n_raw] = raw

    return geo, resample_idx


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
    - **Motion**  : 31 D — Kalman-smoothed trajectory features
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
        "kalman_process_noise": 1e-2,
        "kalman_measurement_noise": 1e-1,
        "resnet_device": "auto",
        "aggregate_stats": ["mean", "std", "min", "max"],
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        cfg = params or {}
        self._kpn = float(cfg.get("kalman_process_noise", 1e-2))
        self._kmn = float(cfg.get("kalman_measurement_noise", 1e-1))
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
        motion = _compute_motion_features(samples, self._kpn, self._kmn)
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

    def finalize_batch(
        self,
        features_list: Sequence[Dict[str, float]],
        *,
        fit: bool = True,
    ) -> Sequence[Dict[str, float]]:
        """Batch PCA on ResNet texture features."""
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
                reduced, self._pca_mean, self._pca_components = _pca_fit_transform(
                    X_raw, target
                )
            else:
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
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        cfg = params or {}
        self._frame_w = float(cfg.get("frame_w", 640.0))
        self._frame_h = float(cfg.get("frame_h", 480.0))
        self._resnet_device = str(cfg.get("resnet_device", "auto"))
        self._tex_pca_dim = int(cfg.get("tex_pca_dim", N_TEX_PCA_TS))

        self._flat_len = N_TS_VARS * N_TS_STEPS
        self._video_keys = [f"ts_{i:04d}" for i in range(self._flat_len)]
        self._subject_keys = self._video_keys  # same after mean

        # PCA state for ResNet texture (fitted in finalize_batch)
        self._pca_mean: Optional[np.ndarray] = None
        self._pca_components: Optional[np.ndarray] = None

    def feature_order(self, level: str = "video") -> Sequence[str]:
        return self._video_keys

    def extract_video(
        self,
        samples: Sequence[FramePrediction],
        video_path: Optional[str] = None,
    ) -> Dict[str, float]:
        # 1. Geometric channels → (N_GEO_VARS, N_TS_STEPS)
        geo, resample_idx = _extract_ts_geo_channels(
            samples, self._frame_w, self._frame_h
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

    def finalize_batch(
        self,
        features_list: Sequence[Dict[str, float]],
        *,
        fit: bool = True,
    ) -> Sequence[Dict[str, float]]:
        """Batch PCA on per-frame ResNet texture → fill channels 10–17.

        1. Collect ``_raw_resnet_ts`` matrices from all videos.
        2. Stack all frames → ``(N_videos × N_TS_STEPS, 512)``.
        3. Fit PCA (or apply) → ``(total, tex_pca_dim)``.
        4. Reshape back, write into texture channels of each video's
           flattened time series.
        """
        raw_mats, indices = [], []
        for i, feat in enumerate(features_list):
            rm = feat.get("_raw_resnet_ts")
            if rm is not None:
                raw_mats.append(np.asarray(rm, dtype=np.float64))
                indices.append(i)
        if not raw_mats:
            return features_list

        # Stack all frames across all videos
        all_frames = np.vstack(raw_mats)  # (total_frames, 512)
        target = self._tex_pca_dim

        if fit:
            reduced_all, self._pca_mean, self._pca_components = (
                _pca_fit_transform(all_frames, target)
            )
        else:
            if self._pca_mean is None or self._pca_components is None:
                reduced_all, self._pca_mean, self._pca_components = (
                    _pca_fit_transform(all_frames, target)
                )
            else:
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

        return result

    # ── Internal helpers ─────────────────────────────────────────────────

    def _extract_resnet_for_ts(
        self,
        samples: Sequence[FramePrediction],
        resample_idx: np.ndarray,
        video_path: Optional[str],
    ) -> np.ndarray:
        """Extract ResNet features for the resampled frame indices.

        Returns ``(N_TS_STEPS, RESNET_FEAT_DIM)`` float32.
        """
        result = np.zeros((N_TS_STEPS, RESNET_FEAT_DIM), dtype=np.float32)
        if not video_path or not samples:
            return result

        resnet = _get_resnet_extractor(self._resnet_device)
        if not resnet.available:
            return result

        # Select only the resampled samples (for efficiency)
        selected = [samples[int(i)] for i in resample_idx]
        feats = resnet.extract_from_video(video_path, selected)  # (len, 512)

        n_actual = min(feats.shape[0], N_TS_STEPS)
        result[:n_actual] = feats[:n_actual]
        return result
