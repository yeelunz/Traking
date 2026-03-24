from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None  # type: ignore

from ...core.interfaces import FramePrediction
from ...core.registry import register_feature_extractor
from ..interfaces import TrajectoryFeatureExtractor
from ..texture_backbone import TextureBackboneWrapper
from .v3pro import _extract_mean_roi_rgb
from ..trajectory_filter import smooth_trajectory_2d as _smooth_trajectory_2d


TAB_V4_MOTION_KEYS: List[str] = [
    "duration",
    "path_length",
    "straightness",
    "speed_mean",
    "speed_std",
    "acc_mean",
    "acc_std",
    "disp_mean",
    "disp_std",
    "disp_max",
    "heading_change_mean",
    "heading_change_std",
    "curve_amplitude",
    "curve_curvature",
    "curve_fit_r2",
]

TAB_V4_STATIC_KEYS: List[str] = [
    "csa_mean",
    "csa_std",
    "csa_strain",
    "swelling_ratio",
    "eq_diam_mean",
    "eq_diam_strain",
    "circularity_mean",
    "circularity_std",
    "aspect_ratio_mean",
    "aspect_ratio_std",
]

TAB_V4_TOTAL_DIM: int = 36


def _safe_std_v4(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.std(arr, ddof=0))


def _safe_mean_v4(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))


def _circularity_v4(area: float, perimeter: float) -> float:
    if perimeter <= 0:
        return 0.0
    return float(4.0 * np.pi * area / (perimeter ** 2))


def _compute_motion_tab_v4(samples: Sequence[FramePrediction]) -> Dict[str, float]:
    zeros = OrderedDict((k, 0.0) for k in TAB_V4_MOTION_KEYS)
    if not samples:
        return zeros

    ordered = sorted(samples, key=lambda s: float(s.frame_index))
    frames = np.asarray([float(s.frame_index) for s in ordered], dtype=np.float64)
    raw_centers = np.asarray([s.center for s in ordered], dtype=np.float64)
    try:
        centers = _smooth_trajectory_2d(raw_centers, frames)
    except Exception:
        centers = raw_centers
    n = int(len(ordered))

    feat: Dict[str, float] = OrderedDict()
    feat["duration"] = float(frames[-1] - frames[0]) if n > 1 else 0.0

    if n > 1:
        diffs = np.diff(centers, axis=0)
        dt = np.diff(frames)
        dt[dt == 0] = 1.0
        step_dist = np.linalg.norm(diffs, axis=1)
        path_length = float(np.sum(step_dist))
        speeds = step_dist / dt
    else:
        path_length = 0.0
        speeds = np.zeros(0, dtype=np.float64)

    net_disp = float(np.linalg.norm(centers[-1] - centers[0])) if n > 1 else 0.0
    feat["path_length"] = path_length
    feat["straightness"] = float(net_disp / max(path_length, 1e-9))
    feat["speed_mean"] = _safe_mean_v4(speeds)
    feat["speed_std"] = _safe_std_v4(speeds)

    if speeds.size > 1:
        accels = np.abs(np.diff(speeds))
    else:
        accels = np.zeros(0, dtype=np.float64)
    feat["acc_mean"] = _safe_mean_v4(accels)
    feat["acc_std"] = _safe_std_v4(accels)

    init_eq_diam = 0.0
    for s in ordered:
        if s.segmentation and s.segmentation.stats:
            init_eq_diam = float(
                getattr(s.segmentation.stats, "equivalent_diameter_px", 0.0) or 0.0
            )
            if init_eq_diam > 0:
                break
    if init_eq_diam <= 0:
        bw0, bh0 = float(ordered[0].bbox[2]), float(ordered[0].bbox[3])
        init_eq_diam = float(np.sqrt(max(bw0 * bh0, 1e-9)))
    scale_ref = max(init_eq_diam, 1e-9)

    median_pos = np.median(centers, axis=0)
    rel_displacements = np.linalg.norm(centers - median_pos, axis=1) / scale_ref
    feat["disp_mean"] = float(rel_displacements.mean())
    feat["disp_std"] = _safe_std_v4(rel_displacements)
    feat["disp_max"] = float(rel_displacements.max())

    if n > 2:
        diffs_all = np.diff(centers, axis=0)
        angles = np.arctan2(diffs_all[:, 1], diffs_all[:, 0])
        ang_changes = np.abs(np.diff(angles))
        ang_changes = np.minimum(ang_changes, 2.0 * np.pi - ang_changes)
        feat["heading_change_mean"] = _safe_mean_v4(ang_changes)
        feat["heading_change_std"] = _safe_std_v4(ang_changes)
    else:
        feat["heading_change_mean"] = 0.0
        feat["heading_change_std"] = 0.0

    if n >= 3:
        t = np.linspace(0.0, 1.0, n, dtype=np.float64)
        cx = centers[:, 0]
        cy = centers[:, 1]
        try:
            coeff_x = np.polyfit(t, cx, deg=2)
            coeff_y = np.polyfit(t, cy, deg=2)

            fit_x = np.polyval(coeff_x, t)
            fit_y = np.polyval(coeff_y, t)
            residual = np.sqrt((cx - fit_x) ** 2 + (cy - fit_y) ** 2)
            feat["curve_amplitude"] = float(np.max(np.cumsum(residual)))

            dx_dt = 2.0 * coeff_x[0] * t + coeff_x[1]
            dy_dt = 2.0 * coeff_y[0] * t + coeff_y[1]
            d2x = np.full_like(t, 2.0 * coeff_x[0])
            d2y = np.full_like(t, 2.0 * coeff_y[0])
            num = np.abs(dx_dt * d2y - dy_dt * d2x)
            den = np.power(dx_dt * dx_dt + dy_dt * dy_dt, 1.5) + 1e-9
            curvature = num / den
            feat["curve_curvature"] = float(np.mean(curvature))

            sse = float(np.sum((cx - fit_x) ** 2 + (cy - fit_y) ** 2))
            sst = float(np.sum((cx - np.mean(cx)) ** 2 + (cy - np.mean(cy)) ** 2))
            feat["curve_fit_r2"] = float(1.0 - sse / (sst + 1e-9)) if sst > 1e-9 else 0.0
        except Exception:  # noqa: BLE001
            feat["curve_amplitude"] = 0.0
            feat["curve_curvature"] = 0.0
            feat["curve_fit_r2"] = 0.0
    else:
        feat["curve_amplitude"] = 0.0
        feat["curve_curvature"] = 0.0
        feat["curve_fit_r2"] = 0.0

    for key in TAB_V4_MOTION_KEYS:
        value = float(feat.get(key, 0.0))
        feat[key] = value if np.isfinite(value) else 0.0
    return feat


def _compute_static_tab_v4(samples: Sequence[FramePrediction]) -> Dict[str, float]:
    zeros = OrderedDict((k, 0.0) for k in TAB_V4_STATIC_KEYS)
    if not samples:
        return zeros

    usable = [
        s
        for s in samples
        if s.segmentation and s.segmentation.stats and getattr(s.segmentation.stats, "area_px", 0.0) > 0
    ]

    feat: Dict[str, float] = OrderedDict()
    if usable:
        areas = np.array([s.segmentation.stats.area_px for s in usable], dtype=np.float64)
        eq_diams = np.array(
            [s.segmentation.stats.equivalent_diameter_px for s in usable],
            dtype=np.float64,
        )
        circs = np.array(
            [
                _circularity_v4(
                    s.segmentation.stats.area_px,
                    s.segmentation.stats.perimeter_px,
                )
                for s in usable
            ],
            dtype=np.float64,
        )
    else:
        areas = np.zeros(0, dtype=np.float64)
        eq_diams = np.zeros(0, dtype=np.float64)
        circs = np.zeros(0, dtype=np.float64)

    if areas.size:
        init_area = max(float(areas[0]), 1e-9)
        feat["csa_mean"] = float(areas.mean())
        feat["csa_std"] = _safe_std_v4(areas)
        feat["csa_strain"] = float((areas.max() - areas.min()) / init_area)
        feat["swelling_ratio"] = float(areas.max() / init_area)
    else:
        feat["csa_mean"] = 0.0
        feat["csa_std"] = 0.0
        feat["csa_strain"] = 0.0
        feat["swelling_ratio"] = 0.0

    if eq_diams.size:
        init_eq = max(float(eq_diams[0]), 1e-9)
        feat["eq_diam_mean"] = float(eq_diams.mean())
        feat["eq_diam_strain"] = float((eq_diams.max() - eq_diams.min()) / init_eq)
    else:
        feat["eq_diam_mean"] = 0.0
        feat["eq_diam_strain"] = 0.0

    if circs.size:
        feat["circularity_mean"] = float(circs.mean())
        feat["circularity_std"] = _safe_std_v4(circs)
    else:
        feat["circularity_mean"] = 0.0
        feat["circularity_std"] = 0.0

    bboxes = np.array([s.bbox for s in samples], dtype=np.float64)
    if bboxes.size > 0:
        aspects = bboxes[:, 2] / (bboxes[:, 3] + 1e-9)
        feat["aspect_ratio_mean"] = float(aspects.mean())
        feat["aspect_ratio_std"] = _safe_std_v4(aspects)
    else:
        feat["aspect_ratio_mean"] = 0.0
        feat["aspect_ratio_std"] = 0.0

    for key in TAB_V4_STATIC_KEYS:
        value = float(feat.get(key, 0.0))
        feat[key] = value if np.isfinite(value) else 0.0
    return feat


def _normalize_texture_mode_v4(mode: str) -> str:
    mode_lc = str(mode or "freeze").strip().lower()
    aliases = {
        "freez": "freeze",
        "frozen": "freeze",
        "pretrained": "pretrain",
    }
    mode_lc = aliases.get(mode_lc, mode_lc)
    if mode_lc not in {"freeze", "learnable", "pretrain"}:
        return "freeze"
    return mode_lc


@register_feature_extractor("tab_v4")
class MotionStaticV4FeatureExtractor(TrajectoryFeatureExtractor):
    name = "MotionStaticV4Features"
    DEFAULT_CONFIG: Dict[str, Any] = {
        "n_texture_frames": 5,
        "texture_image_size": 96,
        "roi_pad_ratio": 0.15,
        "texture_mode": "freeze",  # freeze | learnable | pretrain
        "texture_backbone": "convnext_tiny",
        "texture_pretrain_ckpt": None,
        "pretrained_backbone": True,
        "texture_device": "auto",
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        cfg = {**self.DEFAULT_CONFIG, **(params or {})}
        self._n_texture_frames = int(cfg.get("n_texture_frames", 5))
        self._texture_image_size = int(cfg.get("texture_image_size", 96))
        self._roi_pad_ratio = float(cfg.get("roi_pad_ratio", 0.15))
        self._texture_mode = _normalize_texture_mode_v4(str(cfg.get("texture_mode", "freeze")))
        self._texture_backbone = str(cfg.get("texture_backbone", "convnext_tiny"))
        self._texture_pretrain_ckpt = cfg.get("texture_pretrain_ckpt")
        self._pretrained_backbone = bool(cfg.get("pretrained_backbone", True))
        self._texture_device = str(cfg.get("texture_device", "auto"))
        self._texture_wrapper: Optional[TextureBackboneWrapper] = None

        self._motion_keys = list(TAB_V4_MOTION_KEYS)
        self._static_keys = list(TAB_V4_STATIC_KEYS)
        self._non_deep_dim = len(self._motion_keys) + len(self._static_keys)
        self._texture_dim = TAB_V4_TOTAL_DIM - self._non_deep_dim
        self._texture_keys = [f"cnn_features_{i:02d}" for i in range(self._texture_dim)]

        self._video_keys = self._motion_keys + self._static_keys + self._texture_keys
        self._subject_keys = list(self._video_keys)

        self._pca_mean: Optional[np.ndarray] = None
        self._pca_components: Optional[np.ndarray] = None
        self._feat_mean: Optional[np.ndarray] = None
        self._feat_std: Optional[np.ndarray] = None

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
    def _pca_fit_transform(X: np.ndarray, target_dim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            reduced = np.hstack(
                [reduced, np.zeros((n, target_dim - reduced.shape[1]), dtype=reduced.dtype)]
            )
        return reduced, mean, comps

    @staticmethod
    def _pca_transform(
        X: np.ndarray,
        mean: np.ndarray,
        components: np.ndarray,
        target_dim: int,
    ) -> np.ndarray:
        Xc = X - mean
        try:
            reduced = Xc @ components.T
        except Exception:  # noqa: BLE001
            reduced = Xc[:, :target_dim]
        if reduced.shape[1] < target_dim:
            reduced = np.hstack(
                [
                    reduced,
                    np.zeros((reduced.shape[0], target_dim - reduced.shape[1]), dtype=reduced.dtype),
                ]
            )
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
        motion = _compute_motion_tab_v4(samples)
        static = _compute_static_tab_v4(samples)
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
                            "tab_v4 freeze mode requires fitted PCA state; run finalize_batch(..., fit=True) first."
                        )
                    reduced = self._pca_transform(X, self._pca_mean, self._pca_components, self._texture_dim)

            index_map = {src_idx: red_idx for red_idx, src_idx in enumerate(indices)}
            for i, feat in enumerate(features_list):
                out = OrderedDict()
                for k in self._motion_keys:
                    out[k] = float(feat.get(k, 0.0))
                for k in self._static_keys:
                    out[k] = float(feat.get(k, 0.0))
                if reduced is not None and i in index_map:
                    tex_vec = reduced[index_map[i]]
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
            self._feat_mean[self._non_deep_dim:] = 0.0
            self._feat_std[self._non_deep_dim:] = 1.0
        else:
            if self._feat_mean is None or self._feat_std is None:
                raise RuntimeError(
                    "tab_v4 global z-score is not fitted. finalize_batch(fit=True) must run first."
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
        return OrderedDict((k, float(v)) for k, v in zip(self._video_keys, mean_vec))

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
            raise ValueError("tab_v4 state must be a dict")
        self._pca_mean = state.get("pca_mean")
        self._pca_components = state.get("pca_components")
        self._feat_mean = state.get("feat_mean")
        self._feat_std = state.get("feat_std")
