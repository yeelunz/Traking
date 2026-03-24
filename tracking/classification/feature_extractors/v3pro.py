from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import logging

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None  # type: ignore

from ...core.interfaces import FramePrediction
from ...core.registry import PREPROC_REGISTRY, register_feature_extractor
from ..interfaces import TrajectoryFeatureExtractor
from .v3lite import (
    LITE_MOTION_KEYS,
    LITE_STATIC_KEYS,
    _compute_motion_lite,
    _compute_static_lite,
)
from ..texture_backbone import TextureBackboneWrapper


logger = logging.getLogger(__name__)


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


def _resolve_texture_roi_bbox(
    sample: FramePrediction,
    pad_ratio: float,
) -> Tuple[Tuple[float, float, float, float], float]:
    seg = getattr(sample, "segmentation", None)
    seg_roi_bbox = getattr(seg, "roi_bbox", None) if seg is not None else None
    if seg_roi_bbox is not None and len(seg_roi_bbox) == 4:
        return (
            float(seg_roi_bbox[0]),
            float(seg_roi_bbox[1]),
            float(seg_roi_bbox[2]),
            float(seg_roi_bbox[3]),
        ), 0.0
    bbox = getattr(sample, "bbox", None) or (0.0, 0.0, 0.0, 0.0)
    return (
        float(bbox[0]),
        float(bbox[1]),
        float(bbox[2]),
        float(bbox[3]),
    ), float(pad_ratio)


def _normalize_texture_pooling(mode: str) -> str:
    mode_lc = str(mode or "score_weighted").strip().lower()
    aliases = {
        "avg": "mean",
        "average": "mean",
        "weighted": "score_weighted",
        "score-weighted": "score_weighted",
    }
    mode_lc = aliases.get(mode_lc, mode_lc)
    if mode_lc not in {"mean", "score_weighted"}:
        return "score_weighted"
    return mode_lc


def _normalize_texture_weight_source(source: str) -> str:
    source_lc = str(source or "detection_score").strip().lower()
    aliases = {
        "score": "detection_score",
        "det_score": "detection_score",
        "detector_score": "detection_score",
        "conf": "confidence",
        "iou": "iou_pred",
        "iou_predicted": "iou_pred",
    }
    source_lc = aliases.get(source_lc, source_lc)
    if source_lc not in {"detection_score", "confidence", "iou_pred"}:
        return "detection_score"
    return source_lc


def _sample_weight_value(sample: FramePrediction, weight_source: str) -> float:
    source = _normalize_texture_weight_source(weight_source)
    value: Optional[float] = None
    if source == "iou_pred":
        comps = getattr(sample, "confidence_components", None) or {}
        for key in ("iou_pred", "iou", "pred_iou"):
            raw = comps.get(key)
            if raw is not None:
                value = float(raw)
                break
        if value is None:
            seg = getattr(sample, "segmentation", None)
            if seg is not None:
                seg_score = getattr(seg, "score", None)
                if seg_score is not None:
                    value = float(seg_score)
    elif source == "confidence":
        conf = getattr(sample, "confidence", None)
        if conf is not None:
            value = float(conf)
    else:
        score = getattr(sample, "score", None)
        if score is not None:
            value = float(score)

    if value is None:
        score_fb = getattr(sample, "score", None)
        conf_fb = getattr(sample, "confidence", None)
        if score_fb is not None:
            value = float(score_fb)
        elif conf_fb is not None:
            value = float(conf_fb)
        else:
            value = 0.0
    return max(0.0, value)


def _apply_preprocs_frame_like_segmentation(frame: np.ndarray, preprocs: Sequence[Any]) -> np.ndarray:
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if not preprocs:
        return frame
    if frame.ndim == 2:
        out = frame
        for p in preprocs:
            out = p.apply_to_frame(out)
        return out
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    for p in preprocs:
        rgb = p.apply_to_frame(rgb)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _build_runtime_preprocs_from_cfg(runtime_cfg: Dict[str, Any]) -> Tuple[List[Any], List[Any]]:
    runtime = dict(runtime_cfg or {})
    scheme_raw = runtime.get("scheme", "A")
    scheme = str(scheme_raw).strip().upper()
    if scheme in {"GLOBAL", "A"}:
        scheme = "A"
    elif scheme in {"ROI", "B"}:
        scheme = "B"
    elif scheme in {"HYBRID", "C"}:
        scheme = "C"
    else:
        scheme = "A"

    step_defs = runtime.get("preproc_steps") or []
    if isinstance(step_defs, (str, bytes)) or not isinstance(step_defs, Sequence):
        return [], []

    global_preprocs: List[Any] = []
    roi_preprocs: List[Any] = []
    for step in step_defs:
        if not isinstance(step, dict):
            continue
        name = step.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        cls = PREPROC_REGISTRY.get(name)
        if cls is None:
            continue
        params = step.get("params", {})
        if scheme == "A":
            global_preprocs.append(cls(params))
        elif scheme == "B":
            roi_preprocs.append(cls(params))
        else:
            # Match classification pretrain ROI generation behavior.
            roi_preprocs.append(cls(params))
    return global_preprocs, roi_preprocs


def _extract_roi_rgb_batch(
    samples: Sequence[FramePrediction],
    video_path: Optional[str],
    image_size: int,
    n_frames: int,
    pad_ratio: float,
    weight_source: str,
    global_preprocs: Optional[Sequence[Any]] = None,
    roi_preprocs: Optional[Sequence[Any]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    image_size = max(32, int(image_size))
    empty_x = np.zeros((0, 3, image_size, image_size), dtype=np.float32)
    empty_w = np.zeros((0,), dtype=np.float64)
    if not samples or not video_path:
        return empty_x, empty_w

    n = len(samples)
    k = max(1, int(n_frames))
    if n <= k:
        pick = list(range(n))
    else:
        scored = sorted(
            range(n),
            key=lambda i: _sample_weight_value(samples[i], weight_source),
            reverse=True,
        )
        pick = sorted(scored[:k])
    if not pick:
        return empty_x, empty_w

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return empty_x, empty_w

    patches: List[np.ndarray] = []
    weights: List[float] = []
    try:
        for i in pick:
            s = samples[i]
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(s.frame_index))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frame_for_crop = frame
            if global_preprocs:
                frame_for_crop = _apply_preprocs_frame_like_segmentation(frame_for_crop, global_preprocs)

            crop_bbox, crop_pad_ratio = _resolve_texture_roi_bbox(s, pad_ratio)
            roi = _crop_roi_with_padding(frame_for_crop, crop_bbox, crop_pad_ratio)
            if roi is None:
                continue

            if roi_preprocs:
                roi = _apply_preprocs_frame_like_segmentation(roi, roi_preprocs)

            resized = cv2.resize(roi, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32)
            patches.append(chw)
            weights.append(_sample_weight_value(s, weight_source))
    finally:
        cap.release()

    if not patches:
        return empty_x, empty_w
    return np.stack(patches, axis=0), np.asarray(weights, dtype=np.float64)


def _aggregate_roi_features(
    roi_features: np.ndarray,
    roi_weights: np.ndarray,
    pooling_mode: str,
) -> np.ndarray:
    if roi_features.ndim != 2 or roi_features.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    if pooling_mode == "score_weighted":
        w = np.asarray(roi_weights, dtype=np.float64).reshape(-1)
        if w.shape[0] == roi_features.shape[0]:
            denom = float(w.sum())
            if denom > 1e-9:
                w = w / denom
                return (roi_features * w[:, None]).sum(axis=0)
    return roi_features.mean(axis=0)


@register_feature_extractor("tab_v3_pro")
class MotionStaticV3ProFeatureExtractor(TrajectoryFeatureExtractor):
    name = "MotionStaticV3ProFeatures"
    DEFAULT_CONFIG: Dict[str, Any] = {
        "n_texture_frames": 5,
        "texture_image_size": 96,
        "roi_pad_ratio": 0.15,
        "texture_pooling": "score_weighted",  # mean | score_weighted
        "texture_pooling_weight_source": "detection_score",  # detection_score | confidence | iou_pred
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
        self._texture_pooling = _normalize_texture_pooling(str(cfg.get("texture_pooling", "score_weighted")))
        self._texture_pooling_weight_source = _normalize_texture_weight_source(
            str(cfg.get("texture_pooling_weight_source", "detection_score"))
        )
        self._texture_mode = str(cfg.get("texture_mode", "freeze")).lower()
        self._texture_backbone = str(cfg.get("texture_backbone", "convnext_tiny"))
        self._texture_dim = int(cfg.get("texture_dim", 10))
        self._texture_pretrain_ckpt = cfg.get("texture_pretrain_ckpt")
        self._pretrained_backbone = bool(cfg.get("pretrained_backbone", True))
        self._texture_device = str(cfg.get("texture_device", "auto"))
        _runtime_pp = dict(cfg.get("_runtime_texture_preprocessing") or {})
        self._runtime_global_preprocs, self._runtime_roi_preprocs = _build_runtime_preprocs_from_cfg(_runtime_pp)
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
        roi_batch, roi_weights = _extract_roi_rgb_batch(
            samples=samples,
            video_path=video_path,
            image_size=self._texture_image_size,
            n_frames=self._n_texture_frames,
            pad_ratio=self._roi_pad_ratio,
            weight_source=self._texture_pooling_weight_source,
            global_preprocs=self._runtime_global_preprocs,
            roi_preprocs=self._runtime_roi_preprocs,
        )
        if roi_batch.shape[0] == 0:
            return {"tex": zeros}
        if torch is None:
            return {"tex": zeros}
        wrapper = self._ensure_texture_wrapper()
        try:
            dev = next(wrapper.parameters()).device
        except Exception:  # noqa: BLE001
            dev = torch.device("cpu")
        xb = torch.tensor(roi_batch, dtype=torch.float32, device=dev)

        # Automatic routing:
        # - freeze: use raw backbone feature then PCA in finalize_batch
        # - learnable/pretrain: use wrapper projection output directly
        if self._texture_mode == "freeze":
            with torch.no_grad():
                x_norm = (xb - wrapper.img_mean) / (wrapper.img_std + 1e-8)
                raw = wrapper._forward_backbone(x_norm).detach().cpu().numpy().astype(np.float64)
            return {
                "tex": zeros,
                "_raw_texture_backbone_batch": raw,
                "_raw_texture_weights": roi_weights,
            }

        with torch.no_grad():
            feat = wrapper(xb).detach().cpu().numpy().astype(np.float64)
        feat = feat.reshape(feat.shape[0], -1)
        pooled = _aggregate_roi_features(feat, roi_weights, self._texture_pooling)
        out = np.zeros((self._texture_dim,), dtype=np.float32)
        out[: min(self._texture_dim, pooled.shape[0])] = pooled[: min(self._texture_dim, pooled.shape[0])]
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

        raw_batch = tex_info.get("_raw_texture_backbone_batch")
        if raw_batch is not None:
            out["_raw_texture_backbone_batch"] = np.asarray(raw_batch, dtype=np.float64)
            out["_raw_texture_weights"] = np.asarray(
                tex_info.get("_raw_texture_weights", np.zeros((0,), dtype=np.float64)),
                dtype=np.float64,
            )
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
            fit_rows: List[np.ndarray] = []
            for feat in features_list:
                raw_batch = feat.get("_raw_texture_backbone_batch")
                if raw_batch is not None:
                    arr2d = np.asarray(raw_batch, dtype=np.float64)
                    if arr2d.ndim == 1:
                        arr2d = arr2d.reshape(1, -1)
                    elif arr2d.ndim > 2:
                        arr2d = arr2d.reshape(arr2d.shape[0], -1)
                    if arr2d.shape[0] > 0 and arr2d.shape[1] > 0:
                        fit_rows.append(arr2d)
                    continue
                # Backward-compatibility for old cache shape.
                raw_legacy = feat.get("_raw_texture_backbone")
                if raw_legacy is not None:
                    arr1d = np.asarray(raw_legacy, dtype=np.float64).reshape(1, -1)
                    if arr1d.shape[1] > 0:
                        fit_rows.append(arr1d)

            if fit_rows:
                X = np.vstack(fit_rows)
                if fit:
                    _, self._pca_mean, self._pca_components = self._pca_fit_transform(X, self._texture_dim)
                else:
                    if self._pca_mean is None or self._pca_components is None:
                        raise RuntimeError(
                            "tab_v3_pro freeze mode requires fitted PCA state; run finalize_batch(..., fit=True) first."
                        )
            for i, feat in enumerate(features_list):
                out = OrderedDict()
                for k in self._motion_keys:
                    out[k] = float(feat.get(k, 0.0))
                for k in self._static_keys:
                    out[k] = float(feat.get(k, 0.0))

                raw_batch = feat.get("_raw_texture_backbone_batch")
                weights = feat.get("_raw_texture_weights")
                if raw_batch is None:
                    raw_legacy = feat.get("_raw_texture_backbone")
                    if raw_legacy is not None:
                        raw_batch = np.asarray(raw_legacy, dtype=np.float64).reshape(1, -1)
                        weights = np.ones((1,), dtype=np.float64)

                tex_vec = np.zeros((self._texture_dim,), dtype=np.float64)
                if raw_batch is not None:
                    arr2d = np.asarray(raw_batch, dtype=np.float64)
                    if arr2d.ndim == 1:
                        arr2d = arr2d.reshape(1, -1)
                    elif arr2d.ndim > 2:
                        arr2d = arr2d.reshape(arr2d.shape[0], -1)
                    if arr2d.shape[0] > 0 and arr2d.shape[1] > 0:
                        if self._pca_mean is None or self._pca_components is None:
                            if fit and fit_rows:
                                raise RuntimeError(
                                    "tab_v3_pro freeze mode could not build PCA state for per-ROI texture reduction."
                                )
                        else:
                            roi_reduced = self._pca_transform(
                                arr2d,
                                self._pca_mean,
                                self._pca_components,
                                self._texture_dim,
                            )
                            w = np.asarray(weights if weights is not None else np.ones((roi_reduced.shape[0],), dtype=np.float64), dtype=np.float64)
                            pooled = _aggregate_roi_features(roi_reduced, w, self._texture_pooling)
                            tex_vec[: min(self._texture_dim, pooled.shape[0])] = pooled[: min(self._texture_dim, pooled.shape[0])]

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
            "texture_pooling": self._texture_pooling,
            "texture_pooling_weight_source": self._texture_pooling_weight_source,
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
        if "texture_pooling" in state:
            self._texture_pooling = _normalize_texture_pooling(str(state.get("texture_pooling")))
        if "texture_pooling_weight_source" in state:
            self._texture_pooling_weight_source = _normalize_texture_weight_source(
                str(state.get("texture_pooling_weight_source"))
            )
        self._pca_mean = state.get("pca_mean")
        self._pca_components = state.get("pca_components")
        self._feat_mean = state.get("feat_mean")
        self._feat_std = state.get("feat_std")
