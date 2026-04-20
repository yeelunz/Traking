from __future__ import annotations

import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:  # noqa: BLE001
    torch = None  # type: ignore
    nn = None  # type: ignore

from ...core.interfaces import FramePrediction
from ...core.registry import register_feature_extractor
from ..interfaces import TrajectoryFeatureExtractor
from ..texture_backbone import TextureBackboneWrapper
from ..trajectory_filter import pchip_interpolate_1d
from ..trajectory_filter import smooth_trajectory_2d as _smooth_trajectory_2d
from .v3pro import (
    _aggregate_roi_features,
    _build_runtime_preprocs_from_cfg,
    _extract_roi_rgb_batch,
    _normalize_texture_pooling,
    _normalize_texture_weight_source,
)
from .v4 import (
    TAB_V4_STATIC_KEYS,
    _compute_static_tab_v4,
    _normalize_texture_mode_v4,
)


TAB_V5_MOTION_KEYS: List[str] = [f"moment_motion_{i:02d}" for i in range(12)]
TAB_V5_STATIC_KEYS: List[str] = list(TAB_V4_STATIC_KEYS)
TAB_V5_TOTAL_DIM: int = 33

TAB_V5_LITE_MOTION_KEYS: List[str] = [f"moment_motion_{i:02d}" for i in range(15)]
TAB_V5_LITE_TOTAL_DIM: int = 30
TAB_V5_LITE_TEXTURE_DIM: int = TAB_V5_LITE_TOTAL_DIM - len(TAB_V5_LITE_MOTION_KEYS)

_MOMENT_PIPELINE_CACHE: Dict[Tuple[str, str], Any] = {}
_MOTION_MODES = {"freeze", "learnable", "pretrain"}


def _resample_centers_v5(
    samples: Sequence[FramePrediction],
    target_steps: int,
) -> np.ndarray:
    if target_steps <= 0:
        raise ValueError("target_steps must be positive")
    if not samples:
        return np.zeros((target_steps, 2), dtype=np.float64)

    ordered = sorted(samples, key=lambda s: float(s.frame_index))
    frames = np.asarray([float(s.frame_index) for s in ordered], dtype=np.float64)
    raw_centers = np.asarray([s.center for s in ordered], dtype=np.float64)
    if raw_centers.shape[0] <= 4:
        centers = raw_centers
    else:
        try:
            centers = _smooth_trajectory_2d(raw_centers, frames, skip_hampel=True)
        except Exception:
            centers = raw_centers

    if centers.shape[0] == 1:
        return np.repeat(centers, int(target_steps), axis=0)

    if np.allclose(frames, frames[0]):
        frames = np.arange(centers.shape[0], dtype=np.float64)

    t_query = np.linspace(float(frames[0]), float(frames[-1]), int(target_steps), dtype=np.float64)
    x_res = pchip_interpolate_1d(frames, centers[:, 0], t_query)
    y_res = pchip_interpolate_1d(frames, centers[:, 1], t_query)
    return np.column_stack([x_res, y_res]).astype(np.float64)


def _positions_to_step_displacements_v5(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.float64)
    return np.diff(arr, prepend=arr[0]).astype(np.float64)


def _trajectory_to_moment_input_v5(
    samples: Sequence[FramePrediction],
    target_steps: int,
    displacement_scale: float = 1.0,
) -> np.ndarray:
    centers = _resample_centers_v5(samples, target_steps=target_steps)
    dx = _positions_to_step_displacements_v5(centers[:, 0])
    dy = _positions_to_step_displacements_v5(centers[:, 1])
    try:
        scale = float(displacement_scale)
    except Exception:  # noqa: BLE001
        scale = 1.0
    if np.isfinite(scale) and not np.isclose(scale, 1.0):
        dx = dx * scale
        dy = dy * scale
    return np.stack([dx, dy], axis=0).astype(np.float32)


def _resolve_moment_device_v5(device: str) -> str:
    dev = str(device or "auto").strip().lower()
    if dev == "auto":
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return dev


def _normalize_motion_mode_v5(mode: str) -> str:
    mode_lc = str(mode or "freeze").strip().lower()
    aliases = {
        "freez": "freeze",
        "frozen": "freeze",
        "pretrained": "pretrain",
    }
    mode_lc = aliases.get(mode_lc, mode_lc)
    if mode_lc not in _MOTION_MODES:
        return "freeze"
    return mode_lc


def _extract_projection_state_dict_v5(payload: Any) -> Optional[Dict[str, Any]]:
    if isinstance(payload, dict):
        for key in (
            "projection_state_dict",
            "projection",
            "proj_state_dict",
            "projector_state_dict",
            "proj",
        ):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
    return None


def _strip_projection_prefixes_v5(state: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in state.items():
        clean = str(key)
        for prefix in ("module.", "projection.", "proj.", "projector.", "model."):
            if clean.startswith(prefix):
                clean = clean[len(prefix) :]
        out[clean] = value
    return out


_ModuleBase = nn.Module if nn is not None else object


class MomentProjectionWrapper(_ModuleBase):
    def __init__(
        self,
        *,
        input_dim: int,
        motion_dim: int,
        mode: str,
        pretrain_ckpt: Optional[str] = None,
        seed: int = 42,
    ):
        if torch is None or nn is None:
            raise RuntimeError("PyTorch is required for motion projection wrapper.")
        super().__init__()
        self.input_dim = int(input_dim)
        self.motion_dim = int(motion_dim)
        self.motion_mode = _normalize_motion_mode_v5(mode)

        with torch.random.fork_rng():
            torch.manual_seed(int(seed))
            self.proj = nn.Sequential(
                nn.Linear(self.input_dim, self.motion_dim),
                nn.LayerNorm(self.motion_dim),
                nn.GELU(),
            )

        if self.motion_mode == "pretrain":
            if not pretrain_ckpt:
                raise ValueError("motion_mode='pretrain' requires motion_pretrain_ckpt.")
            self._load_pretrain_ckpt(pretrain_ckpt)

        self.eval()
        self.requires_grad_(False)

    def _load_pretrain_ckpt(self, ckpt_path: str) -> None:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"motion_pretrain_ckpt not found: {ckpt_path}")
        payload = torch.load(ckpt_path, map_location="cpu")

        ckpt_input_dim = None
        ckpt_motion_dim = None
        if isinstance(payload, dict):
            ckpt_input_dim = payload.get("motion_input_dim")
            ckpt_motion_dim = payload.get("motion_embedding_dim")
        if ckpt_input_dim is not None and int(ckpt_input_dim) != self.input_dim:
            raise ValueError(
                f"Motion checkpoint input_dim mismatch: ckpt={ckpt_input_dim}, requested={self.input_dim}"
            )
        if ckpt_motion_dim is not None and int(ckpt_motion_dim) != self.motion_dim:
            raise ValueError(
                f"Motion checkpoint embedding_dim mismatch: ckpt={ckpt_motion_dim}, requested={self.motion_dim}"
            )

        proj_state = _extract_projection_state_dict_v5(payload)
        if not isinstance(proj_state, dict) or not proj_state:
            raise ValueError("motion pretrain ckpt has no projection_state_dict.")
        proj_clean = _strip_projection_prefixes_v5(proj_state)
        incompatible = self.proj.load_state_dict(proj_clean, strict=False)
        missing = list(getattr(incompatible, "missing_keys", []) or [])
        unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
        if missing or unexpected:
            raise ValueError(
                "Motion projection checkpoint loaded with key mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )

    def forward(self, x: Any) -> Any:
        return self.proj(x)


def _load_moment_pipeline_v5(model_name: str, device: str):  # noqa: ANN001
    cache_key = (str(model_name), str(device))
    cached = _MOMENT_PIPELINE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if torch is None:
        raise RuntimeError("PyTorch is required for tab_v5 MOMENT motion features.")

    try:
        from momentfm import MOMENTPipeline
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "momentfm is required for tab_v5. Install/import the package successfully before using this extractor."
        ) from exc

    model = MOMENTPipeline.from_pretrained(
        str(model_name),
        model_kwargs={"task_name": "embedding"},
    )
    model.init()
    model = model.to(str(device))
    model.eval()
    _MOMENT_PIPELINE_CACHE[cache_key] = model
    return model


@register_feature_extractor("tab_v5")
class MotionStaticV5FeatureExtractor(TrajectoryFeatureExtractor):
    name = "MotionStaticV5Features"
    DEFAULT_CONFIG: Dict[str, Any] = {
        "n_texture_frames": 5,
        "texture_image_size": 96,
        "roi_pad_ratio": 0.2,
        "texture_pooling": "score_weighted",
        "texture_pooling_weight_source": "detection_score",
        "texture_mode": "freeze",
        "texture_backbone": "convnext_tiny",
        "texture_pretrain_ckpt": None,
        "pretrained_backbone": True,
        "texture_device": "auto",
        "moment_model_name": "AutonLab/MOMENT-1-base",
        "moment_input_steps": 256,
        "moment_pca_dim": len(TAB_V5_MOTION_KEYS),
        "motion_mode": "freeze",
        "motion_pretrain_ckpt": None,
        "motion_projection_seed": 42,
        "moment_device": "auto",
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        cfg = {**self.DEFAULT_CONFIG, **(params or {})}
        self._n_texture_frames = int(cfg.get("n_texture_frames", 5))
        self._texture_image_size = int(cfg.get("texture_image_size", 96))
        self._roi_pad_ratio = float(cfg.get("roi_pad_ratio", 0.2))
        self._texture_pooling = _normalize_texture_pooling(str(cfg.get("texture_pooling", "score_weighted")))
        self._texture_pooling_weight_source = _normalize_texture_weight_source(
            str(cfg.get("texture_pooling_weight_source", "detection_score"))
        )
        self._texture_mode = _normalize_texture_mode_v4(str(cfg.get("texture_mode", "freeze")))
        self._texture_backbone = str(cfg.get("texture_backbone", "convnext_tiny"))
        self._texture_pretrain_ckpt = cfg.get("texture_pretrain_ckpt")
        self._pretrained_backbone = bool(cfg.get("pretrained_backbone", True))
        self._texture_device = str(cfg.get("texture_device", "auto"))
        _runtime_pp = dict(cfg.get("_runtime_texture_preprocessing") or {})
        self._runtime_global_preprocs, self._runtime_roi_preprocs = _build_runtime_preprocs_from_cfg(
            _runtime_pp
        )
        self._texture_wrapper: Optional[TextureBackboneWrapper] = None

        self._moment_model_name = str(cfg.get("moment_model_name", "AutonLab/MOMENT-1-base"))
        self._moment_input_steps = int(cfg.get("moment_input_steps", 256))
        self._moment_pca_dim = int(cfg.get("moment_pca_dim", len(TAB_V5_MOTION_KEYS)))
        self._motion_mode = _normalize_motion_mode_v5(str(cfg.get("motion_mode", "freeze")))
        self._motion_pretrain_ckpt = cfg.get("motion_pretrain_ckpt")
        self._motion_projection_seed = int(cfg.get("motion_projection_seed", 42))
        self._moment_device = _resolve_moment_device_v5(str(cfg.get("moment_device", "auto")))
        self._moment_pipeline = None
        self._motion_wrapper: Optional[MomentProjectionWrapper] = None

        self._motion_keys = list(TAB_V5_MOTION_KEYS)
        self._static_keys = list(TAB_V5_STATIC_KEYS)
        self._non_deep_dim = len(self._motion_keys) + len(self._static_keys)
        self._texture_dim = TAB_V5_TOTAL_DIM - self._non_deep_dim
        self._texture_keys = [f"cnn_features_{i:02d}" for i in range(self._texture_dim)]
        self._video_keys = self._motion_keys + self._static_keys + self._texture_keys
        self._subject_keys = list(self._video_keys)

        self._motion_pca_mean: Optional[np.ndarray] = None
        self._motion_pca_components: Optional[np.ndarray] = None
        self._moment_raw_dim: Optional[int] = None
        self._motion_projection_state_dict: Optional[Dict[str, Any]] = None
        self._motion_projection_input_dim: Optional[int] = None
        self._tex_pca_mean: Optional[np.ndarray] = None
        self._tex_pca_components: Optional[np.ndarray] = None
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
            dev = _resolve_moment_device_v5(self._texture_device)
            wrapper = wrapper.to(dev)
            wrapper.eval()
        self._texture_wrapper = wrapper
        return wrapper

    def _ensure_moment_pipeline(self):  # noqa: ANN001
        if self._moment_pipeline is not None:
            return self._moment_pipeline
        self._moment_pipeline = _load_moment_pipeline_v5(self._moment_model_name, self._moment_device)
        return self._moment_pipeline

    def _ensure_motion_wrapper(self, input_dim: int) -> MomentProjectionWrapper:
        if torch is None or nn is None:
            raise RuntimeError("PyTorch is required for non-freeze motion_mode.")
        if self._motion_mode == "freeze":
            raise RuntimeError("Motion projection wrapper should not be used in freeze mode.")
        if self._motion_wrapper is not None:
            if int(input_dim) != int(self._motion_projection_input_dim or input_dim):
                raise RuntimeError(
                    "Existing motion projection wrapper input_dim does not match new MOMENT embedding size."
                )
            return self._motion_wrapper

        pretrain_ckpt = None
        if self._motion_mode == "pretrain" and self._motion_projection_state_dict is None:
            pretrain_ckpt = self._motion_pretrain_ckpt
        wrapper = MomentProjectionWrapper(
            input_dim=int(input_dim),
            motion_dim=int(self._moment_pca_dim),
            mode=self._motion_mode,
            pretrain_ckpt=pretrain_ckpt,
            seed=self._motion_projection_seed,
        )
        if self._motion_projection_state_dict is not None:
            incompatible = wrapper.proj.load_state_dict(self._motion_projection_state_dict, strict=False)
            missing = list(getattr(incompatible, "missing_keys", []) or [])
            unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
            if missing or unexpected:
                raise RuntimeError(
                    "Saved motion projection state could not be restored cleanly: "
                    f"missing={missing}, unexpected={unexpected}"
                )
        wrapper = wrapper.to(self._moment_device)
        wrapper.eval()
        self._motion_wrapper = wrapper
        self._motion_projection_input_dim = int(input_dim)
        self._motion_projection_state_dict = {
            k: v.detach().cpu() for k, v in wrapper.proj.state_dict().items()
        }
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

    def _extract_moment_motion_vector(
        self,
        samples: Sequence[FramePrediction],
        *,
        displacement_scale: float = 1.0,
    ) -> Dict[str, Any]:
        zeros = np.zeros((self._moment_pca_dim,), dtype=np.float32)
        if not samples:
            return {"motion": zeros}
        if torch is None:
            return {"motion": zeros}

        model = self._ensure_moment_pipeline()
        try:
            dev = next(model.parameters()).device
        except Exception:  # noqa: BLE001
            dev = torch.device("cpu")

        motion_input = _trajectory_to_moment_input_v5(
            samples,
            target_steps=self._moment_input_steps,
            displacement_scale=displacement_scale,
        )
        x_enc = torch.tensor(motion_input[None, ...], dtype=torch.float32, device=dev)
        input_mask = torch.ones((1, self._moment_input_steps), dtype=torch.float32, device=dev)

        with torch.no_grad():
            output = model(x_enc=x_enc, input_mask=input_mask, reduction="none")

        emb = output.embeddings
        if hasattr(emb, "detach"):
            emb_np = emb.detach().cpu().numpy().astype(np.float64)
        else:
            emb_np = np.asarray(emb, dtype=np.float64)

        if emb_np.ndim == 4:
            pooled = emb_np.mean(axis=(1, 2))[0]
        elif emb_np.ndim == 3:
            pooled = emb_np.mean(axis=1)[0]
        elif emb_np.ndim == 2:
            pooled = emb_np[0]
        else:
            pooled = emb_np.reshape(-1)

        pooled = np.asarray(pooled, dtype=np.float64).reshape(-1)
        self._moment_raw_dim = int(pooled.size)
        if self._motion_mode != "freeze":
            wrapper = self._ensure_motion_wrapper(pooled.size)
            try:
                dev = next(wrapper.parameters()).device
            except Exception:  # noqa: BLE001
                dev = torch.device("cpu")
            x_proj = torch.tensor(pooled[None, ...], dtype=torch.float32, device=dev)
            with torch.no_grad():
                proj = wrapper(x_proj).detach().cpu().numpy().astype(np.float64).reshape(-1)
            motion = np.zeros((self._moment_pca_dim,), dtype=np.float32)
            motion[: min(self._moment_pca_dim, proj.shape[0])] = proj[: min(self._moment_pca_dim, proj.shape[0])]
            self._motion_projection_state_dict = {
                k: v.detach().cpu() for k, v in wrapper.proj.state_dict().items()
            }
            return {
                "motion": motion,
                "_raw_moment_embedding": pooled,
            }
        return {
            "motion": zeros,
            "_raw_moment_embedding": pooled,
        }

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
        motion_info = self._extract_moment_motion_vector(samples)
        static = _compute_static_tab_v4(samples)
        tex_info = self._extract_texture_vector(samples, video_path)
        motion_vec = np.asarray(motion_info.get("motion"), dtype=np.float32)
        tex_vec = np.asarray(tex_info.get("tex"), dtype=np.float32)

        out: Dict[str, float] = OrderedDict()
        for i, k in enumerate(self._motion_keys):
            out[k] = float(motion_vec[i]) if i < motion_vec.size else 0.0
        for k in self._static_keys:
            out[k] = float(static.get(k, 0.0))
        for i, k in enumerate(self._texture_keys):
            out[k] = float(tex_vec[i]) if i < tex_vec.size else 0.0

        raw_motion = motion_info.get("_raw_moment_embedding")
        if raw_motion is not None:
            out["_raw_moment_embedding"] = np.asarray(raw_motion, dtype=np.float64)

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

        motion_rows: List[np.ndarray] = []
        if self._motion_mode == "freeze":
            for feat in features_list:
                raw_motion = feat.get("_raw_moment_embedding")
                if raw_motion is None:
                    continue
                arr = np.asarray(raw_motion, dtype=np.float64).reshape(1, -1)
                if arr.shape[1] > 0:
                    motion_rows.append(arr)

            if motion_rows:
                X_motion = np.vstack(motion_rows)
                if fit:
                    _, self._motion_pca_mean, self._motion_pca_components = self._pca_fit_transform(
                        X_motion,
                        self._moment_pca_dim,
                    )
                    self._moment_raw_dim = int(X_motion.shape[1])
                elif self._motion_pca_mean is None or self._motion_pca_components is None:
                    raise RuntimeError(
                        "tab_v5 MOMENT PCA state is not fitted. finalize_batch(..., fit=True) must run first."
                    )

        result: List[Dict[str, float]] = []
        if self._texture_mode != "freeze":
            texture_result_ready = True
        else:
            texture_result_ready = False
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
            if fit_rows:
                X_tex = np.vstack(fit_rows)
                if fit:
                    _, self._tex_pca_mean, self._tex_pca_components = self._pca_fit_transform(
                        X_tex,
                        self._texture_dim,
                    )
                elif self._tex_pca_mean is None or self._tex_pca_components is None:
                    raise RuntimeError(
                        "tab_v5 freeze mode requires fitted texture PCA state; run finalize_batch(..., fit=True) first."
                    )

        for feat in features_list:
            out = OrderedDict()

            if self._motion_mode == "freeze":
                motion_vec = np.zeros((self._moment_pca_dim,), dtype=np.float64)
                raw_motion = feat.get("_raw_moment_embedding")
                if raw_motion is not None:
                    arr = np.asarray(raw_motion, dtype=np.float64).reshape(1, -1)
                    if arr.shape[1] > 0:
                        if self._motion_pca_mean is None or self._motion_pca_components is None:
                            if fit and motion_rows:
                                raise RuntimeError("tab_v5 could not build MOMENT PCA state.")
                        else:
                            reduced = self._pca_transform(
                                arr,
                                self._motion_pca_mean,
                                self._motion_pca_components,
                                self._moment_pca_dim,
                            )
                            motion_vec[: min(self._moment_pca_dim, reduced.shape[1])] = reduced[0, : self._moment_pca_dim]
            else:
                motion_vec = np.array([feat.get(k, 0.0) for k in self._motion_keys], dtype=np.float64)

            for i, k in enumerate(self._motion_keys):
                out[k] = float(motion_vec[i]) if i < motion_vec.shape[0] else 0.0

            for k in self._static_keys:
                out[k] = float(feat.get(k, 0.0))

            if self._texture_mode != "freeze":
                tex_vec = np.array([feat.get(k, 0.0) for k in self._texture_keys], dtype=np.float64)
            else:
                tex_vec = np.zeros((self._texture_dim,), dtype=np.float64)
                raw_batch = feat.get("_raw_texture_backbone_batch")
                weights = feat.get("_raw_texture_weights")
                if raw_batch is not None:
                    arr2d = np.asarray(raw_batch, dtype=np.float64)
                    if arr2d.ndim == 1:
                        arr2d = arr2d.reshape(1, -1)
                    elif arr2d.ndim > 2:
                        arr2d = arr2d.reshape(arr2d.shape[0], -1)
                    if arr2d.shape[0] > 0 and arr2d.shape[1] > 0:
                        if self._tex_pca_mean is None or self._tex_pca_components is None:
                            if fit and not texture_result_ready:
                                raise RuntimeError("tab_v5 freeze mode could not build texture PCA state.")
                        else:
                            roi_reduced = self._pca_transform(
                                arr2d,
                                self._tex_pca_mean,
                                self._tex_pca_components,
                                self._texture_dim,
                            )
                            w = np.asarray(
                                weights
                                if weights is not None
                                else np.ones((roi_reduced.shape[0],), dtype=np.float64),
                                dtype=np.float64,
                            )
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
            motion_dim = len(self._motion_keys)
            static_dim = len(self._static_keys)
            texture_start = motion_dim + static_dim
            if self._motion_mode != "freeze":
                self._feat_mean[:motion_dim] = 0.0
                self._feat_std[:motion_dim] = 1.0
            self._feat_mean[texture_start:] = 0.0
            self._feat_std[texture_start:] = 1.0
        else:
            if self._feat_mean is None or self._feat_std is None:
                raise RuntimeError(
                    "tab_v5 global z-score is not fitted. finalize_batch(fit=True) must run first."
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
            "texture_pooling": self._texture_pooling,
            "texture_pooling_weight_source": self._texture_pooling_weight_source,
            "moment_model_name": self._moment_model_name,
            "moment_input_steps": self._moment_input_steps,
            "motion_mode": self._motion_mode,
            "moment_pca_dim": self._moment_pca_dim,
            "moment_pca_mean": self._motion_pca_mean,
            "moment_pca_components": self._motion_pca_components,
            "moment_raw_dim": self._moment_raw_dim,
            "motion_projection_input_dim": self._motion_projection_input_dim,
            "motion_projection_state_dict": self._motion_projection_state_dict,
            "tex_pca_mean": self._tex_pca_mean,
            "tex_pca_components": self._tex_pca_components,
            "feat_mean": self._feat_mean,
            "feat_std": self._feat_std,
            "texture_dim": self._texture_dim,
            "video_keys": list(self._video_keys),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise ValueError("tab_v5 state must be a dict")
        if "texture_pooling" in state:
            self._texture_pooling = _normalize_texture_pooling(str(state.get("texture_pooling")))
        if "texture_pooling_weight_source" in state:
            self._texture_pooling_weight_source = _normalize_texture_weight_source(
                str(state.get("texture_pooling_weight_source"))
            )
        if "moment_model_name" in state:
            self._moment_model_name = str(state.get("moment_model_name"))
        if "moment_input_steps" in state:
            self._moment_input_steps = int(state.get("moment_input_steps"))
        if "motion_mode" in state:
            self._motion_mode = _normalize_motion_mode_v5(str(state.get("motion_mode")))
        self._motion_pca_mean = state.get("moment_pca_mean")
        self._motion_pca_components = state.get("moment_pca_components")
        self._moment_raw_dim = state.get("moment_raw_dim")
        self._motion_projection_input_dim = state.get("motion_projection_input_dim")
        self._motion_projection_state_dict = state.get("motion_projection_state_dict")
        self._motion_wrapper = None
        self._tex_pca_mean = state.get("tex_pca_mean")
        self._tex_pca_components = state.get("tex_pca_components")
        self._feat_mean = state.get("feat_mean")
        self._feat_std = state.get("feat_std")


@register_feature_extractor("tab_v5_lite")
class MotionStaticV5LiteFeatureExtractor(MotionStaticV5FeatureExtractor):
    name = "MotionStaticV5LiteFeatures"
    DEFAULT_CONFIG: Dict[str, Any] = {
        **MotionStaticV5FeatureExtractor.DEFAULT_CONFIG,
        "moment_pca_dim": len(TAB_V5_LITE_MOTION_KEYS),
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params=params)
        if self._moment_pca_dim != len(TAB_V5_LITE_MOTION_KEYS):
            raise ValueError(
                "tab_v5_lite requires moment_pca_dim=15 to keep motion/texture at 15D each."
            )

        self._motion_keys = list(TAB_V5_LITE_MOTION_KEYS)
        self._static_keys = []
        self._non_deep_dim = len(self._motion_keys)
        self._texture_dim = int(TAB_V5_LITE_TEXTURE_DIM)
        self._texture_keys = [f"cnn_features_{i:02d}" for i in range(self._texture_dim)]
        self._video_keys = self._motion_keys + self._texture_keys
        self._subject_keys = list(self._video_keys)

    def extract_video(
        self,
        samples: Sequence[FramePrediction],
        video_path: Optional[str] = None,
    ) -> Dict[str, float]:
        motion_info = self._extract_moment_motion_vector(samples)
        tex_info = self._extract_texture_vector(samples, video_path)
        motion_vec = np.asarray(motion_info.get("motion"), dtype=np.float32)
        tex_vec = np.asarray(tex_info.get("tex"), dtype=np.float32)

        out: Dict[str, float] = OrderedDict()
        for i, k in enumerate(self._motion_keys):
            out[k] = float(motion_vec[i]) if i < motion_vec.size else 0.0
        for i, k in enumerate(self._texture_keys):
            out[k] = float(tex_vec[i]) if i < tex_vec.size else 0.0

        raw_motion = motion_info.get("_raw_moment_embedding")
        if raw_motion is not None:
            out["_raw_moment_embedding"] = np.asarray(raw_motion, dtype=np.float64)

        raw_batch = tex_info.get("_raw_texture_backbone_batch")
        if raw_batch is not None:
            out["_raw_texture_backbone_batch"] = np.asarray(raw_batch, dtype=np.float64)
            out["_raw_texture_weights"] = np.asarray(
                tex_info.get("_raw_texture_weights", np.zeros((0,), dtype=np.float64)),
                dtype=np.float64,
            )
        return out
