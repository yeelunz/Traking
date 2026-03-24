from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:  # pragma: no cover - optional CUDA specific exception
    from torch.cuda import CudaError as _TorchCudaError  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - CPU-only envs
    _TorchCudaError = RuntimeError
from torch.utils.data import DataLoader

try:  # pragma: no cover - optional dependency
    from segmentation_annotator import auto_mask as _auto_mask_module
except Exception:  # pragma: no cover
    _auto_mask_module = None

from ..core.interfaces import FramePrediction, MaskStats, SegmentationData, PreprocessingModule
from ..core.registry import SEGMENTATION_MODEL_REGISTRY
from ..utils.annotations import load_coco_vid
from .dataset import SegmentationCropDataset, attach_ground_truth_segmentation
from .metrics import centroid_distance, dice_coefficient, intersection_over_union, summarise_metrics
from . import model as _load_segmentation_models  # noqa: F401
from .utils import (
    BoundingBox,
    compute_mask_stats,
    crop_with_bbox,
    ensure_dir,
    expand_bbox,
    fill_holes,
    keep_largest_component,
    place_mask_on_canvas,
)

if _auto_mask_module is not None:  # pragma: no cover - optional path
    _AUTO_MASK_EXTRACT_ROI = getattr(_auto_mask_module, "_extract_roi_with_reflect", None)
    _AUTO_MASK_ENSURE_GRAY = getattr(_auto_mask_module, "_ensure_gray", None)
    _AUTO_MASK_RUN_GRABCUT = getattr(_auto_mask_module, "_run_grabcut_seed", None)
    _AUTO_MASK_FALLBACK = getattr(_auto_mask_module, "_fallback_center_ellipse", None)
    _AUTO_MASK_STRIP_PADDING = getattr(_auto_mask_module, "_strip_padding", None)
    _AUTO_MASK_REMOVE_BOUNDARY = getattr(_auto_mask_module, "_remove_boundary_components", None)
    _AUTO_MASK_LARGEST = getattr(_auto_mask_module, "_largest_component", None)
    _AUTO_MASK_GUIDED_FILTER = getattr(_auto_mask_module, "_guided_filter", None)
    _AUTO_MASK_MGAC = getattr(_auto_mask_module, "morphological_geodesic_active_contour", None)
else:  # pragma: no cover
    _AUTO_MASK_EXTRACT_ROI = None
    _AUTO_MASK_ENSURE_GRAY = None
    _AUTO_MASK_RUN_GRABCUT = None
    _AUTO_MASK_FALLBACK = None
    _AUTO_MASK_STRIP_PADDING = None
    _AUTO_MASK_REMOVE_BOUNDARY = None
    _AUTO_MASK_LARGEST = None
    _AUTO_MASK_GUIDED_FILTER = None
    _AUTO_MASK_MGAC = None

_AUTO_MASK_SUPPORT = all(
    fn is not None
    for fn in (
        _AUTO_MASK_EXTRACT_ROI,
        _AUTO_MASK_ENSURE_GRAY,
        _AUTO_MASK_RUN_GRABCUT,
        _AUTO_MASK_FALLBACK,
        _AUTO_MASK_STRIP_PADDING,
        _AUTO_MASK_REMOVE_BOUNDARY,
        _AUTO_MASK_LARGEST,
        _AUTO_MASK_GUIDED_FILTER,
    )
)

AUTO_MASK_RUNTIME_AVAILABLE = _AUTO_MASK_SUPPORT


def _build_empty_segmentation_from_bbox(
    bbox_raw: Sequence[float],
    frame_shape: Sequence[int],
) -> SegmentationData:
    frame_h = float(frame_shape[0]) if len(frame_shape) > 0 else 0.0
    frame_w = float(frame_shape[1]) if len(frame_shape) > 1 else 0.0
    try:
        x, y, w, h = map(float, bbox_raw)
    except Exception:
        x, y, w, h = 0.0, 0.0, frame_w, frame_h
    if frame_w > 0.0 and frame_h > 0.0:
        x = max(0.0, min(x, frame_w - 1.0))
        y = max(0.0, min(y, frame_h - 1.0))
        w = max(0.0, min(w, frame_w - x))
        h = max(0.0, min(h, frame_h - y))
    area = float(max(w * h, 0.0))
    perimeter = float(max(2.0 * (w + h), 0.0))
    eq_diam = float(math.sqrt(max(4.0 * area / math.pi, 0.0))) if area > 0.0 else 0.0
    stats = MaskStats(
        area_px=area,
        bbox=(x, y, w, h),
        centroid=(x + (w / 2.0), y + (h / 2.0)),
        perimeter_px=perimeter,
        equivalent_diameter_px=eq_diam,
    )
    return SegmentationData(mask_path=None, stats=stats, roi_bbox=(x, y, w, h), centroid_error_px=None)


@dataclass
class SegmentationConfig:
    model_name: str = "unetpp"
    model_params: Dict[str, Any] = field(default_factory=dict)
    padding_train_min: float = 0.10
    padding_train_max: float = 0.15
    padding_inference: float = 0.15
    batch_size: int = 8
    num_workers: int = 0
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-5
    threshold: float = 0.5
    redundancy: int = 1
    device: str = "auto"
    val_ratio: float = 0.0
    seed: int = 0
    dice_weight: float = 1.0
    bce_weight: float = 1.0
    train: bool = True
    pretrained_external: Optional[str] = None
    inference_checkpoint: Optional[str] = None
    jitter: float = 0.0
    auto_pretrained: bool = False
    target_size: Tuple[int, int] = (256, 256)

    @classmethod
    def from_dict(cls, cfg: Optional[Dict]) -> "SegmentationConfig":
        params = dict(cfg or {})
        method_cfg = params.get("method")
        model_cfg = params.get("model")
        model_name = params.get("model_name")
        model_params: Dict[str, Any] = {}
        config_candidates: List[Dict[str, Any]] = []
        if isinstance(model_cfg, dict):
            config_candidates.append(model_cfg)
        if isinstance(method_cfg, dict):
            config_candidates.append(method_cfg)
        for entry in config_candidates:
            if isinstance(entry.get("params"), dict):
                model_params.update(entry.get("params") or {})
            name_candidate = entry.get("name")
            if isinstance(name_candidate, str) and name_candidate.strip():
                model_name = name_candidate
        if isinstance(params.get("model_params"), dict):
            model_params.update(params.get("model_params") or {})
        # backward compatibility
        if "encoder_name" in params and "encoder_name" not in model_params:
            model_params["encoder_name"] = params["encoder_name"]
        if "encoder_weights" in params and "encoder_weights" not in model_params:
            model_params["encoder_weights"] = params["encoder_weights"]
        name = str(model_name or "unetpp").strip().lower()
        train_raw = params.get("train", params.get("enabled", True))
        if isinstance(train_raw, str):
            train_norm = train_raw.strip().lower()
            train_flag = train_norm not in {"", "0", "false", "no", "n"}
        else:
            train_flag = bool(train_raw) if train_raw is not None else True

        legacy_shared = params.get("pretrained_weights")
        pretrained_raw = (
            params.get("pretrained_external")
            or params.get("external_pretrained")
            or legacy_shared
        )
        auto_flag = bool(params.get("auto_pretrained", False))
        target_size_val = params.get("target_size")
        if target_size_val is None:
            target_size_val = params.get("resize")
        if target_size_val is None:
            target_size_val = params.get("crop_size")
        target_size_tuple = cls._parse_target_size(target_size_val)
        pretrained_val = None
        if isinstance(pretrained_raw, str):
            candidate = pretrained_raw.strip()
            if candidate.lower() in {"auto", "default"}:
                auto_flag = True
                pretrained_val = None
            else:
                pretrained_val = candidate or None

        inference_raw = (
            params.get("inference_checkpoint")
            or params.get("checkpoint")
            or params.get("weights")
            or legacy_shared
        )
        inference_val = None
        if isinstance(inference_raw, str):
            candidate = inference_raw.strip()
            if candidate.lower() in {"auto", "default"}:
                auto_flag = True
                inference_val = None
            else:
                inference_val = candidate or None
        if inference_val is None and not train_flag:
            inference_val = pretrained_val
        jitter_raw = params.get("jitter", params.get("jitter_translate", 0.0))
        try:
            jitter_val = float(jitter_raw)
        except Exception:
            jitter_val = 0.0

        return cls(
            model_name=name,
            model_params=model_params,
            padding_train_min=float(params.get("padding_min", 0.10)),
            padding_train_max=float(params.get("padding_max", 0.15)),
            padding_inference=float(params.get("padding_inference", 0.15)),
            batch_size=int(params.get("batch_size", 8)),
            num_workers=int(params.get("num_workers", 0)),
            epochs=int(params.get("epochs", 20)),
            lr=float(params.get("lr", 1e-3)),
            weight_decay=float(params.get("weight_decay", 1e-5)),
            threshold=float(params.get("threshold", 0.5)),
            redundancy=int(params.get("redundancy", 1)),
            device=str(params.get("device", "auto")),
            val_ratio=float(params.get("val_ratio", 0.0)),
            seed=int(params.get("seed", 0)),
            dice_weight=float(params.get("dice_weight", 1.0)),
            bce_weight=float(params.get("bce_weight", 1.0)),
            train=train_flag,
            pretrained_external=pretrained_val,
            inference_checkpoint=inference_val,
            jitter=jitter_val,
            auto_pretrained=auto_flag,
            target_size=target_size_tuple,
        )

    @staticmethod
    def _parse_target_size(value: Any) -> Tuple[int, int]:
        default = (256, 256)
        if value in (None, "", "auto", "default"):
            return default
        try:
            if isinstance(value, (list, tuple)):
                nums = [int(float(v)) for v in value if float(v) > 0]
                if len(nums) >= 2:
                    return (nums[0], nums[1])
                if len(nums) == 1:
                    return (nums[0], nums[0])
            if isinstance(value, str):
                norm = value.lower()
                if "x" in norm:
                    parts = [p.strip() for p in norm.replace("*", "x").split("x") if p.strip()]
                    nums = [int(float(p)) for p in parts if float(p) > 0]
                    if len(nums) >= 2:
                        return (nums[0], nums[1])
                    if len(nums) == 1:
                        return (nums[0], nums[0])
                else:
                    v = int(float(norm))
                    if v > 0:
                        return (v, v)
            if isinstance(value, (int, float)):
                v = int(value)
                if v > 0:
                    return (v, v)
        except Exception:
            pass
        return default


def _resolve_device(device_hint: str) -> torch.device:
    hint = device_hint.strip().lower()
    if hint in {"", "auto"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():  # pragma: no cover
            return torch.device("mps")
        return torch.device("cpu")
    if hint.startswith("cuda") and torch.cuda.is_available():
        return torch.device(hint)
    if hint in {"mps", "metal"} and torch.backends.mps.is_available():  # pragma: no cover
        return torch.device("mps")
    return torch.device("cpu")


def _dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    num = (probs * target).sum(dim=(1, 2, 3)) * 2.0 + smooth
    den = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + smooth
    dice = num / den
    return 1.0 - dice.mean()


class SegmentationWorkflow:
    def __init__(
        self,
        config: Optional[Dict] = None,
        dataset_root: Optional[str] = None,
        results_dir: Optional[str] = None,
        logger=None,
    ) -> None:
        self.cfg = SegmentationConfig.from_dict(config)
        self.dataset_root = dataset_root or "."
        self.results_dir = results_dir or os.path.join(os.getcwd(), "results", "segmentation")
        ensure_dir(self.results_dir)
        self.logger = logger or (lambda msg: None)
        try:
            self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        except Exception:
            self.project_root = os.getcwd()
        self.device = _resolve_device(self.cfg.device)
        model_key = str(self.cfg.model_name or "").strip().lower()
        model_params = dict(self.cfg.model_params)
        self.model_name = model_key or "unetpp"
        self.using_auto_mask = self.model_name == "auto_mask"
        self.using_medsam = self.model_name == "medsam"
        self.model: Optional[torch.nn.Module] = None
        self.train_enabled = bool(self.cfg.train)
        self.auto_pretrained = bool(getattr(self.cfg, "auto_pretrained", False))
        self.input_size: Optional[Tuple[int, int]] = tuple(int(v) for v in self.cfg.target_size)
        self.pretrained_external = self._resolve_checkpoint_path(self.cfg.pretrained_external)
        self.inference_checkpoint = self._resolve_checkpoint_path(self.cfg.inference_checkpoint)
        self.best_checkpoint: Optional[str] = None
        self.using_default_pretrained = False
        self.auto_mask_options: Dict[str, Any] = {}
        self._cuda_fallback_triggered = False
        # Runtime-injected preproc chains.
        # - preprocs: applied to full frames before ROI cropping (global scope)
        # - roi_preprocs: applied to ROI crops after cropping (ROI scope)
        self.preprocs: List[PreprocessingModule] = []
        self.roi_preprocs: List[PreprocessingModule] = []

        if self.using_auto_mask:
            if not _AUTO_MASK_SUPPORT:
                raise ImportError(
                    "Auto-mask segmentation requires the 'segmentation_annotator.auto_mask' helpers to be available."
                )
            self.train_enabled = False
            self.auto_pretrained = False
            self.auto_mask_options = self._init_auto_mask_options(model_params)
            self._auto_mask_warning_emitted = False
            self.logger("[Segmentation] Auto-mask mode enabled (GrabCut + MGAC refinements).")
        else:
            model_cls = SEGMENTATION_MODEL_REGISTRY.get(self.model_name)
            if model_cls is None:
                raise KeyError(f"Unknown segmentation model: {self.model_name}")
            model_params.setdefault("in_channels", 3)
            model_params.setdefault("classes", 1)
            if self.using_medsam and "checkpoint" not in model_params:
                fallback_ckpt = self.inference_checkpoint or self.pretrained_external
                if fallback_ckpt:
                    model_params["checkpoint"] = fallback_ckpt
            # For MedSAM, pass train_enabled to enable fine-tuning mode
            if self.using_medsam:
                model_params["train_enabled"] = self.train_enabled
            self.model = model_cls(model_params).to(self.device)
            if self.using_medsam:
                if not self.train_enabled:
                    self.auto_pretrained = False
                    self.input_size = None
                    self.logger("[Segmentation] MedSAM mode enabled (inference-only).")
                else:
                    self.input_size = tuple(int(v) for v in self.cfg.target_size)
                    self.logger("[Segmentation] MedSAM mode enabled (fine-tuning mask decoder).")
            if self.auto_pretrained or (
                not self.train_enabled
                and not self.pretrained_external
                and not self.inference_checkpoint
            ):
                self._maybe_use_default_pretrained_model()

    @staticmethod
    def _apply_preprocs_frame(frame: np.ndarray, preprocs: Sequence[PreprocessingModule]) -> np.ndarray:
        """Apply preprocessing modules to a frame (BGR or grayscale).

        Preprocessing modules operate on RGB for 3-channel images.
        Frames are always converted to grayscale first to ensure consistent processing.
        """
        if frame.ndim == 3:
            _g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(_g, cv2.COLOR_GRAY2BGR)
        if not preprocs:
            return frame
        if frame.ndim == 2:
            out = frame
            for p in preprocs:
                if getattr(p, "train_only", False):
                    continue
                out = p.apply_to_frame(out)
            return out
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for p in preprocs:
            if getattr(p, "train_only", False):
                continue
            rgb = p.apply_to_frame(rgb)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # ------------------------------------------------------------------
    @staticmethod
    def _coerce_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            norm = value.strip().lower()
            if norm in {"", "0", "false", "no", "off"}:
                return False
            if norm in {"1", "true", "yes", "on"}:
                return True
            return default
        return bool(value)

    def _init_auto_mask_options(self, params: Dict[str, Any]) -> Dict[str, Any]:
        options: Dict[str, Any] = {}
        margin_raw = params.get("margin", params.get("expand_ratio", 0.2))
        try:
            margin_val = float(margin_raw)
        except Exception:
            margin_val = 0.2
        options["margin"] = max(0.0, float(margin_val))

        num_iter_raw = params.get("num_iter", params.get("iterations", 300))
        try:
            num_iter_val = int(num_iter_raw)
        except Exception:
            num_iter_val = 300
        options["num_iter"] = max(10, num_iter_val)

        guided_radius_raw = params.get("guided_radius", params.get("radius", 4))
        try:
            guided_radius_val = int(guided_radius_raw)
        except Exception:
            guided_radius_val = 4
        options["guided_radius"] = max(1, guided_radius_val)

        guided_eps_raw = params.get("guided_eps", params.get("epsilon", 1e-3))
        try:
            guided_eps_val = float(guided_eps_raw)
        except Exception:
            guided_eps_val = 1e-3
        options["guided_eps"] = max(1e-6, guided_eps_val)

        edge_weight_beta_raw = params.get("edge_weight_beta", 10.0)
        try:
            edge_weight_beta_val = float(edge_weight_beta_raw)
        except Exception:
            edge_weight_beta_val = 10.0
        options["edge_weight_beta"] = max(0.1, edge_weight_beta_val)

        kernel_raw = params.get("postprocess_kernel", params.get("kernel", 3))
        try:
            kernel_val = int(kernel_raw)
        except Exception:
            kernel_val = 3
        if kernel_val < 1:
            kernel_val = 3
        if kernel_val % 2 == 0:
            kernel_val += 1
        options["postprocess_kernel"] = min(max(kernel_val, 1), 15)

        open_iter_raw = params.get("open_iter", 1)
        close_iter_raw = params.get("close_iter", 1)
        dilate_iter_raw = params.get("dilate_iter", 0)
        try:
            options["open_iter"] = max(0, int(open_iter_raw))
        except Exception:
            options["open_iter"] = 1
        try:
            options["close_iter"] = max(0, int(close_iter_raw))
        except Exception:
            options["close_iter"] = 1
        try:
            options["dilate_iter"] = max(0, int(dilate_iter_raw))
        except Exception:
            options["dilate_iter"] = 0

        canny_low_raw = params.get("canny_low", 30)
        canny_high_raw = params.get("canny_high", 80)
        try:
            canny_low = int(canny_low_raw)
        except Exception:
            canny_low = 30
        try:
            canny_high = int(canny_high_raw)
        except Exception:
            canny_high = 80
        if canny_high <= canny_low:
            canny_high = canny_low + 30
        options["canny_low"] = max(0, canny_low)
        options["canny_high"] = max(options["canny_low"] + 1, canny_high)

        use_mgac_raw = params.get("use_mgac", params.get("mgac", True))
        options["use_mgac"] = self._coerce_bool(use_mgac_raw, default=True)

        mgac_balloon_raw = params.get("mgac_balloon", params.get("balloon", 1.0))
        try:
            options["mgac_balloon"] = float(mgac_balloon_raw)
        except Exception:
            options["mgac_balloon"] = 1.0

        mgac_smoothing_raw = params.get("mgac_smoothing", params.get("smoothing", 1))
        try:
            options["mgac_smoothing"] = max(0, int(mgac_smoothing_raw))
        except Exception:
            options["mgac_smoothing"] = 1

        mgac_iter_scale_raw = params.get("mgac_iter_scale", params.get("mgac_fraction", 0.25))
        try:
            options["mgac_iter_scale"] = max(0.05, float(mgac_iter_scale_raw))
        except Exception:
            options["mgac_iter_scale"] = 0.25

        mgac_min_iter_raw = params.get("mgac_min_iter", 60)
        try:
            options["mgac_min_iter"] = max(1, int(mgac_min_iter_raw))
        except Exception:
            options["mgac_min_iter"] = 60

        return options

    # ------------------------------------------------------------------
    def train(
        self,
        train_videos: Sequence[str],
        val_videos: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, float]:
        if self.using_auto_mask:
            self.logger("[Segmentation] Auto-mask mode is inference-only; skipping training stage.")
            return {"status": "auto_mask_inference_only"}
        if self.using_medsam and not self.train_enabled:
            self.logger("[Segmentation] MedSAM mode is inference-only; skipping training stage.")
            return {"status": "medsam_inference_only"}
        if not self.train_enabled:
            self.logger("[Segmentation] Training disabled via config; skipping training stage.")
            return {}
        if not train_videos:
            self.logger("[Segmentation] No training videos provided; skipping training.")
            return {}
        if self.pretrained_external:
            resolved_pretrained = self._resolve_checkpoint_path(self.pretrained_external)
            if resolved_pretrained and os.path.exists(resolved_pretrained):
                self.logger(f"[Segmentation] Warm-start from external weights: {resolved_pretrained}")
                self.load_checkpoint(resolved_pretrained)
            else:
                self.logger(f"[Segmentation] External pretrained weights not found: {self.pretrained_external}")
        train_seed = self.cfg.seed if seed is None else int(seed)
        dataset = SegmentationCropDataset(
            train_videos,
            dataset_root=self.dataset_root,
            padding_range=(self.cfg.padding_train_min, self.cfg.padding_train_max),
            redundancy=max(1, self.cfg.redundancy),
            seed=train_seed,
            jitter=self.cfg.jitter,
            target_size=self.input_size,
            preprocs=self.preprocs,
            roi_preprocs=self.roi_preprocs,
        )
        if getattr(dataset, "missing_annotations", None):
            missing = dataset.missing_annotations
            preview = ", ".join(os.path.basename(p) for p in missing[:5])
            more = "" if len(missing) <= 5 else " …"
            self.logger(
                f"[Segmentation] Missing annotations for {len(missing)} video(s): {preview}{more}"
            )

        if len(dataset) == 0:
            self.logger("[Segmentation] No training samples available after filtering; skipping training stage.")
            return {"status": "no_data"}

        loader = DataLoader(
            dataset,
            batch_size=max(1, self.cfg.batch_size),
            shuffle=True,
            num_workers=max(0, self.cfg.num_workers),
            pin_memory=self.device.type == "cuda",
        )
        val_loader = None
        if val_videos:
            val_dataset = SegmentationCropDataset(
                val_videos,
                dataset_root=self.dataset_root,
                padding_range=(self.cfg.padding_inference, self.cfg.padding_inference),
                redundancy=1,
                seed=train_seed,
                jitter=0.0,
                target_size=self.input_size,
                preprocs=self.preprocs,
                roi_preprocs=self.roi_preprocs,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=max(1, self.cfg.batch_size),
                shuffle=False,
                num_workers=max(0, self.cfg.num_workers),
                pin_memory=self.device.type == "cuda",
            )
        # Select parameters to optimize based on model type
        if self.using_medsam and hasattr(self.model, "get_trainable_parameters"):
            params_to_optimize = list(self.model.get_trainable_parameters())
            if not params_to_optimize:
                self.logger("[Segmentation] MedSAM has no trainable parameters; skipping training.")
                return {"status": "no_trainable_params"}
            self.logger(f"[Segmentation] Training MedSAM mask decoder with {len(params_to_optimize)} parameter groups.")
        else:
            params_to_optimize = self.model.parameters()

        optimiser = torch.optim.AdamW(
            params_to_optimize,
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        best_val = math.inf
        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        for epoch in range(1, self.cfg.epochs + 1):
            self.model.train()
            loss_epoch = 0.0
            for batch in loader:
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                if images.shape[0] < 2 or images.shape[2] < 2 or images.shape[3] < 2:
                    if not getattr(self, "_warned_tiny_batch", False):
                        self.logger("[Segmentation] Skipping tiny batch (batch<2 or spatial<2) to avoid BatchNorm error.")
                        self._warned_tiny_batch = True
                    continue
                optimiser.zero_grad(set_to_none=True)
                logits = self.model(images)
                loss_bce = F.binary_cross_entropy_with_logits(logits, masks)
                loss_dice = _dice_loss(logits, masks)
                loss = self.cfg.bce_weight * loss_bce + self.cfg.dice_weight * loss_dice
                loss.backward()
                optimiser.step()
                loss_epoch += float(loss.detach().cpu()) * images.size(0)
            loss_epoch /= len(loader.dataset)
            history["train_loss"].append(loss_epoch)
            self.logger(f"[Segmentation] Epoch {epoch}/{self.cfg.epochs} loss={loss_epoch:.4f}")
            if val_loader is not None:
                val_loss = self._evaluate(val_loader)
                history["val_loss"].append(val_loss)
                if val_loss < best_val:
                    best_val = val_loss
                    self.best_checkpoint = os.path.join(self.results_dir, "segmentation_best.pt")
                    self._save_checkpoint(self.best_checkpoint)
                    self.logger(f"[Segmentation] Saved best checkpoint @ {self.best_checkpoint}")
            else:
                # save latest when no validation
                self.best_checkpoint = os.path.join(self.results_dir, "segmentation_last.pt")
                self._save_checkpoint(self.best_checkpoint)
        return {k: v[-1] for k, v in history.items() if v}

    def _save_checkpoint(self, path: str) -> None:
        """Save model checkpoint, handling MedSAM's special save method."""
        if self.using_medsam and hasattr(self.model, "save_finetuned"):
            self.model.save_finetuned(path)
        else:
            torch.save(self.model.state_dict(), path)

    # ------------------------------------------------------------------
    def _evaluate(self, loader: DataLoader) -> float:
        if self.using_auto_mask:
            return 0.0
        self.model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                logits = self.model(images)
                loss_bce = F.binary_cross_entropy_with_logits(logits, masks)
                loss_dice = _dice_loss(logits, masks)
                loss = self.cfg.bce_weight * loss_bce + self.cfg.dice_weight * loss_dice
                total += float(loss.cpu()) * images.size(0)
                count += images.size(0)
        return total / max(1, count)

    # ------------------------------------------------------------------
    def load_checkpoint(self, path: Optional[str] = None) -> None:
        if self.using_auto_mask:
            self.logger("[Segmentation] Auto-mask mode does not use checkpoints; skipping load request.")
            return
        if self.using_medsam and not self.train_enabled:
            self.logger("[Segmentation] MedSAM weights are provided via model.params.checkpoint; skipping load request.")
            return
        ckpt_hint = path if path not in (None, "") else None
        if ckpt_hint is None:
            if self.best_checkpoint:
                ckpt_hint = self.best_checkpoint
            elif not self.train_enabled:
                ckpt_hint = (
                    self.inference_checkpoint
                    or self.cfg.inference_checkpoint
                    or self.pretrained_external
                    or self.cfg.pretrained_external
                )
            else:
                ckpt_hint = self.pretrained_external or self.cfg.pretrained_external
        if getattr(self, "using_default_pretrained", False) and ckpt_hint is None:
            self.logger("[Segmentation] Using built-in pretrained weights; no checkpoint to load.")
            return
        ckpt_path = self._resolve_checkpoint_path(ckpt_hint)
        if ckpt_path and os.path.exists(ckpt_path):
            if self.using_medsam and hasattr(self.model, "load_finetuned"):
                # Try to load as MedSAM fine-tuned checkpoint
                try:
                    self.model.load_finetuned(ckpt_path)
                    self.best_checkpoint = ckpt_path
                    self.logger(f"[Segmentation] Loaded MedSAM fine-tuned checkpoint: {ckpt_path}")
                    return
                except Exception:
                    # Fall through to standard loading
                    pass
            state = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(state)
            self.best_checkpoint = ckpt_path
            self.logger(f"[Segmentation] Loaded checkpoint: {ckpt_path}")
        else:
            if ckpt_hint:
                self.logger(f"[Segmentation] No checkpoint found at: {ckpt_hint}")
            else:
                self.logger("[Segmentation] No checkpoint found to load.")

    # ------------------------------------------------------------------
    def predict_dataset(
        self,
        video_predictions: Dict[str, List[FramePrediction]],
        output_root: str,
        gt_annotations: Optional[Dict[str, Dict]] = None,
        viz_settings: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ensure_dir(output_root)
        metrics_by_video: Dict[str, Dict[str, float]] = {}
        dataset_accum: Dict[str, List[float]] = {"dice": [], "iou": [], "centroid": [], "fps": []}
        for video_path, predictions in video_predictions.items():
            gt = None
            if gt_annotations and video_path in gt_annotations:
                gt = gt_annotations[video_path]
            # Two-level layout: {subject_id}/{vid_stem} to avoid collision when
            # different subjects share the same video filename.
            subject_id = os.path.basename(os.path.dirname(video_path))
            vid_stem = os.path.splitext(os.path.basename(video_path))[0]
            out_dir = os.path.join(output_root, subject_id, vid_stem)
            ensure_dir(out_dir)
            metrics, raw_accum = self.predict_video(video_path, predictions, out_dir, gt, viz_settings)
            metrics_by_video[video_path] = metrics
            for key, values in raw_accum.items():
                if not values:
                    continue
                dataset_accum.setdefault(key, []).extend(values)
            if "fps" in metrics:
                try:
                    dataset_accum.setdefault("fps", []).append(float(metrics["fps"]))
                except Exception:
                    pass
        summary = summarise_metrics(dataset_accum)
        return {"summary": summary, "videos": metrics_by_video}

    # ------------------------------------------------------------------
    def predict_video(
        self,
        video_path: str,
        predictions: List[FramePrediction],
        output_dir: str,
        gt_annotation: Optional[Dict] = None,
        viz_settings: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
        ensure_dir(output_dir)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video for segmentation inference: {video_path}")
        fallback_frame_shape = (
            int(max(1, round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1))),
            int(max(1, round(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1))),
        )
        model_to_restore = getattr(self, "model", None)
        restore_training_state: Optional[bool] = None
        if not self.using_auto_mask and model_to_restore is not None:
            if hasattr(model_to_restore, "eval") and hasattr(model_to_restore, "train"):
                restore_training_state = bool(getattr(model_to_restore, "training", False))
                model_to_restore.eval()
        gt_by_frame: Dict[int, FramePrediction] = {}
        if gt_annotation is not None:
            gt_samples = attach_ground_truth_segmentation(
                gt_annotation, self.dataset_root, video_path=video_path
            )
            gt_by_frame = {sample.frame_index: sample for sample in gt_samples}
        accum: Dict[str, list] = {"dice": [], "iou": [], "centroid": []}
        per_frame_metrics: Dict[int, Dict[str, Optional[float]]] = {}
        mask_files: Dict[int, str] = {}
        total_infer_time = 0.0
        frame_counter = 0

        def _ensure_mean_std_keys(summary: Dict[str, float]) -> Dict[str, float]:
            # Viewer expects *_mean/std; keep legacy keys too.
            for k in ("dice", "iou", "centroid"):
                if k in summary and f"{k}_mean" not in summary:
                    try:
                        summary[f"{k}_mean"] = float(summary.get(k, 0.0))
                    except Exception:
                        summary[f"{k}_mean"] = 0.0
                    summary.setdefault(f"{k}_std", 0.0)
            return summary

        # If GT exists, we should evaluate ALL GT frames.
        # In addition, inference must still run on ALL prediction frames so
        # downstream classification can consume per-frame segmentation outputs.
        # Missing bbox handling order:
        # 1) Use bbox at the same frame (if any)
        # 2) Fallback to previous bbox
        # 3) If even the first bbox is missing, fallback to full-frame ROI
        gt_frame_indices: List[int] = sorted(int(i) for i in gt_by_frame.keys()) if gt_by_frame else []
        pred_by_frame: Dict[int, FramePrediction] = {int(p.frame_index): p for p in predictions}
        work_predictions: List[FramePrediction] = []
        try:
            if gt_frame_indices:
                infer_frame_indices = sorted(set(gt_frame_indices) | set(pred_by_frame.keys()))
                # last_bbox_raw stores an unpadded/unexpanded bbox estimate (either from detector
                # or from bootstrap using the segmentation mask). We intentionally do NOT store
                # padded/expanded ROI bboxes here to avoid expansion accumulating over time.
                last_bbox_raw: Optional[tuple] = None
                last_bbox_source: Optional[str] = None
                for frame_idx in infer_frame_indices:
                    pred = pred_by_frame.get(int(frame_idx))
                    use_bbox_raw: Optional[tuple] = None
                    used_prev_bbox = False
                    forced_full_frame = False
                    bbox_source = "detector"
                    pad_fraction = float(self.cfg.padding_inference)
                    pred_bbox_raw: Optional[tuple] = None
                    min_raw_wh = 2.0
                    if pred is not None:
                        try:
                            pred_bbox_raw = tuple(map(float, pred.bbox))
                        except Exception:
                            pred_bbox_raw = None

                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        if pred is not None and pred.segmentation is None:
                            pred.segmentation = _build_empty_segmentation_from_bbox(
                                getattr(pred, "bbox", (0.0, 0.0, 0.0, 0.0)),
                                fallback_frame_shape,
                            )
                        continue
                    frame = self._apply_preprocs_frame(frame, self.preprocs)
                    frame_h, frame_w = frame.shape[:2]

                    pred_bbox_valid = None
                    if pred_bbox_raw is not None and pred_bbox_raw[2] >= min_raw_wh and pred_bbox_raw[3] >= min_raw_wh:
                        pred_bbox_valid = pred_bbox_raw

                    if pred_bbox_valid is not None:
                        # Best case: detector bbox for this frame (non-tiny).
                        use_bbox_raw = pred_bbox_valid
                        last_bbox_raw = pred_bbox_valid
                        last_bbox_source = "detector"
                        bbox_source = "detector"
                    elif last_bbox_raw is not None and last_bbox_raw[2] >= min_raw_wh and last_bbox_raw[3] >= min_raw_wh:
                        # Strategy 1: detector bbox missing or tiny -> reuse previous bbox.
                        # When reusing, expand by at least 15% per side to tolerate motion/dropouts,
                        # but keep last_bbox_raw unexpanded to avoid growth across frames.
                        pad_fraction = max(pad_fraction, 0.15)
                        use_bbox_raw = last_bbox_raw
                        used_prev_bbox = True
                        bbox_source = "prev_segmentation" if last_bbox_source == "segmentation_bootstrap" else "prev_bbox"
                    else:
                        # Strategy 0: no bbox yet (or only tiny bbox) -> full-frame segmentation.
                        use_bbox_raw = (0.0, 0.0, float(frame_w), float(frame_h))
                        forced_full_frame = True
                        bbox_source = "full_frame"

                    if pred is not None:
                        work_pred = FramePrediction(
                            frame_index=int(frame_idx),
                            bbox=use_bbox_raw,
                            score=pred.score,
                            confidence=getattr(pred, "confidence", None),
                            confidence_components=getattr(pred, "confidence_components", None),
                            segmentation=getattr(pred, "segmentation", None),
                            is_fallback=bool(getattr(pred, "is_fallback", False) or used_prev_bbox or forced_full_frame),
                            bbox_source=bbox_source,
                        )
                    else:
                        work_pred = FramePrediction(
                            frame_index=int(frame_idx),
                            bbox=use_bbox_raw,
                            score=None,
                            is_fallback=used_prev_bbox or forced_full_frame,
                            bbox_source=bbox_source,
                        )
                    work_predictions.append(work_pred)
                    bbox = expand_bbox(use_bbox_raw, pad_fraction, (frame_h, frame_w))
                    if bbox.w < 2.0 or bbox.h < 2.0:
                        # Expanded bbox still unusably small -> treat as detection failure and use full frame.
                        bbox = BoundingBox(0.0, 0.0, float(frame_w), float(frame_h))
                        forced_full_frame = True
                    roi = crop_with_bbox(frame, bbox)
                    if roi.size == 0:
                        # If ROI collapses to empty, fall back to full-frame zeros to ensure metrics are recorded.
                        roi = np.zeros((1, 1, 3), dtype=frame.dtype)
                        full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                        mask_roi = np.zeros((int(max(1, round(bbox.h))), int(max(1, round(bbox.w)))), dtype=np.uint8)
                        computed_mask = False
                    else:
                        full_mask = None  # will be set below
                        computed_mask = True
                    # ROI-scope preprocs are applied after cropping.
                    if self.roi_preprocs:
                        roi = self._apply_preprocs_frame(roi, self.roi_preprocs)
                    if self.using_medsam and self.model is not None and hasattr(self.model, "set_prompts"):
                        prompt_box = self._project_bbox_to_roi(use_bbox_raw, bbox, roi.shape[:2])
                        if prompt_box is not None:
                            box_flat = prompt_box.reshape(-1, 4)[0]
                            cx = float((box_flat[0] + box_flat[2]) * 0.5)
                            cy = float((box_flat[1] + box_flat[3]) * 0.5)
                            self.model.set_prompts([
                                {
                                    "box": prompt_box,
                                    "point_coords": [[cx, cy]],
                                    "point_labels": [1],
                                }
                            ])
                        else:
                            roi_h, roi_w = roi.shape[:2]
                            if roi_h > 0 and roi_w > 0:
                                cx = float(max(0.0, min(roi_w - 1.0, roi_w / 2.0)))
                                cy = float(max(0.0, min(roi_h - 1.0, roi_h / 2.0)))
                                self.model.set_prompts([
                                    {
                                        "point_coords": [[cx, cy]],
                                        "point_labels": [1],
                                    }
                                ])
                            else:
                                self.model.set_prompts(None)

                    if self.using_auto_mask:
                        start_time = time.perf_counter()
                        auto_full = self._auto_mask_generate_full(frame, bbox)
                        infer_elapsed = time.perf_counter() - start_time
                        total_infer_time += infer_elapsed
                        frame_counter += 1
                        if auto_full is None:
                            if not getattr(self, "_auto_mask_warning_emitted", False):
                                self.logger("[Segmentation] Auto-mask generation failed for a frame; falling back to empty mask.")
                                self._auto_mask_warning_emitted = True
                            auto_full = np.zeros(frame.shape[:2], dtype=np.uint8)
                        mask_roi = crop_with_bbox(auto_full, bbox)
                        if mask_roi.size == 0:
                            mask_roi = np.zeros(
                                (
                                    int(max(1, round(bbox.h))),
                                    int(max(1, round(bbox.w))),
                                ),
                                dtype=np.uint8,
                            )
                        else:
                            if mask_roi.dtype != np.uint8:
                                mask_roi = (mask_roi > 0).astype(np.uint8) * 255
                        mask_roi = keep_largest_component(mask_roi)
                        mask_roi = fill_holes(mask_roi)
                        full_mask = place_mask_on_canvas(frame.shape[:2], mask_roi, bbox)
                    else:
                        if roi.ndim == 2:
                            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
                        orig_h, orig_w = roi.shape[:2]
                        if orig_h < 2 or orig_w < 2:
                            # Too small for BatchNorm; skip model and emit empty mask.
                            full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                            mask_roi = np.zeros((int(max(1, round(bbox.h))), int(max(1, round(bbox.w)))), dtype=np.uint8)
                            computed_mask = False
                        else:
                            if self.input_size:
                                roi_resized = cv2.resize(
                                    roi,
                                    (self.input_size[1], self.input_size[0]),
                                    interpolation=cv2.INTER_LINEAR,
                                )
                            else:
                                roi_resized = roi
                            roi_tensor = (
                                torch.from_numpy(roi_resized.astype(np.float32) / 255.0)
                                .permute(2, 0, 1)
                                .unsqueeze(0)
                            )
                            roi_tensor = roi_tensor.to(self.device)
                            start_time = time.perf_counter()
                            logits = self._run_model_with_fallback(roi_tensor)
                            probs = torch.sigmoid(logits)
                            infer_elapsed = time.perf_counter() - start_time
                            total_infer_time += infer_elapsed
                            frame_counter += 1
                            mask_roi = (probs.squeeze().cpu().numpy() > self.cfg.threshold).astype(np.uint8) * 255
                            mask_roi = keep_largest_component(mask_roi)
                            mask_roi = fill_holes(mask_roi)
                            if self.input_size and (orig_h, orig_w) != self.input_size:
                                mask_roi = cv2.resize(mask_roi, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                            mask_roi = keep_largest_component(mask_roi)
                            mask_roi = fill_holes(mask_roi)
                            full_mask = place_mask_on_canvas(frame.shape[:2], mask_roi, bbox)

                    # If earlier we forced an empty ROI, ensure full_mask is defined
                    if full_mask is None:
                        full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                        mask_roi = np.zeros((int(max(1, round(bbox.h))), int(max(1, round(bbox.w)))), dtype=np.uint8)
                        computed_mask = False

                    stats = compute_mask_stats(full_mask)

                    # Strategy 2 (bootstrap): if we were forced to do full-frame segmentation and
                    # it produced a non-empty mask, derive a bbox from the mask and store it as the
                    # next frame's last_bbox_raw. This allows subsequent frames to use Strategy 1.
                    if forced_full_frame and float(getattr(stats, "area_px", 0.0)) > 0.0:
                        try:
                            bx, by, bw, bh = tuple(map(float, stats.bbox))
                        except Exception:
                            bx = by = bw = bh = 0.0
                        if bw > 0.0 and bh > 0.0:
                            last_bbox_raw = (bx, by, bw, bh)
                            last_bbox_source = "segmentation_bootstrap"
                            # Also reflect the bootstrapped bbox in the per-frame prediction object.
                            try:
                                work_pred.bbox = last_bbox_raw  # type: ignore[assignment]
                                # This bbox itself is derived from the segmentation mask, but note that
                                # the ROI used for THIS frame may still have been full-frame.
                                if getattr(work_pred, "bbox_source", "") == "full_frame":
                                    work_pred.bbox_source = "segmentation_bootstrap"  # type: ignore[assignment]
                            except Exception:
                                pass
                    dice = None
                    iou = None
                    centroid_err = None
                    gt_mask = None
                    has_gt = frame_idx in gt_by_frame
                    if has_gt:
                        gt_entry = gt_by_frame[frame_idx]
                        gt_mask = self._load_mask_from_annotation(video_path, gt_entry.segmentation)
                        if gt_mask is not None:
                            dice = dice_coefficient(full_mask, gt_mask)
                            iou = intersection_over_union(full_mask, gt_mask)
                            centroid_err = centroid_distance(stats, gt_entry.segmentation.stats) if gt_entry.segmentation else None
                            accum["dice"].append(float(dice))
                            accum["iou"].append(float(iou))
                            if centroid_err is not None:
                                accum["centroid"].append(float(centroid_err))
                        else:
                            # GT exists but cannot load mask -> treat as miss.
                            dice = 0.0
                            iou = 0.0
                            centroid_err = None
                            accum["dice"].append(0.0)
                            accum["iou"].append(0.0)

                    # If GT exists but mask computation failed (empty ROI, etc.), count as miss (0).
                    if has_gt and (gt_mask is None or not computed_mask):
                        if dice is None:
                            dice = 0.0
                            accum["dice"].append(0.0)
                        if iou is None:
                            iou = 0.0
                            accum["iou"].append(0.0)
                        # centroid_err stays None in this fallback

                    per_frame_metrics[frame_idx] = {
                        "dice": float(dice) if has_gt else None,
                        "iou": float(iou) if has_gt else None,
                        "centroid": float(centroid_err) if centroid_err is not None else None,
                    }
                    mask_filename = os.path.join(output_dir, f"frame_{frame_idx:06d}.png")
                    cv2.imwrite(mask_filename, full_mask)
                    mask_files[frame_idx] = mask_filename
                    _seg_data = SegmentationData(
                        mask_path=mask_filename,
                        stats=stats,
                        roi_bbox=bbox.as_tuple(),
                        centroid_error_px=centroid_err,
                    )
                    work_pred.segmentation = _seg_data
                    # Also write back to the ORIGINAL FramePrediction so that
                    # downstream stages (classification) can read pred.segmentation.
                    # In the GT-evaluation path, work_pred is a NEW object; without
                    # this writeback the original pred in test_predictions stays None.
                    orig_pred = pred_by_frame.get(frame_idx)
                    if orig_pred is not None:
                        orig_pred.segmentation = _seg_data
            else:
                # Inference-only mode: only process frames where we have bbox predictions.
                work_predictions = list(predictions)
                last_bbox_raw: Optional[tuple] = None
                last_bbox_source: Optional[str] = None
                for pred in predictions:
                    frame_idx = int(pred.frame_index)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        if pred.segmentation is None:
                            pred.segmentation = _build_empty_segmentation_from_bbox(
                                getattr(pred, "bbox", (0.0, 0.0, 0.0, 0.0)),
                                fallback_frame_shape,
                            )
                        continue
                    frame = self._apply_preprocs_frame(frame, self.preprocs)

                    # Detection fallback: prefer current bbox; otherwise reuse last good bbox; otherwise full-frame.
                    pad_fraction = float(self.cfg.padding_inference)
                    forced_full_frame = False
                    min_raw_wh = 2.0
                    try:
                        pred_bbox_raw = tuple(map(float, pred.bbox))
                    except Exception:
                        pred_bbox_raw = None
                    use_bbox_raw = None
                    bbox_source = "detector"
                    if pred_bbox_raw is not None and pred_bbox_raw[2] >= min_raw_wh and pred_bbox_raw[3] >= min_raw_wh:
                        use_bbox_raw = pred_bbox_raw
                        last_bbox_source = "detector"
                        bbox_source = "detector"
                    elif last_bbox_raw is not None and last_bbox_raw[2] >= min_raw_wh and last_bbox_raw[3] >= min_raw_wh:
                        pad_fraction = max(pad_fraction, 0.15)
                        use_bbox_raw = last_bbox_raw
                        bbox_source = "prev_segmentation" if last_bbox_source == "segmentation_bootstrap" else "prev_bbox"
                    else:
                        use_bbox_raw = (0.0, 0.0, float(frame.shape[1]), float(frame.shape[0]))
                        forced_full_frame = True
                        bbox_source = "full_frame"

                    try:
                        pred.bbox_source = bbox_source  # type: ignore[attr-defined]
                    except Exception:
                        pass

                    bbox = expand_bbox(use_bbox_raw, pad_fraction, frame.shape[:2])
                    if bbox.w < 2.0 or bbox.h < 2.0:
                        bbox = BoundingBox(0.0, 0.0, float(frame.shape[1]), float(frame.shape[0]))
                        forced_full_frame = True

                    roi = crop_with_bbox(frame, bbox)
                    if roi.size == 0:
                        roi = np.zeros((1, 1, 3), dtype=frame.dtype)
                        full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                        mask_roi = np.zeros((int(max(1, round(bbox.h))), int(max(1, round(bbox.w)))), dtype=np.uint8)
                        computed_mask = False
                    else:
                        full_mask = None
                        computed_mask = True

                    # ROI-scope preprocs are applied after cropping.
                    if self.roi_preprocs:
                        roi = self._apply_preprocs_frame(roi, self.roi_preprocs)
                    if self.using_medsam and self.model is not None and hasattr(self.model, "set_prompts"):
                        prompt_box = self._project_bbox_to_roi(use_bbox_raw, bbox, roi.shape[:2])
                        if prompt_box is not None:
                            box_flat = prompt_box.reshape(-1, 4)[0]
                            cx = float((box_flat[0] + box_flat[2]) * 0.5)
                            cy = float((box_flat[1] + box_flat[3]) * 0.5)
                            self.model.set_prompts([
                                {
                                    "box": prompt_box,
                                    "point_coords": [[cx, cy]],
                                    "point_labels": [1],
                                }
                            ])
                        else:
                            roi_h, roi_w = roi.shape[:2]
                            if roi_h > 0 and roi_w > 0:
                                cx = float(max(0.0, min(roi_w - 1.0, roi_w / 2.0)))
                                cy = float(max(0.0, min(roi_h - 1.0, roi_h / 2.0)))
                                self.model.set_prompts([
                                    {
                                        "point_coords": [[cx, cy]],
                                        "point_labels": [1],
                                    }
                                ])
                            else:
                                self.model.set_prompts(None)

                    if self.using_auto_mask:
                        start_time = time.perf_counter()
                        auto_full = self._auto_mask_generate_full(frame, bbox)
                        infer_elapsed = time.perf_counter() - start_time
                        total_infer_time += infer_elapsed
                        frame_counter += 1
                        if auto_full is None:
                            if not getattr(self, "_auto_mask_warning_emitted", False):
                                self.logger("[Segmentation] Auto-mask generation failed for a frame; falling back to empty mask.")
                                self._auto_mask_warning_emitted = True
                            auto_full = np.zeros(frame.shape[:2], dtype=np.uint8)
                        mask_roi = crop_with_bbox(auto_full, bbox)
                        if mask_roi.size == 0:
                            mask_roi = np.zeros(
                                (
                                    int(max(1, round(bbox.h))),
                                    int(max(1, round(bbox.w))),
                                ),
                                dtype=np.uint8,
                            )
                        else:
                            if mask_roi.dtype != np.uint8:
                                mask_roi = (mask_roi > 0).astype(np.uint8) * 255
                        mask_roi = keep_largest_component(mask_roi)
                        mask_roi = fill_holes(mask_roi)
                        full_mask = place_mask_on_canvas(frame.shape[:2], mask_roi, bbox)
                    else:
                        if roi.ndim == 2:
                            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
                        orig_h, orig_w = roi.shape[:2]
                        if orig_h < 2 or orig_w < 2:
                            full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                            mask_roi = np.zeros((int(max(1, round(bbox.h))), int(max(1, round(bbox.w)))), dtype=np.uint8)
                            computed_mask = False
                        else:
                            if self.input_size:
                                roi_resized = cv2.resize(
                                    roi,
                                    (self.input_size[1], self.input_size[0]),
                                    interpolation=cv2.INTER_LINEAR,
                                )
                            else:
                                roi_resized = roi
                            roi_tensor = (
                                torch.from_numpy(roi_resized.astype(np.float32) / 255.0)
                                .permute(2, 0, 1)
                                .unsqueeze(0)
                            )
                            roi_tensor = roi_tensor.to(self.device)
                            start_time = time.perf_counter()
                            logits = self._run_model_with_fallback(roi_tensor)
                            probs = torch.sigmoid(logits)
                            infer_elapsed = time.perf_counter() - start_time
                            total_infer_time += infer_elapsed
                            frame_counter += 1
                            mask_roi = (probs.squeeze().cpu().numpy() > self.cfg.threshold).astype(np.uint8) * 255
                            mask_roi = keep_largest_component(mask_roi)
                            mask_roi = fill_holes(mask_roi)
                            if self.input_size and (orig_h, orig_w) != self.input_size:
                                mask_roi = cv2.resize(mask_roi, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                            mask_roi = keep_largest_component(mask_roi)
                            mask_roi = fill_holes(mask_roi)
                            full_mask = place_mask_on_canvas(frame.shape[:2], mask_roi, bbox)

                    if full_mask is None:
                        full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                        mask_roi = np.zeros((int(max(1, round(bbox.h))), int(max(1, round(bbox.w)))), dtype=np.uint8)
                        computed_mask = False

                    stats = compute_mask_stats(full_mask)

                    # Bootstrap bbox from successful segmentation for later fallback.
                    try:
                        if getattr(stats, "area_px", 0.0) > 0.0:
                            bx, by, bw, bh = tuple(map(float, stats.bbox))
                            if bw > 0.0 and bh > 0.0:
                                last_bbox_raw = (bx, by, bw, bh)
                                last_bbox_source = "segmentation_bootstrap"
                                try:
                                    pred.bbox = last_bbox_raw  # type: ignore[assignment]
                                    if getattr(pred, "bbox_source", "") == "full_frame":
                                        pred.bbox_source = "segmentation_bootstrap"  # type: ignore[attr-defined]
                                except Exception:
                                    pass
                    except Exception:
                        pass

                    dice = None
                    iou = None
                    centroid_err = None
                    gt_mask = None
                    if frame_idx in gt_by_frame:
                        gt_entry = gt_by_frame[frame_idx]
                        gt_mask = self._load_mask_from_annotation(video_path, gt_entry.segmentation)
                        if gt_mask is not None:
                            dice = dice_coefficient(full_mask, gt_mask)
                            iou = intersection_over_union(full_mask, gt_mask)
                            accum["dice"].append(dice)
                            accum["iou"].append(iou)
                            centroid_err = centroid_distance(stats, gt_entry.segmentation.stats) if gt_entry.segmentation else None
                            if centroid_err is not None:
                                accum["centroid"].append(centroid_err)
                    per_frame_metrics[frame_idx] = {
                        "dice": float(dice) if frame_idx in gt_by_frame and gt_mask is not None else None,
                        "iou": float(iou) if frame_idx in gt_by_frame and gt_mask is not None else None,
                        "centroid": float(centroid_err) if centroid_err is not None else None,
                    }
                    mask_filename = os.path.join(output_dir, f"frame_{frame_idx:06d}.png")
                    cv2.imwrite(mask_filename, full_mask)
                    mask_files[frame_idx] = mask_filename
                    pred.segmentation = SegmentationData(
                        mask_path=mask_filename,
                        stats=stats,
                        roi_bbox=bbox.as_tuple(),
                        centroid_error_px=centroid_err,
                    )
        finally:
            cap.release()
            if restore_training_state is not None and model_to_restore is not None:
                model_to_restore.train(restore_training_state)
        if mask_files:
            self._render_segmentation_visualizations(
                video_path,
                work_predictions,
                mask_files,
                output_dir,
                gt_by_frame,
                viz_settings or {},
            )
        metrics_summary = summarise_metrics(accum)
        # If we have GT frames but ended up with no evaluable samples, penalize instead of implicitly skipping.
        if gt_by_frame and not accum.get("dice") and not accum.get("iou"):
            metrics_summary = {
                "dice_mean": 0.0,
                "dice_std": 0.0,
                "iou_mean": 0.0,
                "iou_std": 0.0,
                "centroid_mean": 0.0,
                "centroid_std": 0.0,
            }
        metrics_summary = _ensure_mean_std_keys(metrics_summary)
        # Debug counters (non-breaking)
        try:
            metrics_summary.setdefault("debug_gt_frames", float(len(gt_by_frame) if gt_by_frame else 0))
            metrics_summary.setdefault("debug_pred_frames", float(len(predictions) if predictions else 0))
            metrics_summary.setdefault("debug_processed_frames", float(frame_counter))
        except Exception:
            pass
        if frame_counter > 0 and total_infer_time > 0.0:
            fps_value = float(frame_counter) / float(total_infer_time)
            metrics_summary["fps"] = fps_value
        try:
            with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(metrics_summary, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        try:
            with open(os.path.join(output_dir, "metrics_per_frame.json"), "w", encoding="utf-8") as f:
                json.dump({str(k): v for k, v in per_frame_metrics.items()}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        # Save ROI/bbox trace for downstream visualization (e.g., annotate detection visualizations
        # when bbox is derived from previous segmentation).
        try:
            trace: Dict[str, Any] = {}
            for p in work_predictions:
                key = str(int(getattr(p, "frame_index", -1)))
                if key == "-1":
                    continue
                row: Dict[str, Any] = {
                    "bbox": [float(x) for x in getattr(p, "bbox", (0.0, 0.0, 0.0, 0.0))],
                    "bbox_source": str(getattr(p, "bbox_source", "detector")),
                    "is_fallback": bool(getattr(p, "is_fallback", False)),
                }
                seg = getattr(p, "segmentation", None)
                if seg is not None:
                    roi_bbox = getattr(seg, "roi_bbox", None)
                    if roi_bbox is not None:
                        try:
                            row["roi_bbox"] = [float(x) for x in roi_bbox]
                        except Exception:
                            pass
                trace[key] = row
            if trace:
                with open(os.path.join(output_dir, "roi_trace.json"), "w", encoding="utf-8") as f:
                    json.dump(trace, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return metrics_summary, accum

    # ------------------------------------------------------------------
    def _auto_mask_generate_full(self, frame: np.ndarray, bbox: BoundingBox) -> Optional[np.ndarray]:
        if not self.using_auto_mask or not _AUTO_MASK_SUPPORT:
            return None
        try:
            margin = float(self.auto_mask_options.get("margin", 0.2))
        except Exception:
            margin = 0.2
        try:
            roi, bounds, pads = _AUTO_MASK_EXTRACT_ROI(frame, bbox.as_tuple(), expand_ratio=margin)
        except Exception:
            return None
        if roi is None or roi.size == 0:
            return None
        try:
            roi_gray = _AUTO_MASK_ENSURE_GRAY(roi)
        except Exception:
            return None
        try:
            mask = _AUTO_MASK_RUN_GRABCUT(roi, bbox.as_tuple())
        except Exception:
            mask = None
        if mask is None or np.count_nonzero(mask) == 0:
            try:
                mask = _AUTO_MASK_FALLBACK(roi_gray.shape, bbox.as_tuple())
            except Exception:
                mask = None
        if mask is None or mask.size == 0 or not np.any(mask):
            return None
        mask_bool = mask > 0
        canny_low = int(self.auto_mask_options.get("canny_low", 30))
        canny_high = int(self.auto_mask_options.get("canny_high", max(canny_low + 1, 80)))
        edges = cv2.Canny((roi_gray * 255).astype(np.uint8), canny_low, canny_high)
        beta = float(self.auto_mask_options.get("edge_weight_beta", 10.0))
        edge_weight = np.exp(-(edges.astype(np.float32) / 255.0) * beta)
        denom = float(np.ptp(edge_weight)) if edge_weight.size else 0.0
        edge_weight = (edge_weight - edge_weight.min()) / (denom + 1e-6)
        iterations = int(self.auto_mask_options.get("num_iter", 300))
        use_mgac = self._coerce_bool(self.auto_mask_options.get("use_mgac", True), True) and _AUTO_MASK_MGAC is not None
        mask_refined: Optional[np.ndarray] = None
        if use_mgac:
            try:
                mgac_iter_scale = float(self.auto_mask_options.get("mgac_iter_scale", 0.25))
                mgac_min_iter = int(self.auto_mask_options.get("mgac_min_iter", 60))
                mgac_iters = max(mgac_min_iter, int(iterations * mgac_iter_scale))
                mask_refined = _AUTO_MASK_MGAC(  # type: ignore[misc]
                    edge_weight,
                    mgac_iters,
                    init_level_set=mask_bool,
                    smoothing=int(self.auto_mask_options.get("mgac_smoothing", 1)),
                    threshold="auto",
                    balloon=float(self.auto_mask_options.get("mgac_balloon", 1.0)),
                )
                mask = mask_refined.astype(np.uint8) * 255
            except Exception:
                mask_refined = None
        if not use_mgac or mask_refined is None:
            blended = mask_bool.astype(np.float32) * edge_weight
            mask = (blended > 0.5).astype(np.uint8) * 255
        kernel_size = int(self.auto_mask_options.get("postprocess_kernel", 3))
        kernel_size = max(1, kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        open_iter = int(self.auto_mask_options.get("open_iter", 1))
        close_iter = int(self.auto_mask_options.get("close_iter", 1))
        if open_iter > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iter)
        if close_iter > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
        guided = _AUTO_MASK_GUIDED_FILTER(
            roi_gray,
            mask.astype(np.float32) / 255.0,
            radius=int(self.auto_mask_options.get("guided_radius", 4)),
            eps=float(self.auto_mask_options.get("guided_eps", 1e-3)),
        )
        mask = (guided > 0.5).astype(np.uint8) * 255
        dilate_iter = int(self.auto_mask_options.get("dilate_iter", 0))
        if dilate_iter > 0:
            mask = cv2.dilate(mask, kernel, iterations=dilate_iter)
        mask = _AUTO_MASK_STRIP_PADDING(mask, pads)
        mask = _AUTO_MASK_REMOVE_BOUNDARY(mask)
        mask = _AUTO_MASK_LARGEST(mask)
        if mask is None or mask.size == 0 or np.count_nonzero(mask) == 0:
            return None
        height, width = frame.shape[:2]
        full_mask = np.zeros((height, width), dtype=np.uint8)
        x0_clip, y0_clip, x1_clip, y1_clip = bounds
        roi_h = max(0, y1_clip - y0_clip)
        roi_w = max(0, x1_clip - x0_clip)
        if roi_h == 0 or roi_w == 0:
            return None
        trimmed_mask = mask[:roi_h, :roi_w]
        if trimmed_mask.shape[0] != roi_h or trimmed_mask.shape[1] != roi_w:
            trimmed_mask = cv2.resize(trimmed_mask, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
        full_mask[y0_clip:y1_clip, x0_clip:x1_clip] = trimmed_mask
        return full_mask

    # ------------------------------------------------------------------
    @staticmethod
    def _select_evenly_spaced(indices: Sequence[int], limit: int) -> List[int]:
        if not indices or limit <= 0:
            return []
        ordered = sorted(int(v) for v in indices)
        if len(ordered) <= limit:
            return ordered
        if limit == 1:
            return [ordered[0]]
        step = (len(ordered) - 1) / float(limit - 1)
        result: List[int] = []
        for i in range(limit):
            pos = int(round(i * step))
            if pos >= len(ordered):
                pos = len(ordered) - 1
            candidate = ordered[pos]
            if not result or candidate != result[-1]:
                result.append(candidate)
        if result and result[-1] != ordered[-1]:
            result[-1] = ordered[-1]
        return result

    def _compute_segmentation_roi_bbox(
        self,
        prediction: Optional[FramePrediction],
        gt_entry: Optional[FramePrediction],
        frame_shape: Tuple[int, int],
        pad_fraction: float,
    ) -> BoundingBox:
        img_h, img_w = frame_shape
        bbox_tuple: Optional[Tuple[float, float, float, float]] = None
        if gt_entry and gt_entry.segmentation and gt_entry.segmentation.stats:
            bbox_tuple = tuple(gt_entry.segmentation.stats.bbox)
        if bbox_tuple is None and prediction and prediction.segmentation and prediction.segmentation.roi_bbox:
            bbox_tuple = tuple(prediction.segmentation.roi_bbox)
        if bbox_tuple is None and prediction:
            bbox_tuple = tuple(prediction.bbox)
        if bbox_tuple is None:
            bbox_tuple = (0.0, 0.0, float(img_w), float(img_h))
        try:
            roi = expand_bbox(bbox_tuple, pad_fraction, frame_shape)
        except Exception:
            roi = BoundingBox(float(bbox_tuple[0]), float(bbox_tuple[1]), float(bbox_tuple[2]), float(bbox_tuple[3]))
        if roi.w <= 1 or roi.h <= 1:
            return BoundingBox(0.0, 0.0, float(img_w), float(img_h))
        return roi

    @staticmethod
    def _project_bbox_to_roi(
        original_bbox: Tuple[float, float, float, float],
        roi_bbox: BoundingBox,
        roi_shape: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        roi_h, roi_w = roi_shape
        if roi_h <= 1 or roi_w <= 1:
            return None
        x0 = float(original_bbox[0] - roi_bbox.x)
        y0 = float(original_bbox[1] - roi_bbox.y)
        x1 = x0 + float(original_bbox[2])
        y1 = y0 + float(original_bbox[3])
        x0 = max(0.0, min(x0, roi_w - 1.0))
        y0 = max(0.0, min(y0, roi_h - 1.0))
        x1 = max(0.0, min(x1, roi_w - 1.0))
        y1 = max(0.0, min(y1, roi_h - 1.0))
        if x1 - x0 < 1.0 or y1 - y0 < 1.0:
            return None
        return np.array([[x0, y0, x1, y1]], dtype=np.float32)

    @staticmethod
    def _to_binary_mask(mask: Optional[np.ndarray]) -> np.ndarray:
        if mask is None:
            return np.zeros((0, 0), dtype=bool)
        arr = np.asarray(mask)
        if arr.ndim == 3:
            # collapse channel dimension by taking first channel
            arr = arr[..., 0]
        return (arr > 0)

    @staticmethod
    def _paint_mask(canvas: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int]) -> None:
        if mask is None:
            return
        mask_arr = np.asarray(mask)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[..., 0]
        mask_bool = mask_arr.astype(bool)
        if mask_bool.shape[:2] != canvas.shape[:2]:
            # resize mask to canvas size if needed
            mask_bool = cv2.resize(mask_bool.astype(np.uint8), (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        if not np.any(mask_bool):
            return
        ys, xs = np.nonzero(mask_bool)
        if ys.size == 0:
            return
        canvas[ys, xs] = color

    def _render_segmentation_visualizations(
        self,
        video_path: str,
        predictions: List[FramePrediction],
        mask_files: Dict[int, str],
        output_dir: str,
        gt_by_frame: Dict[int, FramePrediction],
        viz_settings: Dict[str, Any],
    ) -> None:
        if not bool(viz_settings.get("include_segmentation", True)):
            return
        frame_indices = sorted(int(idx) for idx in mask_files.keys())
        if not frame_indices:
            return
        try:
            sample_count = int(viz_settings.get("samples", 10) or 10)
        except Exception:
            sample_count = 10
        sample_count = max(1, sample_count)
        ensure_first_last = bool(viz_settings.get("ensure_first_last", False))
        if ensure_first_last and len(frame_indices) >= 2:
            sample_count = max(sample_count, 2)
        limit = min(sample_count, len(frame_indices))
        selected = self._select_evenly_spaced(frame_indices, limit)
        if ensure_first_last and len(frame_indices) >= 2:
            mandatory = {frame_indices[0], frame_indices[-1]}
            selected = sorted({*selected, *mandatory})
        pred_map = {int(p.frame_index): p for p in predictions}
        try:
            roi_expand = float(viz_settings.get("roi_expand", 0.2))
        except Exception:
            roi_expand = 0.2
        roi_expand = max(0.0, roi_expand)
        try:
            scale_factor = int(viz_settings.get("scale", 4) or 4)
        except Exception:
            scale_factor = 4
        scale_factor = max(1, min(scale_factor, 8))
        vis_dir = os.path.join(output_dir, "visualizations_roi")
        ensure_dir(vis_dir)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return
        color_gt = (0, 255, 0)
        color_pred = (0, 0, 255)
        color_overlap = (0, 255, 255)
        color_fp = (255, 179, 0)
        color_fn = (0, 153, 255)
        text_color = (255, 255, 255)
        for frame_idx in selected:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frame = self._apply_preprocs_frame(frame, self.preprocs)
            frame_h, frame_w = frame.shape[:2]
            pred_entry = pred_map.get(frame_idx)
            gt_entry = gt_by_frame.get(frame_idx)
            roi_bbox = self._compute_segmentation_roi_bbox(pred_entry, gt_entry, (frame_h, frame_w), roi_expand)
            roi_frame = crop_with_bbox(frame, roi_bbox)
            if roi_frame.size == 0:
                continue
            if self.roi_preprocs:
                roi_frame = self._apply_preprocs_frame(roi_frame, self.roi_preprocs)
            mask_path = mask_files.get(frame_idx)
            if not mask_path or not os.path.exists(mask_path):
                continue
            # Guard against zero-byte files: Ultralytics patches cv2.imread to use
            # np.fromfile + imdecode; empty files produce an assertion failure.
            if os.path.getsize(mask_path) == 0:
                continue
            try:
                pred_mask_full = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            except Exception:
                continue
            if pred_mask_full is None:
                continue
            roi_pred_mask = crop_with_bbox(pred_mask_full, roi_bbox)
            if roi_pred_mask.size == 0:
                continue
            has_gt_mask = False
            roi_gt_mask = None
            if gt_entry and gt_entry.segmentation:
                gt_mask_full = self._load_mask_from_annotation(video_path, gt_entry.segmentation)
                if gt_mask_full is not None:
                    roi_gt_mask = crop_with_bbox(gt_mask_full, roi_bbox)
                    if roi_gt_mask is not None and roi_gt_mask.size > 0:
                        has_gt_mask = True
            pred_binary = self._to_binary_mask(roi_pred_mask)
            if pred_binary.size == 0:
                continue
            dice_val = None
            iou_val = None
            if has_gt_mask and roi_gt_mask is not None:
                gt_binary = self._to_binary_mask(roi_gt_mask)
                if gt_binary.shape != pred_binary.shape:
                    gt_binary = cv2.resize(
                        gt_binary.astype(np.uint8),
                        (pred_binary.shape[1], pred_binary.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ) > 0
                dice_val = dice_coefficient(roi_pred_mask, roi_gt_mask)
                iou_val = intersection_over_union(roi_pred_mask, roi_gt_mask)
            else:
                gt_binary = np.zeros_like(pred_binary, dtype=bool)
            intersection = pred_binary & gt_binary
            pred_only = pred_binary & (~gt_binary)
            gt_only = gt_binary & (~pred_binary)
            base_gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            overlay_map = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)
            error_map = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)
            if np.any(gt_only):
                self._paint_mask(overlay_map, gt_only, color_gt)
            if np.any(pred_only):
                self._paint_mask(overlay_map, pred_only, color_pred)
            if np.any(intersection):
                self._paint_mask(overlay_map, intersection, color_overlap)
            if not has_gt_mask or roi_gt_mask is None:
                if np.any(pred_binary):
                    self._paint_mask(overlay_map, pred_binary, color_pred)
            fp_mask = pred_only
            fn_mask = gt_only
            if np.any(fp_mask):
                self._paint_mask(error_map, fp_mask, color_fp)
            if np.any(fn_mask):
                self._paint_mask(error_map, fn_mask, color_fn)
            overlay_up = cv2.resize(
                overlay_map,
                None,
                fx=scale_factor,
                fy=scale_factor,
                interpolation=cv2.INTER_NEAREST,
            )
            error_up = cv2.resize(
                error_map,
                None,
                fx=scale_factor,
                fy=scale_factor,
                interpolation=cv2.INTER_NEAREST,
            )
            header_lines = [f"frame {frame_idx}"]
            if dice_val is not None and iou_val is not None:
                header_lines.append(f"Dice={dice_val:.3f}")
                header_lines.append(f"IoU={iou_val:.3f}")
            elif pred_entry and getattr(pred_entry.segmentation, "centroid_error_px", None) is not None:
                header_lines.append(f"Centroid Δ={pred_entry.segmentation.centroid_error_px:.2f}px")
            for line_idx, text in enumerate(header_lines):
                y = 28 + line_idx * 24
                cv2.putText(overlay_up, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(overlay_up, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv2.LINE_AA)
            legend_entries = [
                (color_gt, "GT"),
                (color_pred, "Pred"),
                (color_overlap, "Overlap"),
            ] if has_gt_mask else [
                (color_pred, "Prediction")
            ]
            legend_y = overlay_up.shape[0] - 24
            for idx, (color, label) in enumerate(legend_entries):
                x0 = 12 + idx * 160
                cv2.rectangle(overlay_up, (x0, legend_y - 18), (x0 + 36, legend_y + 2), color, -1)
                cv2.rectangle(overlay_up, (x0, legend_y - 18), (x0 + 36, legend_y + 2), (0, 0, 0), 1)
                cv2.putText(overlay_up, label, (x0 + 44, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(overlay_up, label, (x0 + 44, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
            fp_count = int(np.count_nonzero(fp_mask))
            fn_count = int(np.count_nonzero(fn_mask))
            err_lines = [f"FP px={fp_count}", f"FN px={fn_count}"]
            for line_idx, text in enumerate(err_lines):
                y = 28 + line_idx * 24
                cv2.putText(error_up, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(error_up, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv2.LINE_AA)
            err_legend_entries = [
                (color_fp, "FP (Pred only)"),
                (color_fn, "FN (GT only)"),
            ]
            err_legend_y = error_up.shape[0] - 24
            for idx, (color, label) in enumerate(err_legend_entries):
                x0 = 12 + idx * 200
                cv2.rectangle(error_up, (x0, err_legend_y - 18), (x0 + 36, err_legend_y + 2), color, -1)
                cv2.rectangle(error_up, (x0, err_legend_y - 18), (x0 + 36, err_legend_y + 2), (0, 0, 0), 1)
                cv2.putText(error_up, label, (x0 + 44, err_legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(error_up, label, (x0 + 44, err_legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
            overlay_path = os.path.join(vis_dir, f"frame_{frame_idx:06d}_overlay.png")
            error_path = os.path.join(vis_dir, f"frame_{frame_idx:06d}_error.png")
            try:
                cv2.imwrite(overlay_path, overlay_up)
                cv2.imwrite(error_path, error_up)
            except Exception:
                continue
        cap.release()

    # ------------------------------------------------------------------
    def _load_mask_from_annotation(self, video_path: str, segmentation: Optional[SegmentationData]) -> Optional[np.ndarray]:
        if segmentation is None or not segmentation.mask_path:
            return None
        mask_path = segmentation.mask_path
        if os.path.isabs(mask_path):
            abs_path = mask_path
            if not os.path.exists(abs_path):
                return None
        else:
            video_dir = os.path.dirname(os.path.abspath(video_path)) if video_path else None
            from .dataset import _resolve_gt_mask_path  # avoid circular at module level
            resolved = _resolve_gt_mask_path(mask_path, self.dataset_root, video_dir)
            if resolved is None:
                return None
            abs_path = resolved
        # Guard against zero-byte files: Ultralytics patches cv2.imread to use
        # np.fromfile + imdecode; empty files produce an assertion failure.
        if os.path.getsize(abs_path) == 0:
            return None
        try:
            mask = cv2.imread(abs_path, cv2.IMREAD_GRAYSCALE)
        except Exception:
            return None
        if mask is None:
            return None
        mask_bin = (mask > 0).astype(np.uint8)
        if mask_bin.size == 0 or np.count_nonzero(mask_bin) == 0:
            return None
        mask_bin = mask_bin * 255
        mask_filled = fill_holes(mask_bin)
        mask_clean = keep_largest_component(mask_filled)
        return mask_clean

    def _resolve_checkpoint_path(self, path: Optional[str]) -> Optional[str]:
        if path in (None, ""):
            return None
        candidate = str(path).strip()
        if not candidate:
            return None
        if os.path.isabs(candidate):
            return candidate
        normalized = candidate.replace("\\", "/")
        while normalized.startswith("./"):
            normalized = normalized[2:]
        normalized = os.path.normpath(normalized)
        search_paths: List[str] = []

        def _add_candidate(base: Optional[str], rel: str) -> None:
            if not base:
                return
            abs_path = os.path.normpath(os.path.join(base, rel))
            if abs_path not in search_paths:
                search_paths.append(abs_path)

        _add_candidate(getattr(self, "project_root", None), normalized)
        _add_candidate(self.dataset_root, normalized)
        _add_candidate(os.getcwd(), normalized)
        if not search_paths:
            search_paths.append(os.path.abspath(normalized))
        for cand in search_paths:
            if os.path.exists(cand):
                return cand
        return search_paths[0] if search_paths else None

    def _maybe_use_default_pretrained_model(self) -> bool:
        fallback_key = "torchvision_fcn_resnet50"
        if self.model_name == fallback_key:
            self.using_default_pretrained = True
            self.train_enabled = False
            self.logger("[Segmentation] Using torchvision FCN ResNet50 pretrained weights for inference.")
            return True
        fallback_cls = SEGMENTATION_MODEL_REGISTRY.get(fallback_key)
        if fallback_cls is None:
            self.logger("[Segmentation] Default pretrained segmentation model is unavailable.")
            return False
        try:
            fallback = fallback_cls({}).to(self.device)
            fallback.eval()
        except Exception as exc:
            self.logger(f"[Segmentation] Failed to initialize default pretrained model: {exc}")
            return False
        self.model = fallback
        self.model_name = fallback_key
        self.train_enabled = False
        self.using_default_pretrained = True
        self.logger("[Segmentation] Using torchvision FCN ResNet50 pretrained weights for inference.")
        return True

    # ------------------------------------------------------------------
    def _run_model_with_fallback(self, roi_tensor: torch.Tensor) -> torch.Tensor:
        """Execute model inference, falling back to CPU if CUDA errors occur."""
        if self.model is None:
            raise RuntimeError("Segmentation model is not initialised")
        try:
            with torch.no_grad():
                return self.model(roi_tensor)
        except (_TorchCudaError, RuntimeError) as exc:
            if self._maybe_fallback_to_cpu(exc):
                roi_tensor_cpu = roi_tensor.to(self.device)
                with torch.no_grad():
                    return self.model(roi_tensor_cpu)
            raise

    def _maybe_fallback_to_cpu(self, exc: Exception) -> bool:
        """Switch to CPU execution when GPU becomes unstable."""
        if self.device.type != "cuda":
            return False
        message = str(exc).lower()
        keywords = (
            "cuda error",
            "cublas",
            "cudnn",
            "device-side assert",
            "illegal memory access",
        )
        if not any(keyword in message for keyword in keywords):
            return False
        if not self._cuda_fallback_triggered:
            self.logger(
                "[Segmentation] CUDA inference failure detected ({}). Falling back to CPU for this run.".format(
                    message
                )
            )
            self._cuda_fallback_triggered = True
        try:  # pragma: no cover - only executes when CUDA is available
            torch.cuda.synchronize()
        except Exception:
            pass
        try:  # pragma: no cover - only executes when CUDA is available
            torch.cuda.empty_cache()
        except Exception:
            pass
        self.device = torch.device("cpu")
        if self.model is not None:
            self.model.to(self.device)
        return True


__all__ = ["SegmentationWorkflow", "AUTO_MASK_RUNTIME_AVAILABLE"]
