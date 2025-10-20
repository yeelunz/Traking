from __future__ import annotations

import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

import random

import numpy as np

try:
    import cv2  # type: ignore
except Exception as ex:  # pragma: no cover - keep runtime error informative
    cv2 = None  # type: ignore
    _CV2_IMPORT_ERROR = ex

try:
    import torch
except Exception as ex:  # pragma: no cover
    torch = None  # type: ignore
    _TORCH_IMPORT_ERROR = ex

from ..core.interfaces import FramePrediction, PreprocessingModule, TrackingModel
from ..core.registry import register_model
from ..utils.init_bbox import resolve_first_frame_bbox


_MIXFORMER_ROOT = Path(__file__).resolve().parents[2] / "libs" / "MixFormerV2"
_MIXFORMER_BOOTSTRAPPED = False


@dataclass
class _MixFormerResult:
    bbox: Optional[Tuple[float, float, float, float]]
    score: Optional[float]


def _ensure_mixformer_bootstrap() -> Path:
    """Ensure MixFormerV2 repo is discoverable on sys.path and return its root."""
    global _MIXFORMER_BOOTSTRAPPED
    root = _MIXFORMER_ROOT
    if not root.exists():
        raise RuntimeError(
            "MixFormerV2 repository not found under 'libs/MixFormerV2'. "
            "Please initialise the submodule or clone the upstream repo."
        )

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    tracking_dir = root / "tracking"
    tracking_str = str(tracking_dir)
    if tracking_dir.is_dir() and tracking_str not in sys.path:
        sys.path.insert(0, tracking_str)

    if not _MIXFORMER_BOOTSTRAPPED:
        init_file = tracking_dir / "_init_paths.py"
        if init_file.exists():
            spec = importlib.util.spec_from_file_location("_mixformer_init_paths", init_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                try:  # pragma: no cover - harmless if it fails, path already added
                    spec.loader.exec_module(module)  # type: ignore[arg-type]
                except Exception:
                    pass
        _MIXFORMER_BOOTSTRAPPED = True
    return root


def _ensure_admin_local(root: Path) -> None:
    module_name = "lib.train.admin.local"
    if module_name in sys.modules:
        return

    try:
        importlib.import_module(module_name)
        return
    except Exception:
        pass

    module = ModuleType(module_name)

    class EnvironmentSettings:  # pragma: no cover - simple configuration holder
        def __init__(self):
            workspace = root / "tracking_workspace"
            self.workspace_dir = str(workspace)
            self.tensorboard_dir = str(workspace / "tensorboard")
            self.pretrained_networks = str(root / "models")
            self.lasot_dir = ""
            self.got10k_dir = ""
            self.trackingnet_dir = ""
            self.coco_dir = ""
            self.lvis_dir = ""
            self.sbd_dir = ""
            self.imagenet_dir = ""
            self.imagenetdet_dir = ""
            self.ecssd_dir = ""
            self.hkuis_dir = ""
            self.msra10k_dir = ""
            self.davis_dir = ""
            self.youtubevos_dir = ""

    module.EnvironmentSettings = EnvironmentSettings  # type: ignore[attr-defined]
    sys.modules[module_name] = module


def _ensure_local_env(root: Path) -> None:
    """Provide lib.test.evaluation.local without writing to disk."""
    module_name = "lib.test.evaluation.local"
    if module_name in sys.modules:
        return

    try:
        from lib.test.evaluation.environment import EnvSettings  # type: ignore
    except Exception as ex:
        raise RuntimeError(f"Failed to import MixFormerV2 environment helpers: {ex}") from ex

    results_dir = root / "tracking_results"
    seg_dir = root / "segmentation_results"
    nets_dir = root / "lib" / "test" / "networks"
    plots_dir = root / "result_plots"
    models_dir = root / "models"

    for path in (results_dir, seg_dir, nets_dir, plots_dir, models_dir):
        path.mkdir(parents=True, exist_ok=True)

    def local_env_settings() -> Any:
        settings = EnvSettings()
        settings.prj_dir = str(root)
        settings.save_dir = str(root)
        settings.results_path = str(results_dir)
        settings.segmentation_path = str(seg_dir)
        settings.network_path = str(nets_dir)
        settings.result_plot_path = str(plots_dir)
        return settings

    module = ModuleType(module_name)
    module.local_env_settings = local_env_settings  # type: ignore[attr-defined]
    sys.modules[module_name] = module


def _allow_mixformer_globals() -> None:
    """Allow MixFormer checkpoints saved with custom pickled classes."""
    if torch is None:
        return

    serialization = getattr(torch, "serialization", None)
    if serialization is None:
        return

    add_safe = getattr(serialization, "add_safe_globals", None)
    if not callable(add_safe):  # PyTorch < 2.6
        return

    safe_types: List[type] = []

    try:
        from lib.train.admin.stats import AverageMeter, StatValue  # type: ignore

        safe_types.extend([AverageMeter, StatValue])
    except Exception:
        pass

    try:
        from lib.train.admin.settings import Settings  # type: ignore

        safe_types.append(Settings)
    except Exception:
        pass

    env_cls: Optional[type] = None
    try:
        local_module = importlib.import_module("lib.train.admin.local")
        candidate = getattr(local_module, "EnvironmentSettings", None)
        if isinstance(candidate, type):
            env_cls = candidate
    except Exception:
        env_cls = None

    if env_cls is None:
        class _FallbackEnvSettings:  # pragma: no cover - simple placeholder
            pass

        env_cls = _FallbackEnvSettings

    safe_types.append(env_cls)

    if not safe_types:
        return

    try:
        add_safe(safe_types)
    except Exception:
        pass


def _resolve_tracker(tracker_name: str) -> Tuple[Callable[..., Any], Callable[..., Any]]:
    registry = {
        "mixformer2_vit": (
            "lib.test.parameter.mixformer2_vit",
            "lib.test.tracker.mixformer2_vit",
        ),
        "mixformer2_vit_online": (
            "lib.test.parameter.mixformer2_vit_online",
            "lib.test.tracker.mixformer2_vit_online",
        ),
    }
    if tracker_name not in registry:
        raise ValueError(
            f"Unsupported MixFormerV2 tracker '{tracker_name}'. "
            f"Available keys: {sorted(registry)}"
        )

    param_mod_name, tracker_mod_name = registry[tracker_name]
    param_module = importlib.import_module(param_mod_name)
    tracker_module = importlib.import_module(tracker_mod_name)

    parameters_fn = getattr(param_module, "parameters", None)
    tracker_cls_factory = getattr(tracker_module, "get_tracker_class", None)
    if not callable(parameters_fn) or not callable(tracker_cls_factory):
        raise RuntimeError(
            f"Tracker module '{tracker_name}' is missing required factories."
        )
    return parameters_fn, tracker_cls_factory


def _is_valid_bbox(bbox: Optional[Tuple[float, float, float, float]]) -> bool:
    if bbox is None:
        return False
    x, y, w, h = bbox
    return all(np.isfinite([x, y, w, h])) and w > 0 and h > 0


@register_model("MixFormerV2")
class MixFormerV2Tracker(TrackingModel):
    """Wrapper around the official MixFormerV2 tracker for the pipeline orchestrator."""

    name = "MixFormerV2"

    DEFAULT_CONFIG: Dict[str, Any] = {
        "tracker": "mixformer2_vit_online",
        "parameter": "288_depth8_score",
        "checkpoint": "models/mixformerv2_base.pth.tar",
        "search_area_scale": None,
        "online_size": 1,
        "update_interval": None,
        "min_confidence": 0.0,
        "fallback_last_prediction": True,
        "dataset_name": "custom",
        "train_enabled": False,
        "train_epochs": 1,
        "train_lr": None,
        "train_weight_decay": None,
        "train_freeze_backbone": True,
        "train_max_samples": 0,
        "train_log_interval": 20,
        "train_template_update": "keep",
        "first_frame_source": "gt",
        "first_frame_fallback": "gt",
        "init_detector_weights": "best.pt",
        "init_detector_conf": 0.25,
        "init_detector_iou": 0.5,
        "init_detector_imgsz": 640,
        "init_detector_device": "auto",
        "init_detector_classes": None,
        "init_detector_max_det": 50,
    }

    def __init__(self, config: Dict[str, Any]):
        if cv2 is None:
            detail = (
                f" (import error: {_CV2_IMPORT_ERROR!r})"
                if "_CV2_IMPORT_ERROR" in globals() else ""
            )
            raise RuntimeError("OpenCV is required for MixFormerV2 tracker." + detail)
        if torch is None:
            detail = (
                f" (import error: {_TORCH_IMPORT_ERROR!r})"
                if "_TORCH_IMPORT_ERROR" in globals() else ""
            )
            raise RuntimeError("PyTorch is required for MixFormerV2 tracker." + detail)
        if not torch.cuda.is_available():  # pragma: no cover - depends on runtime
            raise RuntimeError(
                "MixFormerV2 currently requires a CUDA-capable GPU because the official "
                "implementation executes preprocessing and models on CUDA tensors."
            )

        root = _ensure_mixformer_bootstrap()
        _ensure_admin_local(root)
        _allow_mixformer_globals()
        _ensure_local_env(root)

        merged = {**self.DEFAULT_CONFIG, **(config or {})}
        self.tracker_key = str(merged.get("tracker", self.DEFAULT_CONFIG["tracker"]))
        self.parameter_name = str(merged.get("parameter", self.DEFAULT_CONFIG["parameter"]))
        self.search_area_scale = merged.get("search_area_scale", None)
        online_size_cfg = merged.get("online_size", self.DEFAULT_CONFIG["online_size"])
        self.online_size = online_size_cfg if online_size_cfg is not None else self.DEFAULT_CONFIG["online_size"]
        update_interval_cfg = merged.get("update_interval", self.DEFAULT_CONFIG["update_interval"])
        self.update_interval = (
            update_interval_cfg if update_interval_cfg is not None else self.DEFAULT_CONFIG["update_interval"]
        )
        self.min_confidence = float(merged.get("min_confidence", 0.0))
        self.fallback_last_prediction = bool(merged.get("fallback_last_prediction", True))
        self.dataset_name = str(merged.get("dataset_name", "custom"))
        self.train_enabled = bool(merged.get("train_enabled", self.DEFAULT_CONFIG["train_enabled"]))
        self.train_epochs = max(1, int(merged.get("train_epochs", self.DEFAULT_CONFIG["train_epochs"])))
        self.train_lr = merged.get("train_lr", self.DEFAULT_CONFIG["train_lr"])
        self.train_weight_decay = merged.get("train_weight_decay", self.DEFAULT_CONFIG["train_weight_decay"])
        self.train_freeze_backbone = bool(merged.get("train_freeze_backbone", self.DEFAULT_CONFIG["train_freeze_backbone"]))
        self.train_max_samples = max(0, int(merged.get("train_max_samples", self.DEFAULT_CONFIG["train_max_samples"])) )
        self.train_log_interval = max(1, int(merged.get("train_log_interval", self.DEFAULT_CONFIG["train_log_interval"])) )
        self.train_template_update = str(
            merged.get("train_template_update", self.DEFAULT_CONFIG["train_template_update"])
        ).lower()

        self.first_frame_source = str(merged.get("first_frame_source", self.DEFAULT_CONFIG["first_frame_source"]) or "gt").lower()
        raw_fallback = merged.get("first_frame_fallback", self.DEFAULT_CONFIG["first_frame_fallback"])
        if raw_fallback is None:
            self.first_frame_fallback: Optional[str] = None
        else:
            fb = str(raw_fallback).strip().lower()
            self.first_frame_fallback = fb if fb not in ("", "none", "null") else None
        self.init_detector_params = {
            "weights": str(merged.get("init_detector_weights", self.DEFAULT_CONFIG["init_detector_weights"])),
            "conf": float(merged.get("init_detector_conf", self.DEFAULT_CONFIG["init_detector_conf"])),
            "iou": float(merged.get("init_detector_iou", self.DEFAULT_CONFIG["init_detector_iou"])),
            "imgsz": int(merged.get("init_detector_imgsz", self.DEFAULT_CONFIG["init_detector_imgsz"])),
            "device": str(merged.get("init_detector_device", self.DEFAULT_CONFIG["init_detector_device"])),
            "classes": merged.get("init_detector_classes", self.DEFAULT_CONFIG["init_detector_classes"]),
            "max_det": int(merged.get("init_detector_max_det", self.DEFAULT_CONFIG["init_detector_max_det"])),
        }

        checkpoint_cfg = merged.get("checkpoint", self.DEFAULT_CONFIG["checkpoint"])
        if isinstance(checkpoint_cfg, (list, tuple)):
            raise TypeError("checkpoint must be a string path, not a list/tuple")
        checkpoint_str = str(checkpoint_cfg)
        # Accept absolute path or relative-to-root path
        ckpt_path = Path(checkpoint_str)
        if not ckpt_path.is_absolute():
            candidate = root / checkpoint_str
            if candidate.exists():
                ckpt_path = candidate
        if not ckpt_path.exists():
            raise RuntimeError(
                "MixFormerV2 checkpoint not found. Expected at '{}'. "
                "Place the pretrained .pth.tar under libs/MixFormerV2/models or provide an absolute path.".format(ckpt_path)
            )
        self._checkpoint_arg = str(ckpt_path)

        self._parameters_fn, tracker_factory = _resolve_tracker(self.tracker_key)
        self._tracker_cls = tracker_factory()

        self.preprocs: List[PreprocessingModule] = []
        self._finetuned_state_dict: Optional[Dict[str, Any]] = None
        self._finetuned_summary: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # TrackingModel API
    # ------------------------------------------------------------------
    def train(self, train_dataset, val_dataset=None, seed: int = 0, output_dir: Optional[str] = None):
        if not self.train_enabled:
            return {"status": "skipped", "reason": "train_disabled"}
        if cv2 is None:
            detail = (
                f" (import error: {_CV2_IMPORT_ERROR!r})"
                if "_CV2_IMPORT_ERROR" in globals() else ""
            )
            raise RuntimeError("OpenCV is required for MixFormerV2 training." + detail)
        if torch is None:
            detail = (
                f" (import error: {_TORCH_IMPORT_ERROR!r})"
                if "_TORCH_IMPORT_ERROR" in globals() else ""
            )
            raise RuntimeError("PyTorch is required for MixFormerV2 training." + detail)

        dataset_len = len(train_dataset) if hasattr(train_dataset, "__len__") else None
        if not dataset_len:
            return {"status": "no_data", "reason": "empty_dataset"}

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        params_kwargs: Dict[str, Any] = {
            "yaml_name": self.parameter_name,
            "model": self._checkpoint_arg,
        }
        if self.search_area_scale is not None:
            params_kwargs["search_area_scale"] = self.search_area_scale
        if self.online_size is not None:
            params_kwargs["online_size"] = self.online_size
        if self.update_interval is not None:
            params_kwargs["update_interval"] = self.update_interval

        params = self._parameters_fn(**params_kwargs)
        cfg = params.cfg

        mixformer_module = importlib.import_module("lib.models.mixformer2_vit")
        build_mixformer2_vit_online = getattr(mixformer_module, "build_mixformer2_vit_online")
        tracker_utils_module = importlib.import_module("lib.test.tracker.tracker_utils")
        Preprocessor_wo_mask = getattr(tracker_utils_module, "Preprocessor_wo_mask")
        processing_utils_module = importlib.import_module("lib.train.data.processing_utils")
        sample_target = getattr(processing_utils_module, "sample_target")
        import torch.nn.functional as F
        from torch.nn.utils import clip_grad_norm_

        device = torch.device("cuda")
        settings = SimpleNamespace(static_model=self._checkpoint_arg)
        net = build_mixformer2_vit_online(cfg, settings=settings, train=True)
        checkpoint_obj = torch.load(self._checkpoint_arg, map_location="cpu")
        ckpt_state = checkpoint_obj.get("net") if isinstance(checkpoint_obj, dict) else checkpoint_obj
        net.load_state_dict(ckpt_state, strict=False)
        net = net.to(device)
        net.train()

        if self.train_freeze_backbone:
            for name, param in net.named_parameters():
                if name.startswith("backbone"):
                    param.requires_grad = False

        trainable_params = [p for p in net.parameters() if p.requires_grad]
        if not trainable_params:
            trainable_params = list(net.parameters())

        base_lr = float(getattr(cfg.TRAIN, "LR", 1e-4)) if hasattr(cfg, "TRAIN") else 1e-4
        lr = float(self.train_lr) if self.train_lr is not None else base_lr * 0.1
        weight_decay = (
            float(self.train_weight_decay)
            if self.train_weight_decay is not None
            else float(getattr(cfg.TRAIN, "WEIGHT_DECAY", 1e-4))
        )
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        grad_clip = float(getattr(cfg.TRAIN, "GRAD_CLIP_NORM", 0.0)) if hasattr(cfg, "TRAIN") else 0.0

        logger = getattr(self, "logger", None)

        def _log(msg):
            if logger is not None and hasattr(logger, "info"):
                try:
                    logger.info(msg)
                    return
                except Exception:
                    pass
            print(msg)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            log_file = os.path.join(output_dir, "train_log.txt")
        else:
            log_file = None

        def _write_log(line):
            if not log_file:
                return
            try:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                pass

        preproc = Preprocessor_wo_mask()
        search_size = float(params.search_size)

        def map_box_back(pred_box, prev_bbox, resize_factor):
            half_side = 0.5 * search_size / resize_factor
            cx_prev = prev_bbox[0] + 0.5 * prev_bbox[2]
            cy_prev = prev_bbox[1] + 0.5 * prev_bbox[3]
            cx_real = pred_box[0] + (cx_prev - half_side)
            cy_real = pred_box[1] + (cy_prev - half_side)
            return torch.stack([
                cx_real - 0.5 * pred_box[2],
                cy_real - 0.5 * pred_box[3],
                pred_box[2],
                pred_box[3],
            ])

        def bbox_iou_tensor(a, b):
            x1 = torch.max(a[0], b[0])
            y1 = torch.max(a[1], b[1])
            x2 = torch.min(a[0] + a[2], b[0] + b[2])
            y2 = torch.min(a[1] + a[3], b[1] + b[3])
            inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
            area_a = torch.clamp(a[2], min=0) * torch.clamp(a[3], min=0)
            area_b = torch.clamp(b[2], min=0) * torch.clamp(b[3], min=0)
            union = area_a + area_b - inter
            return inter / (union + 1e-6)

        template_strategy = self.train_template_update.lower()
        max_samples = self.train_max_samples
        log_interval = max(1, self.train_log_interval)
        epochs = max(1, self.train_epochs)

        progress_cb = getattr(self, "progress_callback", None)

        def iter_video_samples(video_path, annotation):
            frames = (annotation or {}).get("frames", {})
            if not frames:
                return
            idx_to_bbox = {}
            for key, boxes in frames.items():
                if not boxes:
                    continue
                try:
                    frame_idx = int(key)
                except Exception:
                    continue
                bbox = boxes[0]
                if bbox and len(bbox) >= 4 and bbox[2] > 0 and bbox[3] > 0:
                    idx_to_bbox[frame_idx] = (
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[2]),
                        float(bbox[3]),
                    )
            if len(idx_to_bbox) < 2:
                return
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                _log(f"[Train] Skipped video (unreadable): {video_path}")
                return
            try:
                template_frame = None
                template_bbox = None
                prev_bbox = None
                target_indices = set(idx_to_bbox.keys())
                frame_idx = 0
                while True:
                    ok, frame_bgr = cap.read()
                    if not ok:
                        break
                    if frame_idx in target_indices:
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        gt_bbox = idx_to_bbox[frame_idx]
                        if template_frame is None:
                            template_frame = frame_rgb
                            template_bbox = gt_bbox
                            prev_bbox = gt_bbox
                        else:
                            yield template_frame, template_bbox, prev_bbox, frame_rgb, gt_bbox
                            prev_bbox = gt_bbox
                            if template_strategy == "update":
                                template_frame = frame_rgb
                                template_bbox = gt_bbox
                    frame_idx += 1
            finally:
                cap.release()

        total_steps = 0
        total_bbox_loss = 0.0
        total_score_loss = 0.0
        total_iou = 0.0

        for epoch in range(epochs):
            if callable(progress_cb):
                try:
                    progress_cb("train_epoch_start", epoch + 1, epochs)
                except Exception:
                    pass
            _log(f"[Train] Epoch {epoch+1}/{epochs}")
            if log_file:
                _write_log(f"# Epoch {epoch+1}/{epochs}")

            for item_idx in range(dataset_len):
                sample_item = train_dataset[item_idx]
                video_path = sample_item.get("video_path") if isinstance(sample_item, dict) else sample_item.get("video_path")
                annotation = sample_item.get("annotation") if isinstance(sample_item, dict) else sample_item.get("annotation")
                if not video_path or annotation is None:
                    continue
                for template_frame, template_bbox, prev_bbox, frame_rgb, gt_bbox in iter_video_samples(video_path, annotation):
                    template_patch, _, _ = sample_target(
                        template_frame,
                        list(template_bbox),
                        params.template_factor,
                        output_sz=params.template_size,
                    )
                    template_tensor = preproc.process(template_patch)

                    search_patch, resize_factor, _ = sample_target(
                        frame_rgb,
                        list(prev_bbox),
                        params.search_factor,
                        output_sz=params.search_size,
                    )
                    search_tensor = preproc.process(search_patch)

                    optimizer.zero_grad(set_to_none=True)
                    out = net(template_tensor, template_tensor, search_tensor, softmax=True, run_score_head=True)
                    pred_boxes = out.get("pred_boxes")
                    if pred_boxes is None:
                        continue
                    pred_vec = pred_boxes.view(-1, 4).mean(dim=0)
                    pred_vec = pred_vec * (search_size / float(resize_factor))
                    prev_bbox_tensor = torch.tensor(prev_bbox, dtype=torch.float32, device=device)
                    gt_tensor = torch.tensor(gt_bbox, dtype=torch.float32, device=device)
                    pred_abs = map_box_back(pred_vec, prev_bbox_tensor, float(resize_factor))

                    bbox_loss = F.smooth_l1_loss(pred_abs, gt_tensor)
                    score_logits = out.get("pred_scores")
                    if score_logits is not None:
                        score_loss = F.binary_cross_entropy_with_logits(
                            score_logits.view(-1),
                            torch.ones_like(score_logits.view(-1), device=device),
                        )
                    else:
                        score_loss = torch.tensor(0.0, device=device)

                    loss = bbox_loss + 0.3 * score_loss
                    loss.backward()
                    if grad_clip > 0:
                        clip_grad_norm_(trainable_params, grad_clip)
                    optimizer.step()

                    with torch.no_grad():
                        iou_val = bbox_iou_tensor(pred_abs, gt_tensor).item()

                    total_steps += 1
                    total_bbox_loss += float(bbox_loss.detach().item())
                    total_score_loss += float(score_loss.detach().item())
                    total_iou += float(iou_val)

                    if total_steps % log_interval == 0:
                        msg = (
                            f"[Train] step={total_steps} loss={loss.detach().item():.4f} "
                            f"bbox={bbox_loss.detach().item():.4f} score={score_loss.detach().item():.4f} "
                            f"IoU={iou_val:.4f}"
                        )
                        _log(msg)
                        _write_log(msg)

                    if max_samples and total_steps >= max_samples:
                        break
                if max_samples and total_steps >= max_samples:
                    break
            if callable(progress_cb):
                try:
                    progress_cb("train_epoch_end", epoch + 1, epochs)
                except Exception:
                    pass
            if max_samples and total_steps >= max_samples:
                _log(f"[Train] Reached max_samples={max_samples}, stopping early.")
                break

        if total_steps == 0:
            return {"status": "no_data", "reason": "no_valid_samples"}

        avg_bbox_loss = total_bbox_loss / total_steps
        avg_score_loss = total_score_loss / total_steps
        mean_iou = total_iou / total_steps

        finetuned_state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        self._finetuned_state_dict = finetuned_state
        summary = {
            "status": "trained",
            "epochs_completed": epoch + 1,
            "steps": total_steps,
            "avg_bbox_loss": avg_bbox_loss,
            "avg_score_loss": avg_score_loss,
            "mean_iou": mean_iou,
            "learning_rate": lr,
            "trainable_params": len(trainable_params),
        }
        self._finetuned_summary = summary
        if log_file:
            _write_log(f"summary={json.dumps(summary, ensure_ascii=False)}")

        return summary

    def should_train(self, *_args, **_kwargs) -> bool:
        return self.train_enabled

    def load_checkpoint(self, ckpt_path: str):
        if not ckpt_path:
            raise ValueError("ckpt_path must be a non-empty string")
        self._checkpoint_arg = ckpt_path

    # ------------------------------------------------------------------
    def _apply_preprocs(self, frame_bgr: np.ndarray) -> np.ndarray:
        if not self.preprocs:
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        for proc in self.preprocs:
            rgb = proc.apply_to_frame(rgb)
        return rgb

    def _build_tracker(self) -> Any:
        def _emit_weights_warning() -> None:
            if getattr(self, "_weights_only_warned", False):
                return
            self._weights_only_warned = True
            warn_msg = (
                "MixFormerV2: forcing torch.load(weights_only=False) for official checkpoints. "
                "Only load weights from trusted sources."
            )
            logger = getattr(self, "logger", None)
            if logger is not None and hasattr(logger, "warning"):
                try:
                    logger.warning(warn_msg)
                except Exception:
                    print(warn_msg)
            else:
                print(warn_msg)

        patched_torch_load = False
        original_torch_load: Optional[Callable[..., Any]] = None
        if torch is not None:
            candidate = getattr(torch, "load", None)
            if callable(candidate) and not getattr(candidate, "_mixformer_forced_weights_only", False):
                original_torch_load = candidate

                def _patched_torch_load(*args: Any, **kwargs: Any):
                    kwargs.setdefault("weights_only", False)
                    _emit_weights_warning()
                    return original_torch_load(*args, **kwargs)  # type: ignore[misc]

                setattr(_patched_torch_load, "_mixformer_forced_weights_only", True)
                torch.load = _patched_torch_load  # type: ignore[assignment]
                patched_torch_load = True

        kwargs: Dict[str, Any] = {
            "yaml_name": self.parameter_name,
            "model": self._checkpoint_arg,
        }
        if self.search_area_scale is not None:
            kwargs["search_area_scale"] = self.search_area_scale
        if self.online_size is not None:
            kwargs["online_size"] = self.online_size
        if self.update_interval is not None:
            kwargs["update_interval"] = self.update_interval

        try:
            params = self._parameters_fn(**kwargs)
            if not hasattr(params, "debug"):
                params.debug = 0
            if not hasattr(params, "vis_attn"):
                params.vis_attn = 0
            tracker = self._tracker_cls(params, self.dataset_name)
            finetuned_state = getattr(self, "_finetuned_state_dict", None)
            if finetuned_state:
                try:
                    tracker.network.load_state_dict(finetuned_state, strict=False)
                except Exception as ex:
                    warn_msg = f"MixFormerV2: failed to load finetuned weights, falling back to checkpoint: {ex}"
                    logger = getattr(self, "logger", None)
                    if logger is not None and hasattr(logger, "warning"):
                        try:
                            logger.warning(warn_msg)
                        except Exception:
                            print(warn_msg)
                    else:
                        print(warn_msg)
        finally:
            if patched_torch_load and torch is not None and original_torch_load is not None:
                torch.load = original_torch_load  # type: ignore[assignment]

        original_load = getattr(tracker, "load_state", None)
        if callable(original_load):

            def _load_with_weights(path, **load_kwargs):
                load_kwargs.setdefault("weights_only", False)
                return original_load(path, **load_kwargs)

            tracker.load_state = _load_with_weights  # type: ignore[attr-defined]

        return tracker

    def _predict_frame(self, tracker: Any, frame_rgb: np.ndarray) -> _MixFormerResult:
        out = tracker.track(frame_rgb, {})
        bbox = out.get("target_bbox") if isinstance(out, dict) else None
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            bbox = tuple(float(b) for b in bbox[:4])  # type: ignore
        else:
            bbox = None
        score = out.get("conf_score") if isinstance(out, dict) else None
        if score is not None:
            try:
                score = float(score)
            except Exception:
                score = None
        return _MixFormerResult(bbox, score)

    def predict(self, video_path: str) -> List[FramePrediction]:
        if cv2 is None:
            detail = (
                f" (import error: {_CV2_IMPORT_ERROR!r})"
                if "_CV2_IMPORT_ERROR" in globals() else ""
            )
            raise RuntimeError("OpenCV is required for MixFormerV2 tracker." + detail)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        init_bbox = resolve_first_frame_bbox(
            video_path,
            mode=self.first_frame_source,
            detector=self.init_detector_params,
            fallback=self.first_frame_fallback,
        )
        if not _is_valid_bbox(init_bbox):
            cap.release()
            raise RuntimeError(
                "MixFormerV2 requires a valid first-frame bounding box. "
                f"Failed to obtain one (mode={self.first_frame_source}) for '{os.path.basename(video_path)}'."
            )

        tracker = self._build_tracker()

        preds: List[FramePrediction] = []
        frame_idx = 0
        last_bbox: Optional[Tuple[float, float, float, float]] = init_bbox

        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                frame_rgb = self._apply_preprocs(frame_bgr)
                if frame_idx == 0:
                    init_bbox_list = [float(v) for v in init_bbox]
                    tracker.initialize(frame_rgb, {"init_bbox": init_bbox_list})
                    init_bbox_tuple = tuple(init_bbox_list)
                    preds.append(FramePrediction(frame_idx, init_bbox_tuple, 1.0))
                    last_bbox = init_bbox_tuple
                else:
                    result = self._predict_frame(tracker, frame_rgb)
                    if _is_valid_bbox(result.bbox) and (result.score is None or result.score >= self.min_confidence):
                        preds.append(FramePrediction(frame_idx, result.bbox, result.score))
                        last_bbox = result.bbox
                    elif self.fallback_last_prediction and _is_valid_bbox(last_bbox):
                        preds.append(FramePrediction(frame_idx, last_bbox, result.score))
                frame_idx += 1
        finally:
            cap.release()

        return preds
