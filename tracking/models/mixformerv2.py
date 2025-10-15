from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Tuple

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
from ..utils.annotations import load_coco_vid


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


def _first_gt_bbox(video_path: str) -> Optional[Tuple[float, float, float, float]]:
    json_path = os.path.splitext(video_path)[0] + ".json"
    if not os.path.exists(json_path):
        return None
    try:
        gt = load_coco_vid(json_path)
    except Exception:
        return None
    frames = gt.get("frames", {}) or {}
    if not frames:
        return None
    valid = sorted(int(k) for k, boxes in frames.items() if boxes)
    if not valid:
        return None
    bboxes = frames.get(valid[0]) or []
    if not bboxes:
        return None
    x, y, w, h = bboxes[0]
    return float(x), float(y), float(w), float(h)


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

    # ------------------------------------------------------------------
    # TrackingModel API
    # ------------------------------------------------------------------
    def train(self, train_dataset, val_dataset=None, seed: int = 0, output_dir: Optional[str] = None):
        if not self.train_enabled:
            return {"status": "skipped", "reason": "train_disabled"}
        cb = getattr(self, "progress_callback", None)
        if callable(cb):
            try:
                cb("train_epoch_start", 1, 1)
                cb("train_epoch_end", 1, 1)
            except Exception:
                pass
        return {"status": "no_training", "reason": "not_implemented"}

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

        init_bbox = _first_gt_bbox(video_path)
        if not _is_valid_bbox(init_bbox):
            cap.release()
            raise RuntimeError(
                "MixFormerV2 requires a valid first-frame ground-truth bbox. "
                f"No usable bbox found next to '{os.path.basename(video_path)}'."
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
