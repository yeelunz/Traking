from __future__ import annotations
from typing import Dict, Any, Optional
import os

# Lazy/safe import of Ultralytics RTDETR
_ULTRA_IMPORT_ERROR: Optional[Exception] = None
try:
    from ultralytics import RTDETR  # type: ignore
except Exception as e:  # pragma: no cover - environment without ultralytics
    raise ImportError(
        "Failed to import ultralytics RTDETR for tracking.models.rtdetrv2. Install ultralytics."
    ) from e

from ..core.registry import register_model
from ..utils.init_bbox import resolve_weights_path
from .yolov11 import YOLOv11Model


@register_model("RTDETRv2")
@register_model("RT-DETRv2")
@register_model("RTDETR")
class RTDETRv2Model(YOLOv11Model):
    """Single-object tracking via per-frame detection using Ultralytics RT-DETR.

    This model keeps the same pipeline contract as YOLOv11Model, including
    optional interpolation-based hole filling and training/export behavior.
    """

    name = "RTDETRv2"
    DEFAULT_CONFIG = {
        **YOLOv11Model.DEFAULT_CONFIG,
        "weights": "rtdetr-l.pt",
    }

    def __init__(self, config: Dict[str, Any]):
        # --- Config params ---
        self.weights = str(config.get("weights", self.DEFAULT_CONFIG["weights"]))
        self._weights_path = resolve_weights_path(self.weights)
        self.conf = float(config.get("conf", self.DEFAULT_CONFIG["conf"]))
        self.iou = float(config.get("iou", self.DEFAULT_CONFIG["iou"]))
        self.imgsz = int(config.get("imgsz", self.DEFAULT_CONFIG["imgsz"]))
        self.device = str(config.get("device", self.DEFAULT_CONFIG["device"]))
        self.classes = config.get("classes", self.DEFAULT_CONFIG["classes"])  # type: ignore
        self.max_det = int(config.get("max_det", self.DEFAULT_CONFIG["max_det"]))
        self.min_confidence = float(config.get("min_confidence", self.DEFAULT_CONFIG["min_confidence"]))
        self.fallback_last_prediction = bool(
            config.get("fallback_last_prediction", self.DEFAULT_CONFIG["fallback_last_prediction"])
        )
        self.fallback_missing_interpolation = bool(
            config.get("fallback_missing_interpolation", self.DEFAULT_CONFIG["fallback_missing_interpolation"])
        )
        self.interpolation_max_gap = int(
            config.get("interpolation_max_gap", self.DEFAULT_CONFIG["interpolation_max_gap"])
        )
        self.include_empty_frames = bool(
            config.get("include_empty_frames", self.DEFAULT_CONFIG.get("include_empty_frames", False))
        )

        # --- Training-specific ---
        self.epochs = int(config.get("epochs", self.DEFAULT_CONFIG["epochs"]))
        self.batch = int(config.get("batch", self.DEFAULT_CONFIG["batch"]))
        self.lr0 = float(config.get("lr0", self.DEFAULT_CONFIG["lr0"]))
        self.patience = int(config.get("patience", self.DEFAULT_CONFIG["patience"]))
        self.workers = int(config.get("workers", self.DEFAULT_CONFIG["workers"]))
        self.train_enabled = bool(config.get("train_enabled", self.DEFAULT_CONFIG["train_enabled"]))

        # --- Runtime-injected preproc chain ---
        self.preprocs = []

        # --- Build / load weights with corruption fallback ---
        try:
            self.model = RTDETR(self._weights_path)
        except Exception as e:
            try:
                if os.path.exists(self._weights_path) and os.path.isfile(self._weights_path):
                    size = -1
                    try:
                        size = os.path.getsize(self._weights_path)
                    except Exception:
                        pass
                    if 0 <= size < 1024:
                        try:
                            os.remove(self._weights_path)
                        except Exception:
                            pass
                        try:
                            self.model = RTDETR(self._weights_path)
                            e = None
                        except Exception:
                            pass
            except Exception:
                pass
            if e is not None:
                raise RuntimeError(
                    f"Failed to load RTDETRv2 weights '{self.weights}': {e}.\n"
                    "Common causes: corrupted or partial .pt file. Remove it and retry, or set params.weights to a valid checkpoint."
                )

        # --- Resolve device (fallback to CPU if CUDA not available) ---
        try:
            import torch  # type: ignore

            if self.device != "cpu" and not torch.cuda.is_available():
                self._device_str = "cpu"
            else:
                self._device_str = self.device
        except Exception:
            self._device_str = "cpu"

    def load_checkpoint(self, ckpt_path: str):
        if RTDETR is None:
            detail = f" underlying import error: {_ULTRA_IMPORT_ERROR!r}" if _ULTRA_IMPORT_ERROR else ""
            raise RuntimeError(f"Ultralytics not available.{detail}")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(ckpt_path)
        self.model = RTDETR(ckpt_path)
