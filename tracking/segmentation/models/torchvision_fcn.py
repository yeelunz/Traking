from __future__ import annotations

import importlib
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

try:  # pragma: no cover - optional dependency resolution
    from torchvision.models.segmentation import (  # type: ignore
        fcn_resnet50,
        FCN_ResNet50_Weights,
    )
except Exception as exc:  # pragma: no cover
    fcn_resnet50 = None  # type: ignore
    FCN_ResNet50_Weights = None  # type: ignore
    _TV_IMPORT_ERROR = exc
else:
    _TV_IMPORT_ERROR = None

from ...core.registry import register_segmentation_model


@register_segmentation_model("torchvision_fcn_resnet50")
class TorchvisionFCNSegmenter(nn.Module):
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__()
        fcn_ctor = fcn_resnet50
        weights_cls = FCN_ResNet50_Weights
        # Allow runtime overrides (e.g., tests monkeypatching
        # tracking.segmentation.model.fcn_resnet50 / FCN_ResNet50_Weights).
        try:
            seg_model_module = importlib.import_module("tracking.segmentation.model")
            fcn_override = getattr(seg_model_module, "fcn_resnet50", None)
            weights_override = getattr(seg_model_module, "FCN_ResNet50_Weights", None)
            if callable(fcn_override):
                fcn_ctor = fcn_override
            if weights_override is not None:
                weights_cls = weights_override
        except Exception:
            pass

        if fcn_ctor is None:  # pragma: no cover
            raise ImportError(
                "torchvision is required for 'torchvision_fcn_resnet50'."
                f" Install via `pip install torchvision`. Original error: {_TV_IMPORT_ERROR}"
            )
        cfg = dict(params or {})
        weights_spec = cfg.get("weights", "DEFAULT")
        weights_enum = None
        if weights_cls is not None:
            if isinstance(weights_spec, str):
                key = weights_spec.upper()
                weights_enum = getattr(weights_cls, key, None)
                if weights_enum is None and hasattr(weights_cls, "DEFAULT"):
                    weights_enum = weights_cls.DEFAULT
            else:
                weights_enum = weights_spec
        self.weights_enum = weights_enum
        self.model = fcn_ctor(weights=weights_enum)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.transform = None
        if weights_enum is not None and hasattr(weights_enum, "transforms"):
            try:
                self.transform = weights_enum.transforms(antialias=True)
            except TypeError:
                self.transform = weights_enum.transforms()
        self.train_enabled = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.transform is not None:
            x = self.transform(x)
        outputs = self.model(x)
        if isinstance(outputs, dict):
            logits = outputs.get("out")
        else:  # pragma: no cover - unexpected API
            logits = outputs
        if logits is None:
            raise RuntimeError("FCN model did not return 'out' tensor")
        if logits.ndim != 4:
            raise RuntimeError(f"Unexpected FCN output shape: {logits.shape}")
        probs = torch.softmax(logits, dim=1)
        if probs.size(1) == 1:
            fg_prob = probs
        else:
            fg_prob, _ = probs[:, 1:, :, :].max(dim=1, keepdim=True)
        eps = 1e-6
        fg_prob = fg_prob.clamp(eps, 1.0 - eps)
        fg_logit = torch.log(fg_prob / (1.0 - fg_prob))
        return fg_logit


__all__ = ["TorchvisionFCNSegmenter"]
