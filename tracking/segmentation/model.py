from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
except Exception as exc:  # pragma: no cover
    smp = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

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

from ..core.registry import register_segmentation_model


def _build_smp_model(model_ctor, params: Optional[Dict[str, Any]] = None) -> nn.Module:
    if smp is None:  # pragma: no cover - handled at runtime
        raise ImportError(
            "segmentation_models_pytorch is required for segmentation models."
            f" Import error: {_IMPORT_ERROR}"
        )
    cfg = dict(params or {})
    encoder_name = cfg.pop("encoder_name", "resnet34")
    encoder_weights = cfg.pop("encoder_weights", "imagenet")
    in_channels = int(cfg.pop("in_channels", 3))
    classes = int(cfg.pop("classes", 1))
    activation = cfg.pop("activation", None)
    return model_ctor(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=activation,
        **cfg,
    )


@register_segmentation_model("unet")
class Unet(nn.Module):
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__()
        if smp is None:  # pragma: no cover
            raise ImportError(
                "segmentation_models_pytorch is required for 'unet'."
                f" Install via `pip install segmentation-models-pytorch`. Original error: {_IMPORT_ERROR}"
            )
        self.model = _build_smp_model(smp.Unet, params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@register_segmentation_model("unetpp")
class UnetPlusPlus(nn.Module):
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__()
        if smp is None:  # pragma: no cover
            raise ImportError(
                "segmentation_models_pytorch is required for 'unetpp'."
                f" Install via `pip install segmentation-models-pytorch`. Original error: {_IMPORT_ERROR}"
            )
        self.model = _build_smp_model(smp.UnetPlusPlus, params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@register_segmentation_model("torchvision_fcn_resnet50")
class TorchvisionFCNSegmenter(nn.Module):
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__()
        if fcn_resnet50 is None:  # pragma: no cover
            raise ImportError(
                "torchvision is required for 'torchvision_fcn_resnet50'."
                f" Install via `pip install torchvision`. Original error: {_TV_IMPORT_ERROR}"
            )
        cfg = dict(params or {})
        weights_spec = cfg.get("weights", "DEFAULT")
        weights_enum = None
        if FCN_ResNet50_Weights is not None:
            if isinstance(weights_spec, str):
                key = weights_spec.upper()
                weights_enum = getattr(FCN_ResNet50_Weights, key, None)
                if weights_enum is None and hasattr(FCN_ResNet50_Weights, "DEFAULT"):
                    weights_enum = FCN_ResNet50_Weights.DEFAULT
            else:
                weights_enum = weights_spec
        self.weights_enum = weights_enum
        self.model = fcn_resnet50(weights=weights_enum)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


__all__ = ["Unet", "UnetPlusPlus", "TorchvisionFCNSegmenter"]
