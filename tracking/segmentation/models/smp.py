from __future__ import annotations

from typing import Any, Dict, Optional

import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
except Exception as exc:  # pragma: no cover
    smp = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from ...core.registry import register_segmentation_model


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
                " Install via `pip install segmentation-models-pytorch`."
                f" Original error: {_IMPORT_ERROR}"
            )
        self.model = _build_smp_model(smp.Unet, params)

    def forward(self, x):  # type: ignore[override]
        return self.model(x)


@register_segmentation_model("unetpp")
class UnetPlusPlus(nn.Module):
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__()
        if smp is None:  # pragma: no cover
            raise ImportError(
                "segmentation_models_pytorch is required for 'unetpp'."
                " Install via `pip install segmentation-models-pytorch`."
                f" Original error: {_IMPORT_ERROR}"
            )
        self.model = _build_smp_model(smp.UnetPlusPlus, params)

    def forward(self, x):  # type: ignore[override]
        return self.model(x)


@register_segmentation_model("deeplabv3+")
class DeepLabV3PlusSegmenter(nn.Module):
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__()
        if smp is None:  # pragma: no cover
            raise ImportError(
                "segmentation_models_pytorch is required for 'deeplabv3+'."
                " Install via `pip install segmentation-models-pytorch`."
                f" Original error: {_IMPORT_ERROR}"
            )
        if not hasattr(smp, "DeepLabV3Plus"):
            raise AttributeError(
                "Installed segmentation_models_pytorch does not expose DeepLabV3Plus."
                " Please upgrade to a build that includes DeepLabV3Plus."
            )
        self.model = _build_smp_model(smp.DeepLabV3Plus, params)

    def forward(self, x):  # type: ignore[override]
        return self.model(x)


__all__ = [
    "Unet",
    "UnetPlusPlus",
    "DeepLabV3PlusSegmenter",
    "_build_smp_model",
]
