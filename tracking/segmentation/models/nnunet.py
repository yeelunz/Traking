from __future__ import annotations

import json
import os
import warnings
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from ...core.registry import register_segmentation_model
from .common import load_json_file, resolve_path

try:  # pragma: no cover - optional heavy dependency
    from nnunetv2.utilities.get_network_from_plans import (  # type: ignore[import]
        get_network_from_plans as _nnunet_get_network,
    )
    from nnunetv2.utilities.plans_handling.plans_handler import (  # type: ignore[import]
        PlansManager as _NnUNetPlansManager,
    )
except Exception as exc:  # pragma: no cover
    _nnunet_get_network = None  # type: ignore
    _NnUNetPlansManager = None  # type: ignore
    _NNUNET_IMPORT_ERROR = exc
else:
    _NNUNET_IMPORT_ERROR = None


def _normalize_architecture_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("nnU-Net architecture definition must be a dict")
    if "network_class_name" not in data or "arch_kwargs" not in data:
        raise ValueError(
            "nnU-Net architecture dict requires 'network_class_name' and 'arch_kwargs' keys"
        )
    arch_kwargs = dict(data["arch_kwargs"])
    requires = data.get("_kw_requires_import", [])
    if isinstance(requires, str):
        requires = [requires]
    requires_list = [str(entry) for entry in requires]
    normalized = {
        "network_class_name": str(data["network_class_name"]),
        "arch_kwargs": arch_kwargs,
        "_kw_requires_import": requires_list,
    }
    if "deep_supervision" in data:
        normalized["deep_supervision"] = bool(data["deep_supervision"])
    return normalized


def _architecture_from_plans(plans_path: str, configuration: Optional[str]) -> Dict[str, Any]:
    if _NnUNetPlansManager is None:
        raise ImportError(
            "nnUNetv2 is required to read plans files. Install it via `pip install -e ./libs/nnUNet`"
            f". Original error: {_NNUNET_IMPORT_ERROR}"
        )
    resolved = resolve_path(plans_path)
    manager = _NnUNetPlansManager(resolved)
    config_name = str(configuration or "2d").strip() or "2d"
    config = manager.get_configuration(config_name)
    arch = {
        "network_class_name": config.network_arch_class_name,
        "arch_kwargs": dict(config.network_arch_init_kwargs),
        "_kw_requires_import": list(config.network_arch_init_kwargs_req_import),
    }
    config_dict = getattr(config, "configuration", {})
    if isinstance(config_dict, dict) and "deep_supervision" in config_dict:
        arch["deep_supervision"] = bool(config_dict["deep_supervision"])
    return _normalize_architecture_dict(arch)


def _default_nnunet_architecture(params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = dict(params or {})
    spatial_dim = int(cfg.get("dimension", cfg.get("spatial_dims", 2)) or 2)
    spatial_dim = 3 if spatial_dim == 3 else 2
    n_stages = max(3, int(cfg.get("n_stages", 5)))
    base_features = max(8, int(cfg.get("base_features", cfg.get("base_num_features", 32))))
    max_features = max(base_features, int(cfg.get("max_features", 512)))
    encoder_convs_setting = cfg.get("n_conv_per_stage", 2)
    if isinstance(encoder_convs_setting, (list, tuple)):
        encoder_convs = [max(1, int(v)) for v in encoder_convs_setting]
        if len(encoder_convs) < n_stages:
            encoder_convs.extend([encoder_convs[-1]] * (n_stages - len(encoder_convs)))
        else:
            encoder_convs = encoder_convs[:n_stages]
    else:
        encoder_convs = [max(1, int(encoder_convs_setting))] * n_stages
    decoder_convs_setting = cfg.get("n_conv_per_stage_decoder", 2)
    if isinstance(decoder_convs_setting, (list, tuple)):
        decoder_convs = [max(1, int(v)) for v in decoder_convs_setting]
        if len(decoder_convs) < n_stages - 1:
            decoder_convs.extend([decoder_convs[-1]] * ((n_stages - 1) - len(decoder_convs)))
        else:
            decoder_convs = decoder_convs[: max(1, n_stages - 1)]
    else:
        decoder_convs = [max(1, int(decoder_convs_setting))] * max(1, n_stages - 1)
    features: List[int] = []
    for stage in range(n_stages):
        feat = min(base_features * (2 ** stage), max_features)
        features.append(feat)
    if spatial_dim == 3:
        conv_op = "torch.nn.modules.conv.Conv3d"
        norm_op = "torch.nn.modules.instancenorm.InstanceNorm3d"
        base_kernel = [3, 3, 3]
        stride_unit = [1, 1, 1]
    else:
        conv_op = "torch.nn.modules.conv.Conv2d"
        norm_op = "torch.nn.modules.instancenorm.InstanceNorm2d"
        base_kernel = [3, 3]
        stride_unit = [1, 1]
    kernel_setting = cfg.get("kernel_sizes")
    if (
        isinstance(kernel_setting, (list, tuple))
        and kernel_setting
        and isinstance(kernel_setting[0], (list, tuple))
    ):
        kernel_sizes = [
            [int(max(1, v)) for v in list(stage_kernel)[:spatial_dim]]
            for stage_kernel in kernel_setting[:n_stages]
        ]
        while len(kernel_sizes) < n_stages:
            kernel_sizes.append(list(kernel_sizes[-1]))
    else:
        base_kernel_setting = cfg.get("kernel_size")
        if isinstance(base_kernel_setting, (list, tuple)):
            base_kernel = [int(max(1, v)) for v in base_kernel_setting]
        elif isinstance(base_kernel_setting, int):
            base_kernel = [max(1, int(base_kernel_setting))] * spatial_dim
        if len(base_kernel) < spatial_dim:
            base_kernel.extend([base_kernel[-1]] * (spatial_dim - len(base_kernel)))
        else:
            base_kernel = base_kernel[:spatial_dim]
        kernel_sizes = [list(base_kernel) for _ in range(n_stages)]
    stride_setting = cfg.get("strides")
    if (
        isinstance(stride_setting, (list, tuple))
        and stride_setting
        and isinstance(stride_setting[0], (list, tuple))
    ):
        strides = [
            [int(max(1, v)) for v in list(stage_stride)[:spatial_dim]]
            for stage_stride in stride_setting[:n_stages]
        ]
        while len(strides) < n_stages:
            strides.append(list(strides[-1]))
    else:
        if isinstance(stride_setting, (list, tuple)):
            stride_step = [max(1, int(v)) for v in stride_setting]
        elif isinstance(stride_setting, int):
            stride_step = [max(1, int(stride_setting))] * spatial_dim
        else:
            alt_stride = cfg.get("downsample_stride") or cfg.get("stride_step")
            if isinstance(alt_stride, (list, tuple)):
                stride_step = [max(1, int(v)) for v in alt_stride]
            elif isinstance(alt_stride, int):
                stride_step = [max(1, int(alt_stride))] * spatial_dim
            else:
                stride_step = [2] * spatial_dim
        if len(stride_step) < spatial_dim:
            stride_step.extend([stride_step[-1]] * (spatial_dim - len(stride_step)))
        else:
            stride_step = stride_step[:spatial_dim]
        strides = [list(stride_unit)] + [list(stride_step) for _ in range(n_stages - 1)]
    arch = {
        "network_class_name": "dynamic_network_architectures.architectures.unet.PlainConvUNet",
        "arch_kwargs": {
            "n_stages": n_stages,
            "features_per_stage": features,
            "conv_op": conv_op,
            "kernel_sizes": kernel_sizes,
            "strides": strides,
            "n_conv_per_stage": encoder_convs,
            "n_conv_per_stage_decoder": decoder_convs,
            "conv_bias": True,
            "norm_op": norm_op,
            "norm_op_kwargs": {"eps": 1e-5, "affine": True},
            "dropout_op": None,
            "dropout_op_kwargs": None,
            "nonlin": "torch.nn.LeakyReLU",
            "nonlin_kwargs": {"inplace": True},
        },
        "_kw_requires_import": ["conv_op", "norm_op", "dropout_op", "nonlin"],
    }
    if "deep_supervision" in cfg:
        arch["deep_supervision"] = bool(cfg.get("deep_supervision"))
    return arch


def _select_nnunet_architecture(params: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(params)
    arch_spec = cfg.get("architecture")
    if isinstance(arch_spec, str):
        arch_spec = load_json_file(arch_spec)
    if isinstance(arch_spec, dict):
        return _normalize_architecture_dict(arch_spec)
    arch_path = cfg.get("architecture_path") or cfg.get("architecture_file")
    if isinstance(arch_path, str) and arch_path.strip():
        return _normalize_architecture_dict(load_json_file(arch_path))
    plans_path = cfg.get("plans_path") or cfg.get("plans")
    if isinstance(plans_path, str) and plans_path.strip():
        configuration = cfg.get("configuration") or cfg.get("configuration_name")
        return _architecture_from_plans(plans_path, configuration)
    return _normalize_architecture_dict(_default_nnunet_architecture(cfg))


@register_segmentation_model("nnunet")
class NnUNetSegmenter(nn.Module):
    """Thin wrapper around nnU-Net v2 architectures via the official helper APIs."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__()
        if _nnunet_get_network is None:
            raise ImportError(
                "nnUNetv2 is required for 'nnunet'. Install via `pip install -e ./libs/nnUNet`"
                f". Original error: {_NNUNET_IMPORT_ERROR}"
            )
        cfg = dict(params or {})
        arch_spec = _select_nnunet_architecture(cfg)
        deep_supervision = cfg.get("deep_supervision")
        if deep_supervision is None:
            deep_supervision = arch_spec.get("deep_supervision")
        deep_supervision_flag = None
        if deep_supervision is not None:
            deep_supervision_flag = bool(deep_supervision)
        self.deep_supervision = bool(deep_supervision_flag)
        self.return_highres_only = bool(cfg.get("return_highres_only", True))
        input_channels = int(cfg.get("in_channels", cfg.get("input_channels", 3)))
        output_channels = int(cfg.get("classes", cfg.get("num_classes", 1)))
        allow_init = bool(cfg.get("init_weights", True))
        self.network = _nnunet_get_network(
            arch_class_name=arch_spec["network_class_name"],
            arch_kwargs=arch_spec["arch_kwargs"],
            arch_kwargs_req_import=arch_spec["_kw_requires_import"],
            input_channels=input_channels,
            output_channels=output_channels,
            allow_init=allow_init,
            deep_supervision=deep_supervision_flag,
        )
        checkpoint = cfg.get("checkpoint") or cfg.get("weights") or cfg.get("pretrained")
        if checkpoint:
            self.load_checkpoint(checkpoint, strict=bool(cfg.get("strict_checkpoint", True)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        outputs = self.network(x)
        if isinstance(outputs, (list, tuple)):
            if not outputs:
                raise RuntimeError("nnU-Net network returned no outputs")
            return outputs[0] if self.return_highres_only else outputs[-1]
        return outputs

    def load_checkpoint(self, path: str, strict: bool = True):
        resolved = resolve_path(path)
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"nnU-Net checkpoint not found: {path}")
        state = torch.load(resolved, map_location="cpu")
        if isinstance(state, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                if isinstance(state.get(key), dict):
                    state = state[key]
                    break
        if not isinstance(state, dict):
            raise RuntimeError("Unsupported nnU-Net checkpoint format; expected a state dict")
        cleaned_state = {}
        for key, value in state.items():
            new_key = key[7:] if key.startswith("module.") else key
            cleaned_state[new_key] = value
        load_info = self.network.load_state_dict(cleaned_state, strict=strict)
        missing = getattr(load_info, "missing_keys", [])
        unexpected = getattr(load_info, "unexpected_keys", [])
        if missing or unexpected:
            warnings.warn(
                f"Loaded nnU-Net checkpoint with missing={missing} unexpected={unexpected}"
            )
        return load_info


__all__ = [
    "NnUNetSegmenter",
    "_nnunet_get_network",
    "_NnUNetPlansManager",
    "_NNUNET_IMPORT_ERROR",
    "_select_nnunet_architecture",
    "_default_nnunet_architecture",
]
