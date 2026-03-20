from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # noqa: BLE001
    torch = None  # type: ignore
    nn = None  # type: ignore
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

try:
    import timm
except Exception as exc:  # noqa: BLE001
    timm = None  # type: ignore
    _TIMM_IMPORT_ERROR = exc
else:
    _TIMM_IMPORT_ERROR = None


logger = logging.getLogger(__name__)


TEXTURE_MODES = {"freeze", "learnable", "pretrain"}


def _require_runtime() -> None:
    if torch is None or nn is None:
        raise RuntimeError(
            "PyTorch is required for texture backbone wrapper."
            + (f" Import error: {_TORCH_IMPORT_ERROR}" if _TORCH_IMPORT_ERROR else "")
        )
    if timm is None:
        raise RuntimeError(
            "timm is required for texture backbone wrapper."
            + (f" Import error: {_TIMM_IMPORT_ERROR}" if _TIMM_IMPORT_ERROR else "")
        )


def _extract_state_dict(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        for key in (
            "backbone_state_dict",
            "backbone",
            "state_dict",
            "model",
            "model_state_dict",
            "full_model_state_dict",
        ):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        if any(isinstance(v, torch.Tensor) for v in payload.values()):
            return payload
    raise ValueError("Could not locate a valid state_dict in checkpoint payload.")


def _extract_projection_state_dict(payload: Any) -> Optional[Dict[str, Any]]:
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


def _strip_prefixes(state: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in state.items():
        key = str(k)
        for p in ("module.", "backbone.", "model.", "net.", "proj.", "projection."):
            if key.startswith(p):
                key = key[len(p) :]
        out[key] = v
    return out


def _has_official_pretrained(backbone_name: str) -> bool:
    if timm is None:
        return False
    try:
        models = timm.list_models(pretrained=True)
    except Exception:  # noqa: BLE001
        return True
    return str(backbone_name) in set(models)


class TextureBackboneWrapper(nn.Module):
    """Texture branch wrapper with explicit mode separation.

    Modes
    -----
    - freeze: ImageNet-pretrained backbone, frozen weights.
    - learnable: trainable backbone, end-to-end gradients.
    - pretrain: load domain checkpoint, frozen in main pipeline.
    """

    def __init__(
        self,
        *,
        mode: str = "freeze",
        backbone_name: str = "convnext_tiny",
        texture_dim: int = 32,
        image_size: int = 96,
        pretrain_ckpt: Optional[str] = None,
        pretrained_imagenet: bool = True,
    ):
        _require_runtime()
        super().__init__()

        mode_lc = str(mode).strip().lower()
        if mode_lc not in TEXTURE_MODES:
            raise ValueError(f"Unsupported texture_mode={mode!r}. Choose from {sorted(TEXTURE_MODES)}")

        self.texture_mode = mode_lc
        self.backbone_name = str(backbone_name)
        self.texture_dim = int(texture_dim)
        self.image_size = int(image_size)

        use_official_pretrained = False
        if self.texture_mode == "freeze":
            if _has_official_pretrained(self.backbone_name):
                use_official_pretrained = True
            else:
                logger.warning(
                    "texture_mode=freeze requested official pretrained weights, but no official weights were found for backbone=%s. Falling back to random init.",
                    self.backbone_name,
                )
        elif self.texture_mode == "learnable":
            use_official_pretrained = bool(pretrained_imagenet)

        try:
            self.backbone = timm.create_model(
                self.backbone_name,
                pretrained=use_official_pretrained,
                num_classes=0,
                global_pool="avg",
            )
        except Exception as exc:  # noqa: BLE001
            if use_official_pretrained:
                logger.warning(
                    "Failed to load official pretrained weights for backbone=%s (%s). Falling back to random init.",
                    self.backbone_name,
                    exc,
                )
                self.backbone = timm.create_model(
                    self.backbone_name,
                    pretrained=False,
                    num_classes=0,
                    global_pool="avg",
                )
            else:
                raise

        with torch.no_grad():
            dummy = torch.zeros((1, 3, self.image_size, self.image_size), dtype=torch.float32)
            raw = self.backbone(dummy)
            backbone_dim = int(raw.shape[-1])

        self.proj = nn.Sequential(
            nn.Linear(backbone_dim, self.texture_dim),
            nn.LayerNorm(self.texture_dim),
            nn.GELU(),
        )

        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1))

        if self.texture_mode == "pretrain":
            if not pretrain_ckpt:
                raise ValueError("texture_mode='pretrain' requires texture_pretrain_ckpt.")
            self._load_pretrain_ckpt(pretrain_ckpt)

        self._apply_mode()

    def _apply_mode(self) -> None:
        trainable = self.texture_mode == "learnable"
        self.backbone.requires_grad_(trainable)
        if trainable:
            self.backbone.train()
        else:
            self.backbone.eval()

    def _load_pretrain_ckpt(self, ckpt_path: str) -> None:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"texture_pretrain_ckpt not found: {ckpt_path}")

        payload = torch.load(ckpt_path, map_location="cpu")
        if isinstance(payload, dict):
            ckpt_backbone = payload.get("backbone_name")
            if isinstance(ckpt_backbone, str) and ckpt_backbone and ckpt_backbone != self.backbone_name:
                raise ValueError(
                    f"Checkpoint backbone mismatch: ckpt={ckpt_backbone}, requested={self.backbone_name}"
                )

        raw_state = _extract_state_dict(payload)
        state = _strip_prefixes(raw_state)
        backbone_keys = set(self.backbone.state_dict().keys())
        overlap = backbone_keys.intersection(set(state.keys()))
        if not overlap:
            raise ValueError(
                "Checkpoint does not contain matching backbone parameters for "
                f"{self.backbone_name}."
            )
        incompatible = self.backbone.load_state_dict(state, strict=False)

        missing_keys = list(getattr(incompatible, "missing_keys", []) or [])
        unexpected_keys = list(getattr(incompatible, "unexpected_keys", []) or [])
        if missing_keys or unexpected_keys:
            logger.warning(
                "texture pretrain ckpt loaded with key mismatch: missing=%d unexpected=%d",
                len(missing_keys),
                len(unexpected_keys),
            )

        if len(overlap) < max(1, int(0.2 * len(backbone_keys))):
            logger.warning(
                "Low backbone key overlap when loading pretrain ckpt: overlap=%d total_backbone=%d",
                len(overlap),
                len(backbone_keys),
            )

        proj_state = _extract_projection_state_dict(payload)
        if isinstance(proj_state, dict) and proj_state:
            proj_clean = _strip_prefixes(proj_state)
            proj_incompatible = self.proj.load_state_dict(proj_clean, strict=False)
            proj_missing = list(getattr(proj_incompatible, "missing_keys", []) or [])
            proj_unexpected = list(getattr(proj_incompatible, "unexpected_keys", []) or [])
            if proj_missing or proj_unexpected:
                logger.warning(
                    "texture pretrain projection loaded with key mismatch: missing=%d unexpected=%d",
                    len(proj_missing),
                    len(proj_unexpected),
                )
        else:
            logger.warning(
                "texture pretrain ckpt has no projection_state_dict; projection will use fresh init."
            )

    def _forward_backbone(self, x_norm: "torch.Tensor") -> "torch.Tensor":
        if self.texture_mode == "learnable":
            return self.backbone(x_norm)
        with torch.no_grad():
            return self.backbone(x_norm)

    def forward(self, x_image: "torch.Tensor") -> "torch.Tensor":
        x_norm = (x_image - self.img_mean) / (self.img_std + 1e-8)
        feat = self._forward_backbone(x_norm)
        return self.proj(feat)
