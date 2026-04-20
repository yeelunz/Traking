from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # noqa: BLE001
    torch = None  # type: ignore
    nn = None  # type: ignore
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

from ..core.registry import FUSION_MODULE_REGISTRY, register_fusion_module


LEARNABLE_FUSION_MODULES = {"gating", "attention", "attention_gating", "gated_attention"}


def is_learnable_fusion_module(name: str) -> bool:
    return str(name or "").strip().lower() in LEARNABLE_FUSION_MODULES


def _normalize_splits(feature_splits: Optional[Iterable[int]]) -> List[int]:
    if not feature_splits:
        return []
    out: List[int] = []
    for value in feature_splits:
        iv = int(value)
        if iv > 0:
            out.append(iv)
    return out


class _NumpyFusionBase:
    name = "fusion_base"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = dict(params or {})
        self.feature_splits = _normalize_splits(self.params.get("feature_splits"))

    def fit(self, X, y=None):  # noqa: ANN001
        _ = (X, y)
        return self

    def transform(self, X):  # noqa: ANN001
        return np.asarray(X, dtype=np.float32)

    def fit_transform(self, X, y=None):  # noqa: ANN001
        return self.fit(X, y).transform(X)


@register_fusion_module("concat")
class ConcatFusionModule(_NumpyFusionBase):
    name = "concat"


@register_fusion_module("gating")
class GatingFusionModule(_NumpyFusionBase):
    name = "gating"


@register_fusion_module("attention")
class AttentionFusionModule(_NumpyFusionBase):
    name = "attention"


_ModuleBase = nn.Module if nn is not None else object


class _TorchAttentionGatingFusion(_ModuleBase):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, x):  # noqa: ANN001
        attn_logits = self.attn(x).squeeze(-1)
        attn = torch.softmax(attn_logits, dim=1)
        gate = torch.sigmoid(self.gate(x).squeeze(-1))
        weight = attn * gate
        weight = weight / (weight.sum(dim=1, keepdim=True) + 1e-8)
        return (weight.unsqueeze(-1) * x).sum(dim=1)


class _TorchAttentionFusion(_ModuleBase):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, x):  # noqa: ANN001
        attn_logits = self.attn(x).squeeze(-1)
        attn = torch.softmax(attn_logits, dim=1)
        return (attn.unsqueeze(-1) * x).sum(dim=1)


class _TorchGatingFusion(_ModuleBase):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, x):  # noqa: ANN001
        gate = torch.sigmoid(self.gate(x).squeeze(-1))
        weight = gate / (gate.sum(dim=1, keepdim=True) + 1e-8)
        return (weight.unsqueeze(-1) * x).sum(dim=1)


class _TorchConcatFusion(_ModuleBase):
    def __init__(self, dim: int, branches: int):
        super().__init__()
        self.dim = int(dim)
        self.branches = int(branches)
        self.out = nn.Sequential(
            nn.Linear(self.dim * self.branches, self.dim),
            nn.LayerNorm(self.dim),
            nn.GELU(),
        )

    def forward(self, x):  # noqa: ANN001
        b, n, d = x.shape
        if d != self.dim or n != self.branches:
            raise ValueError(
                f"Concat fusion input shape mismatch: got {(b, n, d)}, expected (*, {self.branches}, {self.dim})"
            )
        return self.out(x.reshape(b, n * d))


def build_torch_fusion_module(name: str, dim: int, branches: int):
    if torch is None or nn is None:
        raise RuntimeError(
            "PyTorch is required for trainable fusion modules."
            + (f" Import error: {_TORCH_IMPORT_ERROR}" if _TORCH_IMPORT_ERROR else "")
        )
    mode = str(name or "concat").strip().lower()
    if mode == "concat":
        return None
    if mode == "attention":
        return _TorchAttentionFusion(dim=int(dim))
    if mode == "gating":
        return _TorchGatingFusion(dim=int(dim))
    if mode in {"attention_gating", "gated_attention"}:
        return _TorchAttentionGatingFusion(dim=int(dim))
    raise ValueError(f"Unknown trainable fusion module: {mode}")


def create_fusion_module(cfg: Optional[Dict[str, Any]]):
    if not cfg or not isinstance(cfg, dict):
        return None
    name = str(cfg.get("name") or "").strip().lower()
    if not name:
        return None
    cls = FUSION_MODULE_REGISTRY.get(name)
    if cls is None:
        raise KeyError(f"Unknown fusion module: {name}")
    params = cfg.get("params") if isinstance(cfg.get("params"), dict) else {}
    return cls(params)
