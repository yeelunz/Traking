from __future__ import annotations

from typing import Any, Optional

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


def _require_runtime() -> None:
    if torch is None or nn is None:
        raise RuntimeError(
            "PyTorch is required for TexturePretrainModel."
            + (f" Import error: {_TORCH_IMPORT_ERROR}" if _TORCH_IMPORT_ERROR else "")
        )
    if timm is None:
        raise RuntimeError(
            "timm is required for TexturePretrainModel."
            + (f" Import error: {_TIMM_IMPORT_ERROR}" if _TIMM_IMPORT_ERROR else "")
        )


class TexturePretrainModel(nn.Module):
    """Stage-1 texture pretraining model: backbone + classifier head."""

    def __init__(
        self,
        backbone: str = "convnext_tiny",
        num_classes: int = 2,
        pretrained: bool = True,
        embedding_dim: int = 32,
        head_type: str = "linear",
        hidden_dim: int = 256,
        dropout: float = 0.2,
        input_size: int = 224,
    ):
        _require_runtime()
        super().__init__()

        self.backbone_name = str(backbone)
        self.num_classes = int(num_classes)
        self.input_size = int(input_size)
        self.embedding_dim = int(embedding_dim)

        self.backbone = timm.create_model(
            self.backbone_name,
            pretrained=bool(pretrained),
            num_classes=0,
            global_pool="avg",
        )

        feature_dim = int(getattr(self.backbone, "num_features", 0) or 0)
        if feature_dim <= 0:
            with torch.no_grad():
                dummy = torch.zeros((1, 3, self.input_size, self.input_size), dtype=torch.float32)
                feature_dim = int(self.backbone(dummy).shape[-1])
        self.feature_dim = int(feature_dim)

        self.proj = nn.Sequential(
            nn.Linear(self.feature_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.GELU(),
        )

        head = str(head_type).strip().lower()
        if head == "linear":
            self.head = nn.Linear(self.embedding_dim, self.num_classes)
        elif head == "mlp":
            self.head = nn.Sequential(
                nn.Linear(self.embedding_dim, int(hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(hidden_dim), self.num_classes),
            )
        else:
            raise ValueError("head_type must be 'linear' or 'mlp'.")

    def forward_features(self, x: Any) -> Any:
        return self.backbone(x)

    def forward_embedding(self, x: Any) -> Any:
        feat = self.forward_features(x)
        emb = self.proj(feat)
        return emb

    def forward(self, x: Any) -> Any:
        emb = self.forward_embedding(x)
        logits = self.head(emb)
        return logits

    def get_backbone_state_dict(self):
        return {k: v.detach().cpu() for k, v in self.backbone.state_dict().items()}

    def get_model_state_dict(self):
        return {k: v.detach().cpu() for k, v in self.state_dict().items()}
