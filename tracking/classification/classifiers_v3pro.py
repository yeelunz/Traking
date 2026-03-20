from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except Exception as exc:  # noqa: BLE001
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    DataLoader = None  # type: ignore
    TensorDataset = None  # type: ignore
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

from sklearn.model_selection import train_test_split

from ..core.registry import register_classifier
from .interfaces import SubjectClassifier
from .texture_backbone import TextureBackboneWrapper


def _require_runtime() -> None:
    if torch is None or nn is None or F is None:
        raise RuntimeError(
            "PyTorch is required for fusion_mlp classifier."
            + (f" Import error: {_TORCH_IMPORT_ERROR}" if _TORCH_IMPORT_ERROR else "")
        )


def _device_from_pref(pref: str) -> Any:
    _require_runtime()
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


class _AttentionGatingFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

    def forward(self, x: Any) -> Any:
        attn_logits = self.attn(x).squeeze(-1)
        attn = torch.softmax(attn_logits, dim=1)
        gate = torch.sigmoid(self.gate(x).squeeze(-1))
        weight = attn * gate
        weight = weight / (weight.sum(dim=1, keepdim=True) + 1e-8)
        fused = (weight.unsqueeze(-1) * x).sum(dim=1)
        return fused


class _V3ProFusionNet(nn.Module):
    def __init__(
        self,
        motion_dim: int,
        static_dim: int,
        texture_image_size: int,
        fusion_dim: int,
        fusion_mode: str,
        texture_mode: str,
        texture_backbone: str,
        texture_dim: int,
        texture_pretrain_ckpt: Optional[str],
        pretrained_backbone: bool,
        dropout: float,
        texture_input_mode: str,
    ):
        super().__init__()
        self.texture_image_size = int(texture_image_size)
        self.fusion_dim = int(fusion_dim)
        self.fusion_mode = str(fusion_mode)
        self.texture_input_mode = str(texture_input_mode)

        self.motion_proj = nn.Sequential(
            nn.Linear(motion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )
        self.static_proj = nn.Sequential(
            nn.Linear(static_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )
        if self.texture_input_mode == "image":
            self.texture_branch = TextureBackboneWrapper(
                mode=str(texture_mode),
                backbone_name=str(texture_backbone),
                texture_dim=int(texture_dim),
                image_size=self.texture_image_size,
                pretrain_ckpt=texture_pretrain_ckpt,
                pretrained_imagenet=bool(pretrained_backbone),
            )
        else:
            self.texture_branch = None

        if int(texture_dim) != self.fusion_dim or self.texture_input_mode != "image":
            in_dim = int(texture_dim)
            self.texture_align = nn.Sequential(
                nn.Linear(in_dim, self.fusion_dim),
                nn.LayerNorm(self.fusion_dim),
                nn.GELU(),
            )
        else:
            self.texture_align = nn.Identity()

        if self.fusion_mode == "concat":
            self.fusion = nn.Sequential(
                nn.Linear(fusion_dim * 3, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.GELU(),
            )
        elif self.fusion_mode == "attention_gating":
            self.fusion = _AttentionGatingFusion(fusion_dim)
        else:
            raise ValueError(f"Unsupported fusion mode: {self.fusion_mode}")

        self.head = nn.Sequential(
            nn.Dropout(float(dropout)),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(fusion_dim, 2),
        )

    def forward(
        self,
        x_motion: Any,
        x_static: Any,
        x_texture: Any,
    ) -> Any:
        m = self.motion_proj(x_motion)
        s = self.static_proj(x_static)
        if self.texture_input_mode == "image":
            assert self.texture_branch is not None
            t = self.texture_align(self.texture_branch(x_texture))
        else:
            t = self.texture_align(x_texture)

        if self.fusion_mode == "concat":
            z = self.fusion(torch.cat([m, s, t], dim=1))
        else:
            z = self.fusion(torch.stack([m, s, t], dim=1))

        return self.head(z)


@dataclass
class _ParsedInput:
    x_motion: np.ndarray
    x_static: np.ndarray
    x_texture: np.ndarray
    texture_input_mode: str


@register_classifier("fusion_gating_mlp")
@register_classifier("fusion_mlp")
@register_classifier("v3pro_fusion")
class FusionMLPClassifier(SubjectClassifier):
    name = "FusionMLPClassifier"
    DEFAULT_CONFIG: Dict[str, Any] = {
        "motion_dim": 13,
        "static_dim": 10,
        "texture_image_size": 96,
        "texture_mode": "freeze",  # freeze | learnable | pretrain
        "texture_backbone": "convnext_tiny",
        "texture_dim": 32,
        "texture_input_mode": "auto",  # auto | image | compact
        "texture_pretrain_ckpt": None,
        "fusion_mode": "attention_gating",
        "fusion_dim": 32,
        "pretrained_backbone": True,
        "epochs": 40,
        "batch_size": 16,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "dropout": 0.2,
        "val_ratio": 0.2,
        "patience": 8,
        "seed": 42,
        "device": "auto",
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        _require_runtime()
        cfg = {**self.DEFAULT_CONFIG, **(params or {})}

        if "texture_mode" not in cfg:
            backbone_trainable = bool(cfg.get("backbone_trainable", False))
            cfg["texture_mode"] = "learnable" if backbone_trainable else "freeze"

        if cfg.get("texture_dim") is None:
            cfg["texture_dim"] = int(cfg.get("fusion_dim", 32))

        self._cfg = cfg
        self._device = _device_from_pref(str(cfg.get("device", "auto")))
        self._model: Optional[_V3ProFusionNet] = None
        self.classes_: Optional[np.ndarray] = None
        self._state_dict: Optional[Dict[str, Any]] = None

    def _parse_X(self, X: np.ndarray) -> _ParsedInput:
        X = np.asarray(X, dtype=np.float32)
        motion_dim = int(self._cfg["motion_dim"])
        static_dim = int(self._cfg["static_dim"])
        tex_size = int(self._cfg["texture_image_size"])
        tex_img_dim = 3 * tex_size * tex_size
        tex_compact_dim = int(self._cfg.get("texture_dim", 32))

        mode_pref = str(self._cfg.get("texture_input_mode", "auto")).lower()
        expected_img = motion_dim + static_dim + tex_img_dim
        expected_compact = motion_dim + static_dim + tex_compact_dim

        if mode_pref == "image":
            texture_input_mode = "image"
            expected = expected_img
        elif mode_pref == "compact":
            texture_input_mode = "compact"
            expected = expected_compact
        else:
            if X.shape[1] == expected_compact:
                texture_input_mode = "compact"
                expected = expected_compact
            elif X.shape[1] == expected_img:
                texture_input_mode = "image"
                expected = expected_img
            else:
                raise ValueError(
                    "fusion_mlp input mismatch: "
                    f"got {X.shape[1]}, expected compact={expected_compact} or image={expected_img}"
                )

        if X.shape[1] != expected:
            raise ValueError(
                f"fusion_mlp expects {expected} features in mode={texture_input_mode}, got {X.shape[1]}"
            )

        start = 0
        x_motion = X[:, start : start + motion_dim]
        start += motion_dim
        x_static = X[:, start : start + static_dim]
        start += static_dim
        x_tex = X[:, start:]
        return _ParsedInput(
            x_motion=x_motion,
            x_static=x_static,
            x_texture=x_tex,
            texture_input_mode=texture_input_mode,
        )

    def _build_model(self, texture_input_mode: str) -> _V3ProFusionNet:
        return _V3ProFusionNet(
            motion_dim=int(self._cfg["motion_dim"]),
            static_dim=int(self._cfg["static_dim"]),
            texture_image_size=int(self._cfg["texture_image_size"]),
            fusion_dim=int(self._cfg["fusion_dim"]),
            fusion_mode=str(self._cfg["fusion_mode"]),
            texture_mode=str(self._cfg["texture_mode"]),
            texture_backbone=str(self._cfg.get("texture_backbone", "convnext_tiny")),
            texture_dim=int(self._cfg.get("texture_dim", self._cfg["fusion_dim"])),
            texture_pretrain_ckpt=self._cfg.get("texture_pretrain_ckpt"),
            pretrained_backbone=bool(self._cfg["pretrained_backbone"]),
            dropout=float(self._cfg["dropout"]),
            texture_input_mode=texture_input_mode,
        )

    def fit(self, X, y) -> Dict[str, Any]:
        parsed = self._parse_X(np.asarray(X, dtype=np.float32))
        y_arr = np.asarray(y, dtype=np.int64)
        if np.unique(y_arr).size < 2:
            raise ValueError("fusion_mlp requires at least 2 classes in training labels.")

        self.classes_ = np.unique(y_arr)
        if not np.array_equal(self.classes_, np.array([0, 1], dtype=np.int64)):
            mapping = {int(c): i for i, c in enumerate(self.classes_.tolist())}
            y_arr = np.array([mapping[int(v)] for v in y_arr], dtype=np.int64)

        idx = np.arange(len(y_arr))
        val_ratio = float(self._cfg.get("val_ratio", 0.2))
        if 0.0 < val_ratio < 0.9 and len(y_arr) >= 8:
            tr_idx, va_idx = train_test_split(
                idx,
                test_size=val_ratio,
                random_state=int(self._cfg.get("seed", 42)),
                stratify=y_arr,
            )
        else:
            tr_idx, va_idx = idx, np.array([], dtype=np.int64)

        self._cfg["texture_input_mode"] = parsed.texture_input_mode
        model = self._build_model(parsed.texture_input_mode).to(self._device)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(self._cfg["lr"]),
            weight_decay=float(self._cfg["weight_decay"]),
        )
        criterion = nn.CrossEntropyLoss()

        def _to_loader(sel_idx: np.ndarray, shuffle: bool):
            x_m = torch.tensor(parsed.x_motion[sel_idx], dtype=torch.float32)
            x_s = torch.tensor(parsed.x_static[sel_idx], dtype=torch.float32)
            if parsed.texture_input_mode == "image":
                x_t = torch.tensor(parsed.x_texture[sel_idx], dtype=torch.float32).view(
                    len(sel_idx), 3, int(self._cfg["texture_image_size"]), int(self._cfg["texture_image_size"])
                )
            else:
                x_t = torch.tensor(parsed.x_texture[sel_idx], dtype=torch.float32)
            yy = torch.tensor(y_arr[sel_idx], dtype=torch.long)
            ds = TensorDataset(x_m, x_s, x_t, yy)
            return DataLoader(ds, batch_size=int(self._cfg["batch_size"]), shuffle=shuffle)

        train_loader = _to_loader(np.asarray(tr_idx, dtype=np.int64), shuffle=True)
        val_loader = _to_loader(np.asarray(va_idx, dtype=np.int64), shuffle=False) if va_idx.size > 0 else None

        best_state = None
        best_val = float("inf")
        patience = int(self._cfg.get("patience", 8))
        bad_epochs = 0

        for _ in range(int(self._cfg["epochs"])):
            model.train()
            for x_m, x_s, x_t, yy in train_loader:
                x_m = x_m.to(self._device)
                x_s = x_s.to(self._device)
                x_t = x_t.to(self._device)
                yy = yy.to(self._device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x_m, x_s, x_t)
                loss = criterion(logits, yy)
                loss.backward()
                optimizer.step()

            if val_loader is None:
                continue

            model.eval()
            val_loss = 0.0
            total = 0
            with torch.no_grad():
                for x_m, x_s, x_t, yy in val_loader:
                    x_m = x_m.to(self._device)
                    x_s = x_s.to(self._device)
                    x_t = x_t.to(self._device)
                    yy = yy.to(self._device)
                    logits = model(x_m, x_s, x_t)
                    loss = criterion(logits, yy)
                    bs = yy.shape[0]
                    val_loss += float(loss.item()) * bs
                    total += bs
            val_loss = val_loss / max(1, total)

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        self._model = model.to(self._device).eval()
        self._state_dict = {k: v.detach().cpu() for k, v in self._model.state_dict().items()}
        return {
            "n_params": int(sum(p.numel() for p in self._model.parameters())),
            "fusion_mode": str(self._cfg["fusion_mode"]),
            "fusion_dim": int(self._cfg["fusion_dim"]),
            "texture_mode": str(self._cfg.get("texture_mode", "freeze")),
            "texture_backbone": str(self._cfg.get("texture_backbone", "convnext_tiny")),
            "texture_dim": int(self._cfg.get("texture_dim", self._cfg["fusion_dim"])),
        }

    def _forward_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model is not fitted.")
        parsed = self._parse_X(X)
        x_m = torch.tensor(parsed.x_motion, dtype=torch.float32, device=self._device)
        x_s = torch.tensor(parsed.x_static, dtype=torch.float32, device=self._device)
        if parsed.texture_input_mode == "image":
            x_t = torch.tensor(parsed.x_texture, dtype=torch.float32, device=self._device).view(
                X.shape[0], 3, int(self._cfg["texture_image_size"]), int(self._cfg["texture_image_size"])
            )
        else:
            x_t = torch.tensor(parsed.x_texture, dtype=torch.float32, device=self._device)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(x_m, x_s, x_t)
            proba = F.softmax(logits, dim=1).cpu().numpy()
        return proba

    def predict_proba(self, X):
        X_arr = np.asarray(X, dtype=np.float32)
        proba = self._forward_proba(X_arr)
        if self.classes_ is None:
            return proba
        if len(self.classes_) == 2 and np.array_equal(self.classes_, np.array([0, 1], dtype=np.int64)):
            return proba
        out = np.zeros((proba.shape[0], len(self.classes_)), dtype=np.float32)
        out[:, : proba.shape[1]] = proba
        return out

    def predict(self, X):
        proba = self.predict_proba(X)
        pred_idx = np.argmax(proba, axis=1)
        if self.classes_ is None:
            return pred_idx
        return self.classes_[pred_idx]

    def save(self, path: str) -> None:
        if self._model is None and self._state_dict is None:
            raise RuntimeError("Model is not fitted.")
        payload = {
            "cfg": self._cfg,
            "classes_": self.classes_,
            "state_dict": self._state_dict,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self._cfg = payload["cfg"]
        self.classes_ = payload.get("classes_")
        texture_input_mode = str(self._cfg.get("texture_input_mode", "image"))
        self._model = self._build_model(texture_input_mode).to(self._device)
        state = payload.get("state_dict")
        if state is not None:
            self._model.load_state_dict(state)
        self._state_dict = {k: v.detach().cpu() for k, v in self._model.state_dict().items()}
        self._model.eval()


# Backward-compatible symbol alias.
V3ProFusionClassifier = FusionMLPClassifier
