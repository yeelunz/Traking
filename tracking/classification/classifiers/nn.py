from __future__ import annotations

import pickle
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as exc:  # noqa: BLE001
    raise ImportError(
        "Failed to import PyTorch dependencies for tracking.classification.classifiers.nn. "
        "Install torch."
    ) from exc
else:
    _TORCH_IMPORT_ERROR = None

try:  # pragma: no cover - optional dependency guard
    from sklearn.model_selection import train_test_split
except Exception as exc:  # noqa: BLE001
    raise ImportError(
        "Failed to import scikit-learn dependency train_test_split for "
        "tracking.classification.classifiers.nn. Install scikit-learn."
    ) from exc
else:
    _SKLEARN_IMPORT_ERROR = None

from ...core.registry import register_classifier
from ..interfaces import SubjectClassifier


def _require_torch() -> None:
    if torch is None or nn is None:
        msg = "PyTorch is required for MLP/Linear-head classifiers."
        if _TORCH_IMPORT_ERROR is not None:
            msg += f" Import error: {_TORCH_IMPORT_ERROR}"
        raise RuntimeError(msg)


def _resolve_device(pref: str) -> Any:
    _require_torch()
    pref = str(pref or "auto").lower()
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


_ModuleBase = nn.Module


class _MLPLinearNet(_ModuleBase):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        mode: str,
        hidden_dims: List[int],
        dropout: float,
    ):
        super().__init__()
        mode = str(mode).lower()
        if mode not in {"mlp", "linear"}:
            raise ValueError(f"Unsupported mode: {mode}. Expected 'mlp' or 'linear'.")

        if mode == "linear":
            self.net = nn.Linear(input_dim, num_classes)
            return

        dims = [int(input_dim)] + [int(d) for d in hidden_dims if int(d) > 0]
        layers: List[Any] = []
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                nn.GELU(),
            ])
            if float(dropout) > 0:
                layers.append(nn.Dropout(float(dropout)))
        layers.append(nn.Linear(dims[-1], num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Any) -> Any:
        return self.net(x)


@register_classifier("mlp_linear_head")
@register_classifier("mlp_head")
@register_classifier("mlp")
@register_classifier("linear_head")
class MLPLinearHeadClassifier(SubjectClassifier):
    """Differentiable tabular classifier with selectable MLP / linear head."""

    name = "MLPLinearHeadClassifier"
    DEFAULT_CONFIG: Dict[str, Any] = {
        "mode": "mlp",  # mlp | linear
        "hidden_dims": [256, 128],
        "dropout": 0.2,
        "epochs": 60,
        "batch_size": 64,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "val_ratio": 0.2,
        "patience": 10,
        "class_weight": "balanced",  # balanced | none
        "normalize": True,
        "seed": 42,
        "device": "auto",
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        _require_torch()
        cfg = {**self.DEFAULT_CONFIG, **(params or {})}
        self._cfg = cfg
        self._device = _resolve_device(str(cfg.get("device", "auto")))
        self._model: Optional[_MLPLinearNet] = None
        self.classes_: Optional[np.ndarray] = None
        self._state_dict: Optional[Dict[str, Any]] = None
        self._x_mean: Optional[np.ndarray] = None
        self._x_std: Optional[np.ndarray] = None
        self._input_dim: Optional[int] = None
        self._num_classes: Optional[int] = None

    def _normalize_fit(self, X: np.ndarray) -> np.ndarray:
        if not bool(self._cfg.get("normalize", True)):
            self._x_mean = None
            self._x_std = None
            return X
        mean = np.mean(X, axis=0, dtype=np.float64)
        std = np.std(X, axis=0, dtype=np.float64)
        std = np.where(std < 1e-8, 1.0, std)
        self._x_mean = mean.astype(np.float32)
        self._x_std = std.astype(np.float32)
        return ((X - self._x_mean) / self._x_std).astype(np.float32)

    def _normalize_transform(self, X: np.ndarray) -> np.ndarray:
        if self._x_mean is None or self._x_std is None:
            return X.astype(np.float32)
        return ((X - self._x_mean) / self._x_std).astype(np.float32)

    def _build_model(self, input_dim: int, num_classes: int) -> _MLPLinearNet:
        return _MLPLinearNet(
            input_dim=input_dim,
            num_classes=num_classes,
            mode=str(self._cfg.get("mode", "mlp")),
            hidden_dims=list(self._cfg.get("hidden_dims", [256, 128]) or []),
            dropout=float(self._cfg.get("dropout", 0.2)),
        )

    def fit(self, X, y) -> Dict[str, Any]:
        X = np.asarray(X, dtype=np.float32)
        y_raw = np.asarray(y, dtype=np.int64)
        if X.ndim != 2:
            raise ValueError("MLPLinearHeadClassifier expects 2D feature matrix.")
        if len(X) == 0:
            raise ValueError("No training data provided for classifier.")

        self.classes_ = np.unique(y_raw)
        if self.classes_.size < 2:
            raise ValueError("MLPLinearHeadClassifier requires at least two classes.")

        cls_to_idx = {int(c): i for i, c in enumerate(self.classes_.tolist())}
        y_idx = np.array([cls_to_idx[int(v)] for v in y_raw], dtype=np.int64)

        Xn = self._normalize_fit(X)
        self._input_dim = int(Xn.shape[1])
        self._num_classes = int(len(self.classes_))

        idx = np.arange(len(y_idx))
        val_ratio = float(self._cfg.get("val_ratio", 0.2))
        if 0.0 < val_ratio < 0.9 and len(y_idx) >= 10:
            tr_idx, va_idx = train_test_split(
                idx,
                test_size=val_ratio,
                random_state=int(self._cfg.get("seed", 42)),
                stratify=y_idx,
            )
        else:
            tr_idx, va_idx = idx, np.array([], dtype=np.int64)

        model = self._build_model(self._input_dim, self._num_classes).to(self._device)
        optim = torch.optim.AdamW(
            model.parameters(),
            lr=float(self._cfg.get("lr", 1e-3)),
            weight_decay=float(self._cfg.get("weight_decay", 1e-4)),
        )

        if str(self._cfg.get("class_weight", "balanced")).lower() == "balanced":
            cls_counts = np.bincount(y_idx, minlength=self._num_classes).astype(np.float64)
            cls_counts = np.where(cls_counts <= 0, 1.0, cls_counts)
            inv = 1.0 / cls_counts
            w = (inv / inv.sum()) * self._num_classes
            class_weight = torch.tensor(w, dtype=torch.float32, device=self._device)
            criterion = nn.CrossEntropyLoss(weight=class_weight)
        else:
            criterion = nn.CrossEntropyLoss()

        def _make_loader(sel: np.ndarray, shuffle: bool):
            xx = torch.tensor(Xn[sel], dtype=torch.float32)
            yy = torch.tensor(y_idx[sel], dtype=torch.long)
            ds = TensorDataset(xx, yy)
            return DataLoader(ds, batch_size=int(self._cfg.get("batch_size", 64)), shuffle=shuffle)

        train_loader = _make_loader(np.asarray(tr_idx, dtype=np.int64), shuffle=True)
        val_loader = _make_loader(np.asarray(va_idx, dtype=np.int64), shuffle=False) if va_idx.size > 0 else None

        best_state = None
        best_val = float("inf")
        patience = int(self._cfg.get("patience", 10))
        bad_epochs = 0
        trained_epochs = 0

        for _ in range(int(self._cfg.get("epochs", 60))):
            trained_epochs += 1
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(self._device)
                yb = yb.to(self._device)
                optim.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optim.step()

            if val_loader is None:
                continue

            model.eval()
            val_loss = 0.0
            total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self._device)
                    yb = yb.to(self._device)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    bs = int(yb.shape[0])
                    val_loss += float(loss.item()) * bs
                    total += bs
            val_loss = val_loss / max(total, 1)
            if val_loss + 1e-8 < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= max(1, patience):
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        self._model = model
        self._state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        return {
            "epochs_trained": trained_epochs,
            "input_dim": self._input_dim,
            "num_classes": self._num_classes,
            "mode": str(self._cfg.get("mode", "mlp")),
        }

    def _ensure_model(self) -> _MLPLinearNet:
        if self._model is not None:
            return self._model
        if self._state_dict is None or self._input_dim is None or self._num_classes is None:
            raise RuntimeError("Classifier is not fitted or loaded.")
        model = self._build_model(self._input_dim, self._num_classes)
        model.load_state_dict(self._state_dict)
        model.to(self._device)
        self._model = model
        return model

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("MLPLinearHeadClassifier expects 2D feature matrix.")
        Xn = self._normalize_transform(X)
        model = self._ensure_model()
        model.eval()
        with torch.no_grad():
            xb = torch.tensor(Xn, dtype=torch.float32, device=self._device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        pred_idx = np.argmax(probs, axis=1)
        if self.classes_ is None:
            return pred_idx.astype(np.int64)
        return self.classes_[pred_idx]

    def save(self, path: str) -> None:
        payload = {
            "cfg": self._cfg,
            "classes": self.classes_,
            "state_dict": self._state_dict,
            "x_mean": self._x_mean,
            "x_std": self._x_std,
            "input_dim": self._input_dim,
            "num_classes": self._num_classes,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self._cfg = payload.get("cfg", self._cfg)
        self._device = _resolve_device(str(self._cfg.get("device", "auto")))
        self.classes_ = payload.get("classes")
        self._state_dict = payload.get("state_dict")
        self._x_mean = payload.get("x_mean")
        self._x_std = payload.get("x_std")
        self._input_dim = payload.get("input_dim")
        self._num_classes = payload.get("num_classes")
        self._model = None
