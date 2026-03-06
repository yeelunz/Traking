"""Extended classifiers for CTS subject-level classification.

提供以下新分類器（配合 classifiers.py 中已有的 RF / LightGBM / SVM）：

表格式（需 motion_texture_static 特徵）：
  - ``xgboost``   : XGBoost gradient-boosted trees
  - ``tabpfn_v2`` : TabPFN v2（in-context learning；小樣本高效能）

時序式（需 time_series 特徵提取器）：
  - ``multirocket``  : MultiRocket — 官方 numba 實作包裝
  - ``patchtst``     : PatchTST  — 忠實 PyTorch 實作（多頭注意力 + RevIN + 多層 Encoder）
  - ``timemachine``  : TimeMachine — 忠實 PyTorch 實作（4 交錯 Mamba + 殘差 + RevIN）

版本說明：
  MultiRocket 直接包裝 libs/MultiRocket 官方 numba 程式碼（Tan et al., 2022）。
  PatchTST 按照 Nie et al. (2023) 官方 repo 架構，改寫為分類模型。
  TimeMachine 按照 Ahamed & Cheng (2024) 官方 repo 架構，改寫為分類模型。
"""
from __future__ import annotations

import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Module-level progress logger injected by engine.py before classifier.fit().
# Defaults to print so standalone usage still produces output.
_progress_log_fn = print


def set_progress_logger(fn) -> None:
    """Set the progress logging callable used during classifier training.

    Call this with the runner's ``logger`` before ``classifier.fit()`` so that
    epoch messages appear in the UI.  Pass ``None`` to revert to ``print``.
    """
    global _progress_log_fn
    _progress_log_fn = fn if fn is not None else print

# ═════════════════════════════════════════════════════════════════════════════
# Dependency checks
# ═════════════════════════════════════════════════════════════════════════════

# ── sklearn ──────────────────────────────────────────────────────────────────
try:
    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_OK = True
except Exception:
    RidgeClassifierCV = None  # type: ignore[assignment,misc]
    StandardScaler = None  # type: ignore[assignment,misc]
    _SKLEARN_OK = False

# ── XGBoost ──────────────────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier  # type: ignore[import-not-found]
    _XGB_OK = True
except Exception:
    XGBClassifier = None  # type: ignore[assignment,misc]
    _XGB_OK = False

# ── TabPFN ───────────────────────────────────────────────────────────────────
try:
    from tabpfn import TabPFNClassifier  # type: ignore[import-not-found]
    _TABPFN_OK = True
except Exception:
    TabPFNClassifier = None  # type: ignore[assignment,misc]
    _TABPFN_OK = False

# ── PyTorch ──────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_OK = True
except Exception:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_OK = False

# ── Official MultiRocket (numba JIT) from libs/ ─────────────────────────────
_LIBS_DIR = Path(__file__).resolve().parent.parent.parent / "libs"
_MR_DIR = str(_LIBS_DIR / "MultiRocket")
_MULTIROCKET_OK = False
_mr_fit: Any = None
_mr_transform: Any = None
try:
    if _MR_DIR not in sys.path:
        sys.path.insert(0, _MR_DIR)
    from multirocket.multirocket_multivariate import (  # type: ignore[import-not-found]
        fit as _mr_fit,
        transform as _mr_transform,
    )
    _MULTIROCKET_OK = True
except Exception as _mr_err:
    logger.warning("Cannot import official MultiRocket from libs: %s", _mr_err)

# ── Mamba SSM ────────────────────────────────────────────────────────────────
_MAMBA_OK = False
_MambaBlock: Any = None
try:
    from mamba_ssm import Mamba as _MambaBlock  # type: ignore[import-not-found]
    _MAMBA_OK = True
except Exception:
    pass

# ── Internal registry & base class ──────────────────────────────────────────
from ..core.registry import register_classifier
from .classifiers import _PickleSubjectClassifier
from .feature_extractors_ext import N_TS_VARS, N_TS_STEPS

# Base class for torch nn.Module (safe when torch unavailable)
_TorchModule: type = nn.Module if _TORCH_OK else object  # type: ignore[assignment]

# ═════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═════════════════════════════════════════════════════════════════════════════


def _to_ts3d(
    X: np.ndarray,
    n_vars: int = N_TS_VARS,
    n_steps: int = N_TS_STEPS,
) -> np.ndarray:
    """Reshape flat (n_samples, n_vars*n_steps) → (n_samples, n_vars, n_steps)."""
    n = X.shape[0]
    expected = n_vars * n_steps
    if X.shape[1] < expected:
        pad = np.zeros((n, expected - X.shape[1]), dtype=X.dtype)
        X = np.hstack([X, pad])
    return X[:, :expected].reshape(n, n_vars, n_steps)


def _get_torch_device(preference: str = "auto") -> "torch.device":
    """Select compute device: 'auto' picks CUDA if available."""
    if not _TORCH_OK:
        raise RuntimeError("PyTorch is required.")
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def _train_torch_classifier(
    model: "nn.Module",
    X_3d: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int = 80,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    batch_size: int = 64,
    patience: int = 15,
    device: Optional["torch.device"] = None,
) -> "nn.Module":
    """Train a PyTorch classification model with early stopping.

    - Class-balanced CrossEntropyLoss
    - AdamW optimizer with cosine annealing
    - Gradient clipping (max norm 1.0)
    """
    if device is None:
        device = _get_torch_device()

    model = model.to(device)
    model.train()

    X_t = torch.tensor(X_3d, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)

    # Class-balanced weights
    classes, counts = np.unique(y, return_counts=True)
    w = 1.0 / np.clip(counts.astype(np.float64), 1, None)
    w = w / w.sum() * len(classes)
    class_w = torch.tensor(w, dtype=torch.float32, device=device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")
    best_state: Optional[Dict[str, Any]] = None
    no_improve = 0
    n = len(y)
    _log_every = max(1, epochs // 10)  # print ~10 lines per training run

    for _epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            logits = model(X_t[idx])
            loss = criterion(logits, y_t[idx])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg = epoch_loss / max(n_batches, 1)

        is_best = avg < best_loss - 1e-6
        if (_epoch + 1) % _log_every == 0 or _epoch == 0:
            _progress_log_fn(
                f"[Classification] Epoch {_epoch + 1}/{epochs} loss={avg:.4f}"
                + (" *" if is_best else "")
            )

        if is_best:
            best_loss = avg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            _progress_log_fn(
                f"[Classification] Early stopping at epoch {_epoch + 1}/{epochs}"
                f" (patience={patience}, best_loss={best_loss:.4f})"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


# ═════════════════════════════════════════════════════════════════════════════
# RevIN — Reversible Instance Normalization (Kim et al., 2021)
# ═════════════════════════════════════════════════════════════════════════════


class _RevIN(_TorchModule):
    """RevIN (from official PatchTST/TimeMachine repos).

    Input shape: (batch, seq_len, channels).
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        self.mean: Optional[torch.Tensor] = None
        self.stdev: Optional[torch.Tensor] = None

    def forward(self, x: "torch.Tensor", mode: str) -> "torch.Tensor":
        if mode == "norm":
            dim2reduce = tuple(range(1, x.ndim - 1))
            self.mean = x.mean(dim=dim2reduce, keepdim=True).detach()
            self.stdev = torch.sqrt(
                x.var(dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
            ).detach()
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.weight + self.bias
        elif mode == "denorm":
            if self.affine:
                x = (x - self.bias) / (self.weight + self.eps * self.eps)
            x = x * self.stdev + self.mean
        return x


# ═════════════════════════════════════════════════════════════════════════════
# SimpleMamba — fallback when mamba_ssm is not available
# ═════════════════════════════════════════════════════════════════════════════


class _SimpleMamba(_TorchModule):
    """Lightweight selective SSM block faithfully mimicking Mamba (Gu & Dao, 2023).

    Architecture:
      1. Input projection → (x_proj, z_gate), both of size d_inner
      2. Causal depthwise 1D-conv on x_proj for local context
      3. SiLU activation
      4. Data-dependent (selective) SSM: learnable A (diagonal), projected B/C/dt
      5. Sequential scan: h_t = discretize(A,dt)*h_{t-1} + discretize(B,dt)*x_t
      6. D feed-through skip + SiLU gate with z
      7. Output projection → d_model

    This captures all essential Mamba mechanisms without the CUDA-optimized kernels.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 1):
        super().__init__()
        d_inner = max(d_model * expand, 1)
        self.d_inner = d_inner
        self.d_state = d_state

        # Input projection: d_model → 2*d_inner (x and gating z)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)

        # Causal depthwise conv: padding on left only (trim right after)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, d_conv, padding=d_conv - 1, groups=d_inner,
        )

        # SSM parameter projections
        self.x_proj = nn.Linear(d_inner, d_state * 2, bias=False)  # → B, C
        self.dt_proj = nn.Linear(d_inner, d_inner, bias=True)       # → dt

        # Learnable SSM state matrix (log-space for stability)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A.repeat(d_inner, 1)))  # (d_inner, d_state)

        # Feed-through (D) skip connection
        self.D = nn.Parameter(torch.ones(d_inner))

        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (batch, seq_len, d_model)
        batch, seq_len, _ = x.shape

        # 1. Project
        xz = self.in_proj(x)                        # (B, L, 2*d_inner)
        x_proj, z = xz.chunk(2, dim=-1)             # each (B, L, d_inner)

        # 2. Causal conv
        x_conv = x_proj.permute(0, 2, 1)            # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # trim for causality
        x_conv = x_conv.permute(0, 2, 1)            # (B, L, d_inner)
        x_conv = F.silu(x_conv)

        # 3. SSM parameters (data-dependent → selective)
        A = -torch.exp(self.A_log)                   # (d_inner, d_state), negative for stability
        bc = self.x_proj(x_conv)                     # (B, L, 2*d_state)
        B, C = bc.split(self.d_state, dim=-1)        # each (B, L, d_state)
        dt = F.softplus(self.dt_proj(x_conv))        # (B, L, d_inner), positive

        # 4. Discretize: dA = exp(dt * A), dB = dt * B
        dA = torch.exp(dt.unsqueeze(-1) * A)         # (B, L, d_inner, d_state)
        dB = dt.unsqueeze(-1) * B.unsqueeze(2)       # (B, L, d_inner, d_state)

        # 5. Sequential scan
        h = torch.zeros(batch, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(seq_len):
            h = dA[:, t] * h + dB[:, t] * x_conv[:, t].unsqueeze(-1)
            y_t = (h * C[:, t].unsqueeze(1)).sum(-1)  # (B, d_inner)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)                    # (B, L, d_inner)

        # 6. D skip + gate
        y = y + x_conv * self.D
        y = y * F.silu(z)

        # 7. Output
        return self.out_proj(y)                       # (B, L, d_model)


def _make_mamba(d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 1) -> "nn.Module":
    """Create a Mamba block: official mamba_ssm if available, else _SimpleMamba."""
    if _MAMBA_OK and _MambaBlock is not None:
        return _MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    if _TORCH_OK:
        return _SimpleMamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    raise RuntimeError("PyTorch is required for Mamba / TimeMachine classifier.")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  XGBoost                                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


@register_classifier("xgboost")
class XGBoostSubjectClassifier(_PickleSubjectClassifier):
    """XGBoost gradient-boosted trees for tabular CTS features."""

    name = "XGBoostClassifier"
    DEFAULT_CONFIG: Dict[str, Any] = {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.0,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "n_jobs": -1,
        "random_state": 42,
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        if not _XGB_OK:
            raise RuntimeError(
                "xgboost is required for XGBoostSubjectClassifier. "
                "Install via: pip install xgboost"
            )
        cfg = {**self.DEFAULT_CONFIG, **(params or {})}
        self._model = XGBClassifier(
            n_estimators=int(cfg["n_estimators"]),
            max_depth=int(cfg["max_depth"]),
            learning_rate=float(cfg["learning_rate"]),
            subsample=float(cfg["subsample"]),
            colsample_bytree=float(cfg["colsample_bytree"]),
            gamma=float(cfg["gamma"]),
            reg_lambda=float(cfg["reg_lambda"]),
            reg_alpha=float(cfg["reg_alpha"]),
            use_label_encoder=cfg.get("use_label_encoder", False),
            eval_metric=str(cfg.get("eval_metric", "logloss")),
            n_jobs=int(cfg.get("n_jobs", -1)),
            random_state=int(cfg.get("random_state", 42)),
        )
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X, y) -> Dict[str, Any]:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        self._model.fit(X, y, verbose=False)
        self.classes_ = getattr(self._model, "classes_", None)
        return {"feature_importances": self._model.feature_importances_.tolist()}

    def predict(self, X):
        return self._model.predict(np.asarray(X, dtype=np.float32))

    def predict_proba(self, X):
        return self._model.predict_proba(np.asarray(X, dtype=np.float32))


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  TabPFN v2                                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


@register_classifier("tabpfn_v2")
class TabPFNV2SubjectClassifier(_PickleSubjectClassifier):
    """TabPFN v2: in-context learning for small tabular classification."""

    name = "TabPFNv2"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        if not _TABPFN_OK:
            raise RuntimeError(
                "tabpfn is required for TabPFNV2SubjectClassifier. "
                "Install via: pip install tabpfn"
            )
        cfg = params or {}
        n_ens = int(cfg.get("n_ensemble_configurations", 32))
        kwargs: Dict[str, Any] = {"N_ensemble_configurations": n_ens}
        device = cfg.get("device", "cpu")
        if device:
            kwargs["device"] = str(device)
        self._model = TabPFNClassifier(**kwargs)
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X, y) -> Dict[str, Any]:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        self._model.fit(X, y)
        self.classes_ = np.unique(y)
        return {"n_train_samples": len(y)}

    def predict(self, X):
        return self._model.predict(np.asarray(X, dtype=np.float32))

    def predict_proba(self, X):
        return self._model.predict_proba(np.asarray(X, dtype=np.float32))

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self._model, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self._model = pickle.load(f)
        self.classes_ = None


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  MultiRocket — Official implementation wrapper                          ║
# ║  Tan et al. (2022): Multiple pooling operators and transformations for  ║
# ║  fast and effective time series classification                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


@register_classifier("multirocket")
class MultiRocketSubjectClassifier(_PickleSubjectClassifier):
    """MultiRocket (official numba 實作包裝).

    演算法（忠實於 Tan et al., 2022）：
      1. 固定 84 個核心 C(9,3)，搭配 log-spaced dilations
      2. α/γ 公式卷積（非隨機權重）：C = C_α + C_γ[i₀] + C_γ[i₁] + C_γ[i₂]
      3. 以訓練資料卷積輸出的 quantile 作為 bias（data-fitted）
      4. 4 種池化算子：PPV, LSPV (longest stretch), MPV, MIPV
      5. 雙變換：base series + first-order difference（特徵量加倍）
      6. RidgeClassifierCV（交叉驗證正則化 α）

    需要: numba, scikit-learn, libs/MultiRocket (已 clone)
    """

    name = "MultiRocket"
    DEFAULT_CONFIG: Dict[str, Any] = {
        "num_features": 50_000,
        "max_dilations_per_kernel": 32,
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        if not _MULTIROCKET_OK:
            raise RuntimeError(
                "Official MultiRocket (numba) is not available. "
                "Ensure `numba` is installed and libs/MultiRocket is present. "
                f"Searched at: {_MR_DIR}"
            )
        if not _SKLEARN_OK:
            raise RuntimeError("scikit-learn is required for MultiRocket.")

        cfg = {**self.DEFAULT_CONFIG, **(params or {})}
        total = int(cfg["num_features"])
        self._n_features_per_kernel = 4  # PPV, LSPV, MPV, MIPV (official)
        self._num_kernels = int(total / 2 / self._n_features_per_kernel)
        self._max_dilations = int(cfg["max_dilations_per_kernel"])

        self._base_params: Any = None
        self._diff_params: Any = None
        self._scaler = StandardScaler()
        self._clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        self.classes_: Optional[np.ndarray] = None

    # ── internal ─────────────────────────────────────────────────────────────

    @staticmethod
    def _ensure_min_length(X_3d: np.ndarray, min_len: int = 10) -> np.ndarray:
        """Pad very short series (official MultiRocket requires length ≥ 10)."""
        if X_3d.shape[2] < min_len:
            padded = np.zeros(
                (X_3d.shape[0], X_3d.shape[1], min_len), dtype=X_3d.dtype
            )
            padded[:, :, : X_3d.shape[2]] = X_3d
            return padded
        return X_3d

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        X_3d = _to_ts3d(np.asarray(X, dtype=np.float32)).astype(np.float64)
        return self._ensure_min_length(X_3d)

    def _transform(self, X_3d: np.ndarray) -> np.ndarray:
        X_diff = np.diff(X_3d, axis=2)
        features = _mr_transform(
            X_3d, X_diff,
            self._base_params, self._diff_params,
            self._n_features_per_kernel,
        )
        return np.nan_to_num(features).astype(np.float32)

    # ── public interface ─────────────────────────────────────────────────────

    def fit(self, X, y) -> Dict[str, Any]:
        X_3d = self._preprocess(X)
        y = np.asarray(y, dtype=np.int64)

        X_diff = np.diff(X_3d, axis=2)

        self._base_params = _mr_fit(
            X_3d,
            num_features=self._num_kernels,
            max_dilations_per_kernel=self._max_dilations,
        )
        self._diff_params = _mr_fit(
            X_diff,
            num_features=self._num_kernels,
            max_dilations_per_kernel=self._max_dilations,
        )

        X_feat = self._transform(X_3d)
        X_feat = self._scaler.fit_transform(X_feat)
        self._clf.fit(X_feat, y)
        self.classes_ = np.unique(y)

        return {
            "n_features": X_feat.shape[1],
            "best_alpha": float(self._clf.alpha_),
        }

    def predict(self, X):
        X_3d = self._preprocess(X)
        X_feat = self._scaler.transform(self._transform(X_3d))
        return self._clf.predict(X_feat)

    def predict_proba(self, X):
        X_3d = self._preprocess(X)
        X_feat = self._scaler.transform(self._transform(X_3d))
        dec = self._clf.decision_function(X_feat)
        # RidgeClassifierCV has no predict_proba → sigmoid on decision function
        if dec.ndim == 1:
            p = 1.0 / (1.0 + np.exp(-np.clip(dec, -500, 500)))
            return np.column_stack([1 - p, p]).astype(np.float32)
        exp_d = np.exp(dec - dec.max(axis=1, keepdims=True))
        return (exp_d / exp_d.sum(axis=1, keepdims=True)).astype(np.float32)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "base_params": self._base_params,
                    "diff_params": self._diff_params,
                    "scaler": self._scaler,
                    "clf": self._clf,
                    "classes_": self.classes_,
                    "num_kernels": self._num_kernels,
                    "n_features_per_kernel": self._n_features_per_kernel,
                    "max_dilations": self._max_dilations,
                },
                f,
            )

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._base_params = state["base_params"]
        self._diff_params = state["diff_params"]
        self._scaler = state["scaler"]
        self._clf = state["clf"]
        self.classes_ = state.get("classes_")
        self._num_kernels = state.get("num_kernels", self._num_kernels)
        self._n_features_per_kernel = state.get("n_features_per_kernel", 4)
        self._max_dilations = state.get("max_dilations", 32)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PatchTST — Faithful PyTorch implementation for classification          ║
# ║  Nie et al. (2023): A Time Series is Worth 64 Words                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


class _PatchTSTClassification(_TorchModule):
    """PatchTST backbone adapted for time-series classification.

    Architecture (faithful to official repo):
      1. RevIN normalization (per channel)
      2. Patching: unfold with (patch_len, stride)
      3. **Channel-independent** processing: reshape (bs, c_in, n_patches, patch_len)
         → (bs*c_in, n_patches, patch_len)
      4. Linear projection: patch_len → d_model  (Eq. 1 in paper)
      5. Learnable positional encoding (additive)
      6. N-layer Transformer encoder:
           Multi-head self-attention + Add&Norm + FFN + Add&Norm
           (post-norm, GELU activation, residual connections)
      7. Mean pooling over patches → (bs*c_in, d_model)
      8. Reshape → (bs, c_in * d_model) → classification head
    """

    def __init__(
        self,
        c_in: int = N_TS_VARS,
        seq_len: int = N_TS_STEPS,
        patch_len: int = 16,
        stride: int = 8,
        n_layers: int = 2,
        d_model: int = 32,
        n_heads: int = 4,
        d_ff: int = 64,
        dropout: float = 0.3,
        n_classes: int = 2,
    ):
        super().__init__()
        self.c_in = c_in
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride

        # RevIN
        self.revin = _RevIN(c_in, affine=True)

        # Patching geometry
        self.n_patches = int((seq_len - patch_len) / stride + 1)

        # Linear projection: patch_len → d_model  (W_P in official code)
        self.W_P = nn.Linear(patch_len, d_model)

        # Learnable positional encoding  (W_pos in official code)
        self.W_pos = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.uniform_(self.W_pos, -0.02, 0.02)

        # Residual dropout
        self.input_dropout = nn.Dropout(dropout)

        # Transformer encoder (multi-head attention + FFN + residual + post-norm)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(c_in * d_model),
            nn.Dropout(dropout),
            nn.Linear(c_in * d_model, n_classes),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (bs, c_in, seq_len)
        bs = x.shape[0]

        # 1. RevIN (expects bs, seq_len, c_in)
        x = x.permute(0, 2, 1)
        x = self.revin(x, "norm")
        x = x.permute(0, 2, 1)                              # (bs, c_in, seq_len)

        # 2. Patching
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # → (bs, c_in, n_patches, patch_len)

        # 3. Channel-independent
        x = x.reshape(bs * self.c_in, self.n_patches, self.patch_len)

        # 4. Patch embedding
        x = self.W_P(x)                                     # (bs*c_in, n_patches, d_model)

        # 5. Positional encoding + dropout
        x = self.input_dropout(x + self.W_pos)

        # 6. Transformer encoder
        x = self.encoder(x)                                  # (bs*c_in, n_patches, d_model)

        # 7. Mean pooling over patches
        x = x.mean(dim=1)                                    # (bs*c_in, d_model)

        # 8. Reshape and classify
        x = x.reshape(bs, -1)                                # (bs, c_in*d_model)
        return self.head(x)                                   # (bs, n_classes)


@register_classifier("patchtst")
class PatchTSTSubjectClassifier(_PickleSubjectClassifier):
    """PatchTST: A Time Series is Worth 64 Words (Nie et al., 2023).

    忠實 PyTorch 分類版實作，依照官方 repo 架構：
      - Channel-independent patch 嵌入
      - 多頭自注意力 + FFN + 殘差連接 + LayerNorm (多層堆疊)
      - RevIN (Reversible Instance Normalization)
      - 可學習位置編碼
      - 分類頭: mean pooling → LayerNorm → Dropout → Linear

    Input X 來自 ``time_series`` 特徵提取器（flat 2304-dim = 18 ch × 128 steps）。
    需要: torch
    """

    name = "PatchTSTClassifier"
    DEFAULT_CONFIG: Dict[str, Any] = {
        "patch_len": 16,
        "stride": 8,
        "d_model": 32,
        "n_heads": 4,
        "d_ff": 64,
        "n_layers": 2,
        "dropout": 0.3,
        "epochs": 80,
        "lr": 1e-3,
        "weight_decay": 1e-3,
        "batch_size": 64,
        "patience": 15,
        "seed": 42,
        "device": "auto",
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        if not _TORCH_OK:
            raise RuntimeError("PyTorch is required for PatchTST classifier.")
        cfg = {**self.DEFAULT_CONFIG, **(params or {})}
        self._cfg = cfg
        self._seed = int(cfg["seed"])
        self._model: Optional[_PatchTSTClassification] = None
        self._device = _get_torch_device(str(cfg.get("device", "auto")))
        self.classes_: Optional[np.ndarray] = None

    def _build_model(self, n_classes: int = 2) -> _PatchTSTClassification:
        torch.manual_seed(self._seed)
        return _PatchTSTClassification(
            c_in=N_TS_VARS,
            seq_len=N_TS_STEPS,
            patch_len=int(self._cfg["patch_len"]),
            stride=int(self._cfg["stride"]),
            n_layers=int(self._cfg["n_layers"]),
            d_model=int(self._cfg["d_model"]),
            n_heads=int(self._cfg["n_heads"]),
            d_ff=int(self._cfg["d_ff"]),
            dropout=float(self._cfg["dropout"]),
            n_classes=n_classes,
        )

    def fit(self, X, y) -> Dict[str, Any]:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        X_3d = _to_ts3d(X)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        self._model = self._build_model(n_classes)
        self._model = _train_torch_classifier(
            self._model,
            X_3d,
            y,
            epochs=int(self._cfg["epochs"]),
            lr=float(self._cfg["lr"]),
            weight_decay=float(self._cfg["weight_decay"]),
            batch_size=int(self._cfg["batch_size"]),
            patience=int(self._cfg["patience"]),
            device=self._device,
        )
        return {"n_params": sum(p.numel() for p in self._model.parameters())}

    def predict(self, X):
        X_3d = _to_ts3d(np.asarray(X, dtype=np.float32))
        X_t = torch.tensor(X_3d, dtype=torch.float32, device=self._device)
        self._model.to(self._device).eval()
        with torch.no_grad():
            logits = self._model(X_t)
        return self.classes_[logits.argmax(dim=1).cpu().numpy()]

    def predict_proba(self, X):
        X_3d = _to_ts3d(np.asarray(X, dtype=np.float32))
        X_t = torch.tensor(X_3d, dtype=torch.float32, device=self._device)
        self._model.to(self._device).eval()
        with torch.no_grad():
            logits = self._model(X_t)
        return F.softmax(logits, dim=1).cpu().numpy()

    def save(self, path: str) -> None:
        state = {
            "model_state": (
                {k: v.cpu() for k, v in self._model.state_dict().items()}
                if self._model
                else None
            ),
            "cfg": self._cfg,
            "classes_": self.classes_,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._cfg = state["cfg"]
        self.classes_ = state.get("classes_")
        n_classes = len(self.classes_) if self.classes_ is not None else 2
        self._model = self._build_model(n_classes)
        if state.get("model_state"):
            self._model.load_state_dict(state["model_state"])
        self._model.to(self._device).eval()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  TimeMachine — Faithful PyTorch implementation for classification       ║
# ║  Ahamed & Cheng (2024): A Time Series is Worth 4 Mambas for            ║
# ║  Long-term Forecasting                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


class _TimeMachineClassification(_TorchModule):
    """TimeMachine architecture adapted for time-series classification.

    Architecture (faithful to official repo):
      1. RevIN normalization
      2. Channel-independent reshaping: (bs, c_in, L) → (bs*c_in, 1, L)
      3. Stage 1:
           lin1(seq_len → n1) → dropout
           mamba3: temporal SSM on (…, 1, n1), d_model=n1
           mamba4: cross-dim SSM on permuted (…, n1, 1), d_model=d_param2
           x4 = mamba4_output + mamba3_output
      4. Stage 2:
           lin2(n1 → n2) → dropout
           mamba1: cross-dim SSM on permuted (…, n2, 1), d_model=d_param1
           mamba2: temporal SSM on (…, 1, n2), d_model=n2
           x = mamba1_output + x_res2 + mamba2_output  (residual)
      5. lin3(n2 → n1) + x_res1  (residual)
         cat([x, x4]) → (…, 2*n1)
      6. Pool over channels → classification head
    """

    def __init__(
        self,
        c_in: int = N_TS_VARS,
        seq_len: int = N_TS_STEPS,
        n1: int = 64,
        n2: int = 16,
        d_state: int = 16,
        d_conv: int = 2,
        expand: int = 1,
        dropout: float = 0.1,
        ch_ind: bool = True,
        residual: bool = True,
        n_classes: int = 2,
    ):
        super().__init__()
        self.c_in = c_in
        self.seq_len = seq_len
        self.ch_ind = ch_ind
        self.residual = residual
        self.n1 = n1
        self.n2 = n2

        # RevIN
        self.revin = _RevIN(c_in, affine=True)

        # d_model for cross-dim Mamba blocks depends on ch_ind
        d_param1 = 1 if ch_ind else n2
        d_param2 = 1 if ch_ind else n1

        # Linear projections (official: operates on last dim)
        self.lin1 = nn.Linear(seq_len, n1)
        self.dropout1 = nn.Dropout(dropout)
        self.lin2 = nn.Linear(n1, n2)
        self.dropout2 = nn.Dropout(dropout)
        self.lin3 = nn.Linear(n2, n1)

        # 4 Mamba blocks — exact placement following official code
        self.mamba1 = _make_mamba(d_param1, d_state, d_conv, expand)  # cross-dim @ stage2
        self.mamba2 = _make_mamba(n2, d_state, d_conv, expand)        # temporal  @ stage2
        self.mamba3 = _make_mamba(n1, d_state, d_conv, expand)        # temporal  @ stage1
        self.mamba4 = _make_mamba(d_param2, d_state, d_conv, expand)  # cross-dim @ stage1

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(2 * n1),
            nn.Dropout(dropout),
            nn.Linear(2 * n1, n_classes),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (bs, c_in, seq_len)
        bs = x.shape[0]

        # RevIN: (bs, seq_len, c_in)
        x = x.permute(0, 2, 1)
        x = self.revin(x, "norm")
        x = x.permute(0, 2, 1)                              # (bs, c_in, seq_len)

        # Channel-independent
        if self.ch_ind:
            x = x.reshape(bs * self.c_in, 1, self.seq_len)  # (bs*C, 1, L)

        # ── Stage 1 ──────────────────────────────────────────────────────────
        x = self.lin1(x)                                     # (…, 1, n1)
        x_res1 = x
        x = self.dropout1(x)

        # mamba3: temporal SSM → d_model=n1
        x3 = self.mamba3(x)                                  # (…, 1, n1)

        # mamba4: cross-dim SSM → permute so d_model=d_param2
        if self.ch_ind:
            x4 = x.permute(0, 2, 1)                          # (…, n1, 1)
        else:
            x4 = x
        x4 = self.mamba4(x4)
        if self.ch_ind:
            x4 = x4.permute(0, 2, 1)                         # (…, 1, n1)

        x4 = x4 + x3                                         # residual within stage1

        # ── Stage 2 ──────────────────────────────────────────────────────────
        x = self.lin2(x)                                      # (…, 1, n2)
        x_res2 = x
        x = self.dropout2(x)

        # mamba1: cross-dim SSM → permute so d_model=d_param1
        if self.ch_ind:
            x1 = x.permute(0, 2, 1)                          # (…, n2, 1)
        else:
            x1 = x
        x1 = self.mamba1(x1)
        if self.ch_ind:
            x1 = x1.permute(0, 2, 1)                         # (…, 1, n2)

        # mamba2: temporal SSM → d_model=n2
        x2 = self.mamba2(x)                                   # (…, 1, n2)

        if self.residual:
            x = x1 + x_res2 + x2
        else:
            x = x1 + x2

        x = self.lin3(x)                                      # (…, 1, n1)
        if self.residual:
            x = x + x_res1

        x = torch.cat([x, x4], dim=2)                        # (…, 1, 2*n1)

        # ── Classification: pool ─────────────────────────────────────────────
        if self.ch_ind:
            x = x.squeeze(1)                                  # (bs*C, 2*n1)
            x = x.reshape(bs, self.c_in, -1)                 # (bs, C, 2*n1)
            x = x.mean(dim=1)                                 # (bs, 2*n1)
        else:
            x = x.mean(dim=1)                                 # (bs, 2*n1)

        return self.head(x)                                    # (bs, n_classes)


@register_classifier("timemachine")
class TimeMachineSubjectClassifier(_PickleSubjectClassifier):
    """TimeMachine: A Time Series is Worth 4 Mambas (Ahamed & Cheng, 2024).

    忠實 PyTorch 分類版實作，依照官方 repo 架構：
      - RevIN + channel-independent 處理
      - 2-stage 線性壓縮/擴展 (seq_len → n1 → n2 → n1)
      - 4 個交錯 Mamba blocks:
          mamba3 / mamba2: 時序維度 SSM (d_model = n1 / n2)
          mamba4 / mamba1: 跨維度 SSM (permuted, d_model = 1 with ch_ind)
      - 殘差連接 (x_res1, x_res2)
      - 分類頭: mean pool → LayerNorm → Dropout → Linear

    使用 mamba_ssm.Mamba（若有安裝），否則自動回退至 _SimpleMamba。
    Input X 來自 ``time_series`` 特徵提取器（flat 2304-dim = 18 ch × 128 steps）。
    需要: torch (+ 可選: mamba_ssm)
    """

    name = "TimeMachineClassifier"
    DEFAULT_CONFIG: Dict[str, Any] = {
        "n1": 64,
        "n2": 16,
        "d_state": 16,
        "d_conv": 2,
        "expand": 1,
        "dropout": 0.1,
        "ch_ind": True,
        "residual": True,
        "epochs": 80,
        "lr": 1e-3,
        "weight_decay": 1e-3,
        "batch_size": 64,
        "patience": 15,
        "seed": 42,
        "device": "auto",
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        if not _TORCH_OK:
            raise RuntimeError("PyTorch is required for TimeMachine classifier.")
        cfg = {**self.DEFAULT_CONFIG, **(params or {})}
        self._cfg = cfg
        self._seed = int(cfg["seed"])
        self._model: Optional[_TimeMachineClassification] = None
        self._device = _get_torch_device(str(cfg.get("device", "auto")))
        self.classes_: Optional[np.ndarray] = None

    def _build_model(self, n_classes: int = 2) -> _TimeMachineClassification:
        torch.manual_seed(self._seed)
        return _TimeMachineClassification(
            c_in=N_TS_VARS,
            seq_len=N_TS_STEPS,
            n1=int(self._cfg["n1"]),
            n2=int(self._cfg["n2"]),
            d_state=int(self._cfg["d_state"]),
            d_conv=int(self._cfg["d_conv"]),
            expand=int(self._cfg["expand"]),
            dropout=float(self._cfg["dropout"]),
            ch_ind=bool(self._cfg["ch_ind"]),
            residual=bool(self._cfg["residual"]),
            n_classes=n_classes,
        )

    def fit(self, X, y) -> Dict[str, Any]:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        X_3d = _to_ts3d(X)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        self._model = self._build_model(n_classes)
        self._model = _train_torch_classifier(
            self._model,
            X_3d,
            y,
            epochs=int(self._cfg["epochs"]),
            lr=float(self._cfg["lr"]),
            weight_decay=float(self._cfg["weight_decay"]),
            batch_size=int(self._cfg["batch_size"]),
            patience=int(self._cfg["patience"]),
            device=self._device,
        )
        return {
            "n_params": sum(p.numel() for p in self._model.parameters()),
            "mamba_backend": "official" if _MAMBA_OK else "SimpleMamba",
        }

    def predict(self, X):
        X_3d = _to_ts3d(np.asarray(X, dtype=np.float32))
        X_t = torch.tensor(X_3d, dtype=torch.float32, device=self._device)
        self._model.to(self._device).eval()
        with torch.no_grad():
            logits = self._model(X_t)
        return self.classes_[logits.argmax(dim=1).cpu().numpy()]

    def predict_proba(self, X):
        X_3d = _to_ts3d(np.asarray(X, dtype=np.float32))
        X_t = torch.tensor(X_3d, dtype=torch.float32, device=self._device)
        self._model.to(self._device).eval()
        with torch.no_grad():
            logits = self._model(X_t)
        return F.softmax(logits, dim=1).cpu().numpy()

    def save(self, path: str) -> None:
        state = {
            "model_state": (
                {k: v.cpu() for k, v in self._model.state_dict().items()}
                if self._model
                else None
            ),
            "cfg": self._cfg,
            "classes_": self.classes_,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._cfg = state["cfg"]
        self.classes_ = state.get("classes_")
        n_classes = len(self.classes_) if self.classes_ is not None else 2
        self._model = self._build_model(n_classes)
        if state.get("model_state"):
            self._model.load_state_dict(state["model_state"])
        self._model.to(self._device).eval()
