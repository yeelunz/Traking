"""Extended classifiers for CTS subject-level classification.

提供以下新分類器（配合 classifiers.py 中已有的 RF / LightGBM / SVM）：

表格式（需 tab_v2 特徵）：
  - ``xgboost``   : XGBoost gradient-boosted trees
    - ``tabpfn_v2`` : TabPFN 2.5（相容舊名稱）
    - ``tabpfn_2_5`` / ``tabpfn25`` / ``tabpfn2_5`` : TabPFN 2.5

時序式（需 tsc_v2 特徵提取器）：
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
import inspect
import importlib
import subprocess
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


def _try_import_xgboost() -> bool:
    """Attempt to import xgboost and refresh module-level handles."""
    global XGBClassifier, _XGB_OK
    try:
        from xgboost import XGBClassifier as _XGBClassifier  # type: ignore[import-not-found]
        XGBClassifier = _XGBClassifier  # type: ignore[assignment]
        _XGB_OK = True
        return True
    except Exception:
        XGBClassifier = None  # type: ignore[assignment,misc]
        _XGB_OK = False
        return False

# ── TabPFN ───────────────────────────────────────────────────────────────────
try:
    from tabpfn import TabPFNClassifier  # type: ignore[import-not-found]
    _TABPFN_OK = True
except Exception:
    TabPFNClassifier = None  # type: ignore[assignment,misc]
    _TABPFN_OK = False

# ── TabPFN Extensions (interpretability) ─────────────────────────────────────
try:
    from tabpfn_extensions.interpretability.feature_selection import (  # type: ignore[import-not-found]
        feature_selection as _tabpfn_feature_selection,
    )
    _TABPFN_EXT_INTERPRET_OK = True
except Exception:
    _tabpfn_feature_selection = None  # type: ignore[assignment]
    _TABPFN_EXT_INTERPRET_OK = False

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
def _repo_root() -> Path:
    """Return repository root directory for this project."""
    return Path(__file__).resolve().parents[3]


_LIBS_DIR = _repo_root() / "libs"
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
from ...core.registry import register_classifier
from .base import _PickleSubjectClassifier
from ..feature_extractors_ext import N_TS_VARS, N_TS_STEPS

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
    """Reshape flat (n_samples, n_vars*n_steps) → (n_samples, n_vars, n_steps).

    When ``n_vars`` and ``n_steps`` match the legacy defaults (18/128) *and*
    the input width suggests a different factorisation (e.g. 3072 = 12 × 256
    from v3lite), auto-detection is attempted.  Explicit overrides always win.
    """
    n = X.shape[0]
    expected = n_vars * n_steps

    # ── Auto-detect v3lite dimensions ────────────────────────────────────────
    if X.shape[1] != expected and n_vars == N_TS_VARS and n_steps == N_TS_STEPS:
        from ..feature_extractors_v3lite import N_TS_CHANNELS_LITE, N_TS_STEPS_LITE
        lite_expected = N_TS_CHANNELS_LITE * N_TS_STEPS_LITE
        if X.shape[1] == lite_expected:
            n_vars = N_TS_CHANNELS_LITE
            n_steps = N_TS_STEPS_LITE
            expected = lite_expected
            logger.info(
                "_to_ts3d: auto-detected v3lite dims (%d ch × %d steps)",
                n_vars, n_steps,
            )
        elif X.shape[1] % N_TS_CHANNELS_LITE == 0:
            # v3lite with custom n_steps
            n_steps = X.shape[1] // N_TS_CHANNELS_LITE
            n_vars = N_TS_CHANNELS_LITE
            expected = n_vars * n_steps
            logger.info(
                "_to_ts3d: auto-detected v3lite dims (%d ch × %d steps)",
                n_vars, n_steps,
            )
    # ─────────────────────────────────────────────────────────────────────────

    if X.shape[1] < expected:
        pad = np.zeros((n, expected - X.shape[1]), dtype=X.dtype)
        X = np.hstack([X, pad])
    elif X.shape[1] > expected:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "_to_ts3d: input has %d features but expected %d "
            "(n_vars=%d × n_steps=%d); truncating.",
            X.shape[1], expected, n_vars, n_steps,
        )
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

    # Remap labels to contiguous 0-indexed integers.
    # This is a safety net: if the caller passes labels like [1, 2] or [0, 1]
    # we always get a clean [0, 1, ...] range expected by CrossEntropyLoss.
    unique_classes = np.unique(y)
    label_to_idx = {int(c): i for i, c in enumerate(unique_classes)}
    y_remapped = np.array([label_to_idx[int(v)] for v in y], dtype=np.int64)

    X_t = torch.tensor(X_3d, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_remapped, dtype=torch.long, device=device)

    # Class-balanced weights (computed on remapped labels)
    classes, counts = np.unique(y_remapped, return_counts=True)
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
        "auto_install_dependencies": True,
        "dependency_packages": ["xgboost"],
    }

    @staticmethod
    def _ensure_xgboost_dependency(cfg: Dict[str, Any]) -> None:
        if _XGB_OK:
            return

        deps = cfg.get("dependency_packages", ["xgboost"])
        if isinstance(deps, (str, bytes)):
            deps = [str(deps)]
        deps = [str(pkg).strip() for pkg in (deps or []) if str(pkg).strip()]
        if not deps:
            deps = ["xgboost"]

        if not bool(cfg.get("auto_install_dependencies", True)):
            raise RuntimeError(
                "xgboost is required for XGBoostSubjectClassifier. "
                "Install via: pip install xgboost. "
                f"Active interpreter: {sys.executable}"
            )

        cmd = [sys.executable, "-m", "pip", "install", *deps]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "unknown pip failure").strip()
            raise RuntimeError(
                "Failed to install XGBoost dependency at runtime "
                f"({', '.join(deps)}). pip output: {detail}"
            )

        if not _try_import_xgboost():
            raise RuntimeError(
                "xgboost remains unavailable after runtime installation. "
                f"Active interpreter: {sys.executable}"
            )

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        cfg = {**self.DEFAULT_CONFIG, **(params or {})}
        self._ensure_xgboost_dependency(cfg)
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
        self._init_dim_reduction(cfg)

    def fit(self, X, y) -> Dict[str, Any]:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        n_input_features = int(X.shape[1]) if X.ndim == 2 else 0
        X = self._fit_reduce(X, y)
        self._model.fit(X, y, verbose=False)
        self.classes_ = getattr(self._model, "classes_", None)
        return self._build_feature_importance_summary(
            getattr(self._model, "feature_importances_", np.array([])),
            n_input_features,
        )

    def predict(self, X):
        X = self._transform_reduce(np.asarray(X, dtype=np.float32))
        return self._model.predict(X)

    def predict_proba(self, X):
        X = self._transform_reduce(np.asarray(X, dtype=np.float32))
        return self._model.predict_proba(X)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  TabPFN 2.5                                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


@register_classifier("tabpfn_v2")
@register_classifier("tabpfn_2_5")
@register_classifier("tabpfn25")
@register_classifier("tabpfn2_5")
class TabPFNV2SubjectClassifier(_PickleSubjectClassifier):
    """TabPFN 2.5: in-context learning for small tabular classification.

    If ``tabpfn`` is not installed **or** the gated model cannot be
    downloaded (e.g. missing HuggingFace authentication), the classifier
    automatically falls back to ``XGBClassifier`` (if available) or
    ``RandomForestClassifier`` so that the training pipeline does not
    abort.
    """

    name = "TabPFN2.5"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        cfg = params or {}
        self._cfg = dict(cfg)
        self._fallback: bool = False
        self._fallback_name: str = ""
        self._finetune_requested: bool = bool(cfg.get("finetune", cfg.get("fine_tune", False)))
        self._finetune_applied: bool = False

        if _TABPFN_OK:
            try:
                n_ens = int(cfg.get("n_ensemble_configurations", 32))
                kwargs: Dict[str, Any] = {}
                if "n_estimators" in cfg:
                    kwargs["n_estimators"] = int(cfg.get("n_estimators", n_ens))
                else:
                    kwargs["n_estimators"] = n_ens
                device = cfg.get("device", "cpu")
                if device:
                    kwargs["device"] = str(device)
                self._model = TabPFNClassifier(**kwargs)
            except Exception as _init_err:
                logger.warning(
                    "TabPFN init failed (%s); will attempt fallback on fit().",
                    _init_err,
                )
                self._model = None
        else:
            logger.warning(
                "tabpfn package not available; TabPFN classifier will use a fallback."
            )
            self._model = None

        self.classes_: Optional[np.ndarray] = None
        self._init_dim_reduction(cfg)

    # ── helpers ──────────────────────────────────────────────────────────

    def _ensure_fallback(self) -> None:
        """Lazily create a fallback estimator if the primary model is unavailable."""
        if self._model is not None and not self._fallback:
            return

        if self._fallback:
            return  # already created

        if _XGB_OK:
            self._model = XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                use_label_encoder=False, random_state=42, n_jobs=-1,
            )
            self._fallback = True
            self._fallback_name = "XGBoost"
            logger.warning("TabPFN unavailable – falling back to XGBoost.")
        else:
            try:
                from sklearn.ensemble import RandomForestClassifier as _RFC
                self._model = _RFC(
                    n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1,
                )
                self._fallback = True
                self._fallback_name = "RandomForest"
                logger.warning("TabPFN unavailable – falling back to RandomForest.")
            except Exception:
                raise RuntimeError(
                    "TabPFN is unavailable and no fallback classifier (xgboost / sklearn) is installed."
                )

    @staticmethod
    def _supports_parameter(fn: Any, name: str) -> bool:
        try:
            sig = inspect.signature(fn)
        except Exception:
            return False
        for p in sig.parameters.values():
            if p.kind == inspect.Parameter.VAR_KEYWORD:
                return True
        return name in sig.parameters

    @staticmethod
    def _filter_supported_kwargs(fn: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in (kwargs or {}).items():
            if TabPFNV2SubjectClassifier._supports_parameter(fn, k):
                out[k] = v
        return out

    def _build_finetune_fit_kwargs(self) -> Dict[str, Any]:
        if self._model is None:
            return {}
        fit_fn = getattr(self._model, "fit", None)
        if fit_fn is None:
            return {}

        kwargs: Dict[str, Any] = {}
        if self._supports_parameter(fit_fn, "fit_mode"):
            kwargs["fit_mode"] = str(self._cfg.get("fit_mode", "finetune"))
        if self._supports_parameter(fit_fn, "finetune"):
            kwargs["finetune"] = True
        if self._supports_parameter(fit_fn, "fine_tune"):
            kwargs["fine_tune"] = True

        optional_keys = (
            "max_steps",
            "n_steps",
            "epochs",
            "learning_rate",
            "lr",
            "batch_size",
            "weight_decay",
            "patience",
        )
        for key in optional_keys:
            if key in self._cfg and self._supports_parameter(fit_fn, key):
                kwargs[key] = self._cfg[key]

        return kwargs

    # ── public interface ─────────────────────────────────────────────────

    def fit(self, X, y) -> Dict[str, Any]:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        X = self._fit_reduce(X, y)

        if self._model is not None and not self._fallback:
            try:
                fit_kwargs: Dict[str, Any] = {}
                if self._finetune_requested:
                    fit_kwargs = self._build_finetune_fit_kwargs()
                    if fit_kwargs:
                        logger.info("TabPFN fine-tuning enabled with args: %s", sorted(fit_kwargs.keys()))
                    else:
                        logger.warning(
                            "finetune=True requested, but current tabpfn fit API does not expose finetune arguments; using standard fit()."
                        )

                try:
                    self._model.fit(X, y, **fit_kwargs)
                    self._finetune_applied = bool(self._finetune_requested and fit_kwargs)
                except TypeError as _fit_kw_err:
                    if self._finetune_requested and fit_kwargs:
                        logger.warning(
                            "TabPFN fine-tune kwargs rejected (%s); retrying standard fit().",
                            _fit_kw_err,
                        )
                        self._model.fit(X, y)
                        self._finetune_applied = False
                    else:
                        raise

                self.classes_ = np.unique(y)
                return {
                    "n_train_samples": len(y),
                    "backend": "tabpfn_2_5",
                    "finetune_requested": bool(self._finetune_requested),
                    "finetune_applied": bool(self._finetune_applied),
                }
            except Exception as _fit_err:
                logger.warning(
                    "TabPFN fit failed (%s); switching to fallback classifier.",
                    _fit_err,
                )
                self._model = None  # force fallback creation

        self._ensure_fallback()
        if self._fallback_name == "XGBoost":
            self._model.fit(X, y, verbose=False)
        else:
            self._model.fit(X, y)
        self.classes_ = getattr(self._model, "classes_", np.unique(y))
        return {"n_train_samples": len(y), "backend": f"fallback_{self._fallback_name}"}

    def predict(self, X):
        self._ensure_fallback()
        X = self._transform_reduce(np.asarray(X, dtype=np.float32))
        return self._model.predict(X)

    def predict_proba(self, X):
        self._ensure_fallback()
        X = self._transform_reduce(np.asarray(X, dtype=np.float32))
        return self._model.predict_proba(X)

    def get_feature_importance(
        self,
        X,
        y,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        self._ensure_fallback()
        if self._model is None:
            return {}

        Xn = self._transform_reduce(np.asarray(X, dtype=np.float32))
        yn = np.asarray(y, dtype=np.int64)
        if Xn.ndim != 2 or Xn.shape[0] == 0 or Xn.shape[1] == 0:
            return {}

        names = list(feature_names or [])
        if len(names) != Xn.shape[1]:
            names = [f"f_{i:04d}" for i in range(Xn.shape[1])]

        out: Dict[str, Any] = {}

        if hasattr(self._model, "feature_importances_"):
            try:
                values = np.asarray(getattr(self._model, "feature_importances_"), dtype=np.float64)
                if values.ndim == 1 and values.shape[0] == Xn.shape[1]:
                    rows = [
                        {
                            "feature_index": int(i),
                            "feature_name": str(names[i]),
                            "importance": float(values[i]),
                        }
                        for i in range(values.shape[0])
                    ]
                    rows.sort(key=lambda r: r["importance"], reverse=True)
                    out["tree_model_feature_importances"] = {
                        "method": "model_feature_importances_",
                        "importances": rows,
                    }
            except Exception:
                pass

        if _TABPFN_EXT_INTERPRET_OK and _tabpfn_feature_selection is not None:
            try:
                max_features = int(self._cfg.get("interpretability_max_features", 96))
                n_select = int(self._cfg.get("interpretability_n_features_to_select", 12))
                n_select = max(1, n_select)

                if Xn.shape[1] > max_features:
                    variances = np.var(Xn, axis=0)
                    keep_idx = np.argsort(variances)[-max_features:]
                    keep_idx = np.sort(keep_idx)
                else:
                    keep_idx = np.arange(Xn.shape[1])

                X_sel = Xn[:, keep_idx]
                names_sel = [names[int(i)] for i in keep_idx.tolist()]
                n_select = min(n_select, X_sel.shape[1])

                selector = _tabpfn_feature_selection(
                    estimator=self._model,
                    X=X_sel,
                    y=yn,
                    n_features_to_select=n_select,
                    feature_names=names_sel,
                )
                support = np.asarray(selector.get_support(), dtype=bool)

                selected_rows: List[Dict[str, Any]] = []
                binary_rows: List[Dict[str, Any]] = []
                for local_idx, keep_j in enumerate(keep_idx.tolist()):
                    is_selected = bool(support[local_idx]) if local_idx < support.shape[0] else False
                    row = {
                        "feature_index": int(keep_j),
                        "feature_name": str(names[int(keep_j)]),
                        "importance": float(1.0 if is_selected else 0.0),
                    }
                    binary_rows.append(row)
                    if is_selected:
                        selected_rows.append(dict(row))

                binary_rows.sort(key=lambda r: r["importance"], reverse=True)
                out["tabpfn_extensions_interpretability"] = {
                    "method": "tabpfn_extensions.interpretability.feature_selection",
                    "n_features_total": int(Xn.shape[1]),
                    "n_features_evaluated": int(X_sel.shape[1]),
                    "n_features_selected": int(len(selected_rows)),
                    "selected_features": selected_rows,
                    "importances": binary_rows,
                }
            except Exception as exc:  # noqa: BLE001
                out["tabpfn_extensions_interpretability"] = {
                    "method": "tabpfn_extensions.interpretability.feature_selection",
                    "error": str(exc),
                }

        return out

    def save(self, path: str) -> None:
        dr_state = self._dim_reducer.get_state() if self._dim_reducer is not None else None
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": self._model,
                    "fallback": self._fallback,
                    "fallback_name": self._fallback_name,
                    "tabular_pipeline": getattr(self, "_tabular_pipeline", None),
                    "tabular_pipeline_enabled": bool(
                        getattr(self, "_tabular_pipeline_enabled", False)
                    ),
                    "dim_reduction_name": getattr(self, "_dim_reduction_name", None),
                    "dim_reducer_state": dr_state,
                },
                f,
            )

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        if isinstance(state, dict) and "model" in state:
            self._model = state["model"]
            self._fallback = state.get("fallback", False)
            self._fallback_name = state.get("fallback_name", "")
            self._tabular_pipeline = state.get("tabular_pipeline")
            self._tabular_pipeline_enabled = bool(
                state.get("tabular_pipeline_enabled", self._tabular_pipeline is not None)
            )
            dr_name = state.get("dim_reduction_name")
            dr_state = state.get("dim_reducer_state")
            if dr_name and dr_state:
                from ..dim_reduction import get_tabular_reducer
                self._dim_reducer = get_tabular_reducer(dr_name, {})
                self._dim_reducer.set_state(dr_state)
                self._dim_reduction_name = dr_name
            else:
                self._dim_reducer = None
                self._dim_reduction_name = None
        else:
            # Legacy format: raw model
            self._model = state
            self._fallback = False
            self._tabular_pipeline = None
            self._tabular_pipeline_enabled = False
            self._dim_reducer = None
            self._dim_reduction_name = None
        self.classes_ = getattr(self._model, "classes_", None)


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
        # v3lite adaptive dims — leave None to auto-detect.
        "n_vars": None,
        "n_steps": None,
        # Dimensionality reduction (two independent axes):
        # Time axis:    None | "autoencoder"        (auto-encoder T → dim_reduction_target)
        # Channel axis: set channel_reduction_target + method (pca/lda/umap)
        "dim_reduction": None,
        "dim_reduction_target": 128,       # target time steps after T-reduction
        "channel_reduction_target": None,  # target channels after C-reduction (None = off)
        "channel_reduction_method": "pca",  # pca | lda | umap
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

        # Adaptive n_vars / n_steps
        self._cfg_n_vars = cfg.get("n_vars")    # None → auto-detect
        self._cfg_n_steps = cfg.get("n_steps")   # None → auto-detect

        # Dim reduction (two-axis, non-differentiable)
        self._dim_reduction_name = cfg.get("dim_reduction")
        self._dim_reduction_target = int(cfg.get("dim_reduction_target", 128))
        self._channel_reduction_target: Optional[int] = (
            int(cfg["channel_reduction_target"])
            if cfg.get("channel_reduction_target") is not None else None
        )
        self._channel_reduction_method: str = str(
            cfg.get("channel_reduction_method", "pca")
        ).strip().lower()
        if self._channel_reduction_method not in {"pca", "lda", "umap"}:
            raise ValueError(
                "channel_reduction_method must be one of: pca, lda, umap"
            )
        self._ae_reducer: Any = None
        self._channel_reducer: Any = None

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

    def _preprocess(
        self,
        X: np.ndarray,
        *,
        fit_ae: bool = False,
        y: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        nv = int(self._cfg_n_vars) if self._cfg_n_vars else N_TS_VARS
        ns = int(self._cfg_n_steps) if self._cfg_n_steps else N_TS_STEPS
        X_3d = _to_ts3d(np.asarray(X, dtype=np.float32), nv, ns).astype(np.float64)

        # Step A: Channel reduction (pca/lda/umap; independent of time axis)
        if self._channel_reduction_target is not None:
            if fit_ae:
                from ..dim_reduction import (
                    ChannelLDAReducer,
                    ChannelPCAReducer,
                    ChannelUMAPReducer,
                )
                if self._channel_reduction_method == "pca":
                    self._channel_reducer = ChannelPCAReducer(
                        n_components=self._channel_reduction_target
                    )
                elif self._channel_reduction_method == "lda":
                    self._channel_reducer = ChannelLDAReducer(
                        n_components=self._channel_reduction_target
                    )
                else:
                    self._channel_reducer = ChannelUMAPReducer(
                        n_components=self._channel_reduction_target
                    )
                orig_c = X_3d.shape[1]
                X_3d_in = X_3d.astype(np.float32)
                if self._channel_reduction_method == "lda":
                    if y is None:
                        raise ValueError(
                            "channel_reduction_method='lda' requires labels y during fit()."
                        )
                    X_3d = self._channel_reducer.fit_transform(X_3d_in, y).astype(np.float64)
                elif self._channel_reduction_method == "umap":
                    X_3d = self._channel_reducer.fit_transform(X_3d_in, y).astype(np.float64)
                else:
                    X_3d = self._channel_reducer.fit_transform(X_3d_in).astype(np.float64)
                logger.info(
                    "MultiRocket channel reduction (%s): %d → %d channels",
                    self._channel_reduction_method,
                    orig_c,
                    int(X_3d.shape[1]),
                )
            elif self._channel_reducer is not None:
                X_3d = self._channel_reducer.transform(
                    X_3d.astype(np.float32)
                ).astype(np.float64)

        # Step B: Temporal reduction (autoencoder, independent of channel axis)
        if self._dim_reduction_name == "autoencoder":
            if fit_ae:
                from ..dim_reduction import TSAutoEncoderReducer
                self._ae_reducer = TSAutoEncoderReducer(
                    n_channels=X_3d.shape[1],
                    seq_len=X_3d.shape[2],
                    target_len=self._dim_reduction_target,
                    device="auto",
                )
                X_3d = self._ae_reducer.fit_transform(X_3d.astype(np.float32)).astype(np.float64)
                logger.info(
                    "MultiRocket autoencoder: %d → %d time steps",
                    ns if self._cfg_n_steps else N_TS_STEPS,
                    self._dim_reduction_target,
                )
            elif self._ae_reducer is not None:
                X_3d = self._ae_reducer.transform(X_3d.astype(np.float32)).astype(np.float64)

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
        y = np.asarray(y, dtype=np.int64)
        X_3d = self._preprocess(X, fit_ae=True, y=y)

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
        state = {
            "base_params": self._base_params,
            "diff_params": self._diff_params,
            "scaler": self._scaler,
            "clf": self._clf,
            "classes_": self.classes_,
            "num_kernels": self._num_kernels,
            "n_features_per_kernel": self._n_features_per_kernel,
            "max_dilations": self._max_dilations,
            "cfg_n_vars": self._cfg_n_vars,
            "cfg_n_steps": self._cfg_n_steps,
            "dim_reduction": self._dim_reduction_name,
            "dim_reduction_target": self._dim_reduction_target,
            "channel_reduction_target": self._channel_reduction_target,
            "channel_reduction_method": self._channel_reduction_method,
            "ae_state": (
                self._ae_reducer.get_state()
                if self._ae_reducer is not None else None
            ),
            "channel_reducer_state": (
                self._channel_reducer.get_state()
                if self._channel_reducer is not None else None
            ),
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

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
        self._cfg_n_vars = state.get("cfg_n_vars")
        self._cfg_n_steps = state.get("cfg_n_steps")
        self._dim_reduction_name = state.get("dim_reduction")
        self._dim_reduction_target = state.get("dim_reduction_target", 128)
        self._channel_reduction_target = state.get("channel_reduction_target")
        self._channel_reduction_method = state.get("channel_reduction_method", "pca")
        ae_state = state.get("ae_state")
        if ae_state is not None:
            from ..dim_reduction import TSAutoEncoderReducer
            self._ae_reducer = TSAutoEncoderReducer(
                n_channels=ae_state["n_channels"],
                seq_len=ae_state["seq_len"],
                target_len=ae_state["target_len"],
            )
            self._ae_reducer.set_state(ae_state)
        ch_state = state.get("channel_reducer_state")
        if ch_state is not None:
            from ..dim_reduction import (
                ChannelLDAReducer,
                ChannelPCAReducer,
                ChannelUMAPReducer,
            )
            method = self._channel_reduction_method
            if method == "lda":
                self._channel_reducer = ChannelLDAReducer(
                    n_components=ch_state["n_components"]
                )
            elif method == "umap":
                self._channel_reducer = ChannelUMAPReducer(
                    n_components=ch_state["n_components"]
                )
            else:
                self._channel_reducer = ChannelPCAReducer(
                    n_components=ch_state["n_components"]
                )
            self._channel_reducer.set_state(ch_state)


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
        temporal_projection_target: Optional[int] = None,
        channel_projection_target: Optional[int] = None,
    ):
        super().__init__()
        self.c_in = c_in
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride

        # ── (A) Optional channel projection: reduces C → C_target (end-to-end) ──
        if channel_projection_target and channel_projection_target < c_in:
            from ..dim_reduction import build_learned_channel_projection
            self.channel_proj = build_learned_channel_projection(c_in, channel_projection_target)
            self.effective_c_in = channel_projection_target
        else:
            self.channel_proj = None
            self.effective_c_in = c_in

        # ── (B) Optional temporal projection: reduces T → T_target (end-to-end) ──
        if temporal_projection_target and temporal_projection_target < seq_len:
            from ..dim_reduction import build_learned_projection
            self.temporal_proj = build_learned_projection(seq_len, temporal_projection_target)
            effective_len = temporal_projection_target
        else:
            self.temporal_proj = None
            effective_len = seq_len

        # RevIN (operates on original c_in channels — applied before channel_proj)
        self.revin = _RevIN(c_in, affine=True)

        # Patching geometry (uses effective_len after optional projection)
        self.n_patches = int((effective_len - patch_len) / stride + 1)

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

        # Classification head (uses effective_c_in after optional channel proj)
        self.head = nn.Sequential(
            nn.LayerNorm(self.effective_c_in * d_model),
            nn.Dropout(dropout),
            nn.Linear(self.effective_c_in * d_model, n_classes),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (bs, c_in, seq_len)
        bs = x.shape[0]

        # 1. RevIN (expects bs, seq_len, c_in) — applied on original channels
        x = x.permute(0, 2, 1)
        x = self.revin(x, "norm")
        x = x.permute(0, 2, 1)                              # (bs, c_in, seq_len)

        # 1a. Optional channel projection: C → C_target  (independent per timestep)
        if self.channel_proj is not None:
            x = self.channel_proj(x)                         # (bs, effective_c_in, seq_len)

        # 1b. Optional temporal projection: T → T_target  (independent per channel)
        if self.temporal_proj is not None:
            x = self.temporal_proj(x)                        # (bs, effective_c_in, target_len)

        # 2. Patching
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # → (bs, effective_c_in, n_patches, patch_len)

        # 3. Channel-independent
        x = x.reshape(bs * self.effective_c_in, self.n_patches, self.patch_len)

        # 4. Patch embedding
        x = self.W_P(x)                                     # (bs*effective_c_in, n_patches, d_model)

        # 5. Positional encoding + dropout
        x = self.input_dropout(x + self.W_pos)

        # 6. Transformer encoder
        x = self.encoder(x)                                  # (bs*effective_c_in, n_patches, d_model)

        # 7. Mean pooling over patches
        x = x.mean(dim=1)                                    # (bs*effective_c_in, d_model)

        # 8. Reshape and classify
        x = x.reshape(bs, -1)                                # (bs, effective_c_in*d_model)
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

    Input X 來自 ``tsc_v2`` 特徵提取器（flat 4608-dim = 18 ch × 256 steps）
    或 ``tsc_v3_lite`` 特徵提取器（flat 3072-dim = 12 ch × 256 steps）。

    透過 ``n_vars`` / ``n_steps`` 指定維度，或由 auto-detection 推斷。
    ``dim_reduction`` 支援 ``"learned_projection"``（end-to-end 時間軸降維）。
    ``channel_reduction_target`` 支援獨立的 channel 維度降維（Linear per-timestep）。
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
        # v3lite adaptive dims — leave None to auto-detect.
        "n_vars": None,
        "n_steps": None,
        # Dimensionality reduction (two independent axes):
        # Time axis:    None | "learned_projection"   (Linear T → dim_reduction_target)
        # Channel axis: set channel_reduction_target to an int to enable
        "dim_reduction": None,
        "dim_reduction_target": 128,       # target time steps after T-reduction
        "channel_reduction_target": None,  # target channels after C-reduction (None = off)
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
        # dim-reduction config
        self._cfg_n_vars: Optional[int] = cfg.get("n_vars")
        self._cfg_n_steps: Optional[int] = cfg.get("n_steps")
        self._dim_reduction_name: Optional[str] = cfg.get("dim_reduction")
        self._dim_reduction_target: int = int(cfg.get("dim_reduction_target", 128))
        self._channel_reduction_target: Optional[int] = (
            int(cfg["channel_reduction_target"])
            if cfg.get("channel_reduction_target") is not None else None
        )

    def _build_model(self, n_classes: int = 2) -> _PatchTSTClassification:
        torch.manual_seed(self._seed)
        c_in = int(self._cfg_n_vars) if self._cfg_n_vars else N_TS_VARS
        seq_len = int(self._cfg_n_steps) if self._cfg_n_steps else N_TS_STEPS
        tp_target = (
            self._dim_reduction_target
            if self._dim_reduction_name == "learned_projection" else None
        )
        cp_target = self._channel_reduction_target
        return _PatchTSTClassification(
            c_in=c_in,
            seq_len=seq_len,
            patch_len=int(self._cfg["patch_len"]),
            stride=int(self._cfg["stride"]),
            n_layers=int(self._cfg["n_layers"]),
            d_model=int(self._cfg["d_model"]),
            n_heads=int(self._cfg["n_heads"]),
            d_ff=int(self._cfg["d_ff"]),
            dropout=float(self._cfg["dropout"]),
            n_classes=n_classes,
            temporal_projection_target=tp_target,
            channel_projection_target=cp_target,
        )

    def _reshape(self, X: np.ndarray) -> np.ndarray:
        nv = int(self._cfg_n_vars) if self._cfg_n_vars else N_TS_VARS
        ns = int(self._cfg_n_steps) if self._cfg_n_steps else N_TS_STEPS
        return _to_ts3d(X, nv, ns)

    def fit(self, X, y) -> Dict[str, Any]:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        X_3d = self._reshape(X)

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
        X_3d = self._reshape(np.asarray(X, dtype=np.float32))
        X_t = torch.tensor(X_3d, dtype=torch.float32, device=self._device)
        self._model.to(self._device).eval()
        with torch.no_grad():
            logits = self._model(X_t)
        return self.classes_[logits.argmax(dim=1).cpu().numpy()]

    def predict_proba(self, X):
        X_3d = self._reshape(np.asarray(X, dtype=np.float32))
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
            "cfg_n_vars": self._cfg_n_vars,
            "cfg_n_steps": self._cfg_n_steps,
            "dim_reduction": self._dim_reduction_name,
            "dim_reduction_target": self._dim_reduction_target,
            "channel_reduction_target": self._channel_reduction_target,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._cfg = state["cfg"]
        self.classes_ = state.get("classes_")
        self._cfg_n_vars = state.get("cfg_n_vars")
        self._cfg_n_steps = state.get("cfg_n_steps")
        self._dim_reduction_name = state.get("dim_reduction")
        self._dim_reduction_target = state.get("dim_reduction_target", 128)
        self._channel_reduction_target = state.get("channel_reduction_target")
        n_classes = len(self.classes_) if self.classes_ is not None else 2
        self._model = self._build_model(n_classes)
        if state.get("model_state"):
            self._model.load_state_dict(state["model_state"])
        self._model.to(self._device).eval()


class _TimesNetClassification(_TorchModule):
    """TimesNet classification wrapper around THUML Time-Series-Library."""

    def __init__(
        self,
        *,
        c_in: int,
        seq_len: int,
        n_classes: int,
        e_layers: int = 2,
        d_model: int = 32,
        d_ff: int = 64,
        top_k: int = 3,
        num_kernels: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()

        ts_lib_root = _LIBS_DIR / "Time-Series-Library"
        if not ts_lib_root.exists():
            raise RuntimeError(
                f"Time-Series-Library not found at {ts_lib_root}. "
                "Please clone https://github.com/thuml/Time-Series-Library into libs/."
            )
        ts_lib_root_str = str(ts_lib_root)
        if ts_lib_root_str not in sys.path:
            sys.path.insert(0, ts_lib_root_str)

        try:
            timesnet_mod = importlib.import_module("models.TimesNet")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Failed to import TimesNet from Time-Series-Library. "
                f"Original error: {exc}"
            ) from exc

        class _Cfg:
            pass

        cfg = _Cfg()
        cfg.task_name = "classification"
        cfg.seq_len = int(seq_len)
        cfg.label_len = 0
        cfg.pred_len = 0
        cfg.e_layers = int(e_layers)
        cfg.top_k = int(top_k)
        cfg.d_model = int(d_model)
        cfg.d_ff = int(d_ff)
        cfg.num_kernels = int(num_kernels)
        cfg.enc_in = int(c_in)
        cfg.embed = "fixed"
        cfg.freq = "h"
        cfg.dropout = float(dropout)
        cfg.num_class = int(n_classes)
        cfg.c_out = int(c_in)

        self._model = timesnet_mod.Model(cfg)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # input x: (B, C, T) -> TimesNet expects (B, T, C)
        x_enc = x.permute(0, 2, 1)
        x_mark_enc = torch.ones(
            (x_enc.shape[0], x_enc.shape[1]),
            dtype=x_enc.dtype,
            device=x_enc.device,
        )
        return self._model(x_enc, x_mark_enc, None, None)


@register_classifier("timesnet")
class TimesNetSubjectClassifier(_PickleSubjectClassifier):
    """TimesNet classifier integrated from THUML Time-Series-Library."""

    name = "TimesNetClassifier"
    DEFAULT_CONFIG: Dict[str, Any] = {
        "e_layers": 2,
        "d_model": 32,
        "d_ff": 64,
        "top_k": 3,
        "num_kernels": 6,
        "dropout": 0.1,
        "epochs": 80,
        "lr": 1e-3,
        "weight_decay": 1e-3,
        "batch_size": 64,
        "patience": 15,
        "seed": 42,
        "device": "auto",
        "n_vars": None,
        "n_steps": None,
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        if not _TORCH_OK:
            raise RuntimeError("PyTorch is required for TimesNet classifier.")
        cfg = {**self.DEFAULT_CONFIG, **(params or {})}
        self._cfg = cfg
        self._seed = int(cfg["seed"])
        self._model: Optional[_TimesNetClassification] = None
        self._device = _get_torch_device(str(cfg.get("device", "auto")))
        self.classes_: Optional[np.ndarray] = None
        self._cfg_n_vars: Optional[int] = cfg.get("n_vars")
        self._cfg_n_steps: Optional[int] = cfg.get("n_steps")

    def _reshape(self, X: np.ndarray) -> np.ndarray:
        n_vars = int(self._cfg_n_vars) if self._cfg_n_vars else N_TS_VARS
        n_steps = int(self._cfg_n_steps) if self._cfg_n_steps else N_TS_STEPS
        return _to_ts3d(X, n_vars=n_vars, n_steps=n_steps)

    def _build_model(self, n_classes: int = 2) -> _TimesNetClassification:
        torch.manual_seed(self._seed)
        n_vars = int(self._cfg_n_vars) if self._cfg_n_vars else N_TS_VARS
        n_steps = int(self._cfg_n_steps) if self._cfg_n_steps else N_TS_STEPS
        return _TimesNetClassification(
            c_in=n_vars,
            seq_len=n_steps,
            n_classes=n_classes,
            e_layers=int(self._cfg["e_layers"]),
            d_model=int(self._cfg["d_model"]),
            d_ff=int(self._cfg["d_ff"]),
            top_k=int(self._cfg["top_k"]),
            num_kernels=int(self._cfg["num_kernels"]),
            dropout=float(self._cfg["dropout"]),
        )

    def fit(self, X, y) -> Dict[str, Any]:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        X_3d = self._reshape(X)

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
        X_3d = self._reshape(np.asarray(X, dtype=np.float32))
        X_t = torch.tensor(X_3d, dtype=torch.float32, device=self._device)
        self._model.to(self._device).eval()
        with torch.no_grad():
            logits = self._model(X_t)
        return self.classes_[logits.argmax(dim=1).cpu().numpy()]

    def predict_proba(self, X):
        X_3d = self._reshape(np.asarray(X, dtype=np.float32))
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
            "cfg_n_vars": self._cfg_n_vars,
            "cfg_n_steps": self._cfg_n_steps,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._cfg = state["cfg"]
        self.classes_ = state.get("classes_")
        self._cfg_n_vars = state.get("cfg_n_vars")
        self._cfg_n_steps = state.get("cfg_n_steps")
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
        temporal_projection_target: Optional[int] = None,
        channel_projection_target: Optional[int] = None,
    ):
        super().__init__()
        self.c_in = c_in
        self.seq_len = seq_len
        self.ch_ind = ch_ind
        self.residual = residual
        self.n1 = n1
        self.n2 = n2

        # ── (A) Optional channel projection: C → C_target (end-to-end) ──
        if channel_projection_target and channel_projection_target < c_in:
            from ..dim_reduction import build_learned_channel_projection
            self.channel_proj = build_learned_channel_projection(c_in, channel_projection_target)
            self.effective_c_in = channel_projection_target
        else:
            self.channel_proj = None
            self.effective_c_in = c_in

        # ── (B) Optional temporal projection: T → T_target (end-to-end) ──
        if temporal_projection_target and temporal_projection_target < seq_len:
            from ..dim_reduction import build_learned_projection
            self.temporal_proj = build_learned_projection(seq_len, temporal_projection_target)
            effective_len = temporal_projection_target
        else:
            self.temporal_proj = None
            effective_len = seq_len

        # RevIN (operates on original c_in channels)
        self.revin = _RevIN(c_in, affine=True)

        # d_model for cross-dim Mamba blocks depends on ch_ind
        d_param1 = 1 if ch_ind else n2
        d_param2 = 1 if ch_ind else n1

        # Linear projections (official: operates on last dim)
        self.lin1 = nn.Linear(effective_len, n1)
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

        # RevIN: (bs, seq_len, c_in) — applied on original channels
        x = x.permute(0, 2, 1)
        x = self.revin(x, "norm")
        x = x.permute(0, 2, 1)                              # (bs, c_in, seq_len)

        # (A) Optional channel projection: C → C_target  (independent per timestep)
        if self.channel_proj is not None:
            x = self.channel_proj(x)                         # (bs, effective_c_in, seq_len)

        # (B) Optional temporal projection: T → T_target  (independent per channel)
        if self.temporal_proj is not None:
            x = self.temporal_proj(x)                        # (bs, effective_c_in, target_len)

        # Channel-independent
        effective_len = x.shape[-1]
        if self.ch_ind:
            x = x.reshape(bs * self.effective_c_in, 1, effective_len)  # (bs*C_eff, 1, L)

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
            x = x.squeeze(1)                                  # (bs*C_eff, 2*n1)
            x = x.reshape(bs, self.effective_c_in, -1)        # (bs, C_eff, 2*n1)
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
    Input X 來自 ``tsc_v2`` 特徵提取器（flat 2304-dim = 18 ch × 128 steps）
    或 ``tsc_v3_lite`` 特徵提取器（flat 3072-dim = 12 ch × 256 steps）。

    可透過 ``n_vars`` / ``n_steps`` 指定維度，或由 auto-detection 推斷。
    ``dim_reduction`` 支援 ``"learned_projection"``（end-to-end Linear 降維）。

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
        # --- v3lite / dim-reduction ---
        "n_vars": None,           # None = auto-detect
        "n_steps": None,          # None = auto-detect
        # Dimensionality reduction (two independent axes):
        # Time axis:    None | "learned_projection"   (Linear T → dim_reduction_target)
        # Channel axis: set channel_reduction_target to an int to enable
        "dim_reduction": None,
        "dim_reduction_target": 128,
        "channel_reduction_target": None,  # target channels after C-reduction (None = off)
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
        # dim-reduction config
        self._cfg_n_vars: Optional[int] = cfg.get("n_vars")
        self._cfg_n_steps: Optional[int] = cfg.get("n_steps")
        self._dim_reduction_name: Optional[str] = cfg.get("dim_reduction")
        self._dim_reduction_target: int = int(cfg.get("dim_reduction_target", 128))
        self._channel_reduction_target: Optional[int] = (
            int(cfg["channel_reduction_target"])
            if cfg.get("channel_reduction_target") is not None else None
        )

    # ---- helpers ----
    def _reshape(self, X: np.ndarray) -> np.ndarray:
        """Reshape flat X → 3-D (N, C, T) using configured / auto-detected dims."""
        n_vars = self._cfg_n_vars if self._cfg_n_vars else N_TS_VARS
        n_steps = self._cfg_n_steps if self._cfg_n_steps else N_TS_STEPS
        return _to_ts3d(X, n_vars=n_vars, n_steps=n_steps)

    def _build_model(self, n_classes: int = 2) -> _TimeMachineClassification:
        torch.manual_seed(self._seed)
        n_vars = self._cfg_n_vars if self._cfg_n_vars else N_TS_VARS
        n_steps = self._cfg_n_steps if self._cfg_n_steps else N_TS_STEPS
        tp_target: Optional[int] = None
        if self._dim_reduction_name == "learned_projection" and self._dim_reduction_target < n_steps:
            tp_target = self._dim_reduction_target
        cp_target = self._channel_reduction_target
        return _TimeMachineClassification(
            c_in=n_vars,
            seq_len=n_steps,
            n1=int(self._cfg["n1"]),
            n2=int(self._cfg["n2"]),
            d_state=int(self._cfg["d_state"]),
            d_conv=int(self._cfg["d_conv"]),
            expand=int(self._cfg["expand"]),
            dropout=float(self._cfg["dropout"]),
            ch_ind=bool(self._cfg["ch_ind"]),
            residual=bool(self._cfg["residual"]),
            n_classes=n_classes,
            temporal_projection_target=tp_target,
            channel_projection_target=cp_target,
        )

    def fit(self, X, y) -> Dict[str, Any]:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        X_3d = self._reshape(X)

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
        X_3d = self._reshape(np.asarray(X, dtype=np.float32))
        X_t = torch.tensor(X_3d, dtype=torch.float32, device=self._device)
        self._model.to(self._device).eval()
        with torch.no_grad():
            logits = self._model(X_t)
        return self.classes_[logits.argmax(dim=1).cpu().numpy()]

    def predict_proba(self, X):
        X_3d = self._reshape(np.asarray(X, dtype=np.float32))
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
            "cfg_n_vars": self._cfg_n_vars,
            "cfg_n_steps": self._cfg_n_steps,
            "dim_reduction": self._dim_reduction_name,
            "dim_reduction_target": self._dim_reduction_target,
            "channel_reduction_target": self._channel_reduction_target,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._cfg = state["cfg"]
        self.classes_ = state.get("classes_")
        self._cfg_n_vars = state.get("cfg_n_vars")
        self._cfg_n_steps = state.get("cfg_n_steps")
        self._dim_reduction_name = state.get("dim_reduction")
        self._dim_reduction_target = state.get("dim_reduction_target", 128)
        self._channel_reduction_target = state.get("channel_reduction_target")
        n_classes = len(self.classes_) if self.classes_ is not None else 2
        self._model = self._build_model(n_classes)
        if state.get("model_state"):
            self._model.load_state_dict(state["model_state"])
        self._model.to(self._device).eval()
