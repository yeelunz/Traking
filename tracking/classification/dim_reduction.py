"""Dimensionality reduction utilities for the v3-lite feature pipeline.

Addresses non-linear information loss of PCA by providing classifier-aware
reduction strategies:

+---------------------+----------------------------+-----------------------------------+
| Classifier type     | Strategy                   | Mechanism                         |
+=====================+============================+===================================+
| PatchTST/TimeMachine| ``learned_projection``     | End-to-end trainable Linear that  |
|                     |                            | maps seq_len→target on-the-fly.   |
+---------------------+----------------------------+-----------------------------------+
| MultiRocket         | ``autoencoder``            | Conv1d autoencoder pre-trained on |
|                     |                            | TSC data; encoder output is used. |
+---------------------+----------------------------+-----------------------------------+
| Tabular (RF, LGBM…) | ``umap`` / ``lda``        | UMAP or sklearn LDA fitted on     |
|                     |                            | training features.                |
+---------------------+----------------------------+-----------------------------------+

Usage
~~~~~
Each reducer follows a **fit / transform / fit_transform** API identical to
scikit-learn.  Classifiers that support ``dim_reduction`` in their config will
instantiate the appropriate reducer automatically via :func:`get_reducer`.

Example YAML config::

    classifier:
      name: multirocket
      params:
        n_vars: 12
        n_steps: 256
        dim_reduction: autoencoder
        dim_reduction_target: 128    # compress 256 → 128 time steps
"""
from __future__ import annotations

import logging
import pickle
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# Optional dependency flags
# ═════════════════════════════════════════════════════════════════════════════

_TORCH_OK = False
try:
    import torch
    import torch.nn as nn
    _TORCH_OK = True
except ImportError:
    torch = None   # type: ignore[assignment]
    nn = None      # type: ignore[assignment]

_UMAP_OK = False
try:
    import umap as _umap_mod  # type: ignore[import-not-found]
    _UMAP_OK = True
except ImportError:
    _umap_mod = None  # type: ignore[assignment]

_SKLEARN_OK = False
try:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as _SklearnLDA
    _SKLEARN_OK = True
except ImportError:
    _SklearnLDA = None  # type: ignore[assignment]


# ═════════════════════════════════════════════════════════════════════════════
# 1. UMAP Reducer  (Tabular classifiers)
# ═════════════════════════════════════════════════════════════════════════════


class UMAPReducer:
    """UMAP non-linear dimensionality reduction for tabular features.

    Parameters
    ----------
    n_components : int
        Target dimensionality (default 16).
    n_neighbors : int
        UMAP neighbourhood size (default 15).
    min_dist : float
        Minimum distance between embedded points (default 0.1).
    metric : str
        Distance metric (default ``"euclidean"``).
    random_state : int
        Reproducibility seed.
    """

    def __init__(
        self,
        n_components: int = 16,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        random_state: int = 42,
    ):
        if not _UMAP_OK:
            raise RuntimeError(
                "umap-learn is required for UMAP dim reduction. "
                "Install with: pip install umap-learn"
            )
        self.n_components = n_components
        self._model = _umap_mod.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )
        self._fitted = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "UMAPReducer":
        """Fit UMAP on training data."""
        X = np.asarray(X, dtype=np.float32)
        self._model.fit(X, y=y)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project *X* into UMAP embedding space."""
        if not self._fitted:
            raise RuntimeError("UMAPReducer has not been fitted yet.")
        X = np.asarray(X, dtype=np.float32)
        return self._model.transform(X).astype(np.float32)

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        self.fit(X, y)
        return self._model.embedding_.astype(np.float32)

    def get_state(self) -> Dict[str, Any]:
        return {"model": self._model, "fitted": self._fitted}

    def set_state(self, state: Dict[str, Any]) -> None:
        self._model = state["model"]
        self._fitted = state["fitted"]


# ═════════════════════════════════════════════════════════════════════════════
# 2. LDA Reducer  (Tabular classifiers)
# ═════════════════════════════════════════════════════════════════════════════


class LDAReducer:
    """Linear Discriminant Analysis for supervised tabular dim reduction.

    Output dimensionality = min(n_classes - 1, n_features, target_dim).
    For binary classification this collapses to **1 D**.

    Parameters
    ----------
    n_components : int or None
        Maximum target dimensionality. ``None`` → LDA default
        (``min(n_classes - 1, n_features)``).
    solver : str
        LDA solver (default ``"svd"``, no covariance estimate needed).
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        solver: str = "svd",
    ):
        if not _SKLEARN_OK:
            raise RuntimeError(
                "scikit-learn is required for LDA dim reduction. "
                "Install with: pip install scikit-learn"
            )
        self.n_components = n_components
        self._model = _SklearnLDA(
            n_components=n_components,
            solver=solver,
        )
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LDAReducer":
        """Fit LDA on labelled training data."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self._model.fit(X, y)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("LDAReducer has not been fitted yet.")
        X = np.asarray(X, dtype=np.float64)
        return self._model.transform(X).astype(np.float32)

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self._model.transform(np.asarray(X, dtype=np.float64)).astype(np.float32)

    def get_state(self) -> Dict[str, Any]:
        return {"model": self._model, "fitted": self._fitted}

    def set_state(self, state: Dict[str, Any]) -> None:
        self._model = state["model"]
        self._fitted = state["fitted"]


# ═════════════════════════════════════════════════════════════════════════════
# 3. Temporal Autoencoder  (MultiRocket — non end-to-end)
# ═════════════════════════════════════════════════════════════════════════════


def _build_ts_autoencoder(
    n_channels: int,
    seq_len: int,
    target_len: int,
    hidden_dim: int = 32,
) -> "nn.Module":
    """Build a symmetric Conv1d autoencoder for temporal compression.

    Encoder:  (N, C, seq_len) → Conv1d → ReLU → AdaptiveAvgPool → (N, C, target_len)
    Decoder:  (N, C, target_len) → ConvTranspose1d → ReLU → Upsample → (N, C, seq_len)

    Channel-independent: each channel shares the same conv weights (grouped).
    """
    if not _TORCH_OK:
        raise RuntimeError("PyTorch required for autoencoder dim reduction.")

    class _TSAutoEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_channels = n_channels
            self.seq_len = seq_len
            self.target_len = target_len

            # Encoder: per-channel 1D conv + adaptive pool
            self.encoder = nn.Sequential(
                nn.Conv1d(n_channels, n_channels * hidden_dim, kernel_size=7,
                          padding=3, groups=n_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(n_channels * hidden_dim, n_channels, kernel_size=5,
                          padding=2, groups=n_channels),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(target_len),
            )

            # Decoder: reverse — upsample then conv
            self.decoder = nn.Sequential(
                nn.Upsample(size=seq_len, mode="linear", align_corners=False),
                nn.Conv1d(n_channels, n_channels * hidden_dim, kernel_size=5,
                          padding=2, groups=n_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(n_channels * hidden_dim, n_channels, kernel_size=7,
                          padding=3, groups=n_channels),
            )

        def encode(self, x: "torch.Tensor") -> "torch.Tensor":
            """Compress (N, C, seq_len) → (N, C, target_len)."""
            return self.encoder(x)

        def decode(self, z: "torch.Tensor") -> "torch.Tensor":
            """Reconstruct (N, C, target_len) → (N, C, seq_len)."""
            return self.decoder(z)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.decode(self.encode(x))

    return _TSAutoEncoder()


class TSAutoEncoderReducer:
    """Temporal autoencoder reducer for TSC features (MultiRocket).

    Trains a channel-independent Conv1d autoencoder to compress the temporal
    axis: ``(N, C, seq_len) → (N, C, target_len)``.

    MultiRocket cannot back-propagate through a front-end, so we pre-train
    the autoencoder on the training TSC data, freeze it, and use only the
    encoder half for feature compression.

    Parameters
    ----------
    n_channels : int
        Number of channels (default 12 for v3lite).
    seq_len : int
        Original sequence length (default 256).
    target_len : int
        Compressed sequence length (default 128).
    hidden_dim : int
        Hidden expansion factor per channel in conv layers.
    epochs : int
        Training epochs for the autoencoder.
    lr : float
        Learning rate.
    batch_size : int
        Mini-batch size.
    device : str
        ``"auto"`` / ``"cpu"`` / ``"cuda"``.
    """

    def __init__(
        self,
        n_channels: int = 12,
        seq_len: int = 256,
        target_len: int = 128,
        hidden_dim: int = 32,
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 64,
        device: str = "auto",
    ):
        if not _TORCH_OK:
            raise RuntimeError("PyTorch required for autoencoder dim reduction.")
        self.n_channels = n_channels
        self.seq_len = seq_len
        self.target_len = target_len
        self.hidden_dim = hidden_dim
        self._epochs = epochs
        self._lr = lr
        self._bs = batch_size
        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)
        self._model: Optional[nn.Module] = None
        self._fitted = False

    def fit(self, X_3d: np.ndarray, y: Optional[np.ndarray] = None) -> "TSAutoEncoderReducer":
        """Train autoencoder on 3D time-series data.

        Parameters
        ----------
        X_3d : ndarray, shape (N, C, seq_len)
        y : ignored (unsupervised; kept for API consistency)
        """
        X_3d = np.asarray(X_3d, dtype=np.float32)
        model = _build_ts_autoencoder(
            self.n_channels, self.seq_len, self.target_len, self.hidden_dim,
        ).to(self._device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self._lr)
        criterion = nn.MSELoss()
        dataset = torch.tensor(X_3d, dtype=torch.float32)
        n = len(dataset)

        model.train()
        for epoch in range(self._epochs):
            perm = torch.randperm(n)
            epoch_loss = 0.0
            n_batches = 0
            for i in range(0, n, self._bs):
                batch = dataset[perm[i:i + self._bs]].to(self._device)
                recon = model(batch)
                loss = criterion(recon, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            if (epoch + 1) % 20 == 0:
                logger.debug(
                    "TSAutoEncoder epoch %d/%d — MSE %.6f",
                    epoch + 1, self._epochs, epoch_loss / max(n_batches, 1),
                )

        model.eval()
        self._model = model
        self._fitted = True
        return self

    def transform(self, X_3d: np.ndarray) -> np.ndarray:
        """Encode (N, C, seq_len) → (N, C, target_len)."""
        if not self._fitted or self._model is None:
            raise RuntimeError("TSAutoEncoderReducer has not been fitted yet.")
        X_3d = np.asarray(X_3d, dtype=np.float32)
        self._model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_3d, dtype=torch.float32, device=self._device)
            Z = self._model.encode(X_t)
        return Z.cpu().numpy()

    def fit_transform(self, X_3d: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        self.fit(X_3d, y)
        return self.transform(X_3d)

    def get_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "n_channels": self.n_channels,
            "seq_len": self.seq_len,
            "target_len": self.target_len,
            "hidden_dim": self.hidden_dim,
            "fitted": self._fitted,
        }
        if self._model is not None:
            state["model_state"] = {
                k: v.cpu() for k, v in self._model.state_dict().items()
            }
        return state

    def set_state(self, state: Dict[str, Any]) -> None:
        self.n_channels = state["n_channels"]
        self.seq_len = state["seq_len"]
        self.target_len = state["target_len"]
        self.hidden_dim = state["hidden_dim"]
        self._fitted = state["fitted"]
        if "model_state" in state:
            model = _build_ts_autoencoder(
                self.n_channels, self.seq_len, self.target_len, self.hidden_dim,
            ).to(self._device)
            model.load_state_dict(state["model_state"])
            model.eval()
            self._model = model


# ═════════════════════════════════════════════════════════════════════════════
# 4. Learned Temporal Projection  (PatchTST / TimeMachine — end-to-end)
# ═════════════════════════════════════════════════════════════════════════════


def build_learned_projection(
    seq_len: int,
    target_len: int,
) -> "nn.Module":
    """Return a trainable 1-D linear temporal projection layer.

    Designed to be inserted at the *front* of PatchTST / TimeMachine so that
    the full pipeline is trained end-to-end.

    ``forward(x)``: ``(bs, c_in, seq_len)`` → ``(bs, c_in, target_len)``
    """
    if not _TORCH_OK:
        raise RuntimeError("PyTorch required for learned projection.")

    class _LearnedProjection(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(seq_len, target_len)
            nn.init.xavier_uniform_(self.proj.weight)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (bs, c_in, seq_len) — project along last dim
            return self.proj(x)

    return _LearnedProjection()


# ═════════════════════════════════════════════════════════════════════════════
# 5. Factory
# ═════════════════════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════════════════════════════
# 5. Channel-dimension Projection  (PatchTST / TimeMachine — end-to-end)
# ═════════════════════════════════════════════════════════════════════════════

class _LearnedChannelProjection(nn.Module if _TORCH_OK else object):  # type: ignore[misc]
    """Per-timestep shared Linear: (N, C, T) → (N, C_target, T).

    Applies ``nn.Linear(n_vars, target_vars)`` independently at every timestep
    by temporarily treating the time axis as the batch axis:
      (N, C, T) → permute → (N, T, C) → Linear → (N, T, C_t) → permute → (N, C_t, T)
    Trained end-to-end with the parent classifier.
    """

    def __init__(self, n_vars: int, target_vars: int):
        if not _TORCH_OK:
            raise RuntimeError("PyTorch required for _LearnedChannelProjection")
        super().__init__()
        self.proj = nn.Linear(n_vars, target_vars, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (N, C, T)
        x = x.permute(0, 2, 1)   # (N, T, C)
        x = self.proj(x)          # (N, T, C_target)
        x = x.permute(0, 2, 1)   # (N, C_target, T)
        return x


def build_learned_channel_projection(n_vars: int, target_vars: int) -> "nn.Module":
    """Return a channel-projection module for end-to-end training.

    Parameters
    ----------
    n_vars : int   Input number of channels C.
    target_vars : int   Output number of channels C_target (< C).
    """
    if not _TORCH_OK:
        raise RuntimeError("PyTorch required for build_learned_channel_projection")
    return _LearnedChannelProjection(n_vars, target_vars)


# ═════════════════════════════════════════════════════════════════════════════
# 6. Channel-dimension PCA Reducer  (MultiRocket — non end-to-end)
# ═════════════════════════════════════════════════════════════════════════════

class ChannelPCAReducer:
    """Reduce the channel (feature) dimension of TSC data via PCA.

    Operates on the (N, C, T) representation by treating each timestep as an
    independent observation of C-dimensional features:

        (N, C, T) → reshape (N·T, C) → sklearn PCA → (N·T, C_target)
                  → reshape (N, C_target, T)

    This preserves temporal structure completely — only the feature dimension
    is compressed, and the compression is learned from ``(N·T, C)`` samples.

    Parameters
    ----------
    n_components : int  Target number of channels after reduction.
    """

    def __init__(self, n_components: int = 8, **_kwargs: Any) -> None:
        self.n_components = n_components
        self._pca: Any = None
        self._fitted = False

    # ---- public API ----
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit PCA on the channel dim and return reduced (N, C_target, T)."""
        N, C, T = X.shape
        try:
            from sklearn.decomposition import PCA as _PCA
        except ImportError as exc:
            raise RuntimeError("scikit-learn required for ChannelPCAReducer") from exc
        Xr = X.transpose(0, 2, 1).reshape(N * T, C).astype(np.float32)
        nc = min(self.n_components, C, N * T)
        self._pca = _PCA(n_components=nc)
        out = self._pca.fit_transform(Xr).astype(np.float32)
        self._fitted = True
        return out.reshape(N, T, nc).transpose(0, 2, 1)  # (N, C_target, T)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted PCA. Returns (N, C_target, T)."""
        if not self._fitted:
            raise RuntimeError("ChannelPCAReducer must be fitted before transform.")
        N, C, T = X.shape
        Xr = X.transpose(0, 2, 1).reshape(N * T, C).astype(np.float32)
        out = self._pca.transform(Xr).astype(np.float32)
        nc = out.shape[1]
        return out.reshape(N, T, nc).transpose(0, 2, 1)  # (N, C_target, T)

    def get_state(self) -> Dict[str, Any]:
        return {
            "method": "pca",
            "pca": self._pca,
            "n_components": self.n_components,
            "fitted": self._fitted,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self._pca = state["pca"]
        self.n_components = state["n_components"]
        self._fitted = state["fitted"]


class ChannelLDAReducer:
    """Reduce channel dimension of TSC data via supervised LDA.

    Operates on ``(N, C, T)`` by reshaping to ``(N·T, C)`` and repeating labels
    per timestep so that LDA learns channel projections with class supervision.

    Notes
    -----
    For LDA, max output dim is ``min(C, n_classes-1)``.
    """

    def __init__(self, n_components: int = 8, solver: str = "svd", **_kwargs: Any) -> None:
        if not _SKLEARN_OK:
            raise RuntimeError("scikit-learn required for ChannelLDAReducer")
        self.n_components = n_components
        self.solver = solver
        self._lda: Any = None
        self._fitted = False

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit LDA on channel dim and return reduced ``(N, C_target, T)``."""
        if y is None:
            raise ValueError("ChannelLDAReducer requires labels y for fitting.")
        N, C, T = X.shape
        Xr = X.transpose(0, 2, 1).reshape(N * T, C).astype(np.float64)
        y = np.asarray(y)
        yr = np.repeat(y, T)
        n_classes = int(np.unique(y).shape[0])
        nc = min(self.n_components, C, max(1, n_classes - 1))
        self._lda = _SklearnLDA(n_components=nc, solver=self.solver)
        out = self._lda.fit_transform(Xr, yr).astype(np.float32)
        self._fitted = True
        return out.reshape(N, T, out.shape[1]).transpose(0, 2, 1)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted LDA. Returns ``(N, C_target, T)``."""
        if not self._fitted:
            raise RuntimeError("ChannelLDAReducer must be fitted before transform.")
        N, C, T = X.shape
        Xr = X.transpose(0, 2, 1).reshape(N * T, C).astype(np.float64)
        out = self._lda.transform(Xr).astype(np.float32)
        return out.reshape(N, T, out.shape[1]).transpose(0, 2, 1)

    def get_state(self) -> Dict[str, Any]:
        return {
            "method": "lda",
            "lda": self._lda,
            "n_components": self.n_components,
            "solver": self.solver,
            "fitted": self._fitted,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self._lda = state["lda"]
        self.n_components = state["n_components"]
        self.solver = state.get("solver", "svd")
        self._fitted = state["fitted"]


class ChannelUMAPReducer:
    """Reduce channel dimension of TSC data via UMAP on ``(N·T, C)``."""

    def __init__(
        self,
        n_components: int = 8,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        random_state: int = 42,
        **_kwargs: Any,
    ) -> None:
        if not _UMAP_OK:
            raise RuntimeError("umap-learn required for ChannelUMAPReducer")
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state
        self._umap: Any = _umap_mod.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )
        self._fitted = False

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        N, C, T = X.shape
        Xr = X.transpose(0, 2, 1).reshape(N * T, C).astype(np.float32)
        yr = np.repeat(np.asarray(y), T) if y is not None else None
        out = self._umap.fit_transform(Xr, y=yr).astype(np.float32)
        self._fitted = True
        return out.reshape(N, T, out.shape[1]).transpose(0, 2, 1)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("ChannelUMAPReducer must be fitted before transform.")
        N, C, T = X.shape
        Xr = X.transpose(0, 2, 1).reshape(N * T, C).astype(np.float32)
        out = self._umap.transform(Xr).astype(np.float32)
        return out.reshape(N, T, out.shape[1]).transpose(0, 2, 1)

    def get_state(self) -> Dict[str, Any]:
        return {
            "method": "umap",
            "umap": self._umap,
            "n_components": self.n_components,
            "n_neighbors": self.n_neighbors,
            "min_dist": self.min_dist,
            "metric": self.metric,
            "random_state": self.random_state,
            "fitted": self._fitted,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self._umap = state["umap"]
        self.n_components = state["n_components"]
        self.n_neighbors = state.get("n_neighbors", 15)
        self.min_dist = state.get("min_dist", 0.1)
        self.metric = state.get("metric", "euclidean")
        self.random_state = state.get("random_state", 42)
        self._fitted = state["fitted"]


# ════════════════════════════════════════════════════════════════════════════
# Factory helpers
# ════════════════════════════════════════════════════════════════════════════

_TABULAR_REDUCERS = {"umap", "lda"}
_TSC_REDUCERS = {"autoencoder", "learned_projection"}


def get_tabular_reducer(
    name: str,
    params: Optional[Dict[str, Any]] = None,
) -> Any:
    """Instantiate a tabular (Non-TSC) dimensionality reducer by name.

    Parameters
    ----------
    name : ``"umap"`` or ``"lda"``
    params : optional dict forwarded to reducer constructor.
    """
    p = params or {}
    if name == "umap":
        return UMAPReducer(
            n_components=int(p.get("n_components", 16)),
            n_neighbors=int(p.get("n_neighbors", 15)),
            min_dist=float(p.get("min_dist", 0.1)),
            metric=str(p.get("metric", "euclidean")),
            random_state=int(p.get("random_state", 42)),
        )
    if name == "lda":
        nc = p.get("n_components")
        return LDAReducer(
            n_components=int(nc) if nc is not None else None,
            solver=str(p.get("solver", "svd")),
        )
    raise ValueError(f"Unknown tabular reducer: {name!r}. Choose from {_TABULAR_REDUCERS}")


def get_tsc_reducer(
    name: str,
    params: Optional[Dict[str, Any]] = None,
) -> Any:
    """Instantiate a TSC dimensionality reducer by name.

    Parameters
    ----------
    name : ``"autoencoder"`` or ``"learned_projection"``
    params : optional dict forwarded to reducer constructor.
    """
    p = params or {}
    if name == "autoencoder":
        return TSAutoEncoderReducer(
            n_channels=int(p.get("n_channels", 12)),
            seq_len=int(p.get("seq_len", 256)),
            target_len=int(p.get("target_len", 128)),
            hidden_dim=int(p.get("hidden_dim", 32)),
            epochs=int(p.get("ae_epochs", 100)),
            lr=float(p.get("ae_lr", 1e-3)),
            batch_size=int(p.get("ae_batch_size", 64)),
            device=str(p.get("device", "auto")),
        )
    if name == "learned_projection":
        # Returns an nn.Module; caller embeds it in the classification model
        return build_learned_projection(
            seq_len=int(p.get("seq_len", 256)),
            target_len=int(p.get("target_len", 128)),
        )
    raise ValueError(f"Unknown TSC reducer: {name!r}. Choose from {_TSC_REDUCERS}")
