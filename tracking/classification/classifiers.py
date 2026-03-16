from __future__ import annotations

import pickle
from typing import Any, Dict, Optional

import numpy as np

try:  # pragma: no cover - optional dependency guard
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.impute import KNNImputer, SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
except Exception as exc:  # noqa: BLE001
    RandomForestClassifier = None  # type: ignore
    BaseEstimator = object  # type: ignore
    TransformerMixin = object  # type: ignore
    KNNImputer = None  # type: ignore
    SimpleImputer = None  # type: ignore
    Pipeline = None  # type: ignore
    SVC = None  # type: ignore
    DecisionTreeClassifier = None  # type: ignore
    _SKLEARN_IMPORT_ERROR = exc
else:
    _SKLEARN_IMPORT_ERROR = None

try:  # pragma: no cover - optional dependency guard
    from lightgbm import LGBMClassifier  # type: ignore[import-not-found]
except Exception as exc:  # noqa: BLE001
    LGBMClassifier = None  # type: ignore
    _LIGHTGBM_IMPORT_ERROR = exc
else:
    _LIGHTGBM_IMPORT_ERROR = None

from ..core.registry import register_classifier
from .interfaces import SubjectClassifier


def _require_estimator(
    owner: str,
    estimator: Any,
    import_error: Optional[Exception],
    dependency: str,
) -> None:
    if estimator is None:
        message = f"{dependency} is required for {owner}."
        if import_error:
            message += f" Import error: {import_error}"
        raise RuntimeError(message)


class _CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop highly correlated tabular columns using training data only."""

    def __init__(self, threshold: float = 0.98, min_features_keep: int = 1):
        self.threshold = float(threshold)
        self.min_features_keep = max(1, int(min_features_keep))
        self.keep_indices_: Optional[np.ndarray] = None

    def fit(self, X, y=None):  # noqa: ANN001
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("CorrelationFilter expects a 2D matrix.")
        n_features = X.shape[1]
        if n_features == 0:
            self.keep_indices_ = np.zeros(0, dtype=np.int64)
            return self
        if n_features <= self.min_features_keep or self.threshold >= 1.0:
            self.keep_indices_ = np.arange(n_features, dtype=np.int64)
            return self

        corr = np.corrcoef(X, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        abs_corr = np.abs(corr)

        keep_mask = np.ones(n_features, dtype=bool)
        for i in range(n_features):
            if not keep_mask[i]:
                continue
            for j in range(i + 1, n_features):
                if keep_mask[j] and abs_corr[i, j] >= self.threshold:
                    keep_mask[j] = False

        keep_indices = np.where(keep_mask)[0]
        if keep_indices.size < self.min_features_keep:
            keep_indices = np.arange(min(self.min_features_keep, n_features), dtype=np.int64)
        self.keep_indices_ = keep_indices.astype(np.int64)
        return self

    def transform(self, X):  # noqa: ANN001
        if self.keep_indices_ is None:
            raise RuntimeError("CorrelationFilter must be fitted before transform.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("CorrelationFilter expects a 2D matrix.")
        return X[:, self.keep_indices_]


class _PickleSubjectClassifier(SubjectClassifier):
    """Common save/load helpers for scikit-learn style estimators.

    支援可選的降維前處理 (``dim_reduction``):
    - ``"umap"``: UMAP 非線性降維
    - ``"lda"``: LDA 監督式線性降維
    子類別在 ``__init__`` 中呼叫 ``_init_dim_reduction(cfg)`` 即可啟用。

    另外支援 tabular 前處理管線（訓練集 fit、驗證/測試集 transform-only）：
    - 缺失值補值（median/mean/KNN）
    - 高相關特徵過濾（Pearson）
    """

    # ---- tabular pre-process + dim-reduction helpers ----
    def _init_tabular_pipeline(self, cfg: Dict[str, Any]) -> None:
        pp_cfg = cfg.get("tabular_preprocess", {}) or {}
        enabled = bool(pp_cfg.get("enabled", True))
        self._tabular_pipeline_enabled = enabled

        if not enabled:
            self._tabular_pipeline = None
            return
        if Pipeline is None or SimpleImputer is None:
            self._tabular_pipeline = None
            self._tabular_pipeline_enabled = False
            return

        imputer_name = str(pp_cfg.get("imputer", "median") or "median").lower()
        if imputer_name in {"none", "passthrough", "skip"}:
            imputer_step: Any = "passthrough"
        elif imputer_name == "knn":
            if KNNImputer is None:
                raise RuntimeError("KNNImputer requires scikit-learn.")
            imputer_step = KNNImputer(
                n_neighbors=int(pp_cfg.get("knn_neighbors", 5)),
                weights=str(pp_cfg.get("knn_weights", "uniform")),
            )
        else:
            strategy = imputer_name if imputer_name in {"mean", "median", "most_frequent", "constant"} else "median"
            fill_value = pp_cfg.get("fill_value", 0.0)
            imputer_step = SimpleImputer(strategy=strategy, fill_value=fill_value)

        corr_threshold = float(pp_cfg.get("corr_threshold", 0.98))
        min_keep = int(pp_cfg.get("corr_min_features_keep", 1))
        if corr_threshold >= 1.0:
            corr_step: Any = "passthrough"
        else:
            corr_step = _CorrelationFilter(
                threshold=corr_threshold,
                min_features_keep=min_keep,
            )

        self._tabular_pipeline = Pipeline(
            steps=[
                ("imputer", imputer_step),
                ("corr", corr_step),
            ]
        )

    def _init_dim_reduction(self, cfg: Dict[str, Any]) -> None:
        """從 config 初始化 tabular 降維器 (umap / lda)。"""
        self._init_tabular_pipeline(cfg)
        dr_name = cfg.get("dim_reduction")
        if dr_name:
            from .dim_reduction import get_tabular_reducer
            dr_params = cfg.get("dim_reduction_params", {})
            self._dim_reducer = get_tabular_reducer(dr_name, dr_params)
            self._dim_reduction_name = dr_name
        else:
            self._dim_reducer = None
            self._dim_reduction_name = None

    def _fit_reduce(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Train-only fit: tabular pipeline first, then optional dim-reducer."""
        X = np.asarray(X, dtype=np.float32)
        if self._tabular_pipeline is not None:
            X = self._tabular_pipeline.fit_transform(X, y)
            X = np.asarray(X, dtype=np.float32)
        if self._dim_reducer is not None:
            return self._dim_reducer.fit_transform(X, y)
        return X

    def _transform_reduce(self, X: np.ndarray) -> np.ndarray:
        """Inference transform-only: tabular pipeline first, then dim-reducer."""
        X = np.asarray(X, dtype=np.float32)
        if self._tabular_pipeline is not None:
            X = self._tabular_pipeline.transform(X)
            X = np.asarray(X, dtype=np.float32)
        if self._dim_reducer is not None:
            return self._dim_reducer.transform(X)
        return X

    # ---- save / load (backward compatible) ----
    def save(self, path: str) -> None:
        state = {
            "model": self._model,
            "tabular_pipeline": self._tabular_pipeline,
            "tabular_pipeline_enabled": self._tabular_pipeline_enabled,
            "dim_reducer_state": (
                self._dim_reducer.get_state() if self._dim_reducer else None
            ),
            "dim_reduction_name": self._dim_reduction_name,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        # backward compat: old format pickled the model directly
        if isinstance(data, dict) and "model" in data:
            self._model = data["model"]
            self._tabular_pipeline = data.get("tabular_pipeline")
            self._tabular_pipeline_enabled = bool(data.get("tabular_pipeline_enabled", self._tabular_pipeline is not None))
            dr_state = data.get("dim_reducer_state")
            dr_name = data.get("dim_reduction_name")
            if dr_state and dr_name:
                from .dim_reduction import get_tabular_reducer
                self._dim_reducer = get_tabular_reducer(dr_name, {})
                self._dim_reducer.set_state(dr_state)
                self._dim_reduction_name = dr_name
            else:
                self._dim_reducer = None
                self._dim_reduction_name = None
        else:
            self._model = data
            self._tabular_pipeline = None
            self._tabular_pipeline_enabled = False
            self._dim_reducer = None
            self._dim_reduction_name = None
        self.classes_ = getattr(self._model, "classes_", None)


@register_classifier("random_forest")
class RandomForestSubjectClassifier(_PickleSubjectClassifier):
    """Random forest classifier for subject-level diagnosis."""

    name = "RandomForest"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        _require_estimator(
            "RandomForestSubjectClassifier",
            RandomForestClassifier,
            _SKLEARN_IMPORT_ERROR,
            "scikit-learn",
        )
        cfg = params or {}
        n_estimators = int(cfg.get("n_estimators", 200))
        max_depth = cfg.get("max_depth")
        if max_depth is not None:
            max_depth = int(max_depth)
        min_samples_leaf = int(cfg.get("min_samples_leaf", 1))
        random_state = cfg.get("random_state")
        if random_state is not None:
            random_state = int(random_state)
        class_weight = cfg.get("class_weight", "balanced")
        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=int(cfg.get("n_jobs", -1)),
            class_weight=class_weight,
        )
        self.classes_: Optional[np.ndarray] = None
        self._init_dim_reduction(cfg)

    def fit(self, X, y) -> Dict[str, Any]:  # noqa: ANN001
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if X.size == 0:
            raise ValueError("No training data provided for classifier.")
        X = self._fit_reduce(X, y)
        self._model.fit(X, y)
        self.classes_ = getattr(self._model, "classes_", None)
        importances = (self._model.feature_importances_).tolist()
        return {"feature_importances": importances}

    def predict(self, X):  # noqa: ANN001
        X = np.asarray(X, dtype=np.float32)
        X = self._transform_reduce(X)
        return self._model.predict(X)

    def predict_proba(self, X):  # noqa: ANN001
        X = np.asarray(X, dtype=np.float32)
        X = self._transform_reduce(X)
        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(X)
        raise RuntimeError("Classifier does not support predict_proba().")


@register_classifier("decision_tree")
class DecisionTreeSubjectClassifier(_PickleSubjectClassifier):
    """Decision tree classifier for subject-level diagnosis."""

    name = "DecisionTree"
    DEFAULT_CONFIG = {
        "max_depth": 5,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "criterion": "gini",
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        _require_estimator(
            "DecisionTreeSubjectClassifier",
            DecisionTreeClassifier,
            _SKLEARN_IMPORT_ERROR,
            "scikit-learn",
        )
        cfg = params or {}
        kwargs: Dict[str, Any] = {
            "max_depth": cfg.get("max_depth"),
            "min_samples_split": int(cfg.get("min_samples_split", 2)),
            "min_samples_leaf": int(cfg.get("min_samples_leaf", 1)),
            "class_weight": cfg.get("class_weight", "balanced"),
            "criterion": cfg.get("criterion", "gini"),
        }
        if kwargs["max_depth"] is not None:
            kwargs["max_depth"] = int(kwargs["max_depth"])
        random_state = cfg.get("random_state")
        if random_state is not None:
            kwargs["random_state"] = int(random_state)
        self._model = DecisionTreeClassifier(**kwargs)
        self.classes_: Optional[np.ndarray] = None
        self._init_dim_reduction(cfg)

    def fit(self, X, y) -> Dict[str, Any]:  # noqa: ANN001
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if X.size == 0:
            raise ValueError("No training data provided for classifier.")
        X = self._fit_reduce(X, y)
        self._model.fit(X, y)
        self.classes_ = getattr(self._model, "classes_", None)
        return {"feature_importances": self._model.feature_importances_.tolist()}

    def predict(self, X):  # noqa: ANN001
        X = np.asarray(X, dtype=np.float32)
        X = self._transform_reduce(X)
        return self._model.predict(X)

    def predict_proba(self, X):  # noqa: ANN001
        X = np.asarray(X, dtype=np.float32)
        X = self._transform_reduce(X)
        return self._model.predict_proba(X)


@register_classifier("svm")
class SVMSubjectClassifier(_PickleSubjectClassifier):
    """Support Vector Machine classifier with optional probability estimates."""

    name = "SupportVectorMachine"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        _require_estimator(
            "SVMSubjectClassifier",
            SVC,
            _SKLEARN_IMPORT_ERROR,
            "scikit-learn",
        )
        cfg = params or {}
        self._probability = bool(cfg.get("probability", True))
        kwargs: Dict[str, Any] = {
            "C": float(cfg.get("C", 1.0)),
            "kernel": cfg.get("kernel", "rbf"),
            "gamma": cfg.get("gamma", "scale"),
            "degree": int(cfg.get("degree", 3)),
            "coef0": float(cfg.get("coef0", 0.0)),
            "probability": self._probability,
            "class_weight": cfg.get("class_weight", "balanced"),
        }
        random_state = cfg.get("random_state")
        if random_state is not None:
            kwargs["random_state"] = int(random_state)
        self._model = SVC(**kwargs)
        self.classes_: Optional[np.ndarray] = None
        self._init_dim_reduction(cfg)

    def fit(self, X, y) -> Dict[str, Any]:  # noqa: ANN001
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if X.size == 0:
            raise ValueError("No training data provided for classifier.")
        X = self._fit_reduce(X, y)
        self._model.fit(X, y)
        self.classes_ = getattr(self._model, "classes_", None)
        summary = {
            "support_vectors": int(self._model.support_.shape[0]),
            "n_support": getattr(self._model, "n_support_", np.array([])).tolist(),
        }
        return summary

    def predict(self, X):  # noqa: ANN001
        X = np.asarray(X, dtype=np.float32)
        X = self._transform_reduce(X)
        return self._model.predict(X)

    def predict_proba(self, X):  # noqa: ANN001
        if not self._probability or not hasattr(self._model, "predict_proba"):
            raise RuntimeError("Classifier does not support predict_proba(); set probability=True.")
        X = np.asarray(X, dtype=np.float32)
        X = self._transform_reduce(X)
        return self._model.predict_proba(X)


@register_classifier("lightgbm")
class LightGBMSubjectClassifier(_PickleSubjectClassifier):
    """Gradient boosted trees via LightGBM."""

    name = "LightGBMClassifier"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        _require_estimator(
            "LightGBMSubjectClassifier",
            LGBMClassifier,
            _LIGHTGBM_IMPORT_ERROR,
            "lightgbm",
        )
        cfg = params or {}
        kwargs: Dict[str, Any] = {
            "n_estimators": int(cfg.get("n_estimators", 200)),
            "learning_rate": float(cfg.get("learning_rate", 0.1)),
            "num_leaves": int(cfg.get("num_leaves", 31)),
            "max_depth": int(cfg.get("max_depth", -1)),
            "subsample": float(cfg.get("subsample", 1.0)),
            "colsample_bytree": float(cfg.get("colsample_bytree", 1.0)),
            "reg_lambda": float(cfg.get("reg_lambda", 0.0)),
            "reg_alpha": float(cfg.get("reg_alpha", 0.0)),
            "class_weight": cfg.get("class_weight", "balanced"),
            "n_jobs": int(cfg.get("n_jobs", -1)),
            "objective": cfg.get("objective", "binary"),
        }
        random_state = cfg.get("random_state")
        if random_state is not None:
            kwargs["random_state"] = int(random_state)
        self._model = LGBMClassifier(**kwargs)
        self.classes_: Optional[np.ndarray] = None
        self._init_dim_reduction(cfg)

    def fit(self, X, y) -> Dict[str, Any]:  # noqa: ANN001
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if X.size == 0:
            raise ValueError("No training data provided for classifier.")
        X = self._fit_reduce(X, y)
        self._model.fit(X, y)
        self.classes_ = getattr(self._model, "classes_", None)
        summary = {
            "feature_importances": getattr(self._model, "feature_importances_", np.array([])).tolist(),
        }
        return summary

    def predict(self, X):  # noqa: ANN001
        X = np.asarray(X, dtype=np.float32)
        X = self._transform_reduce(X)
        return self._model.predict(X)

    def predict_proba(self, X):  # noqa: ANN001
        X = np.asarray(X, dtype=np.float32)
        X = self._transform_reduce(X)
        return self._model.predict_proba(X)
