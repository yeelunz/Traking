from __future__ import annotations

import pickle
from typing import Any, Dict, Optional

import numpy as np

try:  # pragma: no cover - optional dependency guard
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
except Exception as exc:  # noqa: BLE001
    RandomForestClassifier = None  # type: ignore
    LogisticRegression = None  # type: ignore
    SVC = None  # type: ignore
    _SKLEARN_IMPORT_ERROR = exc
else:
    _SKLEARN_IMPORT_ERROR = None

try:  # pragma: no cover - optional dependency guard
    from xgboost import XGBClassifier  # type: ignore[import-not-found]
except Exception as exc:  # noqa: BLE001
    XGBClassifier = None  # type: ignore
    _XGBOOST_IMPORT_ERROR = exc
else:
    _XGBOOST_IMPORT_ERROR = None

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


class _PickleSubjectClassifier(SubjectClassifier):
    """Common save/load helpers for scikit-learn style estimators."""

    _model: Any
    classes_: Optional[np.ndarray]

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self._model, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self._model = pickle.load(f)
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

    def fit(self, X, y) -> Dict[str, Any]:  # noqa: ANN001
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if X.size == 0:
            raise ValueError("No training data provided for classifier.")
        self._model.fit(X, y)
        self.classes_ = getattr(self._model, "classes_", None)
        importances = (self._model.feature_importances_).tolist()
        return {"feature_importances": importances}

    def predict(self, X):  # noqa: ANN001
        X = np.asarray(X, dtype=np.float32)
        return self._model.predict(X)

    def predict_proba(self, X):  # noqa: ANN001
        X = np.asarray(X, dtype=np.float32)
        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(X)
        raise RuntimeError("Classifier does not support predict_proba().")


@register_classifier("logistic_regression")
class LogisticRegressionSubjectClassifier(_PickleSubjectClassifier):
    """Logistic regression classifier with configurable regularisation."""

    name = "LogisticRegression"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        _require_estimator(
            "LogisticRegressionSubjectClassifier",
            LogisticRegression,
            _SKLEARN_IMPORT_ERROR,
            "scikit-learn",
        )
        cfg = params or {}
        penalty = str(cfg.get("penalty", "l2")).lower()
        solver = cfg.get("solver")
        if solver is None:
            if penalty == "l1":
                solver = "liblinear"
            elif penalty == "elasticnet":
                solver = "saga"
            elif penalty == "none":
                solver = "lbfgs"
            else:
                solver = "lbfgs"
        kwargs: Dict[str, Any] = {
            "penalty": penalty,
            "C": float(cfg.get("C", 1.0)),
            "solver": solver,
            "max_iter": int(cfg.get("max_iter", 1000)),
            "class_weight": cfg.get("class_weight", "balanced"),
            "tol": float(cfg.get("tol", 1e-4)),
            "fit_intercept": bool(cfg.get("fit_intercept", True)),
        }
        multi_class = cfg.get("multi_class")
        if multi_class is not None:
            kwargs["multi_class"] = multi_class
        if penalty == "elasticnet":
            kwargs["l1_ratio"] = float(cfg.get("l1_ratio", 0.5))
        random_state = cfg.get("random_state")
        if random_state is not None:
            kwargs["random_state"] = int(random_state)
        self._model = LogisticRegression(**kwargs)
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X, y) -> Dict[str, Any]:  # noqa: ANN001
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if X.size == 0:
            raise ValueError("No training data provided for classifier.")
        self._model.fit(X, y)
        self.classes_ = getattr(self._model, "classes_", None)
        intercept = getattr(self._model, "intercept_", np.zeros(1, dtype=np.float32))
        summary = {
            "coefficients": self._model.coef_.tolist(),
            "intercept": np.asarray(intercept, dtype=np.float32).tolist(),
        }
        return summary

    def predict(self, X):  # noqa: ANN001
        X = np.asarray(X, dtype=np.float32)
        return self._model.predict(X)

    def predict_proba(self, X):  # noqa: ANN001
        X = np.asarray(X, dtype=np.float32)
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

    def fit(self, X, y) -> Dict[str, Any]:  # noqa: ANN001
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if X.size == 0:
            raise ValueError("No training data provided for classifier.")
        self._model.fit(X, y)
        self.classes_ = getattr(self._model, "classes_", None)
        summary = {
            "support_vectors": int(self._model.support_.shape[0]),
            "n_support": getattr(self._model, "n_support_", np.array([])).tolist(),
        }
        return summary

    def predict(self, X):  # noqa: ANN001
        X = np.asarray(X, dtype=np.float32)
        return self._model.predict(X)

    def predict_proba(self, X):  # noqa: ANN001
        if not self._probability or not hasattr(self._model, "predict_proba"):
            raise RuntimeError("Classifier does not support predict_proba(); set probability=True.")
        X = np.asarray(X, dtype=np.float32)
        return self._model.predict_proba(X)


@register_classifier("xgboost")
class XGBoostSubjectClassifier(_PickleSubjectClassifier):
    """Gradient boosted trees via XGBoost."""

    name = "XGBoostClassifier"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        _require_estimator(
            "XGBoostSubjectClassifier",
            XGBClassifier,
            _XGBOOST_IMPORT_ERROR,
            "xgboost",
        )
        cfg = params or {}
        kwargs: Dict[str, Any] = {
            "n_estimators": int(cfg.get("n_estimators", 200)),
            "max_depth": int(cfg.get("max_depth", 6)),
            "learning_rate": float(cfg.get("learning_rate", 0.1)),
            "subsample": float(cfg.get("subsample", 1.0)),
            "colsample_bytree": float(cfg.get("colsample_bytree", 1.0)),
            "gamma": float(cfg.get("gamma", 0.0)),
            "reg_lambda": float(cfg.get("reg_lambda", 1.0)),
            "reg_alpha": float(cfg.get("reg_alpha", 0.0)),
            "n_jobs": int(cfg.get("n_jobs", -1)),
            "use_label_encoder": bool(cfg.get("use_label_encoder", False)),
            "eval_metric": cfg.get("eval_metric", "logloss"),
        }
        if "scale_pos_weight" in cfg:
            kwargs["scale_pos_weight"] = float(cfg["scale_pos_weight"])
        tree_method = cfg.get("tree_method")
        if tree_method is not None:
            kwargs["tree_method"] = tree_method
        random_state = cfg.get("random_state")
        if random_state is not None:
            kwargs["random_state"] = int(random_state)
        self._model = XGBClassifier(**kwargs)
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X, y) -> Dict[str, Any]:  # noqa: ANN001
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if X.size == 0:
            raise ValueError("No training data provided for classifier.")
        self._model.fit(X, y)
        self.classes_ = getattr(self._model, "classes_", None)
        summary = {
            "feature_importances": getattr(self._model, "feature_importances_", np.array([])).tolist(),
        }
        return summary

    def predict(self, X):  # noqa: ANN001
        X = np.asarray(X, dtype=np.float32)
        return self._model.predict(X)

    def predict_proba(self, X):  # noqa: ANN001
        X = np.asarray(X, dtype=np.float32)
        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(X)
        raise RuntimeError("Classifier does not support predict_proba().")


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

    def fit(self, X, y) -> Dict[str, Any]:  # noqa: ANN001
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if X.size == 0:
            raise ValueError("No training data provided for classifier.")
        self._model.fit(X, y)
        self.classes_ = getattr(self._model, "classes_", None)
        summary = {
            "feature_importances": getattr(self._model, "feature_importances_", np.array([])).tolist(),
        }
        return summary

    def predict(self, X):  # noqa: ANN001
        X = np.asarray(X, dtype=np.float32)
        return self._model.predict(X)

    def predict_proba(self, X):  # noqa: ANN001
        X = np.asarray(X, dtype=np.float32)
        return self._model.predict_proba(X)
