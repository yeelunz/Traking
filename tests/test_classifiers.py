import numpy as np
import pytest

from tracking.core.registry import CLASSIFIER_REGISTRY


@pytest.fixture()
def toy_dataset():
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.5, 0.2],
            [0.2, 0.8],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 0, 1, 1, 1, 0], dtype=np.int64)
    return X, y


def _build_classifier(name: str, params=None):
    cls = CLASSIFIER_REGISTRY.get(name)
    assert cls is not None, f"Classifier '{name}' not registered"
    return cls(params or {})


def test_logistic_regression_classifier(toy_dataset):
    X, y = toy_dataset
    clf = _build_classifier(
        "logistic_regression",
        {"penalty": "l2", "max_iter": 500, "C": 1.5, "class_weight": None},
    )
    summary = clf.fit(X, y)
    assert "coefficients" in summary
    preds = clf.predict(X)
    assert preds.shape == y.shape
    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], len(np.unique(y)))


def test_svm_classifier(toy_dataset):
    X, y = toy_dataset
    clf = _build_classifier(
        "svm",
        {"kernel": "rbf", "gamma": "scale", "probability": True, "class_weight": None},
    )
    summary = clf.fit(X, y)
    assert "support_vectors" in summary
    preds = clf.predict(X)
    assert preds.shape == y.shape
    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], len(np.unique(y)))


@pytest.mark.parametrize(
    "name, params",
    [
        ("xgboost", {"n_estimators": 25, "max_depth": 3, "learning_rate": 0.2}),
        ("lightgbm", {"n_estimators": 25, "learning_rate": 0.2, "num_leaves": 25}),
    ],
)
def test_gradient_boosting_classifiers_optional(toy_dataset, name, params):
    if name == "xgboost":
        pytest.importorskip("xgboost")
    if name == "lightgbm":
        pytest.importorskip("lightgbm")
    X, y = toy_dataset
    clf = _build_classifier(name, params)
    summary = clf.fit(X, y)
    assert "feature_importances" in summary
    preds = clf.predict(X)
    assert preds.shape == y.shape
    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], len(np.unique(y)))
