from __future__ import annotations

from types import SimpleNamespace

import pytest

import tracking.classification.classifiers.legacy_ext as legacy_ext
from tracking.classification.classifiers.legacy_ext import XGBoostSubjectClassifier


def test_xgboost_dependency_check_installs_missing(monkeypatch):
    monkeypatch.setattr(legacy_ext, "_XGB_OK", False)
    monkeypatch.setattr(legacy_ext, "_try_import_xgboost", lambda: True)
    monkeypatch.setattr(
        legacy_ext.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    XGBoostSubjectClassifier._ensure_xgboost_dependency(
        {"auto_install_dependencies": True, "dependency_packages": ["xgboost"]}
    )


def test_xgboost_dependency_check_raises_when_still_missing(monkeypatch):
    monkeypatch.setattr(legacy_ext, "_XGB_OK", False)
    monkeypatch.setattr(legacy_ext, "_try_import_xgboost", lambda: False)
    monkeypatch.setattr(
        legacy_ext.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    with pytest.raises(RuntimeError, match="remains unavailable"):
        XGBoostSubjectClassifier._ensure_xgboost_dependency(
            {"auto_install_dependencies": True, "dependency_packages": ["xgboost"]}
        )


def test_xgboost_dependency_check_raises_when_auto_install_disabled(monkeypatch):
    monkeypatch.setattr(legacy_ext, "_XGB_OK", False)

    with pytest.raises(RuntimeError, match="Install via: pip install xgboost"):
        XGBoostSubjectClassifier._ensure_xgboost_dependency(
            {"auto_install_dependencies": False, "dependency_packages": ["xgboost"]}
        )
