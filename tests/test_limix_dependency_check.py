from __future__ import annotations

from types import SimpleNamespace

import pytest

import tracking.classification.classifiers.limix as limix_module
from tracking.classification.classifiers.limix import LimiXSubjectClassifier


def test_limix_dependency_check_installs_missing(monkeypatch):
    clf = LimiXSubjectClassifier({"dependency_packages": ["fake_dep"]})

    calls = {"n": 0}

    def fake_importable(name: str) -> bool:
        calls["n"] += 1
        return calls["n"] > 1

    monkeypatch.setattr(limix_module, "_module_importable", fake_importable)
    monkeypatch.setattr(
        limix_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    clf._ensure_limix_dependencies()


def test_limix_dependency_check_raises_when_still_missing(monkeypatch):
    clf = LimiXSubjectClassifier({"dependency_packages": ["fake_dep"]})

    monkeypatch.setattr(limix_module, "_module_importable", lambda name: False)
    monkeypatch.setattr(
        limix_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    with pytest.raises(RuntimeError, match="remain unavailable"):
        clf._ensure_limix_dependencies()


def test_limix_dependency_force_install(monkeypatch):
    clf = LimiXSubjectClassifier({"dependency_packages": ["kditransform"]})

    monkeypatch.setattr(limix_module, "_module_importable", lambda name: True)

    called = {"n": 0}

    def _fake_run(*args, **kwargs):
        called["n"] += 1
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(limix_module.subprocess, "run", _fake_run)
    clf._ensure_limix_dependencies(force_install=["kditransform"])

    assert called["n"] == 1
