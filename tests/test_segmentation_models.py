from __future__ import annotations

import torch.nn as nn
import pytest

from tracking.core.registry import SEGMENTATION_MODEL_REGISTRY
import tracking.segmentation.model as seg_models


def test_deeplabv3_plus_uses_smp_builder(monkeypatch):
    called = {}

    def _fake_build(model_ctor, params=None):
        called["ctor"] = model_ctor
        called["params"] = dict(params or {})
        return nn.Identity()

    monkeypatch.setattr(seg_models, "_build_smp_model", _fake_build, raising=False)

    class _DummyDeepLab(nn.Module):
        def forward(self, x):  # type: ignore[override]
            return x

    class _DummySMP:
        DeepLabV3Plus = _DummyDeepLab

    monkeypatch.setattr(seg_models, "smp", _DummySMP, raising=False)

    model_cls = SEGMENTATION_MODEL_REGISTRY.get("deeplabv3+")
    assert model_cls is not None

    instance = model_cls({"classes": 2})
    assert isinstance(instance, nn.Module)
    assert called["ctor"] is _DummySMP.DeepLabV3Plus
    assert called["params"]["classes"] == 2


def test_nnunet_defaults_without_plans(monkeypatch):
    captured = {}

    class _Dummy(nn.Module):
        def forward(self, x):  # type: ignore[override]
            return x

    def _fake_get_network(**kwargs):
        captured.update(kwargs)
        return _Dummy()

    monkeypatch.setattr(seg_models, "_nnunet_get_network", _fake_get_network, raising=False)
    monkeypatch.setattr(seg_models, "_NNUNET_IMPORT_ERROR", None, raising=False)

    model_cls = SEGMENTATION_MODEL_REGISTRY.get("nnunet")
    assert model_cls is not None
    instance = model_cls({"in_channels": 1, "classes": 3})
    assert isinstance(instance, nn.Module)
    assert captured["input_channels"] == 1
    assert captured["output_channels"] == 3
    assert captured["arch_class_name"].endswith("PlainConvUNet")


def test_nnunet_uses_plans_manager(monkeypatch, tmp_path):
    captured = {}

    class _DummyNet(nn.Module):
        def forward(self, x):  # type: ignore[override]
            return x

    class _FakeConfig:
        network_arch_class_name = "dynamic.fake.Net"
        network_arch_init_kwargs = {"foo": "bar"}
        network_arch_init_kwargs_req_import = ["foo"]

        def __init__(self):
            self.configuration = {"deep_supervision": True}

    class _FakePlans:
        def __init__(self, _path):
            self._path = _path

        def get_configuration(self, _name):
            return _FakeConfig()

    def _fake_get_network(**kwargs):
        captured.update(kwargs)
        return _DummyNet()

    monkeypatch.setattr(seg_models, "_nnunet_get_network", _fake_get_network, raising=False)
    monkeypatch.setattr(seg_models, "_NnUNetPlansManager", _FakePlans, raising=False)
    monkeypatch.setattr(seg_models, "_NNUNET_IMPORT_ERROR", None, raising=False)

    plans_file = tmp_path / "nnUNetPlans.json"
    plans_file.write_text("{}", encoding="utf-8")

    model_cls = SEGMENTATION_MODEL_REGISTRY["nnunet"]
    instance = model_cls({"plans_path": str(plans_file), "configuration": "2d", "return_highres_only": False})
    assert isinstance(instance, nn.Module)
    assert captured["arch_class_name"] == "dynamic.fake.Net"
    assert captured["deep_supervision"] is True
    assert captured["output_channels"] == 1


def test_nnunet_import_guard(monkeypatch):
    monkeypatch.setattr(seg_models, "_nnunet_get_network", None, raising=False)
    monkeypatch.setattr(seg_models, "_NNUNET_IMPORT_ERROR", RuntimeError("missing"), raising=False)
    model_cls = SEGMENTATION_MODEL_REGISTRY.get("nnunet")
    assert model_cls is not None
    with pytest.raises(ImportError):
        model_cls({})
