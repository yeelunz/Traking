from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import pytest

from tracking.segmentation.workflow import SegmentationWorkflow, SEGMENTATION_MODEL_REGISTRY
import tracking.segmentation.model as seg_models


class _DummySegmentationModel(nn.Module):
    def __init__(self, _params: Dict[str, Any] | None = None):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, h, w = x.shape
        return torch.zeros((batch, 1, h, w), dtype=x.dtype, device=x.device)


class _DummyFCN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def eval(self):  # type: ignore[override]
        return self

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch, _, h, w = x.shape
        return {"out": torch.zeros((batch, 21, h, w), dtype=x.dtype, device=x.device)}


class _DummyWeights:
    def transforms(self, **_kwargs):  # type: ignore[override]
        def _transform(t: torch.Tensor) -> torch.Tensor:
            return t

        return _transform


@pytest.fixture(autouse=True)
def patch_segmentation_registry(monkeypatch):
    original = SEGMENTATION_MODEL_REGISTRY.get("unetpp")
    monkeypatch.setitem(SEGMENTATION_MODEL_REGISTRY, "unetpp", _DummySegmentationModel)
    monkeypatch.setattr(seg_models, "fcn_resnet50", lambda weights=None: _DummyFCN(), raising=False)
    monkeypatch.setattr(seg_models, "FCN_ResNet50_Weights", type("W", (), {"DEFAULT": _DummyWeights()}), raising=False)
    yield
    if original is not None:
        SEGMENTATION_MODEL_REGISTRY["unetpp"] = original


def test_segmentation_auto_pretrained_fallback(tmp_path):
    cfg = {
        "model": {"name": "unetpp"},
        "train": False,
        "auto_pretrained": True,
    }
    workflow = SegmentationWorkflow(cfg, dataset_root=str(tmp_path), results_dir=str(tmp_path))

    assert workflow.model_name == "torchvision_fcn_resnet50"
    assert getattr(workflow, "using_default_pretrained", False) is True
    assert workflow.train_enabled is False
    assert workflow.input_size == (256, 256)
    assert workflow.cfg.target_size == (256, 256)
    assert callable(getattr(workflow, "model", None)) or isinstance(workflow.model, nn.Module)

    # ensure loading checkpoint with auto-pretrained does not raise
    workflow.load_checkpoint()
