from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import cv2
import torch
import torch.nn as nn
import pytest
import numpy as np

from tracking.core.interfaces import FramePrediction
from tracking.segmentation.workflow import SegmentationWorkflow, SEGMENTATION_MODEL_REGISTRY
import tracking.segmentation.model as seg_models
import tracking.segmentation.workflow as seg_workflow
from tracking.segmentation import utils as seg_utils


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


class _EvalDummyModel(nn.Module):
    def __init__(self, _params: Dict[str, Any] | None = None) -> None:
        super().__init__()
        self.eval_called = 0
        self.train_modes: List[bool] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, h, w = x.shape
        return torch.zeros((batch, 1, h, w), dtype=x.dtype, device=x.device)

    def eval(self):  # type: ignore[override]
        self.eval_called += 1
        return super().eval()

    def train(self, mode: bool = True):  # type: ignore[override]
        self.train_modes.append(bool(mode))
        return super().train(mode)


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


def test_resolve_checkpoint_supports_parent_directory(tmp_path):
    dataset_root = tmp_path / "data"
    dataset_root.mkdir()
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    ckpt = weights_dir / "seg.pt"
    ckpt.write_bytes(b"0")

    workflow = SegmentationWorkflow(
        {"model": {"name": "unetpp"}},
        dataset_root=str(dataset_root),
        results_dir=str(tmp_path / "results"),
    )
    workflow.project_root = str(tmp_path)

    resolved = workflow._resolve_checkpoint_path("../weights/seg.pt")
    assert resolved is not None
    assert os.path.normpath(resolved) == os.path.normpath(str(ckpt))


def test_predict_video_switches_to_eval_mode(monkeypatch, tmp_path):
    monkeypatch.setitem(SEGMENTATION_MODEL_REGISTRY, "unetpp", _EvalDummyModel)

    frames = {0: np.zeros((16, 16, 3), dtype=np.uint8)}

    class _FakeCapture:
        def __init__(self, _path):
            self._current = 0

        def isOpened(self):
            return True

        def set(self, prop_id, value):
            if prop_id == cv2.CAP_PROP_POS_FRAMES:
                self._current = int(value)

        def read(self):
            frame = frames.get(self._current)
            if frame is None:
                return False, None
            return True, frame.copy()

        def get(self, prop_id):
            if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
                return 16
            if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
                return 16
            return 0

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda *_args, **_kwargs: _FakeCapture(_args[0] if _args else None))

    written_masks = {}

    def _fake_imwrite(path, image):
        written_masks[path] = image
        return True

    monkeypatch.setattr(cv2, "imwrite", _fake_imwrite)

    cfg = {"model": {"name": "unetpp"}}
    workflow = SegmentationWorkflow(cfg, dataset_root=str(tmp_path), results_dir=str(tmp_path / "results"))
    assert isinstance(workflow.model, _EvalDummyModel)
    dummy_model: _EvalDummyModel = workflow.model

    preds = [FramePrediction(frame_index=0, bbox=(2.0, 2.0, 6.0, 6.0))]
    metrics, accum = workflow.predict_video(
        "dummy.mp4",
        preds,
        str(tmp_path / "output"),
        viz_settings={"include_segmentation": False},
    )

    assert dummy_model.eval_called >= 1
    assert dummy_model.train_modes and dummy_model.train_modes[-1] is True
    assert preds[0].segmentation is not None
    assert written_masks  # ensure masks were "saved"
    assert isinstance(metrics, dict)
    assert isinstance(accum, dict)


def test_segmentation_predict_dataset_reports_fps(monkeypatch, tmp_path):
    def _fake_predict_video(self, video_path, predictions, output_dir, gt, viz):  # type: ignore[override]
        return {"dice_mean": 0.8, "fps": 20.0}, {"dice": [0.8]}

    dummy = SegmentationWorkflow.__new__(SegmentationWorkflow)
    monkeypatch.setattr(SegmentationWorkflow, "predict_video", _fake_predict_video, raising=False)

    result = dummy.predict_dataset({"video.mp4": []}, str(tmp_path))

    assert "fps_mean" in result["summary"]
    assert pytest.approx(result["summary"]["fps_mean"], rel=1e-6) == 20.0


def test_predict_video_empty_predictions_with_gt_penalizes(monkeypatch, tmp_path):
    frames = {
        0: np.zeros((16, 16, 3), dtype=np.uint8),
        14: np.zeros((16, 16, 3), dtype=np.uint8),
    }

    class _FakeCapture:
        def __init__(self, _path):
            self._current = 0

        def isOpened(self):
            return True

        def set(self, prop_id, value):
            if prop_id == cv2.CAP_PROP_POS_FRAMES:
                self._current = int(value)

        def read(self):
            frame = frames.get(self._current)
            if frame is None:
                return False, None
            return True, frame.copy()

        def get(self, prop_id):
            if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
                return 16
            if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
                return 16
            return 0

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda *_args, **_kwargs: _FakeCapture(_args[0] if _args else None))

    # Create a real dummy mask file so _resolve_gt_mask_path finds it on disk.
    _mask_img = np.zeros((16, 16), dtype=np.uint8)
    _mask_img[2:6, 3:7] = 255
    cv2.imwrite(str(tmp_path / "dummy.png"), _mask_img)

    written_masks = {}

    def _fake_imwrite(path, image):
        written_masks[path] = image
        return True

    monkeypatch.setattr(cv2, "imwrite", _fake_imwrite)

    cfg = {"model": {"name": "unetpp"}}
    workflow = SegmentationWorkflow(cfg, dataset_root=str(tmp_path), results_dir=str(tmp_path / "results"))

    # Patch GT mask loader to always return a full-foreground mask, so a zero-prediction model yields dice=0.
    def _fake_load_mask(_self, _video_path, _seg_data):
        return np.ones((16, 16), dtype=np.uint8) * 255

    monkeypatch.setattr(SegmentationWorkflow, "_load_mask_from_annotation", _fake_load_mask, raising=False)

    gt_annotation = {
        "frames": {"0": [(2.0, 2.0, 6.0, 6.0)], "14": [(2.0, 2.0, 6.0, 6.0)]},
        "frame_annotations": {
            "0": [{"mask_path": "dummy.png", "metadata": {"centroid": [0.0, 0.0]}}],
            "14": [{"mask_path": "dummy.png", "metadata": {"centroid": [0.0, 0.0]}}],
        },
    }

    metrics, accum = workflow.predict_video(
        "dummy.mp4",
        predictions=[],
        output_dir=str(tmp_path / "output"),
        gt_annotation=gt_annotation,
        viz_settings={"include_segmentation": False},
    )

    assert "dice_mean" in metrics
    assert metrics["dice_mean"] == pytest.approx(0.0, abs=1e-6)


def test_predict_video_preserves_upstream_fallback_bbox_source(monkeypatch, tmp_path):
    frames = {0: np.zeros((16, 16, 3), dtype=np.uint8)}

    class _FakeCapture:
        def __init__(self, _path):
            self._current = 0

        def isOpened(self):
            return True

        def set(self, prop_id, value):
            if prop_id == cv2.CAP_PROP_POS_FRAMES:
                self._current = int(value)

        def read(self):
            frame = frames.get(self._current)
            if frame is None:
                return False, None
            return True, frame.copy()

        def get(self, prop_id):
            if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
                return 16
            if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
                return 16
            return 0

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda *_args, **_kwargs: _FakeCapture(_args[0] if _args else None))
    monkeypatch.setattr(cv2, "imwrite", lambda *_args, **_kwargs: True)

    cfg = {"model": {"name": "unetpp"}}
    workflow = SegmentationWorkflow(cfg, dataset_root=str(tmp_path), results_dir=str(tmp_path / "results"))
    workflow.input_size = None

    preds = [
        FramePrediction(
            frame_index=0,
            bbox=(2.0, 2.0, 6.0, 6.0),
            score=None,
            is_fallback=True,
            bbox_source="lw_tab_p",
        )
    ]

    workflow.predict_video(
        "dummy.mp4",
        predictions=preds,
        output_dir=str(tmp_path / "output"),
        gt_annotation=None,
        viz_settings={"include_segmentation": False},
    )

    with open(tmp_path / "output" / "roi_trace.json", "r", encoding="utf-8") as f:
        trace = json.load(f)

    assert trace["0"]["bbox_source"] == "lw_tab_p"
    assert trace["0"]["is_fallback"] is True
