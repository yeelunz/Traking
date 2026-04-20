from __future__ import annotations

import json
import os
from types import SimpleNamespace

import cv2
import numpy as np

from tracking.core.interfaces import FramePrediction
from tracking.core.registry import MODEL_REGISTRY
from tracking.orchestrator import runner as runner_module
from tracking.orchestrator.runner import PipelineRunner


class _FakeSparseDetector:
    name = "FakeSparseDetector"

    def __init__(self, config):
        self.train_enabled = bool(config.get("train_enabled", False))
        self._all_preds = [
            FramePrediction(
                frame_index=idx,
                bbox=(10.0 + idx, 20.0 + idx, 30.0, 40.0),
                score=0.95,
                confidence=0.95,
                is_fallback=False,
                bbox_source="detector",
            )
            for idx in range(6)
        ]

    def train(self, *args, **kwargs):
        return {"status": "skipped"}

    def load_checkpoint(self, *args, **kwargs):
        return None

    def predict(self, video_path: str):
        return list(self._all_preds)

    def predict_frames(self, video_path: str, frame_indices):
        wanted = {int(v) for v in frame_indices}
        return [p for p in self._all_preds if int(p.frame_index) in wanted]


class _FakeSegmentationWorkflow:
    def __init__(self, cfg, dataset_root, train_root, logger):
        self.cfg = SimpleNamespace(
            seed=int((cfg or {}).get("seed", 0)),
            val_ratio=float((cfg or {}).get("val_ratio", 0.0)),
            train=bool((cfg or {}).get("train", False)),
        )
        self.model_name = "fake_seg"
        self.inference_checkpoint = None
        self.best_checkpoint = None

    def load_checkpoint(self, *args, **kwargs):
        return None

    def train(self, *args, **kwargs):
        return {}

    def predict_dataset(self, preds_by_video, output_root, gt_annotations=None, viz_settings=None):
        os.makedirs(output_root, exist_ok=True)
        for preds in preds_by_video.values():
            for pred in preds:
                pred.segmentation = None
        return {"summary": {}, "videos": {}}


def _write_test_video(video_path: str, frame_count: int = 6) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(video_path, fourcc, 5.0, (64, 48))
    assert writer.isOpened()
    for idx in range(frame_count):
        frame = np.full((48, 64, 3), idx * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _write_test_annotation(json_path: str) -> None:
    payload = {
        "images": [
            {"id": idx + 1, "frame_index": idx, "file_name": f"frame_{idx:06d}.png"}
            for idx in range(6)
        ],
        "annotations": [
            {"id": 1, "image_id": 2, "category_id": 1, "bbox": [11.0, 21.0, 30.0, 40.0]},
            {"id": 2, "image_id": 4, "category_id": 1, "bbox": [13.0, 23.0, 30.0, 40.0]},
            {"id": 3, "image_id": 6, "category_id": 1, "bbox": [15.0, 25.0, 30.0, 40.0]},
        ],
        "categories": [{"id": 1, "name": "roi"}],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def test_restrict_to_gt_frames_keeps_full_trajectory_for_downstream(tmp_path, monkeypatch):
    dataset_root = tmp_path / "dataset"
    subject_dir = dataset_root / "01"
    subject_dir.mkdir(parents=True)
    video_path = subject_dir / "clip.avi"
    anno_path = subject_dir / "clip.json"
    _write_test_video(str(video_path))
    _write_test_annotation(str(anno_path))

    captured = {}

    def _fake_run_subject_classification(
        config,
        dataset_root,
        train_dataset,
        test_predictions,
        results_dir,
        logger,
        **kwargs,
    ):
        captured["test_predictions"] = test_predictions
        out_dir = os.path.join(results_dir, "classification")
        os.makedirs(out_dir, exist_ok=True)
        summary = {"accuracy": 1.0}
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return summary

    def _fixed_timestamp_dir(self, name: str) -> str:
        out_dir = os.path.join(self.results_root, name)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    monkeypatch.setitem(MODEL_REGISTRY, "FakeSparseDetector", _FakeSparseDetector)
    monkeypatch.setattr(runner_module, "SegmentationWorkflow", _FakeSegmentationWorkflow)
    monkeypatch.setattr(runner_module, "run_subject_classification", _fake_run_subject_classification)
    monkeypatch.setattr(PipelineRunner, "_timestamp_dir", _fixed_timestamp_dir)

    cfg = {
        "seed": 0,
        "dataset": {
            "root": str(dataset_root),
            "split": {
                "method": "video_level",
                "ratios": [0.0, 0.0, 1.0],
                "k_fold": 1,
            },
        },
        "experiments": [
            {
                "name": "full_traj_regression",
                "pipeline": [
                    {
                        "type": "model",
                        "name": "FakeSparseDetector",
                        "params": {"train_enabled": False},
                    }
                ],
            }
        ],
        "evaluation": {
            "evaluator": "BasicEvaluator",
            "restrict_to_gt_frames": True,
            "visualize": {
                "enabled": False,
                "include_detection": False,
                "include_segmentation": False,
            },
        },
        "segmentation": {
            "train": False,
            "detection_conf_threshold": 0.5,
            "padding_inference": 0.2,
        },
        "classification": {
            "enabled": True,
        },
        "output": {
            "results_root": str(tmp_path / "results"),
            "skip_pip_freeze": True,
        },
    }

    runner = PipelineRunner(cfg)
    runner.run()

    assert "test_predictions" in captured
    per_model = captured["test_predictions"]["FakeSparseDetector"]
    assert str(video_path) in per_model
    preds = per_model[str(video_path)]
    assert [int(p.frame_index) for p in preds] == [0, 1, 2, 3, 4, 5]
