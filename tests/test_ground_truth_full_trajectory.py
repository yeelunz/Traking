from __future__ import annotations

import cv2
import numpy as np

from tracking.segmentation.dataset import attach_ground_truth_segmentation_full_trajectory


def _write_mask(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[2:10, 3:11] = 255
    assert cv2.imwrite(str(path), mask)


def test_gt_full_trajectory_interpolates_all_frames(tmp_path):
    dataset_root = tmp_path / "dataset"
    video_path = dataset_root / "001" / "Grasp.avi"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.touch()

    for name in ("f0.png", "f2.png", "f4.png"):
        _write_mask(dataset_root / "001" / name)

    annotation = {
        "frames": {
            0: [(0.0, 0.0, 10.0, 10.0)],
            1: [],
            2: [(20.0, 20.0, 10.0, 10.0)],
            3: [],
            4: [(40.0, 40.0, 10.0, 10.0)],
        },
        "frame_annotations": {
            0: [{"bbox": (0.0, 0.0, 10.0, 10.0), "mask_path": "001/f0.png", "metadata": {"area": 64.0}}],
            2: [{"bbox": (20.0, 20.0, 10.0, 10.0), "mask_path": "001/f2.png", "metadata": {"area": 64.0}}],
            4: [{"bbox": (40.0, 40.0, 10.0, 10.0), "mask_path": "001/f4.png", "metadata": {"area": 64.0}}],
        },
    }

    out = attach_ground_truth_segmentation_full_trajectory(
        annotation,
        str(dataset_root),
        video_path=str(video_path),
    )

    assert [p.frame_index for p in out] == [0, 1, 2, 3, 4]
    assert out[0].bbox_source == "ground_truth"
    assert out[1].bbox_source == "ground_truth_interpolated_pchip"
    assert out[3].bbox_source == "ground_truth_interpolated_pchip"
    assert out[1].segmentation is None
    assert out[3].segmentation is None
    assert out[0].segmentation is not None
    assert out[2].segmentation is not None
    assert out[4].segmentation is not None
    assert out[1].bbox == (10.0, 10.0, 10.0, 10.0)
    assert out[3].bbox == (30.0, 30.0, 10.0, 10.0)


def test_gt_full_trajectory_keeps_bbox_anchor_even_without_mask_file(tmp_path):
    dataset_root = tmp_path / "dataset"
    video_path = dataset_root / "001" / "Grasp.avi"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.touch()

    _write_mask(dataset_root / "001" / "f0.png")
    _write_mask(dataset_root / "001" / "f4.png")

    annotation = {
        "frames": {
            0: [(0.0, 0.0, 10.0, 10.0)],
            1: [],
            2: [(20.0, 20.0, 10.0, 10.0)],
            3: [],
            4: [(40.0, 40.0, 10.0, 10.0)],
        },
        "frame_annotations": {
            0: [{"bbox": (0.0, 0.0, 10.0, 10.0), "mask_path": "001/f0.png", "metadata": {"area": 64.0}}],
            2: [{"bbox": (20.0, 20.0, 10.0, 10.0), "mask_path": "001/missing.png", "metadata": {"area": 64.0}}],
            4: [{"bbox": (40.0, 40.0, 10.0, 10.0), "mask_path": "001/f4.png", "metadata": {"area": 64.0}}],
        },
    }

    out = attach_ground_truth_segmentation_full_trajectory(
        annotation,
        str(dataset_root),
        video_path=str(video_path),
    )

    assert [p.frame_index for p in out] == [0, 1, 2, 3, 4]
    frame2 = out[2]
    assert frame2.bbox == (20.0, 20.0, 10.0, 10.0)
    assert frame2.bbox_source == "ground_truth"
    assert frame2.segmentation is None
    assert out[1].bbox_source == "ground_truth_interpolated_pchip"
    assert out[3].bbox_source == "ground_truth_interpolated_pchip"
