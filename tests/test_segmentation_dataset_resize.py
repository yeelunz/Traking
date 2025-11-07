from __future__ import annotations

import numpy as np
import torch

from tracking.segmentation.dataset import SegmentationCropDataset


class DummySegmentationDataset(SegmentationCropDataset):
    def _load_frame(self, video_path: str, frame_index: int):  # type: ignore[override]
        return np.full((200, 200, 3), 255, dtype=np.uint8)

    def _load_mask(self, video_path: str, mask_path: str | None):  # type: ignore[override]
        return np.full((200, 200), 255, dtype=np.uint8)


def _make_annotation(bboxes):
    frame_annotations = {}
    for idx, bbox in enumerate(bboxes):
        frame_annotations[str(idx)] = [
            {
                "bbox": bbox,
                "mask_path": "mask.png",
            }
        ]
    return {
        "frame_annotations": frame_annotations,
        "raw": {
            "videos": [
                {
                    "width": 200,
                    "height": 200,
                }
            ]
        },
    }


def test_dataset_resizes_crops_to_target_shape():
    video_path = "video.mp4"
    cache = {
        video_path: _make_annotation([
            (10, 10, 40, 60),
            (30, 20, 80, 50),
        ])
    }

    dataset = DummySegmentationDataset(
        [video_path],
        dataset_root=".",
        padding_range=(0.0, 0.0),
        redundancy=1,
        seed=0,
        cache_annotations=cache,
        jitter=0.0,
        target_size=(128, 96),
    )

    assert len(dataset) == 2
    first = dataset[0]
    second = dataset[1]

    assert first["image"].shape == (3, 128, 96)
    assert second["image"].shape == (3, 128, 96)
    assert first["mask"].shape == (1, 128, 96)
    assert second["mask"].shape == (1, 128, 96)

    assert first["original_roi_size"] != second["original_roi_size"]
    assert torch.isclose(first["mask"].max(), torch.tensor(1.0))
    assert torch.isclose(second["mask"].max(), torch.tensor(1.0))
