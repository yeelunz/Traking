import json
from pathlib import Path

from tracking.data.dataset_manager import COCOJsonDatasetManager, SimpleDataset


def _write_dummy_video(path: Path) -> None:
    path.write_bytes(b"fakevideo")


def _write_dummy_annotation(path: Path) -> None:
    payload = {
        "images": [],
        "annotations": [],
    }
    path.write_text(json.dumps(payload))


def test_dataset_manager_scans_recursively(tmp_path):
    dataset_root = tmp_path / "root"
    nested = dataset_root / "nested" / "inner"
    nested.mkdir(parents=True)
    annotated_video = nested / "clip.mp4"
    _write_dummy_video(annotated_video)
    _write_dummy_annotation(annotated_video.with_suffix(".json"))

    missing_video = dataset_root / "outer" / "missing.mp4"
    missing_video.parent.mkdir(parents=True)
    _write_dummy_video(missing_video)

    dm = COCOJsonDatasetManager(str(dataset_root))

    assert str(annotated_video) in dm.videos
    assert str(annotated_video) in dm.ann_by_video
    assert str(missing_video) in dm.videos
    assert str(missing_video) in dm.missing_annotations

    dataset = SimpleDataset(dm.videos, dm.ann_by_video)
    assert len(dataset) == 1
    item = dataset[0]
    assert item["video_path"].endswith("clip.mp4")
    assert isinstance(item["annotation"], dict)
