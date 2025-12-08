from __future__ import annotations

from pathlib import Path

from tracking.data.dataset_manager import COCOJsonDatasetManager


def _make_manager(tmp_path: Path) -> COCOJsonDatasetManager:
    manager = COCOJsonDatasetManager.__new__(COCOJsonDatasetManager)
    manager.root = str(tmp_path)
    return manager


def test_numeric_prefix_extracted_from_filename(tmp_path):
    manager = _make_manager(tmp_path)
    video = tmp_path / "001Rest post.avi"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.touch()

    subject = manager._derive_subject(str(video))

    assert subject == "001"


def test_nested_directory_preserves_folder_subject(tmp_path):
    manager = _make_manager(tmp_path)
    subdir = tmp_path / "subjectA"
    video = subdir / "Grasp.avi"
    subdir.mkdir(parents=True, exist_ok=True)
    video.touch()

    subject = manager._derive_subject(str(video))

    assert subject == "subjectA"


def test_numeric_directory_still_groups_by_digits(tmp_path):
    manager = _make_manager(tmp_path)
    subdir = tmp_path / "002"
    video = subdir / "Relax.avi"
    subdir.mkdir(parents=True, exist_ok=True)
    video.touch()

    subject = manager._derive_subject(str(video))

    assert subject == "002"
