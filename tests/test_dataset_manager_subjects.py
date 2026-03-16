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


def test_split_subject_level_allocates_by_subject_count(tmp_path):
    """Ensure _split_subject_level divides by number of subjects, not video count.

    With 5 subjects of varying video counts totaling 16 videos and 0.6 train
    ratio, we expect 3 train subjects and 2 test subjects (not 4:1 which
    video-count-based allocation would produce).
    """
    import json

    subjects = {
        "A": ["v1.avi", "v2.avi", "v3.avi", "v4.avi", "v5.avi"],  # 5 videos
        "B": ["x1.avi", "x2.avi", "x3.avi", "x4.avi"],            # 4 videos
        "C": ["y1.avi", "y2.avi", "y3.avi"],                       # 3 videos
        "D": ["z1.avi", "z2.avi"],                                  # 2 videos
        "E": ["w1.avi", "w2.avi"],                                  # 2 videos
    }
    for subj, vids in subjects.items():
        d = tmp_path / subj
        d.mkdir(parents=True, exist_ok=True)
        for v in vids:
            (d / v).touch()
            ann = {"frames": {"0": [[0, 0, 10, 10]]}}
            (d / (Path(v).stem + ".json")).write_text(json.dumps(ann))

    dm = COCOJsonDatasetManager(str(tmp_path))
    result = dm.split(method="subject_level", seed=0, ratios=(0.6, 0.0, 0.4))
    train_ds = result["train"]
    test_ds = result["test"]

    train_subjects = set()
    test_subjects = set()
    for i in range(len(train_ds)):
        vp = train_ds[i]["video_path"]
        train_subjects.add(dm.video_subjects.get(vp))
    for i in range(len(test_ds)):
        vp = test_ds[i]["video_path"]
        test_subjects.add(dm.video_subjects.get(vp))

    # 5 subjects * 0.6 = 3 train subjects, 2 test subjects
    assert len(train_subjects) == 3
    assert len(test_subjects) == 2
    # All videos accounted for
    assert len(train_ds) + len(test_ds) == 16
    # No overlap
    assert train_subjects.isdisjoint(test_subjects)
