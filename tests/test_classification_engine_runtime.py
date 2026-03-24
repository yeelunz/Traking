from __future__ import annotations

from tracking.preproc import clahe  # noqa: F401
from tracking.classification.engine import (
    _build_runtime_preprocs,
    _exclude_loso_subjects,
    _subject_from_video_path,
)


def test_subject_from_video_path_normalises_directory_prefix(tmp_path):
    root = tmp_path / "dataset"
    video = root / "001_subjectA" / "Grasp.avi"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.touch()

    subj = _subject_from_video_path(str(video), str(root))
    assert subj == "001"


def test_runtime_preproc_routing_hybrid_scheme():
    cfg = {
        "runtime_preprocessing": {
            "scheme": "C",
            "preproc_steps": [
                {"name": "CLAHE", "params": {"clip_limit": 2.0}},
            ],
        }
    }

    global_preprocs, roi_preprocs = _build_runtime_preprocs(cfg)
    assert len(global_preprocs) == 0
    assert len(roi_preprocs) == 1


def test_exclude_loso_subjects_removes_blocked_subjects():
    entities = ["001/A", "002/B", "003/C"]
    owner = {"001/A": "001", "002/B": "002", "003/C": "003"}

    kept, removed, overlap = _exclude_loso_subjects(entities, owner, ["002"])

    assert kept == ["001/A", "003/C"]
    assert removed == 1
    assert overlap == []
