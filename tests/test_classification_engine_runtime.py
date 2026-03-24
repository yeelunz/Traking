from __future__ import annotations

from tracking.preproc import clahe  # noqa: F401
from tracking.classification.engine import (
    _build_texture_pretrain_auto_root,
    _build_runtime_preprocs,
    _ensure_texture_pretrain_ckpt,
    _inject_runtime_preprocessing_into_feature_cfg,
    _exclude_loso_subjects,
    _subject_from_video_path,
)
from tracking.core.registry import FEATURE_EXTRACTOR_REGISTRY
import tracking.classification.feature_extractors_ext  # noqa: F401


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


def test_runtime_preproc_injected_into_legacy_texture_extractor():
    cfg = {
        "runtime_preprocessing": {
            "scheme": "C",
            "preproc_steps": [
                {"name": "CLAHE", "params": {"clip_limit": 2.0}},
            ],
        }
    }
    feature_cfg = {"name": "tab_v2", "params": {}}

    injected = _inject_runtime_preprocessing_into_feature_cfg(cfg, feature_cfg)
    ext_cls = FEATURE_EXTRACTOR_REGISTRY["tab_v2"]
    ext = ext_cls(injected["params"])

    assert len(ext._runtime_global_preprocs) == 0
    assert len(ext._runtime_roi_preprocs) == 1


def test_texture_pretrain_auto_root_is_shared_across_feature_families(tmp_path):
    train_videos = {
        str(tmp_path / "dataset" / "001" / "Grasp.avi"): [],
        str(tmp_path / "dataset" / "002" / "Rest.avi"): [],
    }
    runtime_cfg = {
        "scheme": "C",
        "preproc_steps": [{"name": "CLAHE", "params": {"clip_limit": 2.0}}],
    }

    root_a = _build_texture_pretrain_auto_root(
        cache_dir=tmp_path / "cache",
        dataset_root=str(tmp_path / "dataset"),
        train_videos=train_videos,
        backbone="convnext_tiny",
        input_size=96,
        max_frames_per_video=16,
        val_ratio=0.2,
        roi_pad_ratio=0.15,
        seed=42,
        runtime_preprocessing=runtime_cfg,
    )
    root_b = _build_texture_pretrain_auto_root(
        cache_dir=tmp_path / "cache",
        dataset_root=str(tmp_path / "dataset"),
        train_videos=train_videos,
        backbone="convnext_tiny",
        input_size=96,
        max_frames_per_video=16,
        val_ratio=0.2,
        roi_pad_ratio=0.15,
        seed=42,
        runtime_preprocessing=runtime_cfg,
    )

    assert root_a == root_b


def test_texture_pretrain_auto_root_changes_when_runtime_preproc_changes(tmp_path):
    train_videos = {
        str(tmp_path / "dataset" / "001" / "Grasp.avi"): [],
    }
    root_a = _build_texture_pretrain_auto_root(
        cache_dir=tmp_path / "cache",
        dataset_root=str(tmp_path / "dataset"),
        train_videos=train_videos,
        backbone="convnext_tiny",
        input_size=96,
        max_frames_per_video=16,
        val_ratio=0.2,
        roi_pad_ratio=0.15,
        seed=42,
        runtime_preprocessing={"scheme": "A", "preproc_steps": []},
    )
    root_b = _build_texture_pretrain_auto_root(
        cache_dir=tmp_path / "cache",
        dataset_root=str(tmp_path / "dataset"),
        train_videos=train_videos,
        backbone="convnext_tiny",
        input_size=96,
        max_frames_per_video=16,
        val_ratio=0.2,
        roi_pad_ratio=0.15,
        seed=42,
        runtime_preprocessing={
            "scheme": "C",
            "preproc_steps": [{"name": "CLAHE", "params": {"clip_limit": 2.0}}],
        },
    )

    assert root_a != root_b


def test_texture_pretrain_uses_feature_roi_pad_ratio_first(tmp_path, monkeypatch):
    feature_cfg = {
        "name": "tab_v4",
        "params": {
            "texture_mode": "pretrain",
            "texture_backbone": "convnext_tiny",
            "roi_pad_ratio": 0.33,
            "texture_pretrain_ckpt": str(tmp_path / "missing.pth"),
        },
    }
    classification_cfg = {
        "runtime_preprocessing": {"scheme": "A", "preproc_steps": []},
        "texture_pretrain": {
            "cache_dir": str(tmp_path / "cache"),
            "roi_pad_ratio": 0.15,
            "cache_enabled": True,
        },
    }
    train_videos = {str(tmp_path / "dataset" / "001" / "Grasp.avi"): []}
    labels = {"001": 1}

    captured = {}

    import tracking.classification.engine as engine_mod
    def _fake_auto_root(**kwargs):
        captured["roi_pad_ratio"] = kwargs["roi_pad_ratio"]
        raise RuntimeError("stop-after-capture")

    monkeypatch.setattr(engine_mod, "_build_texture_pretrain_auto_root", _fake_auto_root)

    try:
        updated = _ensure_texture_pretrain_ckpt(
            classification_cfg=classification_cfg,
            feature_cfg=feature_cfg,
            dataset_root=str(tmp_path / "dataset"),
            train_videos=train_videos,
            labels=labels,
            results_dir=str(tmp_path / "results"),
            logger=lambda _msg: None,
        )
    except RuntimeError as exc:
        assert str(exc) == "stop-after-capture"
        updated = None

    assert updated is None
    assert captured["roi_pad_ratio"] == 0.33


def test_exclude_loso_subjects_removes_blocked_subjects():
    entities = ["001/A", "002/B", "003/C"]
    owner = {"001/A": "001", "002/B": "002", "003/C": "003"}

    kept, removed, overlap = _exclude_loso_subjects(entities, owner, ["002"])

    assert kept == ["001/A", "003/C"]
    assert removed == 1
    assert overlap == []
