from pathlib import Path

from tracking.classification.engine import _build_texture_pretrain_auto_root


def test_texture_pretrain_auto_root_is_stable_across_result_dirs():
    train_videos = {
        r"C:\dataset\merged_extend\001\A.avi": [],
        r"C:\dataset\merged_extend\002\B.avi": [],
    }

    root1 = _build_texture_pretrain_auto_root(
        cache_dir=Path(r"C:\tmp\texture_cache"),
        dataset_root=r"C:\dataset\merged_extend",
        feature_name="tab_v4",
        train_videos=train_videos,
        backbone="convnext_tiny",
        input_size=96,
        max_frames_per_video=16,
        val_ratio=0.2,
        roi_pad_ratio=0.15,
        seed=42,
    )
    root2 = _build_texture_pretrain_auto_root(
        cache_dir=Path(r"C:\tmp\texture_cache"),
        dataset_root=r"C:\dataset\merged_extend",
        feature_name="tab_v4",
        train_videos=train_videos,
        backbone="convnext_tiny",
        input_size=96,
        max_frames_per_video=16,
        val_ratio=0.2,
        roi_pad_ratio=0.15,
        seed=42,
    )

    assert str(root1) == str(root2)
    assert "auto_data" in str(root1)
    assert "results" not in str(root1).lower()


def test_texture_pretrain_auto_root_changes_when_train_videos_change():
    base_args = dict(
        cache_dir=Path(r"C:\tmp\texture_cache"),
        dataset_root=r"C:\dataset\merged_extend",
        feature_name="tab_v4",
        backbone="convnext_tiny",
        input_size=96,
        max_frames_per_video=16,
        val_ratio=0.2,
        roi_pad_ratio=0.15,
        seed=42,
    )

    root_a = _build_texture_pretrain_auto_root(
        train_videos={r"C:\dataset\merged_extend\001\A.avi": []},
        **base_args,
    )
    root_b = _build_texture_pretrain_auto_root(
        train_videos={
            r"C:\dataset\merged_extend\001\A.avi": [],
            r"C:\dataset\merged_extend\002\B.avi": [],
        },
        **base_args,
    )

    assert str(root_a) != str(root_b)
