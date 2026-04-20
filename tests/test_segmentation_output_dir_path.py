import os

from tracking.segmentation.workflow import _build_prediction_output_dir


def test_output_dir_avoids_duplicate_subject_video_leaf():
    output_root = os.path.join("results", "test", "segmentation", "predictions", "nnunet", "YOLOv11")
    video_path = os.path.join("dataset", "extendclen", "control101_left_68yo_female", "control101_left_68yo_female.avi")

    out_dir = _build_prediction_output_dir(output_root, video_path)

    # Should end with only one copy of the same identifier.
    expected_suffix = os.path.join("control101_left_68yo_female")
    assert out_dir.endswith(expected_suffix)
    assert not out_dir.endswith(os.path.join(expected_suffix, "control101_left_68yo_female"))


def test_output_dir_windows_compacts_overly_long_path(monkeypatch):
    monkeypatch.setattr(os, "name", "nt")

    output_root = os.path.join("C:\\", *("very_long_root_segment" for _ in range(8)))
    video_path = os.path.join(
        "dataset",
        "extendclen",
        "subject_with_really_really_long_identifier_name",
        "video_with_really_really_long_identifier_name.avi",
    )

    out_dir = _build_prediction_output_dir(output_root, video_path)

    assert len(os.path.abspath(out_dir)) <= 220
