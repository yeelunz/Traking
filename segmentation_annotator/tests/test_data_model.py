from __future__ import annotations

import json

import cv2
import numpy as np

from segmentation_annotator.data import MASK_ROOT_DEFAULT, SegmentationProject

from segmentation_annotator.data import MaskMetadata, MotionSample, TrackSummary


def test_mask_metadata_from_mask_simple_square():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:8, 3:9] = 255
    meta = MaskMetadata.from_mask(mask)
    assert meta.area_px == 36.0
    assert meta.bbox == (3.0, 2.0, 6.0, 6.0)
    assert meta.centroid == (5.5, 4.5)
    assert meta.perimeter_px > 0
    assert meta.equivalent_diameter_px > 0


def test_track_summary_speed_and_length():
    track = TrackSummary(track_id=1, category_id=1)
    track.samples.extend(
        [
            MotionSample(frame_index=0, centroid=(0.0, 0.0), displacement=(0.0, 0.0), distance=0.0),
            MotionSample(frame_index=1, centroid=(3.0, 4.0), displacement=(3.0, 4.0), distance=5.0),
            MotionSample(frame_index=2, centroid=(6.0, 8.0), displacement=(3.0, 4.0), distance=5.0),
        ]
    )
    assert track.total_path_length == 10.0
    assert track.average_speed_per_frame == 10.0 / 3.0


def _make_dummy_video(path, *, width: int = 8, height: int = 8, frames: int = 2) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (width, height))
    try:
        for i in range(frames):
            frame = np.full((height, width, 3), i * 10, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def test_segmentation_project_enforces_single_mask_per_frame(tmp_path):
    video_path = tmp_path / "sample.mp4"
    _make_dummy_video(video_path)

    project = SegmentationProject(str(video_path), ["nerve"])
    try:
        mask_a = np.zeros((project.height, project.width), dtype=np.uint8)
        mask_a[1:5, 1:5] = 255
        ann_first = project.new_annotation(0, 1, mask_a)

        mask_b = np.zeros((project.height, project.width), dtype=np.uint8)
        mask_b[2:6, 2:6] = 255
        ann_second = project.new_annotation(0, 1, mask_b)

        frame_store = project.annotations_by_frame[0]
        assert len(frame_store) == 1
        stored_ann = next(iter(frame_store.values()))
        assert stored_ann.track_id == ann_first.track_id == ann_second.track_id == project.primary_track_id

        mask_c = np.zeros((project.height, project.width), dtype=np.uint8)
        mask_c[0:3, 0:3] = 255
        project.new_annotation(1, 1, mask_c)
        assert project.annotations_by_frame[1][project.primary_track_id].track_id == project.primary_track_id
    finally:
        project.close()


def test_segmentation_project_clear_frame(tmp_path):
    video_path = tmp_path / "sample.mp4"
    _make_dummy_video(video_path)

    project = SegmentationProject(str(video_path), ["nerve"])
    try:
        mask = np.zeros((project.height, project.width), dtype=np.uint8)
        mask[1:5, 1:5] = 255
        project.new_annotation(0, 1, mask)
        assert 0 in project.annotations_by_frame

        project.clear_frame(0)
        assert 0 not in project.annotations_by_frame
        assert project.get_annotations(0) == {}
    finally:
        project.close()


def test_export_dataset_flat_structure(tmp_path):
    video_path = tmp_path / "sample.mp4"
    _make_dummy_video(video_path, frames=1)

    project = SegmentationProject(str(video_path), ["nerve"])
    try:
        mask = np.zeros((project.height, project.width), dtype=np.uint8)
        mask[1:5, 1:5] = 255
        project.new_annotation(0, 1, mask)

        out_dir = tmp_path / "export"
        json_path = project.export_dataset(str(out_dir))

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["annotations"], "annotations should not be empty"
        mask_path = data["annotations"][0]["mask_path"]
        mask_file = out_dir / mask_path
        assert mask_file.exists()

        expected = f"{MASK_ROOT_DEFAULT}/{project.video_name_without_ext}/frame_00000001.png"
        assert mask_path == expected
    finally:
        project.close()