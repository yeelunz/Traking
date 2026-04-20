from __future__ import annotations

from tracking.core.interfaces import FramePrediction
from tracking.utils.prediction_interpolation import (
    cubic_clip_interpolate_predictions,
    repair_predictions_for_query_frames,
)


def test_interpolate_small_gap_marks_fallback_and_source():
    preds = [
        FramePrediction(frame_index=0, bbox=(10.0, 10.0, 30.0, 30.0), score=0.9),
        FramePrediction(frame_index=2, bbox=(20.0, 20.0, 30.0, 30.0), score=0.9),
    ]

    out = cubic_clip_interpolate_predictions(preds, max_gap=5, interpolated_bbox_source="interpolated_cubic")

    assert [p.frame_index for p in out] == [0, 1, 2]
    mid = out[1]
    assert mid.is_fallback is True
    assert mid.bbox_source == "interpolated_cubic"


def test_interpolate_respects_max_gap():
    preds = [
        FramePrediction(frame_index=0, bbox=(10.0, 10.0, 30.0, 30.0), score=0.9),
        FramePrediction(frame_index=10, bbox=(20.0, 20.0, 30.0, 30.0), score=0.9),
    ]

    out = cubic_clip_interpolate_predictions(preds, max_gap=3)

    assert [p.frame_index for p in out] == [0, 10]


def test_interpolate_is_clipped_to_known_range_per_dimension():
    preds = [
        FramePrediction(frame_index=0, bbox=(0.0, 10.0, 20.0, 30.0), score=1.0),
        FramePrediction(frame_index=3, bbox=(100.0, 40.0, 50.0, 60.0), score=1.0),
        FramePrediction(frame_index=6, bbox=(0.0, 10.0, 20.0, 30.0), score=1.0),
        FramePrediction(frame_index=9, bbox=(100.0, 40.0, 50.0, 60.0), score=1.0),
    ]

    out = cubic_clip_interpolate_predictions(preds, max_gap=5)

    assert [p.frame_index for p in out] == list(range(10))
    for p in out:
        x, y, w, h = p.bbox
        assert 0.0 <= x <= 100.0
        assert 10.0 <= y <= 40.0
        assert 20.0 <= w <= 50.0
        assert 30.0 <= h <= 60.0


def test_repair_predictions_interpolates_low_confidence_when_anchor_count_exceeds_two():
    preds = [
        FramePrediction(frame_index=0, bbox=(0.0, 0.0, 10.0, 10.0), score=0.9, confidence=0.9),
        FramePrediction(frame_index=1, bbox=(20.0, 20.0, 10.0, 10.0), score=0.1, confidence=0.1),
        FramePrediction(frame_index=2, bbox=(40.0, 40.0, 10.0, 10.0), score=0.9, confidence=0.9),
        FramePrediction(frame_index=3, bbox=(60.0, 60.0, 10.0, 10.0), score=0.9, confidence=0.9),
    ]

    out = repair_predictions_for_query_frames(
        preds,
        query_frame_indices=[0, 1, 2, 3],
        confidence_threshold=0.5,
        min_known_points=3,
        interpolated_bbox_source="interpolated_pchip",
    )

    assert [p.frame_index for p in out] == [0, 1, 2, 3]
    assert out[1].bbox == (20.0, 20.0, 10.0, 10.0)
    assert out[1].bbox_source == "interpolated_pchip"
    assert out[1].is_fallback is True


def test_repair_predictions_keeps_missing_when_known_points_not_enough():
    preds = [
        FramePrediction(frame_index=0, bbox=(0.0, 0.0, 10.0, 10.0), score=0.9, confidence=0.9),
        FramePrediction(frame_index=1, bbox=(0.0, 0.0, 0.0, 0.0), score=None, confidence=0.0, is_fallback=True, bbox_source="missing_detector"),
        FramePrediction(frame_index=2, bbox=(40.0, 40.0, 10.0, 10.0), score=0.9, confidence=0.9),
    ]

    out = repair_predictions_for_query_frames(
        preds,
        query_frame_indices=[0, 1, 2],
        confidence_threshold=0.5,
        min_known_points=3,
    )

    assert [p.frame_index for p in out] == [0, 1, 2]
    assert out[1].bbox == (0.0, 0.0, 0.0, 0.0)
    assert out[1].bbox_source == "missing"
    assert out[1].is_fallback is True


def test_repair_predictions_fills_all_query_frames_including_boundaries():
    preds = [
        FramePrediction(frame_index=5, bbox=(10.0, 10.0, 10.0, 10.0), score=0.9, confidence=0.9),
        FramePrediction(frame_index=10, bbox=(20.0, 20.0, 10.0, 10.0), score=0.9, confidence=0.9),
        FramePrediction(frame_index=15, bbox=(30.0, 30.0, 10.0, 10.0), score=0.9, confidence=0.9),
    ]

    out = repair_predictions_for_query_frames(
        preds,
        query_frame_indices=[0, 5, 10, 15, 20],
        confidence_threshold=0.5,
        min_known_points=3,
        interpolated_bbox_source="interpolated_pchip",
    )

    assert [p.frame_index for p in out] == [0, 5, 10, 15, 20]
    assert out[0].bbox_source == "interpolated_pchip"
    assert out[0].is_fallback is True
    assert out[-1].bbox_source == "interpolated_pchip"
    assert out[-1].is_fallback is True
