from __future__ import annotations

from tracking.core.interfaces import FramePrediction
from tracking.utils.prediction_interpolation import cubic_clip_interpolate_predictions


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
