from __future__ import annotations

import json
from pathlib import Path
import math

import pytest

from tracking.utils.confidence_scan import (
    FrameConfidenceSnapshot,
    SequenceConfidenceSummary,
    analyse_prediction_file,
    scan_schedule_confidence,
    summaries_to_csv,
)


@pytest.fixture()
def sample_predictions(tmp_path: Path) -> Path:
    prediction_dir = tmp_path / "schedule" / "exp_a" / "test" / "predictions"
    prediction_dir.mkdir(parents=True)
    data = [
        {"frame_index": 0, "bbox": [0, 0, 100, 100], "score": 1.0, "confidence": 0.92},
        {"frame_index": 1, "bbox": [2, 1, 100, 100], "score": 0.95, "confidence": 0.88},
        {"frame_index": 2, "bbox": [4, 2, 100, 100], "score": 0.9, "confidence": 0.84},
        {"frame_index": 3, "bbox": [90, 90, 100, 100], "score": 0.35, "confidence": 0.40},
        {"frame_index": 4, "bbox": [120, 120, 100, 100], "score": 0.2, "confidence": 0.25},
        {"frame_index": 5, "bbox": [160, 140, 100, 100], "score": 0.15, "confidence": 0.18},
    ]
    path = prediction_dir / "MixFormerV2.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh)
    return path


def test_analyse_prediction_file_detects_low_confidence(sample_predictions: Path):
    summary = analyse_prediction_file(
        sample_predictions,
        experiment="exp_a",
        tracker="MixFormerV2",
        low_threshold=0.6,
        top_k_frames=3,
    )
    assert summary is not None
    assert summary.total_frames == 6
    tracker_scores = [1.0, 0.95, 0.9, 0.35, 0.2, 0.15]
    assert math.isclose(summary.raw_score_mean, sum(tracker_scores) / len(tracker_scores), rel_tol=1e-6)
    # Should flag at least one low confidence frame due to drift and low score
    assert summary.below_threshold >= 1
    assert summary.longest_low_streak >= 1
    # Worst frame should correspond to the final frame with the low score
    worst_indices = [frame.frame_index for frame in summary.worst_frames]
    assert 5 in worst_indices
    # Aggregated stats should be within valid range
    assert 0.0 <= summary.confidence_min <= summary.confidence_mean <= 1.0
    assert summary.drift_pixels_p95 >= 0.0


def test_scan_schedule_confidence_returns_results(tmp_path: Path, sample_predictions: Path):
    # Add a second experiment to ensure iteration works
    exp_b = tmp_path / "schedule" / "exp_b" / "test" / "predictions"
    exp_b.mkdir(parents=True)
    with (exp_b / "MixFormerV2.json").open("w", encoding="utf-8") as fh:
        json.dump(
            [
                {"frame_index": 0, "bbox": [0, 0, 50, 50], "score": 0.8},
                {"frame_index": 1, "bbox": [1, 1, 50, 50], "score": 0.82},
            ],
            fh,
        )

    summaries = scan_schedule_confidence(tmp_path / "schedule", low_threshold=0.6)
    assert len(summaries) == 2
    experiments = {summary.experiment for summary in summaries}
    assert experiments == {"exp_a", "exp_b"}
    # Ensure results are sorted by low percentile confidence (exp_a more risky)
    assert summaries[0].experiment == "exp_a"
    assert summaries[0].confidence_p05 <= summaries[1].confidence_p05


def test_high_scores_still_penalised_by_drift(tmp_path: Path):
    pred_dir = tmp_path / "schedule" / "exp_drift" / "test" / "predictions"
    pred_dir.mkdir(parents=True)
    data = [
        {"frame_index": 0, "bbox": [0, 0, 120, 120], "score": 1.0},
        {"frame_index": 1, "bbox": [200, 0, 120, 120], "score": 1.0},
        {"frame_index": 2, "bbox": [400, 0, 120, 120], "score": 1.0},
        {"frame_index": 3, "bbox": [600, 0, 120, 120], "score": 1.0},
    ]
    with (pred_dir / "MixFormerV2.json").open("w", encoding="utf-8") as fh:
        json.dump(data, fh)

    summary = analyse_prediction_file(
        pred_dir / "MixFormerV2.json",
        experiment="exp_drift",
        tracker="MixFormerV2",
        low_threshold=0.75,
    )
    assert summary is not None
    # Raw score mean should still reflect the saturated tracker outputs
    assert summary.raw_score_mean == 1.0
    # Despite high raw scores, drift should push the blended confidence well below the low-confidence threshold
    assert summary.confidence_min < 0.85
    assert summary.drift_component_mean < 0.5
    assert summary.score_component_mean > summary.drift_component_mean


def test_analyse_prediction_file_uses_stored_components(tmp_path: Path):
    pred_dir = tmp_path / "schedule" / "exp_components" / "test" / "predictions"
    pred_dir.mkdir(parents=True)
    data = [
        {
            "frame_index": 0,
            "bbox": [0, 0, 50, 50],
            "score": 0.9,
            "confidence": 0.8,
            "components": {
                "raw_score": 0.85,
                "token": 0.7,
                "distribution": 0.6,
                "attention": 0.65,
                "short_iou": 0.9,
                "drift": 0.95,
                "blended": 0.82,
            },
        },
        {
            "frame_index": 1,
            "bbox": [1, 0, 50, 50],
            "score": 0.88,
            "confidence": 0.78,
            "components": {
                "raw_score": 0.83,
                "token": 0.72,
                "distribution": 0.58,
                "attention": 0.61,
                "short_iou": 0.88,
                "drift": 0.9,
                "blended": 0.79,
            },
        },
    ]
    with (pred_dir / "MixFormerV2.json").open("w", encoding="utf-8") as fh:
        json.dump(data, fh)

    summary = analyse_prediction_file(
        pred_dir / "MixFormerV2.json",
        experiment="exp_components",
        tracker="MixFormerV2",
    )
    assert summary is not None
    assert not math.isnan(summary.token_component_mean)
    assert math.isclose(summary.token_component_mean, 0.71, rel_tol=1e-6)
    assert math.isclose(summary.distribution_component_mean, 0.59, rel_tol=1e-6)
    assert math.isclose(summary.attention_component_mean, 0.63, rel_tol=1e-6)
    assert summary.confidence_mean == sum(entry["confidence"] for entry in data) / len(data)


def test_summaries_to_csv_exports_metrics(tmp_path: Path):
    summary_a = SequenceConfidenceSummary(
        experiment="exp_a",
        tracker="MixFormerV2",
        source_path=tmp_path / "exp_a" / "test" / "predictions" / "MixFormerV2.json",
        total_frames=6,
        confidence_mean=0.75,
        confidence_std=0.05,
        confidence_p10=0.62,
        confidence_p05=0.55,
        confidence_min=0.32,
        below_threshold=2,
        below_threshold_ratio=0.333333,
        longest_low_streak=2,
        score_component_mean=0.8,
        token_component_mean=0.7,
        distribution_component_mean=0.65,
        attention_component_mean=0.6,
        short_iou_component_mean=0.6,
        drift_component_mean=0.4,
        drift_pixels_mean=12.5,
        drift_pixels_p95=30.0,
        raw_score_mean=0.95,
        raw_score_p10=0.90,
        worst_frames=[
            FrameConfidenceSnapshot(
                frame_index=4,
                confidence=0.33,
                raw_score=0.35,
                components={"score": 0.4, "short_iou": 0.3, "drift": 0.5, "blended": 0.4},
                drift_pixels=25.0,
            )
        ],
    )
    summary_b = SequenceConfidenceSummary(
        experiment="exp_b",
        tracker="MixFormerV2",
        source_path=tmp_path / "exp_b" / "test" / "predictions" / "MixFormerV2.json",
        total_frames=2,
        confidence_mean=0.9,
        confidence_std=0.02,
        confidence_p10=0.88,
        confidence_p05=0.87,
        confidence_min=0.86,
        below_threshold=0,
        below_threshold_ratio=0.0,
        longest_low_streak=0,
        score_component_mean=0.92,
        token_component_mean=math.nan,
        distribution_component_mean=math.nan,
        attention_component_mean=math.nan,
        short_iou_component_mean=0.95,
        drift_component_mean=0.97,
        drift_pixels_mean=1.2,
        drift_pixels_p95=1.5,
        raw_score_mean=None,
        raw_score_p10=None,
        worst_frames=[],
    )

    csv_text = summaries_to_csv([summary_a, summary_b])
    lines = csv_text.strip().splitlines()
    assert len(lines) == 3  # header + 2 rows
    header = lines[0].split(",")
    assert header[:5] == [
        "experiment",
        "tracker",
        "total_frames",
        "confidence_mean",
        "confidence_p10",
    ]
    first_row = lines[1].split(",")
    assert first_row[0] == "exp_a"
    assert first_row[2] == "6"
    assert first_row[3] == "0.750000"
    assert first_row[7] == "2"
    assert first_row[8] == "0.333333"
    second_row = lines[2].split(",")
    assert second_row[0] == "exp_b"
    # raw score columns should be blank when None
    assert second_row[18] == ""
    assert second_row[19] == ""