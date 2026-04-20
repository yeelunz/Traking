import json
from pathlib import Path

import pytest

try:
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover - optional test dependency
    TestClient = None

from tools.schedule_web_viewer.app import create_app, gather_segmentation_metrics


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_gather_segmentation_metrics_parses_mixed_separator_video_paths(tmp_path: Path):
    exp_path = tmp_path / "exp_case"

    _write_json(
        exp_path / "test" / "segmentation" / "metrics_summary.json",
        {"nnunet": {"YOLOv11": {"dice_mean": 0.61}}},
    )
    _write_json(
        exp_path
        / "test"
        / "segmentation"
        / "predictions"
        / "nnunet"
        / "YOLOv11"
        / "metrics_per_video.json",
        {
            r"dataset\extendclen\control101_left_68yo_female\control101_left_68yo_female.avi": {
                "centroid_mean": 2.1
            },
            "dataset/extendclen/normal003/rest_post.avi": {"centroid_mean": 3.4},
        },
    )

    metrics = gather_segmentation_metrics(exp_path)
    labels = {row.get("video") for row in metrics.get("per_video", [])}

    assert "control101_left_68yo_female/control101_left_68yo_female" in labels
    assert "normal003/rest_post" in labels


@pytest.mark.skipif(TestClient is None, reason="fastapi TestClient/httpx not available")
def test_segmentation_visuals_reads_hashed_output_dir_and_ce(tmp_path: Path):
    results_root = tmp_path / "results"
    exp_path = results_root / "exp_hash_case"

    _write_json(
        exp_path / "metadata.json",
        {
            "experiment": {"name": "exp_hash_case", "pipeline": []},
            "dataset": {"split": {"method": "holdout"}},
        },
    )
    _write_json(
        exp_path / "test" / "segmentation" / "metrics_summary.json",
        {"nnunet": {"YOLOv11": {"dice_mean": 0.57, "iou_mean": 0.41}}},
    )
    _write_json(
        exp_path
        / "test"
        / "segmentation"
        / "predictions"
        / "nnunet"
        / "YOLOv11"
        / "metrics_per_video.json",
        {
            "dataset/extendclen/control101_left_68yo_female/control101_left_68yo_female.avi": {
                "centroid_mean": 9.9
            }
        },
    )

    hashed_leaf = "v_07792c17bb48"
    video_dir = (
        exp_path
        / "test"
        / "segmentation"
        / "predictions"
        / "nnunet"
        / "YOLOv11"
        / hashed_leaf
    )
    vis_path = video_dir / "visualizations_roi" / "frame_000001_overlay.png"
    vis_path.parent.mkdir(parents=True, exist_ok=True)
    vis_path.write_bytes(b"fake-png")
    _write_json(video_dir / "metrics_per_frame.json", {"1": {"centroid": 1.23}})

    app = create_app(results_root)
    client = TestClient(app)

    response = client.get(
        "/api/experiments/visuals",
        params={
            "exp_id": "exp_hash_case",
            "category": "segmentation_overlays",
            "limit": 10,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload.get("items"), list)
    assert len(payload["items"]) == 1
    assert payload["items"][0].get("ce_px") == 1.23
