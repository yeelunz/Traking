from __future__ import annotations

import numpy as np

from tools.trajectory_repair_benchmark import (
    BenchmarkConfig,
    TrajectoryRecord,
    _bbox_to_center_xywh,
    _center_xywh_to_bbox,
    _contiguous_segments,
    _corrupt_bbox,
    _detect_outliers,
    _evaluate_prediction,
    _make_tab_context_bbox,
    _make_proxy_bbox,
    _reconstruct_with_poisson,
    run_benchmark,
    split_trajectories,
)


def _make_record(name: str, n: int = 40) -> TrajectoryRecord:
    frame_index = np.arange(n, dtype=np.int64)
    cx = 50.0 + frame_index * 1.5
    cy = 80.0 + frame_index * 0.5
    w = np.full(n, 20.0)
    h = np.full(n, 10.0)
    bbox = _center_xywh_to_bbox(cx, cy, w, h)
    observed = np.ones(n, dtype=bool)
    return TrajectoryRecord(
        trajectory_id=name,
        source_json=f"{name}.json",
        video_name=name,
        frame_index=frame_index,
        gt_bbox_dense=bbox,
        gt_bbox_observed=bbox.copy(),
        observed_mask=observed,
    )


def test_split_trajectories_keeps_whole_trajectory_units():
    records = [_make_record(f"t{i}") for i in range(10)]
    train, test = split_trajectories(records, train_ratio=0.8, seed=3407)

    assert len(train) == 8
    assert len(test) == 2
    assert {r.trajectory_id for r in train}.isdisjoint({r.trajectory_id for r in test})


def test_corrupt_bbox_respects_total_budget():
    rec = _make_record("demo", n=60)
    out = _corrupt_bbox(rec.gt_bbox_dense, rec.frame_index, rng=np.random.default_rng(123), max_ratio=0.30)

    corrupt_mask = np.asarray(out["corrupt_mask"], dtype=bool)
    assert corrupt_mask.sum() <= int(np.floor(len(rec.frame_index) * 0.30))
    assert set(out["pattern_names"]).issubset({"initial_drift", "spike", "short_drift", "two_way_drift", "random_missing"})


def test_poisson_reconstruction_recovers_constant_step():
    n = 25
    dx = np.ones(n, dtype=np.float64) * 2.0
    dx[0] = 0.0
    trusted = np.arange(n, dtype=np.float64) * 2.0 + 10.0
    trusted_mask = np.zeros(n, dtype=bool)
    trusted_mask[[0, 5, 10, 15, 20]] = True

    out = _reconstruct_with_poisson(dx, trusted, trusted_mask)
    expected = np.arange(n, dtype=np.float64) * 2.0 + 10.0
    assert np.allclose(out, expected, atol=1e-6)


def test_proxy_bbox_fills_missing_values():
    rec = _make_record("demo2", n=20)
    raw = rec.gt_bbox_dense.copy()
    raw[[3, 4, 5]] = np.nan

    proxy = _make_proxy_bbox(raw, rec.frame_index)
    assert np.isfinite(proxy).all()


def test_evaluate_prediction_is_perfect_for_identical_boxes():
    rec = _make_record("demo3", n=20)
    center = np.column_stack((50.0 + rec.frame_index * 1.5, 80.0 + rec.frame_index * 0.5))
    metrics = _evaluate_prediction(
        "same",
        rec.gt_bbox_dense,
        center,
        np.zeros(len(rec.frame_index), dtype=np.float64),
        np.zeros(len(rec.frame_index), dtype=bool),
        rec.gt_bbox_dense,
        np.zeros(len(rec.frame_index), dtype=bool),
    )

    assert metrics["iou_mean"] == 1.0
    assert metrics["mse"] == 0.0
    assert metrics["mae"] == 0.0


def test_contiguous_segments_groups_runs():
    segs = _contiguous_segments(np.array([0, 1, 1, 0, 1, 0, 1, 1, 1], dtype=bool))
    assert segs == [(1, 3), (4, 5), (6, 9)]


def test_detect_outliers_lowess_uses_observed_points_and_marks_missing():
    rec = _make_record("demo4", n=24)
    raw = rec.gt_bbox_dense.copy()
    raw[6] = np.nan
    raw[15, 0] += 40.0
    raw[15, 1] -= 25.0

    cfg = BenchmarkConfig(anomaly_residual_quantile=0.80, lowess_frac=0.45, robust_iterations=3)
    det = _detect_outliers(raw, rec.frame_index, cfg, detector="lowess", seed=123)

    assert det["observed_mask"][6] == 0
    assert det["flag_mask"][6]
    assert det["flag_mask"][15]


def test_run_benchmark_detector_mode_corrupts_detector_raw(monkeypatch, tmp_path):
    train_rec = _make_record("train_case", n=20)
    test_rec = _make_record("test_case", n=20)
    detector_raw = test_rec.gt_bbox_dense.copy()
    detector_raw[:, 0] += 7.0
    detector_raw[:, 1] -= 3.0
    train_raw = train_rec.gt_bbox_dense.copy()
    train_raw[:, 0] += 2.0

    train_rec = TrajectoryRecord(
        trajectory_id=train_rec.trajectory_id,
        source_json=train_rec.source_json,
        video_name=train_rec.video_name,
        frame_index=train_rec.frame_index,
        gt_bbox_dense=train_rec.gt_bbox_dense,
        gt_bbox_observed=train_rec.gt_bbox_observed,
        observed_mask=train_rec.observed_mask,
        raw_bbox_dense=train_raw,
        raw_observed_mask=np.isfinite(train_raw).all(axis=1),
    )
    test_rec = TrajectoryRecord(
        trajectory_id=test_rec.trajectory_id,
        source_json=test_rec.source_json,
        video_name=test_rec.video_name,
        frame_index=test_rec.frame_index,
        gt_bbox_dense=test_rec.gt_bbox_dense,
        gt_bbox_observed=test_rec.gt_bbox_observed,
        observed_mask=test_rec.observed_mask,
        raw_bbox_dense=detector_raw,
        raw_observed_mask=np.isfinite(detector_raw).all(axis=1),
    )

    def _simple_result(raw_bbox: np.ndarray, frame_index: np.ndarray) -> dict[str, np.ndarray]:
        bbox = _make_tab_context_bbox(raw_bbox)
        cx, cy, _, _ = _bbox_to_center_xywh(bbox)
        return {
            "bbox": bbox,
            "center": np.column_stack([cx, cy]),
            "anomaly_score": np.zeros(len(frame_index), dtype=np.float64),
            "flag_mask": np.zeros(len(frame_index), dtype=bool),
            "deleted_bbox": raw_bbox.copy(),
        }

    captured_inputs: list[np.ndarray] = []

    def fake_load_detector_trajectories(root, cfg, cache_tag):
        root_str = str(root)
        if root_str.endswith("train_root"):
            return [train_rec], {"checkpoint": "stub.pt", "cache_root": "train_cache", "model_name": "stub"}
        return [test_rec], {"checkpoint": "stub.pt", "cache_root": "test_cache", "model_name": "stub"}

    def fake_corrupt_bbox(bbox, frame_index, rng, max_ratio, cfg):
        captured_inputs.append(np.asarray(bbox, dtype=np.float64).copy())
        return {
            "raw_bbox": np.asarray(bbox, dtype=np.float64).copy(),
            "corrupt_mask": np.zeros(len(frame_index), dtype=bool),
            "missing_mask": np.zeros(len(frame_index), dtype=bool),
            "pattern_masks": {},
            "pattern_names": ["synthetic_stub"],
            "frame_index": np.asarray(frame_index, dtype=np.int64).copy(),
        }

    monkeypatch.setattr("tools.trajectory_repair_benchmark.load_detector_trajectories", fake_load_detector_trajectories)
    monkeypatch.setattr("tools.trajectory_repair_benchmark._corrupt_bbox", fake_corrupt_bbox)
    monkeypatch.setattr("tools.trajectory_repair_benchmark._fit_tabpfn_regressors", lambda train_X, train_y, cfg: {})
    monkeypatch.setattr("tools.trajectory_repair_benchmark._baseline_method", lambda raw_bbox, frame_index, cfg: _simple_result(raw_bbox, frame_index))
    monkeypatch.setattr("tools.trajectory_repair_benchmark._ransac_bspline_method", lambda raw_bbox, frame_index, cfg: _simple_result(raw_bbox, frame_index))
    monkeypatch.setattr("tools.trajectory_repair_benchmark._msac_poisson_method", lambda raw_bbox, frame_index, cfg, seed: _simple_result(raw_bbox, frame_index))
    monkeypatch.setattr(
        "tools.trajectory_repair_benchmark._tabpfn_method",
        lambda raw_bbox, frame_index, cfg, models, detector, seed: {
            "poisson": _simple_result(raw_bbox, frame_index),
            "bspline": _simple_result(raw_bbox, frame_index),
        },
    )
    monkeypatch.setattr(
        "tools.trajectory_repair_benchmark._detect_outliers",
        lambda raw_bbox, frame_index, cfg, detector, seed: {
            "proxy_bbox": _make_tab_context_bbox(raw_bbox),
            "anomaly_score": np.zeros(len(frame_index), dtype=np.float64),
            "flag_mask": np.zeros(len(frame_index), dtype=bool),
            "observed_mask": np.isfinite(raw_bbox).all(axis=1),
        },
    )

    summary = run_benchmark(
        BenchmarkConfig(
            dataset_root="unused",
            train_dataset_root="train_root",
            test_dataset_root="test_root",
            raw_source="detector",
            output_dir=str(tmp_path),
            target_test_cases=1,
            cases_per_trajectory=1,
        )
    )

    assert summary["raw_source"] == "detector"
    assert len(captured_inputs) >= 1
    assert np.allclose(captured_inputs[0], detector_raw)
