from __future__ import annotations

import numpy as np

from tracking.classification.trajectory_filter import (
    _suppress_long_outlier_runs,
    filter_detections,
    filter_bbox_hampel_only,
    hampel_then_pchip_1d,
    smooth_trajectory_2d,
)


def test_hampel_then_pchip_interpolates_outlier_instead_of_kept_median():
    frame_indices = np.arange(5, dtype=np.int64)
    values = np.array([0.0, 1.0, 100.0, 3.0, 4.0], dtype=np.float64)

    filtered, marked, mask = hampel_then_pchip_1d(
        values,
        frame_indices,
        macro_ratio=0.2,
        macro_sigma=999.0,
        micro_hw=1,
        micro_sigma=1.0,
    )

    assert mask[2]
    assert np.isnan(marked[2])
    assert np.isclose(filtered[2], 2.0, atol=1e-6)


def test_smooth_trajectory_uses_fallback_like_regular_observation_by_default():
    frame_indices = np.arange(5, dtype=np.int64)
    obs = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [100.0, 100.0],
            [3.0, 3.0],
            [4.0, 4.0],
        ],
        dtype=np.float64,
    )

    smoothed = smooth_trajectory_2d(
        obs,
        frame_indices,
        macro_ratio=0.2,
        macro_sigma=999.0,
        micro_hw=1,
        micro_sigma=1.0,
        sg_window=3,
        sg_polyorder=1,
    )

    assert np.allclose(smoothed[2], np.array([2.0, 2.0]), atol=1e-3)


def test_filter_bbox_hampel_only_uses_fallback_like_regular_observation_by_default():
    frame_indices = np.arange(5, dtype=np.int64)
    widths = np.array([10.0, 11.0, 50.0, 13.0, 14.0], dtype=np.float64)
    heights = np.array([20.0, 21.0, 60.0, 23.0, 24.0], dtype=np.float64)

    w_out, h_out = filter_bbox_hampel_only(
        widths,
        heights,
        frame_indices=frame_indices,
        macro_ratio=0.2,
        macro_sigma=999.0,
        micro_hw=1,
        micro_sigma=1.0,
    )

    assert np.isclose(w_out[2], 12.0, atol=1e-3)
    assert np.isclose(h_out[2], 22.0, atol=1e-3)


def test_explicit_missing_mask_is_the_only_input_ignored_by_hampel_then_pchip():
    frame_indices = np.arange(5, dtype=np.int64)
    values = np.array([0.0, 1.0, 50.0, 3.0, 4.0], dtype=np.float64)
    observed_mask = np.array([True, True, False, True, True], dtype=bool)

    filtered, marked, _ = hampel_then_pchip_1d(
        values,
        frame_indices,
        observed_mask=observed_mask,
        macro_ratio=0.2,
        macro_sigma=999.0,
        micro_hw=1,
        micro_sigma=1.0,
    )

    assert np.isnan(marked[2])
    assert np.isclose(filtered[2], 2.0, atol=1e-6)


def test_suppress_long_outlier_runs_keeps_only_short_bursts():
    mask = np.array([False, True, True, True, False, True, False], dtype=bool)
    out = _suppress_long_outlier_runs(mask, max_outlier_run=2)
    assert np.array_equal(out, np.array([False, False, False, False, False, True, False], dtype=bool))


def test_filter_detections_anchor_mask_keeps_detector_frames_close_to_raw():
    fi = np.arange(5, dtype=np.int64)
    cx = np.array([100.0, 100.0, 130.0, 100.0, 100.0], dtype=np.float64)
    cy = np.array([50.0, 50.0, 50.0, 50.0, 50.0], dtype=np.float64)
    w = np.array([20.0, 20.0, 20.0, 20.0, 20.0], dtype=np.float64)
    h = np.array([10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float64)
    s = np.ones(5, dtype=np.float64)
    observed = np.ones(5, dtype=bool)
    anchor = np.array([False, False, True, False, False], dtype=bool)

    out = filter_detections(
        fi,
        cx,
        cy,
        w,
        h,
        s,
        bbox_strategy="hampel_only",
        bbox_params={"macro_ratio": 0.2, "macro_sigma": 999.0, "micro_hw": 1, "micro_sigma": 1.0},
        traj_params={"macro_ratio": 0.2, "macro_sigma": 999.0, "micro_hw": 1, "micro_sigma": 1.0, "sg_window": 3, "sg_polyorder": 1},
        observed_mask=observed,
        anchor_mask=anchor,
        anchor_keep_ratio=0.8,
    )

    # With 80% anchor-keep, detector frame 2 should be a raw/smoothed blend.
    assert np.isclose(out["cx"][2], 124.0, atol=1.0)
