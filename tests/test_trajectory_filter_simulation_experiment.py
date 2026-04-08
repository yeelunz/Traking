from __future__ import annotations

import numpy as np

from tools.trajectory_filter_simulation_experiment import (
    FilterConfig,
    add_jitter,
    build_gt_library,
    evaluate_config,
    inject_pattern,
    run_filter_pipeline,
)


def test_pattern1_swiss_cheese_has_alternating_missing_segment():
    gt = build_gt_library()["A_line"]
    rng = np.random.default_rng(123)
    normal = add_jitter(gt, rng=rng, sigma=0.7)

    raw, info = inject_pattern(normal, "P1_swiss_cheese", rng=np.random.default_rng(123))
    missing = np.asarray(info["missing_mask"], dtype=bool)

    assert missing.sum() >= 10
    assert np.isnan(raw[missing]).all()


def test_filter_pipeline_recovers_injected_isolated_outlier():
    gt = build_gt_library()["A_line"]
    rng = np.random.default_rng(456)
    normal = add_jitter(gt, rng=rng, sigma=0.6)
    raw, info = inject_pattern(normal, "P3_isolated_spikes", rng=np.random.default_rng(456))

    cfg = FilterConfig(
        macro_ratio=0.06,
        macro_sigma=2.0,
        micro_hw=2,
        micro_sigma=2.0,
        max_outlier_run=1,
        sg_window=5,
        sg_polyorder=2,
        anchor_keep_ratio=0.35,
    )
    out = run_filter_pipeline(raw, cfg)
    injected = np.asarray(info["injected_outlier_mask"], dtype=bool)

    assert injected.any()
    raw_err = np.linalg.norm(raw[injected] - gt[injected], axis=1).mean()
    filtered_err = np.linalg.norm(out["filtered"][injected] - gt[injected], axis=1).mean()
    assert filtered_err < raw_err


def test_evaluate_config_clean_data_do_no_harm_is_controlled():
    cfg = FilterConfig(
        macro_ratio=0.06,
        macro_sigma=2.0,
        micro_hw=2,
        micro_sigma=2.0,
        max_outlier_run=1,
        sg_window=5,
        sg_polyorder=2,
        anchor_keep_ratio=0.35,
    )
    gt_library = build_gt_library()
    pattern_names = ["P5_clean_only_jitter"]

    agg, rows, _ = evaluate_config(cfg, gt_library, pattern_names, seed=3407)

    assert len(rows) == 3
    assert float(agg["normal_err_delta"]) < 0.3
    assert float(agg["normal_harm_rate"]) < 0.6
