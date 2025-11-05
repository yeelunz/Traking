from __future__ import annotations

import math

import numpy as np

from tracking.utils.confidence import ConfidenceConfig, ConfidenceEstimator, ConfidenceSignals
HIGH_TOKEN = np.ones((4, 8), dtype=np.float32)
NOISY_TOKEN = np.array(
    [
        [0.1, -1.2, 0.7, 0.5, -0.4, 0.8, -0.6, 1.1],
        [-0.9, 0.3, -0.5, 1.4, 0.6, -1.0, 0.2, 0.9],
        [0.5, -0.7, 1.2, -1.1, 0.8, -0.3, 0.4, -0.8],
        [-1.3, 0.9, -0.2, 0.6, -0.7, 1.0, -0.4, 0.3],
    ],
    dtype=np.float32,
)


def make_signals(*, token: np.ndarray | None = None, edge_left: tuple[float, float] | None = None) -> ConfidenceSignals:
    edge_distributions = None
    if edge_left is not None:
        edge_distributions = {"left": np.array(edge_left, dtype=np.float32)}
    return ConfidenceSignals(token_vectors=token, edge_distributions=edge_distributions)


def test_confidence_matches_requested_logistic_combination():
    est = ConfidenceEstimator()
    bbox = (0.0, 0.0, 100.0, 80.0)
    signals = make_signals(token=HIGH_TOKEN, edge_left=(0.95, 0.05))

    state = est.update(0, bbox, None, signals=signals)
    components = state.raw_components
    expected_logit = (
        est.cfg.distribution_logit_weight * components["distribution"]
        + est.cfg.drift_logit_weight * components["drift"]
        + est.cfg.token_logit_weight * components["token"]
    )
    expected_conf = 1.0 / (1.0 + math.exp(-expected_logit))

    assert math.isclose(components["blended"], expected_conf, rel_tol=1e-6)
    assert math.isclose(state.confidence, expected_conf, rel_tol=1e-6)


def test_confidence_penalises_large_drift_after_smoothing():
    est = ConfidenceEstimator()
    bbox = (0.0, 0.0, 50.0, 50.0)
    signals = make_signals(token=HIGH_TOKEN, edge_left=(0.95, 0.05))

    baseline = est.update(0, bbox, None, signals=signals)
    far_bbox = (400.0, 400.0, 50.0, 50.0)
    state = est.update(1, far_bbox, None, signals=signals)

    components = state.raw_components
    blended = 1.0 / (
        1.0
        + math.exp(
            -(
                est.cfg.distribution_logit_weight * components.get("distribution", 0.0)
                + est.cfg.drift_logit_weight * components.get("drift", 0.0)
                + est.cfg.token_logit_weight * components.get("token", 0.0)
            )
        )
    )
    expected_conf = (1.0 - est.cfg.smoothing_alpha) * baseline.confidence + est.cfg.smoothing_alpha * blended

    assert state.confidence < baseline.confidence
    assert math.isclose(state.raw_components["blended"], blended, rel_tol=1e-6)
    assert math.isclose(state.confidence, expected_conf, rel_tol=1e-6)


def test_high_quality_signals_outperform_noisy_signals():
    bbox = (0.0, 0.0, 60.0, 60.0)
    est_good = ConfidenceEstimator()
    good = est_good.update(0, bbox, None, signals=make_signals(token=HIGH_TOKEN, edge_left=(0.9, 0.1)))

    est_bad = ConfidenceEstimator()
    est_bad.update(0, bbox, None)
    noisy_signals = make_signals(token=NOISY_TOKEN, edge_left=(0.5, 0.5))
    bad = est_bad.update(1, bbox, None, signals=noisy_signals)

    assert good.confidence > bad.confidence


def test_missing_signals_default_to_mid_confidence():
    est = ConfidenceEstimator()
    bbox = (0.0, 0.0, 40.0, 40.0)
    state = est.update(0, bbox, None)
    expected = 1.0 / (1.0 + math.exp(-est.cfg.drift_logit_weight))
    assert math.isclose(state.confidence, expected, rel_tol=1e-6)


def test_evaluate_preview_does_not_mutate_state():
    est = ConfidenceEstimator()
    bbox = (0.0, 0.0, 50.0, 40.0)
    baseline = est.update(0, bbox, None, signals=make_signals(token=HIGH_TOKEN, edge_left=(0.9, 0.1)))

    preview = est.evaluate(frame_index=1, bbox=bbox, raw_score=None, signals=None, commit=False)
    committed = est.update(1, bbox, None, signals=None)

    assert math.isclose(preview.confidence, committed.confidence, rel_tol=1e-6)
    assert math.isclose(committed.previous_confidence, baseline.confidence, rel_tol=1e-6)
