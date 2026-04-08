from __future__ import annotations

import numpy as np

from tracking.classification.lw_tab_p import LWTabPConfig, LWTabPFilter, _center_xywh_to_bbox, _fit_detector_curve_lowess


class _FakeModel:
    def __init__(self, values: np.ndarray):
        self._values = np.asarray(values, dtype=np.float64)

    def predict(self, features: np.ndarray) -> np.ndarray:
        n = len(features)
        if len(self._values) == n:
            return self._values.copy()
        if len(self._values) == 1:
            return np.full(n, float(self._values[0]), dtype=np.float64)
        raise AssertionError("unexpected feature length")


def _make_bbox(n: int = 24) -> tuple[np.ndarray, np.ndarray]:
    frame_index = np.arange(n, dtype=np.int64)
    cx = 100.0 + frame_index * 2.0
    cy = 60.0 + frame_index * 0.75
    w = np.full(n, 18.0, dtype=np.float64)
    h = np.full(n, 12.0, dtype=np.float64)
    return frame_index, _center_xywh_to_bbox(cx, cy, w, h)


def test_lowess_residuals_only_live_on_observed_points():
    frame_index, bbox = _make_bbox()
    cx = bbox[:, 0] + bbox[:, 2] / 2.0
    cx[5:8] = np.nan

    _, residual = _fit_detector_curve_lowess(frame_index, cx, frac=0.45, robust_iterations=3)

    assert np.all(residual[5:8] == 0.0)
    assert np.isfinite(residual).all()


def test_lw_tab_p_repair_marks_missing_and_repairs_without_prefill():
    frame_index, bbox = _make_bbox()
    raw = bbox.copy()
    raw[6:8] = np.nan
    raw[15, 0] += 35.0
    raw[15, 1] -= 20.0

    n = len(frame_index)
    pred_dx = np.zeros(n, dtype=np.float64)
    pred_dy = np.zeros(n, dtype=np.float64)
    pred_w = np.full(n, 18.0, dtype=np.float64)
    pred_h = np.full(n, 12.0, dtype=np.float64)

    filt = LWTabPFilter(LWTabPConfig(anomaly_residual_quantile=0.80, lowess_frac=0.45, robust_iterations=3))
    filt._models = {
        "dx": _FakeModel(pred_dx),
        "dy": _FakeModel(pred_dy),
        "w": _FakeModel(pred_w),
        "h": _FakeModel(pred_h),
    }

    out = filt.repair(raw, frame_index)

    assert out["flag_mask"][6]
    assert out["flag_mask"][7]
    assert out["flag_mask"][15]
    assert np.isfinite(out["bbox"]).all()
