from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .trajectory_filter import _adaptive_savgol, pchip_interpolate_1d

try:
    from tabpfn import TabPFNRegressor
except Exception:  # pragma: no cover
    TabPFNRegressor = None


@dataclass(frozen=True)
class LWTabPConfig:
    seed: int = 3407
    lowess_frac: float = 0.5
    robust_iterations: int = 3
    anomaly_residual_quantile: float = 0.80
    mild_sg_window: int = 5
    mild_sg_polyorder: int = 2
    tabpfn_device: str = "auto"
    tabpfn_max_train_rows: int = 4096


def _bbox_to_center_xywh(bbox: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(bbox[:, 0], dtype=np.float64)
    y = np.asarray(bbox[:, 1], dtype=np.float64)
    w = np.asarray(bbox[:, 2], dtype=np.float64)
    h = np.asarray(bbox[:, 3], dtype=np.float64)
    cx = x + w / 2.0
    cy = y + h / 2.0
    return cx, cy, w, h


def _center_xywh_to_bbox(cx: np.ndarray, cy: np.ndarray, w: np.ndarray, h: np.ndarray) -> np.ndarray:
    x = np.asarray(cx, dtype=np.float64) - np.asarray(w, dtype=np.float64) / 2.0
    y = np.asarray(cy, dtype=np.float64) - np.asarray(h, dtype=np.float64) / 2.0
    w = np.maximum(np.asarray(w, dtype=np.float64), 1.0)
    h = np.maximum(np.asarray(h, dtype=np.float64), 1.0)
    return np.column_stack([x, y, w, h]).astype(np.float64)


def _make_proxy_bbox(raw_bbox: np.ndarray, frame_index: np.ndarray) -> np.ndarray:
    observed = np.isfinite(raw_bbox).all(axis=1)
    cx, cy, w, h = _bbox_to_center_xywh(raw_bbox)
    if observed.sum() >= 2:
        t_obs = frame_index[observed].astype(np.float64)
        t_all = frame_index.astype(np.float64)
        cx_proxy = pchip_interpolate_1d(t_obs, cx[observed], t_all)
        cy_proxy = pchip_interpolate_1d(t_obs, cy[observed], t_all)
        w_proxy = pchip_interpolate_1d(t_obs, w[observed], t_all)
        h_proxy = pchip_interpolate_1d(t_obs, h[observed], t_all)
        return _center_xywh_to_bbox(cx_proxy, cy_proxy, np.maximum(w_proxy, 1.0), np.maximum(h_proxy, 1.0))
    med = np.nanmedian(raw_bbox, axis=0)
    fill = np.where(np.isfinite(med), med, np.array([0.0, 0.0, 1.0, 1.0]))
    return np.tile(fill[None, :], (len(raw_bbox), 1))


def _fill_non_finite(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    finite = np.isfinite(out)
    if finite.all():
        return out
    if not finite.any():
        return np.zeros_like(out, dtype=np.float64)
    first = int(np.where(finite)[0][0])
    out[:first] = out[first]
    for idx in range(first + 1, len(out)):
        if not np.isfinite(out[idx]):
            out[idx] = out[idx - 1]
    last = int(np.where(np.isfinite(out))[0][-1])
    out[last + 1 :] = out[last]
    out[~np.isfinite(out)] = 0.0
    return out


def _make_tab_context_bbox(raw_bbox: np.ndarray) -> np.ndarray:
    bbox = np.asarray(raw_bbox, dtype=np.float64)
    observed = np.isfinite(bbox).all(axis=1)
    if observed.any():
        cx, cy, w, h = _bbox_to_center_xywh(bbox)
        cx_ctx = _fill_non_finite(cx)
        cy_ctx = _fill_non_finite(cy)
        w_ctx = np.maximum(_fill_non_finite(w), 1.0)
        h_ctx = np.maximum(_fill_non_finite(h), 1.0)
        return _center_xywh_to_bbox(cx_ctx, cy_ctx, w_ctx, h_ctx)
    med = np.nanmedian(bbox, axis=0)
    fill = np.where(np.isfinite(med), med, np.array([0.0, 0.0, 1.0, 1.0]))
    return np.tile(fill[None, :], (len(bbox), 1))


def _build_regression_features(bbox: np.ndarray, frame_index: np.ndarray) -> np.ndarray:
    cx, cy, w, h = _bbox_to_center_xywh(bbox)
    t = np.asarray(frame_index, dtype=np.float64)
    dt_prev = np.diff(t, prepend=t[0])
    dt_prev[0] = 1.0
    dx_prev = np.diff(cx, prepend=cx[0])
    dy_prev = np.diff(cy, prepend=cy[0])
    vx = dx_prev / np.maximum(dt_prev, 1.0)
    vy = dy_prev / np.maximum(dt_prev, 1.0)
    ax = np.diff(vx, prepend=vx[0]) / np.maximum(dt_prev, 1.0)
    ay = np.diff(vy, prepend=vy[0]) / np.maximum(dt_prev, 1.0)
    heading = np.arctan2(vy, vx + 1e-12)
    feats = np.column_stack(
        [
            dx_prev,
            dy_prev,
            vx,
            vy,
            ax,
            ay,
            np.cos(heading),
            np.sin(heading),
            dt_prev,
            w,
            h,
        ]
    ).astype(np.float64)
    feats[~np.isfinite(feats)] = np.nan
    for col in range(feats.shape[1]):
        feats[:, col] = _fill_non_finite(feats[:, col])
    return feats


def _build_training_table(trajectories: Sequence[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    feature_rows: list[np.ndarray] = []
    targets: dict[str, list[np.ndarray]] = {"dx": [], "dy": [], "w": [], "h": []}
    for frame_index, bbox in trajectories:
        bbox_arr = np.asarray(bbox, dtype=np.float64)
        fi_arr = np.asarray(frame_index, dtype=np.int64)
        if bbox_arr.ndim != 2 or bbox_arr.shape[1] != 4 or len(bbox_arr) < 2:
            continue
        features = _build_regression_features(bbox_arr, fi_arr)
        cx, cy, w, h = _bbox_to_center_xywh(bbox_arr)
        dx = np.diff(cx, prepend=cx[0])
        dy = np.diff(cy, prepend=cy[0])
        feature_rows.append(features)
        targets["dx"].append(dx)
        targets["dy"].append(dy)
        targets["w"].append(w)
        targets["h"].append(h)
    if not feature_rows:
        return np.zeros((0, 11), dtype=np.float64), {k: np.zeros((0,), dtype=np.float64) for k in targets.keys()}
    X = np.concatenate(feature_rows, axis=0)
    y = {k: np.concatenate(v, axis=0) if v else np.zeros((0,), dtype=np.float64) for k, v in targets.items()}
    return X, y


def _lowess_fit_1d(x: np.ndarray, y: np.ndarray, frac: float, robust_iterations: int) -> np.ndarray:
    n = len(x)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    r = max(4, min(n, int(np.ceil(max(0.05, min(0.95, frac)) * n))))
    robust = np.ones(n, dtype=np.float64)
    pred = np.asarray(y, dtype=np.float64).copy()
    for _ in range(max(1, int(robust_iterations))):
        for i in range(n):
            dist = np.abs(x - x[i])
            h = float(np.partition(dist, min(r - 1, n - 1))[min(r - 1, n - 1)])
            h = h if h > 1e-12 else max(float(np.max(dist)), 1.0)
            u = np.clip(dist / h, 0.0, 1.0)
            weights = (1.0 - u**3) ** 3
            weights *= robust
            if np.count_nonzero(weights > 1e-12) < 2:
                pred[i] = float(np.average(y))
                continue
            x0 = x - x[i]
            design = np.column_stack([np.ones(n, dtype=np.float64), x0])
            sw = np.sqrt(weights)
            coef, *_ = np.linalg.lstsq(design * sw[:, None], y * sw, rcond=None)
            pred[i] = float(coef[0])
        resid = y - pred
        scale = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-6
        u = resid / (6.0 * scale)
        robust = np.where(np.abs(u) < 1.0, (1.0 - u**2) ** 2, 0.0)
    return np.asarray(pred, dtype=np.float64)


def _fit_detector_curve_lowess(
    t: np.ndarray,
    y: np.ndarray,
    frac: float,
    robust_iterations: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(y)
    x = np.asarray(t[finite], dtype=np.float64)
    yy = np.asarray(y[finite], dtype=np.float64)
    if finite.sum() < 4:
        pred = np.full(len(t), np.nan, dtype=np.float64)
        if finite.sum() >= 1:
            pred[np.where(finite)[0]] = yy
            pred = _fill_non_finite(pred)
        else:
            pred.fill(float(np.nanmean(y) if finite.any() else 0.0))
        return np.asarray(pred, dtype=np.float64), np.zeros(len(t), dtype=np.float64)
    pred_fwd = _lowess_fit_1d(x, yy, frac=frac, robust_iterations=robust_iterations)
    x_rev = (x.max() - x[::-1]).astype(np.float64)
    pred_rev = _lowess_fit_1d(x_rev, yy[::-1], frac=frac, robust_iterations=robust_iterations)[::-1]
    pred_obs = 0.5 * (pred_fwd + pred_rev)
    residual_obs = np.abs(yy - pred_obs)
    pred_all = np.full(len(t), np.nan, dtype=np.float64)
    pred_all[np.where(finite)[0]] = pred_obs
    pred_all = _fill_non_finite(pred_all)
    residual_all = np.zeros(len(t), dtype=np.float64)
    residual_all[np.where(finite)[0]] = residual_obs
    return np.asarray(pred_all, dtype=np.float64), residual_all


def _reconstruct_with_poisson(
    predicted_dx: np.ndarray,
    trusted_abs: np.ndarray,
    trusted_mask: np.ndarray,
) -> np.ndarray:
    n = len(predicted_dx)
    A_rows: list[np.ndarray] = []
    b_rows: list[float] = []
    trusted_mask = np.asarray(trusted_mask, dtype=bool)
    trusted_abs = np.asarray(trusted_abs, dtype=np.float64)
    anchor_candidates = np.where(trusted_mask & np.isfinite(trusted_abs))[0]
    if anchor_candidates.size > 0:
        anchor_idx = int(anchor_candidates[0])
        anchor_value = float(trusted_abs[anchor_idx])
    else:
        finite_idx = np.where(np.isfinite(trusted_abs))[0]
        anchor_idx = int(finite_idx[0]) if finite_idx.size > 0 else 0
        anchor_value = float(trusted_abs[anchor_idx]) if finite_idx.size > 0 else 0.0
    row0 = np.zeros(n, dtype=np.float64)
    row0[anchor_idx] = 1.0
    A_rows.append(row0)
    b_rows.append(anchor_value)
    for i in range(1, n):
        row = np.zeros(n, dtype=np.float64)
        row[i] = 1.0
        row[i - 1] = -1.0
        A_rows.append(row)
        b_rows.append(float(predicted_dx[i]))
    for i in np.where(trusted_mask & np.isfinite(trusted_abs))[0]:
        row = np.zeros(n, dtype=np.float64)
        row[i] = 2.0
        A_rows.append(row)
        b_rows.append(float(2.0 * trusted_abs[i]))
    A = np.vstack(A_rows)
    b = np.asarray(b_rows, dtype=np.float64)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.asarray(x, dtype=np.float64)


def build_dense_gt_from_frame_map(
    frame_map: Mapping[int, Sequence[Sequence[float]]],
    smooth_window: int = 5,
    smooth_polyorder: int = 2,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not frame_map:
        return None
    frame_keys = sorted(int(k) for k in frame_map.keys())
    if not frame_keys:
        return None
    dense_frames = np.arange(int(min(frame_keys)), int(max(frame_keys)) + 1, dtype=np.int64)
    n = len(dense_frames)
    bbox_observed = np.full((n, 4), np.nan, dtype=np.float64)
    observed_mask = np.zeros(n, dtype=bool)
    frame_to_dense = {int(f): i for i, f in enumerate(dense_frames)}
    for frame_idx, boxes in frame_map.items():
        idx = frame_to_dense.get(int(frame_idx))
        if idx is None:
            continue
        if not boxes:
            continue
        box = boxes[0]
        if box is None or len(box) < 4:
            continue
        bbox_observed[idx] = np.asarray(box[:4], dtype=np.float64)
        observed_mask[idx] = True
    if observed_mask.sum() < 4:
        return None

    cx_obs, cy_obs, w_obs, h_obs = _bbox_to_center_xywh(bbox_observed)
    t_obs = dense_frames[observed_mask].astype(np.float64)
    t_all = dense_frames.astype(np.float64)
    cx_fill = pchip_interpolate_1d(t_obs, cx_obs[observed_mask], t_all)
    cy_fill = pchip_interpolate_1d(t_obs, cy_obs[observed_mask], t_all)
    w_fill = pchip_interpolate_1d(t_obs, w_obs[observed_mask], t_all)
    h_fill = pchip_interpolate_1d(t_obs, h_obs[observed_mask], t_all)

    cx_gt = _adaptive_savgol(cx_fill, dense_frames, window_length=smooth_window, polyorder=smooth_polyorder)
    cy_gt = _adaptive_savgol(cy_fill, dense_frames, window_length=smooth_window, polyorder=smooth_polyorder)
    w_gt = _adaptive_savgol(np.maximum(w_fill, 1.0), dense_frames, window_length=smooth_window, polyorder=smooth_polyorder)
    h_gt = _adaptive_savgol(np.maximum(h_fill, 1.0), dense_frames, window_length=smooth_window, polyorder=smooth_polyorder)
    gt_bbox_dense = _center_xywh_to_bbox(cx_gt, cy_gt, np.maximum(w_gt, 1.0), np.maximum(h_gt, 1.0))
    return dense_frames, gt_bbox_dense


class LWTabPFilter:
    def __init__(self, cfg: LWTabPConfig):
        self.cfg = cfg
        self._models: dict[str, Any] = {}

    @property
    def fitted(self) -> bool:
        return bool(self._models)

    def fit(self, trajectories: Sequence[tuple[np.ndarray, np.ndarray]]) -> None:
        if TabPFNRegressor is None:
            raise RuntimeError("tabpfn is not installed; lw_tab_p requires TabPFNRegressor")
        X, y = _build_training_table(trajectories)
        if X.shape[0] == 0:
            raise RuntimeError("Empty training table for lw_tab_p regression")

        rng = np.random.default_rng(int(self.cfg.seed))
        max_rows = int(max(64, self.cfg.tabpfn_max_train_rows))
        if len(X) > max_rows:
            keep = rng.choice(np.arange(len(X)), size=max_rows, replace=False)
            keep.sort()
            X_fit = X[keep]
            y_fit = {k: v[keep] for k, v in y.items()}
        else:
            X_fit = X
            y_fit = y

        models: dict[str, Any] = {}
        for target_name in ("dx", "dy", "w", "h"):
            model = TabPFNRegressor(device=self.cfg.tabpfn_device, random_state=int(self.cfg.seed))
            model.fit(X_fit, y_fit[target_name])
            models[target_name] = model
        self._models = models

    def make_proxy_bbox(self, raw_bbox: np.ndarray, frame_index: np.ndarray) -> np.ndarray:
        return _make_tab_context_bbox(np.asarray(raw_bbox, dtype=np.float64))

    def repair(
        self,
        raw_bbox: np.ndarray,
        frame_index: np.ndarray,
        *,
        observed_mask: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        if not self.fitted:
            raise RuntimeError("LWTabPFilter is not fitted")

        raw_bbox = np.asarray(raw_bbox, dtype=np.float64)
        frame_index = np.asarray(frame_index, dtype=np.int64)
        if raw_bbox.ndim != 2 or raw_bbox.shape[1] != 4:
            raise ValueError("raw_bbox must be shaped as (N, 4)")
        if len(frame_index) != len(raw_bbox):
            raise ValueError("frame_index length must match raw_bbox rows")

        observed = np.isfinite(raw_bbox).all(axis=1)
        if observed_mask is not None:
            observed &= np.asarray(observed_mask, dtype=bool)

        cx_raw, cy_raw, w_raw, h_raw = _bbox_to_center_xywh(raw_bbox)

        _, cx_res = _fit_detector_curve_lowess(
            frame_index,
            cx_raw,
            frac=float(self.cfg.lowess_frac),
            robust_iterations=int(self.cfg.robust_iterations),
        )
        _, cy_res = _fit_detector_curve_lowess(
            frame_index,
            cy_raw,
            frac=float(self.cfg.lowess_frac),
            robust_iterations=int(self.cfg.robust_iterations),
        )
        _, w_res = _fit_detector_curve_lowess(
            frame_index,
            w_raw,
            frac=float(self.cfg.lowess_frac),
            robust_iterations=int(self.cfg.robust_iterations),
        )
        _, h_res = _fit_detector_curve_lowess(
            frame_index,
            h_raw,
            frac=float(self.cfg.lowess_frac),
            robust_iterations=int(self.cfg.robust_iterations),
        )
        anomaly_score = np.maximum.reduce([cx_res, cy_res, w_res, h_res])
        if np.nanmax(anomaly_score) > 0:
            anomaly_score = anomaly_score / float(np.nanmax(anomaly_score))

        obs_scores = anomaly_score[observed]
        if obs_scores.size > 0 and float(np.nanmax(obs_scores)) > 0.0:
            threshold = float(np.quantile(obs_scores, float(self.cfg.anomaly_residual_quantile)))
            flag_mask = observed & (anomaly_score >= threshold)
        else:
            flag_mask = np.zeros(len(raw_bbox), dtype=bool)
        flag_mask |= ~observed

        deleted_bbox = raw_bbox.copy()
        deleted_bbox[flag_mask] = np.nan
        context_bbox = _make_tab_context_bbox(deleted_bbox)
        features = _build_regression_features(context_bbox, frame_index)

        pred_dx = np.asarray(self._models["dx"].predict(features), dtype=np.float64)
        pred_dy = np.asarray(self._models["dy"].predict(features), dtype=np.float64)
        pred_w = np.asarray(self._models["w"].predict(features), dtype=np.float64)
        pred_h = np.asarray(self._models["h"].predict(features), dtype=np.float64)

        trusted_mask = (~flag_mask) & observed
        deleted_cx, deleted_cy, deleted_w, deleted_h = _bbox_to_center_xywh(deleted_bbox)
        mixed_dx = pred_dx.copy()
        mixed_dy = pred_dy.copy()
        valid_dx = trusted_mask.copy()
        valid_dx[1:] &= trusted_mask[:-1]
        valid_dx[0] &= np.isfinite(deleted_cx[0]) and np.isfinite(deleted_cy[0])
        obs_dx = np.diff(deleted_cx, prepend=deleted_cx[0])
        obs_dy = np.diff(deleted_cy, prepend=deleted_cy[0])
        obs_dx[0] = 0.0
        obs_dy[0] = 0.0
        mixed_dx[valid_dx] = obs_dx[valid_dx]
        mixed_dy[valid_dx] = obs_dy[valid_dx]

        cx_poisson = _reconstruct_with_poisson(mixed_dx, deleted_cx, trusted_mask)
        cy_poisson = _reconstruct_with_poisson(mixed_dy, deleted_cy, trusted_mask)
        w_final = pred_w.copy()
        h_final = pred_h.copy()
        w_final[trusted_mask] = deleted_w[trusted_mask]
        h_final[trusted_mask] = deleted_h[trusted_mask]

        w_final = _adaptive_savgol(
            np.maximum(w_final, 1.0),
            frame_index,
            window_length=int(self.cfg.mild_sg_window),
            polyorder=int(self.cfg.mild_sg_polyorder),
        )
        h_final = _adaptive_savgol(
            np.maximum(h_final, 1.0),
            frame_index,
            window_length=int(self.cfg.mild_sg_window),
            polyorder=int(self.cfg.mild_sg_polyorder),
        )
        cx_poisson = _adaptive_savgol(
            cx_poisson,
            frame_index,
            window_length=int(self.cfg.mild_sg_window),
            polyorder=int(self.cfg.mild_sg_polyorder),
        )
        cy_poisson = _adaptive_savgol(
            cy_poisson,
            frame_index,
            window_length=int(self.cfg.mild_sg_window),
            polyorder=int(self.cfg.mild_sg_polyorder),
        )

        return {
            "bbox": _center_xywh_to_bbox(cx_poisson, cy_poisson, w_final, h_final),
            "center": np.column_stack([cx_poisson, cy_poisson]),
            "anomaly_score": anomaly_score,
            "flag_mask": flag_mask,
            "deleted_bbox": deleted_bbox,
        }
