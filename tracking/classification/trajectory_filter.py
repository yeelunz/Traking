"""Multi-scale bidirectional Hampel + Savitzky-Golay trajectory filter.

此模組提供完整的軌跡平滑管線，取代原有的 Kalman 濾波器。

設計原則
--------
1. **雙階段 Hampel**:
   - Stage 1 (macro): 大視窗 (序列長度 10–20%)，移除長連續異常。
   - Stage 2 (micro): 小視窗 (11–21 frames)，移除點狀離群值。
2. **邊界鏡像填充 (mirror/reflection padding)**:
   前後端延伸避免邊緣效應造成的失真。
3. **雙向 S-G 平滑**:
   forward + backward 取平均，消除 phase shift。
4. **BBox 尺寸策略** (三種，可選):
   - ``independent``: w/h 各自用較高 polyorder、較小視窗做 S-G。
   - ``fixed_global_roi``: 以 95th percentile w/h 固定 ROI 大小。
   - ``area_constraint``: 面積變化率 > 20% → NaN → 插值修補。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# scipy availability check
# ---------------------------------------------------------------------------
_SCIPY_SIGNAL_OK: bool = False
try:
    from scipy.signal import savgol_filter as _savgol_filter  # noqa: F401
    _SCIPY_SIGNAL_OK = True
except ImportError:
    pass

_SCIPY_INTERP_OK: bool = False
try:
    from scipy.interpolate import CubicSpline as _CubicSpline  # noqa: F401
    _SCIPY_INTERP_OK = True
except ImportError:
    pass

_SCIPY_PCHIP_OK: bool = False
try:
    from scipy.interpolate import PchipInterpolator as _PchipInterpolator  # noqa: F401
    _SCIPY_PCHIP_OK = True
except ImportError:
    pass


def cubic_spline_interpolate_1d(
    t_known: np.ndarray,
    v_known: np.ndarray,
    t_query: np.ndarray,
) -> np.ndarray:
    """Cubic-spline interpolation for a 1-D signal.

    Falls back to ``np.interp`` (linear) when scipy is unavailable or
    when fewer than 4 known points make cubic fitting unreliable.

    Ensures C2 continuity (velocity and acceleration) at interior knots,
    which is critical for physical plausibility of neural trajectories.
    """
    if len(t_known) < 4 or not _SCIPY_INTERP_OK:
        # Fallback: linear (only when cubic is not feasible)
        return np.interp(t_query, t_known, v_known)
    # CubicSpline requires strictly increasing knots; fall back to linear otherwise
    if len(t_known) >= 2 and np.any(np.diff(t_known) <= 0):
        return np.interp(t_query, t_known, v_known)
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(t_known, v_known, extrapolate=True)
    result = cs(t_query)
    # Clamp extrapolated boundary values to the range of known data to
    # prevent cubic polynomial divergence outside the knot span.
    v_min, v_max = float(v_known.min()), float(v_known.max())
    np.clip(result, v_min, v_max, out=result)
    return result


def pchip_interpolate_1d(
    t_known: np.ndarray,
    v_known: np.ndarray,
    t_query: np.ndarray,
) -> np.ndarray:
    """Shape-preserving PCHIP interpolation for a 1-D signal."""
    if len(t_known) < 2:
        if len(t_known) == 0:
            return np.zeros_like(t_query, dtype=np.float64)
        return np.full_like(t_query, float(v_known[0]), dtype=np.float64)
    if not _SCIPY_PCHIP_OK:
        return np.interp(t_query, t_known, v_known)
    if len(t_known) >= 2 and np.any(np.diff(t_known) <= 0):
        return np.interp(t_query, t_known, v_known)
    from scipy.interpolate import PchipInterpolator

    pchip = PchipInterpolator(t_known, v_known, extrapolate=True)
    result = np.asarray(pchip(t_query), dtype=np.float64)
    v_min, v_max = float(np.min(v_known)), float(np.max(v_known))
    np.clip(result, v_min, v_max, out=result)
    return result


def _fill_missing_with_pchip(
    values: np.ndarray,
    frame_indices: np.ndarray,
) -> np.ndarray:
    """Fill NaNs in a 1-D trajectory using PCHIP over frame indices."""
    out = np.asarray(values, dtype=np.float64).copy()
    if out.size == 0 or not np.isnan(out).any():
        return out

    t_axis = np.asarray(frame_indices, dtype=np.float64)
    valid = np.isfinite(out)
    if not valid.any():
        return np.zeros_like(out, dtype=np.float64)
    if valid.sum() == 1:
        out[~valid] = float(out[valid][0])
        return out

    out[~valid] = pchip_interpolate_1d(t_axis[valid], out[valid], t_axis[~valid])
    return out


def _mask_unobserved(values: np.ndarray, observed_mask: Optional[np.ndarray]) -> np.ndarray:
    """Mark explicitly missing samples as NaN so downstream filters ignore them."""
    out = np.asarray(values, dtype=np.float64).copy()
    if observed_mask is None:
        return out
    obs = np.asarray(observed_mask, dtype=bool)
    if obs.shape != out.shape:
        raise ValueError("observed_mask must have the same shape as values")
    out[~obs] = np.nan
    return out


def _suppress_long_outlier_runs(mask: np.ndarray, max_outlier_run: int) -> np.ndarray:
    """Keep only short outlier bursts; long runs are likely regime shifts.

    Hampel is designed for point-like anomalies. When a trajectory enters a
    genuine rapid-motion segment, several adjacent points can be incorrectly
    flagged as outliers, which then causes interpolation drift.
    """
    out = np.asarray(mask, dtype=bool).copy()
    if out.size == 0 or max_outlier_run < 1:
        return out

    i = 0
    n = len(out)
    while i < n:
        if not out[i]:
            i += 1
            continue
        j = i + 1
        while j < n and out[j]:
            j += 1
        run_len = j - i
        if run_len > max_outlier_run:
            out[i:j] = False
        i = j
    return out


def hampel_then_pchip_1d(
    values: np.ndarray,
    frame_indices: np.ndarray,
    *,
    observed_mask: Optional[np.ndarray] = None,
    macro_ratio: float = 0.15,
    macro_sigma: float = 3.0,
    micro_hw: int = 7,
    micro_sigma: float = 3.0,
    max_outlier_run: int = 2,
    skip_hampel: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply missing-aware Hampel detection and recover trajectory with PCHIP.

    Returns
    -------
    filtered : 1-D float array after PCHIP fill
    marked_missing : 1-D float array with missing/outlier samples set to NaN
    outlier_mask : bool array where Hampel marked a finite sample as outlier
    """
    working = _mask_unobserved(values, observed_mask)
    if skip_hampel:
        outlier_mask = np.zeros(len(working), dtype=bool)
    else:
        _, outlier_mask = multiscale_hampel(
            working,
            macro_ratio=macro_ratio,
            macro_sigma=macro_sigma,
            micro_hw=micro_hw,
            micro_sigma=micro_sigma,
        )
        outlier_mask = _suppress_long_outlier_runs(outlier_mask, int(max_outlier_run))
    marked = working.copy()
    if outlier_mask.any():
        marked[outlier_mask] = np.nan
    filled = _fill_missing_with_pchip(marked, frame_indices)
    return filled, marked, outlier_mask


# ═══════════════════════════════════════════════════════════════════════════
# Mirror / Reflection Padding
# ═══════════════════════════════════════════════════════════════════════════

def mirror_pad(values: np.ndarray, pad_len: int) -> Tuple[np.ndarray, int]:
    """Reflection-pad a 1-D array at both ends.

    Parameters
    ----------
    values : 1-D float array
    pad_len : number of samples to add at each end

    Returns
    -------
    padded : 1-D array of length ``len(values) + 2 * pad_len``
    offset : number of prepended samples (== *pad_len*)
    """
    n = len(values)
    if n == 0 or pad_len <= 0:
        return values.copy(), 0
    pad_len = min(pad_len, n - 1)  # can't mirror more than we have
    left = values[1 : pad_len + 1][::-1]
    right = values[-(pad_len + 1) : -1][::-1]
    padded = np.concatenate([left, values, right])
    return padded, pad_len


def _strip_pad(padded: np.ndarray, offset: int, orig_len: int) -> np.ndarray:
    """Remove mirror padding and return array of *orig_len*."""
    return padded[offset : offset + orig_len]


# ═══════════════════════════════════════════════════════════════════════════
# Core Hampel Filter (single-pass)
# ═══════════════════════════════════════════════════════════════════════════

def hampel_filter_1d(
    values: np.ndarray,
    half_window: int = 3,
    n_sigma: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Hampel outlier filter.

    Returns
    -------
    filtered : 1-D cleaned array
    outlier_mask : bool array, True where value was replaced
    """
    n = len(values)
    out = values.copy()
    mask = np.zeros(n, dtype=bool)
    if n < 3:
        return out, mask
    k = min(half_window, (n - 1) // 2)
    if k < 1:
        return out, mask
    for i in range(n):
        lo = max(0, i - k)
        hi = min(n, i + k + 1)
        if not np.isfinite(values[i]):
            continue
        window = values[lo:hi]
        window = window[np.isfinite(window)]
        if window.size == 0:
            continue
        med = np.median(window)
        mad = np.median(np.abs(window - med))
        threshold = n_sigma * 1.4826 * (mad + 1e-12)
        if abs(values[i] - med) > threshold:
            out[i] = med
            mask[i] = True
    return out, mask


# ═══════════════════════════════════════════════════════════════════════════
# Multi-scale Two-stage Hampel
# ═══════════════════════════════════════════════════════════════════════════

def multiscale_hampel(
    values: np.ndarray,
    *,
    macro_ratio: float = 0.15,
    macro_sigma: float = 3.0,
    micro_hw: int = 7,
    micro_sigma: float = 3.0,
    use_mirror_pad: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Two-stage Hampel with macro + micro windows.

    Parameters
    ----------
    values : 1-D array
    macro_ratio : macro window = max(5, int(len * ratio))
    macro_sigma : sigma threshold for stage 1
    micro_hw : half-window for stage 2
    micro_sigma : sigma threshold for stage 2
    use_mirror_pad : apply reflection padding before filtering

    Returns
    -------
    filtered : cleaned 1-D array (same length as input)
    combined_mask : OR of both stage outlier masks
    """
    n = len(values)
    if n < 3:
        return values.copy(), np.zeros(n, dtype=bool)

    # Stage 1: macro
    macro_hw = max(5, int(n * macro_ratio / 2.0))
    pad_len = max(macro_hw, micro_hw) if use_mirror_pad else 0

    if pad_len > 0 and n > 2:
        padded, offset = mirror_pad(values, pad_len)
    else:
        padded = values.copy()
        offset = 0

    stage1, mask1_p = hampel_filter_1d(padded, half_window=macro_hw, n_sigma=macro_sigma)

    # Stage 2: micro
    stage2, mask2_p = hampel_filter_1d(stage1, half_window=micro_hw, n_sigma=micro_sigma)

    # Strip padding
    if offset > 0:
        result = _strip_pad(stage2, offset, n)
        mask1 = _strip_pad(mask1_p.astype(np.float64), offset, n).astype(bool)
        mask2 = _strip_pad(mask2_p.astype(np.float64), offset, n).astype(bool)
    else:
        result = stage2[:n]
        mask1 = mask1_p[:n]
        mask2 = mask2_p[:n]

    combined_mask = mask1 | mask2
    return result, combined_mask


# ═══════════════════════════════════════════════════════════════════════════
# Bidirectional Savitzky-Golay Smoothing
# ═══════════════════════════════════════════════════════════════════════════

def bidirectional_savgol(
    values: np.ndarray,
    window_length: int = 7,
    polyorder: int = 2,
) -> np.ndarray:
    """Bidirectional (forward + backward averaged) S-G smoothing.

    Eliminates phase shift inherent in single-pass filtering.
    Window is adaptively capped to at most ~50 % of the signal length to
    avoid over-smoothing on short sequences.
    """
    n = len(values)
    if n < 4 or not _SCIPY_SIGNAL_OK:
        return values.copy()
    from scipy.signal import savgol_filter

    # Adaptive cap: never use more than half the data
    max_wl = max(5, (n // 2) | 1)  # ensure odd
    wl = min(window_length, max_wl, n)
    if wl % 2 == 0:
        wl = max(wl - 1, 3)
    po = min(polyorder, wl - 1)
    if wl < 3 or po < 1:
        return values.copy()

    fwd = savgol_filter(values, wl, po)
    bwd = savgol_filter(values[::-1], wl, po)[::-1]
    return ((fwd + bwd) / 2.0).astype(values.dtype)


def _adaptive_savgol(
    values: np.ndarray,
    frame_indices: np.ndarray,
    window_length: int = 7,
    polyorder: int = 2,
) -> np.ndarray:
    """Time-aware S-G smoothing for potentially non-uniform frame spacing.

    When frame indices are uniformly spaced (consecutive integers),
    this falls back to :func:`bidirectional_savgol` directly.

    When frame spacing is non-uniform (gaps > 1 between adjacent
    observations), the pipeline is:

    1. PCHIP resample observations to a dense 1-frame grid.
    2. Scale the S-G window proportionally to the density ratio so that
       the temporal coverage is equivalent regardless of sparsity.
    3. Apply bidirectional S-G on the dense grid.
    4. PCHIP extract smoothed values at the original frame positions.
    """
    n = len(values)
    if n < 4:
        return values.copy()

    orig_dtype = values.dtype
    diffs = np.diff(frame_indices)

    # Validate: frame_indices must be sorted and strictly increasing
    if np.any(diffs < 0):
        raise ValueError("_adaptive_savgol: frame_indices must be sorted in non-decreasing order")
    if np.any(diffs == 0):
        raise ValueError(
            "_adaptive_savgol: duplicate frame_indices detected; "
            "each frame must have exactly one observation"
        )

    # Uniform iff all gaps are identical
    is_uniform = diffs.max() == diffs.min()

    if is_uniform:
        return bidirectional_savgol(values, window_length, polyorder)

    # ── Non-uniform path ─────────────────────────────────────────────────
    fi_f = frame_indices.astype(np.float64)
    fi_min, fi_max = fi_f[0], fi_f[-1]
    if fi_max <= fi_min:
        return values.copy()

    n_dense = int(fi_max - fi_min) + 1
    # Guard against memory explosion for extremely sparse data
    _MAX_DENSE = 10_000
    if n_dense > _MAX_DENSE:
        # Too sparse; fall back to direct S-G on observed points
        return bidirectional_savgol(values, window_length, polyorder)

    uniform_fi = np.arange(fi_min, fi_max + 1, dtype=np.float64)
    n_uniform = len(uniform_fi)
    if n_uniform < 4:
        return bidirectional_savgol(values, window_length, polyorder)

    # Resample to uniform 1-frame grid
    interpolated = pchip_interpolate_1d(fi_f, values.astype(np.float64), uniform_fi)

    # Scale window proportionally: same *temporal* coverage as intended
    density_ratio = n_uniform / max(n, 1)
    scaled_wl = max(3, int(round(window_length * density_ratio)))
    scaled_wl = min(scaled_wl, n_uniform)
    if scaled_wl % 2 == 0:
        scaled_wl = max(scaled_wl - 1, 3)
    scaled_po = min(polyorder, scaled_wl - 1)

    smoothed_uniform = bidirectional_savgol(interpolated, scaled_wl, scaled_po)

    # Extract at original positions, preserving original dtype
    result = pchip_interpolate_1d(uniform_fi, smoothed_uniform, fi_f)
    return result.astype(orig_dtype)


# ═══════════════════════════════════════════════════════════════════════════
# Smooth Centroid (replaces _KalmanSmoother2D.smooth)
# ═══════════════════════════════════════════════════════════════════════════

def smooth_trajectory_2d(
    observations: np.ndarray,
    frame_indices: np.ndarray,
    *,
    observed_mask: Optional[np.ndarray] = None,
    macro_ratio: float = 0.15,
    macro_sigma: float = 3.0,
    micro_hw: int = 7,
    micro_sigma: float = 3.0,
    sg_window: int = 7,
    sg_polyorder: int = 2,
    skip_hampel: bool = False,
) -> np.ndarray:
    """Smooth 2-D centroid trajectory using multi-scale Hampel + bidirectional S-G.

    Drop-in replacement for ``_KalmanSmoother2D.smooth()``.

    Parameters
    ----------
    observations : (N, 2) centroid positions
    frame_indices : (N,) frame numbers
    skip_hampel : bool
        If ``True``, skip the multi-scale Hampel outlier removal stage and
        apply **only** the bidirectional Savitzky-Golay smoothing.  This is
        the correct mode for **ground-truth** trajectories: GT annotations
        should not have outliers, but still benefit from S-G smoothing to
        ensure C2-continuous velocity/acceleration derivatives needed by
        downstream motion-feature extractors.

    Returns
    -------
    smoothed : (N, 2)
    """
    n = observations.shape[0]
    if n <= 2:
        return observations.copy()

    # Sort observations by frame index for correct temporal processing
    order = np.argsort(frame_indices)
    fi_sorted = frame_indices[order]
    obs_sorted = observations[order]

    smoothed_sorted = np.empty_like(obs_sorted)
    obs_mask_sorted = None
    if observed_mask is not None:
        obs_mask_sorted = np.asarray(observed_mask, dtype=bool)[order]
    for dim in range(2):
        raw = obs_sorted[:, dim]
        cleaned, _, _ = hampel_then_pchip_1d(
            raw,
            fi_sorted,
            observed_mask=obs_mask_sorted,
            macro_ratio=macro_ratio,
            macro_sigma=macro_sigma,
            micro_hw=micro_hw,
            micro_sigma=micro_sigma,
            skip_hampel=skip_hampel,
        )
        # Use time-aware S-G that correctly handles non-uniform frame gaps
        smoothed_sorted[:, dim] = _adaptive_savgol(
            cleaned, fi_sorted, window_length=sg_window, polyorder=sg_polyorder,
        )

    # Restore original order
    restore = np.argsort(order)
    return smoothed_sorted[restore]


# ═══════════════════════════════════════════════════════════════════════════
# BBox Size Strategies
# ═══════════════════════════════════════════════════════════════════════════

def filter_bbox_independent(
    widths: np.ndarray,
    heights: np.ndarray,
    *,
    sg_window: int = 7,
    sg_polyorder: int = 2,
    macro_ratio: float = 0.15,
    micro_hw: int = 5,
    skip_hampel: bool = False,
    frame_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """策略一: 獨立時間序列處理 — w/h 各自做 Hampel + S-G (較高 polyorder、較小視窗)。

    Parameters
    ----------
    skip_hampel : bool
        If ``True``, bypass Hampel outlier removal and apply only S-G smoothing.
        Should be set for ground-truth trajectories (no tracking noise in annotations).
    frame_indices : optional array
        When provided, enables time-aware S-G that correctly handles non-uniform
        frame spacing.  Falls back to standard S-G when ``None``.
    """
    if skip_hampel:
        # GT mode: no Hampel on bbox dimensions; S-G only
        w_clean = widths.copy().astype(np.float64)
        h_clean = heights.copy().astype(np.float64)
    else:
        w_clean, _ = multiscale_hampel(widths, macro_ratio=macro_ratio, micro_hw=micro_hw)
        h_clean, _ = multiscale_hampel(heights, macro_ratio=macro_ratio, micro_hw=micro_hw)
    if frame_indices is not None and len(frame_indices) == len(widths):
        w_smooth = _adaptive_savgol(w_clean, frame_indices, sg_window, sg_polyorder)
        h_smooth = _adaptive_savgol(h_clean, frame_indices, sg_window, sg_polyorder)
    else:
        w_smooth = bidirectional_savgol(w_clean, window_length=sg_window, polyorder=sg_polyorder)
        h_smooth = bidirectional_savgol(h_clean, window_length=sg_window, polyorder=sg_polyorder)
    # Ensure positive sizes
    w_smooth = np.maximum(w_smooth, 1.0)
    h_smooth = np.maximum(h_smooth, 1.0)
    return w_smooth, h_smooth


def filter_bbox_fixed_global(
    widths: np.ndarray,
    heights: np.ndarray,
    *,
    percentile: float = 95.0,
    skip_hampel: bool = False,  # accepted for API consistency; no Hampel in this strategy
    frame_indices: Optional[np.ndarray] = None,  # accepted for API consistency
) -> Tuple[np.ndarray, np.ndarray]:
    """策略二: 全局固定尺寸擷取 — 以 95th percentile 固定 w/h。"""
    fixed_w = float(np.percentile(widths, percentile))
    fixed_h = float(np.percentile(heights, percentile))
    fixed_w = max(fixed_w, 1.0)
    fixed_h = max(fixed_h, 1.0)
    return np.full_like(widths, fixed_w), np.full_like(heights, fixed_h)


def filter_bbox_area_constraint(
    widths: np.ndarray,
    heights: np.ndarray,
    *,
    max_area_change_ratio: float = 0.20,
    sg_window: int = 7,
    sg_polyorder: int = 2,
    skip_hampel: bool = False,  # propagated to fallback filter_bbox_independent
    frame_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """策略三: 面積變化率限制 — >20% 變化 → NaN → 插值修補。"""
    areas = widths * heights
    n = len(areas)

    # Compute frame-to-frame area change ratio
    valid = np.ones(n, dtype=bool)
    if n > 1:
        change_ratio = np.abs(np.diff(areas)) / (areas[:-1] + 1e-9)
        for i in range(1, n):
            if change_ratio[i - 1] > max_area_change_ratio:
                valid[i] = False

    # Replace invalid with NaN and interpolate
    w_out = widths.copy().astype(np.float64)
    h_out = heights.copy().astype(np.float64)
    w_out[~valid] = np.nan
    h_out[~valid] = np.nan

    n_valid = int(valid.sum())
    if n_valid < 3:
        # Too few valid frames for meaningful interpolation — fall back to
        # independent Hampel+S-G rather than collapsing to a constant.
        import warnings
        warnings.warn(
            f"filter_bbox_area_constraint: only {n_valid}/{n} valid frames; "
            f"falling back to filter_bbox_independent.",
            RuntimeWarning,
            stacklevel=2,
        )
        return filter_bbox_independent(
            widths, heights, sg_window=sg_window, sg_polyorder=sg_polyorder,
            skip_hampel=skip_hampel, frame_indices=frame_indices,
        )

    # Fill NaN gaps with shape-preserving PCHIP interpolation.
    if frame_indices is not None and len(frame_indices) == n:
        t_axis = frame_indices.astype(np.float64)
    else:
        t_axis = np.arange(n, dtype=np.float64)
    for arr, raw in ((w_out, widths), (h_out, heights)):
        nans = np.isnan(arr)
        if nans.any() and (~nans).any():
            arr[nans] = pchip_interpolate_1d(
                t_axis[~nans], arr[~nans], t_axis[nans],
            )
        elif nans.all():
            # All values marked invalid in this dimension; fall back to raw median
            arr[:] = max(float(np.median(raw)), 1.0)

    # Final S-G smoothing (time-aware if frame_indices known)
    if frame_indices is not None and len(frame_indices) == n:
        w_out = _adaptive_savgol(w_out, frame_indices, sg_window, sg_polyorder)
        h_out = _adaptive_savgol(h_out, frame_indices, sg_window, sg_polyorder)
    else:
        w_out = bidirectional_savgol(w_out, window_length=sg_window, polyorder=sg_polyorder)
        h_out = bidirectional_savgol(h_out, window_length=sg_window, polyorder=sg_polyorder)
    w_out = np.maximum(w_out, 1.0)
    h_out = np.maximum(h_out, 1.0)
    return w_out, h_out


def filter_bbox_none(
    widths: np.ndarray,
    heights: np.ndarray,
    *,
    skip_hampel: bool = False,
    frame_indices: Optional[np.ndarray] = None,
    **_kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """策略零: 保留偵測器原始 w/h，不做任何尺寸濾波。

    Rationale
    ---------
    Detector bbox sizes are learned from GT during training and tend to be
    frame-level accurate.  Savitzky-Golay smoothing treats natural
    frame-to-frame size variation as noise and replaces it with a polynomial
    fit, which *reduces IoU* because smoothed sizes no longer match GT
    sizes on a per-frame basis.

    Using ``"none"`` preserves the original sizes for accurate detection
    metrics while still allowing centre-trajectory smoothing (Hampel + S-G)
    for downstream ROI cropping stability.
    """
    w_out = widths.copy().astype(np.float64)
    h_out = heights.copy().astype(np.float64)
    if frame_indices is None or len(frame_indices) != len(w_out):
        frame_indices = np.arange(len(w_out), dtype=np.float64)
    w_out = _fill_missing_with_pchip(w_out, np.asarray(frame_indices))
    h_out = _fill_missing_with_pchip(h_out, np.asarray(frame_indices))
    return w_out, h_out


def filter_bbox_hampel_only(
    widths: np.ndarray,
    heights: np.ndarray,
    *,
    macro_ratio: float = 0.15,
    macro_sigma: float = 3.0,
    micro_hw: int = 5,
    micro_sigma: float = 3.0,
    skip_hampel: bool = False,
    frame_indices: Optional[np.ndarray] = None,
    observed_mask: Optional[np.ndarray] = None,
    **_kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """策略: 僅做 Hampel 離群值移除，不做 S-G 平滑。

    Removes extreme size outliers (e.g. detector returned a tiny or huge
    bbox on one frame) without the over-smoothing that S-G introduces.
    A good middle-ground between ``"none"`` and ``"independent"``.
    """
    if frame_indices is None or len(frame_indices) != len(widths):
        frame_indices = np.arange(len(widths), dtype=np.float64)
    fi = np.asarray(frame_indices)
    w_clean, _, _ = hampel_then_pchip_1d(
        widths,
        fi,
        observed_mask=observed_mask,
        macro_ratio=macro_ratio,
        macro_sigma=macro_sigma,
        micro_hw=micro_hw,
        micro_sigma=micro_sigma,
        skip_hampel=skip_hampel,
    )
    h_clean, _, _ = hampel_then_pchip_1d(
        heights,
        fi,
        observed_mask=observed_mask,
        macro_ratio=macro_ratio,
        macro_sigma=macro_sigma,
        micro_hw=micro_hw,
        micro_sigma=micro_sigma,
        skip_hampel=skip_hampel,
    )
    w_clean = np.maximum(w_clean, 1.0)
    h_clean = np.maximum(h_clean, 1.0)
    return w_clean, h_clean


# ═══════════════════════════════════════════════════════════════════════════
# Full Detection Trajectory Filter
# ═══════════════════════════════════════════════════════════════════════════

# BBox strategy dispatcher
_BBOX_STRATEGIES = {
    "none": filter_bbox_none,
    "hampel_only": filter_bbox_hampel_only,
    "independent": filter_bbox_independent,
    "fixed_global_roi": filter_bbox_fixed_global,
    "area_constraint": filter_bbox_area_constraint,
}


def resolve_filter_bbox_size(
    bbox_strategy: str,
    filter_bbox_size: Optional[bool],
) -> bool:
    """Resolve whether bbox size filtering should run."""
    if filter_bbox_size is None:
        return str(bbox_strategy).strip().lower() != "none"
    return bool(filter_bbox_size)


def filter_detections(
    frame_indices: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    widths: np.ndarray,
    heights: np.ndarray,
    scores: np.ndarray,
    *,
    bbox_strategy: str = "none",
    bbox_params: Optional[Dict[str, Any]] = None,
    traj_params: Optional[Dict[str, Any]] = None,
    skip_hampel: bool = False,
    observed_mask: Optional[np.ndarray] = None,
    anchor_mask: Optional[np.ndarray] = None,
    anchor_keep_ratio: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Filter a full detection trajectory.

    Parameters
    ----------
    frame_indices : (N,) int
    cx, cy : (N,) float — centroid x, y
    widths, heights : (N,) float — bbox w, h
    scores : (N,) float — detection scores
    bbox_strategy : one of ``"none"`` / ``"hampel_only"`` / ``"independent"`` /
        ``"fixed_global_roi"`` / ``"area_constraint"``
    bbox_params : keyword args dict forwarded to bbox filter
    traj_params : keyword args dict for smooth_trajectory_2d
    skip_hampel : bool
        If ``True``, bypass Hampel outlier removal and apply only S-G
        smoothing.  Designed for ground-truth trajectories that do not
        contain tracking noise but still need physical smoothing.

    Returns
    -------
    dict with keys: frame_indices, cx, cy, widths, heights, scores (all same length)
    """
    n = len(frame_indices)
    if n < 2:
        return {
            "frame_indices": frame_indices.copy(),
            "cx": cx.copy(),
            "cy": cy.copy(),
            "widths": widths.copy(),
            "heights": heights.copy(),
            "scores": scores.copy(),
        }

    # Validate: no duplicate frame_indices (each frame must have one detection)
    if len(np.unique(frame_indices)) != n:
        import warnings
        warnings.warn(
            f"filter_detections: {n - len(np.unique(frame_indices))} duplicate "
            f"frame_indices detected; keeping first occurrence per frame.",
            RuntimeWarning, stacklevel=2,
        )
        _, uniq_idx = np.unique(frame_indices, return_index=True)
        frame_indices = frame_indices[uniq_idx]
        cx = cx[uniq_idx]
        cy = cy[uniq_idx]
        widths = widths[uniq_idx]
        heights = heights[uniq_idx]
        scores = scores[uniq_idx]
        n = len(frame_indices)

    traj_kw = dict(traj_params or {})
    bbox_kw = dict(bbox_params or {})

    # Propagate GT mode (skip Hampel, S-G only) into both smooth_trajectory_2d
    # and bbox strategy functions so that ALL smoothing is Hampel-free for GT.
    if skip_hampel:
        traj_kw["skip_hampel"] = True
        bbox_kw["skip_hampel"] = True

    # Sort by frame index
    order = np.argsort(frame_indices)
    fi_sorted = frame_indices[order]
    cx_sorted = cx[order].astype(np.float64)
    cy_sorted = cy[order].astype(np.float64)
    w_sorted = widths[order].astype(np.float64)
    h_sorted = heights[order].astype(np.float64)
    s_sorted = scores[order].astype(np.float64)
    observed_sorted = None
    if observed_mask is not None:
        observed_sorted = np.asarray(observed_mask, dtype=bool)[order]
    anchor_sorted = None
    if anchor_mask is not None:
        anchor_sorted = np.asarray(anchor_mask, dtype=bool)[order]

    # 1. Smooth centroid trajectory
    obs = np.column_stack([cx_sorted, cy_sorted])
    smoothed = smooth_trajectory_2d(obs, fi_sorted, observed_mask=observed_sorted, **traj_kw)
    cx_filt = smoothed[:, 0]
    cy_filt = smoothed[:, 1]

    # 2. Apply bbox strategy
    strategy_fn = _BBOX_STRATEGIES.get(bbox_strategy)
    if strategy_fn is None:
        raise ValueError(f"Unknown bbox_strategy: {bbox_strategy!r}. "
                         f"Choose from {list(_BBOX_STRATEGIES.keys())}")
    bbox_kw["frame_indices"] = fi_sorted
    if observed_sorted is not None:
        bbox_kw["observed_mask"] = observed_sorted
    w_filt, h_filt = strategy_fn(w_sorted, h_sorted, **bbox_kw)

    # Keep detector-anchor frames close to raw observations so valid detector
    # updates are not overruled by surrounding fallback segments.
    keep = float(np.clip(anchor_keep_ratio, 0.0, 1.0))
    if anchor_sorted is not None and keep > 0.0 and anchor_sorted.any():
        cx_filt[anchor_sorted] = keep * cx_sorted[anchor_sorted] + (1.0 - keep) * cx_filt[anchor_sorted]
        cy_filt[anchor_sorted] = keep * cy_sorted[anchor_sorted] + (1.0 - keep) * cy_filt[anchor_sorted]
        w_filt[anchor_sorted] = keep * w_sorted[anchor_sorted] + (1.0 - keep) * w_filt[anchor_sorted]
        h_filt[anchor_sorted] = keep * h_sorted[anchor_sorted] + (1.0 - keep) * h_filt[anchor_sorted]

    return {
        "frame_indices": fi_sorted,
        "cx": cx_filt,
        "cy": cy_filt,
        "widths": w_filt,
        "heights": h_filt,
        "scores": s_sorted,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Metrics: before vs. after filtering
# ═══════════════════════════════════════════════════════════════════════════

def compute_trajectory_metrics(
    cx: np.ndarray,
    cy: np.ndarray,
    widths: np.ndarray,
    heights: np.ndarray,
    frame_indices: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute trajectory quality metrics for comparison before/after filtering.

    Parameters
    ----------
    frame_indices : optional (N,) int
        When provided, time-differences are normalised by frame gap so that
        jitter and smoothness are measured in *per-frame* units even when
        observations are non-uniformly spaced.

    Metrics
    -------
    - jitter_cx / jitter_cy: std of frame-to-frame centroid displacement
    - jitter_w / jitter_h: std of frame-to-frame size change
    - smoothness_cx / smoothness_cy: mean of second derivative magnitude
    - area_stability: 1 - std(area) / mean(area) (higher = more stable)
    - path_length: total centroid displacement
    """
    n = len(cx)
    metrics: Dict[str, float] = {}

    if n < 2:
        for k in ("jitter_cx", "jitter_cy", "jitter_w", "jitter_h",
                   "smoothness_cx", "smoothness_cy", "area_stability",
                   "path_length"):
            metrics[k] = 0.0
        return metrics

    # Compute time gaps between adjacent observations
    if frame_indices is not None and len(frame_indices) == n:
        dt = np.diff(frame_indices).astype(np.float64)
        dt = np.maximum(dt, 1.0)  # avoid division by zero
    else:
        dt = np.ones(n - 1, dtype=np.float64)

    # Jitter (std of velocity = displacement / dt)
    dcx = np.diff(cx) / dt
    dcy = np.diff(cy) / dt
    dw = np.diff(widths) / dt
    dh = np.diff(heights) / dt
    metrics["jitter_cx"] = float(np.std(dcx))
    metrics["jitter_cy"] = float(np.std(dcy))
    metrics["jitter_w"] = float(np.std(dw))
    metrics["jitter_h"] = float(np.std(dh))

    # Smoothness (mean |second derivative| = |d(velocity)/dt|)
    if n >= 3:
        dt2 = (dt[:-1] + dt[1:]) / 2.0  # central dt for second derivative
        dt2 = np.maximum(dt2, 1.0)
        d2cx = np.diff(dcx) / dt2
        d2cy = np.diff(dcy) / dt2
        metrics["smoothness_cx"] = float(np.mean(np.abs(d2cx)))
        metrics["smoothness_cy"] = float(np.mean(np.abs(d2cy)))
    else:
        metrics["smoothness_cx"] = 0.0
        metrics["smoothness_cy"] = 0.0

    # Area stability
    areas = widths * heights
    mean_area = float(np.mean(areas))
    if mean_area > 1e-9:
        metrics["area_stability"] = max(0.0, 1.0 - float(np.std(areas)) / mean_area)
    else:
        metrics["area_stability"] = 0.0

    # Path length (actual displacement, not divided by dt)
    step_dist = np.sqrt(np.diff(cx) ** 2 + np.diff(cy) ** 2)
    metrics["path_length"] = float(np.sum(step_dist))

    return metrics
