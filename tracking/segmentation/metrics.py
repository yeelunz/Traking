from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from ..core.interfaces import MaskStats


def _normalise_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim == 0:
        return arr.astype(np.float32)
    # squeeze singleton dimensions but preserve spatial extents
    arr = np.squeeze(arr)
    if arr.ndim > 2:
        # If still has more than 2 dims, take the last two as spatial dimensions
        arr = np.reshape(arr, arr.shape[-2:])
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask after normalisation, got shape {arr.shape}")
    return arr.astype(np.float32)


def dice_coefficient(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    pred_bin = (_normalise_mask(pred) > 0).astype(np.float32)
    target_bin = (_normalise_mask(target) > 0).astype(np.float32)
    intersection = float((pred_bin * target_bin).sum())
    union = float(pred_bin.sum() + target_bin.sum())
    if union == 0:
        return 1.0
    return 2.0 * intersection / (union + eps)


def intersection_over_union(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    pred_bin = (_normalise_mask(pred) > 0).astype(np.float32)
    target_bin = (_normalise_mask(target) > 0).astype(np.float32)
    intersection = float((pred_bin * target_bin).sum())
    union = float(((pred_bin + target_bin) > 0).sum())
    if union == 0:
        return 1.0
    return intersection / (union + eps)


def centroid_distance(pred_stats: MaskStats, gt_stats: MaskStats) -> "float | None":
    """Euclidean centroid distance, or None if either mask has zero area."""
    if pred_stats.area_px <= 0 or gt_stats.area_px <= 0:
        return None
    px = pred_stats.centroid[0] - gt_stats.centroid[0]
    py = pred_stats.centroid[1] - gt_stats.centroid[1]
    return float((px ** 2 + py ** 2) ** 0.5)


def summarise_metrics(accumulator: Dict[str, list]) -> Dict[str, float]:
    summary = {}
    for key, values in accumulator.items():
        if not values:
            # NaN signals "no data" so downstream won't confuse with actual 0.0.
            summary[f"{key}_mean"] = float("nan")
            summary[f"{key}_std"] = float("nan")
            summary[f"{key}_count"] = 0
            continue
        arr = np.asarray(values, dtype=np.float32)
        summary[f"{key}_mean"] = float(np.nanmean(arr))
        summary[f"{key}_std"] = float(np.nanstd(arr))
        summary[f"{key}_count"] = len(values)
    return summary
