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
    return (2.0 * intersection + eps) / (union + eps)


def intersection_over_union(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    pred_bin = (_normalise_mask(pred) > 0).astype(np.float32)
    target_bin = (_normalise_mask(target) > 0).astype(np.float32)
    intersection = float((pred_bin * target_bin).sum())
    union = float(((pred_bin + target_bin) > 0).sum())
    if union == 0:
        return 1.0
    return (intersection + eps) / (union + eps)


def centroid_distance(pred_stats: MaskStats, gt_stats: MaskStats) -> float:
    px = pred_stats.centroid[0] - gt_stats.centroid[0]
    py = pred_stats.centroid[1] - gt_stats.centroid[1]
    return float((px ** 2 + py ** 2) ** 0.5)


def summarise_metrics(accumulator: Dict[str, list]) -> Dict[str, float]:
    summary = {}
    for key, values in accumulator.items():
        if not values:
            summary[key] = 0.0
            continue
        arr = np.asarray(values, dtype=np.float32)
        summary[f"{key}_mean"] = float(arr.mean())
        summary[f"{key}_std"] = float(arr.std())
    return summary
