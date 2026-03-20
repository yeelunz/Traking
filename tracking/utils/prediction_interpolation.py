from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from ..classification.trajectory_filter import cubic_spline_interpolate_1d
from ..core.interfaces import FramePrediction


def _score_value(score: float | None) -> float:
    if score is None:
        return float("-inf")
    try:
        return float(score)
    except Exception:
        return float("-inf")


def _deduplicate_by_frame(preds: Sequence[FramePrediction]) -> List[FramePrediction]:
    frame_map: Dict[int, FramePrediction] = {}
    for pred in preds:
        frame_idx = int(pred.frame_index)
        stored = frame_map.get(frame_idx)
        if stored is None:
            frame_map[frame_idx] = pred
            continue

        stored_fallback = bool(getattr(stored, "is_fallback", False))
        current_fallback = bool(getattr(pred, "is_fallback", False))

        if stored_fallback and not current_fallback:
            frame_map[frame_idx] = pred
            continue
        if stored_fallback == current_fallback and _score_value(pred.score) > _score_value(stored.score):
            frame_map[frame_idx] = pred

    return [frame_map[idx] for idx in sorted(frame_map.keys())]


def cubic_clip_interpolate_predictions(
    preds: Sequence[FramePrediction],
    *,
    max_gap: int = 30,
    interpolated_bbox_source: str = "interpolated_cubic",
) -> List[FramePrediction]:
    """Interpolate missing frames with GT-aligned cubic+clip bbox interpolation.

    - Uses the same cubic interpolation helper as GT processing
      (``cubic_spline_interpolate_1d``).
    - Clips interpolated values to known-data min/max range per bbox dimension
      to prevent cubic overshoot.
    - Fills only interior gaps whose length is ``<= max_gap``.
    """
    if len(preds) < 2:
        return list(preds)

    try:
        gap_limit = int(max_gap)
    except Exception:
        gap_limit = 30
    if gap_limit < 2:
        return _deduplicate_by_frame(preds)

    sorted_preds = _deduplicate_by_frame(preds)
    if len(sorted_preds) < 2:
        return sorted_preds

    known_frames = np.asarray([int(p.frame_index) for p in sorted_preds], dtype=np.float64)
    known_boxes = np.asarray([list(p.bbox) for p in sorted_preds], dtype=np.float64)
    if known_boxes.ndim != 2 or known_boxes.shape[1] != 4:
        return sorted_preds

    dim_min = np.min(known_boxes, axis=0)
    dim_max = np.max(known_boxes, axis=0)

    interpolated: List[FramePrediction] = []
    for left, right in zip(sorted_preds[:-1], sorted_preds[1:]):
        left_idx = int(left.frame_index)
        right_idx = int(right.frame_index)
        gap = right_idx - left_idx
        if gap <= 1 or gap > gap_limit:
            continue

        query_frames = np.arange(left_idx + 1, right_idx, dtype=np.float64)
        if query_frames.size == 0:
            continue

        interp_dims: List[np.ndarray] = []
        for dim in range(4):
            values = cubic_spline_interpolate_1d(
                known_frames,
                known_boxes[:, dim],
                query_frames,
            )
            values = np.asarray(values, dtype=np.float64)
            np.clip(values, float(dim_min[dim]), float(dim_max[dim]), out=values)
            interp_dims.append(values)

        for i, frame_value in enumerate(query_frames.tolist()):
            bbox = (
                float(interp_dims[0][i]),
                float(interp_dims[1][i]),
                float(interp_dims[2][i]),
                float(interp_dims[3][i]),
            )
            interpolated.append(
                FramePrediction(
                    frame_index=int(frame_value),
                    bbox=bbox,
                    score=None,
                    is_fallback=True,
                    bbox_source=interpolated_bbox_source,
                )
            )

    merged = list(sorted_preds) + interpolated
    return _deduplicate_by_frame(merged)


__all__ = ["cubic_clip_interpolate_predictions"]
