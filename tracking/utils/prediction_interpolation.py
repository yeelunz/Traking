from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from ..classification.trajectory_filter import pchip_interpolate_1d
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


def _sorted_unique_frames(frame_indices: Optional[Sequence[int]]) -> List[int]:
    if frame_indices is None:
        return []
    unique: set[int] = set()
    for value in frame_indices:
        try:
            unique.add(int(value))
        except Exception:
            continue
    return sorted(unique)


def cubic_clip_interpolate_predictions(
    preds: Sequence[FramePrediction],
    *,
    max_gap: int = 30,
    interpolated_bbox_source: str = "interpolated_pchip",
    query_frame_indices: Optional[Sequence[int]] = None,
    fill_all_queries: bool = False,
) -> List[FramePrediction]:
    """Interpolate missing frames with GT-aligned PCHIP+clip bbox interpolation.

    - Uses the same shape-preserving PCHIP interpolation helper as GT processing.
    - Clips interpolated values to known-data min/max range per bbox dimension
      to prevent overshoot.
    - Fills only interior gaps whose length is ``<= max_gap`` by default.
    - When ``query_frame_indices`` is provided, interpolation is generated for
      the requested frames. If ``fill_all_queries`` is True, all requested
      missing frames use PCHIP (including boundary frames via clipped
      extrapolation).
    """
    sorted_preds = _deduplicate_by_frame(preds)
    if not sorted_preds:
        return []

    try:
        gap_limit = int(max_gap)
    except Exception:
        gap_limit = 30
    if gap_limit < 2:
        gap_limit = 2

    if len(sorted_preds) < 1:
        return sorted_preds

    known_frames = np.asarray([int(p.frame_index) for p in sorted_preds], dtype=np.float64)
    known_boxes = np.asarray([list(p.bbox) for p in sorted_preds], dtype=np.float64)
    if known_boxes.ndim != 2 or known_boxes.shape[1] != 4:
        return sorted_preds
    known_set = {int(v) for v in known_frames.tolist()}

    dim_min = np.min(known_boxes, axis=0)
    dim_max = np.max(known_boxes, axis=0)

    if query_frame_indices is None:
        query_candidates: List[int] = []
        for left, right in zip(sorted_preds[:-1], sorted_preds[1:]):
            left_idx = int(left.frame_index)
            right_idx = int(right.frame_index)
            gap = right_idx - left_idx
            if gap <= 1 or gap > gap_limit:
                continue
            query_candidates.extend(range(left_idx + 1, right_idx))
        query_frames = _sorted_unique_frames(query_candidates)
    else:
        query_frames = _sorted_unique_frames(query_frame_indices)

    if not query_frames:
        return sorted_preds

    frames_to_interp: List[int] = []
    if fill_all_queries:
        frames_to_interp = [fi for fi in query_frames if fi not in known_set]
    else:
        for frame_idx in query_frames:
            if frame_idx in known_set:
                continue
            insert_pos = int(np.searchsorted(known_frames, float(frame_idx), side="left"))
            if insert_pos <= 0 or insert_pos >= len(known_frames):
                continue
            left_idx = int(known_frames[insert_pos - 1])
            right_idx = int(known_frames[insert_pos])
            if frame_idx <= left_idx or frame_idx >= right_idx:
                continue
            if right_idx - left_idx > gap_limit:
                continue
            frames_to_interp.append(frame_idx)

    if not frames_to_interp:
        return sorted_preds

    query_array = np.asarray(frames_to_interp, dtype=np.float64)

    interpolated: List[FramePrediction] = []
    interp_dims: List[np.ndarray] = []
    for dim in range(4):
        values = pchip_interpolate_1d(
            known_frames,
            known_boxes[:, dim],
            query_array,
        )
        values = np.asarray(values, dtype=np.float64)
        np.clip(values, float(dim_min[dim]), float(dim_max[dim]), out=values)
        interp_dims.append(values)

    for i, frame_value in enumerate(query_array.tolist()):
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
