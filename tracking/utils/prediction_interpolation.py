from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

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


def _normalize_bbox(
    raw_bbox: object,
    *,
    min_bbox_wh: float = 2.0,
) -> Optional[Tuple[float, float, float, float]]:
    try:
        x, y, w, h = map(float, raw_bbox)  # type: ignore[arg-type]
    except Exception:
        return None
    if not np.isfinite([x, y, w, h]).all():
        return None
    if w < float(min_bbox_wh) or h < float(min_bbox_wh):
        return None
    return (x, y, w, h)


def _prediction_confidence(pred: Optional[FramePrediction]) -> float:
    if pred is None:
        return float("-inf")
    raw_conf = getattr(pred, "confidence", None)
    if raw_conf is None:
        raw_conf = getattr(pred, "score", None)
    try:
        conf_val = float(raw_conf)
    except Exception:
        return float("-inf")
    if not np.isfinite(conf_val):
        return float("-inf")
    return conf_val


def _select_best_prediction_by_frame(
    preds: Sequence[FramePrediction],
    *,
    min_bbox_wh: float = 2.0,
) -> Dict[int, FramePrediction]:
    selected: Dict[int, FramePrediction] = {}
    for pred in preds:
        frame_idx = int(getattr(pred, "frame_index", -1))
        stored = selected.get(frame_idx)
        if stored is None:
            selected[frame_idx] = pred
            continue

        stored_conf = _prediction_confidence(stored)
        pred_conf = _prediction_confidence(pred)
        stored_bbox_ok = _normalize_bbox(getattr(stored, "bbox", None), min_bbox_wh=min_bbox_wh) is not None
        pred_bbox_ok = _normalize_bbox(getattr(pred, "bbox", None), min_bbox_wh=min_bbox_wh) is not None
        stored_fallback = bool(getattr(stored, "is_fallback", False))
        pred_fallback = bool(getattr(pred, "is_fallback", False))

        if pred_conf > stored_conf:
            selected[frame_idx] = pred
            continue
        if pred_conf < stored_conf:
            continue
        if pred_bbox_ok and not stored_bbox_ok:
            selected[frame_idx] = pred
            continue
        if pred_bbox_ok == stored_bbox_ok and (not pred_fallback) and stored_fallback:
            selected[frame_idx] = pred
    return selected


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


def repair_predictions_for_query_frames(
    preds: Sequence[FramePrediction],
    *,
    query_frame_indices: Optional[Sequence[int]] = None,
    confidence_threshold: float = 0.5,
    min_known_points: int = 3,
    min_bbox_wh: float = 2.0,
    interpolated_bbox_source: str = "interpolated_pchip",
    missing_bbox_source: str = "missing",
    interpolation_max_gap: int = 30,
) -> List[FramePrediction]:
    """Apply detection post-processing aligned with downstream segmentation.

    Flow:
    1. Select the best bbox candidate per frame.
    2. Treat fallback / invalid / low-confidence boxes as missing.
    3. If valid anchor count is greater than or equal to ``min_known_points``,
       fill *all* requested missing query frames with PCHIP+clip interpolation.
    4. Otherwise keep those frames as explicit missing zero boxes.
    """
    selected = _select_best_prediction_by_frame(preds, min_bbox_wh=min_bbox_wh)
    if query_frame_indices is None:
        query_frames = sorted(selected.keys())
    else:
        query_frames = _sorted_unique_frames(query_frame_indices)
    if not query_frames:
        return []

    valid_anchor_preds: List[FramePrediction] = []
    for frame_idx in sorted(selected.keys()):
        pred = selected[frame_idx]
        bbox_valid = _normalize_bbox(getattr(pred, "bbox", None), min_bbox_wh=min_bbox_wh)
        if bbox_valid is None:
            continue
        if bool(getattr(pred, "is_fallback", False)):
            continue
        if _prediction_confidence(pred) < float(confidence_threshold):
            continue
        valid_anchor_preds.append(
            FramePrediction(
                frame_index=int(frame_idx),
                bbox=bbox_valid,
                score=getattr(pred, "score", None),
                confidence=getattr(pred, "confidence", None),
                confidence_components=getattr(pred, "confidence_components", None),
                segmentation=getattr(pred, "segmentation", None),
                is_fallback=False,
                bbox_source=(str(getattr(pred, "bbox_source", "detector")).strip() or "detector"),
            )
        )

    interpolated_by_frame: Dict[int, FramePrediction] = {}
    if len(valid_anchor_preds) >= max(1, int(min_known_points)):
        interpolated = cubic_clip_interpolate_predictions(
            valid_anchor_preds,
            max_gap=int(interpolation_max_gap),
            interpolated_bbox_source=interpolated_bbox_source,
            query_frame_indices=query_frames,
            fill_all_queries=True,
        )
        interpolated_by_frame = {int(p.frame_index): p for p in interpolated}

    repaired: List[FramePrediction] = []
    for frame_idx in query_frames:
        selected_pred = selected.get(int(frame_idx))
        selected_bbox = _normalize_bbox(getattr(selected_pred, "bbox", None), min_bbox_wh=min_bbox_wh)
        selected_is_anchor = (
            selected_bbox is not None
            and not bool(getattr(selected_pred, "is_fallback", False))
            and _prediction_confidence(selected_pred) >= float(confidence_threshold)
        )

        if selected_pred is not None and selected_is_anchor:
            repaired.append(
                FramePrediction(
                    frame_index=int(frame_idx),
                    bbox=selected_bbox,
                    score=getattr(selected_pred, "score", None),
                    confidence=getattr(selected_pred, "confidence", None),
                    confidence_components=getattr(selected_pred, "confidence_components", None),
                    segmentation=getattr(selected_pred, "segmentation", None),
                    is_fallback=False,
                    bbox_source=(str(getattr(selected_pred, "bbox_source", "detector")).strip() or "detector"),
                )
            )
            continue

        interp_pred = interpolated_by_frame.get(int(frame_idx))
        interp_bbox = _normalize_bbox(getattr(interp_pred, "bbox", None), min_bbox_wh=min_bbox_wh)
        if interp_bbox is not None:
            repaired.append(
                FramePrediction(
                    frame_index=int(frame_idx),
                    bbox=interp_bbox,
                    score=getattr(selected_pred, "score", None) if selected_pred is not None else None,
                    confidence=getattr(selected_pred, "confidence", None) if selected_pred is not None else None,
                    confidence_components=getattr(selected_pred, "confidence_components", None) if selected_pred is not None else None,
                    segmentation=getattr(selected_pred, "segmentation", None) if selected_pred is not None else None,
                    is_fallback=True,
                    bbox_source=(str(getattr(interp_pred, "bbox_source", interpolated_bbox_source)).strip() or interpolated_bbox_source),
                )
            )
            continue

        repaired.append(
            FramePrediction(
                frame_index=int(frame_idx),
                bbox=(0.0, 0.0, 0.0, 0.0),
                score=getattr(selected_pred, "score", None) if selected_pred is not None else None,
                confidence=getattr(selected_pred, "confidence", None) if selected_pred is not None else None,
                confidence_components=getattr(selected_pred, "confidence_components", None) if selected_pred is not None else None,
                segmentation=getattr(selected_pred, "segmentation", None) if selected_pred is not None else None,
                is_fallback=True,
                bbox_source=missing_bbox_source,
            )
        )

    return repaired


__all__ = ["cubic_clip_interpolate_predictions", "repair_predictions_for_query_frames"]
