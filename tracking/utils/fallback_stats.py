from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


DIRECT_BBOX_SOURCES = {"detector", "tracker"}


def compute_roi_fallback_stats_from_trace(trace: Mapping[str, Any]) -> Optional[Dict[str, float]]:
    """Compute fallback stats from a loaded ``roi_trace.json`` payload."""
    if not isinstance(trace, Mapping) or not trace:
        return None

    total = 0
    fallback = 0
    for key, row in trace.items():
        try:
            int(key)
        except Exception:
            continue
        if not isinstance(row, Mapping):
            continue
        total += 1
        src = str(row.get("bbox_source", "")).strip().lower()
        is_fallback = bool(row.get("is_fallback", False))
        if is_fallback or src not in DIRECT_BBOX_SOURCES:
            fallback += 1

    if total <= 0:
        return None

    return {
        "roi_total_frames": float(total),
        "roi_fallback_frames": float(fallback),
        "roi_fallback_rate": float(fallback) / float(total),
    }


__all__ = ["DIRECT_BBOX_SOURCES", "compute_roi_fallback_stats_from_trace"]
