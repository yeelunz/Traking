from __future__ import annotations

from tracking.utils.fallback_stats import compute_roi_fallback_stats_from_trace


def test_compute_roi_fallback_stats_prefers_explicit_is_fallback():
    trace = {
        "0": {"bbox_source": "detector", "is_fallback": False},
        "1": {"bbox_source": "detector", "is_fallback": True},
        "2": {"bbox_source": "missing", "is_fallback": False},
        "meta": {"bbox_source": "detector", "is_fallback": False},
    }

    stats = compute_roi_fallback_stats_from_trace(trace)

    assert stats is not None
    assert stats["roi_total_frames"] == 3.0
    assert stats["roi_fallback_frames"] == 2.0
    assert stats["roi_fallback_rate"] == 2.0 / 3.0
