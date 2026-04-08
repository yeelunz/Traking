from __future__ import annotations

from tracking.classification.trajectory_filter import resolve_filter_bbox_size


def test_resolve_filter_bbox_size_enables_non_none_strategy_by_default():
    assert resolve_filter_bbox_size("hampel_only", None) is True
    assert resolve_filter_bbox_size("none", None) is False
    assert resolve_filter_bbox_size("hampel_only", False) is False
