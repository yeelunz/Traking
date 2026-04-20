"""Diagnostic: estimate depth scale per-video from the ruler ticks on each frame.

Shows that different videos have different px_per_cm values.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracking.classification.feature_extractors.v6 import (
    _estimate_depth_scale_without_rules_from_frame,
    _first_readable_frame,
)


def main() -> None:
    dataset_root = PROJECT_ROOT / "dataset" / "merged_extend_control_japan_plus_diseased_japan_20260420"
    if not dataset_root.exists():
        print(f"[ERROR] dataset root not found: {dataset_root}")
        return

    videos = sorted([*dataset_root.rglob("*.avi"), *dataset_root.rglob("*.wmv")])
    if not videos:
        print("[ERROR] no video files found")
        return

    rows = []
    for vp in videos:
        frame = _first_readable_frame(str(vp))
        if frame is None:
            rows.append({"video": vp.parent.name, "error": "no_frame"})
            continue
        est = _estimate_depth_scale_without_rules_from_frame(frame)
        if est is None:
            rows.append({"video": vp.parent.name, "error": "no_scale_detected", "frame_shape": list(frame.shape)})
            continue
        rows.append({
            "video": vp.parent.name,
            "px_per_cm": round(float(est.px_per_cm), 4),
            "cm_per_px": round(float(est.cm_per_px), 6),
            "mm_per_px": round(float(est.cm_per_px * 10), 6),
            "zero_depth_y_px": round(float(est.zero_depth_y_px), 4),
            "rule": est.rule,
            "frame_shape": list(frame.shape),
        })

    # Print summary table
    print(f"\n{'Video':<45} {'px_per_cm':>12} {'mm_per_px':>12} {'rule':<30}")
    print("-" * 100)
    for r in rows:
        name = r["video"]
        if "error" in r:
            print(f"{name:<45} {'ERROR':>12} {r['error']}")
        else:
            print(f"{name:<45} {r['px_per_cm']:>12.4f} {r['mm_per_px']:>12.6f} {r['rule']:<30}")

    # Unique scales
    scales = sorted({r["px_per_cm"] for r in rows if "px_per_cm" in r})
    print(f"\nUnique px_per_cm values: {scales}")
    print(f"Count: {len(scales)}")

    out_path = PROJECT_ROOT / "results" / "diag_per_video_scale.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nFull output: {out_path}")


if __name__ == "__main__":
    main()
