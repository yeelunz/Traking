from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracking.classification.feature_extractors.v6 import (  # noqa: E402
    _estimate_c_wmv,
    _estimate_standard_merged_extend,
    _first_readable_frame,
)


def _estimate_for_video(video_path: Path) -> dict:
    frame = _first_readable_frame(str(video_path))
    if frame is None:
        return {
            "video_path": str(video_path),
            "rule": "no_frame",
            "px_per_cm": None,
            "mm_per_pixel": None,
            "cm_per_px": None,
            "zero_depth_y_px": None,
        }

    est_c = _estimate_c_wmv(frame)
    est_s = _estimate_standard_merged_extend(frame)

    chosen = None
    chosen_rule = None
    if est_c is not None:
        chosen = est_c
        chosen_rule = "extendclen_auto_c_wmv"
    elif est_s is not None:
        chosen = est_s
        chosen_rule = "extendclen_auto_standard"

    if chosen is None:
        return {
            "video_path": str(video_path),
            "rule": "no_scale_detected",
            "px_per_cm": None,
            "mm_per_pixel": None,
            "cm_per_px": None,
            "zero_depth_y_px": None,
            "alt_estimates": {
                "c_wmv": None,
                "standard": None,
            },
        }

    px_per_cm = float(chosen.px_per_cm)
    cm_per_px = float(chosen.cm_per_px)
    mm_per_pixel = float(cm_per_px * 10.0)

    return {
        "video_path": str(video_path),
        "rule": chosen_rule,
        "px_per_cm": px_per_cm,
        "mm_per_pixel": mm_per_pixel,
        "cm_per_px": cm_per_px,
        "zero_depth_y_px": float(chosen.zero_depth_y_px),
        "alt_estimates": {
            "c_wmv": None
            if est_c is None
            else {
                "px_per_cm": float(est_c.px_per_cm),
                "mm_per_pixel": float(est_c.cm_per_px * 10.0),
                "cm_per_px": float(est_c.cm_per_px),
                "zero_depth_y_px": float(est_c.zero_depth_y_px),
                "rule": str(est_c.rule),
            },
            "standard": None
            if est_s is None
            else {
                "px_per_cm": float(est_s.px_per_cm),
                "mm_per_pixel": float(est_s.cm_per_px * 10.0),
                "cm_per_px": float(est_s.cm_per_px),
                "zero_depth_y_px": float(est_s.zero_depth_y_px),
                "rule": str(est_s.rule),
            },
        },
    }


def main() -> None:
    dataset_root = PROJECT_ROOT / "dataset_old" / "extendclen"
    videos = sorted([*dataset_root.rglob("*.avi"), *dataset_root.rglob("*.wmv")])
    rows = [_estimate_for_video(video_path) for video_path in videos]

    out = {
        "dataset_root": str(dataset_root),
        "video_count": len(rows),
        "videos": rows,
    }

    output = PROJECT_ROOT / "results" / "extendclen_video_mm_per_pixel.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
