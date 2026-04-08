from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracking.utils.fallback_stats import compute_roi_fallback_stats_from_trace


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect fallback behaviour from a results experiment directory.")
    ap.add_argument("experiment_dir", help="Path to one experiment under results/")
    args = ap.parse_args()

    exp_dir = Path(args.experiment_dir)
    seg_root = exp_dir / "test" / "segmentation" / "predictions"
    if not seg_root.exists():
        raise SystemExit(f"Segmentation predictions not found: {seg_root}")

    print(f"Experiment: {exp_dir}")
    for trace_path in sorted(seg_root.glob("*/*/*/*/roi_trace.json")):
        trace = _load_json(trace_path)
        stats = compute_roi_fallback_stats_from_trace(trace) or {}
        sources = Counter(str(v.get("bbox_source", "")) for v in trace.values() if isinstance(v, dict))
        explicit_fb = sum(1 for v in trace.values() if isinstance(v, dict) and bool(v.get("is_fallback", False)))
        print(
            f"{trace_path.parent.parent.name}/{trace_path.parent.name}: "
            f"frames={int(stats.get('roi_total_frames', 0))} "
            f"fallback={int(stats.get('roi_fallback_frames', 0))} "
            f"rate={stats.get('roi_fallback_rate', 0.0):.4f} "
            f"is_fallback={explicit_fb} "
            f"sources={dict(sources)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
