from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracking.classification.feature_extractors.v6 import resolve_depth_scale_for_video


@dataclass
class VideoResult:
    video_path: str
    width: int
    height: int
    total_frames: int
    px_per_cm: float
    cm_per_px: float
    zero_depth_y_px: float
    formula_cm_from_y: str
    depth_at_top_px: float
    depth_at_bottom_px: float
    rule: str


def infer_dataset_name(dataset_root: Path) -> str:
    token = str(dataset_root).replace("\\", "/").lower()
    if "merged_extend_control_japan" in token:
        return "merged_extend_control_japan"
    if "merged_extend" in token:
        return "merged_extend"
    return dataset_root.name


def analyze_video(video_path: Path) -> VideoResult | None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()

    scale = resolve_depth_scale_for_video(str(video_path))
    if scale is None:
        return None

    return VideoResult(
        video_path=str(video_path),
        width=width,
        height=height,
        total_frames=total_frames,
        px_per_cm=float(scale.px_per_cm),
        cm_per_px=float(scale.cm_per_px),
        zero_depth_y_px=float(scale.zero_depth_y_px),
        formula_cm_from_y=f"depth_cm = (y_px - ({scale.zero_depth_y_px:.3f})) / {scale.px_per_cm:.3f}",
        depth_at_top_px=float((0.0 - scale.zero_depth_y_px) / scale.px_per_cm),
        depth_at_bottom_px=float(((height - 1) - scale.zero_depth_y_px) / scale.px_per_cm),
        rule=str(scale.rule),
    )


def build_report(dataset_root: Path) -> dict:
    video_results: list[VideoResult] = []
    for video_path in sorted(dataset_root.rglob("*.avi")) + sorted(dataset_root.rglob("*.wmv")):
        result = analyze_video(video_path)
        if result is not None:
            video_results.append(result)

    subject_summary = {}
    grouped: dict[str, list[VideoResult]] = {}
    for result in video_results:
        grouped.setdefault(Path(result.video_path).parent.name, []).append(result)

    for subject_name, results in grouped.items():
        px_per_cm = sorted(r.px_per_cm for r in results)
        zero_y = sorted(r.zero_depth_y_px for r in results)
        median_px_per_cm = float(px_per_cm[len(px_per_cm) // 2])
        median_zero_y = float(zero_y[len(zero_y) // 2])
        subject_summary[subject_name] = {
            "video_count": len(results),
            "px_per_cm_median": median_px_per_cm,
            "zero_depth_y_px_median": median_zero_y,
            "formula_cm_from_y": f"depth_cm = (y_px - ({median_zero_y:.3f})) / {median_px_per_cm:.3f}",
            "rules": sorted(set(r.rule for r in results)),
            "videos": [asdict(r) for r in results],
        }

    return {
        "dataset_root": str(dataset_root),
        "dataset_name": infer_dataset_name(dataset_root),
        "video_result_count": len(video_results),
        "subject_summary": subject_summary,
        "video_results": [asdict(result) for result in video_results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate first-frame ultrasound depth-scale rules.")
    parser.add_argument("dataset_root", type=Path, help="Dataset root that contains the videos.")
    parser.add_argument("--output", type=Path, required=True, help="Where to write the JSON report.")
    args = parser.parse_args()

    report = build_report(args.dataset_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
