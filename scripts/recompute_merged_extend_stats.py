from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tracking.utils.annotations import load_coco_vid


MODALITY_MAP: Dict[str, str] = {
    "doppler": "Doppler",
    "Grasp": "Grasp",
    "Relax": "Relax",
    "Rest": "Rest",
    "Rest post": "Rest post",
    "D": "Doppler",
    "G-R": "Relax",
    "R-G": "Grasp",
    "R1": "Rest",
    "R2": "Rest post",
}

MODALITY_ORDER: List[str] = ["Grasp", "Relax", "Rest", "Rest post", "Doppler"]


def parse_subject_labels(path: Path) -> Dict[str, int]:
    labels: Dict[str, int] = {}
    if not path.exists():
        raise FileNotFoundError(f"Label file not found: {path}")
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        subject_id = parts[0].strip()
        try:
            value = int(parts[1])
        except Exception:
            continue
        labels[subject_id] = 1 if value == 1 else 0
    return labels


def canonical_modality(video_stem: str) -> str | None:
    direct = MODALITY_MAP.get(video_stem)
    if direct is not None:
        return direct

    normalized = str(video_stem).strip()
    lowered = normalized.lower()

    # c-subject aliases (Chinese file naming)
    if lowered.endswith("_d") or lowered.endswith("d"):
        return "Doppler"
    if "伸握伸" in normalized:
        return "Relax"
    if "握伸" in normalized or "伸握" in normalized:
        return "Grasp"

    return None


def count_annotated_frames(json_path: Path) -> int:
    ann = load_coco_vid(str(json_path))
    frames = ann.get("frames", {}) or {}
    total = 0
    for boxes in frames.values():
        if isinstance(boxes, list) and len(boxes) > 0:
            total += 1
    return int(total)


def iter_subject_dirs(root: Path, label_map: Dict[str, int]) -> Iterable[Path]:
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name in label_map:
            yield child


def compute_stats(dataset_root: Path, labels_path: Path) -> Dict:
    label_map = parse_subject_labels(labels_path)
    frame_stats = {
        modality: {"patient": 0, "control": 0}
        for modality in MODALITY_ORDER
    }
    subject_sets = {"patient": set(), "control": set()}

    labeled_subject_sets = {
        "patient": {sid for sid, v in label_map.items() if int(v) == 1},
        "control": {sid for sid, v in label_map.items() if int(v) == 0},
    }

    per_subject: Dict[str, Dict[str, int]] = {}

    for subject_dir in iter_subject_dirs(dataset_root, label_map):
        subject_id = subject_dir.name
        label = int(label_map[subject_id])
        group = "patient" if label == 1 else "control"

        subject_has_data = False
        per_subject.setdefault(subject_id, {m: 0 for m in MODALITY_ORDER})

        for json_path in sorted(subject_dir.glob("*.json")):
            modality = canonical_modality(json_path.stem)
            if modality is None:
                continue
            frame_count = count_annotated_frames(json_path)
            if frame_count <= 0:
                continue

            subject_has_data = True
            frame_stats[modality][group] += int(frame_count)
            per_subject[subject_id][modality] += int(frame_count)

        if subject_has_data:
            subject_sets[group].add(subject_id)

    total_frames = {
        "patient": sum(frame_stats[m]["patient"] for m in MODALITY_ORDER),
        "control": sum(frame_stats[m]["control"] for m in MODALITY_ORDER),
    }

    return {
        "dataset_root": str(dataset_root),
        "labels_path": str(labels_path),
        "subjects": {
            "labeled": {
                "patient": len(labeled_subject_sets["patient"]),
                "control": len(labeled_subject_sets["control"]),
                "patient_ids": sorted(labeled_subject_sets["patient"]),
                "control_ids": sorted(labeled_subject_sets["control"]),
            },
            "with_annotated_frames": {
                "patient": len(subject_sets["patient"]),
                "control": len(subject_sets["control"]),
                "patient_ids": sorted(subject_sets["patient"]),
                "control_ids": sorted(subject_sets["control"]),
            },
        },
        "frames": {
            "total": total_frames,
            "by_modality": frame_stats,
        },
        "per_subject": per_subject,
    }


def render_markdown(stats: Dict) -> str:
    subj = stats["subjects"]
    frames = stats["frames"]
    rows = []
    rows.append("## Number of Subject (Patient / Control)")
    rows.append("")
    rows.append("| Item | Count |")
    rows.append("|---|---:|")
    rows.append(f"| Total number (from ann.txt) | {subj['labeled']['patient']} / {subj['labeled']['control']} |")
    rows.append(
        f"| With annotated frames | {subj['with_annotated_frames']['patient']} / {subj['with_annotated_frames']['control']} |"
    )
    rows.append("")
    rows.append("## Number of Frames (Patient / Control)")
    rows.append("")
    rows.append("| Item | Count |")
    rows.append("|---|---:|")
    rows.append(
        f"| Total number | {frames['total']['patient']} / {frames['total']['control']} |"
    )
    for modality in MODALITY_ORDER:
        p = int(frames["by_modality"][modality]["patient"])
        c = int(frames["by_modality"][modality]["control"])
        rows.append(f"| {modality} | {p} / {c} |")
    rows.append("")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute subject/frame summary table for merged_extend dataset.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset") / "merged_extend",
        help="Dataset root directory (default: dataset/merged_extend)",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Path to ann.txt (default: <dataset-root>/ann.txt)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results") / "merged_extend_stats.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results") / "merged_extend_stats.md",
        help="Output Markdown path",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    labels_path = args.labels.resolve() if args.labels is not None else (dataset_root / "ann.txt").resolve()

    stats = compute_stats(dataset_root, labels_path)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)

    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    md = render_markdown(stats)
    args.output_md.write_text(md, encoding="utf-8")

    print(f"Saved JSON: {args.output_json}")
    print(f"Saved Markdown: {args.output_md}")
    print(md)


if __name__ == "__main__":
    main()
