from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset_old" / "extendclen"
LABEL_FILE = DATASET_ROOT / "ann.txt"
SCALE_FILE = PROJECT_ROOT / "results" / "extendclen_video_mm_per_pixel.json"
OUTPUT_JSON = PROJECT_ROOT / "results" / "extendclen_gt_geometry_stats_scaled.json"
OUTPUT_MD = PROJECT_ROOT / "results" / "extendclen_gt_geometry_stats_scaled.md"


def _norm_path(path: Path | str) -> str:
    try:
        return str(Path(path).resolve()).replace("\\", "/").lower()
    except Exception:
        return str(path).replace("\\", "/").lower()


def _parse_subject_labels(path: Path) -> Dict[str, int]:
    if not path.exists():
        raise FileNotFoundError(f"Label file not found: {path}")
    mapping: Dict[str, int] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "#" in line:
            line = line.split("#", 1)[0].strip()
        parts = [p for p in line.split() if p]
        if len(parts) < 2:
            continue
        subject = parts[0].strip()
        try:
            label = int(parts[1])
        except Exception as exc:
            raise ValueError(f"Invalid label line: {raw}") from exc
        if label not in (0, 1):
            raise ValueError(f"Label must be 0/1, got {label} in line: {raw}")
        mapping[subject] = label
    if not mapping:
        raise RuntimeError(f"No valid labels in {path}")
    return mapping


def _load_mm_per_px(path: Path) -> Dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"Scale file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, float] = {}
    for row in payload.get("videos", []):
        vp = row.get("video_path")
        mm = row.get("mm_per_pixel")
        if not vp:
            continue
        if mm is None:
            continue
        mm_val = float(mm)
        if not math.isfinite(mm_val) or mm_val <= 0.0:
            continue
        out[_norm_path(vp)] = mm_val
    if not out:
        raise RuntimeError("No usable mm_per_pixel entries in scale file")
    return out


def _safe_std(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=0))


def _safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _circularity(area: float, perimeter: float) -> float:
    if perimeter <= 0.0:
        return 0.0
    return float(4.0 * math.pi * area / (perimeter ** 2))


def _iter_annotation_jsons(dataset_root: Path) -> List[Path]:
    json_files: List[Path] = []
    for p in dataset_root.glob("*/*.json"):
        if p.name.lower() == "labels.json":
            continue
        json_files.append(p)
    return sorted(json_files)


def _compute_stats(values: Dict[str, List[float]]) -> Dict[str, float]:
    return {
        "frame_count": float(len(values["csa"])),
        "csa_mean": _safe_mean(values["csa"]),
        "csa_std": _safe_std(values["csa"]),
        "eq_diam_mean": _safe_mean(values["eq_diam"]),
        "eq_diam_std": _safe_std(values["eq_diam"]),
        "circularity_mean": _safe_mean(values["circularity"]),
        "circularity_std": _safe_std(values["circularity"]),
        "aspect_ratio_mean": _safe_mean(values["aspect_ratio"]),
        "aspect_ratio_std": _safe_std(values["aspect_ratio"]),
    }


def _empty_bucket() -> Dict[str, List[float]]:
    return {
        "csa": [],
        "eq_diam": [],
        "circularity": [],
        "aspect_ratio": [],
    }


def main() -> None:
    labels = _parse_subject_labels(LABEL_FILE)
    mm_map = _load_mm_per_px(SCALE_FILE)
    json_files = _iter_annotation_jsons(DATASET_ROOT)
    if not json_files:
        raise RuntimeError(f"No annotation json files found under {DATASET_ROOT}")

    subject_buckets: Dict[str, Dict[str, List[float]]] = {s: _empty_bucket() for s in labels.keys()}
    group_buckets: Dict[str, Dict[str, List[float]]] = {
        "diseased": _empty_bucket(),
        "non_diseased": _empty_bucket(),
    }

    missing_scale: List[str] = []
    missing_label: List[str] = []

    for ann_path in json_files:
        subject = ann_path.parent.name
        if subject not in labels:
            missing_label.append(str(ann_path))
            continue

        payload = json.loads(ann_path.read_text(encoding="utf-8"))
        videos = payload.get("videos", [])
        if not videos:
            continue
        video_name = str(videos[0].get("name", "")).strip()
        if not video_name:
            continue

        video_path = ann_path.parent / video_name
        mm_per_px = mm_map.get(_norm_path(video_path))
        if mm_per_px is None:
            missing_scale.append(str(video_path))
            continue

        cm_per_px = float(mm_per_px) / 10.0
        area_scale = cm_per_px * cm_per_px

        anns = payload.get("annotations", [])
        for ann in anns:
            area_px = float(ann.get("area", 0.0) or 0.0)
            meta = ann.get("metadata") or {}
            eq_diam_px = float(meta.get("equivalent_diameter_px", 0.0) or 0.0)
            perimeter_px = float(meta.get("perimeter_px", 0.0) or 0.0)
            bbox = ann.get("bbox") or [0.0, 0.0, 0.0, 0.0]
            w = float(bbox[2]) if len(bbox) > 2 else 0.0
            h = float(bbox[3]) if len(bbox) > 3 else 0.0

            if area_px <= 0.0 or eq_diam_px <= 0.0 or perimeter_px <= 0.0 or h <= 0.0:
                continue

            csa = area_px * area_scale
            eq_diam = eq_diam_px * cm_per_px
            circ = _circularity(area_px, perimeter_px)
            aspect = w / (h + 1e-9)

            subj_bucket = subject_buckets[subject]
            subj_bucket["csa"].append(float(csa))
            subj_bucket["eq_diam"].append(float(eq_diam))
            subj_bucket["circularity"].append(float(circ))
            subj_bucket["aspect_ratio"].append(float(aspect))

            group_name = "diseased" if int(labels[subject]) == 1 else "non_diseased"
            grp_bucket = group_buckets[group_name]
            grp_bucket["csa"].append(float(csa))
            grp_bucket["eq_diam"].append(float(eq_diam))
            grp_bucket["circularity"].append(float(circ))
            grp_bucket["aspect_ratio"].append(float(aspect))

    if missing_label:
        raise RuntimeError(
            "Some annotation files have no subject label in ann.txt:\n" + "\n".join(sorted(missing_label))
        )
    if missing_scale:
        raise RuntimeError(
            "Some videos have no mm_per_pixel in scale file:\n" + "\n".join(sorted(set(missing_scale)))
        )

    subject_stats = {
        subject: _compute_stats(bucket)
        for subject, bucket in sorted(subject_buckets.items(), key=lambda kv: kv[0])
    }
    group_stats = {
        group: _compute_stats(bucket)
        for group, bucket in group_buckets.items()
    }

    output = {
        "dataset_root": str(DATASET_ROOT),
        "label_file": str(LABEL_FILE),
        "scale_file": str(SCALE_FILE),
        "units": {
            "csa": "cm^2",
            "eq_diam": "cm",
            "circularity": "unitless",
            "aspect_ratio": "unitless",
        },
        "group_stats": group_stats,
        "subject_stats": subject_stats,
    }

    OUTPUT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    def _fmt_pair(mean_val: float, std_val: float) -> str:
        return f"{mean_val:.6f} ± {std_val:.6f}"

    lines: List[str] = []
    lines.append("# Extendclen GT Geometry Statistics (scale-calibrated)")
    lines.append("")
    lines.append("- GT-only: Yes (computed from annotation JSON entries only)")
    lines.append("- Scale calibration: Yes (`results/extendclen_video_mm_per_pixel.json`)")
    lines.append("- Unit: CSA=cm^2, EqDiam=cm")
    lines.append("")
    lines.append("## Group Summary")
    lines.append("")
    lines.append("| Group | Frames | CSA (mean±std) | EqDiam (mean±std) | Circularity (mean±std) | Aspect Ratio (mean±std) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for key in ("diseased", "non_diseased"):
        row = group_stats[key]
        lines.append(
            "| {group} | {n:.0f} | {csa} | {eq} | {cir} | {asp} |".format(
                group=key,
                n=row["frame_count"],
                csa=_fmt_pair(row["csa_mean"], row["csa_std"]),
                eq=_fmt_pair(row["eq_diam_mean"], row["eq_diam_std"]),
                cir=_fmt_pair(row["circularity_mean"], row["circularity_std"]),
                asp=_fmt_pair(row["aspect_ratio_mean"], row["aspect_ratio_std"]),
            )
        )

    lines.append("")
    lines.append("## Subject Summary")
    lines.append("")
    lines.append("| Subject | Label | Frames | CSA (mean±std) | EqDiam (mean±std) | Circularity (mean±std) | Aspect Ratio (mean±std) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for subject, row in subject_stats.items():
        label = int(labels.get(subject, 0))
        lines.append(
            "| {subject} | {label} | {n:.0f} | {csa} | {eq} | {cir} | {asp} |".format(
                subject=subject,
                label=label,
                n=row["frame_count"],
                csa=_fmt_pair(row["csa_mean"], row["csa_std"]),
                eq=_fmt_pair(row["eq_diam_mean"], row["eq_diam_std"]),
                cir=_fmt_pair(row["circularity_mean"], row["circularity_std"]),
                asp=_fmt_pair(row["aspect_ratio_mean"], row["aspect_ratio_std"]),
            )
        )

    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] Wrote {OUTPUT_JSON}")
    print(f"[OK] Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
