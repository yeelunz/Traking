from __future__ import annotations

import argparse
import csv
import json
import math
import os
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import re
from urllib.parse import quote, urlencode

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles


def _sanitize(obj: Any) -> Any:
    """Recursively replace NaN / ±Inf floats with None for JSON safety."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Voting / modality analysis
# ---------------------------------------------------------------------------

# Maps raw entity_id strings (from predictions.json) → canonical modality name.
# Sick subjects use long English names; normal subjects use short codes.
_MODAL_MAP: Dict[str, str] = {
    # sick subject naming
    "doppler":   "doppler",
    "Grasp":     "grasp",
    "Relax":     "relax",
    "Rest":      "rest",
    "Rest post": "rest_post",
    # normal subject naming
    "D":   "doppler",
    "G-R": "grasp",
    "R-G": "relax",
    "R1":  "rest",
    "R2":  "rest_post",
}

_MODALITIES: List[str] = ["doppler", "grasp", "relax", "rest", "rest_post"]


def _compute_voting_analysis(predictions: List[Dict]) -> Dict:
    """Compute per-modality accuracy and voting analysis from predictions.json entries.

    Each entry: {entity_id, subject_id, label_true, label_pred, prob_positive}.
    Returns a dict with keys: modalities, per_modality, five_voting, top_combos.
    """
    # Group by subject_id → {canonical_modality: {pred, true, prob}}
    subjects: Dict[str, Dict[str, Dict]] = {}
    for item in predictions:
        entity_id = item.get("entity_id", "")
        subject_id = item.get("subject_id", "")
        canonical = _MODAL_MAP.get(str(entity_id))
        if canonical is None:
            continue
        if subject_id not in subjects:
            subjects[subject_id] = {}
        subjects[subject_id][canonical] = {
            "pred": item.get("label_pred"),
            "true": item.get("label_true"),
            "prob": item.get("prob_positive"),
        }

    if not subjects:
        return {"modalities": _MODALITIES, "per_modality": [], "five_voting": None, "top_combos": []}

    # Per-modality accuracy
    per_modality: List[Dict] = []
    for m in _MODALITIES:
        correct = 0
        total = 0
        for subj_data in subjects.values():
            if m not in subj_data:
                continue
            total += 1
            if subj_data[m]["pred"] == subj_data[m]["true"]:
                correct += 1
        per_modality.append({
            "modality": m,
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total > 0 else None,
        })

    def _vote_result(mod_list: List[str]) -> Dict:
        """Majority vote across selected modalities for each subject."""
        correct = 0
        total = 0
        details: List[Dict] = []
        for subj_id, subj_data in sorted(subjects.items()):
            avail = [m for m in mod_list if m in subj_data]
            if not avail:
                continue
            # All modalities for a given subject share the same true label
            true_label = subj_data[avail[0]]["true"]
            votes_pos = sum(subj_data[m]["pred"] for m in avail if subj_data[m]["pred"] is not None)
            vote = 1 if votes_pos > len(avail) / 2 else 0
            is_correct = (vote == true_label)
            if is_correct:
                correct += 1
            total += 1
            details.append({
                "subject_id": subj_id,
                "true_label": true_label,
                "vote": vote,
                "votes_for_positive": int(votes_pos),
                "total_votes": len(avail),
                "correct": is_correct,
            })
        return {
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total > 0 else None,
            "details": details,
        }

    five_voting = _vote_result(_MODALITIES)
    five_voting["modalities"] = _MODALITIES

    top_combos: List[Dict] = []
    for combo in combinations(_MODALITIES, 3):
        result = _vote_result(list(combo))
        result["modalities"] = list(combo)
        top_combos.append(result)
    top_combos.sort(key=lambda x: (x["accuracy"] or 0, x["correct"]), reverse=True)

    return {
        "modalities": _MODALITIES,
        "per_modality": per_modality,
        "five_voting": five_voting,
        "top_combos": top_combos,
    }



def load_json(path: Path) -> Optional[Dict]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def first_value(obj: Dict) -> Optional[Dict]:
    if not isinstance(obj, dict):
        return None
    for value in obj.values():
        if isinstance(value, dict):
            return value
    return None


def detection_summary_metrics(exp_path: Path) -> Optional[Dict]:
    summary_path = exp_path / "test" / "detection" / "metrics" / "summary.json"
    if not summary_path.exists():
        return None
    data = load_json(summary_path)
    return first_value(data) if data else None


def segmentation_summary_metrics(exp_path: Path) -> Optional[Dict]:
    summary_path = exp_path / "test" / "segmentation" / "metrics_summary.json"
    if not summary_path.exists():
        return None
    data = load_json(summary_path)
    if not data:
        return None
    first_model = first_value(data)
    return first_value(first_model) if first_model else None


def classification_summary_metrics(exp_path: Path) -> Optional[Dict]:
    """Load classification summary from ``<exp>/classification/summary.json``."""
    summary_path = exp_path / "classification" / "summary.json"
    if not summary_path.exists():
        return None
    return load_json(summary_path)


def gather_classification_metrics(exp_path: Path) -> Dict:
    """Collect classification summary, predictions and artefact metadata."""
    cls_dir = exp_path / "classification"
    summary = load_json(cls_dir / "summary.json") if (cls_dir / "summary.json").exists() else None
    predictions = load_json(cls_dir / "predictions.json") if (cls_dir / "predictions.json").exists() else None
    artefacts = load_json(cls_dir / "artefacts.json") if (cls_dir / "artefacts.json").exists() else None
    return {
        "summary": summary,
        "predictions": predictions or [],
        "artefacts": artefacts,
    }


def _format_loso_fold(subject: Optional[object], fallback: Optional[str] = None) -> str:
    """Normalize fold label for display/grouping.

    Prefer metadata dataset.split.subject. If it's numeric, format as losoXYZ.
    """
    if subject is None:
        return fallback or "loso"
    try:
        s = str(subject).strip()
    except Exception:
        return fallback or "loso"
    if not s:
        return fallback or "loso"
    # numeric subject ids like "4" or "004"
    if s.isdigit():
        return f"loso{int(s):03d}"
    # already looks like loso...
    if s.lower().startswith("loso"):
        return s.lower()
    return s


def _is_loso_run(meta: Dict) -> bool:
    split = (meta.get("dataset") or {}).get("split") or {}
    method = str(split.get("method") or "").strip().lower()
    if method == "loso":
        return True
    if bool(split.get("loso", False)):
        return True
    # heuristic: if subject exists and total_folds > 1
    if split.get("subject") is not None:
        try:
            total_folds = int(split.get("total_folds") or 0)
        except Exception:
            total_folds = 0
        if total_folds > 1:
            return True
    return False


def _aggregate_id(group_path: str, exp_name: str) -> str:
    group = (group_path or "").strip("/")
    name = (exp_name or "").strip()
    if group:
        return f"{group}/{name}"
    return name


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / float(len(values))


def _std(values: List[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    mu = _mean(values)
    if mu is None:
        return None
    var = sum((v - mu) ** 2 for v in values) / float(len(values) - 1)
    return var ** 0.5


def aggregate_preview_dicts(previews: List[Optional[Dict]]) -> Optional[Dict]:
    """Aggregate per-fold preview dicts into mean/std across folds.

    Notes:
    - We only aggregate numeric scalar fields.
    - We skip existing '*_std' inputs and compute std from the corresponding value field.
    - If both 'fps' and 'fps_mean' exist, we prefer '*_mean' variants.
    """
    preview_dicts = [p for p in previews if isinstance(p, dict)]
    if not preview_dicts:
        return None

    all_keys: set[str] = set()
    for p in preview_dicts:
        all_keys.update(p.keys())

    values_by_key: Dict[str, List[float]] = {}
    for p in preview_dicts:
        for key, value in p.items():
            if key.endswith("_std"):
                continue
            if f"{key}_mean" in all_keys:
                continue
            if isinstance(value, (int, float)):
                values_by_key.setdefault(key, []).append(float(value))

    aggregated: Dict[str, float] = {}
    for key, values in values_by_key.items():
        mu = _mean(values)
        if mu is None:
            continue
        aggregated[key] = mu
        if key.endswith("_mean"):
            std_key = f"{key[:-5]}_std"
        else:
            std_key = f"{key}_std"
        sigma = _std(values)
        if sigma is not None:
            aggregated[std_key] = sigma

    return aggregated or None


class ExperimentIndex:
    def __init__(self, root: Path):
        self.root = root.resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"Results root not found: {self.root}")
        self.entries: Dict[str, Dict] = {}
        self.loso_groups: Dict[str, List[Dict]] = {}
        self.refresh()

    def _experiment_dirs(self) -> List[Path]:
        """Find every directory under results root that contains an experiment metadata.json."""
        experiment_dirs: List[Path] = []
        seen: set[Path] = set()
        for meta_path in sorted(self.root.rglob("metadata.json")):
            exp_path = meta_path.parent
            try:
                exp_path.relative_to(self.root)
            except ValueError:
                continue
            if exp_path in seen:
                continue
            seen.add(exp_path)
            experiment_dirs.append(exp_path)
        return experiment_dirs

    def refresh(self) -> None:
        entries: Dict[str, Dict] = {}
        loso_groups: Dict[str, List[Dict]] = {}
        for exp_path in self._experiment_dirs():
            meta_path = exp_path / "metadata.json"
            meta = load_json(meta_path)
            if not meta or not isinstance(meta, dict):
                continue
            experiment_meta = meta.get("experiment") or {}
            pipeline = experiment_meta.get("pipeline") or []
            if not pipeline:
                pipeline = meta.get("config", {}).get("pipeline", [])
            preprocs = [step.get("name") for step in pipeline if step.get("type") == "preproc"]
            models = [step.get("name") for step in pipeline if step.get("type") == "model"]
            rel = self._relative_label(exp_path)
            group_path = self._group_path(rel)
            exp_name = experiment_meta.get("name") or exp_path.name
            is_loso = _is_loso_run(meta)
            split = (meta.get("dataset") or {}).get("split") or {}
            fold_label = _format_loso_fold(split.get("subject"), fallback=None) if is_loso else None
            aggregate_rel = _aggregate_id(group_path, str(exp_name)) if is_loso else None
            det_summary = exp_path / "test" / "detection" / "metrics" / "summary.json"
            seg_summary = exp_path / "test" / "segmentation" / "metrics_summary.json"
            cls_summary_path = exp_path / "classification" / "summary.json"
            det_preview = detection_summary_metrics(exp_path)
            seg_preview = segmentation_summary_metrics(exp_path)
            cls_preview = classification_summary_metrics(exp_path)
            entry = {
                "id": rel,
                "path": exp_path,
                "relative_path": rel,
                "group_path": group_path,
                "is_loso": is_loso,
                "fold": fold_label,
                "aggregate_id": aggregate_rel,
                "name": exp_name,
                "created_at": meta.get("created_at"),
                "preprocs": preprocs,
                "models": models,
                "has_detection": det_summary.exists(),
                "has_segmentation": seg_summary.exists(),
                "has_classification": cls_summary_path.exists(),
                "has_detection_visuals": (exp_path / "test" / "detection" / "visualizations").exists(),
                "has_segmentation_visuals": bool(list((exp_path / "test" / "segmentation").rglob("visualizations_roi"))) if (exp_path / "test" / "segmentation").exists() else False,
                "preview": {
                    "detection": det_preview,
                    "segmentation": seg_preview,
                    "classification": cls_preview,
                },
            }
            entries[rel] = entry
            if is_loso and aggregate_rel and fold_label:
                loso_groups.setdefault(aggregate_rel, []).append({
                    "fold": fold_label,
                    "exp_id": rel,
                })
        self.entries = entries
        self.loso_groups = loso_groups

    def _relative_label(self, path: Path) -> str:
        try:
            rel = path.relative_to(self.root)
            label = rel.as_posix()
            return label if label else "root"
        except ValueError:
            return path.name

    def _group_path(self, rel: str) -> str:
        if rel in {"", "root"}:
            return ""
        parts = rel.split("/")
        if len(parts) <= 1:
            return ""
        return "/".join(parts[:-1])

    def list_entries(self) -> List[Dict]:
        public_entries: List[Dict] = []

        # 1) Non-LOSO (or already aggregated) experiments
        for entry in self.entries.values():
            if entry.get("is_loso"):
                continue
            public_entries.append(self._public_entry(entry))

        # 2) LOSO aggregated experiments (one entry per base experiment across folds)
        for aggregate_id, fold_entries in self.loso_groups.items():
            fold_public: List[Dict] = []
            for fold_item in fold_entries:
                raw_id = fold_item.get("exp_id")
                if not raw_id:
                    continue
                raw_entry = self.entries.get(raw_id)
                if not raw_entry:
                    continue
                fold_public.append({
                    "fold": fold_item.get("fold"),
                    "id": raw_entry.get("id"),
                    "relative_path": raw_entry.get("relative_path"),
                    "created_at": raw_entry.get("created_at"),
                    "has_detection": raw_entry.get("has_detection"),
                    "has_segmentation": raw_entry.get("has_segmentation"),
                    "has_classification": raw_entry.get("has_classification"),
                    "preview": raw_entry.get("preview", {}),
                })

            if not fold_public:
                continue

            # Use the newest fold for name/config display
            fold_public.sort(key=lambda item: item.get("created_at") or "", reverse=True)
            newest_raw = self.entries.get(fold_public[0]["id"])
            if not newest_raw:
                continue

            det_previews = [f.get("preview", {}).get("detection") for f in fold_public]
            seg_previews = [f.get("preview", {}).get("segmentation") for f in fold_public]
            cls_previews = [f.get("preview", {}).get("classification") for f in fold_public]
            aggregate_preview = {
                "detection": aggregate_preview_dicts(det_previews),
                "segmentation": aggregate_preview_dicts(seg_previews),
                "classification": aggregate_preview_dicts(cls_previews),
            }
            created_at = fold_public[0].get("created_at")
            group_path = ""
            try:
                if "/" in aggregate_id:
                    group_path = aggregate_id.rsplit("/", 1)[0]
            except Exception:
                group_path = ""
            public_entries.append({
                "id": aggregate_id,
                "name": newest_raw.get("name"),
                "relative_path": aggregate_id,
                "group_path": group_path,
                "created_at": created_at,
                "preprocs": newest_raw.get("preprocs", []),
                "models": newest_raw.get("models", []),
                "has_detection": any(bool(f.get("has_detection")) for f in fold_public),
                "has_segmentation": any(bool(f.get("has_segmentation")) for f in fold_public),
                "has_classification": any(bool(f.get("has_classification")) for f in fold_public),
                "has_detection_visuals": any(self.entries.get(f.get("id"), {}).get("has_detection_visuals") for f in fold_public),
                "has_segmentation_visuals": any(self.entries.get(f.get("id"), {}).get("has_segmentation_visuals") for f in fold_public),
                "preview": aggregate_preview,
                "is_loso": True,
                "fold_count": len(fold_public),
            })

        return sorted(public_entries, key=lambda item: item.get("created_at") or "", reverse=True)

    def _public_entry(self, entry: Dict) -> Dict:
        return {
            "id": entry["id"],
            "name": entry["name"],
            "relative_path": entry["relative_path"],
            "group_path": entry.get("group_path", ""),
            "created_at": entry["created_at"],
            "preprocs": entry["preprocs"],
            "models": entry["models"],
            "has_detection": entry["has_detection"],
            "has_segmentation": entry["has_segmentation"],
            "has_classification": entry.get("has_classification", False),
            "has_detection_visuals": entry["has_detection_visuals"],
            "has_segmentation_visuals": entry["has_segmentation_visuals"],
            "preview": entry.get("preview", {}),
            "is_loso": bool(entry.get("is_loso", False)),
        }

    def get_path(self, exp_id: str) -> Path:
        entry = self.entries.get(exp_id)
        if not entry:
            raise KeyError(exp_id)
        return Path(entry["path"])

    def get_loso_folds(self, aggregate_id: str) -> List[Dict]:
        return list(self.loso_groups.get(aggregate_id) or [])


def gather_detection_metrics(exp_path: Path) -> Dict:
    summary_path = exp_path / "test" / "detection" / "metrics" / "summary.json"
    per_video_root = summary_path.parent
    summary = load_json(summary_path)
    summary_metrics = first_value(summary) if summary else None

    per_video: List[Dict] = []
    if per_video_root.exists():
        for item in sorted(per_video_root.iterdir()):
            if not item.is_dir():
                continue
            summary_file = item / "summary.json"
            if not summary_file.exists():
                continue
            data = load_json(summary_file)
            metrics = first_value(data) if data else None
            if metrics:
                per_video.append({
                    "video": item.name,
                    "metrics": metrics,
                })
    return {
        "summary": summary_metrics,
        "per_video": per_video,
    }


def gather_segmentation_metrics(exp_path: Path) -> Dict:
    summary_path = exp_path / "test" / "segmentation" / "metrics_summary.json"
    summary = load_json(summary_path)
    summary_metrics = None
    if summary:
        first_model = first_value(summary)
        summary_metrics = first_value(first_model) if first_model else None

    per_video: List[Dict] = []
    preds_root = exp_path / "test" / "segmentation" / "predictions"
    metrics_file = None
    if preds_root.exists():
        for model_dir in preds_root.iterdir():
            if not model_dir.is_dir():
                continue
            for detector_dir in model_dir.iterdir():
                candidate = detector_dir / "metrics_per_video.json"
                if candidate.exists():
                    metrics_file = candidate
                    break
            if metrics_file:
                break
    if metrics_file:
        data = load_json(metrics_file) or {}
        for video_path, metrics in data.items():
            label = os.path.basename(video_path)
            per_video.append({
                "video": label,
                "metrics": metrics,
            })
    return {
        "summary": summary_metrics,
        "per_video": per_video,
    }


def normalize_media_items(exp_path: Path, files: List[Path], exp_id: str) -> List[Dict]:
    items = []
    for file_path in files:
        rel = file_path.relative_to(exp_path)
        rel_posix = rel.as_posix()
        params = urlencode({"exp_id": exp_id, "resource": rel_posix})
        url = f"/media?{params}"
        items.append({
            "label": rel.as_posix(),
            "url": url,
        })
    return items


def list_images(
    exp_path: Path,
    pattern_root: Path,
    patterns: Sequence[str] | str,
    limit: Optional[int],
    exp_id: str,
) -> List[Dict]:
    """Return media items under pattern_root matching patterns.

    limit=None means "no limit" so the UI can show every test image when desired.
    """
    if not pattern_root.exists():
        return []
    if isinstance(patterns, str):
        patterns = [patterns]
    collected: set[Path] = set()
    for pattern in patterns:
        for file_path in pattern_root.rglob(pattern):
            if file_path.is_file():
                collected.add(file_path)
    files = sorted(collected)
    if limit is not None:
        files = files[:limit]
    return normalize_media_items(exp_path, files, exp_id)


def create_app(results_root: Path) -> FastAPI:
    index = ExperimentIndex(results_root)
    app = FastAPI(title="Schedule Results Viewer", version="1.0.0")
    app.index = index  # type: ignore[attr-defined]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"]
,
        allow_headers=["*"],
    )

    static_dir = Path(__file__).parent / "static"
    assets_dir = static_dir
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/api/experiments")
    def list_experiments():
        return Response(
            content=json.dumps(_sanitize({"experiments": index.list_entries(), "root": str(index.root)})),
            media_type="application/json",
        )

    @app.post("/api/experiments/refresh")
    def refresh_index():
        index.refresh()
        return {"count": len(index.entries)}

    def _experiment_payload(exp_id: str):
        # Aggregated LOSO entry (virtual experiment id)
        folds = index.get_loso_folds(exp_id)
        if folds:
            fold_payloads: List[Dict] = []
            det_fold_previews: List[Optional[Dict]] = []
            seg_fold_previews: List[Optional[Dict]] = []
            cls_fold_previews: List[Optional[Dict]] = []
            newest_created_at = None
            newest_entry: Optional[Dict] = None

            for fold in folds:
                raw_id = fold.get("exp_id")
                if not raw_id:
                    continue
                raw_entry = index.entries.get(raw_id)
                if not raw_entry:
                    continue
                created_at = raw_entry.get("created_at")
                if created_at and (newest_created_at is None or str(created_at) > str(newest_created_at)):
                    newest_created_at = created_at
                    newest_entry = raw_entry

                det_fold_previews.append((raw_entry.get("preview") or {}).get("detection"))
                seg_fold_previews.append((raw_entry.get("preview") or {}).get("segmentation"))
                cls_fold_previews.append((raw_entry.get("preview") or {}).get("classification"))
                fold_payloads.append({
                    "fold": fold.get("fold"),
                    "exp_id": raw_id,
                    "created_at": created_at,
                    "preview": raw_entry.get("preview") or {},
                    "has_detection": raw_entry.get("has_detection"),
                    "has_segmentation": raw_entry.get("has_segmentation"),
                    "has_classification": raw_entry.get("has_classification"),
                })

            fold_payloads.sort(key=lambda item: item.get("fold") or "")
            aggregate_detection = aggregate_preview_dicts(det_fold_previews)
            aggregate_segmentation = aggregate_preview_dicts(seg_fold_previews)
            aggregate_classification = aggregate_preview_dicts(cls_fold_previews)

            experiment_name = (newest_entry or {}).get("name") if newest_entry else exp_id
            created_at = newest_created_at

            return {
                "id": exp_id,
                "mode": "aggregate",
                "experiment": {
                    "name": experiment_name,
                    "output_dir": None,
                    "pipeline": None,
                    "created_at": created_at,
                },
                "dataset": None,
                "detection": {"summary": aggregate_detection, "per_video": []},
                "segmentation": {"summary": aggregate_segmentation, "per_video": []},
                "classification": {"summary": aggregate_classification, "predictions": [], "artefacts": None},
                "folds": fold_payloads,
            }

        try:
            exp_path = index.get_path(exp_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Experiment not found")
        meta = load_json(exp_path / "metadata.json") or {}
        dataset_info = meta.get("dataset")
        experiment_info = meta.get("experiment") or {}
        detection = gather_detection_metrics(exp_path) if (exp_path / "test" / "detection").exists() else None
        segmentation = gather_segmentation_metrics(exp_path) if (exp_path / "test" / "segmentation").exists() else None
        classification = gather_classification_metrics(exp_path) if (exp_path / "classification").exists() else None
        return {
            "id": exp_id,
            "mode": "single",
            "experiment": {
                "name": experiment_info.get("name"),
                "output_dir": experiment_info.get("output_dir"),
                "pipeline": experiment_info.get("pipeline"),
                "created_at": meta.get("created_at"),
            },
            "dataset": dataset_info,
            "detection": detection,
            "segmentation": segmentation,
            "classification": classification,
        }

    @app.get("/api/experiments/{exp_id:path}/metrics")
    def experiment_metrics_path(exp_id: str):
        return Response(
            content=json.dumps(_sanitize(_experiment_payload(exp_id))),
            media_type="application/json",
        )

    @app.get("/api/experiments/metrics")
    def experiment_metrics_query(exp_id: str = Query(..., description="Relative experiment id")):
        return Response(
            content=json.dumps(_sanitize(_experiment_payload(exp_id))),
            media_type="application/json",
        )

    @app.get("/api/experiments/{exp_id:path}/visuals")
    def experiment_visuals_path(
        exp_id: str,
        category: str = Query(...),
        limit: Optional[int] = Query(None, ge=1, le=10000, description="Max items to return; leave empty for all"),
    ):
        return _experiment_visuals(exp_id, category, limit)

    @app.get("/api/experiments/visuals")
    def experiment_visuals_query(
        exp_id: str = Query(..., description="Relative experiment id"),
        category: str = Query(...),
        limit: Optional[int] = Query(None, ge=1, le=10000, description="Max items to return; leave empty for all"),
    ):
        return _experiment_visuals(exp_id, category, limit)

    def _experiment_visuals(exp_id: str, category: str, limit: Optional[int]):
        try:
            exp_path = index.get_path(exp_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Experiment not found")
        category = category.lower()
        items: List[Dict] = []
        per_frame_cache: Dict[Path, Dict] = {}

        if category == "detection_visualizations":
            base = exp_path / "test" / "detection" / "visualizations"
            items = list_images(exp_path, base, ["*.png", "*.jpg", "*.jpeg", "*.webp"], limit, exp_id)
            # Attach per-video center error (pixels) if available
            det_metrics = gather_detection_metrics(exp_path)
            ce_map: Dict[str, float] = {}
            for entry in det_metrics.get("per_video", []):
                video = entry.get("video")
                metrics = entry.get("metrics") or {}
                ce_val = metrics.get("ce_mean") if isinstance(metrics, dict) else None
                if video is not None and ce_val is not None:
                    video_name = os.path.basename(str(video))
                    ce_map[str(video_name)] = ce_val
                    ce_map[Path(video_name).stem] = ce_val
            for item in items:
                parts = Path(item.get("label", "")).parts
                if "visualizations" in parts:
                    idx = parts.index("visualizations")
                    if idx + 1 < len(parts):
                        video_name = parts[idx + 1]
                        # Try per-frame metrics first
                        metrics_dir = (exp_path / "test" / "detection" / "metrics" / video_name)
                        per_frame_metrics = None
                        if metrics_dir.exists():
                            per_frame_metrics = per_frame_cache.get(metrics_dir)
                            if per_frame_metrics is None:
                                per_frame_metrics = {}
                                candidates = []
                                candidates.extend(sorted(metrics_dir.glob("*_per_frame.json")))
                                candidates.extend(sorted(metrics_dir.glob("metrics_per_frame.json")))
                                candidates.extend(sorted(metrics_dir.glob("per_frame.json")))
                                for path in candidates:
                                    data = load_json(path) or {}
                                    if data:
                                        per_frame_metrics = data
                                        break
                                # Fallback to CSV if JSON is missing (legacy runs)
                                if not per_frame_metrics:
                                    csv_candidates = sorted(metrics_dir.glob("*_per_frame.csv"))
                                    for csv_path in csv_candidates:
                                        try:
                                            with csv_path.open("r", encoding="utf-8") as fh:
                                                reader = csv.reader(fh)
                                                header = next(reader, None)
                                                frame_idx_col = 0
                                                ce_col = 2 if header and len(header) > 2 else 1
                                                metrics_map = {}
                                                for row in reader:
                                                    if not row or row[0] in {"frame_index", None, ""}:
                                                        continue
                                                    try:
                                                        fi = int(row[frame_idx_col])
                                                        ce_val = float(row[ce_col]) if len(row) > ce_col and row[ce_col] not in {None, ""} else None
                                                    except Exception:
                                                        continue
                                                    metrics_map[str(fi)] = {"ce": ce_val}
                                                if metrics_map:
                                                    per_frame_metrics = metrics_map
                                                    break
                                        except Exception:
                                            continue
                                per_frame_cache[metrics_dir] = per_frame_metrics
                        stem = Path(parts[-1]).stem
                        frame_idx = None
                        if stem.startswith("frame_"):
                            try:
                                frame_idx = int(stem.split("_", 1)[1])
                            except Exception:
                                frame_idx = None
                        if frame_idx is None:
                            match = re.search(r"(\d+)$", stem)
                            if match:
                                try:
                                    frame_idx = int(match.group(1))
                                except Exception:
                                    frame_idx = None
                        ce_val = None
                        if per_frame_metrics and frame_idx is not None:
                            frame_entry = per_frame_metrics.get(str(frame_idx))
                            if isinstance(frame_entry, dict):
                                ce_val = frame_entry.get("ce")
                        if ce_val is None:
                            ce_val = ce_map.get(video_name)
                        if ce_val is not None:
                            item["ce_px"] = ce_val

        elif category == "detection_metrics":
            base = exp_path / "test" / "detection" / "metrics"
            items = list_images(exp_path, base, "*.png", limit, exp_id)

        elif category in {"segmentation_overlays", "segmentation_errors"}:
            base = exp_path / "test" / "segmentation"
            pattern = "*overlay.png" if category == "segmentation_overlays" else "*error.png"
            items = list_images(exp_path, base, pattern, limit, exp_id)
            # Attach centroid error (pixels) per video and per frame if available
            seg_metrics = gather_segmentation_metrics(exp_path)
            ce_map: Dict[str, float] = {}
            for entry in seg_metrics.get("per_video", []):
                video = entry.get("video")
                metrics = entry.get("metrics") or {}
                ce_val = metrics.get("centroid_mean") if isinstance(metrics, dict) else None
                if video is not None and ce_val is not None:
                    # map both basename and stem for robustness
                    video_name = os.path.basename(str(video))
                    ce_map[video_name] = ce_val
                    ce_map[Path(video_name).stem] = ce_val
            per_frame_cache: Dict[Path, Dict[str, Dict]] = {}
            for item in items:
                label = item.get("label", "")
                path_obj = Path(label)
                stem = path_obj.stem
                video_dir = None
                metrics_for_video: Optional[Dict[str, Dict]] = None
                parts = path_obj.parts
                if "visualizations_roi" in parts:
                    idx = parts.index("visualizations_roi")
                    if idx >= 1:
                        video_dir = parts[idx - 1]
                        video_dir_path = exp_path / Path(*parts[: idx])  # path up to video dir
                        metrics_path = video_dir_path / "metrics_per_frame.json"
                        if metrics_path.exists():
                            metrics_for_video = per_frame_cache.get(metrics_path)
                            if metrics_for_video is None:
                                metrics_for_video = load_json(metrics_path) or {}
                                per_frame_cache[metrics_path] = metrics_for_video
                # strip common suffixes
                for suffix in ["_overlay", "_error"]:
                    if stem.endswith(suffix):
                        stem = stem[: -len(suffix)]
                        break
                # Try per-frame centroid error if available
                frame_idx = None
                if stem.startswith("frame_"):
                    try:
                        frame_idx = int(stem.split("_", 1)[1])
                    except Exception:
                        frame_idx = None
                if frame_idx is None:
                    match = re.search(r"(\d+)$", stem)
                    if match:
                        try:
                            frame_idx = int(match.group(1))
                        except Exception:
                            frame_idx = None
                ce_val = None
                if metrics_for_video and frame_idx is not None:
                    frame_entry = metrics_for_video.get(str(frame_idx))
                    if isinstance(frame_entry, dict):
                        ce_val = frame_entry.get("centroid")
                if ce_val is None:
                    ce_val = ce_map.get(video_dir) or ce_map.get(stem) or ce_map.get(path_obj.name)
                if ce_val is not None:
                    item["ce_px"] = ce_val

        else:
            raise HTTPException(status_code=400, detail="Unknown category")

        return {"items": items}

    def _experiment_voting(exp_id: str):
        try:
            exp_path = index.get_path(exp_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Experiment not found")
        pred_file = exp_path / "classification" / "predictions.json"
        if not pred_file.exists():
            raise HTTPException(status_code=404, detail="predictions.json not found")
        preds = load_json(pred_file) or []
        return Response(
            content=json.dumps(_sanitize(_compute_voting_analysis(preds))),
            media_type="application/json",
        )

    @app.get("/api/experiments/{exp_id:path}/voting")
    def experiment_voting_path(exp_id: str):
        return _experiment_voting(exp_id)

    @app.get("/api/experiments/voting")
    def experiment_voting_query(exp_id: str = Query(..., description="Relative experiment id")):
        return _experiment_voting(exp_id)

    @app.get("/media")
    def media(exp_id: str = Query(...), resource: str = Query(...)):
        try:
            exp_path = index.get_path(exp_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Experiment not found")
        target = (exp_path / resource).resolve()
        if not str(target).startswith(str(exp_path)) or not target.exists():
            raise HTTPException(status_code=404, detail="Resource not found")
        return FileResponse(target)

    @app.get("/")
    def index_page():
        index_html = static_dir / "index.html"
        if not index_html.exists():
            raise HTTPException(status_code=500, detail="index.html missing")
        return FileResponse(index_html)

    return app


def main():
    parser = argparse.ArgumentParser(description="Interactive schedule results viewer")
    parser.add_argument("--results", default="results", help="Path to schedule root or single experiment folder")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (for development)")
    args = parser.parse_args()
    results_root = Path(args.results)
    app = create_app(results_root)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
