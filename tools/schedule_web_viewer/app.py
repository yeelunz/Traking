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

from fastapi import Body, FastAPI, HTTPException, Query
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
    "G-R": "relax",
    "R-G": "grasp",
    "R1":  "rest",
    "R2":  "rest_post",
}

_MODALITIES: List[str] = ["doppler", "grasp", "relax", "rest", "rest_post"]


def _compute_voting_analysis(predictions: List[Dict]) -> Dict:
    """Compute per-modality accuracy and voting analysis from predictions.json entries.

    Each entry: {entity_id, subject_id, label_true, label_pred, prob_positive}.
    Returns a dict with keys: modalities, per_modality, five_voting,
    soft_five_voting, top_combos, soft_top_combos.

    Modalities are detected dynamically from the data: known entity_id strings are
    mapped via _MODAL_MAP; unknown strings fall back to their own value.
    This supports datasets with arbitrary video/modality names.
    """
    # Group by subject_id → {canonical_modality: {pred, true, prob}}
    subjects: Dict[str, Dict[str, Dict]] = {}
    for item in predictions:
        entity_id = item.get("entity_id", "")
        subject_id = item.get("subject_id", "")
        # Fall back to identity mapping so new datasets with arbitrary names work too
        canonical = _MODAL_MAP.get(str(entity_id), str(entity_id))
        if not canonical or not subject_id:
            continue
        if subject_id not in subjects:
            subjects[subject_id] = {}
        subjects[subject_id][canonical] = {
            "pred": item.get("label_pred"),
            "true": item.get("label_true"),
            "prob": item.get("prob_positive"),
        }

    # Build effective modality list from data (preserving _MODALITIES order for known ones)
    seen: set = set(m for subj in subjects.values() for m in subj)
    effective_modalities: List[str] = [
        m for m in _MODALITIES if m in seen
    ] + sorted(seen - set(_MODALITIES))

    if not subjects:
        return {
            "modalities": _MODALITIES,
            "per_modality": [],
            "five_voting": None,
            "soft_five_voting": None,
            "top_combos": [],
            "soft_top_combos": [],
        }

    # Per-modality accuracy
    per_modality: List[Dict] = []
    for m in effective_modalities:
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

    def _soft_vote_result(mod_list: List[str]) -> Dict:
        """Soft vote using the mean positive probability across selected modalities."""
        correct = 0
        total = 0
        details: List[Dict] = []
        for subj_id, subj_data in sorted(subjects.items()):
            avail = [m for m in mod_list if m in subj_data]
            if not avail:
                continue
            probs = [subj_data[m]["prob"] for m in avail if subj_data[m].get("prob") is not None]
            if not probs:
                continue
            true_label = subj_data[avail[0]]["true"]
            mean_prob = float(sum(float(p) for p in probs) / len(probs))
            vote = 1 if mean_prob >= 0.5 else 0
            is_correct = (vote == true_label)
            if is_correct:
                correct += 1
            total += 1
            details.append({
                "subject_id": subj_id,
                "true_label": true_label,
                "vote": vote,
                "mean_prob_positive": mean_prob,
                "total_votes": len(probs),
                "correct": is_correct,
            })
        return {
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total > 0 else None,
            "details": details,
        }

    five_voting = _vote_result(effective_modalities)
    five_voting["modalities"] = effective_modalities
    soft_five_voting = _soft_vote_result(effective_modalities)
    soft_five_voting["modalities"] = effective_modalities

    # Generate C(N, k) combos where k = min(3, N); skip if fewer than 2 modalities
    combo_size = min(3, len(effective_modalities))
    top_combos: List[Dict] = []
    soft_top_combos: List[Dict] = []
    if combo_size >= 2 and len(effective_modalities) > combo_size:
        for combo in combinations(effective_modalities, combo_size):
            result = _vote_result(list(combo))
            result["modalities"] = list(combo)
            top_combos.append(result)
            soft_result = _soft_vote_result(list(combo))
            soft_result["modalities"] = list(combo)
            soft_top_combos.append(soft_result)
    top_combos.sort(key=lambda x: (x["accuracy"] or 0, x["correct"]), reverse=True)
    soft_top_combos.sort(key=lambda x: (x["accuracy"] or 0, x["correct"]), reverse=True)
    top_combos = top_combos[:10]
    soft_top_combos = soft_top_combos[:10]

    return {
        "modalities": effective_modalities,
        "per_modality": per_modality,
        "five_voting": five_voting,
        "soft_five_voting": soft_five_voting,
        "top_combos": top_combos,
        "soft_top_combos": soft_top_combos,
    }



def _build_loso_combined(
    fold_entries: List[Dict],
    entries: Dict,
) -> Tuple[Optional[Dict], List[Dict]]:
    """Combine classification data from all LOSO folds.

    Returns ``(combined_summary, all_predictions)`` where:
    - ``combined_summary`` has *integer* TP/FP/FN/TN summed across every
      fold and re-computed overall accuracy / F1 / balanced-accuracy.
    - ``all_predictions`` is the flat list of all per-fold prediction dicts.
    """
    all_predictions: List[Dict] = []
    tp = fp = fn = tn = 0
    fold_thresholds: List[float] = []
    fold_youden_js: List[float] = []
    fold_threshold_method: Optional[str] = None
    total_loo_predictions: int = 0
    for fold in fold_entries:
        raw_id = fold.get("exp_id")
        if not raw_id:
            continue
        raw_entry = entries.get(raw_id)
        if not raw_entry:
            continue
        exp_path = Path(raw_entry["path"])
        preds = load_json(exp_path / "classification" / "predictions.json") or []
        all_predictions.extend(preds)
        s = load_json(exp_path / "classification" / "summary.json") or {}
        tp += int(round(float(s.get("tp") or 0)))
        fp += int(round(float(s.get("fp") or 0)))
        fn += int(round(float(s.get("fn") or 0)))
        tn += int(round(float(s.get("tn") or 0)))
        total_loo_predictions += int(s.get("threshold_n_loo_predictions") or 0)
        # Gather per-fold Youden threshold info from artefacts
        art = load_json(exp_path / "classification" / "artefacts.json") or {}
        t_used = art.get("threshold_used")
        j_val = art.get("youden_j")
        if t_used is not None:
            fold_thresholds.append(float(t_used))
        if j_val is not None:
            fold_youden_js.append(float(j_val))
        if fold_threshold_method is None and art.get("threshold_method"):
            fold_threshold_method = art["threshold_method"]

    total = tp + fp + fn + tn
    if total == 0:
        return None, all_predictions

    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced = (tpr + tnr) / 2

    # Compute AUC-ROC from combined predictions across all LOSO folds.
    # Each fold predicts a held-out subject, so the union of all fold predictions
    # covers the entire dataset without data leakage — making this the correct way
    # to compute AUC-ROC for a LOSO evaluation.
    roc_auc: Optional[float] = None
    brier_score: Optional[float] = None
    reliability: Optional[float] = None
    resolution: Optional[float] = None
    uncertainty: Optional[float] = None
    try:
        y_true_all = [p.get("label_true") for p in all_predictions if p.get("label_true") is not None and p.get("prob_positive") is not None]
        y_prob_all = [p.get("prob_positive") for p in all_predictions if p.get("label_true") is not None and p.get("prob_positive") is not None]
        if len(y_true_all) == len(y_prob_all) and len(y_true_all) > 0:
            y_true_bin = [1.0 if int(y) == 1 else 0.0 for y in y_true_all]
            y_prob_clip = [max(0.0, min(1.0, float(p))) for p in y_prob_all]
            brier_score = float(sum((p - o) ** 2 for p, o in zip(y_prob_clip, y_true_bin)) / float(len(y_true_bin)))

            n = len(y_true_bin)
            o_bar = sum(y_true_bin) / float(n)
            n_bins = 10
            groups: List[List[int]] = [[] for _ in range(n_bins)]
            for idx, p in enumerate(y_prob_clip):
                bin_idx = min(int(p * n_bins), n_bins - 1)
                groups[bin_idx].append(idx)
            rel = 0.0
            res = 0.0
            for indices in groups:
                if not indices:
                    continue
                nk = float(len(indices))
                fk = sum(y_prob_clip[i] for i in indices) / nk
                ok = sum(y_true_bin[i] for i in indices) / nk
                wk = nk / float(n)
                rel += wk * (fk - ok) ** 2
                res += wk * (ok - o_bar) ** 2
            reliability = float(rel)
            resolution = float(res)
            uncertainty = float(o_bar * (1.0 - o_bar))
        if len(set(y_true_all)) >= 2:
            from sklearn.metrics import roc_auc_score as _roc_auc_score  # type: ignore
            roc_auc = float(_roc_auc_score(y_true_all, y_prob_all))
    except Exception:
        roc_auc = None
        brier_score = None
        reliability = None
        resolution = None
        uncertainty = None

    mean_threshold: Optional[float] = (
        float(sum(fold_thresholds) / len(fold_thresholds)) if fold_thresholds else None
    )
    mean_youden_j: Optional[float] = (
        float(sum(fold_youden_js) / len(fold_youden_js)) if fold_youden_js else None
    )

    combined_summary: Dict = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "precision_positive": precision,
        "recall_positive": recall,
        "specificity": specificity,
        "f1_positive": f1,
        "roc_auc": roc_auc,
        "brier_score": brier_score,
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "support_positive": tp + fn,
        "threshold_used": mean_threshold,
        "threshold_method": fold_threshold_method,
        "youden_j": mean_youden_j,
        "threshold_n_loo_predictions": total_loo_predictions or None,
    }
    return combined_summary, all_predictions


def _to_float_list(values: Any) -> Optional[List[float]]:
    if not isinstance(values, list):
        return None
    out: List[float] = []
    for value in values:
        try:
            out.append(float(value))
        except Exception:
            out.append(0.0)
    return out


def _rows_to_feature_map(rows: Any) -> Dict[str, float]:
    if not isinstance(rows, list):
        return {}
    fmap: Dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        name_raw = row.get("feature_name", row.get("feature"))
        if name_raw is None:
            idx = row.get("feature_index")
            if idx is None:
                continue
            name = f"f_{int(idx):04d}"
        else:
            name = str(name_raw)
        try:
            val = float(row.get("importance", 0.0))
        except Exception:
            val = 0.0
        fmap[name] = val
    return fmap


def _extract_feature_map_from_payload(payload: Optional[Dict[str, Any]]) -> Tuple[Dict[str, float], Optional[str]]:
    if not isinstance(payload, dict) or not payload:
        return {}, None

    candidates: List[Tuple[str, Dict[str, float]]] = []

    model_reported = payload.get("model_reported")
    if isinstance(model_reported, dict):
        fmap = _rows_to_feature_map(model_reported.get("importances"))
        if fmap:
            candidates.append(("feature_importance.model_reported", fmap))

    classifier_specific = payload.get("classifier_specific")
    if isinstance(classifier_specific, dict):
        preferred_keys = (
            "tree_model_feature_importances",
            "tabpfn_extensions_interpretability",
        )
        for key in preferred_keys:
            section = classifier_specific.get(key)
            if isinstance(section, dict):
                fmap = _rows_to_feature_map(section.get("importances"))
                if fmap:
                    candidates.append((f"feature_importance.classifier_specific.{key}", fmap))
        for key, section in classifier_specific.items():
            if key in preferred_keys or not isinstance(section, dict):
                continue
            fmap = _rows_to_feature_map(section.get("importances"))
            if fmap:
                candidates.append((f"feature_importance.classifier_specific.{key}", fmap))

    model_agnostic = payload.get("model_agnostic")
    if isinstance(model_agnostic, dict):
        fmap = _rows_to_feature_map(model_agnostic.get("importances"))
        if fmap:
            candidates.append(("feature_importance.model_agnostic", fmap))

    if not candidates:
        return {}, None

    # Prefer classifier-specific explanations, then model-reported, then model-agnostic.
    source_order = {
        "feature_importance.classifier_specific.tree_model_feature_importances": 0,
        "feature_importance.classifier_specific.tabpfn_extensions_interpretability": 1,
        "feature_importance.model_reported": 2,
        "feature_importance.model_agnostic": 3,
    }
    candidates.sort(key=lambda item: source_order.get(item[0], 10))
    return candidates[0][1], candidates[0][0]


def _aligned_feature_importances(
    feature_keys: Sequence[str],
    train_info: Optional[Dict[str, Any]],
) -> Optional[List[float]]:
    if not feature_keys or not isinstance(train_info, dict):
        return None

    values = _to_float_list(train_info.get("feature_importances"))
    if values is None:
        return None

    n_keys = len(feature_keys)
    if len(values) == n_keys:
        return values

    keep_indices_raw = train_info.get("feature_importances_keep_indices")
    reduced_raw = train_info.get("feature_importances_reduced")
    keep_indices = (
        [int(v) for v in keep_indices_raw]
        if isinstance(keep_indices_raw, list)
        else None
    )
    reduced_values = _to_float_list(reduced_raw)

    if isinstance(keep_indices, list) and keep_indices:
        source_values = reduced_values if (reduced_values is not None and len(reduced_values) == len(keep_indices)) else values
        if len(source_values) == len(keep_indices):
            full = [0.0] * n_keys
            for idx, val in zip(keep_indices, source_values):
                if 0 <= idx < n_keys:
                    full[idx] = float(val)
            return full

    # Legacy fallback: truncate / zero-pad by index.
    full = [0.0] * n_keys
    upto = min(len(values), n_keys)
    for i in range(upto):
        full[i] = float(values[i])
    return full


def _feature_importance_map_from_artifacts(
    artefacts: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, float], Optional[str]]:
    if not isinstance(artefacts, dict):
        return {}, None
    feature_keys = artefacts.get("feature_keys")
    if not isinstance(feature_keys, list) or not feature_keys:
        return {}, None
    aligned = _aligned_feature_importances(feature_keys, artefacts.get("train_info"))
    if aligned is None:
        return {}, None
    return (
        {str(k): float(v) for k, v in zip(feature_keys, aligned)},
        "artefacts.train_info.feature_importances",
    )


def _build_feature_importance_view(
    keys: Sequence[str],
    importances: Sequence[float],
    *,
    top_n: int = 10,
    meta: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if not keys or not importances:
        return None
    pairs = [
        {"feature": str(k), "importance": float(v)}
        for k, v in zip(keys, importances)
    ]
    if not pairs:
        return None
    desc = sorted(pairs, key=lambda item: item["importance"], reverse=True)
    asc = sorted(pairs, key=lambda item: item["importance"])
    return {
        "top": desc[:top_n],
        "bottom": asc[:top_n],
        "meta": meta or {},
    }


def _single_feature_importance_view(classification: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(classification, dict):
        return None

    fi_payload = classification.get("feature_importance_data")
    fmap, source = _extract_feature_map_from_payload(fi_payload if isinstance(fi_payload, dict) else None)

    if not fmap:
        artefacts = classification.get("artefacts")
        fmap, source = _feature_importance_map_from_artifacts(
            artefacts if isinstance(artefacts, dict) else None
        )

    if not fmap:
        return None

    keys = list(fmap.keys())
    values = [float(fmap[k]) for k in keys]
    return _build_feature_importance_view(
        keys,
        values,
        top_n=10,
        meta={"mode": "single", "source": source or "unknown"},
    )


def _loso_feature_importance_view(
    fold_entries: List[Dict[str, Any]],
    entries: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    fold_maps: List[Dict[str, float]] = []
    key_order: List[str] = []
    key_seen: set[str] = set()
    total_folds = 0
    used_folds = 0

    for fold in fold_entries:
        raw_id = fold.get("exp_id")
        if not raw_id:
            continue
        total_folds += 1
        raw_entry = entries.get(raw_id)
        if not raw_entry:
            continue
        exp_path = Path(raw_entry["path"])

        fi_payload = load_json(exp_path / "classification" / "feature_importance.json") or {}
        fmap, _ = _extract_feature_map_from_payload(fi_payload)
        if not fmap:
            art = load_json(exp_path / "classification" / "artefacts.json") or {}
            fmap, _ = _feature_importance_map_from_artifacts(art)
        if not fmap:
            continue

        fold_maps.append(fmap)
        used_folds += 1
        for key_str in fmap.keys():
            if key_str not in key_seen:
                key_seen.add(key_str)
                key_order.append(key_str)

    if not fold_maps or not key_order or used_folds <= 0:
        return None

    mean_values: List[float] = []
    for key in key_order:
        total = 0.0
        for fmap in fold_maps:
            total += float(fmap.get(key, 0.0))
        mean_values.append(total / float(used_folds))

    return _build_feature_importance_view(
        key_order,
        mean_values,
        top_n=10,
        meta={
            "mode": "loso_mean",
            "source": "mean_over_loso_folds",
            "folds_used": used_folds,
            "folds_total": total_folds,
        },
    )


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


def trajectory_filter_summary_metrics(exp_path: Path) -> Optional[Dict]:
    """Load trajectory filter summary from ``<exp>/test/trajectory_filter/summary.json``."""
    summary_path = exp_path / "test" / "trajectory_filter" / "summary.json"
    if not summary_path.exists():
        return None
    return load_json(summary_path)


def filtered_detection_summary_metrics(exp_path: Path) -> Optional[Dict]:
    """Load post-filter detection summary from trajectory-filter outputs.

    Source file: ``<exp>/test/trajectory_filter/filtered_detection_summary.json``.
    """
    summary_path = exp_path / "test" / "trajectory_filter" / "filtered_detection_summary.json"
    if not summary_path.exists():
        return None
    data = load_json(summary_path)
    return first_value(data) if data else None


def gather_classification_metrics(exp_path: Path) -> Dict:
    """Collect classification summary, predictions and artefact metadata."""
    cls_dir = exp_path / "classification"
    summary = load_json(cls_dir / "summary.json") if (cls_dir / "summary.json").exists() else None
    predictions = load_json(cls_dir / "predictions.json") if (cls_dir / "predictions.json").exists() else None
    artefacts = load_json(cls_dir / "artefacts.json") if (cls_dir / "artefacts.json").exists() else None
    feature_importance_data = (
        load_json(cls_dir / "feature_importance.json")
        if (cls_dir / "feature_importance.json").exists()
        else None
    )
    return {
        "summary": summary,
        "predictions": predictions or [],
        "artefacts": artefacts,
        "feature_importance_data": feature_importance_data,
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
            tf_preview = trajectory_filter_summary_metrics(exp_path)
            fdet_preview = filtered_detection_summary_metrics(exp_path)
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
                "has_trajectory_filter": (exp_path / "test" / "trajectory_filter" / "summary.json").exists(),
                "has_detection_visuals": (exp_path / "test" / "detection" / "visualizations").exists(),
                "has_segmentation_visuals": bool(list((exp_path / "test" / "segmentation").rglob("visualizations_roi"))) if (exp_path / "test" / "segmentation").exists() else False,
                "preview": {
                    "detection": det_preview,
                    "segmentation": seg_preview,
                    "classification": cls_preview,
                    "trajectory_filter": tf_preview,
                    "filtered_detection": fdet_preview,
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

    def _resolve_note_file(self, group_path: str = "", exp_id: Optional[str] = None) -> Path:
        group_path = str(group_path or "").strip().strip("/")
        if group_path:
            note_dir = (self.root / Path(group_path)).resolve()
            try:
                note_dir.relative_to(self.root)
            except ValueError as exc:
                raise ValueError("Invalid group path") from exc
            return note_dir / "note.txt"
        if exp_id:
            return self.get_path(exp_id) / "note.txt"
        raise KeyError("Missing note target")

    def load_schedule_note(self, group_path: str = "", exp_id: Optional[str] = None) -> str:
        try:
            note_path = self._resolve_note_file(group_path=group_path, exp_id=exp_id)
        except KeyError:
            return ""
        if not note_path.exists():
            return ""
        try:
            return note_path.read_text(encoding="utf-8").strip()
        except Exception:
            return ""

    def save_schedule_note(self, text: str, group_path: str = "", exp_id: Optional[str] = None) -> Path:
        note_path = self._resolve_note_file(group_path=group_path, exp_id=exp_id)
        note_path.parent.mkdir(parents=True, exist_ok=True)
        note_path.write_text(str(text or ""), encoding="utf-8")
        return note_path

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
            # Override classification roc_auc / brier decomposition with correctly pooled values
            # (individual-fold AUC is often null for LOSO with 1 test subject).
            try:
                _fold_list = [{"exp_id": f.get("id")} for f in fold_public]
                _, _pool_preds = _build_loso_combined(_fold_list, self.entries)
                if _pool_preds:
                    _yt = [p.get("label_true") for p in _pool_preds
                           if p.get("label_true") is not None and p.get("prob_positive") is not None]
                    _yp = [p.get("prob_positive") for p in _pool_preds
                           if p.get("label_true") is not None and p.get("prob_positive") is not None]
                    if len(_yt) == len(_yp) and len(_yt) > 0:
                        _yt_bin = [1.0 if int(y) == 1 else 0.0 for y in _yt]
                        _yp_clip = [max(0.0, min(1.0, float(p))) for p in _yp]
                        _pooled_brier = float(sum((p - o) ** 2 for p, o in zip(_yp_clip, _yt_bin)) / float(len(_yt_bin)))
                        _n = len(_yt_bin)
                        _o_bar = sum(_yt_bin) / float(_n)
                        _n_bins = 10
                        _groups: List[List[int]] = [[] for _ in range(_n_bins)]
                        for _idx, _p in enumerate(_yp_clip):
                            _bin_idx = min(int(_p * _n_bins), _n_bins - 1)
                            _groups[_bin_idx].append(_idx)
                        _rel = 0.0
                        _res = 0.0
                        for _indices in _groups:
                            if not _indices:
                                continue
                            _nk = float(len(_indices))
                            _fk = sum(_yp_clip[i] for i in _indices) / _nk
                            _ok = sum(_yt_bin[i] for i in _indices) / _nk
                            _wk = _nk / float(_n)
                            _rel += _wk * (_fk - _ok) ** 2
                            _res += _wk * (_ok - _o_bar) ** 2
                        _unc = float(_o_bar * (1.0 - _o_bar))
                        if aggregate_preview["classification"] is None:
                            aggregate_preview["classification"] = {}
                        aggregate_preview["classification"]["brier_score"] = _pooled_brier
                        aggregate_preview["classification"]["reliability"] = float(_rel)
                        aggregate_preview["classification"]["resolution"] = float(_res)
                        aggregate_preview["classification"]["uncertainty"] = _unc
                    if len(set(_yt)) >= 2:
                        from sklearn.metrics import roc_auc_score as _roc_fn  # noqa: PLC0415
                        _pooled_auc = float(_roc_fn(_yt, _yp))
                        if aggregate_preview["classification"] is None:
                            aggregate_preview["classification"] = {}
                        aggregate_preview["classification"]["roc_auc"] = _pooled_auc
            except Exception:
                pass
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
                "schedule_note": self.load_schedule_note(group_path, newest_raw.get("id")),
                "created_at": created_at,
                "preprocs": newest_raw.get("preprocs", []),
                "models": newest_raw.get("models", []),
                "has_detection": any(bool(f.get("has_detection")) for f in fold_public),
                "has_segmentation": any(bool(f.get("has_segmentation")) for f in fold_public),
                "has_classification": any(bool(f.get("has_classification")) for f in fold_public),
                "has_trajectory_filter": any(
                    self.entries.get(f.get("id"), {}).get("has_trajectory_filter") for f in fold_public
                ),
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
            "schedule_note": self.load_schedule_note(entry.get("group_path", ""), entry.get("id")),
            "created_at": entry["created_at"],
            "preprocs": entry["preprocs"],
            "models": entry["models"],
            "has_detection": entry["has_detection"],
            "has_segmentation": entry["has_segmentation"],
            "has_classification": entry.get("has_classification", False),
            "has_trajectory_filter": entry.get("has_trajectory_filter", False),
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
            if summary_file.exists():
                # Old flat layout: metrics/{vid_stem}/summary.json
                data = load_json(summary_file)
                metrics = first_value(data) if data else None
                if metrics:
                    per_video.append({
                        "video": item.name,
                        "metrics": metrics,
                    })
            else:
                # New two-level layout: metrics/{subject_id}/{vid_stem}/summary.json
                for subitem in sorted(item.iterdir()):
                    if not subitem.is_dir():
                        continue
                    sub_summary_file = subitem / "summary.json"
                    if not sub_summary_file.exists():
                        continue
                    data = load_json(sub_summary_file)
                    metrics = first_value(data) if data else None
                    if metrics:
                        per_video.append({
                            "video": f"{item.name}/{subitem.name}",
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
            # Show subject_id/filename to distinguish videos across subjects
            parts = Path(video_path).parts
            if len(parts) >= 2:
                label = f"{parts[-2]}/{Path(video_path).stem}"
            else:
                label = os.path.basename(video_path)
            per_video.append({
                "video": label,
                "metrics": metrics,
            })
    return {
        "summary": summary_metrics,
        "per_video": per_video,
    }


def gather_trajectory_filter_metrics(exp_path: Path) -> Optional[Dict]:
    """Gather before/after trajectory filter metrics for the web viewer.

    Reads ``test/trajectory_filter/summary.json`` (dataset-level aggregates)
    and ``test/trajectory_filter/metrics.json`` (per-video before/after).
    Also reads ``test/trajectory_filter/filtered_detection_summary.json`` when
    available (IoU / CE / SR re-evaluated on the smoothed predictions).
    """
    tf_dir = exp_path / "test" / "trajectory_filter"
    summary_path = tf_dir / "summary.json"
    metrics_path = tf_dir / "metrics.json"
    fdet_path = tf_dir / "filtered_detection_summary.json"

    if not summary_path.exists():
        return None

    summary = load_json(summary_path)
    per_video_data = load_json(metrics_path) if metrics_path.exists() else None
    filtered_detection_data = load_json(fdet_path) if fdet_path.exists() else None

    # Flatten per-model per-video into a single per-video list for the UI
    per_video: List[Dict] = []
    if isinstance(per_video_data, dict):
        for model_name, videos in per_video_data.items():
            if not isinstance(videos, dict):
                continue
            for vid_stem, vid_metrics in videos.items():
                if not isinstance(vid_metrics, dict):
                    continue
                per_video.append({
                    "model": model_name,
                    "video": vid_stem,
                    "before": vid_metrics.get("before"),
                    "after": vid_metrics.get("after"),
                    "frames": vid_metrics.get("frames"),
                })

    # Build filtered_detection summary using same shape as detection summary
    filtered_detection_summary = first_value(filtered_detection_data) if filtered_detection_data else None

    return {
        "summary": summary,
        "per_video": per_video,
        "filtered_detection_summary": filtered_detection_summary,
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

    @app.post("/api/schedules/note")
    def save_schedule_note(payload: Dict[str, Any] = Body(...)):
        group_path = str(payload.get("group_path") or "").strip()
        exp_id = payload.get("exp_id")
        text = str(payload.get("text") or "")
        try:
            note_path = index.save_schedule_note(text=text, group_path=group_path, exp_id=exp_id)
        except KeyError:
            raise HTTPException(status_code=400, detail="Missing note target")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"ok": True, "note": text, "path": str(note_path)}

    def _experiment_payload(exp_id: str):
        # Aggregated LOSO entry (virtual experiment id)
        folds = index.get_loso_folds(exp_id)
        if folds:
            fold_payloads: List[Dict] = []
            det_fold_previews: List[Optional[Dict]] = []
            seg_fold_previews: List[Optional[Dict]] = []
            cls_fold_previews: List[Optional[Dict]] = []
            tf_fold_previews: List[Optional[Dict]] = []
            fdet_fold_previews: List[Optional[Dict]] = []
            newest_created_at = None
            newest_entry: Optional[Dict] = None

            for fold in folds:
                raw_id = fold.get("exp_id")
                if not raw_id:
                    continue
                raw_entry = index.entries.get(raw_id)
                if not raw_entry:
                    continue
                raw_exp_path = Path(raw_entry["path"])
                created_at = raw_entry.get("created_at")
                if created_at and (newest_created_at is None or str(created_at) > str(newest_created_at)):
                    newest_created_at = created_at
                    newest_entry = raw_entry

                det_fold_previews.append((raw_entry.get("preview") or {}).get("detection"))
                seg_fold_previews.append((raw_entry.get("preview") or {}).get("segmentation"))
                cls_fold_previews.append((raw_entry.get("preview") or {}).get("classification"))
                tf_fold_previews.append((raw_entry.get("preview") or {}).get("trajectory_filter"))
                fdet_direct = filtered_detection_summary_metrics(raw_exp_path)
                fdet_fold_previews.append(
                    fdet_direct if fdet_direct is not None else (raw_entry.get("preview") or {}).get("filtered_detection")
                )
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
            aggregate_trajectory_filter = aggregate_preview_dicts(tf_fold_previews)
            aggregate_filtered_detection = aggregate_preview_dicts(fdet_fold_previews)

            # Combined (summed) classification stats across all LOSO folds
            combined_cls_summary, combined_predictions = _build_loso_combined(
                folds, index.entries
            )
            loso_voting = (
                _compute_voting_analysis(combined_predictions)
                if combined_predictions
                else None
            )

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
                "classification": {
                    "summary": combined_cls_summary or aggregate_classification,
                    "combined": combined_cls_summary,
                    "predictions": combined_predictions,
                    "artefacts": None,
                    "feature_importance": _loso_feature_importance_view(folds, index.entries),
                },
                "trajectory_filter": {"summary": aggregate_trajectory_filter, "per_video": []} if aggregate_trajectory_filter else None,
                "filtered_detection": {"summary": aggregate_filtered_detection} if aggregate_filtered_detection else None,
                "loso_voting": loso_voting,
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
        if isinstance(classification, dict):
            classification["feature_importance"] = _single_feature_importance_view(classification)
        trajectory_filter = gather_trajectory_filter_metrics(exp_path)
        # Expose filtered-detection accuracy (IoU/CE/SR after smoothing) as a
        # dedicated top-level key so the frontend can render it alongside the
        # raw detection summary without altering the trajectory_filter section.
        filtered_detection_summary = (
            (trajectory_filter or {}).get("filtered_detection_summary")
        )
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
            "trajectory_filter": trajectory_filter,
            "filtered_detection": {"summary": filtered_detection_summary} if filtered_detection_summary else None,
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
                    # Store both the full key ("001/Grasp") and basename fallback ("Grasp")
                    ce_map[str(video)] = ce_val
                    video_name = os.path.basename(str(video))
                    ce_map[str(video_name)] = ce_val
                    ce_map[Path(video_name).stem] = ce_val
            for item in items:
                parts = Path(item.get("label", "")).parts
                if "visualizations" in parts:
                    idx = parts.index("visualizations")
                    if idx + 1 < len(parts):
                        video_name = parts[idx + 1]
                        # Support both flat (vid_stem) and two-level (subject_id/vid_stem) layouts
                        # If parts[idx+1] is a subject dir (no summary.json at that level), look one deeper
                        _flat_metrics = exp_path / "test" / "detection" / "metrics" / video_name
                        if not _flat_metrics.exists() and idx + 2 < len(parts):
                            video_name = str(Path(parts[idx + 1]) / parts[idx + 2])
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
                            # Try both full path key ("001/Grasp") and basename key ("Grasp")
                            ce_val = ce_map.get(video_name) or ce_map.get(Path(video_name).name)
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
                    # map full key, basename and stem for robustness
                    ce_map[str(video)] = ce_val
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
        # Aggregate LOSO: combine predictions from all folds
        agg_folds = index.get_loso_folds(exp_id)
        if agg_folds:
            _, all_preds = _build_loso_combined(agg_folds, index.entries)
            return Response(
                content=json.dumps(_sanitize(_compute_voting_analysis(all_preds))),
                media_type="application/json",
            )
        # Single experiment
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
        try:
            target.relative_to(exp_path.resolve())
        except ValueError:
            raise HTTPException(status_code=404, detail="Resource not found")
        if not target.exists():
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
