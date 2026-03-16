from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..core.interfaces import FramePrediction
from ..core.registry import (
    CLASSIFIER_REGISTRY,
    FEATURE_EXTRACTOR_REGISTRY,
)
from .feature_vector import FeatureVectoriser
from .metrics import summarise_classification
from . import feature_extractors as _load_feature_extractors  # noqa: F401
from . import classifiers as _load_classifiers  # noqa: F401
from ..segmentation.dataset import attach_ground_truth_segmentation
from ..utils.annotations import load_coco_vid


def _subject_from_video_path(path: str, dataset_root: str | None = None) -> str:
    """Derive subject ID from video path.

    Strategy:
    1. If *dataset_root* is given, compute the relative path; if there is a
       parent directory component, use the first directory name as subject.
    2. Otherwise, fall back to extracting leading digits from the filename stem.
    """
    # Try directory-based derivation first
    if dataset_root:
        try:
            rel = os.path.relpath(path, dataset_root)
            parts = Path(rel).parts
            if len(parts) > 1:
                # e.g. "n001/D.avi" → subject = "n001"
                return parts[0]
        except Exception:
            pass

    # Fallback: leading digits from filename
    stem = os.path.splitext(os.path.basename(path))[0]
    digits = []
    for ch in stem:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    if digits:
        return "".join(digits)
    if "_" in stem:
        return stem.split("_")[0]
    return stem


def _annotation_to_predictions(annotation: Dict[str, any]) -> List[FramePrediction]:  # noqa: ANN001
    frames = annotation.get("frames", {})
    samples: List[FramePrediction] = []
    for fi_str, boxes in frames.items():
        try:
            fi = int(fi_str)
        except Exception:
            try:
                fi = int(float(fi_str))
            except Exception:
                continue
        if not boxes:
            continue
        bbox = boxes[0]
        if bbox is None:
            continue
        try:
            x, y, w, h = map(float, bbox)
        except Exception:
            continue
        samples.append(FramePrediction(fi, (x, y, w, h), None))
    samples.sort(key=lambda s: s.frame_index)
    return samples


def _load_subject_labels(label_path: str) -> Dict[str, int]:
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")
    mapping: Dict[str, int] = {}
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            subj, label = parts[0], parts[1]
            try:
                mapping[subj] = int(label)
            except Exception:
                continue
    return mapping


def _load_annotations_for_videos(video_paths: Sequence[str]) -> Dict[str, Dict[str, any]]:  # noqa: ANN401
    annotations: Dict[str, Dict[str, any]] = {}
    for video_path in video_paths:
        json_path = os.path.splitext(video_path)[0] + ".json"
        if not os.path.exists(json_path):
            continue
        try:
            annotations[video_path] = load_coco_vid(json_path)
        except Exception:
            continue
    return annotations


def _prepare_entity_features(
    feature_extractor,
    videos: Dict[str, List[FramePrediction]],
    level: str,
    logger: Callable[[str], None],
    dataset_root: str | None = None,
    *,
    fit_batch: bool = True,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, str], Dict[str, List[str]]]:
    """Extract features for every entity (video or subject).

    If the feature_extractor exposes a ``finalize_batch`` method (e.g. for
    batch PCA on texture features), it is called on the collected video-level
    feature dicts **before** subject aggregation.

    Parameters
    ----------
    fit_batch : bool
        Forwarded to ``finalize_batch(fit=...)``.  Use ``True`` for the
        training set (to fit PCA etc.) and ``False`` for the test set.
    """
    level_lc = str(level).lower()
    features: Dict[str, Dict[str, float]] = {}
    owners: Dict[str, str] = {}
    sources: Dict[str, List[str]] = {}
    has_finalize = callable(getattr(feature_extractor, "finalize_batch", None))

    if level_lc == "subject":
        # 1) Extract video-level features
        subject_video_feats: Dict[str, List[Dict[str, float]]] = defaultdict(list)
        source_paths: Dict[str, List[str]] = defaultdict(list)
        all_video_feats: List[Dict[str, float]] = []
        video_meta: List[Tuple[str, int]] = []  # (subject, idx_in_subject_list)

        for video_path, samples in videos.items():
            subject = _subject_from_video_path(video_path, dataset_root)
            vf = feature_extractor.extract_video(samples, video_path=video_path)
            idx = len(subject_video_feats[subject])
            subject_video_feats[subject].append(vf)
            source_paths[subject].append(video_path)
            all_video_feats.append(vf)
            video_meta.append((subject, idx))

        # 2) Batch finalize (PCA etc.) on video-level features
        if has_finalize and all_video_feats:
            all_video_feats = list(feature_extractor.finalize_batch(all_video_feats, fit=fit_batch))
            # Rebuild per-subject dict from updated flat list
            subject_video_feats_new: Dict[str, List[Dict[str, float]]] = defaultdict(list)
            for (subj, _local_idx), vf in zip(video_meta, all_video_feats):
                subject_video_feats_new[subj].append(vf)
            subject_video_feats = subject_video_feats_new

        # 3) Aggregate to subject level
        for subject, feats in subject_video_feats.items():
            features[subject] = feature_extractor.aggregate_subject(feats)
            owners[subject] = subject
            sources[subject] = source_paths[subject]
            logger(f"[Classification] Subject {subject}: aggregated {len(feats)} video(s)")
    else:
        all_video_feats_flat: List[Dict[str, float]] = []
        entity_ids: List[str] = []

        for video_path, samples in videos.items():
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            subject = _subject_from_video_path(video_path, dataset_root)
            # Use subject/video_id as the unique entity key to avoid collision when
            # multiple subjects share the same video filename (e.g. 001/Grasp.avi
            # and 004/Grasp.avi both mapping to entity_id "Grasp").
            unique_eid = f"{subject}/{video_id}"
            vf = feature_extractor.extract_video(samples, video_path=video_path)
            all_video_feats_flat.append(vf)
            entity_ids.append(unique_eid)
            owners[unique_eid] = subject
            sources[unique_eid] = [video_path]
            logger(f"[Classification] Video {video_id} (subject={subject}): extracted {len(samples)} sample(s)")

        if has_finalize and all_video_feats_flat:
            all_video_feats_flat = list(feature_extractor.finalize_batch(all_video_feats_flat, fit=fit_batch))

        for eid, vf in zip(entity_ids, all_video_feats_flat):
            features[eid] = vf

    return features, owners, sources


def _build_vectoriser(
    feature_extractor,
    level: str,
) -> FeatureVectoriser:
    order = feature_extractor.feature_order(level)
    return FeatureVectoriser(order)


def _filter_entities(
    entity_features: Dict[str, Dict[str, float]],
    entity_to_subject: Dict[str, str],
    labels: Dict[str, int],
    logger: Callable[[str], None],
) -> Sequence[str]:
    kept: List[str] = []
    for entity in sorted(entity_features.keys()):
        subject = entity_to_subject.get(entity)
        if subject is None or subject not in labels:
            logger(
                f"[Classification] Skip entity {entity}: "
                + ("missing subject mapping" if subject is None else "label not found")
            )
            continue
        kept.append(entity)
    return kept


def _find_youden_threshold(
    y_true: np.ndarray,
    proba_positive: np.ndarray,
) -> Tuple[float, float]:
    """Find optimal decision threshold using Youden's Index.

    J(t) = Sensitivity(t) + Specificity(t) - 1 = TPR(t) + TNR(t) - 1

    Scanns all midpoint candidates between unique probabilities.

    Returns
    -------
    threshold : float
        Optimal decision threshold in [0, 1].
    j_score : float
        Youden's J statistic at the optimal threshold.
    """
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba_positive, dtype=float)

    unique_labels = np.unique(y_true)
    if len(unique_labels) != 2:
        return 0.5, 0.0  # degenerate: need exactly two classes
    neg_label, pos_label = int(unique_labels[0]), int(unique_labels[1])

    pos_total = int(np.sum(y_true == pos_label))
    neg_total = int(np.sum(y_true == neg_label))
    if pos_total == 0 or neg_total == 0:
        return 0.5, 0.0  # degenerate case

    unique_probs = np.unique(proba)
    if len(unique_probs) > 1:
        midpoints = (unique_probs[:-1] + unique_probs[1:]) / 2.0
        candidates = np.concatenate([[0.0, 0.5], unique_probs, midpoints])
    else:
        candidates = np.array([0.0, 0.5, float(unique_probs[0])])
    candidates = np.clip(np.sort(np.unique(candidates)), 0.0, 1.0)

    best_j = -2.0
    best_t = 0.5

    for t in candidates:
        y_hat = (proba >= t).astype(int)
        tp = int(np.sum((y_hat == 1) & (y_true == pos_label)))
        tn = int(np.sum((y_hat == 0) & (y_true == neg_label)))
        sensitivity = tp / pos_total
        specificity = tn / neg_total
        j = sensitivity + specificity - 1.0
        if j > best_j or (j == best_j and abs(t - 0.5) < abs(best_t - 0.5)):
            best_j = j
            best_t = float(t)

    return best_t, best_j


def _loo_calib_probabilities(
    train_entities: Sequence[str],
    train_features: Dict[str, Dict[str, float]],
    entity_to_subject: Dict[str, str],
    labels: Dict[str, int],
    vectoriser,
    classifier_cls,
    classifier_cfg: Dict,
    logger: Callable[[str], None],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute out-of-sample probabilities for Youden calibration via
    leave-one-subject-out on the training set.

    For every unique training subject we:
      1. Train a temporary classifier on all other training subjects.
      2. Predict probabilities for the left-out subject's entities.

    This guarantees that every prediction is truly out-of-sample regardless
    of how few subjects are available (works with as few as 2 training
    subjects total), and that both class labels appear in the combined
    probability array.

    Returns
    -------
    y_loo : np.ndarray  (N,)  true labels
    p_loo : np.ndarray  (N,)  predicted prob_positive
    """
    entities = list(train_entities)
    subjects = sorted({entity_to_subject[e] for e in entities if entity_to_subject.get(e) in labels})

    if len(subjects) < 2:
        return np.array([], dtype=int), np.array([], dtype=float)

    y_loo: List[int] = []
    p_loo: List[float] = []

    # For iterative deep-learning classifiers (e.g. PatchTST, TimeMachine) the
    # LOO fold training can be very slow with the full epoch budget.  Cap
    # training at a lighter setting so calibration remains feasible.
    # Detect by checking (1) user-supplied params, or (2) the class DEFAULT_CONFIG.
    _base_params: Dict = dict(classifier_cfg.get("params") or {})
    _default_cfg: Dict = getattr(classifier_cls, "DEFAULT_CONFIG", {}) or {}
    _effective_epochs   = int(_base_params.get("epochs",   _default_cfg.get("epochs",   0)))
    _effective_patience = int(_base_params.get("patience", _default_cfg.get("patience", 0)))
    if _effective_epochs > 0:
        _loo_max_epochs = min(_effective_epochs, 25)
        _loo_patience   = min(_effective_patience, 8) if _effective_patience > 0 else 5
        _loo_params: Dict = {**_default_cfg, **_base_params, "epochs": _loo_max_epochs, "patience": _loo_patience}
    else:
        _loo_params = _base_params

    logger(
        f"[Classification] Youden LOO calibration: {len(subjects)} folds"
        + (
            f" (epochs capped at {_loo_params.get('epochs')}, patience={_loo_params.get('patience')})"
            if _effective_epochs > 0
            else ""
        )
    )

    for _fold_idx, held_out in enumerate(subjects, 1):
        fit_ents = [e for e in entities if entity_to_subject.get(e) != held_out]
        val_ents = [e for e in entities if entity_to_subject.get(e) == held_out]

        if not fit_ents or not val_ents:
            logger(f"[Classification] LOO fold {_fold_idx}/{len(subjects)}: skip {held_out} (empty split)")
            continue

        # Ensure at least 2 distinct labels in fit set so classifier can learn
        fit_labels_set = {labels[entity_to_subject[e]] for e in fit_ents}
        if len(fit_labels_set) < 2:
            logger(f"[Classification] LOO fold {_fold_idx}/{len(subjects)}: skip {held_out} (single-class fit set)")
            continue

        logger(f"[Classification] LOO fold {_fold_idx}/{len(subjects)}: held-out={held_out}, fit_n={len(fit_ents)}, val_n={len(val_ents)}")
        X_fit = vectoriser.transform(train_features[e] for e in fit_ents)
        y_fit = np.asarray([labels[entity_to_subject[e]] for e in fit_ents], dtype=np.int64)
        X_val = vectoriser.transform(train_features[e] for e in val_ents)
        y_val_true = [labels[entity_to_subject[e]] for e in val_ents]

        try:
            clf = classifier_cls(_loo_params if _loo_params else None)
            clf.fit(X_fit, y_fit)
            prob_mat = clf.predict_proba(X_val)
            classes_ = getattr(clf, "classes_", None)
            if classes_ is None and hasattr(clf, "_model"):
                classes_ = getattr(getattr(clf, "_model"), "classes_", None)
            if classes_ is not None and 1 in classes_:
                pos_idx = int(np.where(classes_ == 1)[0][0])
            else:
                pos_idx = 1 if prob_mat.shape[1] > 1 else 0
            y_loo.extend(y_val_true)
            p_loo.extend(prob_mat[:, pos_idx].tolist())
        except Exception as _e:
            logger(f"[Classification] LOO calib: skip subject {held_out} ({_e})")
            continue

    return np.asarray(y_loo, dtype=int), np.asarray(p_loo, dtype=float)


def run_subject_classification(
    config: Dict[str, any],  # noqa: ANN001
    dataset_root: str,
    train_dataset,
    test_predictions: Dict[str, Dict[str, List[FramePrediction]]],
    results_dir: str,
    logger: Callable[[str], None],
    *,
    split_method: str = "video_level",
    cached_classifier_path: Optional[str] = None,
    cached_seg_path: Optional[str] = None,  # deprecated – kept for API compatibility; no longer used
) -> Optional[Dict[str, float]]:
    import traceback as _traceback_mod
    try:
        return _run_subject_classification_impl(
            config, dataset_root, train_dataset, test_predictions, results_dir, logger,
            split_method=split_method,
            cached_classifier_path=cached_classifier_path,
        )
    except Exception as _exc:
        logger(f"[Classification] TRACEBACK:\n{_traceback_mod.format_exc()}")
        raise


def _run_subject_classification_impl(
    config: Dict[str, any],  # noqa: ANN001
    dataset_root: str,
    train_dataset,
    test_predictions: Dict[str, Dict[str, List[FramePrediction]]],
    results_dir: str,
    logger: Callable[[str], None],
    *,
    split_method: str = "video_level",
    cached_classifier_path: Optional[str] = None,
) -> Optional[Dict[str, float]]:
    if not config.get("enabled", False):
        logger("[Classification] Stage disabled via config.")
        return None

    feature_cfg = config.get("feature_extractor", {"name": "basic", "params": {}})
    feature_name = feature_cfg.get("name", "basic")
    feature_cls = FEATURE_EXTRACTOR_REGISTRY.get(feature_name)
    if feature_cls is None:
        raise KeyError(f"Unknown feature extractor: {feature_name}")
    feature_extractor = feature_cls(feature_cfg.get("params"))

    classifier_cfg = config.get("classifier", {"name": "random_forest", "params": {}})
    classifier_name = classifier_cfg.get("name", "random_forest")
    classifier_cls = CLASSIFIER_REGISTRY.get(classifier_name)
    if classifier_cls is None:
        raise KeyError(f"Unknown classifier: {classifier_name}")
    classifier = classifier_cls(classifier_cfg.get("params"))

    legacy_level = config.get("target_level", None)
    if legacy_level is not None:
        logger("[Classification] 'target_level' is deprecated; using dataset split method instead.")
    _explicit_level = str(config.get("level") or "").strip().lower()
    if _explicit_level in {"video", "subject"}:
        level = _explicit_level
        logger(f"[Classification] Target level overridden by config: {level}")
    else:
        level = "subject" if str(split_method).lower() == "subject_level" else "video"
    logger(f"[Classification] Target level: {level}")

    seg_cfg = config.get("segmentation", {}) or {}
    seg_enabled = bool(seg_cfg.get("enabled", True))
    if "segmentation" in getattr(feature_extractor, "name", "").lower() and not seg_enabled:
        raise RuntimeError(
            "Segmentation-aware feature extractor selected but classification.segmentation.enabled is False."
        )

    label_path = config.get("label_file") or os.path.join(dataset_root, "ann.txt")
    labels = _load_subject_labels(label_path)
    logger(f"[Classification] Loaded {len(labels)} subject labels from {label_path}")

    # Collect training annotations and segmentation sequences
    train_video_paths: List[str] = []
    for idx in range(len(train_dataset)):
        item = train_dataset[idx]
        video_path = item.get("video_path")
        if video_path:
            train_video_paths.append(video_path)
    train_annotations = _load_annotations_for_videos(train_video_paths)

    # The classification stage does NOT train its own segmentation model.
    # Segmentation masks on test_predictions are produced by the main pipeline's
    # segmentation stage (e.g. MedNeXt) and should be used as-is.
    if not seg_enabled:
        logger("[Classification] Segmentation stage disabled via config; using existing trajectories only.")
    else:
        logger("[Classification] Segmentation masks from the main pipeline will be used for feature extraction (no internal seg model).")

    train_videos: Dict[str, List[FramePrediction]] = {}
    for video_path, annotation in train_annotations.items():
        samples = attach_ground_truth_segmentation(annotation, dataset_root, video_path=video_path)
        if samples:
            train_videos[video_path] = samples
    logger(f"[Classification] Prepared GT segmentation trajectories for {len(train_videos)} train videos")

    # ── GT trajectory smoothing ──────────────────────────────────────────
    # Ground-truth bboxes are clean (no tracking noise), so we skip the
    # Hampel outlier-removal stage.  However, bidirectional Savitzky-Golay
    # smoothing is still mandatory to guarantee C2-continuous centroid
    # trajectories, which in turn yields physically plausible velocity and
    # acceleration features that are consistent with those produced for
    # detector-output test trajectories.
    try:
        from .trajectory_filter import filter_detections as _filter_detections_gt
        _gt_traj_cfg = dict((config.get("trajectory_filter", {}) or {}).get("traj_params", {}) or {})
        # GT mode uses lightweight S-G parameters to remove only annotation jitter —
        # lower polyorder and smaller window preserve the biomechanical envelope more
        # faithfully than the standard inference-path parameters (sg_window=11, polyorder=2).
        # These defaults can be overridden via config["trajectory_filter"]["gt_traj_params"].
        _gt_traj_override = dict(
            (config.get("trajectory_filter", {}) or {}).get("gt_traj_params", {}) or {}
        )
        _GT_SG_WINDOW_DEFAULT = 7   # lighter than inference (11)
        _GT_SG_POLYORDER_DEFAULT = 1  # lighter than inference (2)
        _gt_traj_cfg.setdefault("sg_window", _GT_SG_WINDOW_DEFAULT)
        _gt_traj_cfg.setdefault("sg_polyorder", _GT_SG_POLYORDER_DEFAULT)
        # Allow explicit gt_traj_params in config to override even the above defaults
        _gt_traj_cfg.update(_gt_traj_override)
        _gt_bbox_strategy = str(
            (config.get("trajectory_filter", {}) or {}).get("bbox_strategy", "independent")
        )
        _gt_bbox_params = dict(
            (config.get("trajectory_filter", {}) or {}).get("bbox_params", {}) or {}
        )
        _gt_filtered_videos: Dict[str, List[FramePrediction]] = {}
        _gt_smoothed_count = 0
        for _gt_vp, _gt_preds in train_videos.items():
            if len(_gt_preds) < 2:
                _gt_filtered_videos[_gt_vp] = _gt_preds
                continue
            _gt_fi = np.array([p.frame_index for p in _gt_preds], dtype=np.int64)
            _gt_bb = np.array([list(p.bbox) for p in _gt_preds], dtype=np.float64)
            _gt_cx = _gt_bb[:, 0] + _gt_bb[:, 2] / 2.0
            _gt_cy = _gt_bb[:, 1] + _gt_bb[:, 3] / 2.0
            _gt_w = _gt_bb[:, 2].copy()
            _gt_h = _gt_bb[:, 3].copy()
            _gt_sc = np.array(
                # GT predictions: default to 1.0 (ground-truth is fully reliable)
                [float(p.score if p.score is not None else 1.0) for p in _gt_preds],
                dtype=np.float64,
            )
            _gt_result = _filter_detections_gt(
                _gt_fi, _gt_cx, _gt_cy, _gt_w, _gt_h, _gt_sc,
                bbox_strategy=_gt_bbox_strategy,
                bbox_params=_gt_bbox_params,
                traj_params=_gt_traj_cfg,
                skip_hampel=True,  # GT mode: bypass Hampel, force S-G only
            )
            # Rebuild FramePrediction list with smoothed bboxes.
            # Use a dict lookup keyed by frame_index so we handle duplicates/reordering.
            _gt_fi_result = _gt_result["frame_indices"]
            _gt_fi_map = {int(fi): idx for idx, fi in enumerate(_gt_fi_result)}
            _gt_new_preds: list = []
            for _gi, _gp in enumerate(_gt_preds):
                _gk = _gt_fi_map.get(int(_gp.frame_index))
                if _gk is None:
                    # Frame was removed during filtering (e.g. dedup); keep original.
                    _gt_new_preds.append(_gp)
                    continue
                _ncx = float(_gt_result["cx"][_gk])
                _ncy = float(_gt_result["cy"][_gk])
                _nw = float(_gt_result["widths"][_gk])
                _nh = float(_gt_result["heights"][_gk])
                _nx = _ncx - _nw / 2.0
                _ny = _ncy - _nh / 2.0
                _gt_new_preds.append(FramePrediction(
                    frame_index=_gp.frame_index,
                    bbox=(_nx, _ny, _nw, _nh),
                    score=_gp.score,
                    segmentation=getattr(_gp, "segmentation", None),
                ))
            _gt_filtered_videos[_gt_vp] = _gt_new_preds
            _gt_smoothed_count += 1
        train_videos = _gt_filtered_videos
        logger(
            f"[Classification] GT trajectory smoothing applied (skip_hampel=True, S-G only) "
            f"to {_gt_smoothed_count} train videos"
        )
    except Exception as _gt_err:
        logger(f"[Classification] WARNING: GT trajectory smoothing failed ({_gt_err}); using raw GT bboxes")

    train_features, train_owner, train_sources = _prepare_entity_features(
        feature_extractor,
        train_videos,
        level,
        logger,
        dataset_root=dataset_root,
    )
    train_entities = _filter_entities(train_features, train_owner, labels, logger)
    if not train_entities:
        raise RuntimeError(
            "No training entities with labels available for classification stage."
        )

    vectoriser = _build_vectoriser(feature_extractor, level)
    X_train = vectoriser.transform(train_features[e] for e in train_entities)
    y_train = np.asarray([labels[train_owner[e]] for e in train_entities], dtype=np.int64)
    logger(f"[Classification] X_train shape={X_train.shape}, n_entities={len(train_entities)}, n_feat_keys={len(list(vectoriser.keys))}")

    # ── Threshold calibration via Youden's Index ─────────────────────────────
    threshold_cfg = config.get("threshold", {}) or {}
    threshold_method = str(threshold_cfg.get("method", "youden")).lower()

    decision_threshold: float = float(threshold_cfg.get("value", 0.5)) if threshold_method == "fixed" else 0.5
    youden_j_val: Optional[float] = None
    n_loo_predictions: int = 0  # number of out-of-sample predictions used for Youden calibration

    if threshold_method == "youden":
        # Use leave-one-subject-out on the training set to get out-of-sample
        # probabilities for every training entity.  This works robustly even
        # when only 2–3 subjects per class are available (common in LOSO).
        try:
            from tracking.classification.classifiers_ext import set_progress_logger  # noqa: PLC0415
            set_progress_logger(None)
        except ImportError:
            pass
        y_loo, p_loo = _loo_calib_probabilities(
            train_entities, train_features, train_owner, labels,
            vectoriser, classifier_cls, classifier_cfg, logger,
        )
        try:
            from tracking.classification.classifiers_ext import set_progress_logger  # noqa: PLC0415
            set_progress_logger(logger)
        except ImportError:
            pass

        n_unique_labels = len(set(y_loo.tolist())) if len(y_loo) else 0
        if n_unique_labels >= 2:
            decision_threshold, youden_j_val = _find_youden_threshold(y_loo, p_loo)
            n_loo_predictions = int(len(y_loo))
            logger(
                f"[Classification] Youden (LOO) threshold = {decision_threshold:.4f} "
                f"(J = {youden_j_val:.4f}, n_loo_predictions = {len(y_loo)})"
            )
        else:
            logger(
                f"[Classification] Youden LOO calibration degenerate "
                f"(n_loo={len(y_loo)}, unique_labels={n_unique_labels}); using 0.5"
            )

    if threshold_method == "fixed":
        logger(f"[Classification] Using fixed threshold: {decision_threshold:.4f}")
    # ─────────────────────────────────────────────────────────────────────────

    model_dir = os.path.join(results_dir, "classification")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "classifier.pkl")
    train_info: Optional[Dict] = None

    if cached_classifier_path and os.path.exists(cached_classifier_path):
        logger(f"[Classification] Cache hit – loading classifier from: {cached_classifier_path}")
        try:
            classifier.load(cached_classifier_path)
            # Copy to current experiment's dir so artifacts are self-contained
            if os.path.abspath(cached_classifier_path) != os.path.abspath(model_path):
                import shutil
                shutil.copy2(cached_classifier_path, model_path)
        except Exception as _load_err:
            logger(f"[Classification] Cache load failed ({_load_err}); falling back to training.")
            logger(f"[Classification] Training classifier on {len(train_entities)} entities")
            try:
                from tracking.classification.classifiers_ext import set_progress_logger
                set_progress_logger(logger)
            except ImportError:
                pass
            train_info = classifier.fit(X_train, y_train)
            try:
                from tracking.classification.classifiers_ext import set_progress_logger
                set_progress_logger(None)
            except ImportError:
                pass
            try:
                classifier.save(model_path)
            except Exception:
                logger("[Classification] Warning: classifier save skipped (not implemented)")
    else:
        logger(f"[Classification] Training classifier on {len(train_entities)} entities")
        try:
            from tracking.classification.classifiers_ext import set_progress_logger
            set_progress_logger(logger)
        except ImportError:
            pass
        train_info = classifier.fit(X_train, y_train)
        try:
            from tracking.classification.classifiers_ext import set_progress_logger
            set_progress_logger(None)  # revert to print
        except ImportError:
            pass
        try:
            classifier.save(model_path)
        except Exception:
            logger("[Classification] Warning: classifier save skipped (not implemented)")

    # Determine model predictions for test subjects
    source_model = config.get("source_model")
    if source_model is None and test_predictions:
        source_model = next(iter(test_predictions.keys()))
        logger(f"[Classification] source_model not specified, defaulting to {source_model}")
    if not source_model or source_model not in test_predictions:
        raise RuntimeError("Classification source_model not found in tracking predictions")

    if seg_enabled:
        missing_masks = sum(
            1 for _video, preds in test_predictions[source_model].items() for pred in preds if pred.segmentation is None
        )
        if missing_masks:
            # Warn but do not abort – frames without masks will contribute zero
            # CTS static features, which is acceptable for a partial prediction.
            logger(
                f"[Classification] Warning: {missing_masks} test frame(s) are missing "
                "segmentation masks from the main pipeline. CTS/TS static features will be partial for those frames."
            )

    test_features, test_owner, test_sources = _prepare_entity_features(
        feature_extractor,
        test_predictions[source_model],
        level,
        logger,
        dataset_root=dataset_root,
        fit_batch=False,
    )
    test_entities = _filter_entities(test_features, test_owner, labels, logger)
    if not test_entities:
        raise RuntimeError(
            "No test entities with labels available for classification evaluation."
        )

    X_test = vectoriser.transform(test_features[e] for e in test_entities)
    y_test = np.asarray([labels[test_owner[e]] for e in test_entities], dtype=np.int64)
    logger(f"[Classification] X_train shape: {X_train.shape}, X_test shape: {X_test.shape}, n_feat_keys: {len(list(vectoriser.keys))}")

    logger(f"[Classification] Evaluating classifier on {len(test_entities)} entities")
    y_pred_default = classifier.predict(X_test)
    prob_positive: List[float] = []
    try:
        prob_matrix = classifier.predict_proba(X_test)
        class_indices = getattr(classifier, "classes_", None)
        if class_indices is None and hasattr(classifier, "_model"):
            class_indices = getattr(getattr(classifier, "_model"), "classes_", None)
        if class_indices is not None and 1 in class_indices:
            pos_idx = int(np.where(class_indices == 1)[0][0])
        else:
            pos_idx = 1 if prob_matrix.shape[1] > 1 else 0
        prob_positive = prob_matrix[:, pos_idx].tolist()
    except Exception:
        prob_positive = [0.0 for _ in y_pred_default]

    # Apply the calibrated threshold (Youden or fixed) to prob_positive
    if prob_positive and any(p > 0.0 for p in prob_positive):
        y_pred = np.where(
            np.asarray(prob_positive) >= decision_threshold, 1, 0
        ).astype(np.int64)
        logger(
            f"[Classification] Threshold applied: {decision_threshold:.4f} "
            f"(method={threshold_method})"
        )
    else:
        # No valid probabilities – fall back to default predict()
        y_pred = y_pred_default
        decision_threshold = 0.5
        logger("[Classification] Falling back to default predict() (no valid probabilities).")

    metrics = summarise_classification(y_test, y_pred, prob_positive)
    metrics["threshold_used"] = round(decision_threshold, 6)
    metrics["threshold_method"] = threshold_method
    if youden_j_val is not None:
        metrics["youden_j"] = round(youden_j_val, 6)
    if n_loo_predictions:
        metrics["threshold_n_loo_predictions"] = n_loo_predictions
    logger(
        "[Classification] Metrics: "
        + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float)))
    )

    with open(os.path.join(model_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    artefacts = {
        "target_level": level,
        "train_entities": list(train_entities),
        "train_entity_sources": train_sources,
        "train_entity_subject_map": train_owner,
        "test_entities": list(test_entities),
        "test_entity_sources": test_sources,
        "test_entity_subject_map": test_owner,
        "feature_keys": list(vectoriser.keys),
        "train_info": train_info,
        "threshold_method": threshold_method,
        "threshold_used": round(decision_threshold, 6),
        "youden_j": round(youden_j_val, 6) if youden_j_val is not None else None,
        "threshold_n_loo_predictions": n_loo_predictions,
    }
    with open(os.path.join(model_dir, "artefacts.json"), "w", encoding="utf-8") as f:
        json.dump(artefacts, f, ensure_ascii=False, indent=2)

    rows = []
    for idx, entity in enumerate(test_entities):
        subject_id = test_owner.get(entity)
        source_paths = test_sources.get(entity, [])
        # Strip the subject/ prefix added for uniqueness in video-level mode so
        # that entity_id in the output is just the video stem (e.g. "Grasp"),
        # which the web viewer's canonical-modality mapping expects.
        display_entity_id = entity.rsplit("/", 1)[-1] if "/" in entity else entity
        rows.append(
            {
                "entity_id": display_entity_id,
                "subject_id": subject_id,
                "source_videos": source_paths,
                "label_true": int(y_test[idx]),
                "label_pred": int(y_pred[idx]),
                "prob_positive": float(prob_positive[idx]) if prob_positive else 0.0,
            }
        )
    with open(os.path.join(model_dir, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    return metrics

