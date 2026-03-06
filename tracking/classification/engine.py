from __future__ import annotations

import json
import os
import random
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
from ..segmentation import SegmentationWorkflow
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
            vf = feature_extractor.extract_video(samples, video_path=video_path)
            all_video_feats_flat.append(vf)
            entity_ids.append(video_id)
            owners[video_id] = _subject_from_video_path(video_path, dataset_root)
            sources[video_id] = [video_path]
            logger(f"[Classification] Video {video_id}: extracted {len(samples)} sample(s)")

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
) -> Optional[Dict[str, float]]:
    import traceback as _traceback_mod
    try:
        return _run_subject_classification_impl(
            config, dataset_root, train_dataset, test_predictions, results_dir, logger,
            split_method=split_method, cached_classifier_path=cached_classifier_path,
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

    legacy_level = config.pop("target_level", None)
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
    annotated_train_paths = list(train_annotations.keys())

    seg_results_root = os.path.join(results_dir, "test", "segmentation")
    seg_train_root = os.path.join(results_dir, "train_full", "segmentation", "classification")
    seg_workflow: Optional[SegmentationWorkflow] = None
    seg_metrics_by_model: Dict[str, Dict[str, Dict[str, float]]] = {}
    if seg_enabled:
        os.makedirs(seg_results_root, exist_ok=True)
        os.makedirs(seg_train_root, exist_ok=True)
        seg_params = seg_cfg.get("params")
        seg_workflow = SegmentationWorkflow(seg_params, dataset_root, seg_train_root, logger)
        seg_seed = int(seg_cfg.get("seed", 0))
        val_ratio = float(seg_cfg.get("val_ratio", 0.0))
        train_targets = annotated_train_paths[:]
        val_videos: Optional[Sequence[str]] = None
        if val_ratio > 0.0 and len(train_targets) > 1:
            val_count = max(1, int(len(train_targets) * val_ratio))
            rng = random.Random(seg_seed)
            shuffled = train_targets[:]
            rng.shuffle(shuffled)
            val_videos = shuffled[:val_count]
            train_targets = shuffled[val_count:] or shuffled
        if train_targets:
            seg_train_info = seg_workflow.train(train_targets, val_videos, seed=seg_seed)
            if seg_train_info:
                logger("[Segmentation] Training summary: " + ", ".join(f"{k}={v:.4f}" for k, v in seg_train_info.items()))
            seg_workflow.load_checkpoint()
        else:
            logger("[Segmentation] No annotated training videos found; skipping segmentation training.")
            seg_workflow = None
    else:
        logger("[Classification] Segmentation stage disabled via config; using existing trajectories only.")

    train_videos: Dict[str, List[FramePrediction]] = {}
    for video_path, annotation in train_annotations.items():
        samples = attach_ground_truth_segmentation(annotation, dataset_root)
        if samples:
            train_videos[video_path] = samples
    logger(f"[Classification] Prepared GT segmentation trajectories for {len(train_videos)} train videos")

    test_video_paths: List[str] = sorted({vp for model_map in test_predictions.values() for vp in model_map.keys()})
    test_annotations = _load_annotations_for_videos(test_video_paths)
    if seg_workflow is not None:
        seg_model_name = getattr(seg_workflow, "model_name", "segmentation_model")
        predictions_root = os.path.join(seg_results_root, "predictions", seg_model_name)
        os.makedirs(predictions_root, exist_ok=True)
        for model_name, preds_by_video in test_predictions.items():
            out_dir = os.path.join(predictions_root, model_name)
            os.makedirs(out_dir, exist_ok=True)
            # Use inference-only mode for test predictions inside the classification
            # engine. GT-evaluation mode would only cover GT-annotated frames, leaving
            # detector-only frames without segmentation masks and causing a crash in
            # the missing-masks check below.  Inference mode ensures every detected
            # frame receives a mask from the just-trained internal segmentation model.
            metrics_payload = seg_workflow.predict_dataset(preds_by_video, out_dir, gt_annotations=None)
            summary_metrics = metrics_payload.get("summary", {})
            per_video_metrics = metrics_payload.get("videos", {})
            if per_video_metrics:
                try:
                    with open(os.path.join(out_dir, "metrics_per_video.json"), "w", encoding="utf-8") as f:
                        json.dump(per_video_metrics, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass
            seg_metrics_by_model.setdefault(seg_model_name, {})[model_name] = summary_metrics
            if summary_metrics:
                summary_text = ", ".join(f"{k}={v:.4f}" for k, v in summary_metrics.items())
            else:
                summary_text = "no metrics"
            logger(
                f"[Segmentation] Summary metrics | seg_model={seg_model_name} det_model={model_name}: {summary_text}"
            )

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

    if seg_workflow is not None:
        missing_masks = sum(
            1 for _video, preds in test_predictions[source_model].items() for pred in preds if pred.segmentation is None
        )
        if missing_masks:
            # Warn but do not abort – frames without masks will contribute zero
            # CTS static features, which is acceptable for a partial prediction.
            logger(
                f"[Classification] Warning: {missing_masks} test frame(s) are missing "
                "segmentation masks. CTS static features will be partial for those frames."
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
    y_pred = classifier.predict(X_test)
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
        prob_positive = [0.0 for _ in y_pred]

    metrics = summarise_classification(y_test, y_pred, prob_positive)
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
    }
    with open(os.path.join(model_dir, "artefacts.json"), "w", encoding="utf-8") as f:
        json.dump(artefacts, f, ensure_ascii=False, indent=2)

    rows = []
    for idx, entity in enumerate(test_entities):
        subject_id = test_owner.get(entity)
        source_paths = test_sources.get(entity, [])
        rows.append(
            {
                "entity_id": entity,
                "subject_id": subject_id,
                "source_videos": source_paths,
                "label_true": int(y_test[idx]),
                "label_pred": int(y_pred[idx]),
                "prob_positive": float(prob_positive[idx]) if prob_positive else 0.0,
            }
        )
    with open(os.path.join(model_dir, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    if seg_metrics_by_model:
        seg_metrics_path = os.path.join(seg_results_root, "metrics_summary.json")
        with open(seg_metrics_path, "w", encoding="utf-8") as f:
            json.dump(seg_metrics_by_model, f, ensure_ascii=False, indent=2)

    return metrics

