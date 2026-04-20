from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from tracking.classification.classifiers.nn import MLPLinearHeadClassifier
from tracking.classification.feature_extractors.v5 import (
    _load_moment_pipeline_v5,
    _resolve_moment_device_v5,
    _trajectory_to_moment_input_v5,
)
from tracking.classification.feature_extractors.v6 import (
    resolve_depth_scale_for_video,
    scale_predictions_to_cm,
)
from tracking.classification.metrics import summarise_classification
from tracking.core.interfaces import FramePrediction
from tracking.utils.annotations import load_coco_vid

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None  # type: ignore


@dataclass
class VideoRecord:
    subject: str
    label: int
    video_path: str
    annotation_path: str
    samples: List[FramePrediction]


def _load_labels(label_path: str) -> Dict[str, int]:
    labels: Dict[str, int] = {}
    with open(label_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            labels[str(parts[0]).strip()] = int(parts[1])
    return labels


def _build_samples_from_annotation(annotation: Dict[str, Any]) -> List[FramePrediction]:
    frame_map = annotation.get("frames", {}) or {}
    samples: List[FramePrediction] = []
    for frame_idx, bbox_list in frame_map.items():
        if not bbox_list:
            continue
        bbox = bbox_list[0]
        samples.append(
            FramePrediction(
                frame_index=int(frame_idx),
                bbox=tuple(float(v) for v in bbox),
                score=None,
            )
        )
    samples.sort(key=lambda item: int(item.frame_index))
    return samples


def _discover_records(dataset_root: str, labels: Dict[str, int]) -> List[VideoRecord]:
    root = Path(dataset_root)
    records: List[VideoRecord] = []
    for subject_dir in sorted(root.iterdir()):
        if not subject_dir.is_dir():
            continue
        subject = subject_dir.name
        if subject not in labels:
            continue
        for video_path in sorted(subject_dir.iterdir()):
            if video_path.suffix.lower() not in {".avi", ".wmv", ".mp4", ".mov"}:
                continue
            annotation_path = video_path.with_suffix(".json")
            if not annotation_path.is_file():
                continue
            annotation = load_coco_vid(str(annotation_path))
            samples = _build_samples_from_annotation(annotation)
            if len(samples) < 2:
                continue
            records.append(
                VideoRecord(
                    subject=subject,
                    label=int(labels[subject]),
                    video_path=str(video_path),
                    annotation_path=str(annotation_path),
                    samples=samples,
                )
            )
    return records


def _pooled_moment_embedding(
    model: Any,
    samples: Sequence[FramePrediction],
    *,
    target_steps: int,
) -> np.ndarray:
    if torch is None:
        raise RuntimeError("PyTorch is required for MOMENT benchmarking.")
    try:
        device = next(model.parameters()).device
    except Exception:  # noqa: BLE001
        device = torch.device("cpu")

    motion_input = _trajectory_to_moment_input_v5(samples, target_steps=target_steps)
    x_enc = torch.tensor(motion_input[None, ...], dtype=torch.float32, device=device)
    input_mask = torch.ones((1, target_steps), dtype=torch.float32, device=device)

    with torch.no_grad():
        output = model(x_enc=x_enc, input_mask=input_mask, reduction="none")

    emb = output.embeddings
    emb_np = emb.detach().cpu().numpy().astype(np.float64) if hasattr(emb, "detach") else np.asarray(emb, dtype=np.float64)
    if emb_np.ndim == 4:
        pooled = emb_np.mean(axis=(1, 2))[0]
    elif emb_np.ndim == 3:
        pooled = emb_np.mean(axis=1)[0]
    elif emb_np.ndim == 2:
        pooled = emb_np[0]
    else:
        pooled = emb_np.reshape(-1)
    return np.asarray(pooled, dtype=np.float64).reshape(-1)


def _extract_embeddings(
    records: Sequence[VideoRecord],
    *,
    use_scale_cm: bool,
    moment_model_name: str,
    moment_input_steps: int,
    moment_device: str,
) -> Dict[str, np.ndarray]:
    resolved_device = _resolve_moment_device_v5(moment_device)
    model = _load_moment_pipeline_v5(moment_model_name, resolved_device)
    out: Dict[str, np.ndarray] = {}
    for record in records:
        samples = record.samples
        if use_scale_cm:
            depth_scale = resolve_depth_scale_for_video(record.video_path)
            samples = scale_predictions_to_cm(samples, depth_scale)
        out[record.video_path] = _pooled_moment_embedding(
            model,
            samples,
            target_steps=moment_input_steps,
        )
    return out


def _pca_fit(X: np.ndarray, target_dim: int) -> tuple[np.ndarray, np.ndarray]:
    n, d = X.shape
    if d <= target_dim:
        mean = np.zeros((d,), dtype=np.float64)
        comps = np.eye(target_dim, d, dtype=np.float64)
        return mean, comps
    mean = X.mean(axis=0)
    Xc = X - mean
    actual = min(target_dim, n, d)
    try:
        _, _, vt = np.linalg.svd(Xc, full_matrices=False)
        comps = vt[:actual]
    except np.linalg.LinAlgError:
        comps = np.eye(target_dim, d, dtype=np.float64)
    return mean, comps


def _pca_transform(X: np.ndarray, mean: np.ndarray, comps: np.ndarray, target_dim: int) -> np.ndarray:
    reduced = (X - mean) @ comps.T
    if reduced.shape[1] < target_dim:
        reduced = np.hstack(
            [reduced, np.zeros((reduced.shape[0], target_dim - reduced.shape[1]), dtype=reduced.dtype)]
        )
    return reduced[:, :target_dim]


def _subject_mean(
    records: Sequence[VideoRecord],
    features_by_video: Dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    grouped: Dict[str, List[np.ndarray]] = {}
    labels: Dict[str, int] = {}
    for record in records:
        grouped.setdefault(record.subject, []).append(np.asarray(features_by_video[record.video_path], dtype=np.float64))
        labels[record.subject] = int(record.label)
    subjects = sorted(grouped)
    X = np.vstack([np.vstack(grouped[subj]).mean(axis=0) for subj in subjects]).astype(np.float64)
    y = np.asarray([labels[subj] for subj in subjects], dtype=np.int64)
    return X, y, subjects


def _fit_logreg(train_X: np.ndarray, train_y: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=4000,
        random_state=42,
    )
    clf.fit(train_X, train_y)
    return clf


def _positive_proba_from_model(clf: Any, X: np.ndarray) -> np.ndarray:
    probs = np.asarray(clf.predict_proba(X), dtype=np.float64)
    classes = np.asarray(getattr(clf, "classes_", np.array([0, 1])), dtype=np.int64)
    if probs.ndim != 2 or probs.shape[0] != X.shape[0]:
        raise RuntimeError("Classifier returned unexpected probability shape.")
    if 1 in classes.tolist():
        pos_idx = int(np.where(classes == 1)[0][0])
    else:
        pos_idx = int(probs.shape[1] - 1)
    return probs[:, pos_idx]


def _evaluate_freeze_pca(
    records: Sequence[VideoRecord],
    embeddings_by_video: Dict[str, np.ndarray],
    *,
    pca_dim: int,
) -> Dict[str, Any]:
    subjects = sorted({record.subject for record in records})
    y_true: List[int] = []
    y_prob: List[float] = []
    fold_details: List[Dict[str, Any]] = []

    for heldout_subject in subjects:
        train_records = [record for record in records if record.subject != heldout_subject]
        test_records = [record for record in records if record.subject == heldout_subject]
        train_video_matrix = np.vstack([embeddings_by_video[record.video_path] for record in train_records]).astype(np.float64)
        mean, comps = _pca_fit(train_video_matrix, pca_dim)

        reduced_by_video: Dict[str, np.ndarray] = {}
        for record in train_records + test_records:
            vec = np.asarray(embeddings_by_video[record.video_path], dtype=np.float64).reshape(1, -1)
            reduced_by_video[record.video_path] = _pca_transform(vec, mean, comps, pca_dim)[0]

        train_video_reduced = np.vstack([reduced_by_video[record.video_path] for record in train_records]).astype(np.float64)
        feat_mean = train_video_reduced.mean(axis=0)
        feat_std = train_video_reduced.std(axis=0)
        feat_std = np.where(feat_std < 1e-9, 1.0, feat_std)

        standardized_by_video = {
            video_path: (vec - feat_mean) / feat_std for video_path, vec in reduced_by_video.items()
        }
        train_subject_X, train_subject_y, train_subjects = _subject_mean(train_records, standardized_by_video)
        test_subject_X, test_subject_y, test_subjects = _subject_mean(test_records, standardized_by_video)

        clf = _fit_logreg(train_subject_X, train_subject_y)
        test_prob = _positive_proba_from_model(clf, test_subject_X)
        y_true.extend(test_subject_y.tolist())
        y_prob.extend(test_prob.tolist())
        fold_details.append(
            {
                "heldout_subject": heldout_subject,
                "train_subjects": train_subjects,
                "test_subjects": test_subjects,
                "y_true": int(test_subject_y[0]),
                "y_prob": float(test_prob[0]),
            }
        )

    y_pred = [1 if prob >= 0.5 else 0 for prob in y_prob]
    metrics = summarise_classification(y_true, y_pred, y_prob=y_prob, positive_label=1)
    metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return {
        "mode": "freeze_pca_logreg",
        "n_subjects": len(subjects),
        "n_videos": len(records),
        "metrics": metrics,
        "folds": fold_details,
    }


def _evaluate_projection_head(
    records: Sequence[VideoRecord],
    embeddings_by_video: Dict[str, np.ndarray],
    *,
    projection_dim: int,
    device: str,
    mode: str = "mlp",
    hidden_dims: Optional[Sequence[int]] = None,
    epochs: int = 160,
    dropout: float = 0.1,
) -> Dict[str, Any]:
    subjects = sorted({record.subject for record in records})
    y_true: List[int] = []
    y_prob: List[float] = []
    fold_details: List[Dict[str, Any]] = []

    mode_norm = str(mode).strip().lower()
    if mode_norm not in {"mlp", "linear"}:
        raise ValueError(f"Unsupported head mode: {mode}. Expected 'mlp' or 'linear'.")

    if hidden_dims is None:
        resolved_hidden_dims: List[int] = [int(projection_dim)] if mode_norm == "mlp" else []
    else:
        resolved_hidden_dims = [int(v) for v in hidden_dims if int(v) > 0]
        if mode_norm == "linear":
            resolved_hidden_dims = []
        elif not resolved_hidden_dims:
            resolved_hidden_dims = [int(projection_dim)]

    for heldout_subject in subjects:
        train_records = [record for record in records if record.subject != heldout_subject]
        test_records = [record for record in records if record.subject == heldout_subject]
        train_subject_X, train_subject_y, train_subjects = _subject_mean(train_records, embeddings_by_video)
        test_subject_X, test_subject_y, test_subjects = _subject_mean(test_records, embeddings_by_video)

        clf = MLPLinearHeadClassifier(
            {
                "mode": mode_norm,
                "hidden_dims": resolved_hidden_dims,
                "dropout": float(dropout),
                "epochs": int(epochs),
                "batch_size": 8,
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "val_ratio": 0.2,
                "patience": 20,
                "class_weight": "balanced",
                "normalize": True,
                "seed": 42,
                "device": device,
            }
        )
        clf.fit(train_subject_X.astype(np.float32), train_subject_y.astype(np.int64))
        test_prob = _positive_proba_from_model(clf, test_subject_X.astype(np.float32))
        y_true.extend(test_subject_y.tolist())
        y_prob.extend(test_prob.tolist())
        fold_details.append(
            {
                "heldout_subject": heldout_subject,
                "train_subjects": train_subjects,
                "test_subjects": test_subjects,
                "y_true": int(test_subject_y[0]),
                "y_prob": float(test_prob[0]),
            }
        )

    y_pred = [1 if prob >= 0.5 else 0 for prob in y_prob]
    metrics = summarise_classification(y_true, y_pred, y_prob=y_prob, positive_label=1)
    metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return {
        "mode": "learnable_projection_head" if mode_norm == "mlp" else "direct_linear_head",
        "n_subjects": len(subjects),
        "n_videos": len(records),
        "classifier_config": {
            "mode": mode_norm,
            "hidden_dims": resolved_hidden_dims,
            "dropout": float(dropout),
            "epochs": int(epochs),
            "batch_size": 8,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "val_ratio": 0.2,
            "patience": 20,
            "class_weight": "balanced",
            "normalize": True,
            "seed": 42,
            "device": device,
        },
        "metrics": metrics,
        "folds": fold_details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark pure MOMENT trajectory features on merged_extend.")
    parser.add_argument("--dataset-root", default="dataset/merged_extend")
    parser.add_argument("--label-file", default=None)
    parser.add_argument("--moment-model-name", default="AutonLab/MOMENT-1-base")
    parser.add_argument("--moment-input-steps", type=int, default=256)
    parser.add_argument("--moment-device", default="auto")
    parser.add_argument("--pca-dim", type=int, default=12)
    parser.add_argument("--head-epochs", type=int, default=160)
    parser.add_argument("--direct-head-epochs", type=int, default=None)
    parser.add_argument("--direct-mlp-hidden-dim", type=int, default=256)
    parser.add_argument("--direct-mlp-epochs", type=int, default=None)
    parser.add_argument("--output", default="analysis/moment_trajectory_benchmark_merged_extend.json")
    args = parser.parse_args()

    dataset_root = os.path.abspath(args.dataset_root)
    label_file = os.path.abspath(args.label_file or os.path.join(dataset_root, "ann.txt"))
    labels = _load_labels(label_file)
    records = _discover_records(dataset_root, labels)
    if not records:
        raise RuntimeError(f"No usable records found under {dataset_root}")

    emb_px = _extract_embeddings(
        records,
        use_scale_cm=False,
        moment_model_name=str(args.moment_model_name),
        moment_input_steps=int(args.moment_input_steps),
        moment_device=str(args.moment_device),
    )
    emb_cm = _extract_embeddings(
        records,
        use_scale_cm=True,
        moment_model_name=str(args.moment_model_name),
        moment_input_steps=int(args.moment_input_steps),
        moment_device=str(args.moment_device),
    )
    sample_dim = int(next(iter(emb_cm.values())).shape[0])
    resolved_device = _resolve_moment_device_v5(str(args.moment_device))
    head_epochs = int(args.head_epochs)
    direct_head_epochs = int(args.direct_head_epochs) if args.direct_head_epochs is not None else head_epochs
    direct_mlp_epochs = int(args.direct_mlp_epochs) if args.direct_mlp_epochs is not None else head_epochs

    results = {
        "dataset_root": dataset_root,
        "label_file": label_file,
        "n_subjects": len({record.subject for record in records}),
        "n_videos": len(records),
        "moment_model_name": str(args.moment_model_name),
        "moment_input_steps": int(args.moment_input_steps),
        "moment_device": resolved_device,
        "raw_embedding_dim": sample_dim,
        "head_epochs": head_epochs,
        "direct_head_epochs": direct_head_epochs,
        "direct_mlp_hidden_dim": int(args.direct_mlp_hidden_dim),
        "direct_mlp_epochs": direct_mlp_epochs,
        "experiments": {
            "raw_px_freeze_pca_logreg": _evaluate_freeze_pca(records, emb_px, pca_dim=int(args.pca_dim)),
            "cm_freeze_pca_logreg": _evaluate_freeze_pca(records, emb_cm, pca_dim=int(args.pca_dim)),
            "cm_learnable_projection_head": _evaluate_projection_head(
                records,
                emb_cm,
                projection_dim=int(args.pca_dim),
                device=resolved_device,
                mode="mlp",
                hidden_dims=[int(args.pca_dim)],
                epochs=head_epochs,
            ),
            "cm_direct_linear_head": _evaluate_projection_head(
                records,
                emb_cm,
                projection_dim=sample_dim,
                device=resolved_device,
                mode="linear",
                hidden_dims=[],
                epochs=direct_head_epochs,
                dropout=0.0,
            ),
            "cm_direct_mlp_head": _evaluate_projection_head(
                records,
                emb_cm,
                projection_dim=sample_dim,
                device=resolved_device,
                mode="mlp",
                hidden_dims=[int(args.direct_mlp_hidden_dim)],
                epochs=direct_mlp_epochs,
            ),
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
