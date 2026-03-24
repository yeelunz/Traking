from __future__ import annotations

import json
import os
import csv
import pickle
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..core.interfaces import FramePrediction
from ..core.registry import (
    CLASSIFIER_REGISTRY,
    FEATURE_EXTRACTOR_REGISTRY,
    PREPROC_REGISTRY,
)
from .feature_vector import FeatureVectoriser
from .metrics import summarise_classification
from . import feature_extractors as _load_feature_extractors  # noqa: F401
from . import classifiers as _load_classifiers  # noqa: F401
from . import classifiers_limix as _load_classifiers_limix  # noqa: F401
from . import fusion_modules as _load_fusion_modules  # noqa: F401
from .fusion_modules import create_fusion_module, is_learnable_fusion_module
from ..segmentation.dataset import attach_ground_truth_segmentation
from ..utils.annotations import load_coco_vid


_TEXTURE_BACKBONE_FEATURES = {
    "tab_v3_pro",
    "tab_v4",
    "tab_v2",
    "tab_v2_extend",
    "tsc_v2",
    "tsc_v2_extend",
    "tsc_v3_pro",
}


def _normalise_subject_token(token: Optional[str]) -> Optional[str]:
    if token is None:
        return None
    cleaned = str(token).strip()
    if not cleaned:
        return None
    digits = []
    for ch in cleaned:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    if digits:
        return "".join(digits)
    return cleaned


def _infer_texture_embedding_dim(feature_name: str, params: Dict[str, Any], tp_cfg: Dict[str, Any]) -> int:
    if "embedding_dim" in tp_cfg:
        return int(tp_cfg.get("embedding_dim", 32))
    if "texture_dim" in params:
        return int(params.get("texture_dim", 32))

    if feature_name == "tab_v3_pro":
        try:
            from .feature_extractors_v3pro import MotionStaticV3ProFeatureExtractor

            default_cfg = dict(getattr(MotionStaticV3ProFeatureExtractor, "DEFAULT_CONFIG", {}) or {})
            return int(default_cfg.get("texture_dim", 10))
        except Exception:
            return 10
    if feature_name == "tab_v4":
        try:
            from .feature_extractors_v4 import TAB_V4_TOTAL_DIM, TAB_V4_MOTION_KEYS, TAB_V4_STATIC_KEYS

            return int(TAB_V4_TOTAL_DIM - len(TAB_V4_MOTION_KEYS) - len(TAB_V4_STATIC_KEYS))
        except Exception:
            return 11
    if feature_name == "tsc_v3_pro":
        try:
            from .feature_extractors_v3pro_tsc import N_TEX_CHANNELS_V3PRO

            return int(N_TEX_CHANNELS_V3PRO)
        except Exception:
            return 3
    if feature_name == "tab_v2":
        try:
            from .feature_extractors import MOTION_FEATURE_KEYS
            from .feature_extractors_ext import CTS_STATIC_FEATURE_KEYS

            return int(len(MOTION_FEATURE_KEYS) + len(CTS_STATIC_FEATURE_KEYS))
        except Exception:
            return 66
    if feature_name == "tab_v2_extend":
        try:
            from .feature_extractors import MOTION_FEATURE_KEYS
            from .feature_extractors_ext import CTS_STATIC_V2_FEATURE_KEYS

            return int(len(MOTION_FEATURE_KEYS) + len(CTS_STATIC_V2_FEATURE_KEYS))
        except Exception:
            return 90
    if feature_name == "tsc_v2":
        try:
            from .feature_extractors_ext import N_TEX_PCA_TS

            return int(params.get("tex_pca_dim", N_TEX_PCA_TS))
        except Exception:
            return int(params.get("tex_pca_dim", 8))
    if feature_name == "tsc_v2_extend":
        try:
            from .feature_extractors_ext import N_TEX_PCA_TS_V2

            return int(params.get("tex_pca_dim", N_TEX_PCA_TS_V2))
        except Exception:
            return int(params.get("tex_pca_dim", 8))
    return 32


_EXTRACTOR_STATE_ATTRS = (
    "_pca_mean",
    "_pca_components",
    "_proj_matrix",
    "_feat_mean",
    "_feat_std",
    "_channel_mean",
    "_channel_std",
    "_tex_pca_mean",
    "_tex_pca_components",
)


def _get_feature_extractor_state(feature_extractor) -> Optional[Dict[str, any]]:  # noqa: ANN001
    getter = getattr(feature_extractor, "get_state", None)
    if callable(getter):
        state = getter()
        if isinstance(state, dict) and state:
            return state

    state: Dict[str, any] = {}  # noqa: ANN001
    for attr in _EXTRACTOR_STATE_ATTRS:
        if hasattr(feature_extractor, attr):
            value = getattr(feature_extractor, attr)
            if value is not None:
                state[attr] = value
    return state if state else None


def _set_feature_extractor_state(feature_extractor, state: Dict[str, any]) -> None:  # noqa: ANN001
    setter = getattr(feature_extractor, "set_state", None)
    if callable(setter):
        setter(state)
        return
    for attr, value in (state or {}).items():
        if hasattr(feature_extractor, attr):
            setattr(feature_extractor, attr, value)


def _save_feature_extractor_state(
    feature_extractor,
    model_dir: str,
    logger: Callable[[str], None],
) -> Optional[str]:
    state = _get_feature_extractor_state(feature_extractor)
    if not state:
        return None
    out_path = os.path.join(model_dir, "feature_extractor_state.pkl")
    payload = {
        "extractor_class": feature_extractor.__class__.__name__,
        "state": state,
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    logger(f"[Classification] Saved feature extractor state: {out_path}")
    return out_path


def _try_load_feature_extractor_state(
    feature_extractor,
    config: Dict[str, any],  # noqa: ANN001
    logger: Callable[[str], None],
) -> Optional[str]:
    path = str((config or {}).get("feature_extractor_state_path", "")).strip()
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"feature_extractor_state_path not found: {path}")
    with open(path, "rb") as f:
        payload = pickle.load(f)
    state = payload.get("state") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise RuntimeError("Invalid feature extractor state payload.")
    _set_feature_extractor_state(feature_extractor, state)
    logger(f"[Classification] Loaded feature extractor state: {path}")
    return path


def _resolve_feature_names(feature_keys: Sequence[str], n_features: int) -> List[str]:
    keys = list(feature_keys or [])
    if len(keys) == int(n_features):
        return keys
    return [f"f_{i:04d}" for i in range(int(n_features))]


def _positive_class_probabilities(classifier, X):  # noqa: ANN001
    prob_matrix = classifier.predict_proba(X)
    class_indices = getattr(classifier, "classes_", None)
    if class_indices is None and hasattr(classifier, "_model"):
        class_indices = getattr(getattr(classifier, "_model"), "classes_", None)
    if class_indices is not None and 1 in class_indices:
        pos_idx = int(np.where(class_indices == 1)[0][0])
    else:
        pos_idx = 1 if prob_matrix.shape[1] > 1 else 0
    return np.asarray(prob_matrix[:, pos_idx], dtype=np.float64)


def _compute_permutation_feature_importance(
    classifier,
    X,
    y,
    feature_names: Sequence[str],
    *,
    max_features: int = 96,
    max_samples: int = 256,
    random_state: int = 42,
) -> Optional[Dict[str, Any]]:  # noqa: ANN401
    Xn = np.asarray(X, dtype=np.float32)
    yn = np.asarray(y, dtype=np.int64)
    if Xn.ndim != 2 or Xn.shape[0] == 0 or Xn.shape[1] == 0:
        return None

    rng = np.random.default_rng(int(random_state))

    if Xn.shape[0] > int(max_samples):
        picked_rows = rng.choice(Xn.shape[0], size=int(max_samples), replace=False)
        X_eval = Xn[picked_rows]
        y_eval = yn[picked_rows]
    else:
        X_eval = Xn
        y_eval = yn

    feat_names = _resolve_feature_names(feature_names, X_eval.shape[1])
    if X_eval.shape[1] > int(max_features):
        var = np.var(X_eval, axis=0)
        keep_idx = np.argsort(var)[-int(max_features):]
        keep_idx = np.sort(keep_idx)
    else:
        keep_idx = np.arange(X_eval.shape[1])

    try:
        base_probs = _positive_class_probabilities(classifier, X_eval)
    except Exception:
        return None

    base_pred = (base_probs >= 0.5).astype(np.int64)
    base_acc = float(np.mean(base_pred == y_eval))

    scored: List[Dict[str, Any]] = []
    for j in keep_idx.tolist():
        X_perm = X_eval.copy()
        X_perm[:, j] = X_perm[rng.permutation(X_perm.shape[0]), j]
        try:
            perm_probs = _positive_class_probabilities(classifier, X_perm)
        except Exception:
            continue
        perm_pred = (perm_probs >= 0.5).astype(np.int64)
        perm_acc = float(np.mean(perm_pred == y_eval))
        scored.append(
            {
                "feature_index": int(j),
                "feature_name": str(feat_names[j]),
                "importance": float(base_acc - perm_acc),
                "base_accuracy": base_acc,
                "permuted_accuracy": perm_acc,
            }
        )

    scored.sort(key=lambda r: r["importance"], reverse=True)
    return {
        "method": "permutation_accuracy_drop",
        "base_accuracy": base_acc,
        "n_samples_evaluated": int(X_eval.shape[0]),
        "n_features_total": int(X_eval.shape[1]),
        "n_features_evaluated": int(len(scored)),
        "importances": scored,
    }


def _collect_feature_importance_payload(
    *,
    classifier,
    train_info: Optional[Dict[str, Any]],
    X_eval,
    y_eval,
    feature_names: Sequence[str],
    logger: Callable[[str], None],
) -> Optional[Dict[str, Any]]:
    payload: Dict[str, Any] = {}

    if isinstance(train_info, dict) and isinstance(train_info.get("feature_importances"), list):
        values = [float(v) for v in train_info.get("feature_importances", [])]
        names = _resolve_feature_names(feature_names, len(values))
        rows = [
            {
                "feature_index": int(i),
                "feature_name": str(names[i]),
                "importance": float(values[i]),
            }
            for i in range(len(values))
        ]
        rows.sort(key=lambda r: r["importance"], reverse=True)
        payload["model_reported"] = {
            "method": "model_reported",
            "importances": rows,
        }

    getter = getattr(classifier, "get_feature_importance", None)
    if callable(getter):
        try:
            custom = getter(X_eval, y_eval, feature_names=_resolve_feature_names(feature_names, np.asarray(X_eval).shape[1]))
            if isinstance(custom, dict) and custom:
                payload["classifier_specific"] = custom
        except Exception as exc:  # noqa: BLE001
            logger(f"[Classification] Warning: classifier-specific feature importance failed: {exc}")

    try:
        perm = _compute_permutation_feature_importance(
            classifier,
            X_eval,
            y_eval,
            feature_names=_resolve_feature_names(feature_names, np.asarray(X_eval).shape[1]),
        )
        if perm:
            payload["model_agnostic"] = perm
    except Exception as exc:  # noqa: BLE001
        logger(f"[Classification] Warning: permutation feature importance failed: {exc}")

    return payload or None


def _crop_roi_safe(frame, bbox, pad_ratio: float = 0.15):
    if frame is None or bbox is None or len(bbox) != 4:
        return None
    h_img, w_img = frame.shape[:2]
    x, y, w, h = map(float, bbox)
    if w <= 1.0 or h <= 1.0:
        return None

    pad_ratio = max(0.0, float(pad_ratio))
    pw = w * pad_ratio
    ph = h * pad_ratio
    x1 = int(max(0, min(w_img - 1, round(x - pw))))
    y1 = int(max(0, min(h_img - 1, round(y - ph))))
    x2 = int(max(x1 + 1, min(w_img, round(x + w + pw))))
    y2 = int(max(y1 + 1, min(h_img, round(y + h + ph))))
    roi = frame[y1:y2, x1:x2]
    if roi is None or roi.size == 0:
        return None
    return roi


def _apply_preprocs_frame_like_segmentation(frame: np.ndarray, preprocs: Sequence[Any]) -> np.ndarray:
    import cv2  # noqa: PLC0415

    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if not preprocs:
        return frame
    if frame.ndim == 2:
        out = frame
        for p in preprocs:
            out = p.apply_to_frame(out)
        return out
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    for p in preprocs:
        rgb = p.apply_to_frame(rgb)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _build_runtime_preprocs(classification_cfg: Dict[str, Any]) -> Tuple[List[Any], List[Any]]:
    runtime = dict((classification_cfg or {}).get("runtime_preprocessing") or {})
    scheme_raw = runtime.get("scheme", "A")
    scheme = str(scheme_raw).strip().upper()
    if scheme in {"GLOBAL", "A"}:
        scheme = "A"
    elif scheme in {"ROI", "B"}:
        scheme = "B"
    elif scheme in {"HYBRID", "C"}:
        scheme = "C"
    else:
        scheme = "A"

    step_defs = runtime.get("preproc_steps") or []
    if isinstance(step_defs, (str, bytes)) or not isinstance(step_defs, Sequence):
        return [], []

    global_preprocs: List[Any] = []
    roi_preprocs: List[Any] = []
    for step in step_defs:
        if not isinstance(step, dict):
            continue
        name = step.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        cls = PREPROC_REGISTRY.get(name)
        if cls is None:
            continue
        params = step.get("params", {})
        if scheme == "A":
            global_preprocs.append(cls(params))
        elif scheme == "B":
            roi_preprocs.append(cls(params))
        else:
            # Scheme C follows segmentation runtime behavior:
            # detector keeps global preprocs, but segmentation (and texture-pretrain
            # ROI extraction that mirrors segmentation) crops from RAW then applies
            # ROI preprocs. Therefore classification-side ROI generation should not
            # apply any global preprocs here.
            roi_preprocs.append(cls(params))
    return global_preprocs, roi_preprocs


def _inject_runtime_preprocessing_into_feature_cfg(
    classification_cfg: Dict[str, Any],
    feature_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Attach runtime preprocessing config to texture-capable feature extractors.

    This keeps inference-time texture ROI extraction aligned with texture pretrain
    ROI generation, which already consumes ``runtime_preprocessing`` from
    classification config.
    """
    cfg = dict(feature_cfg or {})
    feature_name = str(cfg.get("name", "")).strip().lower()
    if feature_name not in _TEXTURE_BACKBONE_FEATURES:
        return cfg

    runtime_cfg = dict((classification_cfg or {}).get("runtime_preprocessing") or {})
    params = dict(cfg.get("params") or {})
    params["_runtime_texture_preprocessing"] = runtime_cfg
    cfg["params"] = params
    return cfg


def _build_texture_pretrain_auto_root(
    *,
    cache_dir: Path,
    dataset_root: str,
    train_videos: Dict[str, List[FramePrediction]],
    backbone: str,
    input_size: int,
    max_frames_per_video: int,
    val_ratio: float,
    roi_pad_ratio: float,
    seed: int,
    runtime_preprocessing: Optional[Dict[str, Any]] = None,
) -> Path:
    runtime_cfg = dict(runtime_preprocessing or {})
    step_defs = runtime_cfg.get("preproc_steps") or []
    normalized_steps: List[Dict[str, Any]] = []
    if isinstance(step_defs, Sequence) and not isinstance(step_defs, (str, bytes)):
        for step in step_defs:
            if not isinstance(step, dict):
                continue
            name = str(step.get("name", "")).strip()
            if not name:
                continue
            normalized_steps.append(
                {
                    "name": name,
                    "params": step.get("params", {}) or {},
                }
            )
    identity = {
        "schema": "texture_pretrain_auto_data_v2",
        "dataset_root": str(Path(dataset_root).resolve()),
        "train_videos": sorted(str(v) for v in train_videos.keys()),
        "backbone": str(backbone),
        "input_size": int(input_size),
        "max_frames_per_video": int(max_frames_per_video),
        "val_ratio": float(val_ratio),
        "roi_pad_ratio": float(roi_pad_ratio),
        "seed": int(seed),
        "runtime_preprocessing": {
            "scheme": str(runtime_cfg.get("scheme", "A")).strip().upper(),
            "preproc_steps": normalized_steps,
        },
    }
    packed = json.dumps(identity, ensure_ascii=False, sort_keys=True)
    key = hashlib.sha256(packed.encode("utf-8")).hexdigest()[:24]
    return cache_dir / "auto_data" / f"shared_{key}"


def _ensure_texture_pretrain_ckpt(
    *,
    classification_cfg: Dict[str, any],  # noqa: ANN001
    feature_cfg: Dict[str, any],  # noqa: ANN001
    dataset_root: str,
    train_videos: Dict[str, List[FramePrediction]],
    labels: Dict[str, int],
    results_dir: str,
    logger: Callable[[str], None],
) -> Dict[str, any]:  # noqa: ANN001
    feature_name = str((feature_cfg or {}).get("name", "")).strip().lower()
    params = dict((feature_cfg or {}).get("params") or {})

    if feature_name not in _TEXTURE_BACKBONE_FEATURES:
        return feature_cfg

    mode = str(params.get("texture_mode", "")).strip().lower()

    if mode != "pretrain":
        return feature_cfg

    ckpt = params.get("texture_pretrain_ckpt")
    if isinstance(ckpt, str) and ckpt.strip() and os.path.exists(ckpt):
        return feature_cfg

    logger(
        "[Classification] texture_mode=pretrain detected without valid checkpoint; "
        "auto-running Stage-1 texture pretraining (with cache)."
    )
    tp_cfg = dict((classification_cfg or {}).get("texture_pretrain") or {})
    embedding_dim = _infer_texture_embedding_dim(feature_name, params, tp_cfg)

    backbone = str(params.get("texture_backbone", tp_cfg.get("backbone", "convnext_tiny")))
    input_size = int(tp_cfg.get("input_size", params.get("texture_image_size", 96)))
    max_frames_per_video = int(tp_cfg.get("max_frames_per_video", 16))
    val_ratio = float(tp_cfg.get("val_ratio", 0.2))
    roi_pad_ratio = float(params.get("roi_pad_ratio", tp_cfg.get("roi_pad_ratio", 0.15)))

    tp_seed = int(tp_cfg.get("seed", 42))
    cache_dir = Path(tp_cfg.get("cache_dir", "ckpt/texture_pretrain_cache")).resolve()
    runtime_cfg = dict((classification_cfg or {}).get("runtime_preprocessing") or {})
    auto_root = _build_texture_pretrain_auto_root(
        cache_dir=cache_dir,
        dataset_root=dataset_root,
        train_videos=train_videos,
        backbone=backbone,
        input_size=input_size,
        max_frames_per_video=max_frames_per_video,
        val_ratio=val_ratio,
        roi_pad_ratio=roi_pad_ratio,
        seed=tp_seed,
        runtime_preprocessing=runtime_cfg,
    )
    roi_dir = auto_root / "roi_images"
    roi_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Tuple[str, int, str]] = []  # (subject, label, image_path)
    runtime_global_preprocs, runtime_roi_preprocs = _build_runtime_preprocs(classification_cfg)
    for video_path, preds in sorted(train_videos.items(), key=lambda kv: kv[0]):
        subject = _subject_from_video_path(video_path, dataset_root)
        if subject not in labels:
            continue
        label = int(labels[subject])
        if not preds:
            continue

        idx_pick = np.linspace(
            0,
            max(0, len(preds) - 1),
            num=min(max_frames_per_video, len(preds)),
            dtype=int,
        )
        idx_pick = sorted(set(int(v) for v in idx_pick.tolist()))

        try:
            import cv2  # noqa: PLC0415
            cap = cv2.VideoCapture(video_path)
        except Exception:
            cap = None

        if cap is None or not cap.isOpened():
            continue

        try:
            for i in idx_pick:
                p = preds[i]
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(p.frame_index))
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue

                frame_for_crop = frame
                if runtime_global_preprocs:
                    frame_for_crop = _apply_preprocs_frame_like_segmentation(frame_for_crop, runtime_global_preprocs)

                seg = getattr(p, "segmentation", None)
                seg_roi_bbox = getattr(seg, "roi_bbox", None) if seg is not None else None
                use_bbox = tuple(seg_roi_bbox) if seg_roi_bbox is not None else p.bbox
                pad = 0.0 if seg_roi_bbox is not None else roi_pad_ratio

                roi = _crop_roi_safe(frame_for_crop, use_bbox, pad_ratio=pad)
                if roi is None:
                    continue

                if runtime_roi_preprocs:
                    roi = _apply_preprocs_frame_like_segmentation(roi, runtime_roi_preprocs)

                fn = f"{subject}__{Path(video_path).stem}__f{int(p.frame_index):06d}.png"
                out_path = roi_dir / fn
                cv2.imwrite(str(out_path), roi)
                rows.append((subject, label, str(out_path.resolve())))
        finally:
            cap.release()

    if not rows:
        raise RuntimeError(
            "Auto texture pretraining failed: no ROI samples could be generated from training videos."
        )

    # subject-level split
    subjects = sorted(set(r[0] for r in rows))
    rng = np.random.default_rng(int(tp_cfg.get("seed", 42)))
    rng.shuffle(subjects)
    n_val = int(round(len(subjects) * val_ratio)) if len(subjects) > 1 else 0
    n_val = max(0, min(n_val, max(0, len(subjects) - 1)))
    val_subjects = set(subjects[:n_val])

    train_rows = [r for r in rows if r[0] not in val_subjects]
    val_rows = [r for r in rows if r[0] in val_subjects]

    manifest_dir = auto_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    train_manifest = manifest_dir / "train.csv"
    val_manifest = manifest_dir / "val.csv"

    with train_manifest.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "label"])
        for _, lbl, img in train_rows:
            w.writerow([img, lbl])

    with val_manifest.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "label"])
        for _, lbl, img in val_rows:
            w.writerow([img, lbl])

    pretrain_yaml = auto_root / "texture_pretrain_auto.yaml"
    pretrain_save = Path(tp_cfg.get("save_path", str(auto_root / "best.pth"))).resolve()
    payload = {
        "texture_pretrain": {
            "enable": True,
            "backbone": backbone,
            "num_classes": int(tp_cfg.get("num_classes", 2)),
            "embedding_dim": int(tp_cfg.get("embedding_dim", embedding_dim)),
            "input_size": input_size,
            "batch_size": int(tp_cfg.get("batch_size", 32)),
            "epochs": int(tp_cfg.get("epochs", 30)),
            "lr": float(tp_cfg.get("lr", 1e-4)),
            "weight_decay": float(tp_cfg.get("weight_decay", 1e-4)),
            "save_path": str(pretrain_save),
            "save_backbone_only": bool(tp_cfg.get("save_backbone_only", True)),
            "optimizer": str(tp_cfg.get("optimizer", "adamw")),
            "scheduler": str(tp_cfg.get("scheduler", "cosine")),
            "num_workers": int(tp_cfg.get("num_workers", 0)),
            "device": str(tp_cfg.get("device", "auto")),
            "amp": bool(tp_cfg.get("amp", True)),
            "head_type": str(tp_cfg.get("head_type", "linear")),
            "hidden_dim": int(tp_cfg.get("hidden_dim", 256)),
            "dropout": float(tp_cfg.get("dropout", 0.2)),
            "pretrained_imagenet": bool(tp_cfg.get("pretrained_imagenet", True)),
            "force_official_pretrained": bool(tp_cfg.get("force_official_pretrained", True)),
            "cache_enabled": bool(tp_cfg.get("cache_enabled", True)),
            "cache_dir": str(cache_dir),
            "train_manifest": str(train_manifest.resolve()),
            "val_manifest": str(val_manifest.resolve()) if val_rows else None,
        }
    }

    with pretrain_yaml.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    try:
        import yaml as _yaml  # noqa: PLC0415

        with pretrain_yaml.open("w", encoding="utf-8") as f:
            _yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
    except Exception:
        pass

    from ..texture_pretrain.train_texture import run_from_config as _run_texture_pretrain  # noqa: PLC0415

    summary = _run_texture_pretrain(str(pretrain_yaml))
    best_ckpt = str(summary.get("best_ckpt") or "").strip()
    if not best_ckpt or not os.path.exists(best_ckpt):
        raise RuntimeError("Auto texture pretraining finished but no valid best_ckpt was produced.")

    params["texture_mode"] = "pretrain"
    params["texture_pretrain_ckpt"] = best_ckpt
    _cached = bool(summary.get("cached", False))
    _epochs = summary.get("epochs")
    _device = summary.get("device")
    _best_metric = summary.get("best_metric")
    logger(
        "[Classification] Auto texture pretrain summary: "
        f"cached={_cached}, epochs={_epochs}, device={_device}, best_metric={_best_metric}"
    )
    logger(f"[Classification] Auto texture pretrain ready: ckpt={best_ckpt}")
    if summary.get("cache_key"):
        logger(f"[Classification] Auto texture pretrain cache_key={summary.get('cache_key')}")

    updated = dict(feature_cfg)
    updated["params"] = params
    return updated


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
                subject = _normalise_subject_token(parts[0])
                if subject:
                    return subject
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
        prefix = _normalise_subject_token(stem.split("_")[0])
        if prefix:
            return prefix
    return _normalise_subject_token(stem) or stem


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


def _exclude_loso_subjects(
    train_entities: Sequence[str],
    entity_to_subject: Dict[str, str],
    blocked_subjects: Sequence[str],
) -> Tuple[List[str], int, List[str]]:
    blocked = {
        _normalise_subject_token(str(token))
        for token in blocked_subjects
        if token is not None and _normalise_subject_token(str(token))
    }
    if not blocked:
        return list(train_entities), 0, []

    kept = [
        entity
        for entity in train_entities
        if _normalise_subject_token(entity_to_subject.get(entity)) not in blocked
    ]
    removed = int(len(train_entities) - len(kept))
    overlap = sorted(
        {
            _normalise_subject_token(entity_to_subject.get(entity))
            for entity in kept
            if _normalise_subject_token(entity_to_subject.get(entity)) in blocked
        }
    )
    return kept, removed, [str(v) for v in overlap if v is not None]


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
    feature_cfg = _inject_runtime_preprocessing_into_feature_cfg(config, feature_cfg)
    feature_name = feature_cfg.get("name", "basic")
    feature_cls = FEATURE_EXTRACTOR_REGISTRY.get(feature_name)
    if feature_cls is None:
        raise KeyError(f"Unknown feature extractor: {feature_name}")
    feature_extractor = feature_cls(feature_cfg.get("params"))

    classifier_cfg = config.get("classifier", {"name": "limix", "params": {}})
    classifier_name = classifier_cfg.get("name", "limix")
    classifier_cls = CLASSIFIER_REGISTRY.get(classifier_name)
    if classifier_cls is None:
        raise KeyError(f"Unknown classifier: {classifier_name}")

    fusion_cfg = config.get("fusion_module") if isinstance(config.get("fusion_module"), dict) else None
    fusion_name = str((fusion_cfg or {}).get("name", "")).strip().lower()
    fusion_learnable = is_learnable_fusion_module(fusion_name) if fusion_name else False
    differentiable_classifiers = {
        "mlp",
        "mlp_head",
        "mlp_linear_head",
        "linear_head",
        "patchtst",
        "timemachine",
        "transformer",
        "fusion_mlp",
        "fusion_gating_mlp",
        "v3pro_fusion",
    }

    clf_params = dict(classifier_cfg.get("params") or {})
    if fusion_cfg is not None:
        if fusion_learnable:
            classifier_name_lc = str(classifier_name).strip().lower()
            if classifier_name_lc not in differentiable_classifiers:
                raise RuntimeError(
                    "Learnable fusion module requires a differentiable classifier. "
                    f"Got fusion_module={fusion_name}, classifier={classifier_name}."
                )
            clf_params["fusion_module"] = fusion_cfg
            logger(
                "[Classification] Learnable fusion_module is enabled and passed to downstream differentiable classifier."
            )
        else:
            # Non-learnable fusion (e.g., concat) runs as an independent transform stage in engine.
            pass

    classifier = classifier_cls(clf_params if clf_params else None)

    def _detect_backend_used() -> Optional[str]:
        name_lc = str(classifier_name).strip().lower()
        if "tabpfn" in name_lc:
            try:
                if bool(getattr(classifier, "_fallback", False)):
                    fb_name = str(getattr(classifier, "_fallback_name", "")).strip() or "unknown"
                    return f"fallback_{fb_name}"
            except Exception:
                pass
            return "tabpfn_2_5"
        return None

    def _validate_tabpfn_backend(train_info_obj: Optional[Dict[str, Any]], *, loaded_from_cache: bool) -> Optional[str]:
        name_lc = str(classifier_name).strip().lower()
        if "tabpfn" not in name_lc:
            return None

        backend = None
        if isinstance(train_info_obj, dict):
            raw_backend = train_info_obj.get("backend")
            if isinstance(raw_backend, str) and raw_backend.strip():
                backend = raw_backend.strip()
        if backend is None:
            backend = _detect_backend_used()

        require_native_backend = bool((classifier_cfg.get("params") or {}).get("require_native_backend", True))
        if backend is not None and backend.lower().startswith("fallback_"):
            msg = (
                f"[Classification] TabPFN is running with fallback backend ({backend}) "
                f"({'cache load' if loaded_from_cache else 'training time'})."
            )
            if require_native_backend:
                raise RuntimeError(
                    msg
                    + " Please install/configure TabPFN correctly (including gated model access), "
                      "or set classifier.params.require_native_backend=false to allow fallback explicitly."
                )
            logger(msg + " Proceeding because require_native_backend=false.")
        return backend

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
            (config.get("trajectory_filter", {}) or {}).get("bbox_strategy", "none")
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

    # Auto bootstrap for texture pretrain mode:
    # if the selected feature extractor uses texture backbone in pretrain mode
    # and no checkpoint is provided, run Stage-1 automatically (with cache).
    try:
        feature_cfg = _ensure_texture_pretrain_ckpt(
            classification_cfg=config,
            feature_cfg=feature_cfg,
            dataset_root=dataset_root,
            train_videos=train_videos,
            labels=labels,
            results_dir=results_dir,
            logger=logger,
        )
        feature_cfg = _inject_runtime_preprocessing_into_feature_cfg(config, feature_cfg)
        feature_extractor = feature_cls(feature_cfg.get("params"))
    except Exception as _auto_pretrain_err:
        raise RuntimeError(
            "Failed to auto-prepare texture pretrain checkpoint for feature extractor "
            f"'{feature_name}': {_auto_pretrain_err}"
        ) from _auto_pretrain_err

    _loaded_fe_state_path = _try_load_feature_extractor_state(feature_extractor, config, logger)

    train_features, train_owner, train_sources = _prepare_entity_features(
        feature_extractor,
        train_videos,
        level,
        logger,
        dataset_root=dataset_root,
    )
    train_entities = _filter_entities(train_features, train_owner, labels, logger)

    runtime_ctx = dict((config or {}).get("runtime_context") or {})
    heldout_subject_raw = runtime_ctx.get("loso_subject")
    heldout_subject = _normalise_subject_token(str(heldout_subject_raw)) if heldout_subject_raw is not None else None

    loso_test_subjects_raw = runtime_ctx.get("loso_test_subjects")
    loso_test_subjects: set[str] = set()
    if isinstance(loso_test_subjects_raw, (list, tuple, set)):
        for token in loso_test_subjects_raw:
            norm = _normalise_subject_token(str(token)) if token is not None else None
            if norm:
                loso_test_subjects.add(norm)
    elif loso_test_subjects_raw is not None:
        norm = _normalise_subject_token(str(loso_test_subjects_raw))
        if norm:
            loso_test_subjects.add(norm)
    if heldout_subject:
        loso_test_subjects.add(heldout_subject)

    if loso_test_subjects:
        train_entities, removed_n, overlap = _exclude_loso_subjects(
            train_entities,
            train_owner,
            sorted(loso_test_subjects),
        )
        if removed_n > 0:
            logger(
                f"[Classification] LOSO guard removed {removed_n} training entit(ies) "
                f"for held-out subject(s)={sorted(loso_test_subjects)}."
            )
        if overlap:
            raise RuntimeError(
                "LOSO leakage detected: train entities still include test subject(s): "
                + ", ".join(str(v) for v in overlap)
            )

    if not train_entities:
        raise RuntimeError(
            "No training entities with labels available for classification stage."
        )

    vectoriser = _build_vectoriser(feature_extractor, level)
    X_train = vectoriser.transform(train_features[e] for e in train_entities)
    y_train = np.asarray([labels[train_owner[e]] for e in train_entities], dtype=np.int64)

    engine_fusion = None
    if fusion_cfg is not None and not fusion_learnable:
        engine_fusion = create_fusion_module(fusion_cfg)
        if engine_fusion is not None:
            X_train = engine_fusion.fit_transform(X_train, y_train)

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
            cached_backend = _validate_tabpfn_backend(None, loaded_from_cache=True)
            if cached_backend is not None:
                train_info = {
                    "backend": cached_backend,
                    "loaded_from_cache": True,
                }
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
            _validate_tabpfn_backend(train_info, loaded_from_cache=False)
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
        _validate_tabpfn_backend(train_info, loaded_from_cache=False)
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
        total_test_frames = sum(len(preds) for preds in test_predictions[source_model].values())
        missing_masks = sum(
            1 for _video, preds in test_predictions[source_model].items() for pred in preds if pred.segmentation is None
        )
        if missing_masks:
            ratio = (float(missing_masks) / float(max(total_test_frames, 1))) * 100.0
            logger(
                f"[Classification] Warning: {missing_masks}/{total_test_frames} test frame(s) ({ratio:.1f}%) are missing "
                "segmentation masks from the main pipeline. CTS/TS static features may be partial for those frames."
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

    if engine_fusion is not None:
        X_test = engine_fusion.transform(X_test)

    logger(f"[Classification] X_train shape: {X_train.shape}, X_test shape: {X_test.shape}, n_feat_keys: {len(list(vectoriser.keys))}")

    logger(f"[Classification] Evaluating classifier on {len(test_entities)} entities")
    y_pred_default = classifier.predict(X_test)
    prob_positive: List[float] = []
    try:
        prob_positive = _positive_class_probabilities(classifier, X_test).tolist()
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

    fi_feature_names = _resolve_feature_names(list(vectoriser.keys), X_test.shape[1])
    feature_importance_payload = _collect_feature_importance_payload(
        classifier=classifier,
        train_info=train_info,
        X_eval=X_test,
        y_eval=y_test,
        feature_names=fi_feature_names,
        logger=logger,
    )
    fi_file_basename: Optional[str] = None
    if feature_importance_payload is not None:
        fi_file_basename = "feature_importance.json"
        fi_path = os.path.join(model_dir, fi_file_basename)
        with open(fi_path, "w", encoding="utf-8") as f:
            json.dump(feature_importance_payload, f, ensure_ascii=False, indent=2)
        logger(f"[Classification] Saved feature importance: {fi_path}")

    fe_state_path = _save_feature_extractor_state(feature_extractor, model_dir, logger)

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
        "feature_importance_file": fi_file_basename,
        "feature_extractor_state_file": os.path.basename(fe_state_path) if fe_state_path else None,
        "feature_extractor_state_source": _loaded_fe_state_path,
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

