from __future__ import annotations
import os
import json
import time
import hashlib
from datetime import datetime
from contextlib import contextmanager
from typing import Dict, Any, List, Callable, Optional, Sequence
import random
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

from ..core.registry import PREPROC_REGISTRY, MODEL_REGISTRY, EVAL_REGISTRY
from ..utils.annotations import load_coco_vid
from ..core.interfaces import FramePrediction
from ..data.dataset_manager import COCOJsonDatasetManager, SimpleDataset
# classification modules
from ..classification.engine import run_subject_classification
from ..segmentation import SegmentationWorkflow
# import built-in plugins to populate registries
from ..preproc import clahe  # noqa: F401
from ..preproc import augment  # noqa: F401
from ..models import template_matching  # noqa: F401
from ..models import csrt  # noqa: F401
from ..models import optical_flow_lk  # noqa: F401
from ..models import faster_rcnn  # noqa: F401
from ..models import yolov11  # noqa: F401
from ..models import fast_speckle  # noqa: F401
from ..models import ocsort  # noqa: F401
from ..models import strongsort  # noqa: F401
from ..models import tomp  # noqa: F401
from ..models import tamos  # noqa: F401
from ..models import mixformerv2  # noqa: F401
from ..eval import evaluator  # noqa: F401
from ..utils.env import capture_env
from ..utils.seed import set_seed
from .pipeline_validator import enforce_or_collect_warnings
import traceback as _tb


def _load_annotations_for_videos(video_paths: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    annotations: Dict[str, Dict[str, Any]] = {}
    for video_path in video_paths:
        json_path = os.path.splitext(video_path)[0] + ".json"
        if not os.path.exists(json_path):
            continue
        try:
            annotations[video_path] = load_coco_vid(json_path)
        except Exception:
            continue
    return annotations


class PipelineRunner:
    def __init__(
        self,
        config: Dict[str, Any],
        logger: Optional[Callable[[str], None]] = None,
        progress_cb: Optional[Callable[[str,int,int,Dict[str,Any]], None]] = None,
        detector_reuse_cache: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    ):
        self.cfg = config
        self.seed = int(config.get("seed", 0))
        dataset_cfg = config.get("dataset", {})
        self.dataset_root = dataset_cfg.get("root", ".")
        out_cfg = config.get("output", {}) or {}
        # If output missing or lacks results_root, default to project folder /results
        try:
            proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        except Exception:
            proj_root = os.getcwd()
        self._proj_root = proj_root
        res_root_cfg = out_cfg.get("results_root")
        if res_root_cfg:
            # If user provided a relative path, anchor it to project root to avoid cwd-dependent outputs.
            self.results_root = res_root_cfg if os.path.isabs(res_root_cfg) else os.path.join(proj_root, res_root_cfg)
        else:
            self.results_root = os.path.join(proj_root, "results")
        os.makedirs(self.results_root, exist_ok=True)
        # Persistent detector cache: lives at {proj_root}/results/detector_cache.json
        # This survives across multiple queue runs so already-trained detectors are reused.
        _global_cache_dir = os.path.join(proj_root, "results")
        os.makedirs(_global_cache_dir, exist_ok=True)
        self._persistent_det_cache_file: str = os.path.join(_global_cache_dir, "detector_cache.json")
        self._persistent_seg_cache_file: str = os.path.join(_global_cache_dir, "segmentation_cache.json")
        self._persistent_clf_cache_file: str = os.path.join(_global_cache_dir, "classification_cache.json")
        self._logger = logger
        self._log_file: Optional[str] = None
        self._progress_cb = progress_cb
        self._detector_reuse_cache = detector_reuse_cache if detector_reuse_cache is not None else {}

    # lightweight wrapper to emit progress events
    def _progress(self, stage: str, current: int, total: int, extra: Optional[Dict[str,Any]] = None):
        if self._progress_cb:
            try:
                self._progress_cb(stage, int(current), int(total), extra or {})
            except Exception:
                pass

    def _timestamp_dir(self, name: str) -> str:
        ts = time.strftime("%Y-%m-%d_%H-%M-%S")
        d = os.path.join(self.results_root, f"{ts}_{name}")
        os.makedirs(d, exist_ok=True)
        return d

    # ------------------------------------------------------------------
    # Persistent detector cache helpers
    # ------------------------------------------------------------------
    def _load_persistent_detector_cache(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Load the project-level persistent detector cache from disk.

        Only entries whose checkpoint file still exists on disk are returned.
        Returns an empty dict on any error.
        """
        try:
            if not os.path.exists(self._persistent_det_cache_file):
                return {}
            with open(self._persistent_det_cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            detectors = data.get("detectors", {})
            if not isinstance(detectors, dict):
                return {}
            valid: Dict[str, Dict[str, Dict[str, Any]]] = {}
            for sig_key, model_map in detectors.items():
                if not isinstance(model_map, dict):
                    continue
                valid_models: Dict[str, Dict[str, Any]] = {}
                for model_name, entry in model_map.items():
                    if not isinstance(entry, dict):
                        continue
                    ckpt = entry.get("checkpoint")
                    if ckpt and isinstance(ckpt, str) and os.path.exists(ckpt):
                        valid_models[model_name] = entry
                if valid_models:
                    valid[sig_key] = valid_models
            return valid
        except Exception:
            return {}

    def _save_entry_to_persistent_cache(
        self,
        sig_key: str,
        model_name: str,
        entry: Dict[str, Any],
    ) -> None:
        """Append / update one detector entry in the project-level persistent cache.

        Thread-safety note: this is intentionally best-effort (single-process use).
        Errors are silently swallowed so the training pipeline is never interrupted.
        """
        try:
            existing_data: Dict[str, Any] = {}
            if os.path.exists(self._persistent_det_cache_file):
                try:
                    with open(self._persistent_det_cache_file, "r", encoding="utf-8") as f:
                        existing_data = json.load(f)
                except Exception:
                    existing_data = {}
            detectors = existing_data.get("detectors", {})
            if not isinstance(detectors, dict):
                detectors = {}
            detectors.setdefault(sig_key, {})[model_name] = entry
            existing_data["detectors"] = detectors
            existing_data["version"] = 1
            existing_data["updated_at"] = datetime.utcnow().isoformat() + "Z"
            with open(self._persistent_det_cache_file, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass  # Best-effort; do not break the main pipeline

    # ------------------------------------------------------------------
    # Generic stage cache helpers (segmentation / classification)
    # Schema: {"version": 1, "entries": {<sig_key>: {<model_key>: {"checkpoint": <path>}}}}
    # ------------------------------------------------------------------
    def _load_persistent_stage_cache(self, cache_file: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Load a generic persistent cache keyed by SHA-256 signature.

        Only entries whose "checkpoint" file still exists on disk are returned.
        Returns an empty dict on any error.
        """
        try:
            if not os.path.exists(cache_file):
                return {}
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            entries = data.get("entries", {})
            if not isinstance(entries, dict):
                return {}
            valid: Dict[str, Dict[str, Dict[str, Any]]] = {}
            for sig_key, model_map in entries.items():
                if not isinstance(model_map, dict):
                    continue
                valid_models: Dict[str, Dict[str, Any]] = {}
                for model_key, entry in model_map.items():
                    if not isinstance(entry, dict):
                        continue
                    ckpt = entry.get("checkpoint")
                    if ckpt and isinstance(ckpt, str) and os.path.exists(ckpt):
                        valid_models[model_key] = entry
                if valid_models:
                    valid[sig_key] = valid_models
            return valid
        except Exception:
            return {}

    def _save_to_stage_cache(
        self,
        cache_file: str,
        sig_key: str,
        model_key: str,
        entry: Dict[str, Any],
    ) -> None:
        """Append / update one entry in a generic persistent stage cache file."""
        try:
            existing_data: Dict[str, Any] = {}
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        existing_data = json.load(f)
                except Exception:
                    existing_data = {}
            entries = existing_data.get("entries", {})
            if not isinstance(entries, dict):
                entries = {}
            entries.setdefault(sig_key, {})[model_key] = entry
            existing_data["entries"] = entries
            existing_data["version"] = 1
            existing_data["updated_at"] = datetime.utcnow().isoformat() + "Z"
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass  # Best-effort; do not break the main pipeline

    def _log(self, msg: str, to_console: bool = True):
        # keep file logging always; console logging can be suppressed when tqdm bars are active
        if self._log_file:
            try:
                with open(self._log_file, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
            except Exception:
                pass
        if to_console and self._logger:
            try:
                self._logger(msg)
            except Exception:
                pass

    def run(self):
        # Validate pipeline design compatibility before any training starts.
        # This catches learnable feature blocks paired with non-differentiable classifiers.
        _warnings = enforce_or_collect_warnings(self.cfg)
        for _msg in _warnings:
            self._log(_msg)
            if self._logger is None:
                try:
                    print(_msg)
                except Exception:
                    pass

        # Only need reproducible dataset splits, not full deterministic ops (which can break CuBLAS).
        set_seed(self.seed, deterministic=False)
        dm = COCOJsonDatasetManager(self.dataset_root)
        total_videos = len(dm.videos)
        annotated_videos = len(dm.ann_by_video)
        missing_ann = list(getattr(dm, "missing_annotations", []) or [])
        self._log(
            f"[Dataset] Videos discovered: {total_videos} | annotated: {annotated_videos} | missing_annotations: {len(missing_ann)}"
        )
        if missing_ann:
            preview_items = []
            base_root = self.dataset_root if isinstance(self.dataset_root, str) and self.dataset_root else None
            for path in missing_ann[:5]:
                try:
                    rel = os.path.relpath(path, base_root) if base_root else path
                except Exception:
                    rel = path
                preview_items.append(rel)
            preview = ", ".join(preview_items)
            more = " …" if len(missing_ann) > 5 else ""
            self._log(f"[Dataset] Missing annotation JSON for: {preview}{more}")
        if annotated_videos == 0:
            self._log("[Dataset] Warning: no annotated videos found; detector training will be skipped.")
        ds_cfg = self.cfg.get("dataset", {})
        split_cfg = (ds_cfg or {}).get("split", {})
        method = split_cfg.get("method", "video_level")
        ratios = split_cfg.get("ratios", [0.8, 0.2])
        k_fold = int(split_cfg.get("k_fold", 1) or 1)
        loso_enabled = str(method).lower() == "loso" or bool(split_cfg.get("loso", False))
        self._log(f"Dataset root: {self.dataset_root}")
        self._log(f"Results root: {self.results_root}")
        def _rel_path(path: str) -> str:
            try:
                base_root = self.dataset_root if isinstance(self.dataset_root, str) and self.dataset_root else None
                return os.path.relpath(path, base_root) if base_root else path
            except Exception:
                return path

        def _dataset_preview(dataset, limit: int = 5) -> List[str]:  # type: ignore[override]
            names: List[str] = []
            try:
                total = len(dataset)  # type: ignore[arg-type]
            except Exception:
                total = 0
            for idx in range(min(total, int(limit))):
                try:
                    item = dataset[idx]
                    if isinstance(item, dict):
                        vp = item.get("video_path")
                    else:
                        vp = getattr(item, "video_path", None)
                    if vp:
                        names.append(_rel_path(vp))
                except Exception:
                    continue
            return names

        loso_folds: List[Dict[str, Any]] = []
        if loso_enabled:
            # Optional LOSO limiting knobs (useful for smoke tests / debugging)
            # - subjects: only run a subset of subject IDs
            # - max_folds: stop after N folds
            # - max_train_videos / max_test_videos: truncate per-fold video lists
            subj_filter_raw = split_cfg.get("subjects") or split_cfg.get("subject") or split_cfg.get("subject_ids")
            subjects_filter = None
            if isinstance(subj_filter_raw, str) and subj_filter_raw.strip():
                subjects_filter = {subj_filter_raw.strip()}
            elif isinstance(subj_filter_raw, (list, tuple)):
                subjects_filter = {str(s).strip() for s in subj_filter_raw if str(s).strip()}
            try:
                max_folds = int(split_cfg.get("max_folds") or 0)
            except Exception:
                max_folds = 0
            try:
                max_train_videos = int(split_cfg.get("max_train_videos") or 0)
            except Exception:
                max_train_videos = 0
            try:
                max_test_videos = int(split_cfg.get("max_test_videos") or 0)
            except Exception:
                max_test_videos = 0

            for fold in dm.loso():
                subject_id = fold.get("subject")
                if subjects_filter is not None and str(subject_id) not in subjects_filter:
                    continue
                train_videos = list(fold.get("train", []) or [])
                test_videos = list(fold.get("test", []) or [])
                if max_train_videos > 0:
                    train_videos = train_videos[:max_train_videos]
                if max_test_videos > 0:
                    test_videos = test_videos[:max_test_videos]
                train_ds = SimpleDataset(train_videos, dm.ann_by_video)
                test_ds = SimpleDataset(test_videos, dm.ann_by_video)
                loso_folds.append({"subject": subject_id, "train": train_ds, "test": test_ds})
                if max_folds > 0 and len(loso_folds) >= max_folds:
                    break
            self._log(f"LOSO folds prepared: {len(loso_folds)} subjects")
        else:
            # accept [train,test] or [train,val,test]; always map to (train, 0.0, test)
            if isinstance(ratios, list) and len(ratios) == 2:
                train_r, test_r = float(ratios[0]), float(ratios[1])
            elif isinstance(ratios, list) and len(ratios) == 3:
                train_r, test_r = float(ratios[0]), float(ratios[2])
            else:
                train_r, test_r = 0.8, 0.2
            split_tt = dm.split(method=method, seed=self.seed, ratios=(train_r, 0.0, test_r))
            train_ds = split_tt["train"]
            test_ds = split_tt["test"]
            loso_folds.append({"subject": None, "train": train_ds, "test": test_ds})

        def _dataset_info(train_ds: Any, test_ds: Any, subject: Optional[str], fold_idx: int, total_folds: int) -> Dict[str, Any]:
            train_count = len(train_ds)
            test_count = len(test_ds)
            info = {
                "root": self.dataset_root,
                "total_videos": total_videos,
                "annotated_videos": annotated_videos,
                "missing_annotations": len(missing_ann),
                "missing_preview": [_rel_path(p) for p in missing_ann[:5]],
                "split": {
                    "method": "loso" if loso_enabled else method,
                    "ratios": ratios,
                    "k_fold": k_fold,
                    "fold": fold_idx + 1,
                    "total_folds": total_folds,
                    "subject": subject,
                },
                "train_videos": train_count,
                "test_videos": test_count,
                "train_preview": _dataset_preview(train_ds),
                "test_preview": _dataset_preview(test_ds),
            }
            return info

        if loso_enabled and not loso_folds:
            self._log("[Dataset] No LOSO folds could be built (no subjects).")
            return
        if not loso_enabled:
            self._log(
                f"Dataset split created (train/test). Train videos: {len(loso_folds[0]['train'])} | Test videos: {len(loso_folds[0]['test'])}"
            )
            if len(loso_folds[0]["train"]) == 0:
                self._log("[Train] No annotated training videos detected. Training steps will be skipped.")

        # orchestrate experiments (per fold if LOSO)
        folds_total = len(loso_folds)
        detector_reuse_cache: Dict[str, Dict[str, Dict[str, Any]]] = self._detector_reuse_cache

        # --- Persistent detector cache: load from disk and merge into in-memory cache ---
        # This allows cross-run reuse: detectors trained in *previous* runs (including
        # separate program sessions and individual "Run" button clicks) are reused
        # automatically without retraining.
        _disk_cache = self._load_persistent_detector_cache()
        if _disk_cache:
            self._log(
                f"[DetectorCache] Loaded {len(_disk_cache)} signature(s) from persistent cache: {self._persistent_det_cache_file}",
                to_console=True,
            )
            for _sig_key, _model_map in _disk_cache.items():
                if _sig_key not in detector_reuse_cache:
                    detector_reuse_cache[_sig_key] = {}
                for _model_name, _entry in _model_map.items():
                    # In-memory cache takes priority (e.g. just-trained within same run)
                    if _model_name not in detector_reuse_cache[_sig_key]:
                        detector_reuse_cache[_sig_key][_model_name] = _entry

        # --- Load persistent caches for segmentation and classification ---
        _seg_disk_cache = self._load_persistent_stage_cache(self._persistent_seg_cache_file)
        if _seg_disk_cache:
            self._log(
                f"[SegCache] Loaded {len(_seg_disk_cache)} signature(s) from persistent cache: {self._persistent_seg_cache_file}",
                to_console=True,
            )
        _clf_disk_cache = self._load_persistent_stage_cache(self._persistent_clf_cache_file)
        if _clf_disk_cache:
            self._log(
                f"[ClfCache] Loaded {len(_clf_disk_cache)} signature(s) from persistent cache: {self._persistent_clf_cache_file}",
                to_console=True,
            )

        def _detector_signature(
            exp_cfg: Dict[str, Any],
            scheme_key: str,
            train_dataset: Any,
            subject_id: Optional[str],
            fold_index: int,
        ) -> Dict[str, Any]:
            detector_preprocs: List[Dict[str, Any]] = []
            for step in exp_cfg.get("pipeline", []) or []:
                if step.get("type") != "preproc":
                    continue
                if scheme_key in {"A", "C"}:
                    detector_preprocs.append(
                        {"name": step.get("name"), "params": step.get("params", {})}
                    )

            detector_models: List[Dict[str, Any]] = []
            for step in exp_cfg.get("pipeline", []) or []:
                if step.get("type") != "model":
                    continue
                detector_models.append(
                    {"name": step.get("name"), "params": step.get("params", {})}
                )

            train_videos: List[str] = []
            try:
                train_videos = sorted({train_dataset[i]["video_path"] for i in range(len(train_dataset))})
            except Exception:
                train_videos = []

            signature_payload = {
                "detector_models": detector_models,
                "detector_preprocs": detector_preprocs,
                "preproc_scheme": scheme_key,
                "dataset_root": self.dataset_root,
                "train_videos": train_videos,
                "subject": subject_id,
                "fold": fold_index + 1,
            }
            sig_raw = json.dumps(signature_payload, sort_keys=True, ensure_ascii=False)
            sig_key = hashlib.sha256(sig_raw.encode("utf-8")).hexdigest()
            return {"key": sig_key, "payload": signature_payload}

        def _segmentation_signature(
            seg_cfg: Dict[str, Any],
            model_name: str,
            global_preprocs: List[Any],
            roi_preprocs: List[Any],
            train_paths: List[str],
            seg_seed: Any,
            preproc_scheme: str,
            subject_id: Optional[str],
            fold_index: int,
        ) -> Dict[str, Any]:
            def _preproc_desc(p: Any) -> str:
                return type(p).__name__
            payload = {
                "seg_config": seg_cfg,
                "model_name": model_name,
                "preproc_scheme": preproc_scheme,
                "global_preprocs": [_preproc_desc(p) for p in (global_preprocs or [])],
                "roi_preprocs": [_preproc_desc(p) for p in (roi_preprocs or [])],
                "dataset_root": self.dataset_root,
                "train_videos": sorted(train_paths),
                "seed": int(seg_seed) if seg_seed is not None else 0,
                "subject": subject_id,
                "fold": fold_index + 1,
            }
            sig_raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
            return {"key": hashlib.sha256(sig_raw.encode("utf-8")).hexdigest(), "payload": payload}

        def _classification_signature(
            clf_cfg: Dict[str, Any],
            train_paths: List[str],
            seg_model_name_hint: str,
            subject_id: Optional[str],
            fold_index: int,
        ) -> Dict[str, Any]:
            payload = {
                "feature_extractor": clf_cfg.get("feature_extractor", {}),
                "classifier": clf_cfg.get("classifier", {}),
                "label_file": clf_cfg.get("label_file"),
                "seg_enabled": bool((clf_cfg.get("segmentation") or {}).get("enabled", True)),
                "seg_model_name": seg_model_name_hint,
                "dataset_root": self.dataset_root,
                "train_videos": sorted(train_paths),
                "subject": subject_id,
                "fold": fold_index + 1,
            }
            sig_raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
            return {"key": hashlib.sha256(sig_raw.encode("utf-8")).hexdigest(), "payload": payload}

        for fold_idx, fold_data in enumerate(loso_folds):
            train_ds = fold_data["train"]
            test_ds = fold_data["test"]
            subject = fold_data.get("subject")
            dataset_info = _dataset_info(train_ds, test_ds, subject, fold_idx, folds_total)

            for exp in self.cfg.get("experiments", []):
                exp_name = exp.get("name", "exp")
                suffix = ""
                if loso_enabled:
                    subj_tag = subject or f"fold{fold_idx+1}"
                    suffix = f"_loso_{subj_tag}"
                out_dir = self._timestamp_dir(exp_name + suffix)
                experiment_start = time.perf_counter()
                out_cfg = self.cfg.get("output", {}) or {}
                skip_freeze = bool(out_cfg.get("skip_pip_freeze", False))
                pipeline_summary: List[Dict[str, Any]] = []
                for step in exp.get("pipeline", []):
                    if not isinstance(step, dict):
                        continue
                    pipeline_summary.append(
                        {
                            "type": step.get("type"),
                            "name": step.get("name"),
                            "params": step.get("params", {}),
                        }
                    )
                stage_metrics_collector: Dict[str, Any] = {}
                stage_records: List[Dict[str, Any]] = []
                try:
                    rel_out = os.path.relpath(out_dir, self.results_root)
                except Exception:
                    rel_out = None
                meta_path = os.path.join(out_dir, "metadata.json")
                meta: Dict[str, Any] = {
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "seed": self.seed,
                    "dataset": dataset_info,
                    "config": exp,
                    "env": capture_env(skip_freeze=skip_freeze),
                    "experiment": {
                        "name": exp_name,
                        "output_dir": out_dir,
                        "results_root": self.results_root,
                        "relative_output": rel_out,
                        "pipeline": pipeline_summary,
                        "k_fold": k_fold,
                        "split_method": "loso" if loso_enabled else method,
                        "ratios": ratios,
                        "fold": fold_idx + 1 if loso_enabled else None,
                        "total_folds": folds_total if loso_enabled else None,
                        "subject": subject,
                    },
                    "metrics": {},
                    "artifacts": {},
                    "stages": stage_records,
                    "runtime": {
                        "started_at": datetime.utcnow().isoformat() + "Z",
                    },
                }

                def _dump_meta() -> None:
                    meta["runtime"]["updated_at"] = datetime.utcnow().isoformat() + "Z"
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(meta, f, ensure_ascii=False, indent=2)

                def _record_stage_skip(name: str, kind: str, reason: str, extra: Optional[Dict[str, Any]] = None) -> None:
                    entry: Dict[str, Any] = {
                        "name": name,
                        "kind": kind,
                        "status": "skipped",
                        "reason": reason,
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                    }
                    if extra:
                        entry.update(extra)
                    stage_records.append(entry)
                    _dump_meta()

                @contextmanager
                def stage_scope(name: str, kind: str, info: Optional[Dict[str, Any]] = None):
                    entry: Dict[str, Any] = {
                        "name": name,
                        "kind": kind,
                        "started_at": datetime.utcnow().isoformat() + "Z",
                    }
                    if info:
                        entry.update(info)
                    stage_records.append(entry)
                    _dump_meta()
                    stage_start = time.perf_counter()
                    try:
                        yield entry
                        entry.setdefault("status", "completed")
                    except Exception as exc:
                        entry["status"] = "failed"
                        entry["error"] = {"type": type(exc).__name__, "message": str(exc)}
                        _dump_meta()
                        raise
                    finally:
                        entry["finished_at"] = datetime.utcnow().isoformat() + "Z"
                        entry["duration_sec"] = max(0.0, time.perf_counter() - stage_start)
                        _dump_meta()

                _dump_meta()
                # init log file for this experiment
                logs_dir = os.path.join(out_dir, "logs")
                os.makedirs(logs_dir, exist_ok=True)
                self._log_file = os.path.join(logs_dir, "run.log")
                self._log(f"Started experiment: {exp_name}")
                self._log(f"Experiment folder: {out_dir}")
                meta.setdefault("artifacts", {})["run_log"] = self._log_file
                _dump_meta()

            # If no experiments were defined, skip this fold (e.g., defs-only queue entries)
            if not self.cfg.get("experiments"):
                continue

            test_root = os.path.join(out_dir, "test")
            detection_root = os.path.join(test_root, "detection")
            segmentation_root = os.path.join(test_root, "segmentation")
            train_root = os.path.join(out_dir, "train_full")
            os.makedirs(detection_root, exist_ok=True)
            os.makedirs(segmentation_root, exist_ok=True)
            os.makedirs(train_root, exist_ok=True)
            os.makedirs(os.path.join(train_root, "detection"), exist_ok=True)
            os.makedirs(os.path.join(train_root, "segmentation"), exist_ok=True)

            # build preproc chain
            # - global: applied to full frames before any downstream stage (affects detector + segmentation)
            # - roi: applied only after ROI crop (affects segmentation only)
            preprocs = []  # detector/global preprocs (kept name for backward-compat)
            preprocs_roi = []  # segmentation ROI-only preprocs

            # Simplified scheme selector (preferred UI control): A / B / C
            # IMPORTANT: this is a "作用域" choice that applies to *all* preprocessing steps, not just CLAHE.
            # - A (Global): preprocs affect detector and segmentation (before crop)
            # - B (ROI): detector sees raw; segmentation applies preprocs only after ROI crop
            # - C (Hybrid): detector uses global-preproc frames for better bbox; segmentation crops from RAW then applies ROI preprocs
            scheme_raw = exp.get("preproc_scheme") or exp.get("preprocessing_scheme") or exp.get("preproc_mode")
            scheme = str(scheme_raw).strip().upper() if scheme_raw is not None else "A"
            if scheme in {"GLOBAL", "A"}:
                scheme = "A"
            elif scheme in {"ROI", "B"}:
                scheme = "B"
            elif scheme in {"HYBRID", "C"}:
                scheme = "C"
            else:
                raise ValueError(f"Invalid preproc_scheme: {scheme_raw!r}. Expected one of: A/B/C (or GLOBAL/ROI/HYBRID).")

            for step in exp.get("pipeline", []):
                if step.get("type") == "preproc":
                    cls = PREPROC_REGISTRY[step["name"]]
                    params = step.get("params", {})

                    # Only scheme-based routing is supported (no per-step `scope`).
                    if scheme == "A":
                        preprocs.append(cls(params))
                    elif scheme == "B":
                        preprocs_roi.append(cls(params))
                    elif scheme == "C":
                        preprocs.append(cls(params))
                        preprocs_roi.append(cls(params))

            self._log(f"Preprocs(global): {[type(p).__name__ for p in preprocs]}")
            if preprocs_roi:
                self._log(f"Preprocs(roi): {[type(p).__name__ for p in preprocs_roi]}")
            self._log(f"Preproc scheme: {scheme}")

            # factory for model(s) to ensure fresh instance per fold/final
            model_steps = [step for step in exp.get("pipeline", []) if step.get("type") == "model"]
            def build_models():
                ms = []
                for step in model_steps:
                    cls = MODEL_REGISTRY[step["name"]]
                    try:
                        m = cls(step.get("params", {}))
                        if hasattr(m, "preprocs"):
                            setattr(m, "preprocs", preprocs)
                        # --- 新增：記錄模型實際使用的參數快照，方便偵錯 UI 傳參是否正確 ---
                        try:
                            # 可能的參數名稱集合（通用 + FasterRCNN 常見）
                            cand_keys = {
                                'device','pretrained','num_classes','epochs','batch','batch_size','lr','lr0','weight_decay','momentum',
                                'optimizer_name','optimizer','step_size','gamma','grad_clip','detect_anomaly','include_empty_frames',
                                'fallback_last_prediction','score_thresh','adamw_betas','adamw_eps','inference_batch','num_workers','workers','pin_memory','patience'
                            }
                            snap = {}
                            for k in sorted(cand_keys):
                                if hasattr(m, k):
                                    v = getattr(m, k)
                                    # 避免張量 / 模型本體被序列化
                                    if not hasattr(v, 'parameters') and not isinstance(v, (type(m.model),)):
                                        snap[k] = v
                            self._log(f"[ModelConfig] model={step['name']} params={snap}")
                        except Exception:
                            pass
                        ms.append((step["name"], m))
                    except Exception as _e_inst:
                        self._log(f"[ERROR] Model init failed | model={step['name']} error={_e_inst}\n{_tb.format_exc()}")
                        # fallback placeholder to keep pipeline running and produce result files
                        class _UnavailableModel:
                            def __init__(self, reason: str):
                                self.name = step["name"]
                                self.reason = reason
                                self.preprocs = []
                            def train(self, *a, **k):
                                return {"status": "unavailable", "reason": self.reason}
                            def predict(self, *a, **k):
                                return []
                            def load_checkpoint(self, *a, **k):
                                pass
                        ms.append((step["name"], _UnavailableModel(str(_e_inst))))
                return ms
            models = build_models()
            # attach model-level progress callbacks (per-epoch)
            for name, m in models:
                try:
                    setattr(m, 'progress_callback', lambda stage, cur, tot, model=name: self._progress(stage, cur, tot, {'model': model}))
                except Exception:
                    pass
            # 顯示實例名稱（對外顯示）
            try:
                display_names = []
                for _reg_name, _m in models:
                    disp = getattr(_m, 'name', _reg_name)
                    display_names.append(disp)
                self._log(f"Models: {display_names}")
            except Exception:
                self._log(f"Models: {[name for name,_ in models]}")

            # evaluator
            eval_name = self.cfg.get("evaluation", {}).get("evaluator", "BasicEvaluator")
            eval_cls = EVAL_REGISTRY.get(eval_name)
            evaluator = eval_cls() if eval_cls else None

            def run_on_dataset(dataset, out_dir_base: str, models_list=None, phase: str = 'eval'):
                nonlocal stage_metrics_collector
                # initialize containers and ensure per-model predictions files are always created
                predictions_all = {}
                # aggregate across entire dataset (all videos) per model
                dataset_agg = {}
                per_model_video_predictions = {}
                met_dir = os.path.join(out_dir_base, "metrics")
                os.makedirs(met_dir, exist_ok=True)
                eval_cfg = self.cfg.get("evaluation", {}) or {}
                restrict_to_gt_frames = bool(eval_cfg.get("restrict_to_gt_frames", True))
                viz_cfg = eval_cfg.get("visualize", {}) or {}
                viz_enabled = bool(viz_cfg.get("enabled", True))
                viz_samples = max(1, int(viz_cfg.get("samples", 10) or 10))
                viz_include_detection = bool(viz_cfg.get("include_detection", True))

                def _select_evenly_spaced(indices: List[int], limit: int) -> List[int]:
                    if not indices or limit <= 0:
                        return []
                    if len(indices) <= limit:
                        result = list(indices)
                    elif limit == 1:
                        result = [indices[0]]
                    else:
                        step = (len(indices) - 1) / float(limit - 1)
                        result = []
                        for i in range(limit):
                            pos = int(round(i * step))
                            if pos >= len(indices):
                                pos = len(indices) - 1
                            result.append(indices[pos])
                        result[0] = indices[0]
                        result[-1] = indices[-1]
                    dedup: List[int] = []
                    seen = set()
                    for value in result:
                        if value not in seen:
                            dedup.append(int(value))
                            seen.add(int(value))
                    return dedup
                # Determine models used and initialize prediction collectors
                use_models = models_list or models
                for model_name, _m in use_models:
                    display_name = getattr(_m, 'name', model_name)
                    predictions_all.setdefault(display_name, [])
                    per_model_video_predictions.setdefault(display_name, {})
                # iterate videos with optional progress bar
                iterable = dataset
                total_videos = None
                try:
                    total_videos = len(dataset)  # type: ignore
                except Exception:
                    total_videos = None
                if tqdm is not None:
                    iterable = tqdm(dataset, total=total_videos, desc="Videos", unit="vid")
                idx_video = 0
                for video_item in iterable:
                    idx_video += 1
                    if total_videos:
                        self._progress('eval_video', idx_video, total_videos, {'phase': phase, 'exp': exp_name})
                    vp = video_item["video_path"]
                    gt_json = os.path.splitext(vp)[0] + ".json"
                    gt = load_coco_vid(gt_json) if os.path.exists(gt_json) else {"frames": {}}
                    # reduce noisy console logs during progress; keep in file only when tqdm is present
                    self._log(f"Predicting video: {os.path.basename(vp)}", to_console=(tqdm is None))
                    pv_predictions = {}
                    per_video_infer_stats: Dict[str, Dict[str, float]] = {}
                    # per-video per-model bar
                    model_iter = use_models
                    if tqdm is not None and len(use_models) > 1:
                        model_iter = tqdm(use_models, desc=f"Models@{os.path.basename(vp)}", unit="mdl", leave=False)
                    for model_name, model in model_iter:
                        frames_targeted = None
                        start_time = time.perf_counter()
                        try:
                            # If we only evaluate GT frames and the model supports sparse inference,
                            # request predictions only on those frames (useful for detection-based ML models)
                            if restrict_to_gt_frames and hasattr(model, 'predict_frames'):
                                gt_frames_sorted = sorted([int(fi) for fi, boxes in gt.get("frames", {}).items() if boxes])
                                frames_targeted = gt_frames_sorted
                                preds = model.predict_frames(vp, gt_frames_sorted)  # type: ignore[attr-defined]
                            else:
                                preds = model.predict(vp)
                        except Exception as e:
                            # Log the failure and continue to next model/video
                            self._log(f"[ERROR] Predict failed | model={getattr(model,'name',model_name)} video={os.path.basename(vp)} error={e}")
                            preds = []
                        elapsed = time.perf_counter() - start_time
                        disp = getattr(model, 'name', model_name)
                        pv_predictions[disp] = preds
                        predictions_all.setdefault(disp, []).extend(preds)
                        per_model_video_predictions.setdefault(disp, {})[vp] = list(preds)
                        unique_frames = {int(getattr(p, 'frame_index', -1)) for p in preds if getattr(p, 'frame_index', None) is not None}
                        frames_processed = 0
                        if frames_targeted:
                            frames_processed = len(frames_targeted)
                        elif unique_frames:
                            frames_processed = len(unique_frames)
                        elif preds:
                            frames_processed = len(preds)
                        per_video_infer_stats[disp] = {
                            "time": per_video_infer_stats.get(disp, {}).get("time", 0.0) + float(elapsed),
                            "frames": per_video_infer_stats.get(disp, {}).get("frames", 0.0) + float(frames_processed),
                        }
                    # --- NEW: write per-video predictions for easier debugging/inspection ---
                    try:
                        vid_stem = os.path.splitext(os.path.basename(vp))[0]
                        _subj_id_pred = dm.video_subjects.get(vp, "unknown")
                        pred_dir = os.path.join(out_dir_base, "predictions_by_video", _subj_id_pred, vid_stem)
                        os.makedirs(pred_dir, exist_ok=True)
                        for display_name, pl in pv_predictions.items():
                            outp_video = os.path.join(pred_dir, f"{display_name}.json")
                            with open(outp_video, "w", encoding="utf-8") as f:
                                rows = []
                                for p in pl:
                                    row = {"frame_index": p.frame_index, "bbox": list(p.bbox), "score": p.score}
                                    conf_val = getattr(p, "confidence", None)
                                    if conf_val is not None:
                                        row["confidence"] = conf_val
                                    if bool(getattr(p, "is_fallback", False)):
                                        row["fallback"] = True
                                    rows.append(row)
                                json.dump(rows, f, ensure_ascii=False, indent=2)
                        self._log(f"Per-video predictions saved: {pred_dir}", to_console=(tqdm is None))
                    except Exception:
                        pass
                    if evaluator:
                        gt_frames_nonempty = {int(fi) for fi, boxes in gt.get("frames", {}).items() if boxes}
                        if restrict_to_gt_frames and not gt_frames_nonempty:
                            self._log(
                                f"[Eval] Skip metrics for {os.path.basename(vp)} (no GT frames).",
                                to_console=(tqdm is None),
                            )
                            res = {}
                            # still keep per-model predictions in per_model_video_predictions
                            continue
                        # Filter predictions to GT frames only for evaluation (avoid penalizing unannotated frames)
                        if restrict_to_gt_frames:
                            gt_frames_set = gt_frames_nonempty
                            eval_pv_predictions = {}
                            if gt_frames_set:
                                min_gt = min(gt_frames_set)
                            for mn, pl in pv_predictions.items():
                                # original filter (no shift)
                                direct = [p for p in pl if int(getattr(p, 'frame_index', -1)) in gt_frames_set]
                                best = direct
                                # try alignment by constant offset: delta = min_gt - min_pred
                                try:
                                    if gt_frames_set and pl:
                                        min_pred = int(min(int(getattr(p, 'frame_index', 0)) for p in pl))
                                        delta = int(min_gt) - int(min_pred)
                                        if delta != 0:
                                            shifted = [
                                                FramePrediction(
                                                    frame_index=int(p.frame_index) + delta,
                                                    bbox=p.bbox,
                                                    score=p.score,
                                                    confidence=getattr(p, "confidence", None),
                                                    confidence_components=getattr(p, "confidence_components", None),
                                                    segmentation=getattr(p, "segmentation", None),
                                                    is_fallback=bool(getattr(p, "is_fallback", False)),
                                                )
                                                for p in pl
                                            ]
                                            shifted_f = [p for p in shifted if int(getattr(p, 'frame_index', -1)) in gt_frames_set]
                                            # choose the one with higher coverage on GT frames
                                            if len(shifted_f) > len(direct):
                                                best = shifted_f
                                except Exception:
                                    pass
                                eval_pv_predictions[mn] = best
                        else:
                            eval_pv_predictions = pv_predictions
                        # per-video evaluation directory to avoid overwriting (split by subject)
                        vid_stem = os.path.splitext(os.path.basename(vp))[0]
                        _subj_id_met = dm.video_subjects.get(vp, "unknown")
                        vid_met_dir = os.path.join(met_dir, _subj_id_met, vid_stem)
                        os.makedirs(vid_met_dir, exist_ok=True)
                        res = evaluator.evaluate(eval_pv_predictions, gt, vid_met_dir)
                        # --- Debug: log coverage stats per model (from evaluator summary) ---
                        try:
                            for _mn, _sm in res.items():
                                dbg_total_gt = _sm.get('debug_total_gt_frames')
                                dbg_total_pred = _sm.get('debug_total_pred_frames')
                                dbg_matched = _sm.get('debug_matched_frames')
                                dbg_cover = _sm.get('debug_coverage_ratio')
                                dbg_gt_rng = _sm.get('debug_gt_frame_range')
                                dbg_pred_rng = _sm.get('debug_pred_frame_range')
                                dbg_offset = _sm.get('debug_suggested_constant_offset')
                                self._log(
                                    f"[DebugCoverage] video={os.path.basename(vp)} model={_mn} "
                                    f"gt_frames={dbg_total_gt} pred_frames={dbg_total_pred} matched={dbg_matched} "
                                    f"coverage={dbg_cover:.3f} gt_range={dbg_gt_rng} pred_range={dbg_pred_rng} suggested_offset={dbg_offset}",
                                    to_console=(tqdm is None)
                                )
                        except Exception:
                            pass
                        # accumulate for dataset-level summary
                        for display_name, sm in res.items():
                            agg = dataset_agg.setdefault(display_name, {
                                "count": 0,
                                "sum_iou": 0.0, "sum_iou_sq": 0.0,
                                "sum_ce": 0.0, "sum_ce_sq": 0.0,
                                # detection aggregation
                                "tp_50": 0, "fp_50": 0, "fn_50": 0,
                                "tp_75": 0, "fp_75": 0, "fn_75": 0,
                                # Success AUC aggregation (average across videos)
                                "sum_success_auc": 0.0, "videos": 0,
                                # Inference performance aggregation
                                "sum_infer_time": 0.0, "sum_infer_frames": 0.0,
                                # Drift aggregation (per-video statistic)
                                "sum_drift_rate": 0.0, "drift_samples": 0,
                            })
                            agg["count"] += int(sm.get("count", 0))
                            agg["sum_iou"] += float(sm.get("sum_iou", 0.0))
                            agg["sum_iou_sq"] += float(sm.get("sum_iou_sq", 0.0))
                            agg["sum_ce"] += float(sm.get("sum_ce", 0.0))
                            agg["sum_ce_sq"] += float(sm.get("sum_ce_sq", 0.0))
                            agg["tp_50"] += int(sm.get("tp_50", 0))
                            agg["fp_50"] += int(sm.get("fp_50", 0))
                            agg["fn_50"] += int(sm.get("fn_50", 0))
                            agg["tp_75"] += int(sm.get("tp_75", 0))
                            agg["fp_75"] += int(sm.get("fp_75", 0))
                            agg["fn_75"] += int(sm.get("fn_75", 0))
                            # Success AUC per-video scalar average
                            if "success_auc" in sm:
                                agg["sum_success_auc"] += float(sm.get("success_auc", 0.0))
                                agg["videos"] += 1
                            stats = per_video_infer_stats.get(display_name)
                            if stats:
                                agg["sum_infer_time"] += float(stats.get("time", 0.0))
                                agg["sum_infer_frames"] += float(stats.get("frames", 0.0))
                            if "drift_rate" in sm:
                                agg["sum_drift_rate"] += float(sm.get("drift_rate", 0.0))
                                agg["drift_samples"] += 1
                        self._log(f"Evaluated metrics written to: {vid_met_dir}", to_console=(tqdm is None))

                        # Optional visualization: draw up to N GT frames per video with GT/pred boxes
                        if viz_enabled and viz_include_detection:
                            try:
                                import cv2  # type: ignore
                                vis_dir = os.path.join(out_dir_base, "visualizations", _subj_id_met, vid_stem)
                                os.makedirs(vis_dir, exist_ok=True)
                                gt_frames_sorted = sorted([int(fi) for fi, boxes in gt.get("frames", {}).items() if boxes])
                                cap = cv2.VideoCapture(vp)
                                if not gt_frames_sorted:
                                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                    gt_frames_sorted = list(range(total_frames))
                                selected_frames = _select_evenly_spaced(gt_frames_sorted, viz_samples)
                                # preload predictions by frame for each model
                                eval_pred_maps = {}
                                for mn, pl in (eval_pv_predictions.items() if restrict_to_gt_frames else pv_predictions.items()):
                                    m = {}
                                    for p in pl:
                                        m[int(p.frame_index)] = {
                                            "bbox": p.bbox,
                                            "fallback": bool(getattr(p, "is_fallback", False)),
                                        }
                                    eval_pred_maps[mn] = m
                                for fi in selected_frames:
                                    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                                    ok, frame = cap.read()
                                    if not ok:
                                        continue
                                    # draw GT boxes in green
                                    for gtb in gt.get("frames", {}).get(fi, []) or []:
                                        x, y, w, h = map(float, gtb)
                                        cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0,255,0), 2)
                                    # draw per-model pred in red (one per model)
                                    for mn, fmap in eval_pred_maps.items():
                                        pb_entry = fmap.get(fi)
                                        if pb_entry is None:
                                            continue
                                        if isinstance(pb_entry, dict):
                                            pb = pb_entry.get("bbox")
                                            is_fb = bool(pb_entry.get("fallback"))
                                        else:
                                            pb = pb_entry
                                            is_fb = False
                                        if pb is None:
                                            continue
                                        x, y, w, h = map(float, pb)
                                        color = (0, 165, 255) if is_fb else (0, 0, 255)
                                        label = f"{mn} (fallback)" if is_fb else mn
                                        cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
                                        cv2.putText(frame, label, (int(x), max(0, int(y)-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                                    out_path = os.path.join(vis_dir, f"frame_{fi:06d}.jpg")
                                    try:
                                        cv2.imwrite(out_path, frame)
                                    except Exception:
                                        pass
                                cap.release()
                                self._log(f"Visualization images saved: {vis_dir}", to_console=(tqdm is None))
                            except Exception as _e_viz:
                                self._log(f"[Warn] Visualization failed: {_e_viz}", to_console=(tqdm is None))
                # dataset-level summary across all videos
                if evaluator and dataset_agg:
                    summary_out = {}
                    for model_name, a in dataset_agg.items():
                        n = max(1, int(a.get("count", 0)))
                        i_mu = a["sum_iou"] / n
                        # E[X^2] - (E[X])^2, guard non-negative
                        i_var = max(0.0, a["sum_iou_sq"] / n - i_mu * i_mu)
                        i_sd = i_var ** 0.5
                        c_mu = a["sum_ce"] / n
                        c_var = max(0.0, a["sum_ce_sq"] / n - c_mu * c_mu)
                        c_sd = c_var ** 0.5
                        # micro-precision as AP at single threshold
                        tp50, fp50 = int(a.get("tp_50", 0)), int(a.get("fp_50", 0))
                        tp75, fp75 = int(a.get("tp_75", 0)), int(a.get("fp_75", 0))
                        success_rate_50 = (tp50 / (tp50 + fp50)) if (tp50 + fp50) > 0 else 0.0
                        success_rate_75 = (tp75 / (tp75 + fp75)) if (tp75 + fp75) > 0 else 0.0
                        # Average Success AUC across videos (if any present)
                        vcnt = max(1, int(a.get("videos", 0)))
                        success_auc_mean = float(a.get("sum_success_auc", 0.0)) / vcnt if int(a.get("videos", 0)) > 0 else 0.0
                        infer_time_total = float(a.get("sum_infer_time", 0.0))
                        infer_frames_total = float(a.get("sum_infer_frames", 0.0))
                        fps = (infer_frames_total / infer_time_total) if infer_time_total > 0.0 else 0.0
                        drift_samples = int(a.get("drift_samples", 0))
                        drift_rate_mean = (float(a.get("sum_drift_rate", 0.0)) / drift_samples) if drift_samples > 0 else 0.0
                        summary_out[model_name] = {
                            "frames_count": n,
                            "iou_mean": i_mu, "iou_std": i_sd,
                            "ce_mean": c_mu, "ce_std": c_sd,
                            "success_rate_50": success_rate_50, "success_rate_75": success_rate_75,
                            "success_auc": success_auc_mean,
                            "fps": fps,
                            "drift_rate": drift_rate_mean,
                        }
                    summary_path = os.path.join(met_dir, "summary.json")
                    with open(summary_path, "w", encoding="utf-8") as f:
                        json.dump(summary_out, f, ensure_ascii=False, indent=2)
                    self._log(f"Dataset metrics summary saved: {summary_path}", to_console=(tqdm is None))
                    stage_metrics_collector[phase] = {
                        "summary": summary_out,
                        "summary_path": summary_path,
                        "metrics_dir": met_dir,
                    }
                pred_dir = os.path.join(out_dir_base, "predictions")
                os.makedirs(pred_dir, exist_ok=True)
                for display_name, preds in predictions_all.items():
                    outp = os.path.join(pred_dir, f"{display_name}.json")
                    with open(outp, "w", encoding="utf-8") as f:
                        rows = []
                        for p in preds:
                            row = {"frame_index": p.frame_index, "bbox": list(p.bbox), "score": p.score}
                            conf_val = getattr(p, "confidence", None)
                            if conf_val is not None:
                                row["confidence"] = conf_val
                            components = getattr(p, "confidence_components", None)
                            if components:
                                row["components"] = {k: float(v) for k, v in components.items()}
                            if bool(getattr(p, "is_fallback", False)):
                                row["fallback"] = True
                            rows.append(row)
                        json.dump(rows, f, ensure_ascii=False, indent=2)
                    self._log(f"Predictions saved: {outp}", to_console=(tqdm is None))

                if phase in stage_metrics_collector:
                    stage_metrics_collector[phase]["predictions_dir"] = pred_dir
                else:
                    stage_metrics_collector[phase] = {"predictions_dir": pred_dir}

                return per_model_video_predictions

            # k-fold within training for validation
            if k_fold > 1 and len(train_ds) > 0:
                with stage_scope(
                    "detector_kfold",
                    "detection",
                    {"folds": k_fold, "train_videos": len(train_ds)},
                ) as stage_entry:
                    self._log(f"Running {k_fold}-Fold validation on training set…")
                    train_vids = [train_ds[i]["video_path"] for i in range(len(train_ds))]
                    rng = random.Random(self.seed)
                    vids = train_vids[:]
                    rng.shuffle(vids)
                    # Round-robin allocation for even fold sizes
                    fold_bins: list[list[str]] = [[] for _ in range(k_fold)]
                    for _rr_idx, _rr_v in enumerate(vids):
                        fold_bins[_rr_idx % k_fold].append(_rr_v)
                    fold_summaries = []
                    fold_iter = range(k_fold)
                    if tqdm is not None:
                        fold_iter = tqdm(range(k_fold), total=k_fold, desc="K-Fold", unit="fold")
                    for fi in fold_iter:
                        phase_key = f"kfold-{fi+1}"
                        self._progress('kfold_fold', fi + 1, k_fold, {'exp': exp_name})
                        val_vids = fold_bins[fi]
                        trn_vids = [v for v in vids if v not in set(val_vids)]
                        val_ds = SimpleDataset(val_vids, dm.ann_by_video)
                        trn_ds = SimpleDataset(trn_vids, dm.ann_by_video)
                        fold_dir = os.path.join(train_root, "detection", f"fold_{fi+1}")
                        os.makedirs(fold_dir, exist_ok=True)
                        fold_models = build_models()
                        for model_name, model in fold_models:
                            if not hasattr(model, "train"):
                                continue
                            if len(trn_ds) == 0:
                                self._log(
                                    f"[Train] Skipped (no annotated data) | model={model_name} | fold={fi+1}/{k_fold}"
                                )
                                continue
                            allow_train = getattr(model, "train_enabled", True)
                            should_train_cb = getattr(model, "should_train", None)
                            if allow_train and callable(should_train_cb):
                                try:
                                    allow_train = allow_train and bool(should_train_cb(trn_ds, val_ds))
                                except Exception:
                                    pass
                            if not allow_train:
                                self._log(
                                    f"[Train] Skipped (disabled) | model={model_name} | fold={fi+1}/{k_fold}",
                                    to_console=(tqdm is None),
                                )
                                continue
                            self._log(
                                f"[Train] Fold {fi+1}/{k_fold} | model={model_name} | train_videos={len(trn_ds)} val_videos={len(val_ds)}"
                            )
                            try:
                                ret = model.train(
                                    trn_ds,
                                    val_ds,
                                    seed=self.seed,
                                    output_dir=os.path.join(fold_dir, "train"),
                                )
                                if isinstance(ret, dict):
                                    self._log(f"[Train] Result: {ret}", to_console=(tqdm is None))
                            except Exception as e:
                                self._log(
                                    f"[ERROR] Training failed | model={model_name} fold={fi+1} error={e}\n{_tb.format_exc()}"
                                )
                        _ = run_on_dataset(val_ds, fold_dir, models_list=fold_models, phase=phase_key)
                        try:
                            with open(os.path.join(fold_dir, "metrics", "summary.json"), "r", encoding="utf-8") as f:
                                fold_summaries.append(json.load(f))
                        except Exception:
                            pass
                    agg = {}
                    for sm in fold_summaries:
                        for model_name, m in sm.items():
                            agg.setdefault(
                                model_name,
                                {"iou_mean": [], "ce_mean": [], "success_rate_50": [], "success_rate_75": [], "success_auc": [], "fps": [], "drift_rate": []},
                            )
                            agg[model_name]["iou_mean"].append(m.get("iou_mean", 0.0))
                            agg[model_name]["ce_mean"].append(m.get("ce_mean", 0.0))
                            if "success_rate_50" in m:
                                agg[model_name]["success_rate_50"].append(m.get("success_rate_50", 0.0))
                            elif "mAP_50" in m:
                                agg[model_name]["success_rate_50"].append(m.get("mAP_50", 0.0))
                            if "success_rate_75" in m:
                                agg[model_name]["success_rate_75"].append(m.get("success_rate_75", 0.0))
                            elif "mAP_75" in m:
                                agg[model_name]["success_rate_75"].append(m.get("mAP_75", 0.0))
                            if "success_auc" in m:
                                agg[model_name]["success_auc"].append(m.get("success_auc", 0.0))
                            if "fps" in m:
                                agg[model_name]["fps"].append(m.get("fps", 0.0))
                            if "drift_rate" in m:
                                agg[model_name]["drift_rate"].append(m.get("drift_rate", 0.0))
                    comp_dir = os.path.join(out_dir, "comparison")
                    os.makedirs(comp_dir, exist_ok=True)
                    agg_out = {}
                    for model_name, vals in agg.items():
                        def mean_std(arr):
                            if not arr:
                                return (0.0, 0.0)
                            mu = sum(arr) / len(arr)
                            sd = (sum((x - mu) ** 2 for x in arr) / len(arr)) ** 0.5
                            return (mu, sd)

                        i_mu, i_sd = mean_std(vals["iou_mean"])
                        c_mu, c_sd = mean_std(vals["ce_mean"])
                        agg_out[model_name] = {
                            "iou_mean_mean": i_mu,
                            "iou_mean_std": i_sd,
                            "ce_mean_mean": c_mu,
                            "ce_mean_std": c_sd,
                            "success_rate_50_mean": mean_std(vals.get("success_rate_50", []))[0],
                            "success_rate_50_std": mean_std(vals.get("success_rate_50", []))[1],
                            "success_rate_75_mean": mean_std(vals.get("success_rate_75", []))[0],
                            "success_rate_75_std": mean_std(vals.get("success_rate_75", []))[1],
                            "success_auc_mean": mean_std(vals.get("success_auc", []))[0],
                            "success_auc_std": mean_std(vals.get("success_auc", []))[1],
                            "fps_mean": mean_std(vals.get("fps", []))[0],
                            "fps_std": mean_std(vals.get("fps", []))[1],
                            "drift_rate_mean": mean_std(vals.get("drift_rate", []))[0],
                            "drift_rate_std": mean_std(vals.get("drift_rate", []))[1],
                        }
                    summary_file = os.path.join(comp_dir, "kfold_summary.json")
                    with open(summary_file, "w", encoding="utf-8") as f:
                        json.dump(agg_out, f, ensure_ascii=False, indent=2)
                    stage_entry["aggregated_metrics"] = agg_out
                    per_fold_details = {}
                    for fi in range(k_fold):
                        key = f"kfold-{fi+1}"
                        info = stage_metrics_collector.get(key)
                        if info:
                            per_fold_details[key] = info
                    if per_fold_details:
                        stage_entry["per_fold_results"] = per_fold_details
                    stage_entry.setdefault("artifacts", {})["summary"] = summary_file
                    meta.setdefault("artifacts", {})["kfold_summary"] = summary_file
                    try:
                        import matplotlib.pyplot as plt  # type: ignore

                        labels = list(agg_out.keys())
                        i_means = [agg_out[k]["iou_mean_mean"] for k in labels]
                        i_stds = [agg_out[k]["iou_mean_std"] for k in labels]
                        plt.figure()
                        plt.bar(labels, i_means, yerr=i_stds, capsize=5)
                        plt.ylabel("IoU mean (±std)")
                        plt.title("K-Fold Aggregate IoU")
                        plt.xticks(rotation=30, ha='right')
                        plt.tight_layout()
                        plt.savefig(os.path.join(comp_dir, "kfold_iou_bar.png"))
                        plt.close()
                        c_means = [agg_out[k]["ce_mean_mean"] for k in labels]
                        c_stds = [agg_out[k]["ce_mean_std"] for k in labels]
                        plt.figure()
                        plt.bar(labels, c_means, yerr=c_stds, capsize=5)
                        plt.ylabel("Center Error (px) mean (±std)")
                        plt.title("K-Fold Aggregate Center Error")
                        plt.xticks(rotation=30, ha='right')
                        plt.tight_layout()
                        plt.savefig(os.path.join(comp_dir, "kfold_ce_bar.png"))
                        plt.close()
                    except Exception:
                        pass
                    meta.setdefault("artifacts", {})["kfold_dir"] = comp_dir
                    _dump_meta()
                    self._log("K-Fold validation finished. Aggregates saved.", to_console=(tqdm is None))
            elif k_fold > 1:
                _record_stage_skip(
                    "detector_kfold",
                    "detection",
                    "no annotated training videos available",
                    {"folds": k_fold, "train_videos": len(train_ds)},
                )

            # final training on full train and evaluate on test
            else:
                if k_fold > 1:
                    self._log("[Train] Skipping k-fold validation (no annotated videos).")

            final_models = build_models()
            final_train_dir = os.path.join(train_root, "detection")
            os.makedirs(final_train_dir, exist_ok=True)
            det_sig = _detector_signature(exp, scheme, train_ds, subject, fold_idx)

            with stage_scope(
                "detector_train_full",
                "detection",
                {
                    "train_videos": len(train_ds),
                    "output_dir": final_train_dir,
                    "reuse_signature": det_sig.get("key"),
                },
            ) as stage_entry:
                model_reports: List[Dict[str, Any]] = []
                stage_entry["models"] = model_reports
                for model_name, model in final_models:
                    report: Dict[str, Any] = {"model": model_name, "train_videos": len(train_ds)}
                    model_reports.append(report)
                    cached = detector_reuse_cache.get(det_sig["key"], {}).get(model_name)
                    cached_ckpt = cached.get("checkpoint") if isinstance(cached, dict) else None
                    if cached_ckpt and os.path.exists(cached_ckpt) and hasattr(model, "load_checkpoint"):
                        try:
                            model.load_checkpoint(cached_ckpt)
                            report["status"] = "reused"
                            report["checkpoint"] = cached_ckpt
                            report["reuse_signature"] = det_sig.get("key")
                            self._log(
                                f"[Train] Reuse detector | model={model_name} | ckpt={cached_ckpt}",
                                to_console=(tqdm is None),
                            )
                            continue
                        except Exception as e:
                            report["reuse_error"] = {"type": type(e).__name__, "message": str(e)}
                    if not hasattr(model, "train"):
                        report["status"] = "unsupported"
                        continue
                    if len(train_ds) == 0:
                        report["status"] = "skipped_no_data"
                        self._log(
                            f"[Train] Skipped (no annotated data) | model={model_name}",
                            to_console=(tqdm is None),
                        )
                        continue
                    allow_train = getattr(model, "train_enabled", True)
                    should_train_cb = getattr(model, "should_train", None)
                    if allow_train and callable(should_train_cb):
                        try:
                            allow_train = allow_train and bool(should_train_cb(train_ds, None))
                        except Exception:
                            pass
                    if not allow_train:
                        report["status"] = "skipped_disabled"
                        self._log(
                            f"[Train] Skipped (disabled) | model={model_name}",
                            to_console=(tqdm is None),
                        )
                        continue
                    self._log(f"[Train] Full train | model={model_name} | train_videos={len(train_ds)}")
                    try:
                        ret = model.train(train_ds, None, seed=self.seed, output_dir=final_train_dir)
                        report["status"] = "trained"
                        if isinstance(ret, dict):
                            report["result"] = ret
                            self._log(f"[Train] Result: {ret}", to_console=(tqdm is None))
                            if ret.get("status") == "no_data":
                                self._log(
                                    "[Warn] No training samples found (check that each video has a matching .json annotation next to it).",
                                    to_console=(tqdm is None),
                                )
                            best_ckpt = ret.get("best_ckpt") or ret.get("best_checkpoint") or ret.get("checkpoint")
                            if best_ckpt and os.path.exists(best_ckpt):
                                cache_entry: Dict[str, Any] = {
                                    "checkpoint": best_ckpt,
                                    "signature": det_sig.get("payload"),
                                    "trained_at": datetime.utcnow().isoformat() + "Z",
                                }
                                detector_reuse_cache.setdefault(det_sig["key"], {})[model_name] = cache_entry
                                report["checkpoint"] = best_ckpt
                                # Persist to disk so subsequent runs (and separate program
                                # sessions) can also skip re-training this detector.
                                self._save_entry_to_persistent_cache(
                                    det_sig["key"], model_name, cache_entry
                                )
                                self._log(
                                    f"[DetectorCache] Saved to persistent cache | sig={det_sig['key'][:16]}… | ckpt={best_ckpt}",
                                    to_console=False,
                                )
                    except Exception as e:
                        report["status"] = "train_failed"
                        report["error"] = {"type": type(e).__name__, "message": str(e)}
                        self._log(f"[ERROR] Training failed | model={model_name} error={e}\n{_tb.format_exc()}")
                _dump_meta()
            test_dir = os.path.join(out_dir, "test")
            os.makedirs(test_dir, exist_ok=True)
            detection_test_dir = os.path.join(test_dir, "detection")
            os.makedirs(detection_test_dir, exist_ok=True)
            with stage_scope(
                "detector_eval",
                "detection",
                {"dataset": "test", "output_dir": detection_test_dir},
            ) as stage_entry:
                test_predictions = run_on_dataset(test_ds, detection_test_dir, models_list=final_models, phase="test")
                detection_metrics = stage_metrics_collector.get("test", {})
                if detection_metrics:
                    stage_entry["metrics"] = detection_metrics.get("summary")
                    stage_entry.setdefault("artifacts", {})["summary_path"] = detection_metrics.get("summary_path")
                    stage_entry.setdefault("artifacts", {})["metrics_dir"] = detection_metrics.get("metrics_dir")
                    stage_entry.setdefault("artifacts", {})["predictions_dir"] = detection_metrics.get("predictions_dir")
                _dump_meta()

            # ── Trajectory Filter Stage ─────────────────────────────────
            # Detection → Trajectory Filtering → Re-cropping → Segmentation
            # Applies multi-scale Hampel + bidirectional S-G to smooth noisy
            # detection bbox trajectories before segmentation ROI cropping.
            traj_filter_cfg = self.cfg.get("trajectory_filter", {}) or {}
            traj_filter_enabled = bool(traj_filter_cfg.get("enabled", True))
            if traj_filter_enabled and test_predictions:
                import numpy as _np_tf
                from ..classification.trajectory_filter import (
                    filter_detections as _filter_detections,
                    compute_trajectory_metrics as _compute_traj_metrics,
                )
                with stage_scope(
                    "trajectory_filter",
                    "trajectory_filter",
                    {
                        "bbox_strategy": str(traj_filter_cfg.get("bbox_strategy", "none")),
                        "detector_models": list(test_predictions.keys()),
                    },
                ) as stage_entry:
                    _tf_bbox_strategy = str(traj_filter_cfg.get("bbox_strategy", "none"))
                    _tf_bbox_params = dict(traj_filter_cfg.get("bbox_params", {}) or {})
                    _tf_traj_params = dict(traj_filter_cfg.get("traj_params", {}) or {})

                    _tf_metrics_all: Dict[str, Dict[str, Dict[str, Any]]] = {}
                    _tf_filtered: Dict[str, Dict[str, list]] = {}

                    for _tf_mn, _tf_pbv in test_predictions.items():
                        _tf_filtered[_tf_mn] = {}
                        _tf_metrics_all[_tf_mn] = {}

                        for _tf_vp, _tf_preds in _tf_pbv.items():
                            if len(_tf_preds) < 2:
                                _tf_filtered[_tf_mn][_tf_vp] = list(_tf_preds)
                                continue

                            # Deduplicate predictions with same frame_index (keep first)
                            _tf_seen_fi: set = set()
                            _tf_dedup: list = []
                            for _tp in _tf_preds:
                                if _tp.frame_index not in _tf_seen_fi:
                                    _tf_seen_fi.add(_tp.frame_index)
                                    _tf_dedup.append(_tp)
                            _tf_preds = _tf_dedup

                            if len(_tf_preds) < 2:
                                _tf_filtered[_tf_mn][_tf_vp] = list(_tf_preds)
                                continue

                            # Extract arrays from FramePrediction objects
                            _tf_fi = _np_tf.array([p.frame_index for p in _tf_preds], dtype=_np_tf.int64)
                            _tf_bb = _np_tf.array([list(p.bbox) for p in _tf_preds], dtype=_np_tf.float64)
                            _tf_cx = _tf_bb[:, 0] + _tf_bb[:, 2] / 2.0
                            _tf_cy = _tf_bb[:, 1] + _tf_bb[:, 3] / 2.0
                            _tf_w = _tf_bb[:, 2].copy()
                            _tf_h = _tf_bb[:, 3].copy()
                            _tf_sc = _np_tf.array(
                                # Detector predictions: default to 0.0 if score missing
                                [float(p.score if p.score is not None else 0.0) for p in _tf_preds],
                                dtype=_np_tf.float64,
                            )

                            # Before-filtering metrics (sort first for correct time order)
                            _tf_sort_b = _np_tf.argsort(_tf_fi)
                            _tf_before = _compute_traj_metrics(
                                _tf_cx[_tf_sort_b], _tf_cy[_tf_sort_b],
                                _tf_w[_tf_sort_b], _tf_h[_tf_sort_b],
                                _tf_fi[_tf_sort_b],
                            )

                            # Apply trajectory filter (multi-scale Hampel + bidirectional S-G)
                            _tf_result = _filter_detections(
                                _tf_fi, _tf_cx, _tf_cy, _tf_w, _tf_h, _tf_sc,
                                bbox_strategy=_tf_bbox_strategy,
                                bbox_params=_tf_bbox_params,
                                traj_params=_tf_traj_params,
                            )

                            # After-filtering metrics
                            _tf_after = _compute_traj_metrics(
                                _tf_result["cx"], _tf_result["cy"],
                                _tf_result["widths"], _tf_result["heights"],
                                _tf_result["frame_indices"],
                            )

                            _tf_vid_stem = os.path.splitext(os.path.basename(_tf_vp))[0]
                            _tf_metrics_all[_tf_mn][_tf_vid_stem] = {
                                "before": _tf_before,
                                "after": _tf_after,
                                "frames": int(len(_tf_fi)),
                            }

                            # Map sorted results back to original prediction order
                            # Use frame_index→result_index lookup for robustness
                            _tf_fi_result = _tf_result["frame_indices"]
                            _tf_fi_map = {int(fi): idx for idx, fi in enumerate(_tf_fi_result)}

                            _tf_new_preds: list = []
                            for _tf_i, _tf_pred in enumerate(_tf_preds):
                                _tf_k = _tf_fi_map.get(int(_tf_pred.frame_index))
                                if _tf_k is None:
                                    # Frame was removed (shouldn't happen after dedup); keep original
                                    _tf_new_preds.append(_tf_pred)
                                    continue
                                _tf_ncx = float(_tf_result["cx"][_tf_k])
                                _tf_ncy = float(_tf_result["cy"][_tf_k])
                                _tf_nw = float(_tf_result["widths"][_tf_k])
                                _tf_nh = float(_tf_result["heights"][_tf_k])
                                _tf_nx = _tf_ncx - _tf_nw / 2.0
                                _tf_ny = _tf_ncy - _tf_nh / 2.0
                                _tf_new_preds.append(FramePrediction(
                                    frame_index=_tf_pred.frame_index,
                                    bbox=(_tf_nx, _tf_ny, _tf_nw, _tf_nh),
                                    score=_tf_pred.score,
                                    confidence=getattr(_tf_pred, "confidence", None),
                                    confidence_components=getattr(_tf_pred, "confidence_components", None),
                                    segmentation=getattr(_tf_pred, "segmentation", None),
                                    is_fallback=bool(getattr(_tf_pred, "is_fallback", False)),
                                    bbox_source=getattr(_tf_pred, "bbox_source", "detector"),
                                ))
                            _tf_filtered[_tf_mn][_tf_vp] = _tf_new_preds

                    # Replace test_predictions with filtered version for downstream stages
                    test_predictions = _tf_filtered

                    # Save per-video and summary metrics
                    _tf_out_dir = os.path.join(test_dir, "trajectory_filter")
                    os.makedirs(_tf_out_dir, exist_ok=True)
                    _tf_metrics_path = os.path.join(_tf_out_dir, "metrics.json")
                    with open(_tf_metrics_path, "w", encoding="utf-8") as f:
                        json.dump(_tf_metrics_all, f, ensure_ascii=False, indent=2)

                    # Aggregate before/after across all videos for dataset-level summary
                    _tf_agg_b: Dict[str, list] = {}
                    _tf_agg_a: Dict[str, list] = {}
                    for _tf_vm in _tf_metrics_all.values():
                        for _tf_vd in _tf_vm.values():
                            for _tk, _tv in _tf_vd.get("before", {}).items():
                                _tf_agg_b.setdefault(_tk, []).append(float(_tv))
                            for _tk, _tv in _tf_vd.get("after", {}).items():
                                _tf_agg_a.setdefault(_tk, []).append(float(_tv))
                    _tf_summary = {
                        "before": {
                            k: float(_np_tf.mean(v)) for k, v in _tf_agg_b.items()
                        } if _tf_agg_b else {},
                        "after": {
                            k: float(_np_tf.mean(v)) for k, v in _tf_agg_a.items()
                        } if _tf_agg_a else {},
                        "config": {
                            "bbox_strategy": _tf_bbox_strategy,
                            "bbox_params": _tf_bbox_params,
                            "traj_params": _tf_traj_params,
                        },
                    }
                    _tf_summary_path = os.path.join(_tf_out_dir, "summary.json")
                    with open(_tf_summary_path, "w", encoding="utf-8") as f:
                        json.dump(_tf_summary, f, ensure_ascii=False, indent=2)

                    # ── Filtered-detection accuracy metrics (IoU / CE / SR) ──────
                    # Re-evaluate the smoothed predictions against GT so the viewer
                    # can show bbox accuracy *after* trajectory filtering.
                    # This mirrors BasicEvaluator's exact IoU / CE / SR definitions.
                    try:
                        from ..utils.annotations import load_coco_vid as _tf_load_gt
                        # restrict_to_gt_frames is defined inside run_on_dataset (local scope);
                        # read it directly from config here so this outer-scope block is safe.
                        _tf_eval_cfg = self.cfg.get("evaluation", {}) or {}
                        restrict_to_gt_frames = bool(_tf_eval_cfg.get("restrict_to_gt_frames", True))

                        def _tf_iou(px, py, pw, ph, gx, gy, gw, gh):
                            ix1 = max(px, gx); iy1 = max(py, gy)
                            ix2 = min(px + pw, gx + gw); iy2 = min(py + ph, gy + gh)
                            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                            union = pw * ph + gw * gh - inter
                            return inter / union if union > 0 else 0.0

                        _tf_fdet_agg: Dict[str, Dict] = {}
                        for _fm, _fpbv in _tf_filtered.items():
                            _fa = _tf_fdet_agg.setdefault(_fm, {
                                "count": 0,
                                "sum_iou": 0.0, "sum_iou_sq": 0.0,
                                "sum_ce": 0.0, "sum_ce_sq": 0.0,
                                "tp_50": 0, "fp_50": 0, "fn_50": 0,
                                "tp_75": 0, "fp_75": 0, "fn_75": 0,
                            })
                            for _fvp, _fpreds in _fpbv.items():
                                if not _fpreds:
                                    continue
                                _fgt_json = os.path.splitext(_fvp)[0] + ".json"
                                if not os.path.exists(_fgt_json):
                                    continue
                                _fgt = _tf_load_gt(_fgt_json)
                                _fgt_frames = _fgt.get("frames", {})
                                if not _fgt_frames:
                                    continue
                                # Build prediction lookup by frame (first prediction wins)
                                _fpred_map: Dict[int, tuple] = {}
                                for _fp in _fpreds:
                                    _fk = int(getattr(_fp, "frame_index", -1))
                                    if _fk not in _fpred_map:
                                        _fpred_map[_fk] = tuple(_fp.bbox)[:4]
                                # Collect all GT frames with boxes
                                _fgt_set = {int(fi) for fi, boxes in _fgt_frames.items() if boxes}
                                # When restrict_to_gt_frames is on, only evaluate on GT frames
                                # (matching how the detector was evaluated in BasicEvaluator)
                                if restrict_to_gt_frames:
                                    _fall_frames = _fgt_set
                                else:
                                    _fall_frames = _fgt_set | set(_fpred_map.keys())
                                for _ffi in _fall_frames:
                                    _has_gt = _ffi in _fgt_set
                                    _has_pred = _ffi in _fpred_map
                                    if _has_gt and _has_pred:
                                        _px, _py, _pw, _ph = _fpred_map[_ffi]
                                        _gbox = _fgt_frames.get(_ffi, [None])[0]
                                        if _gbox is None:
                                            _fa["fp_50"] += 1
                                            _fa["fp_75"] += 1
                                            continue
                                        _gx, _gy, _gw, _gh = list(_gbox)[:4]
                                        _fiou = _tf_iou(_px, _py, _pw, _ph, _gx, _gy, _gw, _gh)
                                        _pce = float(_np_tf.sqrt(
                                            (_px + _pw / 2.0 - (_gx + _gw / 2.0)) ** 2 +
                                            (_py + _ph / 2.0 - (_gy + _gh / 2.0)) ** 2
                                        ))
                                        _fa["count"] += 1
                                        _fa["sum_iou"] += _fiou
                                        _fa["sum_iou_sq"] += _fiou * _fiou
                                        _fa["sum_ce"] += _pce
                                        _fa["sum_ce_sq"] += _pce * _pce
                                        if _fiou >= 0.50:
                                            _fa["tp_50"] += 1
                                        else:
                                            _fa["fp_50"] += 1
                                        if _fiou >= 0.75:
                                            _fa["tp_75"] += 1
                                        else:
                                            _fa["fp_75"] += 1
                                    elif _has_gt and not _has_pred:
                                        _fa["fn_50"] += 1
                                        _fa["fn_75"] += 1
                                    elif _has_pred and not _has_gt:
                                        _fa["fp_50"] += 1
                                        _fa["fp_75"] += 1

                        _tf_fdet_out: Dict[str, Dict] = {}
                        for _fm, _fa in _tf_fdet_agg.items():
                            _fn = max(1, int(_fa["count"]))
                            _fi_mu = _fa["sum_iou"] / _fn
                            _fi_sd = max(0.0, _fa["sum_iou_sq"] / _fn - _fi_mu * _fi_mu) ** 0.5
                            _fc_mu = _fa["sum_ce"] / _fn
                            _fc_sd = max(0.0, _fa["sum_ce_sq"] / _fn - _fc_mu * _fc_mu) ** 0.5
                            # SR = precision = tp / (tp + fp), same as BasicEvaluator
                            _ft50 = int(_fa["tp_50"]); _ff50 = int(_fa["fp_50"])
                            _ft75 = int(_fa["tp_75"]); _ff75 = int(_fa["fp_75"])
                            _tf_fdet_out[_fm] = {
                                "frames_count": int(_fa["count"]),
                                "iou_mean": _fi_mu, "iou_std": _fi_sd,
                                "ce_mean": _fc_mu, "ce_std": _fc_sd,
                                "success_rate_50": (_ft50 / (_ft50 + _ff50)) if (_ft50 + _ff50) > 0 else 0.0,
                                "success_rate_75": (_ft75 / (_ft75 + _ff75)) if (_ft75 + _ff75) > 0 else 0.0,
                            }
                        if _tf_fdet_out:
                            _tf_fdet_path = os.path.join(_tf_out_dir, "filtered_detection_summary.json")
                            with open(_tf_fdet_path, "w", encoding="utf-8") as f:
                                json.dump(_tf_fdet_out, f, ensure_ascii=False, indent=2)
                            self._log(
                                f"[TrajectoryFilter] Filtered-detection accuracy saved → {_tf_fdet_path}",
                                to_console=(tqdm is None),
                            )
                    except Exception as _tf_fdet_err:
                        self._log(
                            f"[TrajectoryFilter] WARNING: filtered detection accuracy metrics failed "
                            f"({_tf_fdet_err})"
                        )

                    stage_entry["metrics"] = _tf_summary
                    stage_entry.setdefault("artifacts", {})["metrics_path"] = _tf_metrics_path
                    stage_entry.setdefault("artifacts", {})["summary_path"] = _tf_summary_path
                    self._log(
                        f"[TrajectoryFilter] bbox_strategy={_tf_bbox_strategy} | "
                        f"videos_filtered={sum(len(v) for v in _tf_metrics_all.values())}",
                        to_console=(tqdm is None),
                    )
                    _dump_meta()
            elif not traj_filter_enabled:
                _record_stage_skip("trajectory_filter", "trajectory_filter", "disabled via config")

            # segmentation stage (mandatory)
            seg_cfg = self.cfg.get("segmentation", {}) or {}
            seg_results_root = os.path.join(test_dir, "segmentation")
            os.makedirs(seg_results_root, exist_ok=True)
            seg_train_root = os.path.join(train_root, "segmentation")
            seg_workflow = SegmentationWorkflow(seg_cfg, self.dataset_root, seg_train_root, self._log)
            # Ensure preprocessing scheme is honored for segmentation:
            # - global preprocs: applied to full frames before ROI cropping
            # - roi preprocs: applied to ROI crops (after cropping) before model inference
            try:
                seg_global_preprocs = preprocs
                seg_roi_preprocs = preprocs_roi

                # Scheme B/C: segmentation should NOT receive global preprocs,
                # because it must crop from RAW frames (C) or keep detector raw (B).
                if scheme in {"B", "C"}:
                    seg_global_preprocs = []

                setattr(seg_workflow, "preprocs", seg_global_preprocs)
                setattr(seg_workflow, "roi_preprocs", seg_roi_preprocs)
            except Exception:
                pass
            train_video_paths = [train_ds[i]["video_path"] for i in range(len(train_ds))]
            train_annotations = _load_annotations_for_videos(train_video_paths)
            annotated_train_paths = list(train_annotations.keys())
            seg_metrics_by_model: Dict[str, Dict[str, Dict[str, float]]] = {}
            seg_seed = seg_cfg.get("seed", seg_workflow.cfg.seed or self.seed)
            val_ratio = seg_cfg.get("val_ratio", seg_workflow.cfg.val_ratio)
            train_enabled = bool(seg_workflow.cfg.train)
            inference_ckpt = getattr(seg_workflow, "inference_checkpoint", None)
            seg_summary_path: Optional[str] = None

            # --- Segmentation persistent cache check ---
            _seg_sig = _segmentation_signature(
                seg_cfg, seg_workflow.model_name,
                getattr(seg_workflow, "preprocs", []),
                getattr(seg_workflow, "roi_preprocs", []),
                annotated_train_paths, seg_seed, scheme, subject, fold_idx,
            )
            _seg_sig_key = _seg_sig["key"]
            _seg_cached_entry = (_seg_disk_cache.get(_seg_sig_key) or {}).get(seg_workflow.model_name)
            _seg_cached_ckpt: Optional[str] = (
                _seg_cached_entry.get("checkpoint")
                if isinstance(_seg_cached_entry, dict) else None
            )

            if _seg_cached_ckpt and os.path.exists(_seg_cached_ckpt):
                self._log(
                    f"[SegCache] Cache hit for model={seg_workflow.model_name} – loading from: {_seg_cached_ckpt}"
                )
                _record_stage_skip(
                    "segmentation_train", "segmentation", "cache hit",
                    {"model": seg_workflow.model_name, "cached_checkpoint": _seg_cached_ckpt},
                )
                seg_workflow.load_checkpoint(_seg_cached_ckpt)
                seg_workflow.best_checkpoint = _seg_cached_ckpt
            elif train_enabled:
                if annotated_train_paths:
                    with stage_scope(
                        "segmentation_train",
                        "segmentation",
                        {
                            "model": seg_workflow.model_name,
                            "train_videos": len(annotated_train_paths),
                            "seed": int(seg_seed),
                        },
                    ) as stage_entry:
                        train_targets = annotated_train_paths[:]
                        val_videos: Optional[Sequence[str]] = None
                        if isinstance(val_ratio, (int, float)) and float(val_ratio) > 0.0 and len(train_targets) > 1:
                            vr = max(0.0, min(float(val_ratio), 0.9))
                            val_count = max(1, int(len(train_targets) * vr))
                            # Clamp val_count so at least 1 video remains for training
                            val_count = min(val_count, len(train_targets) - 1)
                            rng = random.Random(int(seg_seed))
                            shuffled = train_targets[:]
                            rng.shuffle(shuffled)
                            val_videos = shuffled[:val_count]
                            train_targets = shuffled[val_count:]
                        stage_entry["train_targets"] = len(train_targets)
                        stage_entry["val_videos"] = len(val_videos) if val_videos else 0
                        self._log(
                            f"[Segmentation] Training model={seg_workflow.model_name} | train_videos={len(train_targets)}"
                            + (f" val_videos={len(val_videos)}" if val_videos else "")
                        )
                        train_summary = seg_workflow.train(train_targets, val_videos, seed=int(seg_seed))
                        if train_summary:
                            stage_entry["metrics"] = train_summary
                            def _fmt(v: object) -> str:
                                # Format numbers consistently; fall back to str for non-numerics
                                try:
                                    return f"{float(v):.4f}"
                                except (TypeError, ValueError):
                                    return str(v)

                            self._log(
                                "[Segmentation] Training summary: "
                                + ", ".join(f"{k}={_fmt(v)}" for k, v in train_summary.items()),
                                to_console=(tqdm is None),
                            )
                        seg_workflow.load_checkpoint()
                        best_ckpt = getattr(seg_workflow, "best_checkpoint", None)
                        if best_ckpt:
                            stage_entry.setdefault("artifacts", {})["best_checkpoint"] = best_ckpt
                            self._save_to_stage_cache(
                                self._persistent_seg_cache_file, _seg_sig_key,
                                seg_workflow.model_name,
                                {"checkpoint": best_ckpt, "model_name": seg_workflow.model_name},
                            )
                            self._log(f"[SegCache] Saved checkpoint to persistent cache: {best_ckpt}")
                            # also update in-memory so later folds in same run can reuse
                            _seg_disk_cache.setdefault(_seg_sig_key, {})[seg_workflow.model_name] = {
                                "checkpoint": best_ckpt
                            }
                        _dump_meta()
                else:
                    self._log("[Segmentation] Warning: no annotated training videos found; skipping training phase.")
                    _record_stage_skip(
                        "segmentation_train",
                        "segmentation",
                        "no annotated training videos",
                        {"model": seg_workflow.model_name},
                    )
                    seg_workflow.load_checkpoint(inference_ckpt)
            else:
                self._log("[Segmentation] Training disabled via config; skipping training phase.")
                _record_stage_skip(
                    "segmentation_train",
                    "segmentation",
                    "disabled via config",
                    {"model": seg_workflow.model_name},
                )
                seg_workflow.load_checkpoint(inference_ckpt)
            # Run inference only if we have predictions and a model object
            test_video_paths = sorted({vp for model_map in test_predictions.values() for vp in model_map.keys()})
            test_annotations = _load_annotations_for_videos(test_video_paths)
            seg_viz_cfg = (self.cfg.get("evaluation", {}) or {}).get("visualize", {}) or {}
            seg_model_name = getattr(seg_workflow, "model_name", "segmentation_model")
            predictions_root = os.path.join(seg_results_root, "predictions", seg_model_name)
            os.makedirs(predictions_root, exist_ok=True)
            with stage_scope(
                "segmentation_infer",
                "segmentation",
                {
                    "model": seg_model_name,
                    "detector_models": list(test_predictions.keys()),
                    "predictions_root": predictions_root,
                },
            ) as stage_entry:
                def _augment_detection_visuals_with_seg_roi(det_vis_dir: str, roi_trace_path: str, model_label: str) -> None:
                    cv2 = None
                    try:
                        import cv2 as _cv2  # type: ignore
                        cv2 = _cv2
                    except Exception:
                        return
                    try:
                        if not os.path.isdir(det_vis_dir) or not os.path.exists(roi_trace_path):
                            return
                        with open(roi_trace_path, "r", encoding="utf-8") as f:
                            trace = json.load(f) or {}
                        if not isinstance(trace, dict) or not trace:
                            return
                        # Build a lookup frame_index(int) -> row
                        trace_map: Dict[int, Dict[str, Any]] = {}
                        for k, v in trace.items():
                            try:
                                fi = int(k)
                            except Exception:
                                continue
                            if isinstance(v, dict):
                                trace_map[fi] = v
                        if not trace_map:
                            return

                        for fname in os.listdir(det_vis_dir):
                            if not (fname.startswith("frame_") and fname.endswith(".jpg")):
                                continue
                            try:
                                fi = int(fname[len("frame_") : len("frame_") + 6])
                            except Exception:
                                continue
                            row = trace_map.get(fi)
                            if not row:
                                continue
                            if str(row.get("bbox_source", "")) != "prev_segmentation":
                                continue

                            # Prefer ROI bbox (expanded crop) if available; otherwise fall back to raw bbox.
                            bb = row.get("roi_bbox") or row.get("bbox")
                            if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                                continue
                            try:
                                x, y, w, h = map(float, bb)
                            except Exception:
                                continue
                            if w <= 1.0 or h <= 1.0:
                                continue

                            img_path = os.path.join(det_vis_dir, fname)
                            if not os.path.isfile(img_path) or os.path.getsize(img_path) == 0:
                                continue
                            try:
                                img = cv2.imread(img_path)
                            except Exception:
                                continue
                            if img is None:
                                continue

                            color = (255, 0, 0)  # blue (BGR)
                            label = f"{model_label} ROI(From Last Seg)"
                            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
                            cv2.putText(
                                img,
                                label,
                                (int(x), max(0, int(y) - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color,
                                1,
                                cv2.LINE_AA,
                            )
                            try:
                                cv2.imwrite(img_path, img)
                            except Exception:
                                pass
                    except Exception:
                        return

                def _roi_trace_fallback_stats(roi_trace_path: str) -> Optional[Dict[str, float]]:
                    """Compute ROI fallback stats from roi_trace.json.

                    Fallback definition:
                    - Direct ROI: bbox_source in {detector, tracker}
                    - Fallback ROI: everything else (prev_bbox / prev_segmentation / full_frame / segmentation_bootstrap / unknown)
                    Denominator: total frames in roi_trace (int keys).
                    """
                    try:
                        if not os.path.exists(roi_trace_path):
                            return None
                        with open(roi_trace_path, "r", encoding="utf-8") as f:
                            trace = json.load(f) or {}
                        if not isinstance(trace, dict) or not trace:
                            return None
                        direct_sources = {"detector", "tracker"}
                        total = 0
                        fallback = 0
                        for k, v in trace.items():
                            try:
                                _ = int(k)
                            except Exception:
                                continue
                            if not isinstance(v, dict):
                                continue
                            total += 1
                            src = str(v.get("bbox_source", "")).strip().lower()
                            if src not in direct_sources:
                                fallback += 1
                        if total <= 0:
                            return None
                        rate = float(fallback) / float(total)
                        return {
                            "roi_total_frames": float(total),
                            "roi_fallback_frames": float(fallback),
                            "roi_fallback_rate": float(rate),
                        }
                    except Exception:
                        return None

                def _inject_roi_fallback_metrics_into_detection(
                    model_label: str,
                    vid_key: str,
                    stats: Dict[str, float],
                ) -> None:
                    """Update detection metrics JSON files in-place for viewer consumption.

                    ``vid_key`` is ``subject_id/vid_stem`` for per-video files, or
                    empty string for the dataset-level summary.
                    """
                    try:
                        # Per-video summary.json
                        if vid_key:
                            det_vid_summary = os.path.join(detection_test_dir, "metrics", vid_key, "summary.json")
                            if os.path.exists(det_vid_summary):
                                try:
                                    with open(det_vid_summary, "r", encoding="utf-8") as f:
                                        data = json.load(f) or {}
                                except Exception:
                                    data = {}
                                if isinstance(data, dict):
                                    model_metrics = data.get(model_label)
                                    if isinstance(model_metrics, dict):
                                        model_metrics.update(stats)
                                        data[model_label] = model_metrics
                                        with open(det_vid_summary, "w", encoding="utf-8") as f:
                                            json.dump(data, f, ensure_ascii=False, indent=2)
                        # Dataset-level summary.json
                        det_ds_summary = os.path.join(detection_test_dir, "metrics", "summary.json")
                        if os.path.exists(det_ds_summary):
                            try:
                                with open(det_ds_summary, "r", encoding="utf-8") as f:
                                    ds = json.load(f) or {}
                            except Exception:
                                ds = {}
                            if isinstance(ds, dict):
                                mm = ds.get(model_label)
                                if isinstance(mm, dict):
                                    mm.update(stats)
                                    ds[model_label] = mm
                                    with open(det_ds_summary, "w", encoding="utf-8") as f:
                                        json.dump(ds, f, ensure_ascii=False, indent=2)
                    except Exception:
                        return

                for model_name, preds_by_video in test_predictions.items():
                    out_dir_model = os.path.join(predictions_root, model_name)
                    os.makedirs(out_dir_model, exist_ok=True)
                    metrics_payload = seg_workflow.predict_dataset(
                        preds_by_video,
                        out_dir_model,
                        gt_annotations=test_annotations,
                        viz_settings=seg_viz_cfg,
                    )

                    # Compute ROI fallback rate per video from roi_trace.json and inject into detection metrics.
                    # Also aggregate across dataset using total frames as denominator (micro-average).
                    roi_total_sum = 0.0
                    roi_fallback_sum = 0.0
                    try:
                        for vp in preds_by_video.keys():
                            _subj_id_seg = os.path.basename(os.path.dirname(vp))
                            vid_stem = os.path.splitext(os.path.basename(vp))[0]
                            roi_trace_path = os.path.join(out_dir_model, _subj_id_seg, vid_stem, "roi_trace.json")
                            stats = _roi_trace_fallback_stats(roi_trace_path)
                            if not stats:
                                continue
                            roi_total_sum += float(stats.get("roi_total_frames", 0.0))
                            roi_fallback_sum += float(stats.get("roi_fallback_frames", 0.0))
                            vid_key = os.path.join(_subj_id_seg, vid_stem)
                            _inject_roi_fallback_metrics_into_detection(str(model_name), vid_key, stats)
                    except Exception:
                        pass
                    if roi_total_sum > 0.0:
                        ds_rate = float(roi_fallback_sum) / float(roi_total_sum)
                        ds_stats = {
                            "roi_total_frames": float(roi_total_sum),
                            "roi_fallback_frames": float(roi_fallback_sum),
                            "roi_fallback_rate": float(ds_rate),
                        }
                        _inject_roi_fallback_metrics_into_detection(str(model_name), "", ds_stats)
                        try:
                            test_stage = stage_metrics_collector.get("test", {}).get("summary")
                            if isinstance(test_stage, dict) and isinstance(test_stage.get(str(model_name)), dict):
                                test_stage[str(model_name)].update(ds_stats)
                        except Exception:
                            pass

                    # Post-process detection visualization JPGs to include ROI bbox derived from previous segmentation.
                    try:
                        for vp in preds_by_video.keys():
                            _subj_id_seg = os.path.basename(os.path.dirname(vp))
                            vid_stem = os.path.splitext(os.path.basename(vp))[0]
                            det_vis_dir = os.path.join(detection_test_dir, "visualizations", _subj_id_seg, vid_stem)
                            roi_trace_path = os.path.join(out_dir_model, _subj_id_seg, vid_stem, "roi_trace.json")
                            _augment_detection_visuals_with_seg_roi(det_vis_dir, roi_trace_path, str(model_name))
                    except Exception:
                        pass
                    summary_metrics = metrics_payload.get("summary", {})
                    per_video_metrics = metrics_payload.get("videos", {})
                    if per_video_metrics:
                        try:
                            with open(os.path.join(out_dir_model, "metrics_per_video.json"), "w", encoding="utf-8") as f:
                                json.dump(per_video_metrics, f, ensure_ascii=False, indent=2)
                        except Exception:
                            pass
                    seg_metrics_by_model.setdefault(seg_model_name, {})[model_name] = summary_metrics
                    if summary_metrics:
                        summary_text = ", ".join(f"{k}={v:.4f}" for k, v in summary_metrics.items())
                    else:
                        summary_text = "no metrics"
                    self._log(
                        f"[Segmentation] Summary metrics | seg_model={seg_model_name} det_model={model_name}: {summary_text}",
                        to_console=(tqdm is None),
                    )
                if seg_metrics_by_model:
                    seg_summary_path = os.path.join(seg_results_root, "metrics_summary.json")
                    with open(seg_summary_path, "w", encoding="utf-8") as f:
                        json.dump(seg_metrics_by_model, f, ensure_ascii=False, indent=2)
                    stage_entry["metrics"] = seg_metrics_by_model
                    stage_entry.setdefault("artifacts", {})["summary_path"] = seg_summary_path
                    stage_entry.setdefault("artifacts", {})["predictions_root"] = predictions_root
                _dump_meta()

            clf_cfg = self.cfg.get("classification", {}) or {}
            clf_enabled = bool(clf_cfg.get("enabled", True))
            if clf_enabled:
                with stage_scope(
                    "classification",
                    "classification",
                    {"config": clf_cfg},
                ) as stage_entry:
                    try:
                        # --- Classification persistent cache ---
                        _clf_train_paths: List[str] = []
                        try:
                            _clf_train_paths = sorted({
                                train_ds[i]["video_path"]
                                for i in range(len(train_ds))
                                if isinstance(train_ds[i], dict) and "video_path" in train_ds[i]
                            })
                        except Exception:
                            pass
                        _clf_seg_model_hint = getattr(seg_workflow, "model_name", "unknown")
                        _clf_sig = _classification_signature(
                            clf_cfg, _clf_train_paths, _clf_seg_model_hint, subject, fold_idx,
                        )
                        _clf_sig_key = _clf_sig["key"]
                        _clf_model_key = (
                            f"{(clf_cfg.get('feature_extractor') or {}).get('name', 'fe')}"
                            f"_{(clf_cfg.get('classifier') or {}).get('name', 'clf')}"
                        )
                        _clf_cached_entry = (_clf_disk_cache.get(_clf_sig_key) or {}).get(_clf_model_key)
                        _clf_cached_ckpt: Optional[str] = (
                            _clf_cached_entry.get("checkpoint")
                            if isinstance(_clf_cached_entry, dict) else None
                        )
                        _clf_result = run_subject_classification(
                            clf_cfg,
                            self.dataset_root,
                            train_ds,
                            test_predictions,
                            out_dir,
                            self._log,
                            split_method=method,
                            cached_classifier_path=_clf_cached_ckpt,
                        )
                        if isinstance(_clf_result, dict):
                            stage_entry["metrics"] = _clf_result
                        # Save newly produced classifier to cache
                        _clf_produced_pkl = os.path.join(out_dir, "classification", "classifier.pkl")
                        if os.path.exists(_clf_produced_pkl):
                            _clf_cache_entry: Dict[str, str] = {
                                "checkpoint": _clf_produced_pkl,
                                "model_key": _clf_model_key,
                            }
                            self._save_to_stage_cache(
                                self._persistent_clf_cache_file, _clf_sig_key, _clf_model_key,
                                _clf_cache_entry,
                            )
                            self._log(f"[ClfCache] Saved classifier to persistent cache: {_clf_produced_pkl}")
                            _clf_disk_cache.setdefault(_clf_sig_key, {})[_clf_model_key] = _clf_cache_entry
                    except Exception as _e_cls:
                        import traceback as _tb
                        stage_entry["status"] = "failed"
                        _tb_str = _tb.format_exc()
                        stage_entry["error"] = {"type": type(_e_cls).__name__, "message": str(_e_cls), "traceback": _tb_str}
                        self._log(f"[Classification] Error: {_e_cls}\n{_tb_str}")
                    finally:
                        _dump_meta()
            else:
                _record_stage_skip("classification", "classification", "disabled via config", {"config": clf_cfg})

            metrics_section = meta.setdefault("metrics", {})
            if stage_metrics_collector:
                metrics_section["detection_phases"] = stage_metrics_collector
                test_summary = stage_metrics_collector.get("test", {}).get("summary")
                if test_summary:
                    metrics_section["detection"] = test_summary
                detection_artifacts = meta.setdefault("artifacts", {}).setdefault("detection", {})
                test_info = stage_metrics_collector.get("test", {})
                for key in ("summary_path", "metrics_dir", "predictions_dir"):
                    value = test_info.get(key)
                    if value:
                        detection_artifacts[key] = value

            if seg_metrics_by_model:
                metrics_section["segmentation"] = seg_metrics_by_model
                seg_artifacts = meta.setdefault("artifacts", {}).setdefault("segmentation", {})
                if seg_summary_path:
                    seg_artifacts["summary_path"] = seg_summary_path
                seg_artifacts["predictions_root"] = predictions_root

            # Trajectory filter metrics: read from test/trajectory_filter/summary.json
            _tf_summary_file = os.path.join(test_dir, "trajectory_filter", "summary.json")
            if os.path.exists(_tf_summary_file):
                try:
                    with open(_tf_summary_file, "r", encoding="utf-8") as f:
                        metrics_section["trajectory_filter"] = json.load(f)
                    tf_artifacts = meta.setdefault("artifacts", {}).setdefault("trajectory_filter", {})
                    tf_artifacts["summary_path"] = _tf_summary_file
                    tf_artifacts["metrics_path"] = os.path.join(test_dir, "trajectory_filter", "metrics.json")
                except Exception:
                    pass

            # Classification metrics: read from classification/summary.json produced by engine
            _clf_summary_path = os.path.join(out_dir, "classification", "summary.json")
            if os.path.exists(_clf_summary_path):
                try:
                    with open(_clf_summary_path, "r", encoding="utf-8") as f:
                        metrics_section["classification"] = json.load(f)
                except Exception:
                    pass

            meta["runtime"]["finished_at"] = datetime.utcnow().isoformat() + "Z"
            meta["runtime"]["duration_sec"] = max(0.0, time.perf_counter() - experiment_start)
            meta["runtime"]["stages"] = len(stage_records)
            _dump_meta()
