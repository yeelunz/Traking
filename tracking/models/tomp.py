from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception as ex:  # pragma: no cover - ensure graceful failure where OpenCV is unavailable
    cv2 = None  # type: ignore
    _CV2_IMPORT_ERROR = ex

try:
    import torch
except Exception as ex:  # pragma: no cover - keep explicit error messaging if Torch missing
    torch = None  # type: ignore
    _TORCH_IMPORT_ERROR = ex

# Ensure the bundled pytracking checkout is importable
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PYTRACKING_ROOT = _PROJECT_ROOT / "libs" / "pytracking"
if _PYTRACKING_ROOT.is_dir():
    if str(_PYTRACKING_ROOT) not in sys.path:
        sys.path.insert(0, str(_PYTRACKING_ROOT))

try:  # pytracking runtime
    from pytracking.tracker.tomp import ToMP as _PyTrackingToMP  # type: ignore
    from pytracking.utils.params import TrackerParams as _TrackerParams  # type: ignore
    from pytracking.evaluation.environment import env_settings as _env_settings  # type: ignore
except Exception as ex:  # pragma: no cover - keep failure explicit until runtime
    _PyTrackingToMP = None  # type: ignore
    _TrackerParams = None  # type: ignore
    _env_settings = None  # type: ignore
    _PYTRACKING_IMPORT_ERROR = ex

from ..core.interfaces import FramePrediction, PreprocessingModule, TrackingModel
from ..core.registry import register_model
from ..utils.annotations import load_coco_vid


def _resolve_param_module(param_name: str):
    """Resolve ToMP parameter preset to a module with a ``parameters`` factory."""

    preset = (param_name or "").strip().lower()
    if not preset:
        preset = "tomp50"

    aliases = {
        "tomp50": "pytracking.parameter.tomp.tomp50",
        "tomp-50": "pytracking.parameter.tomp.tomp50",
        "tomp101": "pytracking.parameter.tomp.tomp101",
        "tomp-101": "pytracking.parameter.tomp.tomp101",
    }

    module_path = aliases.get(preset, param_name)
    try:
        return importlib.import_module(module_path)
    except Exception as ex:
        raise RuntimeError(f"Failed to import ToMP parameter module '{module_path}': {ex}") from ex


def _first_gt_bbox(video_path: str) -> Optional[Tuple[float, float, float, float]]:
    json_path = os.path.splitext(video_path)[0] + ".json"
    if not os.path.exists(json_path):
        return None
    try:
        gt = load_coco_vid(json_path)
    except Exception:
        return None
    frames = gt.get("frames", {})
    if not frames:
        return None
    valid_keys = sorted(int(k) for k, boxes in frames.items() if boxes)
    if not valid_keys:
        return None
    first_idx = valid_keys[0]
    bbox = frames.get(first_idx, [None])[0]
    if not bbox:
        return None
    x, y, w, h = bbox
    return float(x), float(y), float(w), float(h)


def _is_valid_bbox(bbox: Optional[Tuple[float, float, float, float]]) -> bool:
    if bbox is None:
        return False
    x, y, w, h = bbox
    return all(np.isfinite([x, y, w, h])) and w > 0 and h > 0


@dataclass
class _TrackerOutputs:
    bbox: Optional[Tuple[float, float, float, float]]
    score: Optional[float]


@register_model("ToMP")
class ToMPTracker(TrackingModel):
    """Wrapper that exposes the ToMP tracker from the bundled pytracking checkout."""

    name = "ToMP"

    DEFAULT_CONFIG: Dict[str, Any] = {
        "parameter": "tomp50",
        "device": "cuda",
        "force_cpu_if_no_cuda": True,
        "fallback_last_prediction": True,
        "min_presence_score": 0.0,
        "param_overrides": {},
    }

    def __init__(self, config: Dict[str, Any]):
        if cv2 is None:
            detail = f" (import error: {_CV2_IMPORT_ERROR!r})" if '_CV2_IMPORT_ERROR' in globals() else ""
            raise RuntimeError("OpenCV is required for ToMP tracker." + detail)
        if torch is None:
            detail = f" (import error: {_TORCH_IMPORT_ERROR!r})" if '_TORCH_IMPORT_ERROR' in globals() else ""
            raise RuntimeError("PyTorch is required for ToMP tracker." + detail)
        if _PyTrackingToMP is None or _TrackerParams is None:
            detail = f" (import error: {_PYTRACKING_IMPORT_ERROR!r})" if '_PYTRACKING_IMPORT_ERROR' in globals() else ""
            raise RuntimeError("Bundled pytracking package is required for ToMP tracker." + detail)

        self.parameter_name = str(config.get("parameter", self.DEFAULT_CONFIG["parameter"]))
        self.device_preference = str(config.get("device", self.DEFAULT_CONFIG["device"])).lower()
        self.force_cpu_if_no_cuda = bool(config.get("force_cpu_if_no_cuda", self.DEFAULT_CONFIG["force_cpu_if_no_cuda"]))
        self.fallback_last_prediction = bool(config.get("fallback_last_prediction", self.DEFAULT_CONFIG["fallback_last_prediction"]))
        self.min_presence_score = float(config.get("min_presence_score", self.DEFAULT_CONFIG["min_presence_score"]))
        self._param_overrides = dict(config.get("param_overrides", self.DEFAULT_CONFIG["param_overrides"]))
        self._custom_checkpoint: Optional[str] = None

        param_module = _resolve_param_module(self.parameter_name)
        self._param_module_path = param_module.__name__

        if _env_settings is not None:
            env_cfg = _env_settings()
            net_root = Path(getattr(env_cfg, "network_path", ""))
            if net_root and not net_root.exists():
                raise RuntimeError(
                    "pytracking network_path does not exist: "
                    f"'{net_root}'. Place ToMP weights (e.g. tomp50.pth.tar) there or update local.py."
                )

        # Validate overrides eagerly to surface errors during construction
        self._create_params()

        self.preprocs: List[PreprocessingModule] = []

    # ------------------------------------------------------------------
    # TrackingModel API
    # ------------------------------------------------------------------
    def train(self, train_dataset, val_dataset=None, seed: int = 0, output_dir: Optional[str] = None):
        # ToMP requires no additional training at inference time. Emit a dummy callback for UI responsiveness.
        cb = getattr(self, "progress_callback", None)
        if callable(cb):
            try:
                cb("train_epoch_start", 1, 1)
            except Exception:
                pass
            try:
                cb("train_epoch_end", 1, 1)
            except Exception:
                pass
        return {"status": "no_training"}

    def load_checkpoint(self, ckpt_path: str):
        if not ckpt_path:
            raise ValueError("ckpt_path must be a non-empty string")
        self._custom_checkpoint = ckpt_path

    # ------------------------------------------------------------------
    def _apply_preprocs(self, frame_bgr: np.ndarray) -> np.ndarray:
        if not self.preprocs:
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        for proc in self.preprocs:
            rgb = proc.apply_to_frame(rgb)
        return rgb

    def _initialise_tracker(self, tracker: Any, frame_rgb: np.ndarray, init_bbox: Tuple[float, float, float, float]):
        init_info = {
            "init_bbox": list(init_bbox),
            "object_ids": [1],
        }
        tracker.initialize(frame_rgb, init_info)

    def _track_frame(self, tracker: Any, frame_rgb: np.ndarray) -> _TrackerOutputs:
        out = tracker.track(frame_rgb, {})
        bbox = out.get("target_bbox") if isinstance(out, dict) else None
        score = out.get("object_presence_score") if isinstance(out, dict) else None
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            bbox = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        else:
            bbox = None
        score = float(score) if score is not None else None
        if score is not None and score < self.min_presence_score:
            return _TrackerOutputs(None, score)
        return _TrackerOutputs(bbox, score)

    def _create_params(self) -> Any:
        param_module = importlib.import_module(self._param_module_path)
        params = param_module.parameters()

        wanted_device = self.device_preference
        if wanted_device.startswith("cuda"):
            if torch.cuda.is_available():
                params.use_gpu = True
                params.device = wanted_device
            elif self.force_cpu_if_no_cuda:
                params.use_gpu = False
                params.device = "cpu"
            else:
                raise RuntimeError("CUDA requested for ToMP tracker but no compatible GPU is available.")
        else:
            params.use_gpu = False
            params.device = "cpu"

        for key, value in self._param_overrides.items():
            try:
                setattr(params, key, value)
            except Exception as ex:
                raise RuntimeError(f"Failed to set TrackerParams override '{key}' -> {value!r}: {ex}") from ex

        if self._custom_checkpoint:
            net = getattr(params, "net", None)
            if net is None:
                raise RuntimeError("Tracker parameters do not expose a 'net' attribute to replace checkpoint.")
            setattr(net, "net_path", self._custom_checkpoint)

        return params

    def _build_tracker(self) -> Any:
        params = self._create_params()
        return _PyTrackingToMP(params)

    def predict(self, video_path: str) -> List[FramePrediction]:
        if cv2 is None:
            detail = f" (import error: {_CV2_IMPORT_ERROR!r})" if '_CV2_IMPORT_ERROR' in globals() else ""
            raise RuntimeError("OpenCV is required for ToMP tracker." + detail)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        init_bbox = _first_gt_bbox(video_path)
        if not _is_valid_bbox(init_bbox):
            cap.release()
            raise RuntimeError(
                "ToMP requires a valid initial ground-truth bounding box. "
                f"No usable bbox found next to '{video_path}'."
            )

        tracker = self._build_tracker()

        preds: List[FramePrediction] = []
        frame_idx = 0
        last_bbox: Optional[Tuple[float, float, float, float]] = None

        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                frame_rgb = self._apply_preprocs(frame_bgr)

                if frame_idx == 0:
                    self._initialise_tracker(tracker, frame_rgb, init_bbox)  # type: ignore[arg-type]
                    preds.append(FramePrediction(frame_idx, init_bbox, 1.0))
                    last_bbox = init_bbox
                else:
                    out = self._track_frame(tracker, frame_rgb)
                    if _is_valid_bbox(out.bbox):
                        preds.append(FramePrediction(frame_idx, out.bbox, out.score))
                        last_bbox = out.bbox
                    elif self.fallback_last_prediction and last_bbox is not None:
                        preds.append(FramePrediction(frame_idx, last_bbox, out.score))

                frame_idx += 1
        finally:
            cap.release()

        return preds
