from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import cv2  # type: ignore
except Exception as ex:  # pragma: no cover - surface clear error when OpenCV missing
    cv2 = None  # type: ignore
    _CV2_IMPORT_ERROR = ex

try:
    import torch
except Exception as ex:  # pragma: no cover - keep torch import error explicit
    torch = None  # type: ignore
    _TORCH_IMPORT_ERROR = ex

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PYTRACKING_ROOT = _PROJECT_ROOT / "libs" / "pytracking"
if _PYTRACKING_ROOT.is_dir() and str(_PYTRACKING_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTRACKING_ROOT))

try:  # pragma: no cover - pytracking imports validated at runtime
    from pytracking.tracker.tamos import TaMOs as _PyTrackingTaMOs  # type: ignore
    from pytracking.utils.params import TrackerParams as _TrackerParams  # type: ignore
    from pytracking.evaluation.environment import env_settings as _env_settings  # type: ignore
except Exception as ex:  # pragma: no cover
    _PyTrackingTaMOs = None  # type: ignore
    _TrackerParams = None  # type: ignore
    _env_settings = None  # type: ignore
    _PYTRACKING_IMPORT_ERROR = ex

from ..core.interfaces import FramePrediction, PreprocessingModule, TrackingModel
from ..core.registry import register_model
from ..utils.init_bbox import resolve_first_frame_bbox


def _resolve_param_module(name: str):
    preset = (name or "").strip().lower() or "tamos_resnet50"
    aliases = {
        "tamos": "pytracking.parameter.tamos.tamos_resnet50",
        "tamos50": "pytracking.parameter.tamos.tamos_resnet50",
        "tamos_resnet50": "pytracking.parameter.tamos.tamos_resnet50",
        "tamos-resnet50": "pytracking.parameter.tamos.tamos_resnet50",
        "tamos_swin_base": "pytracking.parameter.tamos.tamos_swin_base",
        "tamos-swin-base": "pytracking.parameter.tamos.tamos_swin_base",
        "tamos_swin": "pytracking.parameter.tamos.tamos_swin_base",
    }
    module_path = aliases.get(preset, name)
    try:
        return importlib.import_module(module_path)
    except Exception as ex:  # pragma: no cover - rely on runtime
        raise RuntimeError(f"Failed to import TaMOs parameter module '{module_path}': {ex}") from ex


def _is_valid_bbox(bbox: Optional[Tuple[float, float, float, float]]) -> bool:
    if bbox is None:
        return False
    x, y, w, h = bbox
    return all(np.isfinite([x, y, w, h])) and w > 0 and h > 0


@register_model("TaMOs")
class TaMOsTracker(TrackingModel):
    """Expose the TaMOs tracker from the bundled pytracking checkout."""

    name = "TaMOs"

    DEFAULT_CONFIG: Dict[str, Any] = {
        "parameter": "tamos_resnet50",
        "device": "cuda",
        "force_cpu_if_no_cuda": True,
        "fallback_last_prediction": True,
        "min_presence_score": 0.0,
        "param_overrides": {},
        "fine_tune": {
            "enabled": False,
            "command": None,
            "cwd": None,
            "checkpoint": None,
            "env": {},
        },
        "first_frame_source": "gt",
        "first_frame_fallback": "gt",
        "init_detector_weights": "best.pt",
        "init_detector_conf": 0.25,
        "init_detector_iou": 0.5,
        "init_detector_imgsz": 640,
        "init_detector_device": "auto",
        "init_detector_classes": None,
        "init_detector_max_det": 50,
    }

    def __init__(self, config: Dict[str, Any]):
        if cv2 is None:
            detail = f" (import error: {_CV2_IMPORT_ERROR!r})" if '_CV2_IMPORT_ERROR' in globals() else ""
            raise RuntimeError("OpenCV is required for TaMOs tracker." + detail)
        if torch is None:
            detail = f" (import error: {_TORCH_IMPORT_ERROR!r})" if '_TORCH_IMPORT_ERROR' in globals() else ""
            raise RuntimeError("PyTorch is required for TaMOs tracker." + detail)
        if _PyTrackingTaMOs is None or _TrackerParams is None:
            detail = f" (import error: {_PYTRACKING_IMPORT_ERROR!r})" if '_PYTRACKING_IMPORT_ERROR' in globals() else ""
            raise RuntimeError("Bundled pytracking package is required for TaMOs tracker." + detail)

        self.parameter_name = str(config.get("parameter", self.DEFAULT_CONFIG["parameter"]))
        self.device_preference = str(config.get("device", self.DEFAULT_CONFIG["device"])).lower()
        self.force_cpu_if_no_cuda = bool(config.get("force_cpu_if_no_cuda", self.DEFAULT_CONFIG["force_cpu_if_no_cuda"]))
        self.fallback_last_prediction = bool(config.get("fallback_last_prediction", self.DEFAULT_CONFIG["fallback_last_prediction"]))
        self.min_presence_score = float(config.get("min_presence_score", self.DEFAULT_CONFIG["min_presence_score"]))
        self._param_overrides = dict(config.get("param_overrides", self.DEFAULT_CONFIG["param_overrides"]))
        self._custom_checkpoint: Optional[str] = None

        fine_tune_cfg = config.get("fine_tune", self.DEFAULT_CONFIG["fine_tune"])
        if isinstance(fine_tune_cfg, bool):
            fine_tune_cfg = {"enabled": bool(fine_tune_cfg)}
        self._fine_tune_enabled = bool(fine_tune_cfg.get("enabled", False))
        self._fine_tune_command: Optional[Union[str, Sequence[str]]] = fine_tune_cfg.get("command")
        self._fine_tune_cwd: Optional[str] = fine_tune_cfg.get("cwd")
        self._fine_tune_checkpoint_template: Optional[str] = fine_tune_cfg.get("checkpoint")
        env_cfg = fine_tune_cfg.get("env", {}) or {}
        self._fine_tune_env: Dict[str, str] = {str(k): str(v) for k, v in env_cfg.items()}

        param_module = _resolve_param_module(self.parameter_name)
        self._param_module_path = param_module.__name__

        if _env_settings is not None:
            env_cfg = _env_settings()
            net_root = Path(getattr(env_cfg, "network_path", ""))
            if net_root and not net_root.exists():
                raise RuntimeError(
                    "pytracking network_path does not exist: "
                    f"'{net_root}'. Place TaMOs weights (e.g. tamos_resnet50.pth.tar) there or update local.py."
                )

        # Construct params early so override errors surface fast
        self._create_params()

        self.preprocs: List[PreprocessingModule] = []

        self.first_frame_source = str(config.get("first_frame_source", self.DEFAULT_CONFIG["first_frame_source"]) or "gt").lower()
        raw_fallback = config.get("first_frame_fallback", self.DEFAULT_CONFIG["first_frame_fallback"])
        if raw_fallback is None:
            self.first_frame_fallback: Optional[str] = None
        else:
            fb = str(raw_fallback).strip().lower()
            self.first_frame_fallback = fb if fb not in ("", "none", "null") else None
        self.init_detector_params = {
            "weights": str(config.get("init_detector_weights", self.DEFAULT_CONFIG["init_detector_weights"])),
            "conf": float(config.get("init_detector_conf", self.DEFAULT_CONFIG["init_detector_conf"])),
            "iou": float(config.get("init_detector_iou", self.DEFAULT_CONFIG["init_detector_iou"])),
            "imgsz": int(config.get("init_detector_imgsz", self.DEFAULT_CONFIG["init_detector_imgsz"])),
            "device": str(config.get("init_detector_device", self.DEFAULT_CONFIG["init_detector_device"])),
            "classes": config.get("init_detector_classes", self.DEFAULT_CONFIG["init_detector_classes"]),
            "max_det": int(config.get("init_detector_max_det", self.DEFAULT_CONFIG["init_detector_max_det"])),
        }

    # ------------------------------------------------------------------
    # TrackingModel API
    # ------------------------------------------------------------------
    def train(self, train_dataset, val_dataset=None, seed: int = 0, output_dir: Optional[str] = None):
        if not self._fine_tune_enabled:
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

        formatted_output_dir = Path(output_dir or Path.cwd() / "finetune_tamos")
        formatted_output_dir.mkdir(parents=True, exist_ok=True)

        command = self._format_fine_tune_command(self._fine_tune_command, formatted_output_dir)
        if command is None:
            raise RuntimeError(
                "TaMOs fine-tune enabled but no command provided. Set fine_tune.command to a shell string or list."
            )

        shell = isinstance(command, str)
        cmd_display = command if shell else " ".join(command)
        cwd = Path(self._fine_tune_cwd or _PYTRACKING_ROOT)
        env = os.environ.copy()
        env.update(self._fine_tune_env)

        cb = getattr(self, "progress_callback", None)
        if callable(cb):
            try:
                cb("train_epoch_start", 1, 1)
            except Exception:
                pass

        try:
            subprocess.run(
                command,
                shell=shell,
                check=True,
                cwd=str(cwd),
                env=env,
            )
        except subprocess.CalledProcessError as ex:
            raise RuntimeError(f"TaMOs fine-tune command failed with exit code {ex.returncode}: {cmd_display}") from ex

        checkpoint_template = self._fine_tune_checkpoint_template
        checkpoint_path: Optional[Path] = None
        if checkpoint_template:
            checkpoint_resolved = self._format_template(checkpoint_template, formatted_output_dir)
            checkpoint_path = self._resolve_checkpoint_path(checkpoint_resolved, formatted_output_dir)
            if not checkpoint_path.exists():
                raise RuntimeError(
                    f"TaMOs fine-tune expected checkpoint at '{checkpoint_path}', but file was not found."
                )
            self._custom_checkpoint = str(checkpoint_path)
            self._cached_params = None
            self._create_params()

        if callable(cb):
            try:
                cb("train_epoch_end", 1, 1)
            except Exception:
                pass

        return {
            "status": "external_fine_tune",
            "command": cmd_display,
            "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        }

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
                raise RuntimeError("CUDA requested for TaMOs tracker but no compatible GPU is available.")
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

        self._cached_params = params
        return params

    def _build_tracker(self) -> Any:
        params = getattr(self, "_cached_params", None) or self._create_params()
        return _PyTrackingTaMOs(params)

    def predict(self, video_path: str) -> List[FramePrediction]:
        if cv2 is None:
            detail = f" (import error: {_CV2_IMPORT_ERROR!r})" if '_CV2_IMPORT_ERROR' in globals() else ""
            raise RuntimeError("OpenCV is required for TaMOs tracker." + detail)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        primary_init = resolve_first_frame_bbox(
            video_path,
            mode=self.first_frame_source,
            detector=self.init_detector_params,
            fallback=self.first_frame_fallback,
        )
        if not _is_valid_bbox(primary_init):
            cap.release()
            raise RuntimeError(
                "TaMOs requires a valid first-frame bounding box. "
                f"Failed to obtain one (mode={self.first_frame_source}) for '{video_path}'."
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
                    init_info = {"init_bbox": list(primary_init)}
                    tracker.initialize(frame_rgb, init_info)
                    preds.append(FramePrediction(frame_idx, primary_init, 1.0))
                    last_bbox = primary_init
                else:
                    raw_out = tracker.track(frame_rgb)
                    bbox_data = raw_out.get("target_bbox") if isinstance(raw_out, dict) else None
                    score_data = raw_out.get("object_presence_score") if isinstance(raw_out, dict) else None

                    if isinstance(bbox_data, dict):
                        # multi-object mode - use first key deterministically
                        if bbox_data:
                            first_key = sorted(bbox_data.keys())[0]
                            bbox_data = bbox_data[first_key]
                            if isinstance(score_data, dict):
                                score_data = score_data.get(first_key)
                        else:
                            bbox_data = None

                    bbox_tuple: Optional[Tuple[float, float, float, float]] = None
                    if bbox_data is not None:
                        if isinstance(bbox_data, torch.Tensor):
                            bbox_data = bbox_data.detach().cpu().numpy().tolist()
                        bbox_tuple = tuple(float(v) for v in list(bbox_data)[:4])

                    score_val = None
                    if score_data is not None:
                        if isinstance(score_data, torch.Tensor):
                            score_data = score_data.detach().cpu().item()
                        score_val = float(score_data)

                    if score_val is not None and score_val < self.min_presence_score:
                        bbox_tuple = None

                    if _is_valid_bbox(bbox_tuple):
                        preds.append(FramePrediction(frame_idx, bbox_tuple, score_val))
                        last_bbox = bbox_tuple
                    elif self.fallback_last_prediction and last_bbox is not None:
                        preds.append(FramePrediction(frame_idx, last_bbox, score_val))

                frame_idx += 1
        finally:
            cap.release()

        return preds

    # ------------------------------------------------------------------
    # Fine-tune helpers
    # ------------------------------------------------------------------
    def _format_fine_tune_command(
        self,
        command: Optional[Union[str, Sequence[str]]],
        output_dir: Path,
    ) -> Optional[Union[str, List[str]]]:
        if command is None:
            return None
        fmt_kwargs = {"output_dir": str(output_dir)}
        if isinstance(command, str):
            return command.format(**fmt_kwargs)
        if isinstance(command, Iterable):
            formatted: List[str] = []
            for part in command:
                formatted.append(str(part).format(**fmt_kwargs))
            return formatted
        raise TypeError("fine_tune.command must be a string or iterable of strings")

    def _format_template(self, template: str, output_dir: Path) -> str:
        fmt_kwargs = {"output_dir": str(output_dir)}
        return template.format(**fmt_kwargs)

    def _resolve_checkpoint_path(self, candidate: str, output_dir: Optional[Path]) -> Path:
        path = Path(candidate)
        if path.is_absolute():
            return path
        search_order: List[Path] = []
        if output_dir is not None:
            search_order.append(output_dir)
        search_order.append(Path.cwd())
        if _PYTRACKING_ROOT.exists():
            search_order.append(_PYTRACKING_ROOT)
        if _env_settings is not None:
            env_cfg = _env_settings()
            network_path = getattr(env_cfg, "network_path", "")
            if isinstance(network_path, (list, tuple)):
                search_order.extend(Path(p) for p in network_path)
            elif network_path:
                search_order.append(Path(network_path))
        for base in search_order:
            resolved = base / candidate
            if resolved.exists():
                return resolved
        return Path(candidate)