from __future__ import annotations

import importlib
import os
import random
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
    from pytracking import TensorDict  # type: ignore
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
from ..utils.init_bbox import detect_bbox_on_frame, resolve_first_frame_bbox
from ..utils.annotations import load_coco_vid


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
        "train_epochs": 1,
        "train_lr": 2e-4,
        "train_weight_decay": 1e-4,
        "train_max_samples": 0,
        "train_log_interval": 10,
        "train_grad_clip": 0.1,
        "train_freeze_backbone": False,
        "train_template_update": "static",
        "first_frame_source": "gt",
        "first_frame_fallback": "gt",
        "init_detector_weights": "best.pt",
        "init_detector_conf": 0.25,
        "init_detector_iou": 0.5,
        "init_detector_imgsz": 640,
        "init_detector_device": "auto",
        "init_detector_classes": None,
        "init_detector_max_det": 50,
        "low_confidence_reinit": {
            "enabled": False,
            "threshold": 0.3,
            "min_interval": 15,
            "detector": {},
            "detector_min_conf": None,
        },
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

        merged = {**self.DEFAULT_CONFIG, **(config or {})}

        self.parameter_name = str(merged.get("parameter", self.DEFAULT_CONFIG["parameter"]))
        self.device_preference = str(merged.get("device", self.DEFAULT_CONFIG["device"])).lower()
        self.force_cpu_if_no_cuda = bool(merged.get("force_cpu_if_no_cuda", self.DEFAULT_CONFIG["force_cpu_if_no_cuda"]))
        self.fallback_last_prediction = bool(merged.get("fallback_last_prediction", self.DEFAULT_CONFIG["fallback_last_prediction"]))
        self.min_presence_score = float(merged.get("min_presence_score", self.DEFAULT_CONFIG["min_presence_score"]))
        self._param_overrides = dict(merged.get("param_overrides", self.DEFAULT_CONFIG["param_overrides"]))
        self._custom_checkpoint: Optional[str] = None

        fine_tune_cfg = merged.get("fine_tune", self.DEFAULT_CONFIG["fine_tune"])
        if isinstance(fine_tune_cfg, bool):
            fine_tune_cfg = {"enabled": bool(fine_tune_cfg)}
        self._fine_tune_enabled = bool(fine_tune_cfg.get("enabled", False))
        self._fine_tune_command: Optional[Union[str, Sequence[str]]] = fine_tune_cfg.get("command")
        self._fine_tune_cwd: Optional[str] = fine_tune_cfg.get("cwd")
        self._fine_tune_checkpoint_template: Optional[str] = fine_tune_cfg.get("checkpoint")
        env_cfg = fine_tune_cfg.get("env", {}) or {}
        self._fine_tune_env: Dict[str, str] = {str(k): str(v) for k, v in env_cfg.items()}

        self.train_epochs = max(1, int(merged.get("train_epochs", self.DEFAULT_CONFIG["train_epochs"])))
        lr_value = merged.get("train_lr", self.DEFAULT_CONFIG["train_lr"])
        self.train_lr = None if lr_value in (None, "none") else float(lr_value)
        wd_value = merged.get("train_weight_decay", self.DEFAULT_CONFIG["train_weight_decay"])
        self.train_weight_decay = None if wd_value in (None, "none") else float(wd_value)
        self.train_max_samples = max(0, int(merged.get("train_max_samples", self.DEFAULT_CONFIG["train_max_samples"])))
        self.train_log_interval = max(1, int(merged.get("train_log_interval", self.DEFAULT_CONFIG["train_log_interval"])))
        self.train_grad_clip = max(0.0, float(merged.get("train_grad_clip", self.DEFAULT_CONFIG["train_grad_clip"])))
        self.train_freeze_backbone = bool(merged.get("train_freeze_backbone", self.DEFAULT_CONFIG["train_freeze_backbone"]))
        self.train_template_update = str(merged.get("train_template_update", self.DEFAULT_CONFIG["train_template_update"]) or "static").lower()
        self._template_interval = None
        if self.train_template_update.startswith("interval"):
            parts = self.train_template_update.split(":", 1)
            if len(parts) == 2 and parts[1].strip().isdigit():
                self._template_interval = max(1, int(parts[1].strip()))
            else:
                self._template_interval = 1
            self.train_template_update = "interval"

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

        self.first_frame_source = str(merged.get("first_frame_source", self.DEFAULT_CONFIG["first_frame_source"]) or "gt").lower()
        raw_fallback = merged.get("first_frame_fallback", self.DEFAULT_CONFIG["first_frame_fallback"])
        if raw_fallback is None:
            self.first_frame_fallback: Optional[str] = None
        else:
            fb = str(raw_fallback).strip().lower()
            self.first_frame_fallback = fb if fb not in ("", "none", "null") else None
        self.init_detector_params = {
            "weights": str(merged.get("init_detector_weights", self.DEFAULT_CONFIG["init_detector_weights"])),
            "conf": float(merged.get("init_detector_conf", self.DEFAULT_CONFIG["init_detector_conf"])),
            "iou": float(merged.get("init_detector_iou", self.DEFAULT_CONFIG["init_detector_iou"])),
            "imgsz": int(merged.get("init_detector_imgsz", self.DEFAULT_CONFIG["init_detector_imgsz"])),
            "device": str(merged.get("init_detector_device", self.DEFAULT_CONFIG["init_detector_device"])),
            "classes": merged.get("init_detector_classes", self.DEFAULT_CONFIG["init_detector_classes"]),
            "max_det": int(merged.get("init_detector_max_det", self.DEFAULT_CONFIG["init_detector_max_det"])),
        }

        lc_default = self.DEFAULT_CONFIG["low_confidence_reinit"]
        lc_cfg = merged.get("low_confidence_reinit", lc_default)
        if isinstance(lc_cfg, bool):
            lc_cfg = {"enabled": bool(lc_cfg)}
        lc_cfg = dict(lc_cfg or {})
        self.low_conf_reinit_enabled = bool(lc_cfg.get("enabled", lc_default["enabled"]))
        self.low_conf_threshold = float(lc_cfg.get("threshold", lc_default["threshold"]))
        self.low_conf_min_interval = max(1, int(lc_cfg.get("min_interval", lc_default["min_interval"])))
        detector_override = lc_cfg.get("detector") or {}
        if not isinstance(detector_override, dict):
            detector_override = {}
        self.low_conf_detector_params = {**self.init_detector_params, **detector_override}
        min_conf_val = lc_cfg.get("detector_min_conf", lc_default.get("detector_min_conf"))
        self.low_conf_detector_min_conf = None if min_conf_val in (None, "none", "") else float(min_conf_val)

        self._finetuned_state_dict = None
        self._finetuned_summary = None

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
            return self._run_internal_finetune(train_dataset, seed, formatted_output_dir)

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

    def _run_internal_finetune(self, train_dataset, seed: int, output_dir: Path):
        if torch is None:
            detail = f" (import error: {_TORCH_IMPORT_ERROR!r})" if '_TORCH_IMPORT_ERROR' in globals() else ""
            raise RuntimeError("PyTorch is required for TaMOs fine-tuning." + detail)
        if cv2 is None:
            detail = f" (import error: {_CV2_IMPORT_ERROR!r})" if '_CV2_IMPORT_ERROR' in globals() else ""
            raise RuntimeError("OpenCV is required for TaMOs fine-tuning." + detail)

        dataset_len = len(train_dataset) if hasattr(train_dataset, "__len__") else None
        if not dataset_len:
            return {"status": "no_data", "reason": "empty_dataset"}

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        params = self._create_params()
        net_wrapper = getattr(params, "net", None)
        if net_wrapper is None:
            raise RuntimeError("Tracker parameters do not expose a 'net' attribute for TaMOs.")
        if getattr(net_wrapper, "net", None) is None:
            net_wrapper.initialize()
        net = net_wrapper.net

        use_gpu = bool(getattr(params, "use_gpu", False)) and torch.cuda.is_available()
        if use_gpu:
            device_name = getattr(params, "device", "cuda")
            device = torch.device(device_name if torch.cuda.is_available() else "cuda")
        else:
            device = torch.device("cpu")
            setattr(params, "use_gpu", False)
        net = net.to(device)
        net.train()

        if self.train_freeze_backbone and hasattr(net, "feature_extractor"):
            for param in net.feature_extractor.parameters():
                param.requires_grad = False

        trainable_params = [p for p in net.parameters() if p.requires_grad]
        if not trainable_params:
            trainable_params = list(net.parameters())

        lr = float(self.train_lr) if self.train_lr is not None else 2e-4
        weight_decay = float(self.train_weight_decay) if self.train_weight_decay is not None else 1e-4

        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        grad_clip = float(self.train_grad_clip)

        try:
            from torch.nn.utils import clip_grad_norm_  # type: ignore
            import ltr.data.transforms as tfm  # type: ignore
            from ltr.data import processing as ltr_processing  # type: ignore
            import ltr.models.loss as ltr_losses  # type: ignore
            from ltr.models.loss.bbr_loss import GIoULoss  # type: ignore
            from ltr.actors.tracking import TaMOsActor  # type: ignore
        except Exception as ex:  # pragma: no cover
            raise RuntimeError(
                "TaMOs fine-tuning dependencies (LTR framework) failed to import: {}".format(ex)
            ) from ex

        processor = self._build_processor(params, net, tfm, ltr_processing)
        objective = {"giou": GIoULoss(), "test_clf": ltr_losses.FocalLoss()}
        loss_weight = {"giou": 1.0, "test_clf": 100.0}
        actor = TaMOsActor(net=net, objective=objective, loss_weight=loss_weight, prob=True)

        log_file = output_dir / "train_log.txt"
        if log_file.exists():
            try:
                log_file.unlink()
            except Exception:
                pass

        total_steps = 0
        total_loss = 0.0
        total_giou = 0.0
        total_clf = 0.0

        cb = getattr(self, "progress_callback", None)
        max_samples = self.train_max_samples if self.train_max_samples > 0 else None
        template_interval = self._template_interval

        for epoch in range(self.train_epochs):
            if callable(cb):
                try:
                    cb("train_epoch_start", epoch + 1, self.train_epochs)
                except Exception:
                    pass

            for item_idx in range(dataset_len):
                try:
                    sample_item = train_dataset[item_idx]
                except Exception:
                    continue

                video_path = None
                annotation = None
                if isinstance(sample_item, dict):
                    video_path = sample_item.get("video_path") or sample_item.get("video")
                    annotation = sample_item.get("annotation")

                if not video_path or not os.path.exists(video_path):
                    continue

                frames = self._extract_annotation_frames(annotation, video_path)
                if not frames:
                    continue

                for template_frame, template_bbox, search_frame, search_bbox in self._iter_training_pairs(
                    video_path, frames, self.train_template_update, template_interval
                ):
                    batch = self._prepare_training_batch(
                        processor,
                        template_frame,
                        template_bbox,
                        search_frame,
                        search_bbox,
                        device,
                        epoch,
                    )
                    if batch is None:
                        continue

                    optimizer.zero_grad(set_to_none=True)
                    loss, stats = actor(batch)
                    if not torch.isfinite(loss):
                        continue

                    loss.backward()
                    if grad_clip > 0:
                        clip_grad_norm_(trainable_params, grad_clip)
                    optimizer.step()

                    step_loss = float(loss.detach().item())
                    total_steps += 1
                    total_loss += step_loss
                    total_giou += float(stats.get("Loss/GIoU", 0.0))
                    total_clf += float(stats.get("Loss/clf_loss_test", 0.0))

                    if total_steps % self.train_log_interval == 0:
                        msg = (
                            f"[TaMOs train] step={total_steps} loss={step_loss:.4f} "
                            f"giou={stats.get('Loss/GIoU', 0.0):.4f} "
                            f"clf={stats.get('Loss/clf_loss_test', 0.0):.4f}"
                        )
                        self._log_training_line(msg, log_file)

                    if max_samples and total_steps >= max_samples:
                        break

                if max_samples and total_steps >= max_samples:
                    break

            if callable(cb):
                try:
                    cb("train_epoch_end", epoch + 1, self.train_epochs)
                except Exception:
                    pass

            if max_samples and total_steps >= max_samples:
                break

        if total_steps == 0:
            return {"status": "no_data", "reason": "no_valid_samples"}

        avg_loss = total_loss / total_steps
        avg_giou = total_giou / total_steps
        avg_clf = total_clf / total_steps

        finetuned_state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        self._finetuned_state_dict = finetuned_state
        summary = {
            "status": "trained",
            "epochs_completed": min(self.train_epochs, epoch + 1),
            "steps": total_steps,
            "avg_loss": avg_loss,
            "avg_giou": avg_giou,
            "avg_clf": avg_clf,
            "learning_rate": lr,
            "trainable_params": len(trainable_params),
        }

        state_path = output_dir / "tamos_finetuned_state.pt"
        try:
            torch.save({"net": finetuned_state}, state_path)
            summary["checkpoint"] = str(state_path)
        except Exception:
            summary["checkpoint"] = None

        self._finetuned_summary = summary
        self._log_training_line(
            f"[TaMOs train] finished steps={total_steps} avg_loss={avg_loss:.4f}", log_file
        )

        return summary

    def _build_processor(self, params: Any, net: Any, tfm_module: Any, processing_module: Any):
        mean_tensor = getattr(params.net, "_mean", torch.tensor([0.485, 0.456, 0.406]))
        std_tensor = getattr(params.net, "_std", torch.tensor([0.229, 0.224, 0.225]))
        mean = mean_tensor.view(-1).tolist()
        std = std_tensor.view(-1).tolist()
        base_transform = tfm_module.Transform(
            tfm_module.ToTensor(),
            tfm_module.Normalize(mean=mean, std=std),
        )

        search_area_factor = float(getattr(params, "search_area_scale", 5.0))
        stride = int(getattr(params, "feature_stride", 16))
        train_feature_size = getattr(params, "train_feature_size", [24, 36])
        if isinstance(train_feature_size, (list, tuple)) and len(train_feature_size) >= 2:
            feature_h = int(train_feature_size[0])
            feature_w = int(train_feature_size[1])
        else:
            feature_h = feature_w = int(train_feature_size[0] if train_feature_size else 24)
        feature_sz = (feature_w, feature_h)

        image_sample_size = getattr(params, "image_sample_size", None)
        if isinstance(image_sample_size, (list, tuple)) and len(image_sample_size) >= 2:
            output_h = int(image_sample_size[0])
            output_w = int(image_sample_size[1])
        else:
            output_w = feature_sz[0] * stride
            output_h = feature_sz[1] * stride

        sigma_factor = (1.0 / 4.0) / max(search_area_factor, 1e-6)
        kernel_sz = getattr(params, "target_filter_sz", 1)
        predictor = getattr(getattr(net, "head", None), "filter_predictor", None)
        max_objects = int(getattr(predictor, "num_tokens", 10))
        label_params = {
            "feature_sz": feature_sz,
            "sigma_factor": sigma_factor,
            "kernel_sz": kernel_sz,
        }

        processor = processing_module.TaMOsProcessing(
            max_num_objects=max_objects,
            search_area_factor=search_area_factor,
            output_sz=(output_w, output_h),
            center_jitter_factor={"train": 0.0, "test": 0.0},
            scale_jitter_factor={"train": 0.0, "test": 0.0},
            crop_type="inside_major",
            mode="sequence",
            stride=stride,
            label_function_params=label_params,
            center_sampling_radius=1.0,
            use_normalized_coords=True,
            include_high_res_labels=True,
            enforce_one_sample_region_per_object=True,
            transform=base_transform,
            train_transform=base_transform,
            test_transform=base_transform,
            joint_transform=None,
        )
        return processor

    def _prepare_training_batch(
        self,
        processor,
        template_frame,
        template_bbox,
        search_frame,
        search_bbox,
        device,
        epoch,
    ):
        template_tensor = torch.tensor(template_bbox, dtype=torch.float32)
        search_tensor = torch.tensor(search_bbox, dtype=torch.float32)
        raw = TensorDict({
            "train_images": [template_frame],
            "train_anno": [{0: template_tensor}],
            "test_images": [search_frame],
            "test_anno": [{0: search_tensor}],
        })

        processed = processor(raw)
        return self._tensor_dict_to_actor_batch(processed, device, epoch)

    def _tensor_dict_to_actor_batch(self, processed, device, epoch):
        train_images = processed.get("train_images")
        test_images = processed.get("test_images")
        train_anno_list = processed.get("train_anno")
        test_anno_list = processed.get("test_anno")

        if train_images is None or test_images is None or not train_anno_list or not test_anno_list:
            return None

        def _stack_annos(anno_seq):
            boxes = []
            for entry in anno_seq:
                bbox = entry.get(0)
                if bbox is None or not torch.isfinite(bbox).all() or bbox[2] <= 0 or bbox[3] <= 0:
                    return None
                boxes.append(bbox)
            return torch.stack(boxes, dim=0)

        train_bb = _stack_annos(train_anno_list)
        test_bb = _stack_annos(test_anno_list)
        if train_bb is None or test_bb is None:
            return None

        if train_images.dim() == 4:
            train_images = train_images.unsqueeze(1)
        if test_images.dim() == 4:
            test_images = test_images.unsqueeze(1)

        batch: Dict[str, Any] = {
            "train_images": train_images.to(device=device, dtype=torch.float32),
            "test_images": test_images.to(device=device, dtype=torch.float32),
            "train_anno": train_bb.unsqueeze(1).to(device=device, dtype=torch.float32),
            "test_anno": test_bb.unsqueeze(1).to(device=device, dtype=torch.float32),
            "train_label": processed["train_label"].to(device=device, dtype=torch.float32),
            "train_ltrb_target": processed["train_ltrb_target"].to(device=device, dtype=torch.float32),
            "train_sample_region": processed["train_sample_region"].to(device=device, dtype=torch.float32),
            "test_label": processed["test_label"].to(device=device, dtype=torch.float32),
            "test_ltrb_target": processed["test_ltrb_target"].to(device=device, dtype=torch.float32),
            "test_sample_region": processed["test_sample_region"].to(device=device, dtype=torch.float32),
            "epoch": torch.tensor(int(epoch), dtype=torch.int64, device=device),
        }

        if "train_label_highres" in processed:
            batch["train_label_highres"] = processed["train_label_highres"].to(device=device, dtype=torch.float32)
        if "train_ltrb_target_highres" in processed:
            batch["train_ltrb_target_highres"] = processed["train_ltrb_target_highres"].to(device=device, dtype=torch.float32)
        if "train_sample_region_highres" in processed:
            batch["train_sample_region_highres"] = processed["train_sample_region_highres"].to(device=device, dtype=torch.float32)
        if "test_label_highres" in processed:
            batch["test_label_highres"] = processed["test_label_highres"].to(device=device, dtype=torch.float32)
        if "test_ltrb_target_highres" in processed:
            batch["test_ltrb_target_highres"] = processed["test_ltrb_target_highres"].to(device=device, dtype=torch.float32)
        if "test_sample_region_highres" in processed:
            batch["test_sample_region_highres"] = processed["test_sample_region_highres"].to(device=device, dtype=torch.float32)

        return batch

    def _iter_training_pairs(
        self,
        video_path: str,
        frames: Dict[int, List[Tuple[float, float, float, float]]],
        strategy: str,
        interval: Optional[int],
    ):
        valid = {
            int(idx): [bbox for bbox in boxes if self._is_valid_bbox_tuple(bbox)]
            for idx, boxes in frames.items()
        }
        idx_to_bbox = {idx: tuple(boxes[0][:4]) for idx, boxes in valid.items() if boxes}
        if len(idx_to_bbox) < 2:
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return

        template_frame = None
        template_bbox: Optional[Tuple[float, float, float, float]] = None
        template_counter = 0
        try:
            frame_idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx in idx_to_bbox:
                    bbox = idx_to_bbox[frame_idx]
                    if template_frame is None:
                        template_frame = frame.copy()
                        template_bbox = bbox
                        template_counter = 0
                    else:
                        yield template_frame.copy(), template_bbox, frame.copy(), bbox
                        template_counter += 1
                        if strategy == "update":
                            template_frame = frame.copy()
                            template_bbox = bbox
                            template_counter = 0
                        elif strategy == "interval" and interval and template_counter >= interval:
                            template_frame = frame.copy()
                            template_bbox = bbox
                            template_counter = 0
                        elif strategy == "random" and random.random() < 0.33:
                            template_frame = frame.copy()
                            template_bbox = bbox
                            template_counter = 0
                frame_idx += 1
        finally:
            cap.release()

    def _extract_annotation_frames(
        self,
        annotation: Optional[Dict[str, Any]],
        video_path: Optional[str],
    ) -> Dict[int, List[Tuple[float, float, float, float]]]:
        frames: Dict[int, List[Tuple[float, float, float, float]]] = {}

        def _normalize(source_frames: Any) -> Dict[int, List[Tuple[float, float, float, float]]]:
            normalized: Dict[int, List[Tuple[float, float, float, float]]] = {}
            if not isinstance(source_frames, dict):
                return normalized
            for key, boxes in source_frames.items():
                try:
                    idx = int(key)
                except Exception:
                    continue
                norm_boxes: List[Tuple[float, float, float, float]] = []
                iterable = boxes if isinstance(boxes, (list, tuple)) else []
                for box in iterable:
                    if isinstance(box, (list, tuple)) and len(box) >= 4:
                        bbox = tuple(float(v) for v in box[:4])
                        if self._is_valid_bbox_tuple(bbox):
                            norm_boxes.append(bbox)
                if norm_boxes:
                    normalized[idx] = norm_boxes
            return normalized

        if isinstance(annotation, dict):
            if "frames" in annotation:
                frames = _normalize(annotation["frames"])
            elif "raw" in annotation and isinstance(annotation["raw"], dict) and "frames" in annotation["raw"]:
                frames = _normalize(annotation["raw"]["frames"])
            elif "annotation_path" in annotation and isinstance(annotation["annotation_path"], str):
                ann_path = annotation["annotation_path"]
                if os.path.exists(ann_path):
                    try:
                        loaded = load_coco_vid(ann_path)
                        frames = _normalize(loaded.get("frames", {}))
                    except Exception:
                        frames = {}
        elif isinstance(annotation, str) and os.path.exists(annotation):
            try:
                loaded = load_coco_vid(annotation)
                frames = _normalize(loaded.get("frames", {}))
            except Exception:
                frames = {}

        if not frames and video_path:
            candidate = Path(video_path).with_suffix(".json")
            if candidate.exists():
                try:
                    loaded = load_coco_vid(str(candidate))
                    frames = _normalize(loaded.get("frames", {}))
                except Exception:
                    frames = {}

        return frames

    def _log_training_line(self, message: str, log_file: Path) -> None:
        logger = getattr(self, "logger", None)
        if logger is not None and hasattr(logger, "info"):
            try:
                logger.info(message)
            except Exception:
                print(message)
        else:
            print(message)
        if log_file:
            try:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(message + "\n")
            except Exception:
                pass

    @staticmethod
    def _is_valid_bbox_tuple(bbox: Tuple[float, float, float, float]) -> bool:
        if bbox is None or len(bbox) < 4:
            return False
        x, y, w, h = bbox[:4]
        return np.isfinite([x, y, w, h]).all() and w > 0 and h > 0

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
        tracker = _PyTrackingTaMOs(params)
        finetuned_state = getattr(self, "_finetuned_state_dict", None)
        if finetuned_state:
            try:
                params_obj = getattr(tracker, "params", None)
                net_wrapper = getattr(params_obj, "net", None) if params_obj is not None else None
                if net_wrapper is not None:
                    if getattr(net_wrapper, "net", None) is None:
                        net_wrapper.initialize()
                    net_wrapper.net.load_state_dict(finetuned_state, strict=False)
            except Exception as ex:
                warn_msg = f"TaMOs: failed to load fine-tuned weights, falling back to checkpoint: {ex}"
                logger = getattr(self, "logger", None)
                if logger is not None and hasattr(logger, "warning"):
                    try:
                        logger.warning(warn_msg)
                    except Exception:
                        print(warn_msg)
                else:
                    print(warn_msg)
        return tracker

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
        last_detector_reinit = -10**9

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

                    should_reinit = (
                        self.low_conf_reinit_enabled
                        and (score_val is None or score_val < self.low_conf_threshold)
                        and (frame_idx - last_detector_reinit) >= self.low_conf_min_interval
                    )

                    if should_reinit:
                        det_bbox, det_conf = detect_bbox_on_frame(
                            frame_bgr,
                            self.low_conf_detector_params,
                            self.low_conf_detector_min_conf,
                        )
                        if det_bbox is not None:
                            tracker = self._build_tracker()
                            tracker.initialize(frame_rgb, {"init_bbox": [float(v) for v in det_bbox]})
                            last_detector_reinit = frame_idx
                            last_bbox = det_bbox
                            conf_out = det_conf if det_conf is not None else score_val
                            if conf_out is not None:
                                conf_out = float(max(0.0, min(1.0, conf_out)))
                            preds.append(FramePrediction(frame_idx, det_bbox, conf_out))
                            frame_idx += 1
                            continue

                    if _is_valid_bbox(bbox_tuple):
                        preds.append(FramePrediction(frame_idx, bbox_tuple, score_val))
                        last_bbox = bbox_tuple
                    elif self.fallback_last_prediction and last_bbox is not None:
                        fallback_score = score_val if score_val is not None else 0.0
                        preds.append(
                            FramePrediction(
                                frame_index=frame_idx,
                                bbox=last_bbox,
                                score=fallback_score,
                                is_fallback=True,
                            )
                        )

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