from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.registry import register_segmentation_model

try:  # pragma: no cover - optional dependency
    from segment_anything import SamPredictor, sam_model_registry  # type: ignore[import]
except Exception as exc:  # pragma: no cover
    SamPredictor = None  # type: ignore
    sam_model_registry = {}  # type: ignore
    _SAM_IMPORT_ERROR = exc
else:
    _SAM_IMPORT_ERROR = None


@register_segmentation_model("medsam")
class MedSAMSegmenter(nn.Module):
    """Wrapper around MedSAM/SAM checkpoints via the segment-anything predictor.

    Supports both inference-only mode (default) and fine-tuning of the mask decoder.
    When train_enabled=True, only the mask decoder is fine-tuned while image encoder
    stays frozen.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__()
        if SamPredictor is None or not sam_model_registry:  # pragma: no cover - runtime guard
            raise ImportError(
                "segment-anything is required for 'medsam'. Install via"
                " `pip install git+https://github.com/facebookresearch/segment-anything.git`"
                f". Original error: {_SAM_IMPORT_ERROR}"
            )
        cfg = dict(params or {})
        checkpoint = (
            cfg.get("checkpoint")
            or cfg.get("weights")
            or cfg.get("path")
        )
        if not checkpoint:
            raise ValueError("MedSAM requires 'checkpoint' in model.params, pointing to medsam_vit_*.pth")
        checkpoint_path = os.path.abspath(os.path.expanduser(str(checkpoint)))
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"MedSAM checkpoint not found: {checkpoint_path}")
        model_type = str(cfg.get("model_type", "vit_b")).lower()
        if model_type not in sam_model_registry:
            raise KeyError(
                f"Unknown SAM backbone '{model_type}'. Available: {', '.join(sorted(sam_model_registry.keys()))}"
            )
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.predictor: Optional[Any] = SamPredictor(self.sam)
        # Basic parameters
        self.multimask_output = bool(cfg.get("multimask_output", False))
        self.use_box_prompt = bool(cfg.get("use_box_prompt", True))
        self.logit_eps = float(cfg.get("logit_eps", 1e-4))
        # Training mode: if True, fine-tune the mask decoder
        self.train_enabled = bool(cfg.get("train_enabled", False))
        self._pending_prompts: Optional[List[Dict[str, Any]]] = None
        # Setup for training/inference mode
        self._setup_training_mode()

    def _setup_training_mode(self) -> None:
        """Configure the model for training or inference mode."""
        if self.train_enabled:
            # Freeze image encoder and prompt encoder, only train mask decoder
            self.sam.image_encoder.eval()
            for param in self.sam.image_encoder.parameters():
                param.requires_grad = False
            self.sam.prompt_encoder.eval()
            for param in self.sam.prompt_encoder.parameters():
                param.requires_grad = False
            # Enable training for mask decoder
            self.sam.mask_decoder.train()
            for param in self.sam.mask_decoder.parameters():
                param.requires_grad = True
        else:
            # Pure inference mode - freeze everything
            self.sam.eval()
            for param in self.sam.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):  # type: ignore[override]
        """Override train() to control which parts are trainable."""
        super().train(mode)
        if self.train_enabled and mode:
            # Keep encoder frozen even in train mode
            self.sam.image_encoder.eval()
            self.sam.prompt_encoder.eval()
            self.sam.mask_decoder.train()
        else:
            self.sam.eval()
        return self

    def _extract_box_array(
        self,
        prompt: Optional[Dict[str, Any]],
        width: int,
        height: int,
    ) -> Optional[np.ndarray]:
        """Return a (N,4) numpy array describing box prompts."""
        box = None
        if prompt and prompt.get("box") is not None:
            box = np.asarray(prompt["box"], dtype=np.float32)
            if box.ndim == 1:
                if box.size != 4:
                    raise ValueError("MedSAM box prompt must have 4 values")
                box = box.reshape(1, 4)
            elif box.shape[-1] != 4:
                raise ValueError("MedSAM box prompt entries must have 4 values")
        if box is None and self.use_box_prompt:
            box = np.array(
                [[0.0, 0.0, float(width - 1), float(height - 1)]],
                dtype=np.float32,
            )
        return box

    def _extract_point_arrays(
        self,
        prompt: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return (coords, labels) numpy arrays for point prompts if present."""
        if not prompt:
            return None, None

        coords_data: Optional[Any] = None
        labels_data: Optional[Any] = None

        if "point_coords" in prompt and "point_labels" in prompt:
            coords_data = prompt.get("point_coords")
            labels_data = prompt.get("point_labels")
        elif "points" in prompt and prompt["points"]:
            coords_list: List[List[float]] = []
            labels_list: List[int] = []
            for entry in prompt["points"]:
                if isinstance(entry, dict):
                    coord = entry.get("coord") or entry.get("coords") or entry.get("point")
                    label = entry.get("label")
                else:
                    if len(entry) < 3:
                        raise ValueError("Each point entry must include x, y, label")
                    coord = entry[:2]
                    label = entry[2]
                if coord is None or label is None:
                    raise ValueError("Point prompts require both coordinates and labels")
                coords_list.append([float(coord[0]), float(coord[1])])
                labels_list.append(int(label))
            coords_data = coords_list
            labels_data = labels_list

        if coords_data is None or labels_data is None:
            return None, None

        coords = np.asarray(coords_data, dtype=np.float32)
        if coords.ndim == 1:
            if coords.size % 2 != 0:
                raise ValueError("point_coords must have pairs of (x, y)")
            coords = coords.reshape(-1, 2)
        elif coords.shape[-1] != 2:
            raise ValueError("point_coords entries must be (x, y)")

        labels = np.asarray(labels_data, dtype=np.int64).reshape(-1)
        if coords.shape[0] != labels.shape[0]:
            raise ValueError("point_coords and point_labels must have the same length")

        return coords, labels

    def _ensure_predictor(self) -> Any:
        if self.predictor is None:
            self.predictor = SamPredictor(self.sam)
        self.predictor.model = self.sam
        return self.predictor

    def to(self, *args, **kwargs):  # type: ignore[override]
        self.sam.to(*args, **kwargs)
        self.predictor = SamPredictor(self.sam)
        return super().to(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Forward pass - uses predictor for inference, direct model for training."""
        if x.ndim != 4:
            raise ValueError(f"MedSAM expects BCHW tensors; got shape {x.shape}")

        if self.train_enabled and self.training:
            return self._forward_train(x)
        else:
            return self._forward_inference(x)

    def _forward_inference(self, x: torch.Tensor) -> torch.Tensor:
        """Inference mode using SAM predictor (no gradients)."""
        predictor = self._ensure_predictor()
        outputs = []
        eps = max(1e-6, self.logit_eps)
        device = x.device
        prompts = self._pending_prompts or []
        self._pending_prompts = None

        with torch.no_grad():
            for idx, sample in enumerate(x):
                sample_np = sample.detach().cpu().numpy()
                if sample_np.shape[0] == 1:
                    sample_np = np.repeat(sample_np, 3, axis=0)
                elif sample_np.shape[0] != 3:
                    sample_np = sample_np[:3]
                roi = np.transpose(sample_np, (1, 2, 0))
                roi = np.clip(roi * 255.0, 0.0, 255.0).astype(np.uint8)
                roi_rgb = roi[..., ::-1].copy()
                predictor.set_image(roi_rgb)
                height, width = roi_rgb.shape[:2]

                prompt = prompts[idx] if idx < len(prompts) else None
                box = self._extract_box_array(prompt, width, height)
                point_coords_np, point_labels_np = self._extract_point_arrays(prompt)

                masks, scores, _ = predictor.predict(
                    point_coords=point_coords_np,
                    point_labels=point_labels_np,
                    box=box,
                    multimask_output=self.multimask_output,
                )
                if masks is None or len(masks) == 0:
                    raise RuntimeError("MedSAM predictor returned no masks")

                if len(masks) > 1 and scores is not None:
                    best_idx = int(np.argmax(scores))
                else:
                    best_idx = 0
                mask_prob = masks[best_idx].astype(np.float32)
                prob_tensor = torch.from_numpy(mask_prob)
                prob_tensor = prob_tensor.clamp(min=eps, max=1.0 - eps)
                logits = torch.log(prob_tensor / (1.0 - prob_tensor))
                outputs.append(logits.unsqueeze(0))

        if not outputs:
            raise RuntimeError("MedSAM forward produced no outputs")
        stacked = torch.stack(outputs, dim=0)
        return stacked.to(device)

    def _forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """Training mode - direct SAM model call with gradients for mask decoder."""
        batch_size = x.size(0)
        device = x.device
        outputs = []
        prompts = self._pending_prompts or []
        self._pending_prompts = None

        for idx in range(batch_size):
            sample = x[idx]
            if sample.shape[0] == 1:
                sample = sample.repeat(3, 1, 1)
            elif sample.shape[0] != 3:
                sample = sample[:3]

            img_for_sam = (sample * 255.0).clamp(0, 255)
            height, width = img_for_sam.shape[1], img_for_sam.shape[2]

            with torch.no_grad():
                img_input = self._preprocess_for_sam(img_for_sam.unsqueeze(0))
                image_embedding = self.sam.image_encoder(img_input)

            prompt = prompts[idx] if idx < len(prompts) else None
            box_arr = self._extract_box_array(prompt, width, height)
            box_torch = (
                torch.from_numpy(box_arr).to(device)
                if box_arr is not None
                else None
            )
            point_coords_np, point_labels_np = self._extract_point_arrays(prompt)
            if point_coords_np is not None:
                point_coords_t = torch.from_numpy(point_coords_np).float().to(device).unsqueeze(0)
                point_labels_t = torch.from_numpy(point_labels_np).long().to(device).unsqueeze(0)
                point_tuple = (point_coords_t, point_labels_t)
            else:
                point_tuple = None

            with torch.no_grad():
                sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                    points=point_tuple,
                    boxes=box_torch,
                    masks=None,
                )

            low_res_masks, iou_predictions = self.sam.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=self.multimask_output,
            )

            masks = F.interpolate(
                low_res_masks,
                (height, width),
                mode="bilinear",
                align_corners=False,
            )

            if masks.shape[1] > 1:
                best_idx = int(torch.argmax(iou_predictions[0]).item())
                mask_logits = masks[0, best_idx:best_idx + 1]
            else:
                mask_logits = masks[0]

            outputs.append(mask_logits)

        stacked = torch.stack(outputs, dim=0)
        return stacked

    def _preprocess_for_sam(self, img: torch.Tensor) -> torch.Tensor:
        """Preprocess image for SAM model input."""
        target_size = 1024
        device = img.device

        img_resized = F.interpolate(
            img,
            (target_size, target_size),
            mode="bilinear",
            align_corners=False,
        )

        pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=device).view(1, 3, 1, 1)
        pixel_std = torch.tensor([58.395, 57.12, 57.375], device=device).view(1, 3, 1, 1)

        img_normalized = (img_resized - pixel_mean) / pixel_std
        return img_normalized

    def set_prompts(self, prompts: Optional[List[Dict[str, Any]]]) -> None:
        """Set prompts for the next forward pass."""
        if prompts is None:
            self._pending_prompts = None
        else:
            self._pending_prompts = list(prompts)

    def get_trainable_parameters(self):
        """Return only the trainable parameters (mask decoder)."""
        if self.train_enabled:
            return self.sam.mask_decoder.parameters()
        else:
            return iter([])

    def save_finetuned(self, path: str) -> None:
        """Save the fine-tuned mask decoder weights."""
        torch.save({
            "mask_decoder": self.sam.mask_decoder.state_dict(),
            "model_type": "vit_b",
        }, path)

    def load_finetuned(self, path: str) -> None:
        """Load fine-tuned mask decoder weights."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fine-tuned weights not found: {path}")
        state = torch.load(path, map_location="cpu")
        if "mask_decoder" in state:
            self.sam.mask_decoder.load_state_dict(state["mask_decoder"])
        else:
            self.sam.mask_decoder.load_state_dict(state)


__all__ = ["MedSAMSegmenter"]
