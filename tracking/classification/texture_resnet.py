"""ResNet-18 based texture feature extraction from segmentation-masked ROIs.

Pipeline per frame
------------------
1. Read video frame → crop bounding-box ROI
2. If segmentation mask available → apply mask (zero-out background)
3. Resize to 224 × 224 → replicate grayscale to 3 channels
4. ImageNet normalisation → forward through truncated ResNet-18 (avgpool) → 512-D embedding

The module provides :class:`MaskedROIResNetExtractor` which lazy-loads the
model on first use.  A graceful fallback to zero vectors is provided when
``torch`` / ``torchvision`` is unavailable.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional, Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)

RESNET_FEAT_DIM: int = 512


def _apply_preprocs_frame_like_segmentation(frame: np.ndarray, preprocs: Sequence) -> np.ndarray:
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

# ── Dependency checks ────────────────────────────────────────────────────────
_TORCH_OK = False
_TV_OK = False
_TORCH_IMPORT_ERROR: Optional[BaseException] = None
_TV_IMPORT_ERROR: Optional[BaseException] = None
try:
    import torch
    import torch.nn as _tnn
    _TORCH_OK = True
except Exception as exc:  # noqa: BLE001
    _TORCH_IMPORT_ERROR = exc
    torch = None  # type: ignore[assignment]
    _tnn = None  # type: ignore[assignment]

try:
    import torchvision.models as _tv_models  # type: ignore[import-not-found]
    _TV_OK = True
except Exception as exc:  # noqa: BLE001
    _TV_IMPORT_ERROR = exc
    _tv_models = None  # type: ignore[assignment]


# ── Helper: resolve mask path ────────────────────────────────────────────────

def _resolve_mask_path(
    mask_path: str,
    dataset_root: Optional[str] = None,
    video_path: Optional[str] = None,
) -> Optional[str]:
    """Return absolute mask file path, trying several base directories."""
    mp = mask_path.replace("/", os.sep)
    if os.path.isabs(mp):
        return mp if os.path.isfile(mp) and os.path.getsize(mp) > 0 else None
    if dataset_root:
        abs1 = os.path.join(dataset_root, mp)
        if os.path.isfile(abs1) and os.path.getsize(abs1) > 0:
            return abs1
    if video_path:
        abs2 = os.path.join(os.path.dirname(video_path), mp)
        if os.path.isfile(abs2) and os.path.getsize(abs2) > 0:
            return abs2
    return None


# ═════════════════════════════════════════════════════════════════════════════


class MaskedROIResNetExtractor:
    """Extract 512-D deep texture features from segmentation-masked ROIs.

    The ResNet-18 backbone (ImageNet-pretrained) is lazy-loaded on the first
    call to :py:meth:`extract_single` or :py:meth:`extract_from_video`.
    All inference runs in ``torch.no_grad()`` mode.

    Parameters
    ----------
    device : str
        ``"auto"`` (default) selects CUDA when available, else CPU.
    batch_size : int
        Mini-batch size for batched forward passes (default 32).
    input_size : int
        Spatial size to which ROI patches are resized (default 224).
    """

    def __init__(
        self,
        device: str = "auto",
        batch_size: int = 32,
        input_size: int = 224,
        pretrain_ckpt: Optional[str] = None,
    ):
        self._device_str = device
        self._batch_size = batch_size
        self._input_size = input_size
        self._pretrain_ckpt = pretrain_ckpt
        self._model = None
        self._projection = None
        self._output_dim = RESNET_FEAT_DIM
        self._device = None
        self._mean = None
        self._std = None

    @staticmethod
    def _extract_state_dict(payload):
        if isinstance(payload, dict):
            for key in ("backbone_state_dict", "backbone", "state_dict", "model"):
                value = payload.get(key)
                if isinstance(value, dict):
                    return value
            if any(hasattr(v, "shape") for v in payload.values()):
                return payload
        return None

    @staticmethod
    def _extract_projection_state_dict(payload):
        if isinstance(payload, dict):
            for key in ("projection_state_dict", "projection", "proj_state_dict", "proj"):
                value = payload.get(key)
                if isinstance(value, dict):
                    return value
        return None

    @staticmethod
    def _strip_prefixes(state):
        cleaned = {}
        for k, v in state.items():
            key = str(k)
            for p in ("module.", "backbone.", "model.", "net."):
                if key.startswith(p):
                    key = key[len(p):]
            cleaned[key] = v
        return cleaned

    @staticmethod
    def _strip_projection_prefixes(state):
        cleaned = {}
        for k, v in state.items():
            key = str(k)
            for p in ("module.", "projection.", "proj.", "model."):
                if key.startswith(p):
                    key = key[len(p):]
            cleaned[key] = v
        return cleaned

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """True when torch + torchvision are importable."""
        return _TORCH_OK and _TV_OK

    @property
    def output_dim(self) -> int:
        return int(self._output_dim)

    # ── Lazy initialization ──────────────────────────────────────────────

    def _ensure_model(self):
        if self._model is not None:
            return
        if not self.available:
            causes = []
            if _TORCH_IMPORT_ERROR is not None:
                causes.append(f"torch: {_TORCH_IMPORT_ERROR}")
            if _TV_IMPORT_ERROR is not None:
                causes.append(f"torchvision: {_TV_IMPORT_ERROR}")
            detail = f" ({'; '.join(causes)})" if causes else ""
            raise RuntimeError(
                "PyTorch / torchvision required for ResNet texture extraction"
                f"{detail}."
            )
        self._device = torch.device(
            "cuda"
            if self._device_str == "auto" and torch.cuda.is_available()
            else "cpu" if self._device_str == "auto" else self._device_str
        )
        # Load ResNet-18 with pretrained weights
        try:
            from torchvision.models import ResNet18_Weights
            base = _tv_models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except (ImportError, AttributeError):
            base = _tv_models.resnet18(pretrained=True)  # type: ignore[call-overload]

        # Truncated model: keep everything up to and including avgpool → 512-D
        self._model = _tnn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
            base.avgpool,
            _tnn.Flatten(1),
        )
        self._model.eval()
        self._model.to(self._device)
        for p in self._model.parameters():
            p.requires_grad_(False)

        if self._pretrain_ckpt:
            try:
                payload = torch.load(self._pretrain_ckpt, map_location="cpu")
                raw_state = self._extract_state_dict(payload)
                if raw_state is not None:
                    state = self._strip_prefixes(raw_state)
                    self._model.load_state_dict(state, strict=False)
                    logger.info(
                        "MaskedROIResNetExtractor: loaded pretrain checkpoint %s",
                        self._pretrain_ckpt,
                    )
                else:
                    logger.warning(
                        "MaskedROIResNetExtractor: checkpoint has no state_dict: %s",
                        self._pretrain_ckpt,
                    )

                proj_state = self._extract_projection_state_dict(payload)
                if isinstance(proj_state, dict) and proj_state:
                    proj_clean = self._strip_projection_prefixes(proj_state)
                    emb_dim = None
                    if isinstance(payload, dict) and payload.get("embedding_dim") is not None:
                        emb_dim = int(payload.get("embedding_dim"))
                    elif "0.weight" in proj_clean and hasattr(proj_clean["0.weight"], "shape"):
                        emb_dim = int(proj_clean["0.weight"].shape[0])

                    if emb_dim is not None and emb_dim > 0:
                        self._projection = _tnn.Sequential(
                            _tnn.Linear(RESNET_FEAT_DIM, emb_dim),
                            _tnn.LayerNorm(emb_dim),
                            _tnn.GELU(),
                        )
                        self._projection.load_state_dict(proj_clean, strict=False)
                        self._projection.eval()
                        self._projection.to(self._device)
                        for p in self._projection.parameters():
                            p.requires_grad_(False)
                        self._output_dim = emb_dim
                        logger.info(
                            "MaskedROIResNetExtractor: loaded projection from pretrain checkpoint (dim=%d)",
                            emb_dim,
                        )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "MaskedROIResNetExtractor: failed to load checkpoint %s (%s)",
                    self._pretrain_ckpt,
                    exc,
                )

        self._mean = torch.tensor(
            [0.485, 0.456, 0.406], device=self._device
        ).view(1, 3, 1, 1)
        self._std = torch.tensor(
            [0.229, 0.224, 0.225], device=self._device
        ).view(1, 3, 1, 1)
        logger.info(
            "MaskedROIResNetExtractor: ResNet-18 loaded on %s", self._device
        )

    # ── Preprocessing ────────────────────────────────────────────────────

    @staticmethod
    def _crop_and_mask(
        frame_gray: np.ndarray,
        bbox,
        mask: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Crop bbox from frame, optionally apply segmentation mask."""
        h, w = frame_gray.shape[:2]
        x, y, bw, bh = bbox
        x0 = max(0, int(np.floor(x)))
        y0 = max(0, int(np.floor(y)))
        x1 = min(w, int(np.ceil(x + bw)))
        y1 = min(h, int(np.ceil(y + bh)))
        if x1 <= x0 or y1 <= y0:
            return None
        patch = frame_gray[y0:y1, x0:x1].copy()
        if mask is not None:
            mask_crop = mask[y0:y1, x0:x1]
            if mask_crop.ndim == 3:
                mask_crop = mask_crop[:, :, 0]
            patch = patch * (mask_crop > 0).astype(np.uint8)
        if patch.size == 0 or min(patch.shape[:2]) < 2:
            return None
        return patch

    def _preprocess_patch(self, gray_patch: np.ndarray) -> "torch.Tensor":
        """(H, W) uint8 → (1, 3, input_size, input_size) float tensor."""
        sz = self._input_size
        resized = cv2.resize(gray_patch, (sz, sz), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(resized.astype(np.float32) / 255.0)
        # (H, W) → (1, 1, H, W) → (1, 3, H, W)
        t = t.unsqueeze(0).unsqueeze(0).expand(1, 3, sz, sz).contiguous()
        return t

    # ── Public API ───────────────────────────────────────────────────────

    def extract_single(
        self,
        frame_gray: np.ndarray,
        bbox,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Extract 512-D feature from one frame's masked ROI.

        Returns ``np.zeros(512, float32)`` on failure.
        """
        zeros = np.zeros(self.output_dim, dtype=np.float32)
        if not self.available:
            return zeros
        self._ensure_model()

        patch = self._crop_and_mask(frame_gray, bbox, mask)
        if patch is None:
            return zeros

        tensor = self._preprocess_patch(patch).to(self._device)
        tensor = (tensor - self._mean) / self._std
        with torch.no_grad():
            feat = self._model(tensor)
            if self._projection is not None:
                feat = self._projection(feat)
        return feat.cpu().numpy().flatten().astype(np.float32)

    def extract_from_video(
        self,
        video_path: str,
        samples: Sequence,
        dataset_root: Optional[str] = None,
        *,
        global_preprocs: Optional[Sequence] = None,
        roi_preprocs: Optional[Sequence] = None,
    ) -> np.ndarray:
        """Extract 512-D features for each *sample* frame.

        Opens the video file once, reads each frame indicated by the
        samples' ``frame_index``, crops the bbox, optionally applies
        the segmentation mask, and runs the patch through ResNet-18.

        Parameters
        ----------
        video_path : str
            Path to the video file.
        samples : Sequence[FramePrediction]
            Frame predictions (must have ``frame_index``, ``bbox``,
            and optionally ``segmentation.mask_path``).
        dataset_root : str, optional
            Base directory used to resolve relative mask paths.

        Returns
        -------
        np.ndarray
            Shape ``(len(samples), 512)``, dtype ``float32``.
        """
        N = len(samples)
        empty = np.zeros((max(N, 1), self.output_dim), dtype=np.float32)
        if not self.available or N == 0:
            return empty[:N] if N else empty
        self._ensure_model()

        # 1. Read all patches from video
        patches: List[Optional[np.ndarray]] = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning("Cannot open video: %s", video_path)
            return np.zeros((N, self.output_dim), dtype=np.float32)

        try:
            for sample in samples:
                frame_idx = max(0, int(round(sample.frame_index)))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok or frame is None:
                    patches.append(None)
                    continue

                frame_for_crop = frame
                if global_preprocs:
                    frame_for_crop = _apply_preprocs_frame_like_segmentation(
                        frame_for_crop,
                        global_preprocs,
                    )

                # Load segmentation mask if available
                mask_img = None
                seg = getattr(sample, "segmentation", None)
                if seg is not None and getattr(seg, "mask_path", None):
                    resolved = _resolve_mask_path(
                        seg.mask_path,
                        dataset_root=dataset_root,
                        video_path=video_path,
                    )
                    if resolved is not None:
                        try:
                            mask_img = cv2.imread(resolved, cv2.IMREAD_GRAYSCALE)
                        except Exception:
                            pass

                seg = getattr(sample, "segmentation", None)
                seg_roi_bbox = getattr(seg, "roi_bbox", None) if seg is not None else None
                crop_bbox = seg_roi_bbox if seg_roi_bbox is not None else sample.bbox
                h, w = frame_for_crop.shape[:2]
                x, y, bw, bh = crop_bbox
                x0 = max(0, int(np.floor(x)))
                y0 = max(0, int(np.floor(y)))
                x1 = min(w, int(np.ceil(x + bw)))
                y1 = min(h, int(np.ceil(y + bh)))
                if x1 <= x0 or y1 <= y0:
                    patches.append(None)
                    continue

                roi = frame_for_crop[y0:y1, x0:x1].copy()
                if roi_preprocs and roi.size > 0:
                    roi = _apply_preprocs_frame_like_segmentation(roi, roi_preprocs)
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
                mask_crop = mask_img[y0:y1, x0:x1] if mask_img is not None else None
                if mask_crop is not None and mask_crop.ndim == 3:
                    mask_crop = mask_crop[:, :, 0]
                if mask_crop is not None:
                    gray_roi = gray_roi * (mask_crop > 0).astype(np.uint8)
                patch = gray_roi if gray_roi.size > 0 and min(gray_roi.shape[:2]) >= 2 else None
                patches.append(patch)
        finally:
            cap.release()

        # 2. Preprocess valid patches → tensors
        valid_indices = [i for i, p in enumerate(patches) if p is not None]
        if not valid_indices:
            return np.zeros((N, RESNET_FEAT_DIM), dtype=np.float32)

        tensors = [self._preprocess_patch(patches[i]) for i in valid_indices]

        # 3. Batched forward pass
        features = np.zeros((N, self.output_dim), dtype=np.float32)
        bs = self._batch_size
        for start in range(0, len(tensors), bs):
            end = min(start + bs, len(tensors))
            batch_tensors = tensors[start:end]
            batch_indices = valid_indices[start:end]
            batch = torch.cat(batch_tensors, dim=0).to(self._device)
            batch = (batch - self._mean) / self._std
            with torch.no_grad():
                out = self._model(batch)
                if self._projection is not None:
                    out = self._projection(out)
            out_np = out.cpu().numpy().astype(np.float32)
            for j, idx in enumerate(batch_indices):
                features[idx] = out_np[j]

        return features
