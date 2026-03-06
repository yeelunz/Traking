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

# ── Dependency checks ────────────────────────────────────────────────────────
_TORCH_OK = False
_TV_OK = False
try:
    import torch
    import torch.nn as _tnn
    _TORCH_OK = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _tnn = None  # type: ignore[assignment]

try:
    import torchvision.models as _tv_models  # type: ignore[import-not-found]
    _TV_OK = True
except ImportError:
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
    ):
        self._device_str = device
        self._batch_size = batch_size
        self._input_size = input_size
        self._model = None
        self._device = None
        self._mean = None
        self._std = None

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """True when torch + torchvision are importable."""
        return _TORCH_OK and _TV_OK

    # ── Lazy initialization ──────────────────────────────────────────────

    def _ensure_model(self):
        if self._model is not None:
            return
        if not self.available:
            raise RuntimeError(
                "PyTorch / torchvision required for ResNet texture extraction."
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
        zeros = np.zeros(RESNET_FEAT_DIM, dtype=np.float32)
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
        return feat.cpu().numpy().flatten().astype(np.float32)

    def extract_from_video(
        self,
        video_path: str,
        samples: Sequence,
        dataset_root: Optional[str] = None,
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
        empty = np.zeros((max(N, 1), RESNET_FEAT_DIM), dtype=np.float32)
        if not self.available or N == 0:
            return empty[:N] if N else empty
        self._ensure_model()

        # 1. Read all patches from video
        patches: List[Optional[np.ndarray]] = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning("Cannot open video: %s", video_path)
            return np.zeros((N, RESNET_FEAT_DIM), dtype=np.float32)

        try:
            for sample in samples:
                frame_idx = max(0, int(round(sample.frame_index)))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok or frame is None:
                    patches.append(None)
                    continue

                gray = (
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if frame.ndim == 3
                    else frame
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

                patch = self._crop_and_mask(gray, sample.bbox, mask_img)
                patches.append(patch)
        finally:
            cap.release()

        # 2. Preprocess valid patches → tensors
        valid_indices = [i for i, p in enumerate(patches) if p is not None]
        if not valid_indices:
            return np.zeros((N, RESNET_FEAT_DIM), dtype=np.float32)

        tensors = [self._preprocess_patch(patches[i]) for i in valid_indices]

        # 3. Batched forward pass
        features = np.zeros((N, RESNET_FEAT_DIM), dtype=np.float32)
        bs = self._batch_size
        for start in range(0, len(tensors), bs):
            end = min(start + bs, len(tensors))
            batch_tensors = tensors[start:end]
            batch_indices = valid_indices[start:end]
            batch = torch.cat(batch_tensors, dim=0).to(self._device)
            batch = (batch - self._mean) / self._std
            with torch.no_grad():
                out = self._model(batch)
            out_np = out.cpu().numpy().astype(np.float32)
            for j, idx in enumerate(batch_indices):
                features[idx] = out_np[j]

        return features
