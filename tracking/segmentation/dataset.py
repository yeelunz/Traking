from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..core.interfaces import FramePrediction, MaskStats, SegmentationData, PreprocessingModule
from ..utils.annotations import load_coco_vid
from .utils import (
    BoundingBox,
    compute_mask_stats,
    crop_with_bbox,
    expand_bbox,
    fill_holes,
    keep_largest_component,
)


@dataclass
class SegmentationSampleDescriptor:
    video_path: str
    frame_index: int
    roi_bbox: BoundingBox
    mask_path: Optional[str]
    original_bbox: Tuple[float, float, float, float]


class SegmentationCropDataset(Dataset):
    """Generate ROI crops for segmentation training using ground-truth masks."""

    def __init__(
        self,
        video_paths: Sequence[str],
        dataset_root: str,
        padding_range: Tuple[float, float] = (0.10, 0.15),
        redundancy: int = 1,
        seed: int = 0,
        cache_annotations: Optional[Dict[str, Dict]] = None,
        jitter: float = 0.0,
        target_size: Optional[Tuple[int, int]] = None,
        preprocs: Optional[Sequence[PreprocessingModule]] = None,
        roi_preprocs: Optional[Sequence[PreprocessingModule]] = None,
    ) -> None:
        super().__init__()
        self.dataset_root = dataset_root
        self.padding_min = float(min(padding_range))
        self.padding_max = float(max(padding_range))
        self.rng = random.Random(seed)
        self._jitter_rng = random.Random(seed + 1337)
        self.jitter = max(0.0, float(jitter))
        if target_size is not None:
            self.target_size = self._normalize_target_size(target_size)
        else:
            self.target_size = None
        self.entries: List[SegmentationSampleDescriptor] = []
        self._annotations_cache: Dict[str, Dict] = dict(cache_annotations or {})
        self.missing_annotations: List[str] = []
        self._mask_valid_cache: Dict[Tuple[str, str], bool] = {}
        self.empty_masks: List[Tuple[str, int]] = []
        # global preprocs apply to full frames before ROI cropping
        self.preprocs: List[PreprocessingModule] = list(preprocs or [])
        # roi preprocs apply to ROI crops after cropping
        self.roi_preprocs: List[PreprocessingModule] = list(roi_preprocs or [])

        for video_path in video_paths:
            ann = self._get_annotation(video_path)
            if not ann:
                self.missing_annotations.append(video_path)
                continue
            frame_ann = ann.get("frame_annotations", {})
            width = ann.get("raw", {}).get("videos", [{}])[0].get("width")
            height = ann.get("raw", {}).get("videos", [{}])[0].get("height")
            if not width or not height:
                width = ann.get("raw", {}).get("images", [{}])[0].get("width", 0)
                height = ann.get("raw", {}).get("images", [{}])[0].get("height", 0)
            image_shape = (int(height), int(width)) if width and height else None
            for frame_idx, items in frame_ann.items():
                if not items:
                    continue
                item = items[0]
                bbox = tuple(item.get("bbox", (0.0, 0.0, 0.0, 0.0)))
                mask_path = item.get("mask_path")
                if not self._mask_has_content(video_path, mask_path):
                    self.empty_masks.append((video_path, int(frame_idx)))
                    continue
                for _ in range(max(1, redundancy)):
                    pad = self._sample_padding()
                    roi_bbox = self._compute_roi(bbox, pad, video_path, frame_idx, image_shape)
                    self.entries.append(
                        SegmentationSampleDescriptor(
                            video_path=video_path,
                            frame_index=int(frame_idx),
                            roi_bbox=roi_bbox,
                            mask_path=mask_path,
                            original_bbox=bbox,
                        )
                    )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        frame = self._load_frame(entry.video_path, entry.frame_index)
        if frame is None:
            raise RuntimeError(f"Failed to read frame {entry.frame_index} from {entry.video_path}")
        mask_full = self._load_mask(entry.video_path, entry.mask_path)
        if mask_full is None:
            # Mask file missing or empty on disk (e.g. zero-area annotation written
            # by the annotator for a frame where nothing was drawn).  Raise a
            # descriptive error so the DataLoader can surface the issue clearly,
            # but also attempt to return a blank mask so training can continue.
            # Re-raise as SkipSample-style error is not supported by default
            # DataLoader; instead we raise RuntimeError with full context.
            raise RuntimeError(
                f"Mask not found or empty for frame {entry.frame_index} in "
                f"{entry.video_path} (mask_path={entry.mask_path!r}). "
                "This usually means the annotation has area=0 or the mask file "
                "does not exist. Remove or re-annotate this frame."
            )
        roi_bbox = entry.roi_bbox
        if self.preprocs:
            frame, mask_full, roi_bbox = self._apply_preprocs_frame_mask_bbox(
                frame,
                mask_full,
                roi_bbox,
                self.preprocs,
            )
        if self.jitter > 0.0:
            roi_bbox = self._jitter_bbox(roi_bbox, frame.shape[:2])
        roi_image = crop_with_bbox(frame, roi_bbox)
        roi_mask = crop_with_bbox(mask_full, roi_bbox)
        if self.roi_preprocs and roi_image.size != 0:
            roi_image, roi_mask = self._apply_preprocs_frame_and_mask(
                roi_image,
                roi_mask,
                self.roi_preprocs,
            )
        orig_size = roi_image.shape[:2]
        if self.target_size is not None:
            roi_image = self._resize_image(roi_image, self.target_size)
            roi_mask = self._resize_mask(roi_mask, self.target_size)
        if roi_image.ndim == 2:
            roi_image = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2BGR)
        roi_image = roi_image.astype(np.float32) / 255.0
        roi_mask = (roi_mask > 0).astype(np.float32)
        image_tensor = torch.from_numpy(roi_image.transpose(2, 0, 1))
        mask_tensor = torch.from_numpy(roi_mask).unsqueeze(0)
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "frame_index": entry.frame_index,
            "video_path": entry.video_path,
            "roi_bbox": roi_bbox.as_tuple(),
            "original_roi_size": orig_size,
            "original_bbox": entry.original_bbox,
        }

    def _apply_preprocs_frame(self, frame: np.ndarray, preprocs: Sequence[PreprocessingModule]) -> np.ndarray:
        """Apply preprocessing modules to a frame (BGR or grayscale).

        Preprocessing modules operate on RGB for 3-channel images.
        Frames are always converted to grayscale first to ensure consistent processing.
        """
        if frame.ndim == 3:
            _g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(_g, cv2.COLOR_GRAY2BGR)
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

    def _apply_preprocs_frame_and_mask(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        preprocs: Sequence[PreprocessingModule],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if frame.ndim == 3:
            _g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(_g, cv2.COLOR_GRAY2BGR)
        if not preprocs:
            return frame, mask
        if frame.ndim == 2:
            out = frame
            out_mask = mask
            for p in preprocs:
                if hasattr(p, "apply_to_frame_and_mask"):
                    out, out_mask = p.apply_to_frame_and_mask(out, out_mask)
                else:
                    out = p.apply_to_frame(out)
            return out, out_mask
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out_mask = mask
        for p in preprocs:
            if hasattr(p, "apply_to_frame_and_mask"):
                rgb, out_mask = p.apply_to_frame_and_mask(rgb, out_mask)
            else:
                rgb = p.apply_to_frame(rgb)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), out_mask

    def _apply_preprocs_frame_mask_bbox(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        bbox: BoundingBox,
        preprocs: Sequence[PreprocessingModule],
    ) -> Tuple[np.ndarray, np.ndarray, BoundingBox]:
        if frame.ndim == 3:
            _g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(_g, cv2.COLOR_GRAY2BGR)
        if not preprocs:
            return frame, mask, bbox
        bbox_tuple = bbox.as_tuple()
        if frame.ndim == 2:
            out = frame
            out_mask = mask
            out_bbox = bbox_tuple
            for p in preprocs:
                if hasattr(p, "apply_to_frame_mask_bbox"):
                    out, out_mask, out_bbox = p.apply_to_frame_mask_bbox(out, out_mask, out_bbox)
                elif hasattr(p, "apply_to_frame_and_mask"):
                    out, out_mask = p.apply_to_frame_and_mask(out, out_mask)
                else:
                    out = p.apply_to_frame(out)
            return out, out_mask, BoundingBox(*out_bbox)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out_mask = mask
        out_bbox = bbox_tuple
        for p in preprocs:
            if hasattr(p, "apply_to_frame_mask_bbox"):
                rgb, out_mask, out_bbox = p.apply_to_frame_mask_bbox(rgb, out_mask, out_bbox)
            elif hasattr(p, "apply_to_frame_and_mask"):
                rgb, out_mask = p.apply_to_frame_and_mask(rgb, out_mask)
            else:
                rgb = p.apply_to_frame(rgb)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), out_mask, BoundingBox(*out_bbox)

    # ---------------- internal helpers ----------------
    def _sample_padding(self) -> float:
        return self.rng.uniform(self.padding_min, self.padding_max)

    def _compute_roi(
        self,
        bbox: Tuple[float, float, float, float],
        pad: float,
        video_path: str,
        frame_idx: int,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> BoundingBox:
        if image_shape is None:
            frame = self._load_frame(video_path, frame_idx)
            if frame is None:
                raise RuntimeError(f"Unable to load frame for ROI computation: {video_path}#{frame_idx}")
            image_shape = frame.shape[:2]
        return expand_bbox(bbox, pad, image_shape)

    def _jitter_bbox(self, bbox: BoundingBox, image_shape: Tuple[int, int]) -> BoundingBox:
        img_h, img_w = image_shape
        if bbox.w <= 0 or bbox.h <= 0:
            return bbox
        max_dx = bbox.w * self.jitter
        max_dy = bbox.h * self.jitter
        dx = self._jitter_rng.uniform(-max_dx, max_dx)
        dy = self._jitter_rng.uniform(-max_dy, max_dy)
        new_x = bbox.x + dx
        new_y = bbox.y + dy
        max_x = max(0.0, img_w - bbox.w)
        max_y = max(0.0, img_h - bbox.h)
        new_x = float(min(max(0.0, new_x), max_x))
        new_y = float(min(max(0.0, new_y), max_y))
        return BoundingBox(new_x, new_y, bbox.w, bbox.h)

    def _get_annotation(self, video_path: str) -> Dict:
        if video_path in self._annotations_cache:
            return self._annotations_cache[video_path]

        candidates: List[Path] = []
        vp = Path(video_path)
        if vp.suffix:
            candidates.append(vp.with_suffix(".json"))
        candidates.append(Path(f"{video_path}.json"))

        if self.dataset_root:
            try:
                rel = Path(video_path)
                if rel.is_absolute():
                    rel = rel.relative_to(Path(self.dataset_root))
                rel_base = rel.with_suffix(".json")
                candidates.append(Path(self.dataset_root) / rel_base)
            except Exception:
                pass

        seen: List[str] = []
        for cand in candidates:
            cand_str = str(cand)
            if cand_str in seen:
                continue
            seen.append(cand_str)
            if cand.exists():
                data = load_coco_vid(cand_str)
                self._annotations_cache[video_path] = data
                return data
        return {}

    def _load_frame(self, video_path: str, frame_index: int) -> Optional[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return None
        return frame

    def _load_mask(self, video_path: str, mask_path: Optional[str]) -> Optional[np.ndarray]:
        if not mask_path:
            return None
        mask_file = mask_path.replace("/", os.sep)
        if os.path.isabs(mask_file):
            abs_path = mask_file
        else:
            abs_path = os.path.join(self.dataset_root, mask_file)
            # Fallback: resolve relative to video directory
            if not os.path.exists(abs_path):
                video_dir = os.path.dirname(video_path)
                alt_path = os.path.join(video_dir, mask_file)
                if os.path.exists(alt_path):
                    abs_path = alt_path
        if not os.path.isfile(abs_path):
            return None
        # Guard against zero-byte files: Ultralytics patches cv2.imread to use
        # np.fromfile + imdecode; empty files produce an assertion failure.
        if os.path.getsize(abs_path) == 0:
            return None
        try:
            mask = cv2.imread(abs_path, cv2.IMREAD_GRAYSCALE)
        except Exception:
            return None
        if mask is None:
            return None
        # ensure binary mask and fill interior holes to avoid ring-shaped annotations
        mask_bin = (mask > 0).astype(np.uint8) * 255
        mask_filled = fill_holes(mask_bin)
        mask_clean = keep_largest_component(mask_filled)
        if mask_clean is None or mask_clean.size == 0 or np.count_nonzero(mask_clean) == 0:
            return None
        return mask_clean

    def _mask_has_content(self, video_path: str, mask_path: Optional[str]) -> bool:
        if not mask_path:
            return False
        # Use (video_path, mask_path) as the cache key so that different subjects
        # with the same relative mask_path (e.g. seg_masks/doppler/frame_xxx.png)
        # do not share a cached result and risk a False-Positive on empty masks.
        cache_key = (video_path, mask_path)
        if cache_key in self._mask_valid_cache:
            return self._mask_valid_cache[cache_key]
        mask = self._load_mask(video_path, mask_path)
        has_content = mask is not None
        self._mask_valid_cache[cache_key] = has_content
        return has_content

    # ---------------- resize helpers -----------------
    @staticmethod
    def _normalize_target_size(target: Tuple[int, int]) -> Tuple[int, int]:
        if isinstance(target, (list, tuple)):
            vals = [int(v) for v in target if int(v) > 0]
            if len(vals) == 1:
                s = max(1, vals[0])
                return (s, s)
            if len(vals) >= 2:
                h, w = vals[0], vals[1]
                return (max(1, h), max(1, w))
        if isinstance(target, (int, float)):
            s = int(target)
            if s > 0:
                return (s, s)
        raise ValueError(f"Invalid target size: {target}")

    @staticmethod
    def _resize_image(image: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
        target_h, target_w = target_hw
        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return resized

    @staticmethod
    def _resize_mask(mask: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
        target_h, target_w = target_hw
        resized = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        return resized


def _resolve_gt_mask_path(
    mask_path: Optional[str],
    dataset_root: Optional[str],
    video_dir: Optional[str] = None,
) -> Optional[str]:
    """Return absolute mask path if the file actually exists on disk, else None.

    Resolution order (merged / subject-level dataset support):
    1. Absolute path as-is
    2. dataset_root / mask_path
    3. video_dir / mask_path  (most reliable for merged layouts)
    4. dataset_root's parent / mask_path  (old flat-export layout)
    5. Strip leading path components from mask_path and retry from each base
    """
    if not mask_path:
        return None
    mp = mask_path.replace("/", os.sep)
    if os.path.isabs(mp):
        return mp if os.path.isfile(mp) and os.path.getsize(mp) > 0 else None

    bases: List[Optional[str]] = [dataset_root, video_dir]
    if dataset_root:
        parent = os.path.dirname(dataset_root.rstrip(os.sep))
        if parent and parent != dataset_root:
            bases.append(parent)

    def _safe_candidate(base: str, rel: str) -> "str | None":
        candidate = os.path.normpath(os.path.join(base, rel))
        # Guard: resolved path must stay under the trusted base directory.
        base_norm = os.path.normpath(base)
        if not candidate.startswith(base_norm + os.sep) and candidate != base_norm:
            return None
        if os.path.isfile(candidate) and os.path.getsize(candidate) > 0:
            return candidate
        return None

    for base in bases:
        if not base:
            continue
        hit = _safe_candidate(base, mp)
        if hit:
            return hit

    # Last-resort: strip leading path components and retry
    parts = Path(mask_path).parts
    for skip in range(1, len(parts)):
        stripped = os.path.join(*parts[skip:])
        for base in bases:
            if not base:
                continue
            hit = _safe_candidate(base, stripped)
            if hit:
                return hit
    return None


def attach_ground_truth_segmentation(
    annotation: Dict,
    dataset_root: str,
    video_path: Optional[str] = None,
) -> List[FramePrediction]:
    """Build a list of ground-truth FramePrediction objects from a COCO-VID annotation dict.

    Parameters
    ----------
    annotation:
        Parsed COCO-VID dict (from :func:`load_coco_vid`).
    dataset_root:
        Root directory of the dataset.  Used as the primary base for resolving
        relative ``mask_path`` values stored inside the JSON.
    video_path:
        Optional absolute path to the video file associated with this annotation.
        When provided, the video's parent directory is used as an additional
        resolution candidate, which is required for the merged / subject-level
        dataset layout (``dataset_root/<subject>/video.avi``).
    """
    video_dir: Optional[str] = os.path.dirname(os.path.abspath(video_path)) if video_path else None
    frames = []
    raw_frames = annotation.get("frames", {})
    frame_ann = annotation.get("frame_annotations", {})
    for frame_idx, bbox_list in raw_frames.items():
        idx = int(frame_idx)
        ann_items = frame_ann.get(frame_idx) or frame_ann.get(str(frame_idx)) or []
        bbox = bbox_list[0] if bbox_list else (0.0, 0.0, 0.0, 0.0)
        mask_path = None
        stats: Optional[MaskStats] = None
        if ann_items:
            ann_entry = ann_items[0]
            mask_path = ann_entry.get("mask_path")
            metadata = ann_entry.get("metadata") or {}
            centroid = metadata.get("centroid", [0.0, 0.0])
            stats = MaskStats(
                area_px=float(metadata.get("area", metadata.get("area_px", 0.0) or ann_entry.get("area", 0.0))),
                bbox=tuple(bbox),
                centroid=(float(centroid[0]), float(centroid[1])),
                perimeter_px=float(metadata.get("perimeter_px", 0.0)),
                equivalent_diameter_px=float(metadata.get("equivalent_diameter_px", 0.0)),
            )

        # Skip frames whose mask file does not exist on disk.
        # If the mask was manually deleted the associated bbox is also unreliable,
        # so we drop the entire frame rather than using it without a mask.
        resolved_mask = _resolve_gt_mask_path(mask_path, dataset_root, video_dir)
        if resolved_mask is None:
            continue

        pred = FramePrediction(
            frame_index=idx,
            bbox=tuple(map(float, bbox)),
            score=None,
            segmentation=SegmentationData(
                mask_path=resolved_mask,
                stats=stats or MaskStats(0.0, (0.0, 0.0, 0.0, 0.0), (0.0, 0.0), 0.0, 0.0),
            ),
        )
        frames.append(pred)
    frames.sort(key=lambda s: s.frame_index)
    return frames
