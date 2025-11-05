from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

MASK_ROOT_DEFAULT = "seg_masks"


def _largest_component(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    if binary.ndim > 2:
        binary = np.squeeze(binary)
    if binary.ndim != 2:
        binary = binary.reshape(mask.shape[0], mask.shape[1])
    num_labels, labels = cv2.connectedComponents(binary)
    if num_labels <= 1:
        return binary * 255
    # Ignore background label 0
    counts = np.bincount(labels.ravel())
    if len(counts) <= 1:
        return binary * 255
    counts[0] = 0
    best_label = int(np.argmax(counts))
    filtered = (labels == best_label).astype(np.uint8)
    return filtered * 255


@dataclass
class MaskMetadata:
    """Derived statistics for a single binary mask.

    These fields are serialised alongside the COCO-style annotation entry so that
    downstream components can recover both bounding boxes and motion-related
    descriptors without recomputing them from scratch.
    """

    area_px: float
    bbox: Tuple[float, float, float, float]
    centroid: Tuple[float, float]
    perimeter_px: float
    equivalent_diameter_px: float

    @classmethod
    def from_mask(cls, mask: np.ndarray) -> "MaskMetadata":
        if mask.dtype != np.uint8:
            binary = (mask > 0).astype(np.uint8)
        else:
            binary = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return cls(area_px=0.0, bbox=(0.0, 0.0, 0.0, 0.0), centroid=(0.0, 0.0), perimeter_px=0.0, equivalent_diameter_px=0.0)
        # Merge to a single contour to compute metrics on disjoint regions.
        merged = np.vstack(contours)
        x, y, w, h = cv2.boundingRect(merged)
        area = float(np.count_nonzero(binary))
        moments = cv2.moments(binary)
        if moments["m00"] > 0:
            cx = float(moments["m10"] / moments["m00"])
            cy = float(moments["m01"] / moments["m00"])
        else:
            cx = float(x + w / 2.0)
            cy = float(y + h / 2.0)
        perimeter = float(cv2.arcLength(merged, True))
        eq_diameter = float(np.sqrt(4.0 * area / np.pi)) if area > 0 else 0.0
        return cls(area_px=area, bbox=(float(x), float(y), float(w), float(h)), centroid=(cx, cy), perimeter_px=perimeter, equivalent_diameter_px=eq_diameter)


@dataclass
class MotionSample:
    frame_index: int
    centroid: Tuple[float, float]
    displacement: Tuple[float, float]
    distance: float


@dataclass
class TrackSummary:
    track_id: int
    category_id: int
    samples: List[MotionSample] = field(default_factory=list)

    @property
    def total_path_length(self) -> float:
        return float(sum(s.distance for s in self.samples))

    @property
    def average_speed_per_frame(self) -> float:
        if not self.samples:
            return 0.0
        return float(np.mean([s.distance for s in self.samples]))

    def to_json(self) -> Dict:
        return {
            "track_id": self.track_id,
            "category_id": self.category_id,
            "samples": [
                {
                    "frame_index": s.frame_index,
                    "centroid": [round(float(s.centroid[0]), 3), round(float(s.centroid[1]), 3)],
                    "displacement": [round(float(s.displacement[0]), 3), round(float(s.displacement[1]), 3)],
                    "distance": round(float(s.distance), 3),
                }
                for s in self.samples
            ],
            "total_path_length": round(self.total_path_length, 3),
            "average_speed_per_frame": round(self.average_speed_per_frame, 3),
        }


@dataclass
class MaskAnnotation:
    frame_index: int
    track_id: int
    category_id: int
    mask: np.ndarray  # binary uint8 mask aligned with the original frame size
    metadata: MaskMetadata
    mask_path: Optional[str] = None  # relative path written on export
    previous_centroid: Optional[Tuple[float, float]] = None

    def update_mask(self, new_mask: np.ndarray) -> None:
        filtered = _largest_component(new_mask)
        self.mask = filtered.astype(np.uint8, copy=False)
        self.metadata = MaskMetadata.from_mask(self.mask)

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return self.metadata.bbox

    @property
    def centroid(self) -> Tuple[float, float]:
        return self.metadata.centroid

    def to_coco_annotation(self, ann_id: int, image_id: int) -> Dict:
        dx = dy = dist = 0.0
        if self.previous_centroid is not None:
            dx = float(self.centroid[0] - self.previous_centroid[0])
            dy = float(self.centroid[1] - self.previous_centroid[1])
            dist = float(np.hypot(dx, dy))
        rec = {
            "id": ann_id,
            "image_id": image_id,
            "category_id": self.category_id,
            "track_id": self.track_id,
            "bbox": [round(v, 2) for v in self.bbox],
            "area": round(self.metadata.area_px, 2),
            "iscrowd": 0,
            "mask_path": self.mask_path,
            "metadata": {
                "centroid": [round(v, 3) for v in self.centroid],
                "perimeter_px": round(self.metadata.perimeter_px, 2),
                "equivalent_diameter_px": round(self.metadata.equivalent_diameter_px, 2),
            },
            "motion": {
                "dx": round(dx, 3),
                "dy": round(dy, 3),
                "distance": round(dist, 3),
            },
        }
        return rec


class SegmentationProject:
    """In-memory representation of a single video's segmentation annotations.

    The project keeps masks in RAM for interactive editing and writes them to
    disk on demand. All paths emitted by :meth:`export_dataset` are relative to
    ``output_dir`` so that the exported dataset remains portable.
    """

    def __init__(self, video_path: str, categories: List[str], *, mask_root: str = MASK_ROOT_DEFAULT):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(video_path)
        self.video_name = self.video_path.name
        self.mask_root = Path(mask_root)
        self.categories = categories
        self.category_to_id = {name: idx + 1 for idx, name in enumerate(categories)}
        self.id_to_category = {idx + 1: name for idx, name in enumerate(categories)}

        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise RuntimeError("Unable to open video")
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 25.0)

        self.annotations_by_frame: Dict[int, Dict[int, MaskAnnotation]] = {}
        self.next_track_id = 1
        self.primary_track_id: Optional[int] = None
        self._load_existing()

    # ------------------------------------------------------------------
    # Video frames -----------------------------------------------------
    # ------------------------------------------------------------------
    def load_frame(self, index: int) -> Optional[np.ndarray]:
        if index < 0 or index >= self.total_frames:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        if not ok:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------------------
    # Annotation management --------------------------------------------
    # ------------------------------------------------------------------
    def get_annotations(self, frame_index: int) -> Dict[int, MaskAnnotation]:
        return self.annotations_by_frame.get(frame_index, {})

    def new_annotation(self, frame_index: int, category_id: int, mask: np.ndarray, track_id: Optional[int] = None) -> MaskAnnotation:
        mask = _largest_component(mask.astype(np.uint8))
        metadata = MaskMetadata.from_mask(mask)
        frame_store = self.annotations_by_frame.setdefault(frame_index, {})
        frame_store.clear()

        if track_id is not None:
            tid = int(track_id)
            self.primary_track_id = self.primary_track_id or tid
            if tid >= self.next_track_id:
                self.next_track_id = tid + 1
        elif self.primary_track_id is not None:
            tid = int(self.primary_track_id)
        else:
            tid = int(self.next_track_id)
            self.primary_track_id = tid
            self.next_track_id += 1

        anno = MaskAnnotation(frame_index=frame_index, track_id=tid, category_id=category_id, mask=mask, metadata=metadata)
        frame_store[tid] = anno
        return anno

    def update_annotation(self, frame_index: int, track_id: int, mask: np.ndarray) -> MaskAnnotation:
        mask = _largest_component(mask.astype(np.uint8))
        frame_store = self.annotations_by_frame.get(frame_index)
        if not frame_store:
            raise KeyError(f"No annotation for frame {frame_index} track {track_id}")
        entry = frame_store.get(track_id)
        if entry is None:
            raise KeyError(f"No annotation for frame {frame_index} track {track_id}")
        entry.update_mask(mask)
        for tid in list(frame_store.keys()):
            if tid != track_id:
                frame_store.pop(tid, None)
        return entry

    def delete_annotation(self, frame_index: int, track_id: int) -> None:
        frame_store = self.annotations_by_frame.get(frame_index)
        if not frame_store:
            return
        frame_store.pop(track_id, None)
        frame_store.clear()
        self.annotations_by_frame.pop(frame_index, None)

    def clear_frame(self, frame_index: int) -> None:
        self.annotations_by_frame.pop(frame_index, None)

    # ------------------------------------------------------------------
    # Export -----------------------------------------------------------
    # ------------------------------------------------------------------
    def export_dataset(self, output_dir: str) -> str:
        """Materialise masks and metadata to ``output_dir``.

        Returns the path to the COCO-VID compatible JSON manifest.
        """

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        mask_root = out_dir / self.mask_root
        mask_root.mkdir(parents=True, exist_ok=True)
        mask_video_dir = mask_root / self.video_name_without_ext
        mask_video_dir.mkdir(parents=True, exist_ok=True)

        images: List[Dict] = []
        annotations: List[Dict] = []
        track_summaries: Dict[int, TrackSummary] = {}

        ann_id = 1
        image_id = 1

        for frame_index in sorted(self.annotations_by_frame.keys()):
            frame_ann = self.annotations_by_frame[frame_index]
            if not frame_ann:
                continue
            frame_entry = {
                "id": image_id,
                "video_id": 1,
                "frame_index": frame_index,
                "file_name": f"{self.video_name_without_ext}/{frame_index + 1:08d}.jpg",
                "height": self.height,
                "width": self.width,
            }
            images.append(frame_entry)

            for track_id, ann in sorted(frame_ann.items()):
                mask_filename = f"frame_{frame_index + 1:08d}.png"
                if len(frame_ann) > 1:
                    mask_filename = f"frame_{frame_index + 1:08d}_track_{track_id:04d}.png"
                rel_mask_path = Path(self.video_name_without_ext) / mask_filename
                full_mask_path = mask_video_dir / mask_filename
                cv2.imwrite(str(full_mask_path), ann.mask)
                ann.mask_path = str(Path(self.mask_root) / rel_mask_path).replace(os.sep, "/")

                prev_centroid: Optional[Tuple[float, float]] = None
                if track_id in track_summaries and track_summaries[track_id].samples:
                    prev_centroid = track_summaries[track_id].samples[-1].centroid
                ann.previous_centroid = prev_centroid

                entry = ann.to_coco_annotation(ann_id=ann_id, image_id=image_id)
                annotations.append(entry)

                dx = dy = dist = 0.0
                if prev_centroid is not None:
                    dx = float(ann.centroid[0] - prev_centroid[0])
                    dy = float(ann.centroid[1] - prev_centroid[1])
                    dist = float(np.hypot(dx, dy))
                sample = MotionSample(frame_index=frame_index, centroid=ann.centroid, displacement=(dx, dy), distance=dist)
                track_summary = track_summaries.setdefault(track_id, TrackSummary(track_id=track_id, category_id=ann.category_id))
                track_summary.samples.append(sample)

                ann_id += 1
            image_id += 1

        categories = [{"id": cid, "name": name} for cid, name in self.id_to_category.items()]

        json_data = {
            "info": {
                "description": "Segmentation annotations with derived motion signals",
                "annotation_type": "bbox+mask",
                "mask_root": self.mask_root.as_posix(),
            },
            "videos": [
                {
                    "id": 1,
                    "name": self.video_name,
                    "height": self.height,
                    "width": self.width,
                    "fps": self.fps,
                    "total_frames": self.total_frames,
                }
            ],
            "images": images,
            "annotations": annotations,
            "categories": categories,
            "tracks": [summary.to_json() for summary in track_summaries.values()],
        }

        json_path = out_dir / f"{self.video_name_without_ext}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False)

        return str(json_path)

    # ------------------------------------------------------------------
    @property
    def video_name_without_ext(self) -> str:
        return self.video_path.stem

    def close(self) -> None:
        if getattr(self, "cap", None) is not None:
            try:
                self.cap.release()
            except Exception:
                pass

    # ------------------------------------------------------------------
    def _load_existing(self) -> None:
        json_path = self.video_path.with_suffix(".json")
        if not json_path.exists():
            return
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return
        images = data.get("images", [])
        annotations = data.get("annotations", [])
        img_to_frame: Dict[int, int] = {}
        for img in images:
            if not isinstance(img, dict):
                continue
            img_id = img.get("id")
            if img_id is None:
                continue
            frame_index = int(img.get("frame_index", int(img_id) - 1))
            img_to_frame[int(img_id)] = frame_index

        max_tid = 0
        for ann in annotations:
            if not isinstance(ann, dict):
                continue
            track_id = int(ann.get("track_id", 0))
            category_id = int(ann.get("category_id", 1))
            image_id = ann.get("image_id")
            if track_id <= 0 or image_id is None:
                continue
            mask_rel = ann.get("mask_path")
            if not mask_rel:
                continue
            frame_index = img_to_frame.get(int(image_id))
            if frame_index is None:
                continue
            mask_file = (json_path.parent / mask_rel).resolve()
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            mask = _largest_component(mask)
            metadata = MaskMetadata.from_mask(mask)
            if self.primary_track_id is None:
                self.primary_track_id = track_id

            tid = self.primary_track_id if self.primary_track_id is not None else track_id
            frame_store = self.annotations_by_frame.setdefault(frame_index, {})
            if frame_store:
                # Keep the annotation with the larger area for this frame
                existing_tid, existing_ann = next(iter(frame_store.items()))
                if metadata.area_px <= existing_ann.metadata.area_px:
                    continue
                frame_store.clear()
                tid = existing_tid

            entry = MaskAnnotation(
                frame_index=frame_index,
                track_id=tid,
                category_id=category_id,
                mask=mask.astype(np.uint8),
                metadata=metadata,
                mask_path=str(mask_rel),
            )
            frame_store[tid] = entry
            max_tid = max(max_tid, tid)
        if max_tid >= self.next_track_id:
            self.next_track_id = max_tid + 1


__all__ = [
    "MaskMetadata",
    "MaskAnnotation",
    "SegmentationProject",
    "TrackSummary",
    "MotionSample",
    "MASK_ROOT_DEFAULT",
]
