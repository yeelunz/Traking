from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, Iterable, Protocol
import numpy as np


@dataclass
class MaskStats:
    area_px: float
    bbox: Tuple[float, float, float, float]
    centroid: Tuple[float, float]
    perimeter_px: float
    equivalent_diameter_px: float


@dataclass
class SegmentationData:
    mask_path: Optional[str]
    stats: MaskStats
    score: Optional[float] = None
    roi_bbox: Optional[Tuple[float, float, float, float]] = None
    centroid_error_px: Optional[float] = None


@dataclass
class FramePrediction:
    frame_index: int
    bbox: Tuple[float, float, float, float]  # x,y,w,h
    score: Optional[float] = None
    confidence: Optional[float] = None
    confidence_components: Optional[Dict[str, float]] = None
    segmentation: Optional[SegmentationData] = None
    is_fallback: bool = False
    # Metadata for downstream visualization/debugging. Typical values:
    # - "detector": bbox provided by detector at this frame
    # - "prev_bbox": bbox reused from previous frame (detector-based)
    # - "prev_segmentation": bbox reused from previous frame, bootstrapped from segmentation mask
    # - "full_frame": forced full-frame ROI due to missing bbox
    # - "segmentation_bootstrap": bbox derived from this frame's segmentation mask (seed for next frames)
    bbox_source: str = "detector"

    @property
    def center(self) -> Tuple[float, float]:
        x, y, w, h = self.bbox
        return (x + w / 2.0, y + h / 2.0)


class Dataset(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> Dict[str, Any]: ...


class DatasetManager(Protocol):
    def split(self, method: str = "video_level", seed: int = 0, ratios=(0.7, 0.2, 0.1)) -> Dict[str, Any]: ...
    def k_fold(self, k: int, seed: int = 0) -> Iterable[Dict[str, Any]]: ...
    def to_pytorch_dataloader(self, subset_name: str, batch_size: int, transforms=None): ...


class PreprocessingModule(Protocol):
    name: str
    def apply_to_frame(self, frame: np.ndarray) -> np.ndarray: ...
    def apply_to_video(self, video_path: str) -> str: ...


class TrackingModel(Protocol):
    name: str
    def train(self, train_dataset: Dataset, val_dataset: Optional[Dataset] = None, seed: int = 0, output_dir: Optional[str] = None) -> Dict[str, Any]: ...
    def predict(self, video_path: str) -> List[FramePrediction]: ...
    def load_checkpoint(self, ckpt_path: str): ...


class Evaluator(Protocol):
    def evaluate(self, predictions: Dict[str, List[FramePrediction]], gt: Dict[str, Any], out_dir: str) -> Dict[str, Any]: ...
