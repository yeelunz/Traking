"""Segmentation annotator package."""

from .data import (
	MASK_ROOT_DEFAULT,
	MaskAnnotation,
	MaskMetadata,
	MotionSample,
	SegmentationProject,
	TrackSummary,
)
from .auto_mask import AutoMaskGenerator, AutoMaskResult
from .main_window import SegmentAnnotatorWindow, launch

__all__ = [
	"MASK_ROOT_DEFAULT",
	"MaskAnnotation",
	"MaskMetadata",
	"MotionSample",
	"SegmentationProject",
	"TrackSummary",
	"AutoMaskGenerator",
	"AutoMaskResult",
	"SegmentAnnotatorWindow",
	"launch",
]
