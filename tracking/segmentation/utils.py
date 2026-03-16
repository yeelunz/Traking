from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from ..core.interfaces import MaskStats


@dataclass
class BoundingBox:
    x: float
    y: float
    w: float
    h: float

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.w, self.h)

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)


def compute_mask_stats(mask: np.ndarray) -> MaskStats:
    if mask.dtype != np.uint8:
        binary = (mask > 0).astype(np.uint8)
    else:
        binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return MaskStats(
            area_px=0.0,
            bbox=(0.0, 0.0, 0.0, 0.0),
            centroid=(0.0, 0.0),
            perimeter_px=0.0,
            equivalent_diameter_px=0.0,
        )
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
    eq_diameter = float(np.sqrt(4.0 * area / math.pi)) if area > 0 else 0.0
    return MaskStats(
        area_px=area,
        bbox=(float(x), float(y), float(w), float(h)),
        centroid=(cx, cy),
        perimeter_px=perimeter,
        equivalent_diameter_px=eq_diameter,
    )


def expand_bbox(bbox: Tuple[float, float, float, float], pad_fraction: float, image_shape: Tuple[int, int]) -> BoundingBox:
    x, y, w, h = bbox
    pad_fraction = max(0.0, float(pad_fraction))
    pad_w = w * pad_fraction
    pad_h = h * pad_fraction
    cx = x + w / 2.0
    cy = y + h / 2.0
    new_w = w + pad_w * 2.0
    new_h = h + pad_h * 2.0
    x0 = cx - new_w / 2.0
    y0 = cy - new_h / 2.0
    img_h, img_w = image_shape
    x0 = max(0.0, x0)
    y0 = max(0.0, y0)
    if x0 + new_w > img_w:
        new_w = img_w - x0
    if y0 + new_h > img_h:
        new_h = img_h - y0
    return BoundingBox(float(x0), float(y0), float(new_w), float(new_h))


def crop_with_bbox(image: np.ndarray, bbox: BoundingBox) -> np.ndarray:
    x0 = max(0, int(round(bbox.x)))
    y0 = max(0, int(round(bbox.y)))
    x1 = min(image.shape[1], int(round(bbox.x + bbox.w)))
    y1 = min(image.shape[0], int(round(bbox.y + bbox.h)))
    if x1 <= x0 or y1 <= y0:
        # Degenerate crop — return an empty ROI with correct channel count
        ch = image.shape[2:] if image.ndim > 2 else ()
        return np.empty((0, 0) + ch, dtype=image.dtype)
    return image[y0:y1, x0:x1]


def place_mask_on_canvas(canvas_shape: Tuple[int, int], mask_roi: np.ndarray, bbox: BoundingBox) -> np.ndarray:
    canvas = np.zeros(canvas_shape, dtype=np.uint8)
    x0 = int(round(bbox.x))
    y0 = int(round(bbox.y))
    h, w = mask_roi.shape[:2]

    # Clamp target slice so we never assign outside the canvas bounds
    x0_clamped = max(0, min(x0, canvas.shape[1]))
    y0_clamped = max(0, min(y0, canvas.shape[0]))
    if x0_clamped >= canvas.shape[1] or y0_clamped >= canvas.shape[0]:
        return canvas
    max_w = canvas.shape[1] - x0_clamped
    max_h = canvas.shape[0] - y0_clamped
    copy_w = min(w, max_w)
    copy_h = min(h, max_h)
    if copy_w <= 0 or copy_h <= 0:
        return canvas
    canvas[y0_clamped:y0_clamped + copy_h, x0_clamped:x0_clamped + copy_w] = mask_roi[:copy_h, :copy_w]
    return canvas


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    if mask is None or mask.size == 0:
        return mask
    binary = (mask > 0).astype(np.uint8)
    if binary.ndim == 3:
        binary = binary[..., 0]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.size == 0:
        return np.zeros_like(binary, dtype=np.uint8)
    largest_idx = int(areas.argmax()) + 1
    largest_mask = (labels == largest_idx).astype(np.uint8)
    if mask.ndim == 3:
        largest_mask = np.expand_dims(largest_mask, axis=-1)
        largest_mask = np.repeat(largest_mask, mask.shape[-1], axis=-1)
    return (largest_mask * 255).astype(mask.dtype)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    if mask is None or mask.size == 0:
        return mask
    binary = (mask > 0).astype(np.uint8)
    if binary.ndim == 3:
        binary = binary[..., 0]
    # use morphology closing followed by hole filling via flood fill
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    h, w = closed.shape[:2]
    flood = closed.copy()
    mask_ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask_ff, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    filled = closed | (flood_inv > 0).astype(np.uint8)
    if mask.ndim == 3:
        filled = np.expand_dims(filled, axis=-1)
        filled = np.repeat(filled, mask.shape[-1], axis=-1)
    return (filled * 255).astype(mask.dtype)
