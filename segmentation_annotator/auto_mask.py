from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
try:
    from skimage.segmentation import morphological_chan_vese, morphological_geodesic_active_contour  # type: ignore
except Exception:  # pragma: no cover - scikit-image is optional in unit tests
    morphological_chan_vese = None  # type: ignore
    morphological_geodesic_active_contour = None  # type: ignore

try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover - ultralytics is optional in unit tests
    YOLO = None  # type: ignore


@dataclass
class AutoMaskResult:
    bbox: Tuple[float, float, float, float]
    mask: np.ndarray
    score: float
    category_id: int
    label: str


def _ensure_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.normalize(gray, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(mask)
    if num_labels <= 2:
        return mask * 255
    areas = np.bincount(labels.ravel())
    areas[:1] = 0  # ignore background
    best_label = int(np.argmax(areas)) if areas.size > 1 else 0
    filtered = (labels == best_label).astype(np.uint8)
    return filtered * 255


def _extract_roi_with_reflect(
    frame: np.ndarray,
    bbox: Tuple[float, float, float, float],
    expand_ratio: float,
) -> Tuple[np.ndarray, Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    height, width = frame.shape[:2]
    x, y, w, h = bbox
    cx = x + w / 2.0
    cy = y + h / 2.0
    half_w = (w / 2.0) + (w * expand_ratio)
    half_h = (h / 2.0) + (h * expand_ratio)

    x0 = int(np.floor(cx - half_w))
    y0 = int(np.floor(cy - half_h))
    x1 = int(np.ceil(cx + half_w))
    y1 = int(np.ceil(cy + half_h))

    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - width)
    pad_bottom = max(0, y1 - height)

    x0_clip = max(0, x0)
    y0_clip = max(0, y0)
    x1_clip = min(width, x1)
    y1_clip = min(height, y1)

    roi = frame[y0_clip:y1_clip, x0_clip:x1_clip]
    if pad_top or pad_bottom or pad_left or pad_right:
        roi = cv2.copyMakeBorder(
            roi,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_REFLECT_101,
        )

    return roi, (x0_clip, y0_clip, x1_clip, y1_clip), (pad_top, pad_bottom, pad_left, pad_right)


def _strip_padding(mask: np.ndarray, pads: Tuple[int, int, int, int]) -> np.ndarray:
    pad_top, pad_bottom, pad_left, pad_right = pads
    trimmed = mask
    if pad_top:
        trimmed = trimmed[pad_top:]
    if pad_bottom:
        trimmed = trimmed[:-pad_bottom]
    if pad_left:
        trimmed = trimmed[:, pad_left:]
    if pad_right:
        trimmed = trimmed[:, :-pad_right]
    return trimmed


def _remove_boundary_components(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    if not np.any(binary):
        return binary
    num_labels, labels = cv2.connectedComponents(binary)
    if num_labels <= 1:
        return binary
    keep = np.zeros_like(binary)
    h, w = binary.shape
    for label in range(1, num_labels):
        component = labels == label
        if (
            component[0, :].any()
            or component[h - 1, :].any()
            or component[:, 0].any()
            or component[:, w - 1].any()
        ):
            continue
        keep[component] = 1
    if keep.any():
        return keep
    return binary


def _guided_filter(image: np.ndarray, guide: np.ndarray, radius: int = 4, eps: float = 1e-3) -> np.ndarray:
    I = image.astype(np.float32)
    p = guide.astype(np.float32)
    win_size = (radius * 2 + 1, radius * 2 + 1)

    mean_I = cv2.boxFilter(I, ddepth=-1, ksize=win_size, normalize=True)
    mean_p = cv2.boxFilter(p, ddepth=-1, ksize=win_size, normalize=True)
    mean_Ip = cv2.boxFilter(I * p, ddepth=-1, ksize=win_size, normalize=True)
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(I * I, ddepth=-1, ksize=win_size, normalize=True)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    mean_a = cv2.boxFilter(a, ddepth=-1, ksize=win_size, normalize=True)
    b = mean_p - a * mean_I
    mean_b = cv2.boxFilter(b, ddepth=-1, ksize=win_size, normalize=True)

    q = mean_a * I + mean_b
    return q


class AutoMaskGenerator:
    """YOLO + Chan–Vese helper used by the segmentation annotator."""

    def __init__(self, weights_path: str, *, device: Optional[str] = None, conf: float = 0.25):
        if YOLO is None:
            raise RuntimeError("ultralytics.YOLO is not available; please install the 'ultralytics' package.")
        self.model = YOLO(weights_path)
        self.device = device or "cpu"
        self.conf = conf

    def __call__(
        self,
        frame: np.ndarray,
        *,
        max_detections: int = 1,
        margin: float = 0.2,
        num_iter: int = 300,
    ) -> List[AutoMaskResult]:
        """Run detection and locally-refined Chan–Vese segmentation on a frame."""

        if morphological_chan_vese is None:
            raise RuntimeError("scikit-image is required for morphological Chan–Vese segmentation.")

        results = self.model.predict(frame, conf=self.conf, device=self.device, verbose=False)
        if not results:
            return []
        det = results[0]
        if det.boxes is None or det.boxes.xyxy is None:
            return []

        margin = max(0.05, float(margin))
        num_iter = int(max(10, num_iter))

        boxes = det.boxes.xyxy.cpu().numpy()
        scores = det.boxes.conf.cpu().numpy()
        cls = det.boxes.cls.cpu().numpy().astype(int)
        order = np.argsort(-scores)
        width = frame.shape[1]
        height = frame.shape[0]

        # 同幀只保留一個遮罩，因此僅處理信心最高的偵測
        max_detections = 1
        outputs: List[AutoMaskResult] = []

        for idx in order[:max_detections]:
            x0, y0, x1, y1 = boxes[idx]
            w = float(max(1.0, x1 - x0))
            h = float(max(1.0, y1 - y0))
            base_bbox = (float(x0), float(y0), w, h)

            roi, bounds, pads = _extract_roi_with_reflect(frame, base_bbox, expand_ratio=margin)
            if roi.size == 0:
                continue

            roi_gray = _ensure_gray(roi)
            init_mask = np.zeros_like(roi_gray, dtype=np.uint8)
            cx = roi_gray.shape[1] // 2
            cy = roi_gray.shape[0] // 2
            axes = (max(1, roi_gray.shape[1] // 3), max(1, roi_gray.shape[0] // 3))
            cv2.ellipse(init_mask, (cx, cy), axes, 0, 0, 360, 1, -1)

            level_set = init_mask.astype(bool)
            seg = morphological_chan_vese(
                roi_gray,
                max(1, num_iter),
                init_level_set=level_set,
                smoothing=3,
                lambda1=1,
                lambda2=1,
            )
            mask = seg.astype(np.uint8) * 255

            # 邊緣阻力：導入局部梯度資訊
            edges = cv2.Canny((roi_gray * 255).astype(np.uint8), 30, 80)
            edge_weight = np.exp(-(edges.astype(np.float32) / 255.0) * 10.0)
            denom = float(np.ptp(edge_weight))
            edge_weight = (edge_weight - edge_weight.min()) / (denom + 1e-6)
            if morphological_geodesic_active_contour is not None:
                mask = morphological_geodesic_active_contour(
                    edge_weight,
                    max(60, num_iter // 4),
                    init_level_set=(mask > 0),
                    smoothing=1,
                    threshold="auto",
                    balloon=-1,
                ).astype(np.uint8) * 255
            else:
                mask = (mask.astype(np.float32) * edge_weight > 0.5).astype(np.uint8) * 255

            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            guided = _guided_filter(roi_gray, mask / 255.0, radius=4, eps=1e-3)
            mask = (guided > 0.5).astype(np.uint8) * 255

            mask = _strip_padding(mask, pads)
            mask = _remove_boundary_components(mask)
            mask = _largest_component(mask)

            if np.count_nonzero(mask) == 0:
                continue

            full_mask = np.zeros((height, width), dtype=np.uint8)
            x0_clip, y0_clip, x1_clip, y1_clip = bounds
            roi_h = max(0, y1_clip - y0_clip)
            roi_w = max(0, x1_clip - x0_clip)
            if roi_h == 0 or roi_w == 0:
                continue
            trimmed_mask = mask[:roi_h, :roi_w]
            if trimmed_mask.shape[0] != roi_h or trimmed_mask.shape[1] != roi_w:
                trimmed_mask = cv2.resize(trimmed_mask, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
            full_mask[y0_clip:y1_clip, x0_clip:x1_clip] = trimmed_mask

            outputs.append(
                AutoMaskResult(
                    bbox=base_bbox,
                    mask=full_mask,
                    score=float(scores[idx]),
                    category_id=int(cls[idx]) + 1,
                    label=str(self.model.names.get(int(cls[idx]), f"cls_{int(cls[idx])}")),
                )
            )
            break

        return outputs


__all__ = ["AutoMaskGenerator", "AutoMaskResult"]
