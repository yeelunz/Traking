from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QKeyEvent, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import QWidget

from .data import MaskAnnotation, SegmentationProject


class MaskCanvas(QWidget):
    maskChanged = Signal(int)
    brushSizeChanged = Signal(int)
    annotationCreated = Signal(int)
    annotationsCleared = Signal()

    HANDLE_COLOR = QColor(255, 0, 0, 180)

    def __init__(self, project: SegmentationProject, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.project = project
        self.frame_index = 0
        self.frame: Optional[np.ndarray] = None
        self.active_annotation: Optional[MaskAnnotation] = None
        self.annotations = {}
        self.preview_mode = False
        self.show_mask_overlay = True
        self.brush_radius = 18
        self._stroke_active = False
        self._stroke_mode = "paint"
        self._cursor_mode = "paint"
        self._mask_snapshot: Optional[np.ndarray] = None
        self._undo_stack: list[np.ndarray] = []
        self._max_undo = 30
        self.base_color = QColor(0, 190, 120)
        self.highlight_color = QColor(255, 85, 140)
        self.base_alpha = 140
        self.highlight_alpha = 210
        self.cursor_pos: Optional[QPointF] = None
        self._category_resolver: Optional[Callable[[], int]] = None
        self.background_offset = 0

    # ------------------------------------------------------------------
    # Data feed --------------------------------------------------------
    # ------------------------------------------------------------------
    def load_frame(self, frame_index: int) -> None:
        frame = self.project.load_frame(frame_index)
        if frame is None:
            return
        self.frame_index = frame_index
        self.frame = frame
        self.annotations = self.project.get_annotations(frame_index)
        if self.active_annotation and self.active_annotation.track_id not in self.annotations:
            self.active_annotation = None
        self.update()

    def set_active_track(self, track_id: Optional[int]) -> None:
        if track_id is None:
            self.active_annotation = None
        else:
            self.active_annotation = self.annotations.get(track_id)
        self.update()

    def set_category_resolver(self, resolver: Callable[[], int]) -> None:
        self._category_resolver = resolver

    # ------------------------------------------------------------------
    # Painting helpers -------------------------------------------------
    # ------------------------------------------------------------------
    def paintEvent(self, event):  # noqa: D401 - Qt override
        if self.frame is None:
            return
        painter = QPainter(self)
        target_rect = self.rect_for_image()
        frame_to_draw = self.frame
        if self.background_offset != 0:
            frame_to_draw = np.clip(
                frame_to_draw.astype(np.int16) + int(self.background_offset),
                0,
                255,
            ).astype(np.uint8)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        qimg = self._to_qimage(frame_to_draw)
        painter.drawImage(target_rect, qimg)

        if self.preview_mode:
            painter.fillRect(target_rect, QColor(20, 20, 20, 140))

        if self.show_mask_overlay or self.preview_mode:
            h, w, _ = self.frame.shape
            for ann in self.annotations.values():
                mask = ann.mask
                tint_preview = self.preview_mode
                tinted = self._mask_to_image(
                    mask,
                    highlight=tint_preview,
                )
                painter.drawImage(target_rect, tinted)
            painter.setOpacity(1.0)

        if self.cursor_pos is not None and target_rect.contains(self.cursor_pos):
            _, w, _ = self.frame.shape
            scale = target_rect.width() / w if w else 1.0
            radius = max(1.5, self.brush_radius * scale)
            color = self.highlight_color if self._cursor_mode == "paint" else QColor(255, 80, 80)
            pen = QPen(color)
            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setOpacity(0.9)
            painter.drawEllipse(self.cursor_pos, radius, radius)
            painter.setOpacity(1.0)

    def rect_for_image(self):
        scale, offset_x, offset_y = self._compute_transform()
        h, w, _ = self.frame.shape
        width = int(w * scale)
        height = int(h * scale)
        return QRectF(offset_x, offset_y, width, height)

    def _overlay_rgba(self, highlight: bool) -> Tuple[int, int, int, int]:
        color = self.highlight_color if highlight else self.base_color
        alpha = self.highlight_alpha if highlight else self.base_alpha
        alpha = max(0, min(255, alpha))
        return color.red(), color.green(), color.blue(), alpha

    def _mask_to_image(self, mask: np.ndarray, *, highlight: bool) -> QImage:
        if mask.ndim > 2:
            mask = np.squeeze(mask)
        binary = (mask > 0).astype(np.uint8)
        r, g, b, a = self._overlay_rgba(highlight)
        color = np.zeros((*binary.shape, 4), dtype=np.uint8)
        color[..., 0] = r
        color[..., 1] = g
        color[..., 2] = b
        color[..., 3] = np.where(binary > 0, a, 0).astype(np.uint8)
        return QImage(color.data, color.shape[1], color.shape[0], QImage.Format.Format_RGBA8888).copy()

    def set_overlay_alpha(self, alpha: int) -> None:
        self.base_alpha = max(0, min(255, int(alpha)))
        if self.base_alpha <= 0:
            self.highlight_alpha = 0
        else:
            scaled = int(round(self.base_alpha * 1.6))
            self.highlight_alpha = max(self.base_alpha, min(255, scaled))
        self.update()

    def set_overlay_colors(self, base: QColor, highlight: QColor) -> None:
        self.base_color = base
        self.highlight_color = highlight
        self.update()

    def set_background_offset(self, offset: int) -> None:
        offset = int(max(-100, min(100, offset)))
        if offset == self.background_offset:
            return
        self.background_offset = offset
        self.update()

    def set_brush_radius(self, radius: int) -> None:
        radius = int(max(1, min(250, radius)))
        if radius == self.brush_radius:
            return
        self.brush_radius = radius
        self.brushSizeChanged.emit(self.brush_radius)
        self.update()

    def refresh_annotations(self) -> None:
        self.annotations = self.project.get_annotations(self.frame_index)
        if self.active_annotation and self.active_annotation.track_id not in self.annotations:
            self.active_annotation = None
        self.update()

    def _to_qimage(self, frame: np.ndarray) -> QImage:
        h, w, _ = frame.shape
        return QImage(frame.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()

    def _compute_transform(self):
        if self.frame is None:
            return 1.0, 0, 0
        h, w, _ = self.frame.shape
        scale = min(self.width() / w, self.height() / h)
        scaled_w = int(w * scale)
        scaled_h = int(h * scale)
        offset_x = (self.width() - scaled_w) // 2
        offset_y = (self.height() - scaled_h) // 2
        return scale, offset_x, offset_y

    def _widget_to_image(self, pos: QPointF) -> Optional[tuple[int, int]]:
        if self.frame is None:
            return None
        scale, offset_x, offset_y = self._compute_transform()
        h, w, _ = self.frame.shape
        x = (pos.x() - offset_x) / scale
        y = (pos.y() - offset_y) / scale
        if x < 0 or y < 0 or x >= w or y >= h:
            return None
        return int(x), int(y)

    # ------------------------------------------------------------------
    # Interaction ------------------------------------------------------
    # ------------------------------------------------------------------
    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self.set_brush_radius(self.brush_radius + 2)
        elif delta < 0:
            self.set_brush_radius(self.brush_radius - 2)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Space:
            self.preview_mode = True
            self.show_mask_overlay = True
            self.update()
            return
        if event.key() == Qt.Key.Key_Z and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.undo()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Space:
            self.preview_mode = False
            self.update()
            return
        super().keyReleaseEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() not in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            return
        self.cursor_pos = event.position()
        coords = self._widget_to_image(event.position())
        if coords is None:
            return
        if self.active_annotation is None and not self._ensure_active_annotation():
            return
        self._stroke_active = True
        self._stroke_mode = "paint" if event.button() == Qt.MouseButton.LeftButton else "erase"
        self._cursor_mode = self._stroke_mode
        self._mask_snapshot = self.active_annotation.mask.copy()
        self._apply_brush(coords, mode=self._stroke_mode)

    def mouseMoveEvent(self, event: QMouseEvent):
        self.cursor_pos = event.position()
        if self._stroke_active and self.active_annotation is not None:
            coords = self._widget_to_image(event.position())
            if coords is not None:
                self._apply_brush(coords, mode=self._stroke_mode)
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() not in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            return
        if not self._stroke_active or self.active_annotation is None:
            self._cursor_mode = "paint"
            return
        self.cursor_pos = event.position()
        self._stroke_active = False
        if self._mask_snapshot is not None:
            self._push_undo(self._mask_snapshot)
            self._mask_snapshot = None
        self.maskChanged.emit(self.active_annotation.track_id)
        self._cursor_mode = "paint"

    def leaveEvent(self, event):  # noqa: D401 - Qt override
        self.cursor_pos = None
        self.update()
        super().leaveEvent(event)

    def _apply_brush(self, coords: tuple[int, int], *, mode: str) -> None:
        if self.active_annotation is None:
            return
        mask = self.active_annotation.mask.copy()
        x, y = coords
        radius = self.brush_radius
        yy, xx = np.ogrid[-y:mask.shape[0]-y, -x:mask.shape[1]-x]
        circle = xx * xx + yy * yy <= radius * radius
        if mode == "paint":
            mask[circle] = 255
        else:
            mask[circle] = 0
        self.active_annotation.update_mask(mask)
        self.update()

    def _ensure_active_annotation(self) -> bool:
        if self.active_annotation is not None:
            return True
        if self.frame is None:
            return False
        category_id = 1
        if self._category_resolver is not None:
            try:
                category_id = int(self._category_resolver())
            except Exception:
                category_id = 1
        blank = np.zeros((self.project.height, self.project.width), dtype=np.uint8)
        anno = self.project.new_annotation(self.frame_index, category_id, blank)
        self.annotations = self.project.get_annotations(self.frame_index)
        self.active_annotation = anno
        self._undo_stack.clear()
        self.annotationCreated.emit(anno.track_id)
        self.update()
        return True

    def clear_frame_annotations(self) -> None:
        had_annotations = bool(self.annotations)
        self.project.clear_frame(self.frame_index)
        self.annotations = {}
        self.active_annotation = None
        self._undo_stack.clear()
        self.update()
        if had_annotations:
            self.annotationsCleared.emit()

    # ------------------------------------------------------------------
    # Undo --------------------------------------------------------------
    # ------------------------------------------------------------------
    def _push_undo(self, snapshot: np.ndarray) -> None:
        self._undo_stack.append(snapshot)
        if len(self._undo_stack) > self._max_undo:
            self._undo_stack.pop(0)

    def undo(self) -> None:
        if self.active_annotation is None or not self._undo_stack:
            return
        previous = self._undo_stack.pop()
        self.active_annotation.update_mask(previous)
        self.update()
        self.maskChanged.emit(self.active_annotation.track_id)

    # ------------------------------------------------------------------
    def set_show_mask(self, enabled: bool) -> None:
        self.show_mask_overlay = enabled
        self.update()


__all__ = ["MaskCanvas"]
