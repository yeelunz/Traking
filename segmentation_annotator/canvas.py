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
    zoomChanged = Signal(float)

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
        self.zoom = 1.0
        self.min_zoom = 0.25
        self.max_zoom = 5.0
        self._pan = QPointF(0.0, 0.0)
        self._panning = False
        self._pan_last = QPointF(0.0, 0.0)
        self._minimap_rect = None
        self._minimap_dragging = False

    # ------------------------------------------------------------------
    # Data feed --------------------------------------------------------
    # ------------------------------------------------------------------
    def load_frame(self, frame_index: int) -> None:
        frame = self.project.load_frame(frame_index)
        if frame is None:
            return
        self.frame_index = frame_index
        self.frame = frame
        previous_track_id = self.active_annotation.track_id if self.active_annotation is not None else None
        self.annotations = self.project.get_annotations(frame_index)
        if previous_track_id is not None:
            self.active_annotation = self.annotations.get(previous_track_id)
        else:
            self.active_annotation = None
        if self.active_annotation is None and self.annotations:
            self.active_annotation = next(iter(self.annotations.values()))
        if self.active_annotation is not None:
            self._undo_stack.clear()
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
        self._minimap_rect = None
        if self.zoom <= 1.0:
            self._minimap_dragging = False
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

        if self.zoom > 1.0:
            self._draw_minimap(painter, qimg)

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

    def _visible_image_rect(self) -> Optional[QRectF]:
        if self.frame is None:
            return None
        scale, offset_x, offset_y = self._compute_transform()
        h, w, _ = self.frame.shape
        x0 = (0.0 - offset_x) / scale
        y0 = (0.0 - offset_y) / scale
        x1 = (self.width() - offset_x) / scale
        y1 = (self.height() - offset_y) / scale
        x0 = max(0.0, min(float(w), x0))
        y0 = max(0.0, min(float(h), y0))
        x1 = max(0.0, min(float(w), x1))
        y1 = max(0.0, min(float(h), y1))
        if x1 <= x0 or y1 <= y0:
            return QRectF(0.0, 0.0, float(w), float(h))
        return QRectF(x0, y0, x1 - x0, y1 - y0)

    def _draw_minimap(self, painter: QPainter, frame_image: QImage) -> None:
        if self.frame is None:
            return
        h, w, _ = self.frame.shape
        if w <= 0 or h <= 0:
            return
        max_dim = min(200, int(self.width() * 0.3))
        if max_dim < 80:
            return
        aspect = h / w
        map_w = max_dim
        map_h = int(round(map_w * aspect))
        if map_h > max_dim:
            map_h = max_dim
            map_w = int(round(map_h / aspect))
        if map_w <= 0 or map_h <= 0:
            return
        margin = 12
        x = self.width() - map_w - margin
        y = self.height() - map_h - margin
        if x < margin:
            x = float(margin)
        if y < margin:
            y = float(margin)
        self._minimap_rect = QRectF(float(x), float(y), float(map_w), float(map_h))

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        backdrop = self._minimap_rect.adjusted(-4.0, -4.0, 4.0, 4.0)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 160))
        painter.drawRoundedRect(backdrop, 6.0, 6.0)

        mini_img = frame_image.scaled(
            int(self._minimap_rect.width()),
            int(self._minimap_rect.height()),
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QColor(220, 220, 220))
        painter.drawImage(self._minimap_rect, mini_img)
        painter.drawRect(self._minimap_rect)

        view_rect = self._visible_image_rect()
        if view_rect is not None and view_rect.width() > 0 and view_rect.height() > 0:
            scale_x = self._minimap_rect.width() / w
            scale_y = self._minimap_rect.height() / h
            viewport = QRectF(
                self._minimap_rect.left() + view_rect.left() * scale_x,
                self._minimap_rect.top() + view_rect.top() * scale_y,
                view_rect.width() * scale_x,
                view_rect.height() * scale_y,
            ).intersected(self._minimap_rect)
            if viewport.width() > 0 and viewport.height() > 0:
                painter.setPen(QPen(QColor(255, 200, 40, 230), 2))
                painter.setBrush(QColor(255, 200, 40, 60))
                painter.drawRect(viewport)

        painter.restore()

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
        if self.frame is None or self.width() <= 0 or self.height() <= 0:
            return 1.0, 0.0, 0.0
        h, w, _ = self.frame.shape
        base_scale = min(self.width() / w, self.height() / h)
        scale = base_scale * self.zoom
        scaled_w = w * scale
        scaled_h = h * scale
        base_offset_x = (self.width() - scaled_w) / 2.0
        base_offset_y = (self.height() - scaled_h) / 2.0
        offset_x = base_offset_x + self._pan.x()
        offset_y = base_offset_y + self._pan.y()
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
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if delta > 0:
                self._apply_zoom(1.1, anchor=event.position())
            elif delta < 0:
                self._apply_zoom(1 / 1.1, anchor=event.position())
        else:
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
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._minimap_rect is not None
            and self._minimap_rect.contains(event.position())
        ):
            image_point = self._minimap_point_to_image(event.position())
            if image_point is not None:
                self._minimap_dragging = True
                self.center_view_on(image_point)
            return
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_last = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
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
        if self._minimap_dragging:
            image_point = self._minimap_point_to_image(event.position())
            if image_point is not None:
                self.center_view_on(image_point)
            return
        if self._panning:
            delta = event.position() - self._pan_last
            self._pan_last = event.position()
            self._apply_pan(delta)
            self.update()
            return
        if self._stroke_active and self.active_annotation is not None:
            coords = self._widget_to_image(event.position())
            if coords is not None:
                self._apply_brush(coords, mode=self._stroke_mode)
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self._minimap_dragging:
            self._minimap_dragging = False
            return
        if event.button() == Qt.MouseButton.MiddleButton and self._panning:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return
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
        self._minimap_dragging = False
        if self._panning:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
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

        if self.annotations:
            # Reuse the existing annotation for this frame instead of overwriting it.
            track_id, anno = next(iter(self.annotations.items()))
            self.active_annotation = anno
            # Reset undo history so new strokes start a fresh stack.
            self._undo_stack.clear()
            self.update()
            return True

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

    def _apply_zoom(self, factor: float, *, anchor: Optional[QPointF] = None) -> None:
        if anchor is None:
            anchor = QPointF(self.width() / 2, self.height() / 2)
        image_point = self._widget_to_image(anchor)
        new_zoom = self.zoom * factor
        self.set_zoom(new_zoom, anchor, image_point=image_point)

    def set_zoom(self, zoom: float, anchor: Optional[QPointF] = None, *, image_point: Optional[tuple[int, int]] = None) -> None:
        zoom = float(max(self.min_zoom, min(self.max_zoom, zoom)))
        if abs(zoom - self.zoom) < 1e-3:
            return
        if anchor is None:
            anchor = QPointF(self.width() / 2, self.height() / 2)
        if image_point is None:
            image_point = self._widget_to_image(anchor)
        self.zoom = zoom
        if image_point is not None:
            self._set_pan_to_keep_point(image_point, anchor)
        else:
            self._pan = QPointF(0.0, 0.0)
            self._clamp_pan()
        self.zoomChanged.emit(self.zoom)
        self.update()

    def reset_view(self) -> None:
        self.zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._clamp_pan()
        self.zoomChanged.emit(self.zoom)
        self.update()

    def _apply_pan(self, delta: QPointF) -> None:
        self._pan = QPointF(self._pan.x() + delta.x(), self._pan.y() + delta.y())
        self._clamp_pan()

    def _set_pan_to_keep_point(self, image_point: tuple[int, int], anchor: QPointF) -> None:
        if self.frame is None or self.width() <= 0 or self.height() <= 0:
            self._pan = QPointF(0.0, 0.0)
            return
        image_x, image_y = image_point
        h, w, _ = self.frame.shape
        base_scale = min(self.width() / w, self.height() / h)
        scale = base_scale * self.zoom
        scaled_w = w * scale
        scaled_h = h * scale
        base_offset_x = (self.width() - scaled_w) / 2.0
        base_offset_y = (self.height() - scaled_h) / 2.0
        desired_offset_x = anchor.x() - image_x * scale
        desired_offset_y = anchor.y() - image_y * scale
        self._pan = QPointF(desired_offset_x - base_offset_x, desired_offset_y - base_offset_y)
        self._clamp_pan()

    def _clamp_pan(self) -> None:
        if self.frame is None or self.width() <= 0 or self.height() <= 0:
            self._pan = QPointF(0.0, 0.0)
            return
        h, w, _ = self.frame.shape
        base_scale = min(self.width() / w, self.height() / h)
        scale = base_scale * self.zoom
        scaled_w = w * scale
        scaled_h = h * scale
        base_offset_x = (self.width() - scaled_w) / 2.0
        base_offset_y = (self.height() - scaled_h) / 2.0

        if scaled_w <= self.width():
            final_offset_x = base_offset_x
        else:
            min_offset_x = self.width() - scaled_w
            max_offset_x = 0.0
            offset_x = base_offset_x + self._pan.x()
            final_offset_x = min(max(offset_x, min_offset_x), max_offset_x)
        if scaled_h <= self.height():
            final_offset_y = base_offset_y
        else:
            min_offset_y = self.height() - scaled_h
            max_offset_y = 0.0
            offset_y = base_offset_y + self._pan.y()
            final_offset_y = min(max(offset_y, min_offset_y), max_offset_y)

        self._pan = QPointF(final_offset_x - base_offset_x, final_offset_y - base_offset_y)

    def _minimap_point_to_image(self, pos: QPointF) -> Optional[tuple[int, int]]:
        if self.frame is None or self._minimap_rect is None:
            return None
        if self._minimap_rect.width() <= 0 or self._minimap_rect.height() <= 0:
            return None
        rel_x = (pos.x() - self._minimap_rect.left()) / self._minimap_rect.width()
        rel_y = (pos.y() - self._minimap_rect.top()) / self._minimap_rect.height()
        rel_x = max(0.0, min(1.0, rel_x))
        rel_y = max(0.0, min(1.0, rel_y))
        h, w, _ = self.frame.shape
        image_x = int(round(rel_x * (w - 1))) if w > 1 else 0
        image_y = int(round(rel_y * (h - 1))) if h > 1 else 0
        return image_x, image_y

    def center_view_on(self, image_point: tuple[int, int]) -> None:
        anchor = QPointF(self.width() / 2.0, self.height() / 2.0)
        self._set_pan_to_keep_point(image_point, anchor)
        self._clamp_pan()
        self.update()


__all__ = ["MaskCanvas"]
