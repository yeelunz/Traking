from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QColor
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .auto_mask import AutoMaskGenerator, AutoMaskResult
from .canvas import MaskCanvas
from .data import SegmentationProject


def _ensure_labels_file(video_path: str) -> List[str]:
    root = os.path.dirname(video_path)
    labels_path = os.path.join(root, "labels.txt")
    if not os.path.exists(labels_path):
        with open(labels_path, "w", encoding="utf-8") as f:
            f.write("median_nerve\n")
    with open(labels_path, "r", encoding="utf-8") as f:
        categories = [line.strip() for line in f if line.strip()]
    return categories or ["median_nerve"]


class SegmentAnnotatorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Segmentation Annotator")
        self.resize(1400, 880)
        self.project: Optional[SegmentationProject] = None
        self.canvas: Optional[MaskCanvas] = None
        self.auto_generator: Optional[AutoMaskGenerator] = None
        self.auto_weights_path: Optional[str] = None
        self.dirty = False
        self.dataset_root: Optional[Path] = None
        self.video_paths: List[Path] = []
        self.current_video_index: int = -1
        self.video_annotation_counts: Dict[Path, int] = {}
        self.default_background_offset = 0
        self.overlay_presets = [
            ("葉綠", QColor(0, 190, 120), QColor(255, 85, 140)),
            ("天空藍", QColor(0, 160, 255), QColor(255, 120, 60)),
            ("暖橙", QColor(255, 150, 60), QColor(220, 60, 60)),
            ("薰衣草", QColor(190, 120, 255), QColor(255, 200, 90)),
            ("灰階", QColor(180, 180, 180), QColor(255, 255, 255)),
        ]

        self._build_ui()
        self._build_menu()

    # ------------------------------------------------------------------
    def _build_menu(self):
        file_menu = self.menuBar().addMenu("檔案")
        open_action = QAction("開啟資料夾", self)
        open_action.triggered.connect(self.open_folder)
        file_menu.addAction(open_action)

        help_menu = self.menuBar().addMenu("說明")
        shortcut_action = QAction("快捷鍵說明", self)
        shortcut_action.triggered.connect(self.show_shortcuts)
        help_menu.addAction(shortcut_action)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # Left: canvas + controls
        canvas_col = QVBoxLayout()
        self.canvas_layout = canvas_col
        self.canvas_placeholder = QLabel("請先開啟影片")
        self.canvas_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        canvas_col.addWidget(self.canvas_placeholder, 1)

        controls_row = QHBoxLayout()
        self.btn_prev = QPushButton("上一幀")
        self.btn_prev.clicked.connect(self.prev_frame)
        self.btn_next = QPushButton("下一幀")
        self.btn_next.clicked.connect(self.next_frame)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self.on_slider)
        self.frame_label = QLabel("Frame: -/-")
        self.frame_input = QSpinBox()
        self.frame_input.setRange(1, 1)
        self.frame_input.setValue(1)
        self.frame_input.setEnabled(False)
        self.frame_input.valueChanged.connect(self.on_frame_input_changed)
        controls_row.addWidget(self.btn_prev)
        controls_row.addWidget(self.btn_next)
        controls_row.addWidget(self.slider, 1)
        controls_row.addWidget(self.frame_input)
        controls_row.addWidget(self.frame_label)
        canvas_col.addLayout(controls_row)

        tool_row = QHBoxLayout()
        self.btn_clear_masks = QPushButton("清除遮罩")
        self.btn_clear_masks.clicked.connect(self.clear_masks)
        self.btn_auto = QPushButton("YOLO+ChanVese")
        self.btn_auto.clicked.connect(self.auto_mask)
        tool_row.addWidget(self.btn_clear_masks)
        tool_row.addWidget(self.btn_auto)
        canvas_col.addLayout(tool_row)

        root.addLayout(canvas_col, 3)

        # Right column: lists
        side = QVBoxLayout()
        side.addWidget(QLabel("影片"))
        self.list_videos = QListWidget()
        self.list_videos.currentRowChanged.connect(self.on_select_video_row)
        side.addWidget(self.list_videos, 2)

        side.addWidget(QLabel("類別"))
        self.list_categories = QListWidget()
        self.list_categories.itemClicked.connect(self.on_select_category)
        side.addWidget(self.list_categories, 1)

        side.addWidget(QLabel("筆刷設定"))
        brush_controls = QVBoxLayout()
        brush_row = QHBoxLayout()
        self.brush_label = QLabel("筆刷: 18")
        brush_row.addWidget(self.brush_label)
        self.brush_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_slider.setRange(1, 200)
        self.brush_slider.setValue(18)
        self.brush_slider.valueChanged.connect(self.on_brush_slider_changed)
        brush_row.addWidget(self.brush_slider, 1)
        self.brush_spin = QSpinBox()
        self.brush_spin.setRange(1, 200)
        self.brush_spin.setValue(18)
        self.brush_spin.valueChanged.connect(self.on_brush_spin_changed)
        brush_row.addWidget(self.brush_spin)
        self.btn_undo = QPushButton("復原")
        self.btn_undo.clicked.connect(self.undo)
        brush_row.addWidget(self.btn_undo)
        brush_controls.addLayout(brush_row)
        self.brush_hint = QLabel("左鍵＝筆刷，右鍵＝橡皮擦")
        self.brush_hint.setWordWrap(True)
        brush_controls.addWidget(self.brush_hint)
        side.addLayout(brush_controls)

        side.addWidget(QLabel("遮罩顯示"))
        overlay_row = QHBoxLayout()
        overlay_row.addWidget(QLabel("透明度"))
        self.slider_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_opacity.setRange(0, 100)
        self.slider_opacity.setValue(45)
        self.slider_opacity.valueChanged.connect(self.on_overlay_opacity_changed)
        overlay_row.addWidget(self.slider_opacity, 1)
        self.opacity_label = QLabel("45%")
        overlay_row.addWidget(self.opacity_label)
        overlay_row.addWidget(QLabel("顏色"))
        self.combo_overlay_color = QComboBox()
        for name, _base, _highlight in self.overlay_presets:
            self.combo_overlay_color.addItem(name)
        self.combo_overlay_color.currentIndexChanged.connect(self.on_overlay_color_changed)
        self.combo_overlay_color.setCurrentIndex(0)
        overlay_row.addWidget(self.combo_overlay_color)
        side.addLayout(overlay_row)
        self.apply_overlay_settings()

        side.addWidget(QLabel("背景亮度"))
        brightness_row = QHBoxLayout()
        self.slider_background = QSlider(Qt.Orientation.Horizontal)
        self.slider_background.setRange(-100, 100)
        self.slider_background.setValue(self.default_background_offset)
        self.slider_background.valueChanged.connect(self.on_background_brightness_changed)
        brightness_row.addWidget(self.slider_background, 1)
        self.background_label = QLabel("0")
        brightness_row.addWidget(self.background_label)
        self.btn_background_reset = QPushButton("回到預設")
        self.btn_background_reset.clicked.connect(self.reset_background_brightness)
        brightness_row.addWidget(self.btn_background_reset)
        side.addLayout(brightness_row)
        self.on_background_brightness_changed(self.default_background_offset)

        side.addWidget(QLabel("已標註幀"))
        self.list_frames = QListWidget()
        self.list_frames.itemClicked.connect(self.on_select_frame_item)
        side.addWidget(self.list_frames, 2)

        self.status_label = QLabel("Ready")
        side.addWidget(self.status_label)

        root.addLayout(side, 1)

    # ------------------------------------------------------------------
    def ensure_canvas(self):
        if self.canvas is not None:
            return
        if self.project is None:
            return
        self.canvas = MaskCanvas(self.project)
        self.canvas.maskChanged.connect(self.on_mask_changed)
        self.canvas.brushSizeChanged.connect(self.on_canvas_brush_size_changed)
        self.canvas.annotationCreated.connect(self.on_canvas_annotation_created)
        self.canvas.annotationsCleared.connect(self.on_canvas_annotations_cleared)
        self.canvas.set_category_resolver(self.current_category_id)
        if self.canvas_placeholder is not None:
            self.canvas_placeholder.hide()
            self.canvas_layout.removeWidget(self.canvas_placeholder)
            self.canvas_placeholder.deleteLater()
            self.canvas_placeholder = None
        self.canvas_layout.insertWidget(0, self.canvas, 1)
        self.on_canvas_brush_size_changed(self.canvas.brush_radius)
        self.apply_overlay_settings()
        self.canvas.set_background_offset(self.slider_background.value())

    # ------------------------------------------------------------------
    def open_folder(self):
        directory = QFileDialog.getExistingDirectory(self, "選擇標註資料夾", os.getcwd())
        if not directory:
            return
        folder = Path(directory)
        videos = self._list_video_files(folder)
        if not videos:
            QMessageBox.information(self, "找不到影片", "該資料夾內沒有支援的影片格式。")
            return
        self.dataset_root = folder
        self.video_paths = videos
        self.current_video_index = -1
        self.video_annotation_counts = {}
        for path in self.video_paths:
            self._refresh_video_count(path)
        self.populate_video_list()
        self.load_video_at(0)
        self.status_label.setText(f"資料夾已載入：{folder}")

    def _list_video_files(self, folder: Path) -> List[Path]:
        allow = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".wmv"}
        return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in allow])

    def _count_project_frames(self, project: SegmentationProject) -> int:
        return sum(1 for ann_map in project.annotations_by_frame.values() if ann_map)

    def _annotation_count_from_json(self, video_path: Path) -> int:
        json_path = video_path.with_suffix(".json")
        if not json_path.exists():
            return 0
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return 0
        images = data.get("images", [])
        return sum(1 for img in images if isinstance(img, dict))

    def _refresh_video_count(self, video_path: Path, project: Optional[SegmentationProject] = None) -> None:
        if project is not None:
            count = self._count_project_frames(project)
        else:
            count = self._annotation_count_from_json(video_path)
        self.video_annotation_counts[video_path] = int(count)

    def populate_video_list(self):
        self.list_videos.blockSignals(True)
        self.list_videos.clear()
        for idx, path in enumerate(self.video_paths):
            count = self.video_annotation_counts.get(path, 0)
            label = f"{path.name}（{count} 幀）"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, idx)
            self.list_videos.addItem(item)
        target_index = self.current_video_index if 0 <= self.current_video_index < len(self.video_paths) else (0 if self.video_paths else -1)
        if target_index >= 0:
            self.list_videos.setCurrentRow(target_index)
        self.list_videos.blockSignals(False)

    def load_video_at(self, index: int):
        if index < 0 or index >= len(self.video_paths):
            return
        if index == self.current_video_index and self.project is not None:
            return
        self.current_video_index = index
        self.list_videos.blockSignals(True)
        self.list_videos.setCurrentRow(index)
        self.list_videos.blockSignals(False)
        self.load_video_path(self.video_paths[index])

    def load_video_path(self, path: Path):
        categories = _ensure_labels_file(str(path))
        if self.project:
            self.project.close()
        if self.canvas:
            self.canvas.setParent(None)
            self.canvas.deleteLater()
            self.canvas = None
        if self.canvas_placeholder is None:
            self.canvas_placeholder = QLabel("請先開啟影片")
            self.canvas_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.canvas_layout.insertWidget(0, self.canvas_placeholder, 1)
        self.project = SegmentationProject(str(path), categories)
        self.auto_generator = None
        self.auto_weights_path = None
        self._refresh_video_count(path, project=self.project)
        self.populate_video_list()
        if self.dataset_root is None:
            self.dataset_root = path.parent
        self.populate_categories(categories)
        self.slider.setMaximum(max(0, self.project.total_frames - 1))
        self.slider.setValue(0)
        total_frames = max(1, self.project.total_frames)
        self.frame_input.blockSignals(True)
        self.frame_input.setRange(1, total_frames)
        self.frame_input.setValue(1)
        self.frame_input.setEnabled(True)
        self.frame_input.blockSignals(False)
        self.frame_label.setText(f"Frame: 1/{self.project.total_frames}")
        self.ensure_canvas()
        if self.canvas:
            self.canvas.load_frame(0)
        self.update_lists()
        self.set_active_category(0)
        self.dirty = False
        self.status_label.setText(f"影片已載入：{path.name}")

    def on_select_video_row(self, row: int):
        if row < 0 or row >= len(self.video_paths):
            return
        if row == self.current_video_index:
            return
        self.load_video_at(row)

    def populate_categories(self, categories: List[str]):
        self.list_categories.clear()
        for name in categories:
            self.list_categories.addItem(name)
        if self.list_categories.count():
            self.list_categories.setCurrentRow(0)

    def set_active_category(self, row: int):
        if row < 0 or row >= self.list_categories.count():
            return
        self.list_categories.setCurrentRow(row)

    # ------------------------------------------------------------------
    def load_frame(self, index: int):
        if self.project is None or self.canvas is None:
            return
        index = max(0, min(index, self.project.total_frames - 1))
        self.canvas.load_frame(index)
        self.slider.blockSignals(True)
        self.slider.setValue(index)
        self.slider.blockSignals(False)
        self.frame_input.blockSignals(True)
        self.frame_input.setValue(index + 1)
        self.frame_input.blockSignals(False)
        self.frame_label.setText(f"Frame: {index + 1}/{self.project.total_frames}")
        self.update_lists()

    def prev_frame(self):
        if self.canvas is None:
            return
        self.load_frame(self.canvas.frame_index - 1)

    def next_frame(self):
        if self.canvas is None:
            return
        self.load_frame(self.canvas.frame_index + 1)

    def on_slider(self, value: int):
        self.load_frame(value)

    # ------------------------------------------------------------------
    def on_frame_input_changed(self, value: int):
        if self.project is None or self.canvas is None:
            return
        total = max(1, self.project.total_frames)
        target = max(1, min(value, total)) - 1
        if target == self.canvas.frame_index:
            return
        self.load_frame(target)

    # ------------------------------------------------------------------
    def current_category_id(self) -> int:
        if not self.project or self.list_categories.currentRow() < 0:
            return 1
        name = self.list_categories.currentItem().text()
        return self.project.category_to_id.get(name, 1)

    def clear_masks(self):
        if self.project is None or self.canvas is None:
            return
        self.canvas.clear_frame_annotations()

    def on_canvas_brush_size_changed(self, size: int):
        self.brush_label.setText(f"筆刷: {size}")
        self.brush_slider.blockSignals(True)
        self.brush_slider.setValue(size)
        self.brush_slider.blockSignals(False)
        self.brush_spin.blockSignals(True)
        self.brush_spin.setValue(size)
        self.brush_spin.blockSignals(False)

    def on_brush_slider_changed(self, value: int):
        self.brush_spin.blockSignals(True)
        self.brush_spin.setValue(value)
        self.brush_spin.blockSignals(False)
        if self.canvas:
            self.canvas.set_brush_radius(value)
        else:
            self.brush_label.setText(f"筆刷: {value}")

    def on_brush_spin_changed(self, value: int):
        self.brush_slider.blockSignals(True)
        self.brush_slider.setValue(value)
        self.brush_slider.blockSignals(False)
        if self.canvas:
            self.canvas.set_brush_radius(value)
        else:
            self.brush_label.setText(f"筆刷: {value}")

    def undo(self):
        if self.canvas:
            self.canvas.undo()

    def _opacity_slider_to_alpha(self, value: int) -> int:
        clamped = max(0, min(100, value))
        normalized = clamped / 100.0
        adjusted = normalized ** 1.3
        return int(round(adjusted * 255))

    def apply_overlay_settings(self):
        alpha = self._opacity_slider_to_alpha(self.slider_opacity.value())
        self.opacity_label.setText(f"{self.slider_opacity.value()}%")
        idx = self.combo_overlay_color.currentIndex()
        if self.canvas:
            self.canvas.set_overlay_alpha(alpha)
            if 0 <= idx < len(self.overlay_presets):
                _, base, highlight = self.overlay_presets[idx]
                self.canvas.set_overlay_colors(base, highlight)

    def on_overlay_opacity_changed(self, value: int):
        if value < 0:
            return
        self.apply_overlay_settings()

    def on_overlay_color_changed(self, index: int):
        if index < 0:
            return
        self.apply_overlay_settings()

    def on_background_brightness_changed(self, value: int):
        if getattr(self, "background_label", None) is not None:
            self.background_label.setText(f"{value:+d}")
        if self.canvas:
            self.canvas.set_background_offset(value)

    def reset_background_brightness(self):
        if getattr(self, "slider_background", None) is None:
            return
        if self.slider_background.value() == self.default_background_offset:
            return
        self.slider_background.setValue(self.default_background_offset)

    # ------------------------------------------------------------------
    def mark_dirty(self) -> None:
        self.dirty = True
        self.auto_save()

    def auto_save(self) -> None:
        if self.project is None:
            return
        target_root = self.dataset_root or self.project.video_path.parent
        try:
            path = self.project.export_dataset(str(target_root))
        except Exception as exc:  # noqa: BLE001
            self.status_label.setText(f"自動儲存失敗：{exc}")
            QMessageBox.critical(self, "自動儲存失敗", str(exc))
            return
        if self.current_video_index >= 0 and self.current_video_index < len(self.video_paths):
            video_path = self.video_paths[self.current_video_index]
            self._refresh_video_count(video_path, project=self.project)
            self.populate_video_list()
        self.dirty = False
        self.status_label.setText(f"已自動儲存：{Path(path).name}")

    # ------------------------------------------------------------------
    def update_lists(self):
        if self.project is None or self.canvas is None:
            return

        # Annotated frame list
        self.list_frames.clear()
        for fi in sorted(self.project.annotations_by_frame.keys()):
            ann_map = self.project.annotations_by_frame[fi]
            if not ann_map:
                continue
            text = f"第 {fi + 1} 幀（{len(ann_map)}）"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, fi)
            self.list_frames.addItem(item)
            if fi == self.canvas.frame_index:
                self.list_frames.setCurrentItem(item)

    def on_select_category(self, item: QListWidgetItem):
        # UI only; painting uses current row
        pass

    def on_select_frame_item(self, item: QListWidgetItem):
        fi = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(fi, int):
            self.load_frame(fi)

    # ------------------------------------------------------------------
    def ensure_auto_generator(self) -> Optional[AutoMaskGenerator]:
        if self.auto_generator is not None:
            return self.auto_generator

        search_paths = []
        root_dir = Path(__file__).resolve().parents[1]
        search_paths.append(root_dir / "best.pt")
        search_paths.append(Path.cwd() / "best.pt")
        if self.dataset_root is not None:
            search_paths.append(self.dataset_root / "best.pt")
        if self.project is not None:
            search_paths.append(self.project.video_path.parent / "best.pt")

        weights_path: Optional[Path] = None
        for candidate in search_paths:
            if candidate.exists():
                weights_path = candidate
                break

        if weights_path is None:
            QMessageBox.warning(self, "模型不存在", "找不到預設權重檔案 best.pt")
            return None

        weights_str = str(weights_path)
        try:
            self.auto_generator = AutoMaskGenerator(weights_str)
            self.auto_weights_path = weights_str
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "載入失敗", str(exc))
            self.auto_generator = None
            return None
        return self.auto_generator

    def auto_mask(self):
        if self.project is None or self.canvas is None:
            return
        generator = self.ensure_auto_generator()
        if generator is None:
            return
        frame = self.project.load_frame(self.canvas.frame_index)
        if frame is None:
            QMessageBox.warning(self, "失敗", "無法載入當前影格")
            return
        results = generator(frame)
        if not results:
            QMessageBox.information(self, "提示", "沒有偵測結果")
            return
        best = results[0]
        self.apply_auto_result(best)

    def apply_auto_result(self, result: AutoMaskResult):
        if self.project is None or self.canvas is None:
            return
        frame_index = self.canvas.frame_index
        target_ann = self.canvas.active_annotation
        category_id = self.current_category_id()
        if result.label in self.project.category_to_id:
            category_id = self.project.category_to_id[result.label]
        if target_ann is None:
            target_ann = self.project.new_annotation(frame_index, category_id, result.mask)
            track_id = target_ann.track_id
        else:
            target_ann.update_mask(result.mask)
            target_ann.category_id = category_id
            track_id = target_ann.track_id
        self.canvas.refresh_annotations()
        self.canvas.set_active_track(track_id)
        self.update_lists()
        self.mark_dirty()
        self.status_label.setText(f"自動遮罩完成，score={result.score:.3f}")

    # ------------------------------------------------------------------
    def on_mask_changed(self, track_id: int):
        self.update_lists()
        self.mark_dirty()

    def on_canvas_annotation_created(self, track_id: int):
        if self.canvas:
            self.canvas.set_active_track(track_id)
        self.update_lists()

    def on_canvas_annotations_cleared(self):
        if self.canvas:
            self.canvas.set_active_track(None)
        self.update_lists()
        self.mark_dirty()

    def show_shortcuts(self):
        tips = (
            "滑鼠左鍵：筆刷填色\n"
            "滑鼠右鍵：橡皮擦清除\n"
            "滑鼠滾輪：調整筆刷大小\n"
            "空白鍵：暫時顯示遮罩預覽\n"
            "Ctrl+Z：復原上一筆\n"
            "上下鍵或按鈕：切換影格"
        )
        QMessageBox.information(self, "快捷鍵", tips)

    # ------------------------------------------------------------------
    def closeEvent(self, event):
        if self.dirty:
            ans = QMessageBox.question(self, "尚未儲存", "變更尚未自動儲存，確定要離開？")
            if ans != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        if self.project:
            self.project.close()
        super().closeEvent(event)


def launch():
    import sys

    app = QApplication(sys.argv)
    win = SegmentAnnotatorWindow()
    win.show()
    sys.exit(app.exec())


__all__ = ["SegmentAnnotatorWindow", "launch"]
