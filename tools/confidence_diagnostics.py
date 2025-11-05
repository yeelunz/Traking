from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QDoubleSpinBox,
    QHeaderView,
    QAbstractItemView,
)

try:  # allow running as part of package or standalone script
    from tracking.utils.confidence import ConfidenceConfig
    from tracking.utils.confidence_scan import (
        FrameConfidenceSnapshot,
        SequenceConfidenceSummary,
        scan_schedule_confidence,
        summaries_to_csv,
    )
    from .schedule_results_viewer import RESULTS_ROOT, _resolve_schedule_path
except ImportError:  # pragma: no cover - fallback for direct execution
    import sys

    _CURRENT_DIR = Path(__file__).resolve().parent
    if str(_CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(_CURRENT_DIR))
    _PROJECT_ROOT = _CURRENT_DIR.parent
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

    from tracking.utils.confidence import ConfidenceConfig
    from tracking.utils.confidence_scan import (
        FrameConfidenceSnapshot,
        SequenceConfidenceSummary,
        scan_schedule_confidence,
        summaries_to_csv,
    )
    from schedule_results_viewer import RESULTS_ROOT, _resolve_schedule_path  # type: ignore


def _format_float(value: Optional[float], precision: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return f"{value:.{precision}f}"


def _format_percent(value: float) -> str:
    if math.isnan(value):
        return ""
    return f"{value * 100:.1f}%"


class ConfidenceDiagnosticsWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("ConfidenceDiagnostics")

        self._summaries: List[SequenceConfidenceSummary] = []
        self._low_threshold: float = 0.6
        self._config = ConfidenceConfig()
        self._current_schedule: Optional[Path] = None

        self._build_ui()

    # ------------------------------------------------------------------ UI setup
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        header = QLabel(
            "分析 MixFormer 及其他追蹤器的信心分數，找出需要重新偵測或調整策略的片段。",
            self,
        )
        header.setWordWrap(True)
        header.setStyleSheet("font-size: 14px; color: #4a5568;")
        layout.addWidget(header)

        # Controls row
        controls = QHBoxLayout()
        self.path_edit = QLineEdit(self)
        self.path_edit.setPlaceholderText(
            "輸入排程資料夾，例如 2025-10-31_17-30-25_schedule_10exp"
        )
        controls.addWidget(self.path_edit, 1)

        browse_btn = QPushButton("瀏覽…", self)
        browse_btn.clicked.connect(self._on_browse)
        controls.addWidget(browse_btn)

        load_btn = QPushButton("載入分析", self)
        load_btn.clicked.connect(self._on_load)
        controls.addWidget(load_btn)

        threshold_label = QLabel("低信心門檻", self)
        controls.addWidget(threshold_label)

        self.threshold_spin = QDoubleSpinBox(self)
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setDecimals(7)
        self.threshold_spin.setSingleStep(0.000001)
        self.threshold_spin.setValue(self._low_threshold)
        self.threshold_spin.valueChanged.connect(self._on_threshold_changed)
        controls.addWidget(self.threshold_spin)

        copy_btn = QPushButton("複製重點指標", self)
        copy_btn.clicked.connect(self._copy_metrics)
        controls.addWidget(copy_btn)

        rescan_btn = QPushButton("重新分析", self)
        rescan_btn.clicked.connect(self._rescan_current)
        controls.addWidget(rescan_btn)

        layout.addLayout(controls)

        splitter = QSplitter(Qt.Horizontal, self)
        layout.addWidget(splitter, 1)

        # Summary table
        summary_container = QWidget(splitter)
        summary_layout = QVBoxLayout(summary_container)
        summary_layout.setContentsMargins(0, 0, 0, 0)
        summary_layout.setSpacing(6)

        summary_label = QLabel("實驗摘要", summary_container)
        summary_label.setStyleSheet("font-weight: 600;")
        summary_layout.addWidget(summary_label)

        self.summary_table = QTableWidget(summary_container)
        self.summary_table.setColumnCount(16)
        self.summary_table.setHorizontalHeaderLabels(
            [
                "實驗",
                "追蹤器",
                "總影格",
                "平均信心",
                "P10",
                "P05",
                "最低",
                "低於門檻",
                "低於比率",
                "最長連續",
                "平均Score組件",
                "平均Token組件",
                "平均分布組件",
                "平均注意力組件",
                "平均IoU組件",
                "平均漂移組件",
            ]
        )
        self.summary_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.summary_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.summary_table.verticalHeader().setVisible(False)
        self.summary_table.itemSelectionChanged.connect(self._on_summary_selection)
        summary_layout.addWidget(self.summary_table, 1)

        # Detail panel
        detail_container = QWidget(splitter)
        detail_layout = QVBoxLayout(detail_container)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.setSpacing(8)

        detail_title = QLabel("詳細指標", detail_container)
        detail_title.setStyleSheet("font-weight: 600;")
        detail_layout.addWidget(detail_title)

        self.stats_table = QTableWidget(detail_container)
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["指標", "數值"])
        self.stats_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.stats_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.stats_table.verticalHeader().setVisible(False)
        self.stats_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        detail_layout.addWidget(self.stats_table)

        worst_label = QLabel("最低信心影格 (Top 8)", detail_container)
        worst_label.setStyleSheet("font-weight: 600;")
        detail_layout.addWidget(worst_label)

        self.worst_table = QTableWidget(detail_container)
        self.worst_table.setColumnCount(7)
        self.worst_table.setHorizontalHeaderLabels(
            [
                "#",
                "Frame",
                "Confidence",
                "Raw Score",
                "ScoreComp",
                "IoUComp",
                "Drift(px)",
            ]
        )
        self.worst_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.worst_table.verticalHeader().setVisible(False)
        self.worst_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        detail_layout.addWidget(self.worst_table)

        self.notes_box = QTextEdit(detail_container)
        self.notes_box.setReadOnly(True)
        self.notes_box.setPlaceholderText("選取任一列以顯示診斷資訊。")
        self.notes_box.setStyleSheet(
            "QTextEdit { background-color: #f7fafc; border: 1px solid #e2e8f0; font-family: Consolas, 'Courier New'; }"
        )
        detail_layout.addWidget(self.notes_box, 1)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 4)

        # Status
        self.status_label = QLabel("Ready", self)
        self.status_label.setStyleSheet("color: #4a5568;")
        layout.addWidget(self.status_label)

        if RESULTS_ROOT.exists():
            self.status_label.setText(f"結果根目錄：{RESULTS_ROOT}")

    # ------------------------------------------------------------------ helpers
    def _on_browse(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "選擇排程資料夾",
            str(RESULTS_ROOT) if RESULTS_ROOT.exists() else str(Path.cwd()),
        )
        if directory:
            self.path_edit.setText(directory)
            self._load_schedule(Path(directory))

    def _on_load(self) -> None:
        text = self.path_edit.text().strip()
        if not text:
            QMessageBox.warning(self, "信心診斷", "請輸入排程資料夾")
            return
        schedule_path = _resolve_schedule_path(text)
        self._load_schedule(schedule_path)

    def _on_threshold_changed(self, value: float) -> None:
        self._low_threshold = float(value)

    def _rescan_current(self) -> None:
        text = self.path_edit.text().strip()
        if not text:
            return
        schedule_path = _resolve_schedule_path(text)
        if schedule_path.is_dir():
            self._load_schedule(schedule_path)

    # ------------------------------------------------------------------ loading
    def _load_schedule(self, schedule_path: Path) -> None:
        if not schedule_path.exists():
            QMessageBox.critical(self, "信心診斷", f"找不到資料夾：{schedule_path}")
            return
        try:
            summaries = scan_schedule_confidence(
                schedule_path,
                config=self._config,
                low_threshold=self._low_threshold,
                top_k_frames=8,
            )
        except Exception as exc:  # pragma: no cover - runtime error handling
            QMessageBox.critical(self, "信心診斷", f"分析失敗：{exc}")
            return

        self._summaries = summaries
        self._current_schedule = schedule_path
        self._populate_summary_table()

        if not summaries:
            self.status_label.setText(f"{schedule_path} 沒有任何預測 JSON")
            self.stats_table.setRowCount(0)
            self.worst_table.setRowCount(0)
            self.notes_box.clear()
            return

        self.status_label.setText(
            f"載入 {len(summaries)} 筆追蹤資料，自 '{schedule_path.name}'"
        )
        self.summary_table.selectRow(0)

    def _copy_metrics(self) -> None:
        if not self._summaries:
            QMessageBox.information(self, "信心診斷", "尚未載入任何追蹤資料，請先執行分析。")
            return

        csv_text = summaries_to_csv(self._summaries, float_precision=4)
        if not csv_text:
            QMessageBox.warning(self, "信心診斷", "找不到可複製的指標資料。")
            return

        QApplication.clipboard().setText(csv_text)
        schedule_name = self._current_schedule.name if self._current_schedule else "目前資料"
        self.status_label.setText(
            f"已複製 {len(self._summaries)} 筆指標至剪貼簿（{schedule_name}）。"
        )

    def _populate_summary_table(self) -> None:
        self.summary_table.setRowCount(len(self._summaries))
        for row, summary in enumerate(self._summaries):
            values = [
                summary.experiment,
                summary.tracker,
                str(summary.total_frames),
                _format_float(summary.confidence_mean),
                _format_float(summary.confidence_p10),
                _format_float(summary.confidence_p05),
                _format_float(summary.confidence_min),
                str(summary.below_threshold),
                _format_percent(summary.below_threshold_ratio),
                str(summary.longest_low_streak),
                _format_float(summary.score_component_mean),
                _format_float(summary.token_component_mean),
                _format_float(summary.distribution_component_mean),
                _format_float(summary.attention_component_mean),
                _format_float(summary.short_iou_component_mean),
                _format_float(summary.drift_component_mean),
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col in (2, 7, 9):
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                elif col >= 3:
                    item.setTextAlignment(Qt.AlignCenter)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.summary_table.setItem(row, col, item)

    def _on_summary_selection(self) -> None:
        indexes = self.summary_table.selectionModel().selectedRows()
        if not indexes:
            return
        row = indexes[0].row()
        if not (0 <= row < len(self._summaries)):
            return
        self._show_details(self._summaries[row])

    # ------------------------------------------------------------------ details
    def _show_details(self, summary: SequenceConfidenceSummary) -> None:
        stats: List[Tuple[str, str]] = [
            ("實驗", summary.experiment),
            ("追蹤器", summary.tracker),
            ("來源檔案", str(summary.source_path)),
            ("總影格", str(summary.total_frames)),
            ("平均信心", _format_float(summary.confidence_mean, 4)),
            ("信心標準差", _format_float(summary.confidence_std, 4)),
            ("信心 10th 百分位", _format_float(summary.confidence_p10, 4)),
            ("信心 5th 百分位", _format_float(summary.confidence_p05, 4)),
            ("最低信心", _format_float(summary.confidence_min, 4)),
            ("低於門檻影格", f"{summary.below_threshold} ({_format_percent(summary.below_threshold_ratio)})"),
            ("最長低信心連續影格", str(summary.longest_low_streak)),
            ("原始分數平均", _format_float(summary.raw_score_mean) if summary.raw_score_mean is not None else ""),
            (
                "原始分數 P10",
                _format_float(summary.raw_score_p10) if summary.raw_score_p10 is not None else "",
            ),
            ("平均 Score 組件", _format_float(summary.score_component_mean)),
            ("平均 Token 組件", _format_float(summary.token_component_mean)),
            ("平均邊界分布組件", _format_float(summary.distribution_component_mean)),
            ("平均注意力組件", _format_float(summary.attention_component_mean)),
            ("平均 IoU 組件", _format_float(summary.short_iou_component_mean)),
            ("平均漂移組件", _format_float(summary.drift_component_mean)),
            ("平均漂移距離 (px)", _format_float(summary.drift_pixels_mean)),
            ("漂移距離 P95 (px)", _format_float(summary.drift_pixels_p95)),
        ]

        self.stats_table.setRowCount(len(stats))
        for idx, (label, value) in enumerate(stats):
            item_label = QTableWidgetItem(label)
            item_label.setFlags(item_label.flags() & ~Qt.ItemIsEditable)
            item_value = QTableWidgetItem(value)
            item_value.setFlags(item_value.flags() & ~Qt.ItemIsEditable)
            self.stats_table.setItem(idx, 0, item_label)
            self.stats_table.setItem(idx, 1, item_value)

        worst_frames = summary.worst_frames
        self.worst_table.setRowCount(len(worst_frames))
        for idx, frame in enumerate(worst_frames):
            row_values = [
                str(idx + 1),
                str(frame.frame_index),
                _format_float(frame.confidence, 4),
                _format_float(frame.raw_score, 4) if frame.raw_score is not None else "",
                _format_float(frame.components.get("raw_score", frame.components.get("score")), 4),
                _format_float(frame.components.get("short_iou"), 4),
                _format_float(frame.drift_pixels, 2),
            ]
            for col, text in enumerate(row_values):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if col in (0, 1):
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                else:
                    item.setTextAlignment(Qt.AlignCenter)
                self.worst_table.setItem(idx, col, item)

        advice_lines = self._build_notes(summary)
        self.notes_box.setPlainText("\n".join(advice_lines))

    def _build_notes(self, summary: SequenceConfidenceSummary) -> List[str]:
        lines = [
            f"建議：檢查 {summary.tracker} 在 {summary.experiment} 中的低信心片段。",
            f"- 低於門檻影格：{summary.below_threshold} ({_format_percent(summary.below_threshold_ratio)})",
            f"- 最長低信心連續影格：{summary.longest_low_streak}",
            "",
        ]
        if summary.drift_pixels_p95 > 0:
            lines.append(f"漂移距離 P95 約為 {_format_float(summary.drift_pixels_p95)} px，預設允許值為追蹤框對角線的 {self._config.drift_normalizer} 倍。")
        if summary.raw_score_mean is not None and summary.raw_score_mean > 0.95:
            lines.append("原始分數普遍接近 1.0，建議依賴 IoU / 漂移訊號做輔助重新偵測。")
        if summary.short_iou_component_mean < 0.5:
            lines.append("短期 IoU 組件偏低，可能代表框位置不穩定。")
        if summary.drift_component_mean < 0.5:
            lines.append("漂移組件偏低，建議檢查初始定位或啟用重新偵測。")
        lines.append("")
        lines.append("可將表格資料複製到試算表，或將整個排程資料夾納入後續報告。")
        return lines


def main() -> None:  # pragma: no cover - UI entry
    import sys

    app = QApplication(sys.argv)
    widget = ConfidenceDiagnosticsWidget()
    widget.show()
    sys.exit(app.exec())


if __name__ == "__main__":  # pragma: no cover
    main()
