"""Simple UI tool to inspect schedule experiment results.

This utility lets users enter a results schedule folder (e.g.
``2025-10-31_17-30-25_schedule_10exp``) and visualise all experiment
metrics in a sortable table. Data is gathered from ``metadata.json`` and
``test/metrics/summary.json`` inside each experiment directory.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QMessageBox,
    QFileDialog,
    QLabel,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"

def _resolve_schedule_path(text: str) -> Path:
    """Resolve user input into an absolute schedule directory path."""
    candidate = Path(text.strip())
    if not candidate:
        return candidate
    if not candidate.is_absolute():
        joined = RESULTS_ROOT / candidate
        if joined.exists():
            candidate = joined
        else:
            candidate = candidate.resolve()
    return candidate


def _format_number(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return f"{value:.4f}" if isinstance(value, float) else str(value)
    return str(value)


def _flatten_metrics(summary: Any) -> Tuple[Dict[str, Any], List[str]]:
    metrics: Dict[str, Any] = {}
    ordered_columns: List[str] = []

    if not isinstance(summary, dict):
        return metrics, ordered_columns

    dict_values = list(summary.values())
    if dict_values and all(isinstance(v, dict) for v in dict_values):
        use_prefix = len(summary) > 1
        for tracker_name, tracker_metrics in summary.items():
            if not isinstance(tracker_metrics, dict):
                continue
            for key, value in tracker_metrics.items():
                column_name = f"{tracker_name}.{key}" if use_prefix else key
                metrics[column_name] = value
                ordered_columns.append(column_name)
    else:
        for key, value in summary.items():
            metrics[key] = value
            ordered_columns.append(key)

    return metrics, ordered_columns


def collect_schedule_rows(schedule_dir: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    column_order: List[str] = []
    if not schedule_dir.is_dir():
        raise FileNotFoundError(f"Schedule directory not found: {schedule_dir}")

    for exp_dir in sorted(schedule_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        summary_path = exp_dir / "test" / "metrics" / "summary.json"
        if not summary_path.is_file():
            continue

        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        metric_values, metric_columns = _flatten_metrics(summary)

        row: Dict[str, Any] = {"experiment": exp_dir.name}
        row.update(metric_values)

        rows.append(row)

        for column in metric_columns:
            if column not in column_order:
                column_order.append(column)

    return rows, ["experiment"] + column_order


def rows_to_csv(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    if not rows:
        return ""
    lines = []
    lines.append(",".join(columns))
    for row in rows:
        values = []
        for key in columns:
            value = row.get(key)
            if isinstance(value, (dict, list)):
                values.append(json.dumps(value, ensure_ascii=False))
            elif isinstance(value, float):
                values.append(f"{value:.6f}")
            elif value is None:
                values.append("")
            else:
                values.append(str(value))
        lines.append(",".join(values))
    return "\n".join(lines)


class ScheduleResultsViewer(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Schedule Results Viewer")
        self.resize(1200, 700)

        self._rows: List[Dict[str, Any]] = []
        self._columns: List[str] = ["experiment"]

        layout = QVBoxLayout(self)

        # Input row
        input_layout = QHBoxLayout()
        self.path_edit = QLineEdit(self)
        self.path_edit.setPlaceholderText(
            "Enter schedule folder (e.g. 2025-10-31_17-30-25_schedule_10exp)"
        )
        browse_btn = QPushButton("Browse", self)
        load_btn = QPushButton("Load", self)
        input_layout.addWidget(self.path_edit)
        input_layout.addWidget(browse_btn)
        input_layout.addWidget(load_btn)

        layout.addLayout(input_layout)

        # Table
        self.table = QTableWidget(self)
        self._configure_table_columns(self._columns)
        self.table.setSortingEnabled(True)
        layout.addWidget(self.table)

        # Status + actions
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Ready.", self)
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        copy_btn = QPushButton("Copy CSV", self)
        save_btn = QPushButton("Save CSV…", self)
        status_layout.addWidget(copy_btn)
        status_layout.addWidget(save_btn)
        layout.addLayout(status_layout)

        browse_btn.clicked.connect(self._on_browse)
        load_btn.clicked.connect(self._on_load)
        copy_btn.clicked.connect(self._copy_csv)
        save_btn.clicked.connect(self._save_csv)

        # If there is a default results root, hint it
        if RESULTS_ROOT.exists():
            self.status_label.setText(f"Results root: {RESULTS_ROOT}")

    # ------------------------------------------------------------------ UI
    def _on_browse(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select schedule directory",
            str(RESULTS_ROOT) if RESULTS_ROOT.exists() else str(PROJECT_ROOT),
        )
        if directory:
            self.path_edit.setText(directory)
            self._load_schedule(Path(directory))

    def _on_load(self) -> None:
        text = self.path_edit.text().strip()
        if not text:
            QMessageBox.warning(self, "Schedule Viewer", "Please enter a folder name.")
            return
        schedule_path = _resolve_schedule_path(text)
        self._load_schedule(schedule_path)

    def _load_schedule(self, path: Path) -> None:
        try:
            rows, columns = collect_schedule_rows(path)
        except FileNotFoundError as exc:
            QMessageBox.critical(self, "Schedule Viewer", str(exc))
            return
        except Exception as exc:  # pragma: no cover - unexpected errors
            QMessageBox.critical(self, "Schedule Viewer", f"Failed to load schedule: {exc}")
            return

        sort_column = self._select_sort_column(columns)
        if sort_column:
            rows = self._sort_rows(rows, sort_column)

        self._configure_table_columns(columns)
        self._rows = rows
        self._populate_table()
        if not rows:
            self.status_label.setText(f"No experiments found in {path}")
        else:
            message = f"Loaded {len(rows)} experiments from {path}."
            if sort_column:
                best = rows[0]
                best_value = best.get(sort_column)
                best_value_str = self._format_status_value(best_value)
                best_name = best.get("experiment", "N/A")
                if best_value_str:
                    message += f" Best {sort_column}: {best_value_str} ({best_name})."
            self.status_label.setText(message)

    def _configure_table_columns(self, columns: Sequence[str]) -> None:
        self._columns = list(columns)
        self.table.setColumnCount(len(self._columns))
        self.table.setHorizontalHeaderLabels(self._columns)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def _select_sort_column(self, columns: Sequence[str]) -> Optional[str]:
        if not columns:
            return None
        preferred_keywords = [
            "success_auc",
            "success_rate_75",
            "success_rate_50",
            "success_rate",
            "auc",
            "map_50",
            "map50",
            "iou_mean",
        ]
        lowered = [col.lower() for col in columns]
        for keyword in preferred_keywords:
            for idx, lower in enumerate(lowered):
                if keyword == lower or lower.endswith(f".{keyword}") or keyword in lower:
                    column = columns[idx]
                    if column != "experiment":
                        return column
        for column in columns:
            if column != "experiment":
                return column
        return None

    def _sort_rows(self, rows: List[Dict[str, Any]], column: str) -> List[Dict[str, Any]]:
        def key(row: Dict[str, Any]) -> float:
            value = row.get(column)
            if isinstance(value, (int, float)):
                return float(value)
            return float("-inf")

        return sorted(rows, key=key, reverse=True)

    @staticmethod
    def _format_status_value(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        if isinstance(value, int):
            return str(value)
        if value:
            return str(value)
        return ""

    def _populate_table(self) -> None:
        self.table.setRowCount(len(self._rows))
        for row_idx, row in enumerate(self._rows):
            for col_idx, key in enumerate(self._columns):
                value = row.get(key)
                if isinstance(value, float):
                    display = f"{value:.4f}"
                elif value is None:
                    display = ""
                else:
                    display = str(value)
                item = QTableWidgetItem(display)
                if isinstance(value, (int, float)):
                    item.setData(Qt.EditRole, value)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(row_idx, col_idx, item)
        self.table.resizeColumnsToContents()

    def _copy_csv(self) -> None:
        if not self._rows:
            QMessageBox.information(self, "Schedule Viewer", "No data to copy.")
            return
        csv_text = rows_to_csv(self._rows, self._columns)
        QApplication.clipboard().setText(csv_text)
        self.status_label.setText("CSV copied to clipboard.")

    def _save_csv(self) -> None:
        if not self._rows:
            QMessageBox.information(self, "Schedule Viewer", "No data to save.")
            return
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save CSV",
            str(PROJECT_ROOT / "schedule_results.csv"),
            "CSV Files (*.csv);;All Files (*)",
        )
        if not filename:
            return
        csv_text = rows_to_csv(self._rows, self._columns)
        try:
            with open(filename, "w", encoding="utf-8", newline="") as f:
                f.write(csv_text)
        except Exception as exc:
            QMessageBox.critical(self, "Schedule Viewer", f"Failed to save CSV: {exc}")
            return
        self.status_label.setText(f"Saved CSV to {filename}")

    # ------------------------------------------------------------------ Events
    def closeEvent(self, event: QCloseEvent) -> None:  # pragma: no cover - interactive
        self._rows.clear()
        super().closeEvent(event)


def main() -> None:  # pragma: no cover - UI entry
    import sys

    app = QApplication(sys.argv)
    viewer = ScheduleResultsViewer()
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":  # pragma: no cover - CLI usage
    main()
