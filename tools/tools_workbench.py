"""Unified desktop UI hosting multiple tracking related tools.

This module provides a single entry point exposing existing utility widgets
such as the schedule results viewer, and stubs for upcoming diagnostics
features. Future tools can be registered here by adding new tabs to the
``ToolsWorkbench`` window.
"""
from __future__ import annotations

from typing import Optional, Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:  # allow running as package or standalone script
    from .schedule_results_viewer import ScheduleResultsViewer
    from .confidence_diagnostics import ConfidenceDiagnosticsWidget
except ImportError:  # pragma: no cover - fallback for "python tools/tools_workbench.py"
    import sys
    from pathlib import Path

    _CURRENT_DIR = Path(__file__).resolve().parent
    if str(_CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(_CURRENT_DIR))
    _PROJECT_ROOT = _CURRENT_DIR.parent
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from schedule_results_viewer import ScheduleResultsViewer  # type: ignore
    from confidence_diagnostics import ConfidenceDiagnosticsWidget  # type: ignore


class _PlaceholderTool(QWidget):
    """Simple placeholder widget that can be swapped with a real tool later."""

    def __init__(
        self,
        title: str,
        description_lines: Sequence[str],
        actions: Optional[Sequence[tuple[str, str, str]]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(16)

        heading = QLabel(title, self)
        heading.setStyleSheet("font-size: 22px; font-weight: 600;")
        heading.setWordWrap(True)
        layout.addWidget(heading)

        for line in description_lines:
            label = QLabel(line, self)
            label.setWordWrap(True)
            label.setStyleSheet("color: #586069; font-size: 14px;")
            layout.addWidget(label)

        if actions:
            for action_text, action_desc, css in actions:
                button = QPushButton(action_text, self)
                if css:
                    button.setStyleSheet(css)
                button.setEnabled(False)
                button.setToolTip(action_desc)
                layout.addWidget(button)

        filler = QTextEdit(self)
        filler.setReadOnly(True)
        filler.setPlaceholderText("此工具開發中。完成後會顯示互動式圖表與統計資料。")
        filler.setStyleSheet(
            "QTextEdit { min-height: 260px; border: 1px dashed #d0d7de; "
            "color: #8c959f; background-color: #f6f8fa; }"
        )
        layout.addWidget(filler)
        layout.addStretch(1)


class ToolsWorkbench(QMainWindow):
    """Main window with tabbed navigation for the tracking utilities."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Tracking Tools Workbench")
        self.resize(1280, 820)

        self._tabs = QTabWidget(self)
        self._tabs.setTabPosition(QTabWidget.North)
        self._tabs.setMovable(False)
        self._tabs.setDocumentMode(True)
        self._tabs.setTabBarAutoHide(False)
        self.setCentralWidget(self._tabs)

        self._init_schedule_results_tab()
        self._init_confidence_diagnostics_tab()
        self._init_future_tools_placeholder()

    def _init_schedule_results_tab(self) -> None:
        widget = ScheduleResultsViewer()
        widget.setParent(self)
        self._tabs.addTab(widget, "排程結果瀏覽")

    def _init_confidence_diagnostics_tab(self) -> None:
        widget = ConfidenceDiagnosticsWidget(self)
        self._tabs.addTab(widget, "信心診斷")

    def _init_future_tools_placeholder(self) -> None:
        placeholder = _PlaceholderTool(
            title="工具擴充入口",
            description_lines=[
                "後續實用工具可在此新增，例如排程統計儀表板、錯誤診斷、批次重測等。",
                "若需快速測試，可以複製此分頁並取代為實際 UI。",
            ],
        )
        placeholder.setObjectName("futureToolsPlaceholder")
        self._tabs.addTab(placeholder, "更多工具")


def main() -> None:  # pragma: no cover - interactive UI
    import sys

    app = QApplication(sys.argv)
    app.setOrganizationName("Traking")
    app.setApplicationName("ToolsWorkbench")
    window = ToolsWorkbench()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":  # pragma: no cover - script entry
    main()
