from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, e):
        e.ignore()  # 防止滾輪誤觸


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, e):
        e.ignore()


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, e):
        e.ignore()


class ModelWeightsComboBox(NoWheelComboBox):
    def __init__(self, fetch_weights: Optional[Callable[[], List[str]]], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._fetch_weights = fetch_weights or (lambda: [])
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.NoInsert)
        self.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        try:
            le = self.lineEdit()
            if le is not None:
                le.setPlaceholderText("選擇或輸入模型檔案")
        except Exception:
            pass
        self.refresh_items()

    def _normalize_entry(self, entry: str) -> str:
        return entry.replace("\\", "/")

    def refresh_items(self):
        try:
            obtained = list(self._fetch_weights())
        except Exception:
            obtained = []
        normalized: List[str] = []
        seen = set()
        for entry in obtained:
            if not entry:
                continue
            norm = self._normalize_entry(str(entry))
            if norm in seen:
                continue
            seen.add(norm)
            normalized.append(norm)
        prev_text = self.currentText()
        if self.isEditable():
            try:
                line_text = self.lineEdit().text()
                if line_text:
                    prev_text = line_text
            except Exception:
                pass
        self.blockSignals(True)
        self.clear()
        for entry in normalized:
            self.addItem(entry, entry)
        if prev_text:
            norm_prev = self._normalize_entry(prev_text)
            idx = self.findData(norm_prev)
            if idx < 0:
                idx = self.findText(norm_prev)
            if idx >= 0:
                self.setCurrentIndex(idx)
            else:
                self.setEditText(prev_text)
        else:
            self.setCurrentIndex(-1)
            if self.isEditable():
                self.setEditText("")
        self.blockSignals(False)

    def set_current_value(self, value: Any):
        text = "" if value is None else str(value)
        self.refresh_items()
        norm = self._normalize_entry(text)
        if norm:
            idx = self.findData(norm)
            if idx < 0:
                idx = self.findText(norm)
            if idx >= 0:
                self.setCurrentIndex(idx)
            else:
                self.setEditText(text)
        else:
            self.setCurrentIndex(-1)
            if self.isEditable():
                self.setEditText("")

    def current_value(self) -> str:
        return self.currentText().strip()

    def showPopup(self):
        self.refresh_items()
        super().showPopup()


class LowConfidenceReinitEditor(QWidget):
    valueChanged = Signal()

    SUPPORTED_DETECTOR_KEYS = {"weights"}

    def __init__(
        self,
        defaults: Dict[str, Any],
        weights_provider: Optional[Callable[[], List[str]]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._defaults = dict(defaults or {})
        self._extra_detector_keys: Dict[str, Any] = {}
        self._weights_provider = weights_provider

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.chk_enabled = QCheckBox("啟用低信心重新偵測")
        self.chk_enabled.stateChanged.connect(self._emit_changed)
        layout.addWidget(self.chk_enabled)

        form = QFormLayout()
        form.setContentsMargins(6, 0, 0, 0)
        layout.addLayout(form)

        self.spn_threshold = NoWheelDoubleSpinBox()
        self.spn_threshold.setRange(0.0, 1.0)
        self.spn_threshold.setSingleStep(0.0001)
        self.spn_threshold.setDecimals(9)
        self.spn_threshold.valueChanged.connect(lambda *_: self._emit_changed())
        self.spn_threshold.setToolTip("Tracker 輸出的 confidence 低於此值時，才會啟動重新偵測流程。")
        form.addRow("Tracker 信心門檻", self.spn_threshold)

        self.spn_min_interval = NoWheelSpinBox()
        self.spn_min_interval.setRange(1, 100000)
        self.spn_min_interval.valueChanged.connect(lambda *_: self._emit_changed())
        self.spn_min_interval.setToolTip("兩次重新偵測之間最少需要間隔的 frame 數，避免頻繁呼叫偵測器。")
        form.addRow("最小間隔 (frame)", self.spn_min_interval)

        detector_row = QHBoxLayout()
        detector_row.setContentsMargins(0, 0, 0, 0)
        if self._weights_provider is not None:
            self.edit_detector_weights = ModelWeightsComboBox(self._weights_provider, parent=self)
            self.edit_detector_weights.refresh_items()
            self.edit_detector_weights.currentIndexChanged.connect(lambda *_: self._emit_changed())
            self.edit_detector_weights.editTextChanged.connect(lambda *_: self._emit_changed())
            try:
                le = self.edit_detector_weights.lineEdit()
                if le is not None:
                    le.setPlaceholderText("留空=沿用 init detector 的權重，例如 best.pt")
            except Exception:
                pass
        else:
            self.edit_detector_weights = QLineEdit()
            self.edit_detector_weights.editingFinished.connect(self._emit_changed)
            self.edit_detector_weights.setPlaceholderText("留空=沿用 init detector 的權重，例如 best.pt")
        detector_row.addWidget(self.edit_detector_weights)
        form.addRow("重偵測權重", detector_row)

        self.edit_detector_min_conf = QLineEdit()
        self.edit_detector_min_conf.setPlaceholderText("留空=沿用 YOLO 預設 (通常 0.25)")
        self.edit_detector_min_conf.editingFinished.connect(self._emit_changed)
        self.edit_detector_min_conf.setToolTip("重新偵測後的框需要達到的 YOLO 置信門檻；留空則使用 YOLO 模型的原始 conf。")
        form.addRow("偵測結果最低置信", self.edit_detector_min_conf)

        hint = QLabel("註：Tracker 信心門檻用來判斷何時觸發重新偵測；偵測結果最低置信則過濾 YOLO 回傳的候選框。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#6a737d; font-size:11px;")
        layout.addWidget(hint)

        layout.addStretch(1)

    def _emit_changed(self):
        self.valueChanged.emit()

    def _parse_float(self, text: str) -> Optional[float]:
        txt = text.strip()
        if not txt:
            return None
        try:
            return float(txt)
        except Exception:
            return None

    def get_value(self) -> Dict[str, Any]:
        cfg = {
            "enabled": bool(self.chk_enabled.isChecked()),
            "threshold": float(self.spn_threshold.value()),
            "min_interval": int(self.spn_min_interval.value()),
            "detector": {},
            "detector_min_conf": None,
        }

        detector_cfg: Dict[str, Any] = dict(self._extra_detector_keys)
        weights_widget = self.edit_detector_weights
        weights = ""
        if hasattr(weights_widget, 'current_value') and callable(getattr(weights_widget, 'current_value', None)):
            try:
                weights = str(weights_widget.current_value()).strip()
            except Exception:
                weights = str(getattr(weights_widget, 'currentText', lambda: "")()).strip()
        elif hasattr(weights_widget, 'currentText'):
            weights = str(weights_widget.currentText()).strip()
        else:
            weights = str(getattr(weights_widget, 'text', lambda: "")()).strip()
        if weights:
            detector_cfg["weights"] = weights
        else:
            detector_cfg.pop("weights", None)

        cfg["detector"] = detector_cfg if detector_cfg else {}

        min_conf = self._parse_float(self.edit_detector_min_conf.text())
        cfg["detector_min_conf"] = min_conf

        return cfg

    def set_value(self, value: Any):
        defaults = self._defaults
        if isinstance(value, bool):
            value = {"enabled": bool(value)}
        if not isinstance(value, dict):
            value = {}

        self.chk_enabled.setChecked(bool(value.get("enabled", defaults.get("enabled", False))))

        thr = value.get("threshold", defaults.get("threshold", 0.3))
        try:
            self.spn_threshold.setValue(float(thr))
        except Exception:
            self.spn_threshold.setValue(float(defaults.get("threshold", 0.3)))

        interval = value.get("min_interval", defaults.get("min_interval", 15))
        try:
            self.spn_min_interval.setValue(max(1, int(interval)))
        except Exception:
            self.spn_min_interval.setValue(int(defaults.get("min_interval", 15)))

        detector_cfg = value.get("detector") or {}
        if not isinstance(detector_cfg, dict):
            detector_cfg = {}

        weights = detector_cfg.get("weights") or ""
        if hasattr(self.edit_detector_weights, 'set_current_value') and callable(getattr(self.edit_detector_weights, 'set_current_value', None)):
            self.edit_detector_weights.set_current_value(weights)
        elif hasattr(self.edit_detector_weights, 'setText'):
            self.edit_detector_weights.setText(str(weights))

        self._extra_detector_keys = {k: v for k, v in detector_cfg.items() if k not in self.SUPPORTED_DETECTOR_KEYS}
        if weights:
            self._extra_detector_keys.pop("weights", None)

        min_conf = value.get("detector_min_conf", defaults.get("detector_min_conf"))
        if min_conf in (None, "", "none"):
            self.edit_detector_min_conf.clear()
        else:
            try:
                self.edit_detector_min_conf.setText(str(float(min_conf)))
            except Exception:
                self.edit_detector_min_conf.setText(str(min_conf))
