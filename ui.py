from __future__ import annotations
import copy
import json, os, re, time
from typing import Optional, Dict, Any, List, Tuple

from PySide6.QtCore import Qt, QRegularExpression, QTimer, QObject, Signal, QThread
from PySide6.QtGui import QRegularExpressionValidator
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QTextEdit, QLineEdit, QMessageBox, QComboBox, QSpinBox, QDoubleSpinBox,
    QFormLayout, QGroupBox, QListWidget, QListWidgetItem, QAbstractItemView, QCheckBox,
    QScrollArea, QSplitter, QProgressBar
)

from tracking.orchestrator.runner import PipelineRunner
from tracking.core.registry import PREPROC_REGISTRY, MODEL_REGISTRY, EVAL_REGISTRY
# populate registries
from tracking.preproc import clahe  # noqa: F401
from tracking.models import template_matching, optical_flow_lk, faster_rcnn, yolov11, fast_speckle  # noqa: F401
"""注意: CSRT 模型已被停用 (不再匯入)。若需恢復，請在 tracking/models/__init__.py 取消註解 csrt 匯入。"""
from tracking.eval import evaluator  # noqa: F401


# ---- Wheel-safe spin boxes -------------------------------------------------
class NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, e):
        e.ignore()  # 防止滾輪誤觸


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, e):
        e.ignore()


class LowConfidenceReinitEditor(QWidget):
    valueChanged = Signal()

    SUPPORTED_DETECTOR_KEYS = {"weights"}

    def __init__(self, defaults: Dict[str, Any], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._defaults = dict(defaults or {})
        self._extra_detector_keys: Dict[str, Any] = {}

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
        self.edit_detector_weights = QLineEdit()
        self.edit_detector_weights.setPlaceholderText("留空=沿用 init detector 的權重，例如 best.pt")
        self.edit_detector_weights.editingFinished.connect(self._emit_changed)
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
        weights = self.edit_detector_weights.text().strip()
        if weights:
            detector_cfg["weights"] = weights
        else:
            detector_cfg.pop("weights", None)

        if detector_cfg:
            cfg["detector"] = detector_cfg
        else:
            cfg["detector"] = {}

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
        self.edit_detector_weights.setText(str(weights))

        self._extra_detector_keys = {
            k: v for k, v in detector_cfg.items() if k not in self.SUPPORTED_DETECTOR_KEYS
        }
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


class SimpleRunnerUI(QMainWindow):
    """UI: 預設即時雙向同步。沒有模式切換、沒有存檔（僅載入）。
    - Builder (左) 為主，但 Raw (右) 也可直接改；雙向解析。
    - Builder 改動: 立即覆寫 Raw（除非 Raw 正在等待解析）。
    - Raw 改動: 停止輸入 debounce 後解析；成功→回填 Builder；失敗→僅顯示錯誤並標紅。
    - 數值欄位禁止滑鼠滾輪誤動。
    """

    DEBOUNCE_MS = 800

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tracking Pipeline UI (Always Bi-Directional)")
        self.resize(1400, 820)

        # --- State ---
        self.pre_params: Dict[str, Dict[str, Any]] = {}
        self.model_params: Dict[str, Dict[str, Any]] = {}
        self._current_pre_name: Optional[str] = None
        self._current_model_name: Optional[str] = None
        self._active_model_defaults: Dict[str, Any] = {}
        self._pre_bindings: List[Tuple[str, Any, Any]] = []
        self._model_bindings: List[Tuple[str, Any, Any]] = []
        self._syncing = False
        self._raw_user_edit = False
        self._updating_raw_programmatically = False
        self._run_thread: Optional[QThread] = None
        self._run_worker: Optional[QObject] = None
        self._queue_items: List[Dict[str, Any]] = []
        self._queue_pending: List[Dict[str, Any]] = []
        self._queue_running: bool = False
        self._queue_error: bool = False
        self._queue_total: int = 0
        self._queue_completed: int = 0
        self._queue_results_root: Optional[str] = None
        self._queue_current_label: Optional[str] = None
        self._setting_exp_name = False

        # --- Root layout ---
        central = QWidget(); self.setCentralWidget(central)
        root_hsplit = QSplitter(Qt.Horizontal)
        lay_root = QVBoxLayout(central); lay_root.setContentsMargins(6,6,6,6); lay_root.addWidget(root_hsplit)

        # --- Left: Builder (scroll) ---
        left_scroll = QScrollArea(); left_scroll.setWidgetResizable(True)
        left_container = QWidget(); left_scroll.setWidget(left_container)
        left_layout = QVBoxLayout(left_container); left_layout.setContentsMargins(4,4,4,4)

        # File + dataset
        file_row = QHBoxLayout(); file_row.addWidget(QLabel("設定檔 (載入):"))
        self.edit_cfg = QLineEdit(); btn_browse = QPushButton("載入…"); btn_browse.clicked.connect(self.browse_and_load)
        file_row.addWidget(self.edit_cfg, 1); file_row.addWidget(btn_browse)
        left_layout.addLayout(file_row)

        ds_row = QHBoxLayout(); ds_row.addWidget(QLabel("Dataset Root:"))
        self.edit_root = QLineEdit(); self.edit_root.textChanged.connect(self._on_builder_changed)
        btn_root = QPushButton("選…"); btn_root.clicked.connect(self.browse_root)
        ds_row.addWidget(self.edit_root, 1); ds_row.addWidget(btn_root)
        left_layout.addLayout(ds_row)

        exp_row = QHBoxLayout(); exp_row.addWidget(QLabel("實驗名稱:"))
        self.edit_exp_name = QLineEdit(); self.edit_exp_name.setPlaceholderText("留空會自動依前處理 / 模型產生")
        self.edit_exp_name.textChanged.connect(self._on_builder_changed)
        btn_exp_reset = QPushButton("重設"); btn_exp_reset.clicked.connect(self._reset_experiment_name)
        exp_row.addWidget(self.edit_exp_name, 1); exp_row.addWidget(btn_exp_reset)
        left_layout.addLayout(exp_row)

        # Split group
        gb_split = QGroupBox("資料切分"); fl_split = QFormLayout(gb_split)
        self.split_method = QComboBox(); self.split_method.addItems(["video_level"]); self.split_method.currentIndexChanged.connect(self._on_builder_changed)
        self.split_r_train = NoWheelDoubleSpinBox(); self._setup_ratio_spin(self.split_r_train, 0.8)
        self.split_r_test = NoWheelDoubleSpinBox(); self._setup_ratio_spin(self.split_r_test, 0.2)
        self.kfold = NoWheelSpinBox(); self.kfold.setRange(1,20); self.kfold.setValue(1); self.kfold.valueChanged.connect(self._on_builder_changed)
        fl_split.addRow("Method", self.split_method)
        fl_split.addRow("Train Ratio", self.split_r_train)
        fl_split.addRow("Test Ratio", self.split_r_test)
        fl_split.addRow("K-Fold (1=off)", self.kfold)
        left_layout.addWidget(gb_split)

        # Preproc selection
        gb_pre = QGroupBox("Preprocessing Pipeline"); vb_pre = QVBoxLayout(gb_pre)
        lists_row = QHBoxLayout()
        self.list_pre_avail = QListWidget(); self.list_pre_avail.setSelectionMode(QAbstractItemView.SingleSelection)
        for name in sorted(PREPROC_REGISTRY.keys()): self.list_pre_avail.addItem(name)
        self.list_pre_sel = QListWidget(); self.list_pre_sel.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_pre_sel.currentRowChanged.connect(self._on_pre_selected_changed)
        lists_row.addWidget(self._wrap_box("可用", self.list_pre_avail), 1)
        btn_col = QVBoxLayout()
        for text, cb in (("→", self._pre_add), ("←", self._pre_remove), ("上移", self._pre_up), ("下移", self._pre_down)):
            b = QPushButton(text); b.clicked.connect(cb); btn_col.addWidget(b)
        btn_col.addStretch(1); lists_row.addLayout(btn_col)
        lists_row.addWidget(self._wrap_box("順序", self.list_pre_sel), 1)
        vb_pre.addLayout(lists_row)
        left_layout.addWidget(gb_pre)

        # Preproc params form
        self.pre_form_layout = QFormLayout(); gb_pre_params = QGroupBox("Preproc 參數（即時）"); gb_pre_params.setLayout(self.pre_form_layout)
        left_layout.addWidget(gb_pre_params)

        # Model + params
        gb_model = QGroupBox("Model"); vb_model = QVBoxLayout(gb_model)
        row_m = QHBoxLayout(); row_m.addWidget(QLabel("Name:"))
        self.combo_model = QComboBox()
        # 排除已停用的 CSRT 模型
        for n in sorted(MODEL_REGISTRY.keys()):
            if n == 'CSRT':
                continue
            self.combo_model.addItem(n)
        self.combo_model.currentIndexChanged.connect(self._on_model_index_changed)
        row_m.addWidget(self.combo_model, 1); vb_model.addLayout(row_m)
        self.model_form_layout = QFormLayout(); gb_model_params = QGroupBox("Model 參數（即時）"); gb_model_params.setLayout(self.model_form_layout); vb_model.addWidget(gb_model_params)
        left_layout.addWidget(gb_model)

        # Evaluation
        gb_eval = QGroupBox("Evaluation"); fl_eval = QFormLayout(gb_eval)
        self.combo_eval = QComboBox(); [self.combo_eval.addItem(n) for n in sorted(EVAL_REGISTRY.keys())]
        self.combo_eval.currentIndexChanged.connect(self._on_builder_changed)
        self.chk_viz = QCheckBox("Visualize"); self.chk_viz.stateChanged.connect(self._on_builder_changed)
        self.spn_viz_samples = NoWheelSpinBox(); self.spn_viz_samples.setRange(1,10000); self.spn_viz_samples.setValue(10); self.spn_viz_samples.valueChanged.connect(self._on_builder_changed)
        fl_eval.addRow("Evaluator", self.combo_eval)
        h_viz = QHBoxLayout(); h_viz.addWidget(self.chk_viz); h_viz.addWidget(QLabel("samples")); h_viz.addWidget(self.spn_viz_samples); h_viz.addStretch(1)
        box_viz = QWidget(); box_viz.setLayout(h_viz); fl_eval.addRow("Visualization", box_viz)
        left_layout.addWidget(gb_eval)

        # Actions
        act_row = QHBoxLayout()
        self.btn_load = QPushButton("載入設定檔"); self.btn_load.clicked.connect(self.load_from_path)
        self.btn_run = QPushButton("執行 Pipeline"); self.btn_run.clicked.connect(self.run_pipeline)
        self.progress = QProgressBar(); self.progress.setMaximumHeight(14); self.progress.setTextVisible(False); self.progress.hide()
        self.lbl_status = QLabel("Ready")
        act_row.addWidget(self.btn_load); act_row.addWidget(self.btn_run); act_row.addWidget(self.progress, 1); act_row.addWidget(self.lbl_status)
        left_layout.addLayout(act_row)

        queue_group = QGroupBox("排程隊列")
        queue_layout = QVBoxLayout(queue_group); queue_layout.setContentsMargins(6,6,6,6)
        header_row = QHBoxLayout()
        self.btn_queue_add = QPushButton("加入排程"); self.btn_queue_add.clicked.connect(self._queue_add_current)
        self.btn_queue_remove = QPushButton("移除選取"); self.btn_queue_remove.clicked.connect(self._queue_remove_selected)
        self.btn_queue_clear = QPushButton("清空"); self.btn_queue_clear.clicked.connect(self._queue_clear)
        header_row.addWidget(self.btn_queue_add); header_row.addWidget(self.btn_queue_remove); header_row.addWidget(self.btn_queue_clear); header_row.addStretch(1)
        queue_layout.addLayout(header_row)
        self.list_queue = QListWidget(); self.list_queue.setSelectionMode(QAbstractItemView.SingleSelection)
        queue_layout.addWidget(self.list_queue, 1)
        run_row = QHBoxLayout()
        self.btn_queue_run = QPushButton("執行排程"); self.btn_queue_run.clicked.connect(self.run_queue)
        run_row.addWidget(self.btn_queue_run); run_row.addStretch(1)
        queue_layout.addLayout(run_row)
        left_layout.addWidget(queue_group)

        left_layout.addStretch(1)

        root_hsplit.addWidget(left_scroll)

        # --- Right: Raw + Logs ---
        right_vsplit = QSplitter(Qt.Vertical); root_hsplit.addWidget(right_vsplit)
        root_hsplit.setStretchFactor(0,3); root_hsplit.setStretchFactor(1,4)

        raw_container = QWidget(); raw_layout = QVBoxLayout(raw_container); raw_layout.setContentsMargins(4,4,4,4)
        raw_header = QHBoxLayout(); raw_header.addWidget(QLabel("Raw Config（可直接編輯，800ms 延遲解析）")); raw_header.addStretch(1)
        raw_layout.addLayout(raw_header)
        self.txt_cfg = QTextEdit(); self.txt_cfg.setPlaceholderText("此處顯示 / 可編輯 YAML（或 JSON）。")
        self.txt_cfg.setStyleSheet("QTextEdit { font-family: Consolas, 'Courier New', monospace; font-size:12px; }")
        raw_layout.addWidget(self.txt_cfg, 1)
        right_vsplit.addWidget(raw_container)

        logs_container = QWidget(); logs_layout = QVBoxLayout(logs_container); logs_layout.setContentsMargins(4,4,4,4)
        logs_layout.addWidget(QLabel("Logs"))
        self.txt_log = QTextEdit(); self.txt_log.setReadOnly(True); logs_layout.addWidget(self.txt_log, 1)
        right_vsplit.addWidget(logs_container); right_vsplit.setStretchFactor(0,4); right_vsplit.setStretchFactor(1,2)

        # Debounce timer
        self._raw_timer = QTimer(self); self._raw_timer.setSingleShot(True); self._raw_timer.timeout.connect(self._attempt_parse_raw)
        self.txt_cfg.textChanged.connect(self._on_raw_text_changed)

        # Init forms
        self._on_model_changed(self.combo_model.currentText())
        self._on_builder_changed(force=True)

    # ---------------- Utility / Setup -----------------
    def _setup_ratio_spin(self, sp: NoWheelDoubleSpinBox, val: float):
        sp.setRange(0.0, 1.0)
        sp.setSingleStep(0.0001)
        sp.setDecimals(6)
        sp.setValue(val)
        sp.valueChanged.connect(self._on_builder_changed)

    def _wrap_box(self, title: str, w: QWidget):
        gb = QGroupBox(title); l = QVBoxLayout(gb); l.setContentsMargins(4,4,4,4); l.addWidget(w); return gb

    def _set_experiment_name(self, name: str):
        if not hasattr(self, 'edit_exp_name'):
            return
        text = name or ""
        self._setting_exp_name = True
        try:
            self.edit_exp_name.blockSignals(True)
            self.edit_exp_name.setText(text)
        finally:
            self.edit_exp_name.blockSignals(False)
            self._setting_exp_name = False

    def _sanitize_experiment_name(self, name: str) -> str:
        text = (name or "").strip()
        if not text:
            return "experiment"
        text = re.sub(r"[\r\n]+", "_", text)
        text = re.sub(r"[\\/:*?\"<>|]", "_", text)
        text = text.strip(" _")
        return text or "experiment"

    def _generate_default_experiment_name(self, pre_list: List[str], model: str) -> str:
        pre = "_".join(pre_list) if pre_list else "no_pre"
        mdl = model or "model"
        return f"exp_{pre}_{mdl}"

    def _reset_experiment_name(self):
        pre_list = [self.list_pre_sel.item(i).text() for i in range(self.list_pre_sel.count())]
        model = self.combo_model.currentText()
        default = self._generate_default_experiment_name(pre_list, model)
        self._set_experiment_name(default)
        if not self._syncing:
            self._on_builder_changed(force=True)

    # ---------------- File ops -----------------
    def browse_and_load(self):
        path, _ = QFileDialog.getOpenFileName(self, "選擇設定檔", os.getcwd(), "YAML/JSON (*.yaml *.yml *.json)")
        if path:
            self.edit_cfg.setText(path); self.load_from_path()

    def load_from_path(self):
        path = self.edit_cfg.text().strip()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "提示", "請先輸入 / 選擇有效路徑")
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                txt = f.read()
            self._set_raw_text_programmatically(txt)
            cfg = self._parse_raw_text()
            if cfg is not None:
                # File load 是唯一會回填 Builder 的情境
                self._apply_cfg_to_builder(cfg)
                self._on_builder_changed(force=True)
                self._set_status("載入並套用", good=True)
            else:
                self._highlight_raw_error(True)
                self._set_status("載入解析失敗", good=False)
                self.log('[Load] 解析失敗，僅顯示文字。')
        except Exception as e:
            QMessageBox.critical(self, "讀取失敗", str(e))

    def browse_root(self):
        d = QFileDialog.getExistingDirectory(self, "選擇 Dataset Root", os.getcwd())
        if d: self.edit_root.setText(d)

    # ---------------- Logging -----------------
    def log(self, msg: str):
        self.txt_log.append(msg)
        try: print(msg, flush=True)
        except Exception: pass

    def _set_status(self, text: str, good: bool | None = None):
        color = None
        if good is True: color = "#22863a"
        elif good is False: color = "#d73a49"
        if color:
            self.lbl_status.setStyleSheet(f"color:{color}; font-weight:600;")
        else:
            self.lbl_status.setStyleSheet("")
        self.lbl_status.setText(text)

    # ---------------- Config build / parse -----------------
    def build_config_dict(self) -> Dict[str, Any]:
        ds_root = self.edit_root.text().strip()
        pre_list = [self.list_pre_sel.item(i).text() for i in range(self.list_pre_sel.count())]
        # Preserve edits
        self._save_current_pre_form(); self._save_model_form()
        model = self.combo_model.currentText()
        evaluator = self.combo_eval.currentText()
        pipeline_steps = []
        for p in pre_list:
            params = dict(self.pre_params.get(p) or getattr(PREPROC_REGISTRY[p], 'DEFAULT_CONFIG', {}))
            pipeline_steps.append({'type': 'preproc', 'name': p, 'params': params})
        defaults = getattr(MODEL_REGISTRY[model], 'DEFAULT_CONFIG', {})
        mdl_user = self.model_params.get(model) or {}
        mdl_params = {**defaults, **mdl_user}
        if model in ('CSRT', 'OpticalFlowLK'): mdl_params.pop('init_box', None)
        pipeline_steps.append({'type': 'model', 'name': model, 'params': mdl_params})
        exp_name_text = self.edit_exp_name.text().strip()
        if not exp_name_text:
            default_name = self._generate_default_experiment_name(pre_list, model)
            if self.edit_exp_name.text() != default_name:
                self._set_experiment_name(default_name)
            exp_name_text = default_name
        sanitized_name = self._sanitize_experiment_name(exp_name_text)
        if sanitized_name != exp_name_text:
            self._set_experiment_name(sanitized_name)
        exp_name_final = sanitized_name
        cfg = {
            'seed': 42,
            'dataset': {
                'root': ds_root,
                'split': {
                    'method': self.split_method.currentText(),
                    'ratios': [self.split_r_train.value(), self.split_r_test.value()],
                    'k_fold': int(self.kfold.value())
                }
            },
            'experiments': [
                {'name': exp_name_final, 'pipeline': pipeline_steps}
            ],
            'evaluation': {'evaluator': evaluator}
        }
        if self.chk_viz.isChecked():
            cfg.setdefault('evaluation', {}).setdefault('visualize', {})['enabled'] = True
            cfg['evaluation']['visualize']['samples'] = int(self.spn_viz_samples.value())
        return cfg

    def _serialize_cfg(self, cfg: Dict[str, Any]) -> str:
        try:
            import yaml  # type: ignore
            return yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
        except Exception:
            return json.dumps(cfg, ensure_ascii=False, indent=2)

    def _parse_raw_text(self) -> Optional[Dict[str, Any]]:
        txt = self.txt_cfg.toPlainText()
        if not txt.strip(): return None
        # Try YAML first (supports JSON subset)
        try:
            import yaml  # type: ignore
            data = yaml.safe_load(txt)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        # Fallback JSON
        try:
            data = json.loads(txt)
            if isinstance(data, dict): return data
        except Exception:
            pass
        return None

    # ---------------- Synchronization -----------------
    def _on_builder_changed(self, force: bool = False):
        if self._setting_exp_name and not force:
            return
        if self._syncing:  # currently applying parsed raw into builder
            return
        # If user currently editing raw (pending parse) skip builder->raw except forced
        if self._raw_user_edit and not force:
            return
        cfg = self.build_config_dict()
        txt = self._serialize_cfg(cfg)
        self._set_raw_text_programmatically(txt)
        self._set_status("Builder → Raw", good=True)

    def _on_raw_text_changed(self):
        if self._updating_raw_programmatically:
            return
        self._raw_user_edit = True
        self._set_status("等待解析…")
        self._raw_timer.start(self.DEBOUNCE_MS)

    def _attempt_parse_raw(self):
        # Raw 編輯成功後回填 Builder
        if not self._raw_user_edit:
            return
        cfg = self._parse_raw_text()
        if cfg is None:
            self._highlight_raw_error(True)
            self._set_status("Raw 解析失敗", good=False)
            return
        try:
            self._syncing = True
            self._apply_cfg_to_builder(cfg)
            self._syncing = False
            self._raw_user_edit = False
            self._highlight_raw_error(False)
            self._set_status("Raw → Builder", good=True)
            # 標準化格式顯示
            self._on_builder_changed(force=True)
            self.log('[RawParse] 已回填 Builder')
        except Exception as e:
            self._syncing = False
            self._highlight_raw_error(True)
            self._set_status("套用失敗", good=False)
            self.log(f'[RawParse][Error] {e}')

    def _highlight_raw_error(self, err: bool):
        if err:
            self.txt_cfg.setStyleSheet("QTextEdit { font-family:Consolas; font-size:12px; border:2px solid #d73a49; }")
        else:
            self.txt_cfg.setStyleSheet("QTextEdit { font-family:Consolas; font-size:12px; }")

    def _set_raw_text_programmatically(self, text: str):
        self._updating_raw_programmatically = True
        self.txt_cfg.blockSignals(True)
        self.txt_cfg.setPlainText(text)
        self.txt_cfg.blockSignals(False)
        self._updating_raw_programmatically = False

    # ---------------- Apply parsed cfg -> Builder -----------------
    def _apply_cfg_to_builder(self, cfg: Dict[str, Any]):
        ds = cfg.get('dataset', {}) or {}
        root = ds.get('root');
        if isinstance(root, str): self.edit_root.setText(root)
        split = ds.get('split', {}) or {}
        ratios = split.get('ratios', [self.split_r_train.value(), self.split_r_test.value()])
        if isinstance(ratios, (list, tuple)) and len(ratios) >= 2:
            try:
                self.split_r_train.setValue(float(ratios[0]))
                self.split_r_test.setValue(float(ratios[1]))
            except Exception: pass
        try: self.kfold.setValue(int(split.get('k_fold', self.kfold.value()) or 1))
        except Exception: pass

        # Experiments (take first)
        exps = cfg.get('experiments') or []
        if exps and isinstance(exps, list):
            first = exps[0] or {}
            pipeline = first.get('pipeline') or []
            pre_names = [s.get('name') for s in pipeline if isinstance(s, dict) and s.get('type') == 'preproc']
            model_steps = [s for s in pipeline if isinstance(s, dict) and s.get('type') == 'model']
            # Pre list
            # capture previous selection set to detect differences
            prev = [self.list_pre_sel.item(i).text() for i in range(self.list_pre_sel.count())]
            if prev != pre_names:
                self.list_pre_sel.clear()
                for name in pre_names:
                    if name in PREPROC_REGISTRY:
                        self.list_pre_sel.addItem(name)
                        step = next((s for s in pipeline if s.get('type') == 'preproc' and s.get('name') == name), None)
                        if step:
                            self.pre_params[name] = dict(step.get('params') or {})
                # force current selection to first item for param form update
                if self.list_pre_sel.count():
                    self.list_pre_sel.setCurrentRow(0)
            # Model
            if model_steps:
                mstep = model_steps[0]
                mname = mstep.get('name')
                if mname in MODEL_REGISTRY:
                    idx = self.combo_model.findText(mname)
                    if idx >= 0:
                        self.combo_model.blockSignals(True)
                        self.combo_model.setCurrentIndex(idx)
                        self.combo_model.blockSignals(False)
                        raw_params = dict(mstep.get('params') or {})
                        if mname in ('CSRT', 'OpticalFlowLK'): raw_params.pop('init_box', None)
                        self.model_params[mname] = raw_params
                        self._on_model_changed(mname)
            exp_name_loaded = first.get('name')
            if isinstance(exp_name_loaded, str) and exp_name_loaded.strip():
                self._set_experiment_name(exp_name_loaded)
            else:
                fallback = self._generate_default_experiment_name(pre_names, self.combo_model.currentText())
                self._set_experiment_name(fallback)
        # Evaluator
        ev = (cfg.get('evaluation') or {}).get('evaluator')
        if isinstance(ev, str):
            idx = self.combo_eval.findText(ev)
            if idx >= 0: self.combo_eval.setCurrentIndex(idx)
        viz = (cfg.get('evaluation') or {}).get('visualize') or {}
        self.chk_viz.setChecked(bool(viz.get('enabled', self.chk_viz.isChecked())))
        try: self.spn_viz_samples.setValue(int(viz.get('samples', self.spn_viz_samples.value())))
        except Exception: pass
        # Refresh forms (preproc current selection)
        self._on_pre_selected_changed(self.list_pre_sel.currentRow())

    # ---------------- Dynamic Forms -----------------
    def _on_pre_selected_changed(self, row: int):
        self._save_current_pre_form()
        name = self.list_pre_sel.item(row).text() if 0 <= row < self.list_pre_sel.count() else None
        self._current_pre_name = name
        self._clear_form(self.pre_form_layout)
        self._pre_bindings = []
        if not name: return
        params = dict(getattr(PREPROC_REGISTRY[name], 'DEFAULT_CONFIG', {}))
        params.update(self.pre_params.get(name, {}))
        for k, v in params.items():
            w, getter, setter = self._make_editor(k, v, scope='preproc'); setter(v)
            self.pre_form_layout.addRow(QLabel(k), w)
            self._pre_bindings.append((k, getter, setter))
        self._on_builder_changed()

    def _on_model_changed(self, name: str):
        if self._current_model_name:
            self._save_model_form(self._current_model_name)
        self._clear_form(self.model_form_layout)
        self._model_bindings = []
        defaults_base = dict(getattr(MODEL_REGISTRY[name], 'DEFAULT_CONFIG', {}))
        self._active_model_defaults = defaults_base
        user = self.model_params.get(name, {})
        defaults = dict(defaults_base)
        defaults.update(user)
        if name in ('CSRT', 'OpticalFlowLK'): defaults.pop('init_box', None)
        for k, v in defaults.items():
            w, getter, setter = self._make_editor(k, v, scope='model'); setter(v)
            self.model_form_layout.addRow(QLabel(k), w)
            self._model_bindings.append((k, getter, setter))
        self._current_model_name = name
        self._on_builder_changed()

    def _on_model_index_changed(self, idx: int):
        if idx < 0: return
        self._on_model_changed(self.combo_model.itemText(idx))

    def _save_current_pre_form(self):
        if not self._current_pre_name or not self._pre_bindings: return
        self.pre_params[self._current_pre_name] = {k: g() for k, g, _ in self._pre_bindings}

    def _save_model_form(self, name: Optional[str] = None):
        if name is None: name = self.combo_model.currentText()
        if not name or not self._model_bindings: return
        params = {k: g() for k, g, _ in self._model_bindings}
        if name in ('CSRT', 'OpticalFlowLK'): params.pop('init_box', None)
        self.model_params[name] = params

    def _clear_form(self, form: QFormLayout):
        while form.rowCount(): form.removeRow(0)

    def _make_editor(self, key: str, value: Any, scope: str):
        def register(widget, signal):
            sig = getattr(widget, signal, None)
            if sig: sig.connect(lambda *_: self._on_param_widget_changed(scope))
        if scope == 'model' and key == 'first_frame_source':
            combo = QComboBox()
            choices: List[Tuple[str, str]] = [
                ("Ground Truth (GT)", "gt"),
                ("YOLO 檢測", "yolo"),
                ("Auto (GT→YOLO)", "auto"),
            ]
            value_to_index: Dict[str, int] = {}
            for idx, (label, data) in enumerate(choices):
                combo.addItem(label, data)
                value_to_index[data] = idx

            def _canonical(v: Any) -> str:
                text = str(v or "").strip().lower()
                alias_map = {
                    "groundtruth": "gt",
                    "ground_truth": "gt",
                    "yolov11": "yolo",
                    "yolo_v11": "yolo",
                    "detector": "yolo",
                }
                if not text:
                    return "gt"
                return alias_map.get(text, text)

            def _get_value() -> str:
                data = combo.currentData()
                if isinstance(data, str):
                    return data
                cur = combo.currentText().strip().lower()
                return _canonical(cur)

            def _set_value(v: Any) -> None:
                canon = _canonical(v)
                idx = value_to_index.get(canon)
                if idx is None:
                    combo.addItem(canon, canon)
                    idx = combo.count() - 1
                    value_to_index[canon] = idx
                combo.setCurrentIndex(idx)

            register(combo, 'currentIndexChanged')
            return combo, _get_value, _set_value
        if scope == 'model' and key == 'low_confidence_reinit':
            defaults = {}
            if isinstance(self._active_model_defaults, dict):
                defaults = dict(self._active_model_defaults.get('low_confidence_reinit', {}))
            widget = LowConfidenceReinitEditor(defaults, parent=self)

            def _get():
                return widget.get_value()

            def _set(v):
                widget.set_value(v)

            register(widget, 'valueChanged')
            return widget, _get, _set
        if isinstance(value, bool):
            w = QCheckBox(); register(w, 'stateChanged')
            return w, lambda: bool(w.isChecked()), lambda v: w.setChecked(bool(v))
        if isinstance(value, int):
            w = NoWheelSpinBox()
            # Provide safer ranges for known keys to avoid invalid hyperparams (e.g., StepLR step_size>0)
            if key in ("batch_size", "epochs", "k_fold", "num_workers", "inference_batch", "step_size"):
                # clamp to at least 1 where required
                lo = 0 if key in ("num_workers",) else 1
                hi = 10**9
                w.setRange(lo, hi)
                def _set_int(v):
                    try:
                        iv = int(v)
                    except Exception:
                        iv = lo
                    if key == "step_size" and iv < 1:
                        iv = 1
                    if key in ("batch_size", "epochs", "k_fold", "inference_batch") and iv < 1:
                        iv = 1
                    if key == "num_workers" and iv < 0:
                        iv = 0
                    w.setValue(iv)
                register(w, 'valueChanged')
                return w, lambda: int(w.value()), _set_int
            # Fallback generic int editor
            w.setRange(-10**9, 10**9); register(w, 'valueChanged')
            return w, lambda: int(w.value()), lambda v: w.setValue(int(v))
        if isinstance(value, float):
            w = QLineEdit();
            try:
                rx = QRegularExpression(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
                w.setValidator(QRegularExpressionValidator(rx))
            except Exception: pass
            orig = float(value)
            def _get():
                t = w.text().strip();
                if not t: return orig
                try: return float(t)
                except Exception: return orig
            def _set(v):
                try: w.setText(format(float(v), '.12g'))
                except Exception: w.setText(str(v))
            register(w, 'editingFinished')
            return w, _get, _set
        if isinstance(value, str):
            w = QLineEdit(); register(w, 'editingFinished')
            return w, lambda: w.text(), lambda v: w.setText(str(v))
        # Fallback JSON-serializable
        w = QLineEdit();
        def _get():
            txt = w.text().strip()
            if not txt:
                return value
            try:
                return json.loads(txt)
            except Exception:
                return value
        def _set(v):
            try: w.setText(json.dumps(v))
            except Exception: w.setText(str(v))
        register(w, 'editingFinished')
        return w, _get, _set

    def _on_param_widget_changed(self, scope: str):
        if scope == 'preproc': self._save_current_pre_form()
        elif scope == 'model': self._save_model_form()
        self._on_builder_changed()

    # ---------------- Preproc selection ops -----------------
    def _pre_add(self):
        it = self.list_pre_avail.currentItem();
        if not it: return
        name = it.text()
        for i in range(self.list_pre_sel.count()):
            if self.list_pre_sel.item(i).text() == name: return
        self.list_pre_sel.addItem(name)
        self._on_builder_changed()

    def _pre_remove(self):
        row = self.list_pre_sel.currentRow()
        if row >= 0:
            self.list_pre_sel.takeItem(row)
            self._on_builder_changed()

    def _pre_up(self):
        row = self.list_pre_sel.currentRow()
        if row > 0:
            it = self.list_pre_sel.takeItem(row)
            self.list_pre_sel.insertItem(row - 1, it)
            self.list_pre_sel.setCurrentRow(row - 1)
            self._on_builder_changed()

    def _pre_down(self):
        row = self.list_pre_sel.currentRow()
        if 0 <= row < self.list_pre_sel.count() - 1:
            it = self.list_pre_sel.takeItem(row)
            self.list_pre_sel.insertItem(row + 1, it)
            self.list_pre_sel.setCurrentRow(row + 1)
            self._on_builder_changed()

    # ---------------- Run -----------------
    # ---------------- Run (threaded with progress) -----------------
    def run_pipeline(self):
        if self._queue_running:
            QMessageBox.information(self, '排程執行中', '排程目前正在執行，請先等待完成。')
            return
        if self._run_thread is not None:
            QMessageBox.information(self, '執行中', 'Pipeline 已在執行。')
            return
        if self._raw_user_edit:
            QMessageBox.warning(self, '尚未驗證', 'Raw 尚未驗證完成，請等待或停止輸入。')
            return
        cfg = self.build_config_dict()
        try:
            proj_root = os.path.dirname(os.path.abspath(__file__))
            cfg.setdefault('output', {})['results_root'] = os.path.join(proj_root, 'results')
        except Exception:
            pass
        self._start_run_thread(cfg)

    def _start_run_thread(self, cfg: dict):
        class _Worker(QObject):
            finished = Signal()
            error = Signal(str)
            progress = Signal(str, int, int, dict)
            log = Signal(str)

            def __init__(self, cfg):
                super().__init__()
                self.cfg = cfg

            def run(self):
                from traceback import format_exc
                try:
                    def _logger(msg: str):
                        self.log.emit(msg)
                    runner = PipelineRunner(
                        self.cfg,
                        logger=_logger,
                        progress_cb=lambda stage, cur, tot, extra: self.progress.emit(stage, cur, tot, extra)
                    )
                    runner.run()
                    self.finished.emit()
                except Exception as e:
                    self.log.emit(f'執行錯誤: {e}\n{format_exc()}')
                    self.error.emit(str(e))

        # UI 更新 (主執行緒)
        self.log('開始執行…')
        self._set_status('執行中…')
        self.progress.show()
        self.progress.setRange(0, 0)
        self.btn_run.setEnabled(False)
        self.btn_load.setEnabled(False)

        # Thread setup
        self._run_thread = QThread(self)
        self._run_worker = _Worker(cfg)
        self._run_worker.moveToThread(self._run_thread)
        self._run_thread.started.connect(self._run_worker.run)
        self._run_worker.finished.connect(self._on_run_finished)
        self._run_worker.error.connect(self._on_run_error)
        self._run_worker.progress.connect(self._on_progress_event)
        self._run_worker.log.connect(self.log)
        self._run_worker.finished.connect(self._cleanup_run_thread)
        self._run_worker.error.connect(self._cleanup_run_thread)
        self._run_thread.start()

    def _on_run_finished(self):
        if self._queue_running:
            self.log('排程項目完成。')
        else:
            self.log('完成。')
            self._set_status('完成', good=True)
        self._end_progress()
        self._queue_handle_run_completion(True)

    def _on_run_error(self, msg: str):
        QMessageBox.critical(self, '執行失敗', msg)
        self._set_status('失敗', good=False)
        self._end_progress()
        self._queue_handle_run_completion(False)

    def _end_progress(self):
        self.progress.hide(); self.progress.setRange(0,100)
        if not self._queue_running:
            self.btn_run.setEnabled(True); self.btn_load.setEnabled(True)

    def _cleanup_run_thread(self):
        if self._run_thread:
            self._run_thread.quit(); self._run_thread.wait(5000)
            self._run_thread = None; self._run_worker = None

    # ---- Queue scheduling helpers ----
    def _queue_default_results_root(self) -> str:
        try:
            proj_root = os.path.dirname(os.path.abspath(__file__))
        except Exception:
            proj_root = os.getcwd()
        return os.path.join(proj_root, 'results')

    def _queue_add_current(self):
        if self._queue_running:
            QMessageBox.warning(self, '排程進行中', '排程執行期間無法新增項目。')
            return
        cfg = self.build_config_dict()
        snapshot = copy.deepcopy(cfg)
        experiments = snapshot.get('experiments') or []
        exp = experiments[0] if experiments else {}
        pipeline_steps = exp.get('pipeline') or []
        model_name = pipeline_steps[-1].get('name') if pipeline_steps else self.combo_model.currentText()
        pre_names = [step.get('name') for step in pipeline_steps if step.get('type') == 'preproc']
        dataset_root = (snapshot.get('dataset') or {}).get('root', '')
        display_name = self.edit_exp_name.text().strip()
        label_base = display_name or exp.get('name') or f"Exp{len(self._queue_items)+1}"
        label_parts = [label_base, f"model={model_name}"]
        if pre_names:
            label_parts.append(f"pre={' + '.join(pre_names)}")
        if dataset_root:
            label_parts.append(os.path.basename(dataset_root))
        label = " | ".join(label_parts)
        entry = {'config': snapshot, 'label': label}
        self._queue_items.append(entry)
        item = QListWidgetItem(label)
        tooltip_lines = [f"資料集: {dataset_root or '(未設定)'}", f"模型: {model_name}"]
        if pre_names:
            tooltip_lines.append(f"前處理: {', '.join(pre_names)}")
        item.setToolTip("\n".join(tooltip_lines))
        self.list_queue.addItem(item)
        self.log(f"已加入排程：{label}")

    def _queue_remove_selected(self):
        if self._queue_running:
            QMessageBox.warning(self, '排程進行中', '排程執行期間無法移除。')
            return
        row = self.list_queue.currentRow()
        if row < 0:
            return
        self.list_queue.takeItem(row)
        self._queue_items.pop(row)

    def _queue_clear(self):
        if self._queue_running:
            QMessageBox.warning(self, '排程進行中', '排程執行期間無法清空排程。')
            return
        self.list_queue.clear()
        self._queue_items.clear()

    def run_queue(self):
        if self._queue_running:
            QMessageBox.information(self, '排程執行中', '排程已在執行，請先等待完成。')
            return
        if self._run_thread is not None:
            QMessageBox.information(self, '執行中', 'Pipeline 已在執行。')
            return
        if not self._queue_items:
            QMessageBox.information(self, '排程為空', '請先加入至少一組實驗。')
            return
        if self._raw_user_edit:
            QMessageBox.warning(self, '尚未驗證', 'Raw 尚未驗證完成，請等待或停止輸入。')
            return
        default_root = self._queue_default_results_root()
        schedule_root: Optional[str] = None
        if len(self._queue_items) > 1:
            stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            schedule_root = os.path.join(default_root, f"{stamp}_schedule_{len(self._queue_items)}exp")
            try:
                os.makedirs(schedule_root, exist_ok=True)
            except Exception as e:
                QMessageBox.critical(self, '建立排程資料夾失敗', str(e))
                return
        queue_snapshot: List[Dict[str, Any]] = []
        for idx, item in enumerate(self._queue_items, start=1):
            cfg_copy = copy.deepcopy(item['config'])
            out_cfg = cfg_copy.setdefault('output', {})
            out_cfg['results_root'] = schedule_root or default_root
            if schedule_root:
                sched_meta = out_cfg.setdefault('schedule', {})
                sched_meta.setdefault('batch_root', schedule_root)
                sched_meta['order'] = idx
                sched_meta['total'] = len(self._queue_items)
            queue_snapshot.append({'config': cfg_copy, 'label': item['label'], 'index': idx})
        self._queue_pending = queue_snapshot
        self._queue_total = len(queue_snapshot)
        self._queue_completed = 0
        self._queue_running = True
        self._queue_error = False
        self._queue_results_root = schedule_root
        self._queue_current_label = None
        self.btn_queue_add.setEnabled(False)
        self.btn_queue_remove.setEnabled(False)
        self.btn_queue_clear.setEnabled(False)
        self.btn_queue_run.setEnabled(False)
        self.list_queue.setEnabled(False)
        self.btn_run.setEnabled(False)
        self.btn_load.setEnabled(False)
        self.log(f"排程開始，共 {self._queue_total} 組實驗。")
        if schedule_root:
            self.log(f"排程結果將存於：{schedule_root}")
        self._set_status(f"排程 0/{self._queue_total} 準備中…")
        self._queue_start_next()

    def _queue_start_next(self):
        if not self._queue_running:
            return
        if not self._queue_pending:
            self._finish_queue()
            return
        payload = self._queue_pending.pop(0)
        idx = self._queue_completed + 1
        label = payload.get('label', f"Exp{idx}")
        self._queue_current_label = label
        if self.list_queue.count():
            display_item = self.list_queue.item(0)
            display_item.setText(f"{label}（執行中 {idx}/{self._queue_total}）")
        self.log(f"[Queue] 開始第 {idx}/{self._queue_total}: {label}")
        self._set_status(f"排程 {idx}/{self._queue_total} 執行中…")
        self._start_run_thread(payload['config'])

    def _queue_handle_run_completion(self, success: bool):
        if not self._queue_running:
            return
        if self.list_queue.count():
            self.list_queue.takeItem(0)
        if self._queue_items:
            self._queue_items.pop(0)
        label = self._queue_current_label or ''
        self._queue_completed += 1
        if success:
            self.log(f"[Queue] 完成 {self._queue_completed}/{self._queue_total}: {label}")
            if self._queue_pending:
                self._set_status(f"排程 {self._queue_completed}/{self._queue_total} 完成，準備下一個…")
                QTimer.singleShot(0, self._queue_start_next)
            else:
                self._finish_queue()
        else:
            self._queue_error = True
            self.log(f"[Queue] 失敗：{label}。剩餘排程已取消。")
            self._queue_pending.clear()
            self._finish_queue()
        self._queue_current_label = None

    def _finish_queue(self):
        if not self._queue_running:
            return
        success = not self._queue_error
        if success and self._queue_results_root:
            self.log(f"排程結果資料夾: {self._queue_results_root}")
        self._queue_running = False
        self._queue_pending = []
        self._queue_total = 0
        self._queue_completed = 0
        self._queue_results_root = None
        self._queue_current_label = None
        self.btn_queue_add.setEnabled(True)
        self.btn_queue_remove.setEnabled(True)
        self.btn_queue_clear.setEnabled(True)
        self.btn_queue_run.setEnabled(True)
        self.list_queue.setEnabled(True)
        self.btn_run.setEnabled(True)
        self.btn_load.setEnabled(True)
        if success:
            self._set_status("排程完成", good=True)
            self.log("排程全部完成。")
        else:
            self._set_status("排程中途失敗", good=False)
            self.log("排程已停止，請檢查錯誤訊息後再試一次。")
        self._queue_error = False

    # ---- Progress event handler ----
    def _on_progress_event(self, stage: str, cur: int, tot: int, extra: dict):
        # stages: train_epoch_start/end, kfold_fold, eval_video
        if tot <= 0:
            self.progress.setRange(0,0); return
        self.progress.setRange(0, tot)
        # choose display value depending on stage
        if stage.startswith('train_epoch'):
            self.progress.setValue(cur)
            self._set_status(f"訓練 Epoch {cur}/{tot}")
        elif stage == 'kfold_fold':
            self.progress.setValue(cur)
            self._set_status(f"K-Fold {cur}/{tot}")
        elif stage == 'eval_video':
            self.progress.setValue(cur)
            self._set_status(f"測試影片 {cur}/{tot}")


def main():
    import sys
    app = QApplication(sys.argv)
    ui = SimpleRunnerUI(); ui.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
