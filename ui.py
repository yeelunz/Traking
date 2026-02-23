from __future__ import annotations
import copy
import json, os, re, time
from typing import Optional, Dict, Any, List, Tuple, Callable

from PySide6.QtCore import Qt, QRegularExpression, QTimer, QObject, Signal, QThread
from PySide6.QtGui import QRegularExpressionValidator
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QTextEdit, QLineEdit, QMessageBox, QComboBox, QSpinBox, QDoubleSpinBox,
    QFormLayout, QGroupBox, QListWidget, QListWidgetItem, QAbstractItemView, QCheckBox,
    QScrollArea, QSplitter, QProgressBar
)

from tracking.orchestrator.runner import PipelineRunner
from tracking.core.registry import (
    PREPROC_REGISTRY,
    MODEL_REGISTRY,
    EVAL_REGISTRY,
    FEATURE_EXTRACTOR_REGISTRY,
    CLASSIFIER_REGISTRY,
    SEGMENTATION_MODEL_REGISTRY,
)
# populate registries
from tracking.preproc import clahe  # noqa: F401
from tracking.models import template_matching, optical_flow_lk, faster_rcnn, yolov11, fast_speckle  # noqa: F401
"""注意: CSRT 模型已被停用 (不再匯入)。若需恢復，請在 tracking/models/__init__.py 取消註解 csrt 匯入。"""
from tracking.eval import evaluator  # noqa: F401
from tracking.classification import feature_extractors as _cls_feat  # noqa: F401
from tracking.classification import classifiers as _cls_clf  # noqa: F401
from tracking import segmentation as _seg_pkg  # noqa: F401

from runner_ui.widgets import (
    LowConfidenceReinitEditor,
    ModelWeightsComboBox,
    NoWheelComboBox,
    NoWheelDoubleSpinBox,
    NoWheelSpinBox,
)

from runner_ui.mixins.queue_mixin import QueueMixin

class SimpleRunnerUI(QMainWindow, QueueMixin):
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
        self._applying_cfg = False
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
        self.class_feature_params: Dict[str, Dict[str, Any]] = {}
        self.class_classifier_params: Dict[str, Dict[str, Any]] = {}
        self._current_class_feature: Optional[str] = None
        self._current_class_classifier: Optional[str] = None
        self._class_feature_bindings: List[Tuple[str, Any, Any]] = []
        self._class_classifier_bindings: List[Tuple[str, Any, Any]] = []

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
        self.split_method = NoWheelComboBox(); self.split_method.addItems(["video_level", "subject_level", "loso"]); self.split_method.currentIndexChanged.connect(self._on_builder_changed)
        self.chk_loso = QCheckBox("啟用 LOSO (Leave-One-Subject-Out)"); self.chk_loso.stateChanged.connect(self._on_loso_toggled)
        self.split_r_train = NoWheelDoubleSpinBox(); self._setup_ratio_spin(self.split_r_train, 0.8)
        self.split_r_test = NoWheelDoubleSpinBox(); self._setup_ratio_spin(self.split_r_test, 0.2)
        self.kfold = NoWheelSpinBox(); self.kfold.setRange(1,20); self.kfold.setValue(1); self.kfold.valueChanged.connect(self._on_builder_changed)
        fl_split.addRow("Method", self.split_method)
        fl_split.addRow("LOSO", self.chk_loso)
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

        # Preproc scope scheme (A/B/C) - applies to ALL preprocessing steps
        scheme_box = QGroupBox("前處理作用域方案 (A/B/C)")
        scheme_form = QFormLayout(scheme_box)
        self.combo_preproc_scheme = NoWheelComboBox()
        self.combo_preproc_scheme.addItem("A：Global（全圖前處理 → YOLO → ROI → Seg）", "A")
        self.combo_preproc_scheme.addItem("B：ROI（YOLO 原圖 → ROI 前處理 → Seg）", "B")
        self.combo_preproc_scheme.addItem("C：Hybrid（全圖前處理給 YOLO；ROI 前處理給 Seg）", "C")
        self.combo_preproc_scheme.setCurrentIndex(0)
        self.combo_preproc_scheme.currentIndexChanged.connect(self._on_builder_changed)
        scheme_form.addRow("Scheme", self.combo_preproc_scheme)
        vb_pre.addWidget(scheme_box)
        left_layout.addWidget(gb_pre)

        # Preproc params form
        self.pre_form_layout = QFormLayout(); gb_pre_params = QGroupBox("Preproc 參數（即時）"); gb_pre_params.setLayout(self.pre_form_layout)
        left_layout.addWidget(gb_pre_params)

        # Model + params
        gb_model = QGroupBox("Model"); vb_model = QVBoxLayout(gb_model)
        row_m = QHBoxLayout(); row_m.addWidget(QLabel("Name:"))
        self.combo_model = NoWheelComboBox()
        # 排除已停用的 CSRT 模型
        for n in sorted(MODEL_REGISTRY.keys()):
            if n == 'CSRT':
                continue
            self.combo_model.addItem(n)
        self.combo_model.currentIndexChanged.connect(self._on_model_index_changed)
        row_m.addWidget(self.combo_model, 1); vb_model.addLayout(row_m)
        self.model_form_layout = QFormLayout(); gb_model_params = QGroupBox("Model 參數（即時）"); gb_model_params.setLayout(self.model_form_layout); vb_model.addWidget(gb_model_params)
        left_layout.addWidget(gb_model)

        # Segmentation stage
        gb_seg = QGroupBox("Segmentation")
        vb_seg = QVBoxLayout(gb_seg)
        self.chk_seg_train = QCheckBox("訓練分割模型")
        self.chk_seg_train.setChecked(True)
        self.chk_seg_train.stateChanged.connect(self._on_seg_train_toggled)
        vb_seg.addWidget(self.chk_seg_train)
        seg_form = QFormLayout()
        self.combo_seg_model = NoWheelComboBox()
        _pretty_seg_labels = {
            "unetpp": "UNet++",
            "deeplabv3+": "DeepLabV3+",
        }
        for key in sorted(SEGMENTATION_MODEL_REGISTRY.keys()):
            label = "UNet++" if key.lower() == "unetpp" else key.upper()
            self.combo_seg_model.addItem(label, key)
        auto_mask_label = "Auto Mask (弱分割)"
        def _item_data_key(row: int) -> str:
            data = self.combo_seg_model.itemData(row)
            if data is None:
                return ""
            return str(data)

        if all(_item_data_key(i).lower() != "auto_mask" for i in range(self.combo_seg_model.count())):
            self.combo_seg_model.insertItem(0, auto_mask_label, "auto_mask")
        default_idx = self.combo_seg_model.findData('unetpp')
        if default_idx < 0:
            default_idx = self.combo_seg_model.findText('UNet++')
        if default_idx >= 0:
            self.combo_seg_model.setCurrentIndex(default_idx)
        self.combo_seg_model.currentIndexChanged.connect(self._on_seg_method_changed)
        seg_form.addRow("Method", self.combo_seg_model)

        self.combo_seg_checkpoint = ModelWeightsComboBox(self._get_segmentation_weights, parent=self)
        self.combo_seg_checkpoint.refresh_items()
        self.combo_seg_checkpoint.currentIndexChanged.connect(self._on_builder_changed)
        self.combo_seg_checkpoint.editTextChanged.connect(self._on_builder_changed)
        try:
            le_ckpt = self.combo_seg_checkpoint.lineEdit()
            if le_ckpt is not None:
                le_ckpt.setPlaceholderText("留空 = 使用本地訓練產出的權重")
        except Exception:
            pass
        seg_form.addRow("推論 checkpoint", self.combo_seg_checkpoint)

        self.edit_seg_encoder = QLineEdit("resnet34"); self.edit_seg_encoder.editingFinished.connect(self._on_builder_changed)
        seg_form.addRow("Encoder", self.edit_seg_encoder)
        self.edit_seg_weights = QLineEdit("imagenet"); self.edit_seg_weights.editingFinished.connect(self._on_builder_changed)
        seg_form.addRow("Encoder Weights", self.edit_seg_weights)

        self.seg_padding_min = NoWheelDoubleSpinBox(); self.seg_padding_min.setRange(0.0, 0.5); self.seg_padding_min.setDecimals(3); self.seg_padding_min.setValue(0.10); self.seg_padding_min.valueChanged.connect(self._on_builder_changed)
        self.seg_padding_max = NoWheelDoubleSpinBox(); self.seg_padding_max.setRange(0.0, 0.5); self.seg_padding_max.setDecimals(3); self.seg_padding_max.setValue(0.15); self.seg_padding_max.valueChanged.connect(self._on_builder_changed)
        self.seg_padding_inf = NoWheelDoubleSpinBox(); self.seg_padding_inf.setRange(0.0, 0.5); self.seg_padding_inf.setDecimals(3); self.seg_padding_inf.setValue(0.15); self.seg_padding_inf.valueChanged.connect(self._on_builder_changed)
        self.seg_jitter = NoWheelDoubleSpinBox(); self.seg_jitter.setRange(0.0, 0.5); self.seg_jitter.setDecimals(3); self.seg_jitter.setValue(0.05); self.seg_jitter.valueChanged.connect(self._on_builder_changed)
        seg_form.addRow("Train padding min", self.seg_padding_min)
        seg_form.addRow("Train padding max", self.seg_padding_max)
        seg_form.addRow("Inference padding", self.seg_padding_inf)
        seg_form.addRow("Random jitter", self.seg_jitter)

        self.seg_epochs = NoWheelSpinBox(); self.seg_epochs.setRange(1, 2000); self.seg_epochs.setValue(20); self.seg_epochs.valueChanged.connect(self._on_builder_changed)
        self.seg_batch = NoWheelSpinBox(); self.seg_batch.setRange(1, 2048); self.seg_batch.setValue(8); self.seg_batch.valueChanged.connect(self._on_builder_changed)
        self.seg_num_workers = NoWheelSpinBox(); self.seg_num_workers.setRange(0, 64); self.seg_num_workers.setValue(0); self.seg_num_workers.valueChanged.connect(self._on_builder_changed)
        seg_form.addRow("Epochs", self.seg_epochs)
        seg_form.addRow("Batch size", self.seg_batch)
        seg_form.addRow("Workers", self.seg_num_workers)

        self.seg_lr = QLineEdit("0.001"); self.seg_lr.editingFinished.connect(self._on_builder_changed)
        self.seg_weight_decay = QLineEdit("1e-5"); self.seg_weight_decay.editingFinished.connect(self._on_builder_changed)
        seg_form.addRow("Learning rate", self.seg_lr)
        seg_form.addRow("Weight decay", self.seg_weight_decay)

        self.seg_threshold = NoWheelDoubleSpinBox(); self.seg_threshold.setRange(0.0, 1.0); self.seg_threshold.setDecimals(3); self.seg_threshold.setValue(0.5); self.seg_threshold.valueChanged.connect(self._on_builder_changed)
        self.seg_val_ratio = NoWheelDoubleSpinBox(); self.seg_val_ratio.setRange(0.0, 0.9); self.seg_val_ratio.setDecimals(3); self.seg_val_ratio.setValue(0.0); self.seg_val_ratio.valueChanged.connect(self._on_builder_changed)
        self.seg_seed = NoWheelSpinBox(); self.seg_seed.setRange(0, 999999); self.seg_seed.setValue(0); self.seg_seed.valueChanged.connect(self._on_builder_changed)
        self.seg_redundancy = NoWheelSpinBox(); self.seg_redundancy.setRange(1, 32); self.seg_redundancy.setValue(1); self.seg_redundancy.valueChanged.connect(self._on_builder_changed)
        self.seg_dice_weight = NoWheelDoubleSpinBox(); self.seg_dice_weight.setRange(0.0, 10.0); self.seg_dice_weight.setDecimals(3); self.seg_dice_weight.setValue(1.0); self.seg_dice_weight.valueChanged.connect(self._on_builder_changed)
        self.seg_bce_weight = NoWheelDoubleSpinBox(); self.seg_bce_weight.setRange(0.0, 10.0); self.seg_bce_weight.setDecimals(3); self.seg_bce_weight.setValue(1.0); self.seg_bce_weight.valueChanged.connect(self._on_builder_changed)
        self.edit_seg_device = QLineEdit("auto"); self.edit_seg_device.editingFinished.connect(self._on_builder_changed)
        seg_form.addRow("Mask threshold", self.seg_threshold)
        seg_form.addRow("Validation ratio", self.seg_val_ratio)
        seg_form.addRow("Seed", self.seg_seed)
        seg_form.addRow("Redundancy", self.seg_redundancy)
        seg_form.addRow("Dice weight", self.seg_dice_weight)
        seg_form.addRow("BCE weight", self.seg_bce_weight)
        seg_form.addRow("Device", self.edit_seg_device)

        self.lbl_nnunet_plans = QLabel("nnUNet plans")
        self.edit_nnunet_plans = QLineEdit()
        self.edit_nnunet_plans.setPlaceholderText("選填：指向 nnUNetPlans.json")
        self.edit_nnunet_plans.editingFinished.connect(self._on_builder_changed)
        seg_form.addRow(self.lbl_nnunet_plans, self.edit_nnunet_plans)

        self.lbl_nnunet_config = QLabel("nnUNet config")
        self.edit_nnunet_config = QLineEdit("2d")
        self.edit_nnunet_config.editingFinished.connect(self._on_builder_changed)
        seg_form.addRow(self.lbl_nnunet_config, self.edit_nnunet_config)

        self.lbl_nnunet_arch = QLabel("Arch JSON")
        self.edit_nnunet_architecture = QLineEdit()
        self.edit_nnunet_architecture.setPlaceholderText("選填：自訂 architecture JSON")
        self.edit_nnunet_architecture.editingFinished.connect(self._on_builder_changed)
        seg_form.addRow(self.lbl_nnunet_arch, self.edit_nnunet_architecture)

        self.lbl_nnunet_highres = QLabel("")
        self.chk_nnunet_highres = QCheckBox("僅輸出最高解析度 head")
        self.chk_nnunet_highres.setChecked(True)
        self.chk_nnunet_highres.stateChanged.connect(lambda _v: self._on_builder_changed())
        seg_form.addRow(self.lbl_nnunet_highres, self.chk_nnunet_highres)

        self._nnunet_widgets = [
            self.lbl_nnunet_plans,
            self.edit_nnunet_plans,
            self.lbl_nnunet_config,
            self.edit_nnunet_config,
            self.lbl_nnunet_arch,
            self.edit_nnunet_architecture,
            self.lbl_nnunet_highres,
            self.chk_nnunet_highres,
        ]
        for widget in self._nnunet_widgets:
            widget.setVisible(False)

        vb_seg.addLayout(seg_form)
        self._seg_train_widgets = [
            self.seg_padding_min,
            self.seg_padding_max,
            self.seg_epochs,
            self.seg_batch,
            self.seg_num_workers,
            self.seg_lr,
            self.seg_weight_decay,
            self.seg_val_ratio,
            self.seg_seed,
            self.seg_redundancy,
            self.seg_dice_weight,
            self.seg_bce_weight,
            self.seg_jitter,
        ]
        self._update_segmentation_train_state()
        left_layout.addWidget(gb_seg)

        # Evaluation
        gb_eval = QGroupBox("Evaluation"); fl_eval = QFormLayout(gb_eval)
        self.combo_eval = NoWheelComboBox(); [self.combo_eval.addItem(n) for n in sorted(EVAL_REGISTRY.keys())]
        self.combo_eval.currentIndexChanged.connect(self._on_builder_changed)
        fl_eval.addRow("Evaluator", self.combo_eval)
        left_layout.addWidget(gb_eval)

        # Classification (optional)
        gb_class = QGroupBox("Classification (可選)")
        vb_class = QVBoxLayout(gb_class)
        self.chk_class_enabled = QCheckBox("啟用分類階段")
        self.chk_class_enabled.stateChanged.connect(self._on_classification_toggled)
        vb_class.addWidget(self.chk_class_enabled)

        form_class_basic = QFormLayout()
        form_class_basic.setContentsMargins(6, 0, 0, 0)

        # Label file row
        self.edit_class_label = QLineEdit()
        self.edit_class_label.setPlaceholderText("留空 = dataset.root/ann.txt")
        self.edit_class_label.editingFinished.connect(self._on_builder_changed)
        btn_class_label = QPushButton("選…")
        btn_class_label.clicked.connect(self._browse_class_label_file)
        self.btn_class_label = btn_class_label
        label_row_widget = QWidget()
        label_row_layout = QHBoxLayout(label_row_widget)
        label_row_layout.setContentsMargins(0, 0, 0, 0)
        label_row_layout.addWidget(self.edit_class_label, 1)
        label_row_layout.addWidget(btn_class_label)
        form_class_basic.addRow("Label 檔案", label_row_widget)

        # Feature extractor selection
        self.combo_class_feature = NoWheelComboBox()
        for name in sorted(FEATURE_EXTRACTOR_REGISTRY.keys()):
            self.combo_class_feature.addItem(name)
        self.combo_class_feature.currentIndexChanged.connect(self._on_class_feature_index_changed)
        form_class_basic.addRow("Feature Extractor", self.combo_class_feature)

        # Classifier selection
        self.combo_class_classifier = NoWheelComboBox()
        for name in sorted(CLASSIFIER_REGISTRY.keys()):
            self.combo_class_classifier.addItem(name)
        self.combo_class_classifier.currentIndexChanged.connect(self._on_class_classifier_index_changed)
        form_class_basic.addRow("Classifier", self.combo_class_classifier)

        vb_class.addLayout(form_class_basic)

        self.class_feature_form_layout = QFormLayout()
        self.gb_class_feat = QGroupBox("Feature Extractor 參數（即時）")
        self.gb_class_feat.setLayout(self.class_feature_form_layout)
        vb_class.addWidget(self.gb_class_feat)

        self.class_classifier_form_layout = QFormLayout()
        self.gb_class_clf = QGroupBox("Classifier 參數（即時）")
        self.gb_class_clf.setLayout(self.class_classifier_form_layout)
        vb_class.addWidget(self.gb_class_clf)

        left_layout.addWidget(gb_class)

        if self.combo_class_feature.count():
            self._on_class_feature_changed(self.combo_class_feature.currentText(), from_builder=True)
        if self.combo_class_classifier.count():
            self._on_class_classifier_changed(self.combo_class_classifier.currentText(), from_builder=True)
        self._update_classification_enabled_state()

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
        self.btn_queue_import = QPushButton("匯入排程…"); self.btn_queue_import.clicked.connect(self._queue_import_from_file)
        self.btn_queue_add = QPushButton("加入排程"); self.btn_queue_add.clicked.connect(self._queue_add_current)
        self.btn_queue_remove = QPushButton("移除選取"); self.btn_queue_remove.clicked.connect(self._queue_remove_selected)
        self.btn_queue_clear = QPushButton("清空"); self.btn_queue_clear.clicked.connect(self._queue_clear)
        header_row.addWidget(self.btn_queue_import)
        header_row.addWidget(self.btn_queue_add)
        header_row.addWidget(self.btn_queue_remove)
        header_row.addWidget(self.btn_queue_clear)
        header_row.addStretch(1)
        queue_layout.addLayout(header_row)
        self.list_queue = QListWidget(); self.list_queue.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_queue.itemDoubleClicked.connect(self._on_queue_item_double_clicked)
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

    def _models_root_dir(self) -> str:
        try:
            return os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        except Exception:
            return os.path.join(os.getcwd(), "models")

    def _collect_weight_files(self, subdir: Optional[str] = None, fallback_to_root: bool = True) -> List[str]:
        root = self._models_root_dir()
        if not os.path.isdir(root):
            return []
        search_dirs: List[str] = []
        if subdir:
            sub_path = os.path.join(root, subdir)
            if os.path.isdir(sub_path):
                search_dirs.append(sub_path)
            elif fallback_to_root:
                search_dirs.append(root)
        else:
            search_dirs.append(root)
        patterns = (".pt", ".pth", ".onnx", ".engine")
        collected: Dict[str, None] = {}
        for base in search_dirs:
            if not os.path.isdir(base):
                continue
            for cur_root, _, files in os.walk(base):
                for fname in files:
                    if not fname.lower().endswith(patterns):
                        continue
                    abs_path = os.path.join(cur_root, fname)
                    rel = os.path.relpath(abs_path, root).replace("\\", "/")
                    if rel.startswith(".."):
                        continue
                    normalized = f"models/{rel}" if rel else "models"
                    collected.setdefault(normalized, None)
        return sorted(collected.keys(), key=lambda x: x.lower())

    def _get_model_weights(self) -> List[str]:
        return self._collect_weight_files()

    def _get_detection_weights(self) -> List[str]:
        return self._collect_weight_files("detection")

    def _get_segmentation_weights(self) -> List[str]:
        return self._collect_weight_files("seg")

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

        # Preserve edits from dynamic forms
        self._save_current_pre_form()
        self._save_model_form()
        self._save_class_feature_form()
        self._save_class_classifier_form()

        model = self.combo_model.currentText()
        evaluator = self.combo_eval.currentText()

        pipeline_steps: List[Dict[str, Any]] = []
        for name in pre_list:
            params = dict(self.pre_params.get(name) or getattr(PREPROC_REGISTRY[name], 'DEFAULT_CONFIG', {}))
            pipeline_steps.append({'type': 'preproc', 'name': name, 'params': params})

        model_defaults = getattr(MODEL_REGISTRY[model], 'DEFAULT_CONFIG', {})
        model_overrides = self.model_params.get(model) or {}
        model_params = {**model_defaults, **model_overrides}
        if model in ('CSRT', 'OpticalFlowLK'):
            model_params.pop('init_box', None)
        pipeline_steps.append({'type': 'model', 'name': model, 'params': model_params})

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

        loso_enabled = bool(self.chk_loso.isChecked()) or self.split_method.currentText().strip().lower() == 'loso'
        method_text = 'loso' if loso_enabled else self.split_method.currentText()

        cfg: Dict[str, Any] = {
            'seed': 42,
            'dataset': {
                'root': ds_root,
                'split': {
                    'method': method_text,
                    'ratios': [self.split_r_train.value(), self.split_r_test.value()],
                    'k_fold': int(self.kfold.value()),
                    'loso': loso_enabled,
                },
            },
            'experiments': [
                {
                    'name': exp_name_final,
                    'pipeline': pipeline_steps,
                    'preproc_scheme': str(self.combo_preproc_scheme.currentData() or 'A').strip().upper() or 'A',
                }
            ],
            'evaluation': {'evaluator': evaluator},
        }
        viz_cfg = cfg.setdefault('evaluation', {}).setdefault('visualize', {})
        viz_cfg.update({
            'enabled': True,
            'samples': 10,
            'strategy': 'even_spread',
            'include_detection': True,
            'include_segmentation': True,
            'ensure_first_last': True,
        })

        # Classification configuration (always emitted for discoverability)
        class_enabled = bool(self.chk_class_enabled.isChecked())
        feature_name = self.combo_class_feature.currentText() if self.combo_class_feature.count() else ''
        classifier_name = self.combo_class_classifier.currentText() if self.combo_class_classifier.count() else ''
        feature_params = dict(self.class_feature_params.get(feature_name, {})) if feature_name else {}
        classifier_params = dict(self.class_classifier_params.get(classifier_name, {})) if classifier_name else {}
        fallback_feature = feature_name or (self.combo_class_feature.itemText(0) if self.combo_class_feature.count() else 'motion_only')
        fallback_classifier = classifier_name or (self.combo_class_classifier.itemText(0) if self.combo_class_classifier.count() else 'random_forest')
        class_cfg: Dict[str, Any] = {
            'enabled': class_enabled,
            'feature_extractor': {
                'name': fallback_feature,
                'params': feature_params,
            },
            'classifier': {
                'name': fallback_classifier,
                'params': classifier_params,
            },
        }
        label_path = self.edit_class_label.text().strip()
        if label_path:
            class_cfg['label_file'] = label_path

        cfg['classification'] = class_cfg

        def _parse_float(value: str, default: float) -> float:
            try:
                return float(value.strip())
            except Exception:
                return default

        seg_method_key_raw = self.combo_seg_model.currentData()
        if seg_method_key_raw is None:
            seg_method_key_raw = self.combo_seg_model.currentText() or 'unetpp'
        seg_method_key = str(seg_method_key_raw).strip() or 'unetpp'
        seg_method_key_lower = seg_method_key.lower()
        seg_method_params: Dict[str, Any] = {}
        if seg_method_key_lower == 'nnunet':
            plans = self.edit_nnunet_plans.text().strip()
            config_name = self.edit_nnunet_config.text().strip()
            arch_path = self.edit_nnunet_architecture.text().strip()
            if plans:
                seg_method_params['plans_path'] = plans
            if config_name:
                seg_method_params['configuration'] = config_name
            if arch_path:
                seg_method_params['architecture_path'] = arch_path
            seg_method_params['return_highres_only'] = bool(self.chk_nnunet_highres.isChecked())
        elif seg_method_key_lower != 'auto_mask':
            seg_method_params = {
                'encoder_name': self.edit_seg_encoder.text().strip() or 'resnet34',
                'encoder_weights': self.edit_seg_weights.text().strip() or 'imagenet',
            }
        method_cfg = {
            'name': seg_method_key,
            'params': seg_method_params,
        }
        seg_cfg = {
            'method': method_cfg,
            'padding_min': float(self.seg_padding_min.value()),
            'padding_max': float(self.seg_padding_max.value()),
            'padding_inference': float(self.seg_padding_inf.value()),
            'jitter': float(self.seg_jitter.value()),
            'epochs': int(self.seg_epochs.value()),
            'batch_size': int(self.seg_batch.value()),
            'num_workers': int(self.seg_num_workers.value()),
            'lr': _parse_float(self.seg_lr.text(), 1e-3),
            'weight_decay': _parse_float(self.seg_weight_decay.text(), 1e-5),
            'threshold': float(self.seg_threshold.value()),
            'val_ratio': float(self.seg_val_ratio.value()),
            'seed': int(self.seg_seed.value()),
            'redundancy': int(self.seg_redundancy.value()),
            'dice_weight': float(self.seg_dice_weight.value()),
            'bce_weight': float(self.seg_bce_weight.value()),
            'device': self.edit_seg_device.text().strip() or 'auto',
        }
        seg_cfg['model'] = {
            'name': method_cfg['name'],
            'params': dict(method_cfg['params']),
        }
        seg_cfg['train'] = bool(self.chk_seg_train.isChecked())
        inference_value = ""
        if hasattr(self, 'combo_seg_checkpoint') and hasattr(self.combo_seg_checkpoint, 'current_value'):
            try:
                inference_value = self.combo_seg_checkpoint.current_value().strip()
            except Exception:
                inference_value = ""
        if inference_value:
            seg_cfg['inference_checkpoint'] = inference_value
        if not seg_cfg['train'] and not inference_value:
            seg_cfg['auto_pretrained'] = True
        cfg['segmentation'] = seg_cfg

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
        # early guard: raw editor might not exist yet during __init__
        if not hasattr(self, 'txt_cfg') or self.txt_cfg is None:
            return
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

    def _on_loso_toggled(self, _state: int):
        if self.chk_loso.isChecked():
            idx = self.split_method.findText('loso')
            if idx < 0:
                self.split_method.addItem('loso')
                idx = self.split_method.findText('loso')
            if idx >= 0:
                self.split_method.blockSignals(True)
                self.split_method.setCurrentIndex(idx)
                self.split_method.blockSignals(False)
        self._on_builder_changed()

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
        if not hasattr(self, 'txt_cfg') or self.txt_cfg is None:
            return
        if err:
            self.txt_cfg.setStyleSheet("QTextEdit { font-family:Consolas; font-size:12px; border:2px solid #d73a49; }")
        else:
            self.txt_cfg.setStyleSheet("QTextEdit { font-family:Consolas; font-size:12px; }")

    def _set_raw_text_programmatically(self, text: str):
        # guard in case called before widget created
        if not hasattr(self, 'txt_cfg') or self.txt_cfg is None:
            return
        self._updating_raw_programmatically = True
        self.txt_cfg.blockSignals(True)
        self.txt_cfg.setPlainText(text)
        self.txt_cfg.blockSignals(False)
        self._updating_raw_programmatically = False

    # ---------------- Apply parsed cfg -> Builder -----------------
    def _apply_cfg_to_builder(self, cfg: Dict[str, Any]):
        prev_applying = self._applying_cfg
        self._applying_cfg = True
        try:
            self._apply_cfg_to_builder_inner(cfg)
        finally:
            self._applying_cfg = prev_applying

    def _apply_cfg_to_builder_inner(self, cfg: Dict[str, Any]):
        ds = cfg.get('dataset', {}) or {}
        root = ds.get('root');
        if isinstance(root, str): self.edit_root.setText(root)
        split = ds.get('split', {}) or {}
        loso_flag = bool(split.get('loso', False))
        method_text = str(split.get('method', self.split_method.currentText() or 'video_level')).strip() or 'video_level'
        if loso_flag:
            method_text = 'loso'
        current_methods = [self.split_method.itemText(i) for i in range(self.split_method.count())]
        if method_text not in current_methods:
            self.split_method.addItem(method_text)
        idx_method = self.split_method.findText(method_text)
        if idx_method >= 0:
            self.split_method.blockSignals(True)
            self.split_method.setCurrentIndex(idx_method)
            self.split_method.blockSignals(False)
        self.chk_loso.blockSignals(True)
        self.chk_loso.setChecked(loso_flag or method_text.lower() == 'loso')
        self.chk_loso.blockSignals(False)
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
            # Preproc scheme
            try:
                scheme = str(first.get('preproc_scheme') or first.get('preprocessing_scheme') or first.get('preproc_mode') or 'A').strip().upper()
                if scheme not in {'A','B','C'}:
                    scheme = 'A'
                idx_scheme = self.combo_preproc_scheme.findData(scheme)
                if idx_scheme < 0:
                    idx_scheme = 0
                self.combo_preproc_scheme.blockSignals(True)
                self.combo_preproc_scheme.setCurrentIndex(idx_scheme)
                self.combo_preproc_scheme.blockSignals(False)
            except Exception:
                pass
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
        # Classification
        class_cfg = cfg.get('classification') or {}
        if not isinstance(class_cfg, dict):
            class_cfg = {}

        self.chk_class_enabled.blockSignals(True)
        self.chk_class_enabled.setChecked(bool(class_cfg.get('enabled', self.chk_class_enabled.isChecked())))
        self.chk_class_enabled.blockSignals(False)

        label_file = class_cfg.get('label_file')
        self.edit_class_label.blockSignals(True)
        if isinstance(label_file, str):
            self.edit_class_label.setText(label_file)
        else:
            self.edit_class_label.clear()
        self.edit_class_label.blockSignals(False)

        feature_cfg = class_cfg.get('feature_extractor') or {}
        feat_name = feature_cfg.get('name')
        feat_params = feature_cfg.get('params') or {}
        if feat_name and self.combo_class_feature.findText(feat_name) < 0:
            self.combo_class_feature.addItem(feat_name)
        if self.combo_class_feature.count():
            if feat_name:
                idx_feat = self.combo_class_feature.findText(feat_name)
                if idx_feat < 0:
                    idx_feat = 0
            else:
                idx_feat = self.combo_class_feature.currentIndex()
            self.combo_class_feature.blockSignals(True)
            if idx_feat >= 0:
                self.combo_class_feature.setCurrentIndex(idx_feat)
            self.combo_class_feature.blockSignals(False)
            active_feat = self.combo_class_feature.currentText()
            if feat_name:
                self.class_feature_params[feat_name] = dict(feat_params)
                self._on_class_feature_changed(feat_name, from_builder=True)
            else:
                self._on_class_feature_changed(active_feat, from_builder=True)

        classifier_cfg = class_cfg.get('classifier') or {}
        clf_name = classifier_cfg.get('name')
        clf_params = classifier_cfg.get('params') or {}
        if clf_name and self.combo_class_classifier.findText(clf_name) < 0:
            self.combo_class_classifier.addItem(clf_name)
        if self.combo_class_classifier.count():
            if clf_name:
                idx_clf = self.combo_class_classifier.findText(clf_name)
                if idx_clf < 0:
                    idx_clf = 0
            else:
                idx_clf = self.combo_class_classifier.currentIndex()
            self.combo_class_classifier.blockSignals(True)
            if idx_clf >= 0:
                self.combo_class_classifier.setCurrentIndex(idx_clf)
            self.combo_class_classifier.blockSignals(False)
            active_clf = self.combo_class_classifier.currentText()
            if clf_name:
                self.class_classifier_params[clf_name] = dict(clf_params)
                self._on_class_classifier_changed(clf_name, from_builder=True)
            else:
                self._on_class_classifier_changed(active_clf, from_builder=True)
        seg_cfg = cfg.get('segmentation') or {}
        method_cfg = seg_cfg.get('method') or seg_cfg.get('model') or {}
        method_params = method_cfg.get('params') or {}
        auto_pretrained = bool(seg_cfg.get('auto_pretrained'))

        train_flag = seg_cfg.get('train', seg_cfg.get('enabled', True))
        if hasattr(self, 'chk_seg_train'):
            self.chk_seg_train.blockSignals(True)
            self.chk_seg_train.setChecked(bool(train_flag))
            self.chk_seg_train.blockSignals(False)
            self._update_segmentation_train_state()
        if auto_pretrained and not train_flag:
            self.log('[Segmentation] 未選擇模型權重且停用訓練，執行時會自動載入官方預訓練權重。')

        inference_path = seg_cfg.get('inference_checkpoint') or seg_cfg.get('checkpoint') or ""
        if hasattr(self, 'combo_seg_checkpoint'):
            self.combo_seg_checkpoint.blockSignals(True)
            self.combo_seg_checkpoint.set_current_value(inference_path)
            self.combo_seg_checkpoint.blockSignals(False)

        if self.combo_seg_model.count():
            key = method_cfg.get('name')
            idx_model = -1
            if key is not None:
                idx_model = self.combo_seg_model.findData(key)
                if idx_model < 0:
                    idx_model = self.combo_seg_model.findText(str(key))
            if idx_model < 0:
                idx_model = 0
            self.combo_seg_model.blockSignals(True)
            if 0 <= idx_model < self.combo_seg_model.count():
                self.combo_seg_model.setCurrentIndex(idx_model)
            self.combo_seg_model.blockSignals(False)
            self._on_seg_method_changed(idx_model)

        def _set_line(widget: QLineEdit, value: Any, fallback: str):
            widget.blockSignals(True)
            text = str(value).strip() if isinstance(value, str) and value.strip() else (str(value) if value not in (None, "") else fallback)
            widget.setText(text)
            widget.blockSignals(False)

        _set_line(self.edit_seg_encoder, method_params.get('encoder_name'), self.edit_seg_encoder.text())
        _set_line(self.edit_seg_weights, method_params.get('encoder_weights'), self.edit_seg_weights.text())
        if hasattr(self, 'edit_nnunet_plans'):
            _set_line(self.edit_nnunet_plans, method_params.get('plans_path') or method_params.get('plans'), self.edit_nnunet_plans.text())
        if hasattr(self, 'edit_nnunet_config'):
            _set_line(self.edit_nnunet_config, method_params.get('configuration'), self.edit_nnunet_config.text())
        if hasattr(self, 'edit_nnunet_architecture'):
            _set_line(
                self.edit_nnunet_architecture,
                method_params.get('architecture_path') or method_params.get('architecture_file'),
                self.edit_nnunet_architecture.text(),
            )
        if hasattr(self, 'chk_nnunet_highres'):
            self.chk_nnunet_highres.blockSignals(True)
            self.chk_nnunet_highres.setChecked(bool(method_params.get('return_highres_only', self.chk_nnunet_highres.isChecked())))
            self.chk_nnunet_highres.blockSignals(False)

        def _set_dspin(widget: NoWheelDoubleSpinBox, key: str, source: Dict[str, Any] = seg_cfg):
            widget.blockSignals(True)
            current = widget.value()
            try:
                new_val = float(source.get(key, current))
                widget.setValue(new_val)
            except Exception:
                widget.setValue(current)
            widget.blockSignals(False)

        def _set_spin(widget: NoWheelSpinBox, key: str, source: Dict[str, Any] = seg_cfg):
            widget.blockSignals(True)
            current = widget.value()
            try:
                new_val = int(source.get(key, current))
                widget.setValue(new_val)
            except Exception:
                widget.setValue(current)
            widget.blockSignals(False)

        _set_dspin(self.seg_padding_min, 'padding_min')
        _set_dspin(self.seg_padding_max, 'padding_max')
        _set_dspin(self.seg_padding_inf, 'padding_inference')
        _set_dspin(self.seg_jitter, 'jitter')
        _set_spin(self.seg_epochs, 'epochs')
        _set_spin(self.seg_batch, 'batch_size')
        _set_spin(self.seg_num_workers, 'num_workers')
        _set_dspin(self.seg_threshold, 'threshold')
        _set_dspin(self.seg_val_ratio, 'val_ratio')
        _set_spin(self.seg_seed, 'seed')
        _set_spin(self.seg_redundancy, 'redundancy')
        _set_dspin(self.seg_dice_weight, 'dice_weight')
        _set_dspin(self.seg_bce_weight, 'bce_weight')

        def _set_line_float(widget: QLineEdit, key: str, default: str):
            widget.blockSignals(True)
            value = seg_cfg.get(key)
            if value is None:
                value = default
            widget.setText(str(value))
            widget.blockSignals(False)

        _set_line_float(self.seg_lr, 'lr', self.seg_lr.text())
        _set_line_float(self.seg_weight_decay, 'weight_decay', self.seg_weight_decay.text())
        _set_line(self.edit_seg_device, seg_cfg.get('device'), self.edit_seg_device.text())

        self._update_classification_enabled_state()
        self._update_segmentation_train_state()
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
            combo = NoWheelComboBox()
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
            widget = LowConfidenceReinitEditor(defaults, weights_provider=self._get_detection_weights, parent=self)

            def _get():
                return widget.get_value()

            def _set(v):
                widget.set_value(v)

            register(widget, 'valueChanged')
            return widget, _get, _set
        if scope == 'model' and key in ('weights', 'init_detector_weights'):
            combo_weights = ModelWeightsComboBox(self._get_detection_weights, parent=self)
            combo_weights.refresh_items()
            combo_weights.currentIndexChanged.connect(lambda *_: self._on_param_widget_changed(scope))
            combo_weights.editTextChanged.connect(lambda *_: self._on_param_widget_changed(scope))

            def _get_weights_value() -> str:
                return combo_weights.current_value()

            def _set_weights_value(v: Any):
                combo_weights.set_current_value(v)

            return combo_weights, _get_weights_value, _set_weights_value
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

    def _on_classification_toggled(self):
        self._update_classification_enabled_state()
        self._on_builder_changed()

    def _on_segmentation_toggled(self):
        self._update_segmentation_enabled_state()
        self._on_builder_changed()

    def _browse_class_label_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "選擇 ann.txt",
            self.edit_class_label.text() or os.getcwd(),
            "Text (*.txt);;All Files (*)",
        )
        if path:
            self.edit_class_label.setText(path)
            self._on_builder_changed()

    def _on_class_feature_index_changed(self, idx: int):
        name = self.combo_class_feature.itemText(idx) if idx >= 0 else None
        self._on_class_feature_changed(name)

    def _on_class_feature_changed(self, name: Optional[str], from_builder: bool = False):
        self._save_class_feature_form()
        self._clear_form(self.class_feature_form_layout)
        self._class_feature_bindings = []
        self._current_class_feature = name
        if not name:
            self.class_feature_params.setdefault('', {})
            if not from_builder:
                self._on_builder_changed()
            return
        defaults: Dict[str, Any] = {}
        feat_cls = FEATURE_EXTRACTOR_REGISTRY.get(name)
        if feat_cls is not None:
            base = getattr(feat_cls, 'DEFAULT_CONFIG', {})
            if isinstance(base, dict):
                defaults.update(base)
        stored = self.class_feature_params.get(name, {})
        params = dict(defaults)
        params.update(stored)
        self.class_feature_params[name] = dict(stored)
        for key, value in params.items():
            widget, getter, setter = self._make_editor(key, value, scope='classification_feature')
            setter(value)
            self.class_feature_form_layout.addRow(QLabel(key), widget)
            self._class_feature_bindings.append((key, getter, setter))
        if not params:
            hint = QLabel("此特徵擷取器無額外參數。")
            hint.setStyleSheet("color:#6a737d; font-size:11px;")
            self.class_feature_form_layout.addRow(hint)
        if not from_builder:
            self._on_builder_changed()

    def _save_class_feature_form(self):
        if not self._current_class_feature:
            return
        if self._class_feature_bindings:
            self.class_feature_params[self._current_class_feature] = {
                key: getter() for key, getter, _ in self._class_feature_bindings
            }
        else:
            self.class_feature_params.setdefault(self._current_class_feature, {})

    def _on_class_classifier_index_changed(self, idx: int):
        name = self.combo_class_classifier.itemText(idx) if idx >= 0 else None
        self._on_class_classifier_changed(name)

    def _on_class_classifier_changed(self, name: Optional[str], from_builder: bool = False):
        self._save_class_classifier_form()
        self._clear_form(self.class_classifier_form_layout)
        self._class_classifier_bindings = []
        self._current_class_classifier = name
        if not name:
            self.class_classifier_params.setdefault('', {})
            if not from_builder:
                self._on_builder_changed()
            return
        defaults: Dict[str, Any] = {}
        clf_cls = CLASSIFIER_REGISTRY.get(name)
        if clf_cls is not None:
            base = getattr(clf_cls, 'DEFAULT_CONFIG', {})
            if isinstance(base, dict):
                defaults.update(base)
        stored = self.class_classifier_params.get(name, {})
        params = dict(defaults)
        params.update(stored)
        self.class_classifier_params[name] = dict(stored)
        for key, value in params.items():
            widget, getter, setter = self._make_editor(key, value, scope='classification_classifier')
            setter(value)
            self.class_classifier_form_layout.addRow(QLabel(key), widget)
            self._class_classifier_bindings.append((key, getter, setter))
        if not params:
            hint = QLabel("此分類器無額外參數。")
            hint.setStyleSheet("color:#6a737d; font-size:11px;")
            self.class_classifier_form_layout.addRow(hint)
        if not from_builder:
            self._on_builder_changed()

    def _save_class_classifier_form(self):
        if not self._current_class_classifier:
            return
        if self._class_classifier_bindings:
            self.class_classifier_params[self._current_class_classifier] = {
                key: getter() for key, getter, _ in self._class_classifier_bindings
            }
        else:
            self.class_classifier_params.setdefault(self._current_class_classifier, {})

    def _update_classification_enabled_state(self):
        enabled = self.chk_class_enabled.isChecked()
        widgets = [
            getattr(self, 'edit_class_label', None),
            getattr(self, 'btn_class_label', None),
            getattr(self, 'combo_class_feature', None),
            getattr(self, 'combo_class_classifier', None),
            getattr(self, 'gb_class_feat', None),
            getattr(self, 'gb_class_clf', None),
            getattr(self, 'chk_seg_enabled', None),
        ]
        for widget in widgets:
            if widget is not None:
                widget.setEnabled(enabled)
        self._update_segmentation_enabled_state()
        self._update_segmentation_train_state()

    def _get_current_seg_method_key(self) -> str:
        if not hasattr(self, 'combo_seg_model') or self.combo_seg_model is None:
            return ""
        idx = self.combo_seg_model.currentIndex()
        if idx < 0:
            return ""
        data = self.combo_seg_model.itemData(idx)
        if isinstance(data, str) and data.strip():
            return data.strip().lower()
        text = self.combo_seg_model.itemText(idx)
        return text.strip().lower()

    def _on_seg_method_changed(self, idx: int):
        _ = idx  # unused
        method_key = self._get_current_seg_method_key()
        auto_selected = method_key == "auto_mask"
        if hasattr(self, 'chk_seg_train') and self.chk_seg_train is not None:
            self.chk_seg_train.blockSignals(True)
            if auto_selected:
                self.chk_seg_train.setChecked(False)
            self.chk_seg_train.setEnabled(not auto_selected)
            self.chk_seg_train.blockSignals(False)
        self._update_segmentation_train_state()
        nnunet_selected = method_key == "nnunet"
        for widget in getattr(self, '_nnunet_widgets', []):
            if widget is not None:
                widget.setVisible(nnunet_selected)
        if not getattr(self, '_syncing', False) and not getattr(self, '_applying_cfg', False):
            self._on_builder_changed()

    def _update_segmentation_enabled_state(self):
        enabled = bool(self.chk_class_enabled.isChecked())
        if not enabled:
            if hasattr(self, 'chk_seg_enabled'):
                self.chk_seg_enabled.setEnabled(False)
        else:
            if hasattr(self, 'chk_seg_enabled'):
                self.chk_seg_enabled.setEnabled(True)
        seg_controls = getattr(self, 'gb_class_seg', None)
        seg_active = enabled and getattr(self, 'chk_seg_enabled', None) is not None and self.chk_seg_enabled.isChecked()
        if seg_controls is not None:
            seg_controls.setEnabled(seg_active)

    def _update_segmentation_train_state(self):
        auto_selected = self._get_current_seg_method_key() == "auto_mask"
        train_enabled = True
        if hasattr(self, 'chk_seg_train'):
            train_enabled = bool(self.chk_seg_train.isChecked())
            if auto_selected:
                if self.chk_seg_train.isChecked():
                    self.chk_seg_train.blockSignals(True)
                    self.chk_seg_train.setChecked(False)
                    self.chk_seg_train.blockSignals(False)
                self.chk_seg_train.setEnabled(False)
            else:
                self.chk_seg_train.setEnabled(True)
        if auto_selected:
            train_enabled = False
        if hasattr(self, 'edit_seg_encoder'):
            self.edit_seg_encoder.setEnabled(not auto_selected)
        if hasattr(self, 'edit_seg_weights'):
            self.edit_seg_weights.setEnabled(not auto_selected)
        for widget in getattr(self, '_seg_train_widgets', []):
            if widget is not None:
                widget.setEnabled(train_enabled)
        if hasattr(self, 'combo_seg_checkpoint'):
            self.combo_seg_checkpoint.setEnabled(not train_enabled)

    def _on_seg_train_toggled(self, *_args):
        self._update_segmentation_train_state()
        self._on_builder_changed()

    def _on_param_widget_changed(self, scope: str):
        if scope == 'preproc':
            self._save_current_pre_form()
        elif scope == 'model':
            self._save_model_form()
        elif scope == 'classification_feature':
            self._save_class_feature_form()
        elif scope == 'classification_classifier':
            self._save_class_classifier_form()
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

    def _start_run_thread(self, cfg: dict, detector_cache: Optional[dict] = None):
        class _Worker(QObject):
            finished = Signal()
            error = Signal(str)
            progress = Signal(str, int, int, dict)
            log = Signal(str)

            def __init__(self, cfg, detector_cache):
                super().__init__()
                self.cfg = cfg
                self.detector_cache = detector_cache

            def run(self):
                from traceback import format_exc
                try:
                    def _logger(msg: str):
                        self.log.emit(msg)
                    runner = PipelineRunner(
                        self.cfg,
                        logger=_logger,
                        progress_cb=lambda stage, cur, tot, extra: self.progress.emit(stage, cur, tot, extra),
                        detector_reuse_cache=self.detector_cache,
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
        self._run_worker = _Worker(cfg, detector_cache)
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
