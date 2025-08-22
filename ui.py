from __future__ import annotations
import json
import os
from typing import Optional

from PySide6.QtCore import Qt, QRegularExpression
from PySide6.QtGui import QRegularExpressionValidator
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QTextEdit, QLineEdit, QMessageBox,
    QComboBox, QSpinBox, QDoubleSpinBox, QFormLayout, QGroupBox,
    QListWidget, QListWidgetItem, QAbstractItemView, QCheckBox,
    QScrollArea, QSplitter
)

from tracking.orchestrator.runner import PipelineRunner
from tracking.core.registry import PREPROC_REGISTRY, MODEL_REGISTRY, EVAL_REGISTRY
# import built-in plugins to populate registries for dropdowns
from tracking.preproc import clahe  # noqa: F401
from tracking.models import template_matching  # noqa: F401
from tracking.models import csrt  # noqa: F401
from tracking.models import optical_flow_lk  # noqa: F401
from tracking.models import faster_rcnn  # noqa: F401
from tracking.models import yolov11  # noqa: F401
from tracking.models import fast_speckle  # noqa: F401
from tracking.eval import evaluator  # noqa: F401


class SimpleRunnerUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tracking Pipeline UI (Lite)")
        self.resize(900, 700)
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        # Top container that will be placed inside a scroll area
        top_container = QWidget()
        top_layout = QVBoxLayout(top_container)

        # Config path row
        row = QHBoxLayout()
        row.addWidget(QLabel("Config (YAML/JSON):"))
        self.edit_cfg = QLineEdit()
        btn_browse = QPushButton("瀏覽…")
        btn_browse.clicked.connect(self.browse_cfg)
        row.addWidget(self.edit_cfg, 1)
        row.addWidget(btn_browse)
        top_layout.addLayout(row)

        # Dataset root row
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Dataset Root:"))
        self.edit_root = QLineEdit()
        btn_root = QPushButton("選資料夾…")
        btn_root.clicked.connect(self.browse_root)
        row2.addWidget(self.edit_root, 1)
        row2.addWidget(btn_root)
        top_layout.addLayout(row2)

        # Guided builder group
        gb = QGroupBox("Pipeline Builder")
        gbl = QVBoxLayout(gb)
        # internal state for params editors (redefined with syncing flag)
        self.pre_params = {}
        self.model_params = {}
        self._current_pre_name = None
        self._pre_bindings = []
        self._model_bindings = []
        self._syncing = False  # prevent save/overwrite during raw sync

        ds_form = QFormLayout()
        self.split_method = QComboBox(); self.split_method.addItems(["video_level"])  # reserve
        self.split_r_train = QDoubleSpinBox(); self.split_r_train.setRange(0,1); self.split_r_train.setSingleStep(0.05); self.split_r_train.setValue(0.8)
        self.split_r_test = QDoubleSpinBox(); self.split_r_test.setRange(0,1); self.split_r_test.setSingleStep(0.05); self.split_r_test.setValue(0.2)
        self.kfold = QSpinBox(); self.kfold.setRange(1, 20); self.kfold.setValue(1)
        gbl.addLayout(ds_form)
        ds_form.addRow("Split Method", self.split_method)
        ds_form.addRow("Ratios (train/test)", self._hbox([self.split_r_train, self.split_r_test]))
        ds_form.addRow("K-Fold (1=off)", self.kfold)

        # Preprocessing selector with add/remove/reorder
        pre_row = QVBoxLayout()
        pre_row_header = QHBoxLayout(); pre_row_header.addWidget(QLabel("Preprocessing:"))
        pre_row.addLayout(pre_row_header)
        pre_mid = QHBoxLayout()
        # available list
        self.list_pre_avail = QListWidget(); self.list_pre_avail.setSelectionMode(QAbstractItemView.SingleSelection)
        for name in sorted(PREPROC_REGISTRY.keys()):
            self.list_pre_avail.addItem(QListWidgetItem(name))
        pre_mid.addWidget(self._with_title("可用模組", self.list_pre_avail), 1)
        # buttons
        btn_col = QVBoxLayout()
        btn_add = QPushButton("→ 新增"); btn_add.clicked.connect(self._pre_add)
        btn_remove = QPushButton("← 移除"); btn_remove.clicked.connect(self._pre_remove)
        btn_up = QPushButton("上移"); btn_up.clicked.connect(self._pre_up)
        btn_down = QPushButton("下移"); btn_down.clicked.connect(self._pre_down)
        for b in (btn_add, btn_remove, btn_up, btn_down): btn_col.addWidget(b)
        btn_col.addStretch(1)
        pre_mid.addLayout(btn_col)
        # selected list (ordered)
        self.list_pre_sel = QListWidget(); self.list_pre_sel.setSelectionMode(QAbstractItemView.SingleSelection)
        pre_mid.addWidget(self._with_title("已選順序", self.list_pre_sel), 1)
        pre_row.addLayout(pre_mid)
        gbl.addLayout(pre_row)

        # Model row
        mdl_row = QHBoxLayout()
        mdl_row.addWidget(QLabel("Model:"))
        self.combo_model = QComboBox()
        for name in sorted(MODEL_REGISTRY.keys()):
            self.combo_model.addItem(name)
        mdl_row.addWidget(self.combo_model, 1)
        gbl.addLayout(mdl_row)

        # Preproc params editor
        pre_params_box = QGroupBox("Preproc 參數（編輯右側已選中的項目）")
        pre_params_l = QVBoxLayout(pre_params_box)
        self.pre_params_vlayout = pre_params_l
        self.pre_form_container = QWidget(); self.pre_form_layout = QFormLayout(self.pre_form_container)
        pre_params_l.addWidget(self.pre_form_container)
        btn_pre_save = QPushButton("儲存 Preproc 參數"); btn_pre_save.clicked.connect(self._save_current_pre_form)
        pre_params_l.addWidget(btn_pre_save)
        gbl.addWidget(pre_params_box)

        # Model params editor
        model_params_box = QGroupBox("Model 參數")
        model_params_l = QVBoxLayout(model_params_box)
        self.model_params_vlayout = model_params_l
        self.model_form_container = QWidget(); self.model_form_layout = QFormLayout(self.model_form_container)
        model_params_l.addWidget(self.model_form_container)
        btn_model_save = QPushButton("儲存 Model 參數"); btn_model_save.clicked.connect(self._save_model_form)
        model_params_l.addWidget(btn_model_save)
        gbl.addWidget(model_params_box)

        # Evaluator row
        ev_row = QHBoxLayout()
        ev_row.addWidget(QLabel("Evaluator:"))
        self.combo_eval = QComboBox()
        for name in sorted(EVAL_REGISTRY.keys()):
            self.combo_eval.addItem(name)
        ev_row.addWidget(self.combo_eval, 1)
        gbl.addLayout(ev_row)

        # Evaluation options: visualization
        viz_row = QHBoxLayout()
        self.chk_viz = QCheckBox("輸出可視化")
        self.spn_viz_samples = QSpinBox(); self.spn_viz_samples.setRange(1, 10000); self.spn_viz_samples.setValue(10)
        viz_row.addWidget(self.chk_viz)
        viz_row.addWidget(QLabel("可視化張數(samples):"))
        viz_row.addWidget(self.spn_viz_samples)
        viz_row.addStretch(1)
        gbl.addLayout(viz_row)

        gbl.addWidget(QLabel("Raw Config (optional):"))
        self.txt_cfg = QTextEdit()
        gbl.addWidget(self.txt_cfg, 1)
        top_layout.addWidget(gb)

        # Controls row
        ctr = QHBoxLayout()
        btn_load = QPushButton("載入檔案"); btn_load.clicked.connect(self.load_file)
        btn_save = QPushButton("存檔"); btn_save.clicked.connect(self.save_file)
        btn_build = QPushButton("由上方選項產生 YAML"); btn_build.clicked.connect(self.build_yaml_from_controls)
        btn_sync_from_raw = QPushButton("同步 Raw 到 Builder"); btn_sync_from_raw.setToolTip("將目前 Raw Config 解析並填入上方控制項"); btn_sync_from_raw.clicked.connect(self.sync_controls_from_raw)
        btn_run = QPushButton("執行 Pipeline"); btn_run.clicked.connect(self.run_pipeline)
        for b in (btn_load, btn_save, btn_build, btn_sync_from_raw, btn_run):
            ctr.addWidget(b)
        top_layout.addLayout(ctr)

        # Logs panel (bottom of splitter)
        logs_widget = QWidget()
        logs_layout = QVBoxLayout(logs_widget)
        logs_layout.setContentsMargins(0, 0, 0, 0)
        logs_layout.addWidget(QLabel("Logs:"))
        self.txt_log = QTextEdit(); self.txt_log.setReadOnly(True)
        logs_layout.addWidget(self.txt_log, 1)

        # Wire events for params editors
        self.list_pre_sel.currentRowChanged.connect(self._on_pre_selected_changed)
        self._current_model_name = self.combo_model.currentText()
        self.combo_model.currentIndexChanged.connect(self._on_model_index_changed)
        self._on_model_changed(self._current_model_name)

        # Scrollable top area + resizable splitter
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(top_container)
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(scroll)
        splitter.addWidget(logs_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root_layout.addWidget(splitter)
    def log(self, msg: str):
        self.txt_log.append(msg)
        # 同步輸出到終端機，便於即時觀察
        try:
            print(msg, flush=True)
        except Exception:
            pass

    def _hbox(self, widgets):
        w = QWidget(); l = QHBoxLayout(w); l.setContentsMargins(0,0,0,0)
        for x in widgets: l.addWidget(x)
        return w

    def browse_cfg(self):
        p, _ = QFileDialog.getOpenFileName(self, "選擇設定檔", os.getcwd(), "YAML/JSON (*.yaml *.yml *.json)")
        if p:
            self.edit_cfg.setText(p)
            self.load_file()

    def browse_root(self):
        d = QFileDialog.getExistingDirectory(self, "選擇資料集根目錄", os.getcwd())
        if d:
            self.edit_root.setText(d)

    def load_file(self):
        path = self.edit_cfg.text().strip()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "錯誤", "請先選擇有效的設定檔")
            return
        with open(path, "r", encoding="utf-8") as f:
            self.txt_cfg.setPlainText(f.read())
        # 重要：自動將剛載入的 Raw Config 解析並同步到上方控制項，避免未按「同步」時用預設值覆蓋自訂參數
        try:
            self.sync_controls_from_raw()
            self.log("[AutoSync] 已自動同步載入檔案的參數到 Builder 控制項。")
        except Exception as e:
            self.log(f"[AutoSync][Warn] 同步失敗: {e}")

    def save_file(self):
        path = self.edit_cfg.text().strip()
        if not path:
            p, _ = QFileDialog.getSaveFileName(self, "儲存設定檔", os.getcwd(), "YAML/JSON (*.yaml *.yml *.json)")
            if not p:
                return
            path = p
            self.edit_cfg.setText(path)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.txt_cfg.toPlainText())
        QMessageBox.information(self, "已儲存", f"寫入 {path}")

    def parse_config(self) -> Optional[dict]:
        import traceback as _tb
        txt = self.txt_cfg.toPlainText()
        if not txt.strip():
            return None
        path = (self.edit_cfg.text() or "").lower()
        # Prefer YAML if副檔名是 yml/yaml；否則嘗試 JSON，失敗再回退 YAML
        if path.endswith((".yaml", ".yml")):
            try:
                import yaml  # type: ignore
                return yaml.safe_load(txt)
            except Exception as e:
                err = f"YAML 解析失敗: {e}\n{_tb.format_exc()}"
                self.log(err)
                QMessageBox.critical(self, "YAML 錯誤", str(e))
                return None
        else:
            # Try JSON first
            try:
                return json.loads(txt)
            except Exception as e_json:
                # Fallback to YAML
                try:
                    import yaml  # type: ignore
                    return yaml.safe_load(txt)
                except Exception as e_yaml:
                    err = (
                        f"設定解析失敗。嘗試 JSON 與 YAML 皆失敗\n"
                        f"JSON 錯誤: {e_json}\nYAML 錯誤: {e_yaml}\n{_tb.format_exc()}"
                    )
                    self.log(err)
                    QMessageBox.critical(self, "設定解析失敗", err)
                    return None

    def build_yaml_from_controls(self):
        ds_root = self.edit_root.text().strip() or ""
        # collect selected preprocs
        selected_pre = [self.list_pre_sel.item(i).text() for i in range(self.list_pre_sel.count())]
        # ensure latest form edits are saved
        self._save_current_pre_form()
        self._save_model_form()
        model = self.combo_model.currentText()
        evaluator = self.combo_eval.currentText()
        ratios = [self.split_r_train.value(), self.split_r_test.value()]
        # build pipeline steps
        pipeline_steps = []
        for pre_name in selected_pre:
            params = dict(self.pre_params.get(pre_name) or getattr(PREPROC_REGISTRY.get(pre_name), 'DEFAULT_CONFIG', {}))
            pipeline_steps.append({
                "type": "preproc",
                "name": pre_name,
                "params": params
            })
        # 取得現有模型參數（若未修改則用預設），再合併：使用者 > 預設
        user_params = dict(self.model_params.get(model) or {})
        default_params = getattr(MODEL_REGISTRY[model], 'DEFAULT_CONFIG', {})
        mdl_params = {**default_params, **user_params}
        try:
            self.log(f"[BuildYAML] Model={model} default={default_params} user={user_params} merged={mdl_params}")
        except Exception:
            pass
        # sanitize known auto-init keys
        if model in ("CSRT", "OpticalFlowLK"):
            mdl_params.pop("init_box", None)
        pipeline_steps.append({
            "type": "model",
            "name": model,
            "params": mdl_params
        })
        cfg = {
            "seed": 42,
            "dataset": {
                "root": ds_root,
                "split": {
                    "method": self.split_method.currentText(),
                    "ratios": ratios,
                    "k_fold": int(self.kfold.value())
                }
            },
            "experiments": [
                {
                    "name": f"exp_{('_'.join(selected_pre) if selected_pre else 'no_pre')}_{model}",
                    "pipeline": pipeline_steps,
                }
            ],
            "evaluation": {
                "evaluator": evaluator,
            },
            # output omitted on purpose; UI will enforce results folder under project
        }
        try:
            import yaml  # type: ignore
            txt = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
        except Exception:
            txt = json.dumps(cfg, ensure_ascii=False, indent=2)
        self.txt_cfg.setPlainText(txt)

    def sync_controls_from_raw(self):
        cfg = self.parse_config()
        if not cfg:
            return
        self._syncing = True
        ds = cfg.get("dataset", {})
        root = ds.get("root")
        if root:
            self.edit_root.setText(root)
        split = (ds or {}).get("split", {})
        ratios = split.get("ratios", [0.8, 0.2])
        if len(ratios) == 3:
            self.split_r_train.setValue(float(ratios[0])); self.split_r_test.setValue(float(ratios[2]))
        elif len(ratios) == 2:
            self.split_r_train.setValue(float(ratios[0])); self.split_r_test.setValue(float(ratios[1]))
        self.kfold.setValue(int(split.get("k_fold", 1) or 1))
        exps = cfg.get("experiments", [])
        if exps:
            exp = exps[0]
            pipeline = exp.get("pipeline", [])
            pre_names = [s.get("name") for s in pipeline if s.get("type") == "preproc"]
            mdl_names = [s.get("name") for s in pipeline if s.get("type") == "model"]
            # rebuild preproc selection
            self.list_pre_sel.clear()
            for pname in pre_names:
                if pname in PREPROC_REGISTRY:
                    self.list_pre_sel.addItem(QListWidgetItem(pname))
                    step = next((s for s in pipeline if s.get("type") == "preproc" and s.get("name") == pname), None)
                    if step:
                        self.pre_params[pname] = dict(step.get("params") or {})
            if mdl_names:
                target = mdl_names[0]
                mdl_step = next((s for s in pipeline if s.get("type") == "model" and s.get("name") == target), None)
                if mdl_step:
                    raw_params = dict(mdl_step.get("params") or {})
                    if target in ("CSRT", "OpticalFlowLK"):
                        raw_params.pop("init_box", None)
                    self.model_params[target] = raw_params
                    try:
                        self.log(f"[SyncModelParams] model={target} raw_loaded={raw_params}")
                    except Exception:
                        pass
                idx = self.combo_model.findText(target)
                if idx >= 0:
                    old = self.combo_model.blockSignals(True)
                    self.combo_model.setCurrentIndex(idx)
                    self.combo_model.blockSignals(old)
                self._on_model_changed(target)
        ev = (cfg.get("evaluation", {}) or {}).get("evaluator")
        if ev:
            idx = self.combo_eval.findText(ev)
            if idx >= 0:
                self.combo_eval.setCurrentIndex(idx)
        ev_viz = (cfg.get("evaluation", {}) or {}).get("visualize", {}) or {}
        self.chk_viz.setChecked(bool(ev_viz.get("enabled", False)))
        try:
            self.spn_viz_samples.setValue(int(ev_viz.get("samples", 10) or 10))
        except Exception:
            self.spn_viz_samples.setValue(10)
        self._on_pre_selected_changed(self.list_pre_sel.currentRow())
        self._syncing = False

    def run_pipeline(self):
        import traceback as _tb
        # Env diagnostics: python, torch/torchvision, CUDA
        try:
            import sys as _sys
            self.log(f"Python: {_sys.executable}")
            self.log(f"Python version: {_sys.version}")
            try:
                import os as _os
                self.log(f"Conda env: {_os.environ.get('CONDA_DEFAULT_ENV', '<unknown>')}")
            except Exception:
                pass
            try:
                import torch as _t
                self.log(f"torch: {_t.__version__}")
                self.log(f"CUDA available: {_t.cuda.is_available()}")
                if _t.cuda.is_available():
                    try:
                        dev_name = _t.cuda.get_device_name(0)
                    except Exception:
                        dev_name = "<unknown>"
                    self.log(f"CUDA device[0]: {dev_name}")
            except Exception as e_t:
                self.log(f"torch import failed: {e_t}")
            try:
                import torchvision as _tv
                self.log(f"torchvision: {_tv.__version__}")
            except Exception as e_tv:
                self.log(f"torchvision import failed: {e_tv}")
            try:
                from PIL import Image as _PIL_Image
                import PIL as _PIL
                self.log(f"Pillow: {_PIL.__version__} ({_PIL_Image.__file__})")
            except Exception as e_pil:
                self.log(f"Pillow import failed: {e_pil}")
        except Exception:
            pass
        cfg = self.parse_config()
        if not cfg:
            self.log("設定內容為空或格式錯誤（詳細見上方 Logs 或終端）")
            QMessageBox.warning(self, "錯誤", "設定內容為空或格式錯誤")
            return
        root = self.edit_root.text().strip()
        if root:
            cfg.setdefault("dataset", {})["root"] = root
        # Override visualize from UI (do not persist to YAML)
        try:
            ev = cfg.setdefault("evaluation", {})
            ev_viz = ev.setdefault("visualize", {})
            ev_viz["enabled"] = bool(self.chk_viz.isChecked())
            ev_viz["samples"] = int(self.spn_viz_samples.value())
            # Keep default behavior for restrict_to_gt_frames unless provided in file
        except Exception:
            pass
        # Force results to be saved under the project folder (where ui.py lives)
        try:
            proj_root = os.path.dirname(os.path.abspath(__file__))
            out = cfg.setdefault("output", {})
            out["results_root"] = os.path.join(proj_root, "results")
            self.log(f"結果將儲存至: {out['results_root']}")
        except Exception:
            pass
        self.log("開始執行…")
        runner = PipelineRunner(cfg, logger=self.log)
        try:
            runner.run()
            self.log("完成。")
        except Exception as e:
            tb = _tb.format_exc()
            self.log(f"執行錯誤: {e}\n{tb}")
            QMessageBox.critical(self, "執行失敗", f"{e}\n（詳細請見 Logs/終端）")

    # Helpers for preprocessing UI
    def _with_title(self, title: str, widget: QWidget) -> QWidget:
        box = QGroupBox(title)
        l = QVBoxLayout(box)
        l.setContentsMargins(5,5,5,5)
        l.addWidget(widget)
        return box

    def _pre_add(self):
        it = self.list_pre_avail.currentItem()
        if not it:
            return
        name = it.text()
        # avoid duplicates to keep config clean; allow duplicates if needed by removing this guard
        for i in range(self.list_pre_sel.count()):
            if self.list_pre_sel.item(i).text() == name:
                return
        self.list_pre_sel.addItem(QListWidgetItem(name))

    def _pre_remove(self):
        row = self.list_pre_sel.currentRow()
        if row >= 0:
            self.list_pre_sel.takeItem(row)

    def _pre_up(self):
        row = self.list_pre_sel.currentRow()
        if row > 0:
            it = self.list_pre_sel.takeItem(row)
            self.list_pre_sel.insertItem(row - 1, it)
            self.list_pre_sel.setCurrentRow(row - 1)

    def _pre_down(self):
        row = self.list_pre_sel.currentRow()
        if row >= 0 and row < self.list_pre_sel.count() - 1:
            it = self.list_pre_sel.takeItem(row)
            self.list_pre_sel.insertItem(row + 1, it)
            self.list_pre_sel.setCurrentRow(row + 1)

    # --- Dynamic params forms ---
    def _on_pre_selected_changed(self, row: int):
        # save previous selection edits
        self._save_current_pre_form()
        name = self.list_pre_sel.item(row).text() if row >= 0 and row < self.list_pre_sel.count() else None
        self._current_pre_name = name
        # rebuild form for current selection
        self._clear_form(self.pre_form_layout)
        self._pre_bindings = []
        if not name:
            return
        defaults = getattr(PREPROC_REGISTRY[name], 'DEFAULT_CONFIG', {})
        params = dict(self.pre_params.get(name) or defaults)
        for key, val in params.items():
            w, getter, setter = self._make_editor(key, val)
            setter(val)
            self.pre_form_layout.addRow(QLabel(key), w)
            self._pre_bindings.append((key, getter, setter))

    def _on_model_changed(self, name: str):
        # Avoid saving form while syncing from raw
        if self._current_model_name and not self._syncing:
            self._save_model_form(self._current_model_name)
        self._clear_form(self.model_form_layout)
        self._model_bindings = []
        defaults = getattr(MODEL_REGISTRY[name], 'DEFAULT_CONFIG', {})
        if name in self.model_params and self.model_params[name]:
            params = {**defaults, **self.model_params[name]}
        else:
            params = dict(defaults)
        # sanitize for GT-initialized trackers
        if name in ("CSRT", "OpticalFlowLK"):
            params.pop("init_box", None)
        for key, val in params.items():
            w, getter, setter = self._make_editor(key, val)
            setter(val)
            self.model_form_layout.addRow(QLabel(key), w)
            self._model_bindings.append((key, getter, setter))
        # update current model name after rebuild
        self._current_model_name = name
        # 診斷：記錄 lr 顯示與來源
        try:
            lr_binding = None
            for k, getter, _ in self._model_bindings:
                if k.lower() == 'lr':
                    lr_binding = (k, getter())
                    break
            self.log(f"[ModelForm] model={name} form_lr={lr_binding[1] if lr_binding else 'N/A'} stored_params_lr={self.model_params.get(name, {}).get('lr','N/A')} default_lr={defaults.get('lr','N/A')}")
        except Exception:
            pass

    def _on_model_index_changed(self, idx: int):
        name = self.combo_model.itemText(idx)
        self._on_model_changed(name)

    def _save_current_pre_form(self):
        name = self._current_pre_name
        if not name or not self._pre_bindings:
            return
        params = {}
        for key, getter, _ in self._pre_bindings:
            params[key] = getter()
        self.pre_params[name] = params

    def _save_model_form(self, name: Optional[str] = None):
        if name is None:
            name = self.combo_model.currentText()
        if not name or not self._model_bindings:
            return
        params = {}
        for key, getter, _ in self._model_bindings:
            params[key] = getter()
        # sanitize known auto-init keys
        if name in ("CSRT", "OpticalFlowLK"):
            params.pop("init_box", None)
        self.model_params[name] = params

    def _clear_form(self, form: QFormLayout):
        # Properly clear a QFormLayout by removing rows
        try:
            while form.rowCount() > 0:
                form.removeRow(0)
        except Exception:
            # Fallback for older bindings
            while form.count():
                item = form.takeAt(0)
                if item is None:
                    break
                w = item.widget()
                if w is not None:
                    w.deleteLater()

    def _make_editor(self, key: str, value):
        # returns (widget, getter, setter)
        if isinstance(value, bool):
            chk = QCheckBox()
            return (
                chk,
                lambda: bool(chk.isChecked()),
                lambda v: chk.setChecked(bool(v)),
            )
        elif isinstance(value, int):
            sp = QSpinBox(); sp.setRange(-10**9, 10**9)
            return (
                sp,
                lambda: int(sp.value()),
                lambda v: sp.setValue(int(v)),
            )
        elif isinstance(value, float):
            # Use line edit to preserve scientific notation (e.g., 1e-8) and avoid rounding to 0.000000
            le = QLineEdit()
            # Accept floats with optional scientific notation
            try:
                rx = QRegularExpression(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
                le.setValidator(QRegularExpressionValidator(rx))
            except Exception:
                # Fallback: no validator
                pass
            orig = float(value)
            def _get():
                txt = le.text().strip()
                if not txt:
                    return orig
                try:
                    return float(txt)
                except Exception:
                    return orig
            def _set(v):
                try:
                    fv = float(v)
                    # .12g keeps significant digits and uses scientific notation for small numbers
                    le.setText(format(fv, ".12g"))
                except Exception:
                    le.setText(str(v))
            return (
                le,
                _get,
                _set,
            )
        elif isinstance(value, str):
            le = QLineEdit()
            return (
                le,
                lambda: le.text(),
                lambda v: le.setText(str(v)),
            )
        else:
            # list/dict/others -> JSON text
            le = QLineEdit()
            import json as _json
            def _get():
                txt = le.text().strip()
                if not txt:
                    return value
                try:
                    return _json.loads(txt)
                except Exception:
                    return value
            def _set(v):
                try:
                    le.setText(_json.dumps(v))
                except Exception:
                    le.setText(str(v))
            return (
                le,
                _get,
                _set,
            )


def main():
    import sys
    app = QApplication(sys.argv)
    ui = SimpleRunnerUI()
    ui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
