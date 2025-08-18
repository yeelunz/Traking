from __future__ import annotations
import json
import os
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QTextEdit, QLineEdit, QMessageBox,
    QComboBox, QSpinBox, QDoubleSpinBox, QFormLayout, QGroupBox,
    QListWidget, QListWidgetItem, QAbstractItemView, QCheckBox
)

from tracking.orchestrator.runner import PipelineRunner
from tracking.core.registry import PREPROC_REGISTRY, MODEL_REGISTRY, EVAL_REGISTRY
# import built-in plugins to populate registries for dropdowns
from tracking.preproc import clahe  # noqa: F401
from tracking.models import template_matching  # noqa: F401
from tracking.models import csrt  # noqa: F401
from tracking.models import optical_flow_lk  # noqa: F401
from tracking.eval import evaluator  # noqa: F401


class SimpleRunnerUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tracking Pipeline UI (Lite)")
        self.resize(900, 700)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Config path row
        row = QHBoxLayout()
        row.addWidget(QLabel("Config (YAML/JSON):"))
        self.edit_cfg = QLineEdit()
        btn_browse = QPushButton("瀏覽…")
        btn_browse.clicked.connect(self.browse_cfg)
        row.addWidget(self.edit_cfg, 1)
        row.addWidget(btn_browse)
        layout.addLayout(row)

        # Dataset root row
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Dataset Root:"))
        self.edit_root = QLineEdit()
        btn_root = QPushButton("選資料夾…")
        btn_root.clicked.connect(self.browse_root)
        row2.addWidget(self.edit_root, 1)
        row2.addWidget(btn_root)
        layout.addLayout(row2)

        # Guided builder
        gb = QGroupBox("Pipeline Builder")
        gbl = QVBoxLayout(gb)
        ds_form = QFormLayout()
        self.split_method = QComboBox(); self.split_method.addItems(["video_level"])  # reserve
        self.split_r_train = QDoubleSpinBox(); self.split_r_train.setRange(0,1); self.split_r_train.setSingleStep(0.05); self.split_r_train.setValue(0.8)
        self.split_r_test = QDoubleSpinBox(); self.split_r_test.setRange(0,1); self.split_r_test.setSingleStep(0.05); self.split_r_test.setValue(0.2)
        self.kfold = QSpinBox(); self.kfold.setRange(1, 20); self.kfold.setValue(1)
        ds_form.addRow("Split Method", self.split_method)
        ds_form.addRow("Ratios (train/test)", self._hbox([self.split_r_train, self.split_r_test]))
        ds_form.addRow("K-Fold (1=off)", self.kfold)
        gbl.addLayout(ds_form)

        # Preprocessing selector with add/remove/reorder
        pre_row = QVBoxLayout()
        pre_row_header = QHBoxLayout()
        pre_row_header.addWidget(QLabel("Preprocessing:"))
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

        mdl_row = QHBoxLayout()
        mdl_row.addWidget(QLabel("Model:"))
        self.combo_model = QComboBox()
        for name in sorted(MODEL_REGISTRY.keys()):
            self.combo_model.addItem(name)
        mdl_row.addWidget(self.combo_model, 1)
        gbl.addLayout(mdl_row)

        # Parameter editors
        self.pre_params = {}  # name -> params dict
        self._current_pre_name = None
        self._pre_bindings = []  # list of (key, getter, setter)
        self.model_params = {}  # model_name -> params dict
        self._model_bindings = []
        self._current_model_name: Optional[str] = None

        # Preproc params editor (edit the currently selected item in right list)
        pre_params_box = QGroupBox("Preproc 參數（編輯右側已選中的項目）")
        pre_params_l = QVBoxLayout(pre_params_box)
        # keep a reference to place/reset forms later
        self.pre_params_vlayout = pre_params_l
        self.pre_form_container = QWidget(); self.pre_form_layout = QFormLayout(self.pre_form_container)
        pre_params_l.addWidget(self.pre_form_container)
        btn_pre_save = QPushButton("儲存 Preproc 參數"); btn_pre_save.clicked.connect(self._save_current_pre_form)
        pre_params_l.addWidget(btn_pre_save)
        gbl.addWidget(pre_params_box)

        # Model params editor
        model_params_box = QGroupBox("Model 參數")
        model_params_l = QVBoxLayout(model_params_box)
        # keep a reference to place/reset forms later
        self.model_params_vlayout = model_params_l
        self.model_form_container = QWidget(); self.model_form_layout = QFormLayout(self.model_form_container)
        model_params_l.addWidget(self.model_form_container)
        btn_model_save = QPushButton("儲存 Model 參數"); btn_model_save.clicked.connect(self._save_model_form)
        model_params_l.addWidget(btn_model_save)
        gbl.addWidget(model_params_box)

        ev_row = QHBoxLayout()
        ev_row.addWidget(QLabel("Evaluator:"))
        self.combo_eval = QComboBox()
        for name in sorted(EVAL_REGISTRY.keys()):
            self.combo_eval.addItem(name)
        ev_row.addWidget(self.combo_eval, 1)
        gbl.addLayout(ev_row)

        gbl.addWidget(QLabel("Raw Config (optional):"))
        self.txt_cfg = QTextEdit()
        gbl.addWidget(self.txt_cfg, 1)
        layout.addWidget(gb)

        # Controls
        ctr = QHBoxLayout()
        btn_load = QPushButton("載入檔案"); btn_load.clicked.connect(self.load_file)
        btn_save = QPushButton("存檔"); btn_save.clicked.connect(self.save_file)
        btn_build = QPushButton("由上方選項產生 YAML"); btn_build.clicked.connect(self.build_yaml_from_controls)
        btn_sync_from_raw = QPushButton("同步 Raw 到 Builder"); btn_sync_from_raw.setToolTip("將目前 Raw Config 解析並填入上方控制項"); btn_sync_from_raw.clicked.connect(self.sync_controls_from_raw)
        btn_run = QPushButton("執行 Pipeline"); btn_run.clicked.connect(self.run_pipeline)
        for b in (btn_load, btn_save, btn_build, btn_sync_from_raw, btn_run):
            ctr.addWidget(b)
        layout.addLayout(ctr)
        # Logs
        layout.addWidget(QLabel("Logs:"))
        self.txt_log = QTextEdit(); self.txt_log.setReadOnly(True)
        layout.addWidget(self.txt_log, 1)

        # wire events for params editors
        self.list_pre_sel.currentRowChanged.connect(self._on_pre_selected_changed)
        # track current model to save the correct one before switching
        self._current_model_name = self.combo_model.currentText()
        self.combo_model.currentIndexChanged.connect(self._on_model_index_changed)
        # init model form
        self._on_model_changed(self._current_model_name)

    def log(self, msg: str):
        self.txt_log.append(msg)

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
        txt = self.txt_cfg.toPlainText()
        if not txt.strip():
            return None
        path = self.edit_cfg.text().lower()
        if path.endswith((".yaml", ".yml")):
            try:
                import yaml  # type: ignore
                return yaml.safe_load(txt)
            except Exception as e:
                QMessageBox.critical(self, "YAML 錯誤", str(e))
                return None
        else:
            try:
                return json.loads(txt)
            except Exception as e:
                QMessageBox.critical(self, "JSON 錯誤", str(e))
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
        mdl_params = dict(self.model_params.get(model) or getattr(MODEL_REGISTRY[model], 'DEFAULT_CONFIG', {}))
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
            "evaluation": {"evaluator": evaluator},
            "output": {"results_root": os.path.join(os.getcwd(), "results")}
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
        # dataset
        ds = cfg.get("dataset", {})
        root = ds.get("root")
        if root:
            self.edit_root.setText(root)
        split = (ds or {}).get("split", {})
        ratios = split.get("ratios", [0.8, 0.2])
        if len(ratios) == 3:
            self.split_r_train.setValue(float(ratios[0]))
            self.split_r_test.setValue(float(ratios[2]))
        elif len(ratios) == 2:
            self.split_r_train.setValue(float(ratios[0]))
            self.split_r_test.setValue(float(ratios[1]))
        self.kfold.setValue(int(split.get("k_fold", 1) or 1))
        # experiments (take first by default)
        exps = cfg.get("experiments", [])
        if exps:
            exp = exps[0]
            pipeline = exp.get("pipeline", [])
            pre_names = [s.get("name") for s in pipeline if s.get("type") == "preproc"]
            mdl_names = [s.get("name") for s in pipeline if s.get("type") == "model"]
            # set preproc selections
            # rebuild selected list
            self.list_pre_sel.clear()
            for name in pre_names:
                if name in PREPROC_REGISTRY:
                    self.list_pre_sel.addItem(QListWidgetItem(name))
                    # store params for each preproc
                    step = next((s for s in pipeline if s.get("type") == "preproc" and s.get("name") == name), None)
                    if step is not None:
                        params = dict(step.get("params") or {})
                        # keep only known keys if DEFAULT_CONFIG exists
                        defaults = getattr(PREPROC_REGISTRY[name], 'DEFAULT_CONFIG', {})
                        for k in list(params.keys()):
                            if defaults and k not in defaults:
                                pass
                        self.pre_params[name] = params
            # set model (first)
            if mdl_names:
                idx = self.combo_model.findText(mdl_names[0])
                if idx >= 0:
                    self.combo_model.setCurrentIndex(idx)
                # store model params
                mdl_step = next((s for s in pipeline if s.get("type") == "model"), None)
                if mdl_step:
                    params = dict(mdl_step.get("params") or {})
                    # remove init_box if present (auto-init from GT)
                    if mdl_names[0] in ("CSRT", "OpticalFlowLK"):
                        params.pop("init_box", None)
                    self.model_params[mdl_names[0]] = params
        # evaluator
        ev = (cfg.get("evaluation", {}) or {}).get("evaluator")
        if ev:
            idx = self.combo_eval.findText(ev)
            if idx >= 0:
                self.combo_eval.setCurrentIndex(idx)
        # refresh forms to reflect stored params
        self._on_pre_selected_changed(self.list_pre_sel.currentRow())
        self._on_model_changed(self.combo_model.currentText())

    def run_pipeline(self):
        cfg = self.parse_config()
        if not cfg:
            QMessageBox.warning(self, "錯誤", "設定內容為空或格式錯誤")
            return
        root = self.edit_root.text().strip()
        if root:
            cfg.setdefault("dataset", {})["root"] = root
        self.log("開始執行…")
        runner = PipelineRunner(cfg, logger=self.log)
        try:
            runner.run()
            self.log("完成。")
        except Exception as e:
            self.log(f"錯誤: {e}")
            QMessageBox.critical(self, "執行失敗", str(e))

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
        # save existing edits for the previous model name
        if self._current_model_name:
            self._save_model_form(self._current_model_name)
        # rebuild form
        self._clear_form(self.model_form_layout)
        self._model_bindings = []
        defaults = getattr(MODEL_REGISTRY[name], 'DEFAULT_CONFIG', {})
        params = dict(self.model_params.get(name) or defaults)
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
            sp = QDoubleSpinBox(); sp.setDecimals(6); sp.setRange(-1e9, 1e9)
            return (
                sp,
                lambda: float(sp.value()),
                lambda v: sp.setValue(float(v)),
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
