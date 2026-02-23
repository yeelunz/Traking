from __future__ import annotations

import copy
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Iterator

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QFileDialog, QListWidgetItem, QMessageBox


class QueueMixin:
    """Queue scheduling helpers extracted from ui.py.

    This mixin expects the host class to provide:
    - UI fields: list_queue, btn_queue_import/add/remove/clear/run, btn_run, btn_load
    - logging/status helpers: log(), _set_status(), _serialize_cfg(), _set_raw_text_programmatically(),
      _apply_cfg_to_builder(), _highlight_raw_error(), _on_builder_changed(), build_config_dict(), _start_run_thread()
    - state fields: _queue_items, _queue_pending, _queue_running, _queue_error, _queue_total,
      _queue_completed, _queue_results_root, _queue_current_label, _raw_user_edit, _run_thread
    - combo_model for fallback model name
    """

    # ---- Queue scheduling helpers ----
    def _queue_iter_config_variants(self, cfg: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        if not isinstance(cfg, dict):
            return
        experiments = cfg.get('experiments')
        if not isinstance(experiments, list) or len(experiments) <= 1:
            yield copy.deepcopy(cfg)
            return
        base = copy.deepcopy(cfg)
        for exp in experiments:
            if not isinstance(exp, dict):
                continue
            variant = copy.deepcopy(base)
            variant['experiments'] = [copy.deepcopy(exp)]
            yield variant

    def _queue_extract_summary(self, cfg: Dict[str, Any]) -> Tuple[str, List[str], str, str]:
        dataset_root = ''
        try:
            dataset_root = str((cfg.get('dataset') or {}).get('root', '') or '')
        except Exception:
            dataset_root = ''
        experiments = cfg.get('experiments') or []
        exp = experiments[0] if experiments and isinstance(experiments[0], dict) else {}
        exp_name = str(exp.get('name') or '').strip() if isinstance(exp, dict) else ''
        pipeline = exp.get('pipeline') if isinstance(exp, dict) else None
        pre_names: List[str] = []
        model_name = ''
        if isinstance(pipeline, list):
            for step in pipeline:
                if not isinstance(step, dict):
                    continue
                step_type = step.get('type')
                step_name = step.get('name')
                if step_type == 'preproc' and step_name:
                    pre_names.append(str(step_name))
                elif step_type == 'model' and step_name:
                    model_name = str(step_name)
        if not model_name:
            model_name = str(self.combo_model.currentText() or '')
        return exp_name, pre_names, model_name, dataset_root

    def _queue_build_label(self, cfg: Dict[str, Any], preferred_label: Optional[str] = None) -> str:
        exp_name, pre_names, model_name, dataset_root = self._queue_extract_summary(cfg)
        if preferred_label:
            custom = preferred_label.strip()
            if custom:
                return custom
        label_base = (exp_name or '').strip()
        if not label_base:
            label_base = f"Exp{len(self._queue_items) + 1}"
        parts = [label_base]
        if model_name:
            parts.append(f"model={model_name}")
        if pre_names:
            parts.append("pre=" + " + ".join(pre_names))
        if dataset_root:
            safe_root = dataset_root.rstrip("\\/") or dataset_root
            parts.append(os.path.basename(os.path.normpath(safe_root)))
        return " | ".join(parts)

    def _queue_build_tooltip(self, cfg: Dict[str, Any]) -> str:
        _, pre_names, model_name, dataset_root = self._queue_extract_summary(cfg)
        lines = [f"資料集: {dataset_root or '(未設定)'}"]
        if model_name:
            lines.append(f"模型: {model_name}")
        if pre_names:
            lines.append(f"前處理: {', '.join(pre_names)}")
        return "\n".join(lines)

    def _on_queue_item_double_clicked(self, item: QListWidgetItem):
        if item is None:
            return
        row = self.list_queue.row(item)
        if row < 0 or row >= len(self._queue_items):
            return
        entry = self._queue_items[row]
        cfg = entry.get('config') if isinstance(entry, dict) else None
        if not isinstance(cfg, dict):
            return
        snapshot = copy.deepcopy(cfg)
        text = self._serialize_cfg(snapshot)
        # Update raw + builder together so the left editor stays in sync with the schedule selection.
        self._set_raw_text_programmatically(text)
        try:
            self._syncing = True
            self._apply_cfg_to_builder(snapshot)
        finally:
            self._syncing = False
        # Normalize raw display after builder sync (avoid debounce waiting).
        self._raw_user_edit = False
        self._highlight_raw_error(False)
        self._on_builder_changed(force=True)
        label = entry.get('label', '') if isinstance(entry, dict) else ''
        status = f"排程檢視：{label}" if label else "排程檢視"
        self._set_status(status, good=True)
        self.log(f"[QueueInspect] 已顯示排程配置：{label or row + 1}")

    def _queue_append_entry(self, cfg: Dict[str, Any], label: Optional[str] = None) -> bool:
        if not isinstance(cfg, dict):
            return False
        snapshot = copy.deepcopy(cfg)
        label_text = self._queue_build_label(snapshot, label)
        entry = {'config': snapshot, 'label': label_text}
        self._queue_items.append(entry)
        item = QListWidgetItem(label_text)
        tooltip = self._queue_build_tooltip(snapshot)
        if tooltip:
            item.setToolTip(tooltip)
        self.list_queue.addItem(item)
        self.log(f"已加入排程：{label_text}")
        return True

    def _queue_read_schedule_file(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(text)
            if data is not None:
                return data
        except Exception:
            pass
        try:
            return json.loads(text)
        except Exception as exc:
            raise ValueError(exc)

    def _queue_parse_schedule_data(self, data: Any) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []

        def _consume(obj: Any, base_label: Optional[str] = None):
            if not isinstance(obj, dict):
                return
            variants = list(self._queue_iter_config_variants(obj))
            total = len(variants)
            for idx, variant in enumerate(variants, start=1):
                label = base_label
                if base_label and total > 1:
                    label = f"{base_label} #{idx}"
                entries.append({'config': variant, 'label': label})

        if isinstance(data, dict):
            queue_section = data.get('queue')
            if isinstance(queue_section, list) and queue_section:
                for item in queue_section:
                    if not isinstance(item, dict):
                        continue
                    cfg_obj = item.get('config') if isinstance(item.get('config'), dict) else item
                    lbl = item.get('label') if isinstance(item.get('label'), str) else None
                    _consume(cfg_obj, lbl)
            else:
                _consume(data)
        elif isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                cfg_obj = item.get('config') if isinstance(item.get('config'), dict) else item
                lbl = item.get('label') if isinstance(item.get('label'), str) else None
                _consume(cfg_obj, lbl)
        return entries

    def _queue_import_from_file(self):
        if self._queue_running:
            QMessageBox.warning(self, '排程進行中', '排程執行期間無法匯入設定。')
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            '選擇排程設定檔',
            os.getcwd(),
            'YAML/JSON (*.yaml *.yml *.json);;All Files (*)',
        )
        if not paths:
            return
        total_added = 0
        for path in paths:
            try:
                data = self._queue_read_schedule_file(path)
            except Exception as exc:
                self.log(f"[QueueImport][錯誤] 無法讀取 {path}: {exc}")
                QMessageBox.warning(self, '匯入失敗', f"{os.path.basename(path)} 讀取失敗：{exc}")
                continue
            items = self._queue_parse_schedule_data(data)
            if not items:
                self.log(f"[QueueImport] {path} 未找到有效設定。")
                QMessageBox.information(self, '匯入結果', f"{os.path.basename(path)} 沒有可用的排程設定。")
                continue
            added = 0
            for item in items:
                if self._queue_append_entry(item.get('config'), item.get('label')):
                    added += 1
            total_added += added
            self.log(f"[QueueImport] {os.path.basename(path)} 匯入 {added} 項。")
        if total_added:
            self._set_status(f"已匯入 {total_added} 組排程", good=True)
        else:
            self._set_status('匯入排程設定未成功', good=False)

    def _queue_default_results_root(self) -> str:
        try:
            proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        except Exception:
            proj_root = os.getcwd()
        return os.path.join(proj_root, 'results')

    def _queue_add_current(self):
        if self._queue_running:
            QMessageBox.warning(self, '排程進行中', '排程執行期間無法新增項目。')
            return
        cfg = self.build_config_dict()
        self._queue_append_entry(cfg)

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
            stamp = time.strftime('%Y-%m-%d_%H-%M-%S')
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
        self._queue_detector_cache = {}
        self.btn_queue_import.setEnabled(False)
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
        self._start_run_thread(payload['config'], detector_cache=self._queue_detector_cache)

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
        self._queue_detector_cache = None
        self._queue_current_label = None
        self.btn_queue_import.setEnabled(True)
        self.btn_queue_add.setEnabled(True)
        self.btn_queue_remove.setEnabled(True)
        self.btn_queue_clear.setEnabled(True)
        self.btn_queue_run.setEnabled(True)
        self.list_queue.setEnabled(True)
        self.btn_run.setEnabled(True)
        self.btn_load.setEnabled(True)
        if success:
            self._set_status('排程完成', good=True)
            self.log('排程全部完成。')
        else:
            self._set_status('排程中途失敗', good=False)
            self.log('排程已停止，請檢查錯誤訊息後再試一次。')
        self._queue_error = False
