from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class Column:
    header: str
    width: int
    getter: Any


def _parse_iso8601(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        text = value.replace("Z", "+00:00")
        return datetime.fromisoformat(text)
    except Exception:
        return None


def _format_timestamp(dt: Optional[datetime]) -> str:
    if not dt:
        return "-"
    try:
        return dt.astimezone().strftime("%Y-%m-%d %H:%M")
    except Exception:
        return dt.strftime("%Y-%m-%d %H:%M")


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    try:
        delta = timedelta(seconds=float(seconds))
    except Exception:
        return "-"
    total_seconds = int(delta.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def _format_float(value: Optional[float], precision: int = 3) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{precision}f}"
    except Exception:
        return "-"


def _format_percent(value: Optional[float], precision: int = 1) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value) * 100:.{precision}f}%"
    except Exception:
        return "-"


def _format_ratio(train: Optional[int], test: Optional[int]) -> str:
    if train is None and test is None:
        return "-"
    if train is None:
        return f"-/ {test or 0}"
    if test is None:
        return f"{train or 0}/-"
    return f"{train}/{test}"


def _numeric_avg(values: Iterable[Optional[float]]) -> Optional[float]:
    nums = [float(v) for v in values if isinstance(v, (int, float))]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _summarise_detection(summary: Optional[Dict[str, Dict[str, Any]]]) -> Dict[str, Optional[float]]:
    if not summary:
        return {}
    return {
        "success_rate_50": _numeric_avg(stats.get("success_rate_50") for stats in summary.values()),
        "success_rate_75": _numeric_avg(stats.get("success_rate_75") for stats in summary.values()),
        "success_auc": _numeric_avg(stats.get("success_auc") for stats in summary.values()),
        "fps": _numeric_avg(stats.get("fps") for stats in summary.values()),
        "iou_mean": _numeric_avg(stats.get("iou_mean") for stats in summary.values()),
        "ce_mean": _numeric_avg(stats.get("ce_mean") for stats in summary.values()),
    }


def _summarise_segmentation(metrics: Optional[Dict[str, Dict[str, Dict[str, Any]]]]) -> Dict[str, Optional[float]]:
    if not metrics:
        return {}
    values: Dict[str, List[float]] = {}
    for seg_map in metrics.values():
        for det_stats in seg_map.values():
            for key in ("dice_mean", "iou_mean", "centroid_mean", "fps_mean"):
                value = det_stats.get(key)
                if isinstance(value, (int, float)):
                    values.setdefault(key, []).append(float(value))
    return {key: (sum(vals) / len(vals) if vals else None) for key, vals in values.items()}


def _summarise_classification(clf_metrics: Optional[Dict[str, Any]], exp_path: Optional[Path] = None) -> Dict[str, Optional[float]]:
    """Extract classification summary from metadata dict or classification/summary.json file.

    Tries ``clf_metrics`` (from metadata.json) first, then falls back to reading
    ``<exp_path>/classification/summary.json`` directly (for older experiment dirs).
    """
    data: Optional[Dict[str, Any]] = None
    if isinstance(clf_metrics, dict) and clf_metrics:
        data = clf_metrics
    elif exp_path is not None:
        summary_file = exp_path / "classification" / "summary.json"
        if summary_file.is_file():
            try:
                data = json.loads(summary_file.read_text(encoding="utf-8"))
            except Exception:
                pass
    if not isinstance(data, dict) or not data:
        return {}
    return {
        "accuracy": data.get("accuracy"),
        "f1": data.get("f1") if data.get("f1") is not None else data.get("f1_positive"),
        "precision": data.get("precision") if data.get("precision") is not None else data.get("precision_positive"),
        "recall": data.get("recall") if data.get("recall") is not None else data.get("recall_positive"),
        "balanced_accuracy": data.get("balanced_accuracy"),
    }


def _overall_stage_status(stages: Sequence[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    summary: List[Dict[str, Any]] = []
    has_failed = False
    has_skipped = False
    for stage in stages:
        name = stage.get("name", "?")
        kind = stage.get("kind", "?")
        status = str(stage.get("status", "unknown")).lower()
        duration = stage.get("duration_sec")
        summary.append(
            {
                "name": name,
                "kind": kind,
                "status": status,
                "duration_sec": duration,
                "details": {k: v for k, v in stage.items() if k not in {"name", "kind", "status", "duration_sec"}},
            }
        )
        if status == "failed":
            has_failed = True
        elif status == "skipped":
            has_skipped = True
    if has_failed:
        return "FAILED", summary
    if has_skipped:
        return "PARTIAL", summary
    return "OK", summary


# ---------------------------------------------------------------------------
# Note helpers – each experiment folder may contain a note.txt
# ---------------------------------------------------------------------------

def _note_path(exp_dir: Path) -> Path:
    return Path(exp_dir) / "note.txt"


def _load_note(exp_dir: Path) -> str:
    p = _note_path(exp_dir)
    if p.is_file():
        try:
            return p.read_text(encoding="utf-8").strip()
        except Exception:
            return ""
    return ""


def _save_note(exp_dir: Path, text: str) -> None:
    Path(exp_dir).mkdir(parents=True, exist_ok=True)
    _note_path(exp_dir).write_text(text, encoding="utf-8")


def _collect_experiments(results_root: Path) -> List[Dict[str, Any]]:
    if not results_root.exists():
        return []
    records: List[Dict[str, Any]] = []
    for entry in sorted(results_root.iterdir()):
        if not entry.is_dir():
            continue
        meta_path = entry / "metadata.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            records.append(
                {
                    "name": entry.name,
                    "status": "BROKEN",
                    "relative_path": str(entry.relative_to(results_root)),
                    "path": str(entry),
                    "error": f"Failed to read metadata.json: {exc}",
                }
            )
            continue
        exp_info = meta.get("experiment", {})
        runtime = meta.get("runtime", {})
        dataset = meta.get("dataset", {})
        metrics_root = meta.get("metrics", {})
        stages = meta.get("stages", [])
        stage_status, stage_details = _overall_stage_status(stages)
        det_summary = _summarise_detection(metrics_root.get("detection")) if isinstance(metrics_root, dict) else {}
        seg_summary = _summarise_segmentation(metrics_root.get("segmentation")) if isinstance(metrics_root, dict) else {}
        clf_summary = _summarise_classification(
            metrics_root.get("classification") if isinstance(metrics_root, dict) else None,
            exp_path=entry,
        )
        started = _parse_iso8601(runtime.get("started_at")) if isinstance(runtime, dict) else None
        finished = _parse_iso8601(runtime.get("finished_at")) if isinstance(runtime, dict) else None
        duration = runtime.get("duration_sec") if isinstance(runtime, dict) else None
        record: Dict[str, Any] = {
            "name": exp_info.get("name") or entry.name,
            "path": str(entry),
            "relative_path": str(entry.relative_to(results_root)),
            "status": stage_status,
            "started_at": started,
            "started_at_iso": runtime.get("started_at") if isinstance(runtime, dict) else None,
            "finished_at_iso": runtime.get("finished_at") if isinstance(runtime, dict) else None,
            "duration_sec": duration if isinstance(duration, (int, float)) else None,
            "stages": stage_details,
            "dataset": {
                "train_videos": dataset.get("train_videos"),
                "test_videos": dataset.get("test_videos"),
                "missing_annotations": dataset.get("missing_annotations"),
                "missing_preview": dataset.get("missing_preview", []),
            },
            "metrics": {
                "detection": det_summary,
                "segmentation": seg_summary,
                "classification": clf_summary,
            },
            "note": _load_note(entry),
        }
        records.append(record)
    return records


def _sort_records(records: List[Dict[str, Any]], key_name: str, descending: bool) -> None:
    def sort_key(rec: Dict[str, Any]):
        if key_name == "start":
            return rec.get("started_at") or datetime.min
        if key_name == "duration":
            dur = rec.get("duration_sec")
            return float(dur) if isinstance(dur, (int, float)) else -1.0
        if key_name == "det@50":
            metrics = rec.get("metrics", {}).get("detection", {})
            val = metrics.get("success_rate_50")
            return float(val) if isinstance(val, (int, float)) else -1.0
        if key_name == "seg_dice":
            metrics = rec.get("metrics", {}).get("segmentation", {})
            val = metrics.get("dice_mean")
            return float(val) if isinstance(val, (int, float)) else -1.0
        if key_name == "clf_acc":
            metrics = rec.get("metrics", {}).get("classification", {})
            val = metrics.get("accuracy")
            return float(val) if isinstance(val, (int, float)) else -1.0
        if key_name == "clf_f1":
            metrics = rec.get("metrics", {}).get("classification", {})
            val = metrics.get("f1")
            return float(val) if isinstance(val, (int, float)) else -1.0
        return rec.get("relative_path")

    records.sort(key=sort_key, reverse=descending)


def _print_table(records: Sequence[Dict[str, Any]]) -> None:
    columns = [
        Column("Started", 17, lambda r: _format_timestamp(r.get("started_at"))),
        Column("Experiment", 22, lambda r: r.get("name", "-")),
        Column("Status", 8, lambda r: r.get("status", "-")),
        Column("Duration", 9, lambda r: _format_duration(r.get("duration_sec"))),
        Column("Det@50", 8, lambda r: _format_percent(r.get("metrics", {}).get("detection", {}).get("success_rate_50"))),
        Column("Det FPS", 8, lambda r: _format_float(r.get("metrics", {}).get("detection", {}).get("fps"))),
        Column("Seg Dice", 9, lambda r: _format_float(r.get("metrics", {}).get("segmentation", {}).get("dice_mean"))),
        Column("Seg FPS", 8, lambda r: _format_float(r.get("metrics", {}).get("segmentation", {}).get("fps_mean"))),
        Column("Clf Acc", 8, lambda r: _format_percent(r.get("metrics", {}).get("classification", {}).get("accuracy"))),
        Column("Clf F1", 7, lambda r: _format_float(r.get("metrics", {}).get("classification", {}).get("f1"))),
        Column("Clf Prec", 8, lambda r: _format_float(r.get("metrics", {}).get("classification", {}).get("precision"))),
        Column("Clf Rec", 8, lambda r: _format_float(r.get("metrics", {}).get("classification", {}).get("recall"))),
        Column("Videos", 11, lambda r: _format_ratio(
            r.get("dataset", {}).get("train_videos"),
            r.get("dataset", {}).get("test_videos"),
        )),
        Column("Note", 35, lambda r: (r.get("note") or "").replace("\n", " ")[:35]),
        Column("Folder", 40, lambda r: r.get("relative_path", "-")),
    ]

    header = "  ".join(col.header.ljust(col.width) for col in columns)
    print(header)
    print("  ".join("-" * col.width for col in columns))
    for rec in records:
        row = []
        for col in columns:
            value = col.getter(rec)
            row.append(str(value)[:col.width].ljust(col.width))
        print("  ".join(row))


def _print_stage_details(records: Sequence[Dict[str, Any]]) -> None:
    for rec in records:
        print(f"\n[{rec.get('status', '-')}] {rec.get('name', '?')} :: {rec.get('relative_path', '')}")
        for stage in rec.get("stages", []):
            status = stage.get("status", "-" ).upper()
            label = f"{stage.get('kind', '?')}::{stage.get('name', '?')}"
            duration = _format_duration(stage.get("duration_sec"))
            print(f"  - {status:8s} {label:40s} {duration}")


def _to_serialisable(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    serialisable: List[Dict[str, Any]] = []
    for rec in records:
        payload = dict(rec)
        payload["started_at"] = rec.get("started_at_iso")
        payload.pop("started_at_iso", None)
        payload.pop("started_at", None)
        serialisable.append(payload)
    return serialisable


def _generate_html_str(records: Sequence[Dict[str, Any]]) -> str:
    data = _to_serialisable(records)
    table_rows = json.dumps(data, indent=2)
    return f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
  <meta charset="utf-8" />
  <title>Experiment Viewer</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; background: #fafafa; color: #222; }}
    table {{ border-collapse: collapse; width: 100%; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }}
    th, td {{ padding: 0.6rem 0.8rem; border-bottom: 1px solid #ddd; text-align: left; }}
    th {{ background: #f0f0f0; cursor: pointer; position: sticky; top: 0; }}
    tr:hover {{ background: #f9f9f9; }}
    .status-FAILED {{ color: #c62828; font-weight: 600; }}
    .status-PARTIAL {{ color: #ef6c00; font-weight: 600; }}
    .status-OK {{ color: #2e7d32; font-weight: 600; }}
    caption {{ text-align: left; font-size: 1.4rem; margin-bottom: 1rem; font-weight: 600; }}
    .note-cell {{ min-width: 200px; max-width: 320px; }}
    .note-text {{ cursor: pointer; white-space: pre-wrap; font-size: 0.85em; color: #444;
                  display: block; min-height: 1.2em; border-radius: 3px; padding: 2px 4px; }}
    .note-text.empty {{ color: #bbb; font-style: italic; }}
    .note-text:hover {{ background: #eef5ff; }}
    textarea.note-edit {{ width: 100%; min-height: 64px; font-family: inherit; font-size: 0.85em;
                           padding: 4px; border: 1px solid #4a90d9; border-radius: 3px;
                           resize: vertical; box-sizing: border-box; }}
    .note-status {{ display: block; font-size: 0.72em; margin-top: 2px; }}
    .note-saving {{ color: #888; }}
    .note-saved {{ color: #2e7d32; }}
    .note-error {{ color: #c62828; }}
  </style>
</head>
<body>
  <caption>Experiment Summary</caption>
  <table id=\"exp-table\">
    <thead>
      <tr>
        <th data-key=\"started_at\">Started</th>
        <th data-key=\"name\">Experiment</th>
        <th data-key=\"status\">Status</th>
        <th data-key=\"duration_sec\">Duration (s)</th>
        <th data-key=\"metrics.detection.success_rate_50\">Det@50 (%)</th>
        <th data-key=\"metrics.detection.fps\">Det FPS</th>
        <th data-key=\"metrics.segmentation.dice_mean\">Seg Dice</th>
        <th data-key=\"metrics.segmentation.fps_mean\">Seg FPS</th>
        <th data-key=\"metrics.classification.accuracy\">Clf Acc (%)</th>
        <th data-key=\"metrics.classification.f1\">Clf F1</th>
        <th data-key=\"metrics.classification.precision\">Clf Prec</th>
        <th data-key=\"metrics.classification.recall\">Clf Rec</th>
        <th data-key=\"dataset.train_videos\">Train Videos</th>
        <th data-key=\"dataset.test_videos\">Test Videos</th>
        <th data-key="relative_path">Folder</th>
        <th>Note</th>
      </tr>
    </thead>
    <tbody></tbody>
  </table>
  <script>
    const data = {table_rows};
    const isServed = window.location.protocol !== 'file:';
    function formatDuration(value) {{
      if (!value && value !== 0) return '-';
      const total = Math.floor(value);
      const h = Math.floor(total / 3600);
      const m = Math.floor((total % 3600) / 60);
      const s = total % 60;
      if (h) {{ return `${{h}}:${{m.toString().padStart(2,'0')}}:${{s.toString().padStart(2,'0')}}`; }}
      return `${{m}}:${{s.toString().padStart(2,'0')}}`;
    }}
    function formatPercent(value) {{
      if (value === null || value === undefined) return '-';
      return `${{(value * 100).toFixed(1)}}%`;
    }}
    function formatFloat(value) {{
      if (value === null || value === undefined) return '-';
      return value.toFixed(3);
    }}
    function valueByKey(obj, key) {{
      return key.split('.').reduce((acc, part) => (acc && acc[part] !== undefined) ? acc[part] : undefined, obj);
    }}
    function makeNoteCell(td, row) {{
      const noteText = row.note || '';
      const span = document.createElement('span');
      span.className = 'note-text' + (noteText ? '' : ' empty');
      span.textContent = noteText || (isServed ? '點此新增備註…' : '(以 --serve 模式開啟以編輯備註)');
      td.appendChild(span);
      if (!isServed) return;
      span.addEventListener('click', () => {{
        const ta = document.createElement('textarea');
        ta.className = 'note-edit';
        ta.value = noteText;
        const st = document.createElement('span');
        st.className = 'note-status note-saving';
        td.innerHTML = '';
        td.appendChild(ta);
        td.appendChild(st);
        ta.focus();
        async function saveNote() {{
          const newText = ta.value;
          try {{
            st.textContent = '儲存中…';
            st.className = 'note-status note-saving';
            const resp = await fetch('/api/note', {{
              method: 'POST',
              headers: {{'Content-Type': 'application/json'}},
              body: JSON.stringify({{path: row.path, text: newText}})
            }});
            const d = await resp.json();
            if (d.ok) {{
              row.note = newText;
              st.textContent = '已儲存 ✓';
              st.className = 'note-status note-saved';
              setTimeout(() => {{ td.innerHTML = ''; makeNoteCell(td, row); }}, 900);
            }} else {{
              st.textContent = '儲存失敗';
              st.className = 'note-status note-error';
            }}
          }} catch(e) {{
            st.textContent = '錯誤: ' + e.message;
            st.className = 'note-status note-error';
          }}
        }}
        ta.addEventListener('blur', saveNote);
        ta.addEventListener('keydown', e => {{
          if (e.key === 'Escape') {{ td.innerHTML = ''; makeNoteCell(td, row); }}
        }});
      }});
    }}
    function render(rows) {{
      const tbody = document.querySelector('#exp-table tbody');
      tbody.innerHTML = '';
      rows.forEach(row => {{
        const tr = document.createElement('tr');
        const started = row.started_at ? new Date(row.started_at).toLocaleString() : '-';
        const duration = formatDuration(row.duration_sec);
        const det50 = formatPercent(valueByKey(row, 'metrics.detection.success_rate_50'));
        const detFps = formatFloat(valueByKey(row, 'metrics.detection.fps'));
        const segDice = formatFloat(valueByKey(row, 'metrics.segmentation.dice_mean'));
        const segFps = formatFloat(valueByKey(row, 'metrics.segmentation.fps_mean'));
        const clfAcc = formatPercent(valueByKey(row, 'metrics.classification.accuracy'));
        const clfF1 = formatFloat(valueByKey(row, 'metrics.classification.f1'));
        const clfPrec = formatFloat(valueByKey(row, 'metrics.classification.precision'));
        const clfRec = formatFloat(valueByKey(row, 'metrics.classification.recall'));
        const cells = [
          started,
          row.name || '-',
          row.status || '-',
          duration,
          det50,
          detFps,
          segDice,
          segFps,
          clfAcc,
          clfF1,
          clfPrec,
          clfRec,
          valueByKey(row, 'dataset.train_videos') ?? '-',
          valueByKey(row, 'dataset.test_videos') ?? '-',
          row.relative_path || '-',
        ];
        cells.forEach((value, idx) => {{
          const td = document.createElement('td');
          td.textContent = value;
          if (idx === 2) {{
            td.classList.add(`status-${{row.status || 'UNKNOWN'}}`);
          }}
          tr.appendChild(td);
        }});
        const noteTd = document.createElement('td');
        noteTd.className = 'note-cell';
        makeNoteCell(noteTd, row);
        tr.appendChild(noteTd);
        tbody.appendChild(tr);
      }});
    }}
    function sortBy(key) {{
      const sorted = [...data].sort((a, b) => {{
        const av = valueByKey(a, key);
        const bv = valueByKey(b, key);
        if (av === bv) return 0;
        if (av === null || av === undefined) return 1;
        if (bv === null || bv === undefined) return -1;
        if (typeof av === 'string' && typeof bv === 'string') {{
          return av.localeCompare(bv);
        }}
        return av > bv ? 1 : -1;
      }});
      render(sorted);
    }}
    document.querySelectorAll('#exp-table th').forEach(th => {{
      th.addEventListener('click', () => sortBy(th.dataset.key));
    }});
    render(data);
  </script>
</body>
</html>
"""


def _write_html(records: Sequence[Dict[str, Any]], destination: Path) -> None:
    destination.write_text(_generate_html_str(records), encoding="utf-8")


def _run_server(results_root: Path, args: argparse.Namespace, port: int) -> None:
    """Serve an interactive HTML dashboard with note editing support."""
    import json as _json
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):  # suppress default request log
            pass

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                recs = _collect_experiments(results_root)
                _sort_records(recs, args.sort, args.descending)
                html = _generate_html_str(recs)
                body = html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if self.path == "/api/note":
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length)
                try:
                    payload = _json.loads(raw)
                    exp_path = Path(payload["path"])
                    text = str(payload.get("text", ""))
                    _save_note(exp_path, text)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"ok":true}')
                except Exception as exc:
                    msg = f'{{"ok":false,"error":"{exc}"}}'.encode()
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(msg)
            else:
                self.send_response(404)
                self.end_headers()

    httpd = HTTPServer(("", port), _Handler)
    url = f"http://localhost:{port}/"
    print(f"Experiment Viewer  →  {url}  (Ctrl-C 停止)")
    try:
        import webbrowser
        webbrowser.open(url)
    except Exception:
        pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise tracking experiments using metadata.json outputs.")
    parser.add_argument("--results-root", type=Path, default=Path("results"), help="Directory containing experiment runs")
    parser.add_argument("--status", choices=["OK", "FAILED", "PARTIAL"], nargs="*", help="Filter by overall status")
    parser.add_argument("--sort", choices=["start", "duration", "det@50", "seg_dice", "clf_acc", "clf_f1"], default="start", help="Column to sort by")
    parser.add_argument("--descending", action="store_true", help="Sort in descending order")
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of rows displayed")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a text table")
    parser.add_argument("--html", type=Path, help="Write an interactive HTML dashboard to the given path")
    parser.add_argument("--stages", action="store_true", help="Print per-stage status details after the table")
    parser.add_argument(
        "--serve", type=int, metavar="PORT", nargs="?", const=8765,
        help="啟動本地 HTTP server（預設 port 8765）以開啟互動式 Note 編輯介面",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if getattr(args, "serve", None) is not None:
        _run_server(args.results_root, args, args.serve)
        return 0
    records = _collect_experiments(args.results_root)
    if not records:
        print(f"No experiments found under {args.results_root}")
        return 0

    if args.status:
        status_set = set(args.status)
        records = [rec for rec in records if rec.get("status") in status_set]
        if not records:
            print("No experiments match the requested status filter")
            return 0

    _sort_records(records, args.sort, args.descending)
    if args.limit and args.limit > 0:
        records = records[: args.limit]

    if args.html:
        _write_html(records, args.html)
        print(f"HTML dashboard written to {args.html}")

    if args.json:
        payload = _to_serialisable(records)
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        if not args.html:
            sys.stdout.write("\n")
    else:
        _print_table(records)
        if args.stages:
            _print_stage_details(records)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
