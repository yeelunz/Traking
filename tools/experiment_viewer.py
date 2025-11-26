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
            },
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
        Column("Videos", 11, lambda r: _format_ratio(
            r.get("dataset", {}).get("train_videos"),
            r.get("dataset", {}).get("test_videos"),
        )),
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


def _write_html(records: Sequence[Dict[str, Any]], destination: Path) -> None:
    data = _to_serialisable(records)
    table_rows = json.dumps(data, indent=2)
    html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
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
        <th data-key=\"dataset.train_videos\">Train Videos</th>
        <th data-key=\"dataset.test_videos\">Test Videos</th>
        <th data-key=\"relative_path\">Folder</th>
      </tr>
    </thead>
    <tbody></tbody>
  </table>
  <script>
    const data = {table_rows};
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
        const cells = [
          started,
          row.name || '-',
          row.status || '-',
          duration,
          det50,
          detFps,
          segDice,
          segFps,
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
    destination.write_text(html, encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise tracking experiments using metadata.json outputs.")
    parser.add_argument("--results-root", type=Path, default=Path("results"), help="Directory containing experiment runs")
    parser.add_argument("--status", choices=["OK", "FAILED", "PARTIAL"], nargs="*", help="Filter by overall status")
    parser.add_argument("--sort", choices=["start", "duration", "det@50", "seg_dice"], default="start", help="Column to sort by")
    parser.add_argument("--descending", action="store_true", help="Sort in descending order")
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of rows displayed")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a text table")
    parser.add_argument("--html", type=Path, help="Write an interactive HTML dashboard to the given path")
    parser.add_argument("--stages", action="store_true", help="Print per-stage status details after the table")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
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
