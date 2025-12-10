from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import re
from urllib.parse import quote, urlencode

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


def load_json(path: Path) -> Optional[Dict]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def first_value(obj: Dict) -> Optional[Dict]:
    if not isinstance(obj, dict):
        return None
    for value in obj.values():
        if isinstance(value, dict):
            return value
    return None


def detection_summary_metrics(exp_path: Path) -> Optional[Dict]:
    summary_path = exp_path / "test" / "detection" / "metrics" / "summary.json"
    if not summary_path.exists():
        return None
    data = load_json(summary_path)
    return first_value(data) if data else None


def segmentation_summary_metrics(exp_path: Path) -> Optional[Dict]:
    summary_path = exp_path / "test" / "segmentation" / "metrics_summary.json"
    if not summary_path.exists():
        return None
    data = load_json(summary_path)
    if not data:
        return None
    first_model = first_value(data)
    return first_value(first_model) if first_model else None


class ExperimentIndex:
    def __init__(self, root: Path):
        self.root = root.resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"Results root not found: {self.root}")
        self.entries: Dict[str, Dict] = {}
        self.refresh()

    def _experiment_dirs(self) -> List[Path]:
        """Find every directory under results root that contains an experiment metadata.json."""
        experiment_dirs: List[Path] = []
        seen: set[Path] = set()
        for meta_path in sorted(self.root.rglob("metadata.json")):
            exp_path = meta_path.parent
            try:
                exp_path.relative_to(self.root)
            except ValueError:
                continue
            if exp_path in seen:
                continue
            seen.add(exp_path)
            experiment_dirs.append(exp_path)
        return experiment_dirs

    def refresh(self) -> None:
        entries: Dict[str, Dict] = {}
        for exp_path in self._experiment_dirs():
            meta_path = exp_path / "metadata.json"
            meta = load_json(meta_path)
            if not meta or not isinstance(meta, dict):
                continue
            experiment_meta = meta.get("experiment") or {}
            pipeline = experiment_meta.get("pipeline") or []
            if not pipeline:
                pipeline = meta.get("config", {}).get("pipeline", [])
            preprocs = [step.get("name") for step in pipeline if step.get("type") == "preproc"]
            models = [step.get("name") for step in pipeline if step.get("type") == "model"]
            rel = self._relative_label(exp_path)
            group_path = self._group_path(rel)
            det_summary = exp_path / "test" / "detection" / "metrics" / "summary.json"
            seg_summary = exp_path / "test" / "segmentation" / "metrics_summary.json"
            det_preview = detection_summary_metrics(exp_path)
            seg_preview = segmentation_summary_metrics(exp_path)
            entry = {
                "id": rel,
                "path": exp_path,
                "relative_path": rel,
                "group_path": group_path,
                "name": experiment_meta.get("name") or exp_path.name,
                "created_at": meta.get("created_at"),
                "preprocs": preprocs,
                "models": models,
                "has_detection": det_summary.exists(),
                "has_segmentation": seg_summary.exists(),
                "has_detection_visuals": (exp_path / "test" / "detection" / "visualizations").exists(),
                "has_segmentation_visuals": bool(list((exp_path / "test" / "segmentation").rglob("visualizations_roi"))) if (exp_path / "test" / "segmentation").exists() else False,
                "preview": {
                    "detection": det_preview,
                    "segmentation": seg_preview,
                },
            }
            entries[rel] = entry
        self.entries = entries

    def _relative_label(self, path: Path) -> str:
        try:
            rel = path.relative_to(self.root)
            label = rel.as_posix()
            return label if label else "root"
        except ValueError:
            return path.name

    def _group_path(self, rel: str) -> str:
        if rel in {"", "root"}:
            return ""
        parts = rel.split("/")
        if len(parts) <= 1:
            return ""
        return "/".join(parts[:-1])

    def list_entries(self) -> List[Dict]:
        public_entries = [self._public_entry(e) for e in self.entries.values()]
        return sorted(public_entries, key=lambda item: item.get("created_at") or "", reverse=True)

    def _public_entry(self, entry: Dict) -> Dict:
        return {
            "id": entry["id"],
            "name": entry["name"],
            "relative_path": entry["relative_path"],
            "group_path": entry.get("group_path", ""),
            "created_at": entry["created_at"],
            "preprocs": entry["preprocs"],
            "models": entry["models"],
            "has_detection": entry["has_detection"],
            "has_segmentation": entry["has_segmentation"],
            "has_detection_visuals": entry["has_detection_visuals"],
            "has_segmentation_visuals": entry["has_segmentation_visuals"],
            "preview": entry.get("preview", {}),
        }

    def get_path(self, exp_id: str) -> Path:
        entry = self.entries.get(exp_id)
        if not entry:
            raise KeyError(exp_id)
        return Path(entry["path"])


def gather_detection_metrics(exp_path: Path) -> Dict:
    summary_path = exp_path / "test" / "detection" / "metrics" / "summary.json"
    per_video_root = summary_path.parent
    summary = load_json(summary_path)
    summary_metrics = first_value(summary) if summary else None

    per_video: List[Dict] = []
    if per_video_root.exists():
        for item in sorted(per_video_root.iterdir()):
            if not item.is_dir():
                continue
            summary_file = item / "summary.json"
            if not summary_file.exists():
                continue
            data = load_json(summary_file)
            metrics = first_value(data) if data else None
            if metrics:
                per_video.append({
                    "video": item.name,
                    "metrics": metrics,
                })
    return {
        "summary": summary_metrics,
        "per_video": per_video,
    }


def gather_segmentation_metrics(exp_path: Path) -> Dict:
    summary_path = exp_path / "test" / "segmentation" / "metrics_summary.json"
    summary = load_json(summary_path)
    summary_metrics = None
    if summary:
        first_model = first_value(summary)
        summary_metrics = first_value(first_model) if first_model else None

    per_video: List[Dict] = []
    preds_root = exp_path / "test" / "segmentation" / "predictions"
    metrics_file = None
    if preds_root.exists():
        for model_dir in preds_root.iterdir():
            if not model_dir.is_dir():
                continue
            for detector_dir in model_dir.iterdir():
                candidate = detector_dir / "metrics_per_video.json"
                if candidate.exists():
                    metrics_file = candidate
                    break
            if metrics_file:
                break
    if metrics_file:
        data = load_json(metrics_file) or {}
        for video_path, metrics in data.items():
            label = os.path.basename(video_path)
            per_video.append({
                "video": label,
                "metrics": metrics,
            })
    return {
        "summary": summary_metrics,
        "per_video": per_video,
    }


def normalize_media_items(exp_path: Path, files: List[Path], exp_id: str) -> List[Dict]:
    items = []
    for file_path in files:
        rel = file_path.relative_to(exp_path)
        rel_posix = rel.as_posix()
        params = urlencode({"exp_id": exp_id, "resource": rel_posix})
        url = f"/media?{params}"
        items.append({
            "label": rel.as_posix(),
            "url": url,
        })
    return items


def list_images(
    exp_path: Path,
    pattern_root: Path,
    patterns: Sequence[str] | str,
    limit: Optional[int],
    exp_id: str,
) -> List[Dict]:
    """Return media items under pattern_root matching patterns.

    limit=None means "no limit" so the UI can show every test image when desired.
    """
    if not pattern_root.exists():
        return []
    if isinstance(patterns, str):
        patterns = [patterns]
    collected: set[Path] = set()
    for pattern in patterns:
        for file_path in pattern_root.rglob(pattern):
            if file_path.is_file():
                collected.add(file_path)
    files = sorted(collected)
    if limit is not None:
        files = files[:limit]
    return normalize_media_items(exp_path, files, exp_id)


def create_app(results_root: Path) -> FastAPI:
    index = ExperimentIndex(results_root)
    app = FastAPI(title="Schedule Results Viewer", version="1.0.0")
    app.index = index  # type: ignore[attr-defined]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"]
,
        allow_headers=["*"],
    )

    static_dir = Path(__file__).parent / "static"
    assets_dir = static_dir
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/api/experiments")
    def list_experiments():
        return {"experiments": index.list_entries(), "root": str(index.root)}

    @app.post("/api/experiments/refresh")
    def refresh_index():
        index.refresh()
        return {"count": len(index.entries)}

    def _experiment_payload(exp_id: str):
        try:
            exp_path = index.get_path(exp_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Experiment not found")
        meta = load_json(exp_path / "metadata.json") or {}
        dataset_info = meta.get("dataset")
        experiment_info = meta.get("experiment") or {}
        detection = gather_detection_metrics(exp_path) if (exp_path / "test" / "detection").exists() else None
        segmentation = gather_segmentation_metrics(exp_path) if (exp_path / "test" / "segmentation").exists() else None
        return {
            "id": exp_id,
            "experiment": {
                "name": experiment_info.get("name"),
                "output_dir": experiment_info.get("output_dir"),
                "pipeline": experiment_info.get("pipeline"),
                "created_at": meta.get("created_at"),
            },
            "dataset": dataset_info,
            "detection": detection,
            "segmentation": segmentation,
        }

    @app.get("/api/experiments/{exp_id:path}/metrics")
    def experiment_metrics_path(exp_id: str):
        return _experiment_payload(exp_id)

    @app.get("/api/experiments/metrics")
    def experiment_metrics_query(exp_id: str = Query(..., description="Relative experiment id")):
        return _experiment_payload(exp_id)

    @app.get("/api/experiments/{exp_id:path}/visuals")
    def experiment_visuals_path(
        exp_id: str,
        category: str = Query(...),
        limit: Optional[int] = Query(None, ge=1, le=10000, description="Max items to return; leave empty for all"),
    ):
        return _experiment_visuals(exp_id, category, limit)

    @app.get("/api/experiments/visuals")
    def experiment_visuals_query(
        exp_id: str = Query(..., description="Relative experiment id"),
        category: str = Query(...),
        limit: Optional[int] = Query(None, ge=1, le=10000, description="Max items to return; leave empty for all"),
    ):
        return _experiment_visuals(exp_id, category, limit)

    def _experiment_visuals(exp_id: str, category: str, limit: Optional[int]):
        try:
            exp_path = index.get_path(exp_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Experiment not found")
        category = category.lower()
        items: List[Dict] = []
        per_frame_cache: Dict[Path, Dict] = {}

        if category == "detection_visualizations":
            base = exp_path / "test" / "detection" / "visualizations"
            items = list_images(exp_path, base, ["*.png", "*.jpg", "*.jpeg", "*.webp"], limit, exp_id)
            # Attach per-video center error (pixels) if available
            det_metrics = gather_detection_metrics(exp_path)
            ce_map: Dict[str, float] = {}
            for entry in det_metrics.get("per_video", []):
                video = entry.get("video")
                metrics = entry.get("metrics") or {}
                ce_val = metrics.get("ce_mean") if isinstance(metrics, dict) else None
                if video is not None and ce_val is not None:
                    video_name = os.path.basename(str(video))
                    ce_map[str(video_name)] = ce_val
                    ce_map[Path(video_name).stem] = ce_val
            for item in items:
                parts = Path(item.get("label", "")).parts
                if "visualizations" in parts:
                    idx = parts.index("visualizations")
                    if idx + 1 < len(parts):
                        video_name = parts[idx + 1]
                        # Try per-frame metrics first
                        metrics_dir = (exp_path / "test" / "detection" / "metrics" / video_name)
                        per_frame_metrics = None
                        if metrics_dir.exists():
                            per_frame_metrics = per_frame_cache.get(metrics_dir)
                            if per_frame_metrics is None:
                                per_frame_metrics = {}
                                candidates = []
                                candidates.extend(sorted(metrics_dir.glob("*_per_frame.json")))
                                candidates.extend(sorted(metrics_dir.glob("metrics_per_frame.json")))
                                candidates.extend(sorted(metrics_dir.glob("per_frame.json")))
                                for path in candidates:
                                    data = load_json(path) or {}
                                    if data:
                                        per_frame_metrics = data
                                        break
                                # Fallback to CSV if JSON is missing (legacy runs)
                                if not per_frame_metrics:
                                    csv_candidates = sorted(metrics_dir.glob("*_per_frame.csv"))
                                    for csv_path in csv_candidates:
                                        try:
                                            with csv_path.open("r", encoding="utf-8") as fh:
                                                reader = csv.reader(fh)
                                                header = next(reader, None)
                                                frame_idx_col = 0
                                                ce_col = 2 if header and len(header) > 2 else 1
                                                metrics_map = {}
                                                for row in reader:
                                                    if not row or row[0] in {"frame_index", None, ""}:
                                                        continue
                                                    try:
                                                        fi = int(row[frame_idx_col])
                                                        ce_val = float(row[ce_col]) if len(row) > ce_col and row[ce_col] not in {None, ""} else None
                                                    except Exception:
                                                        continue
                                                    metrics_map[str(fi)] = {"ce": ce_val}
                                                if metrics_map:
                                                    per_frame_metrics = metrics_map
                                                    break
                                        except Exception:
                                            continue
                                per_frame_cache[metrics_dir] = per_frame_metrics
                        stem = Path(parts[-1]).stem
                        frame_idx = None
                        if stem.startswith("frame_"):
                            try:
                                frame_idx = int(stem.split("_", 1)[1])
                            except Exception:
                                frame_idx = None
                        if frame_idx is None:
                            match = re.search(r"(\d+)$", stem)
                            if match:
                                try:
                                    frame_idx = int(match.group(1))
                                except Exception:
                                    frame_idx = None
                        ce_val = None
                        if per_frame_metrics and frame_idx is not None:
                            frame_entry = per_frame_metrics.get(str(frame_idx))
                            if isinstance(frame_entry, dict):
                                ce_val = frame_entry.get("ce")
                        if ce_val is None:
                            ce_val = ce_map.get(video_name)
                        if ce_val is not None:
                            item["ce_px"] = ce_val

        elif category == "detection_metrics":
            base = exp_path / "test" / "detection" / "metrics"
            items = list_images(exp_path, base, "*.png", limit, exp_id)

        elif category in {"segmentation_overlays", "segmentation_errors"}:
            base = exp_path / "test" / "segmentation"
            pattern = "*overlay.png" if category == "segmentation_overlays" else "*error.png"
            items = list_images(exp_path, base, pattern, limit, exp_id)
            # Attach centroid error (pixels) per video and per frame if available
            seg_metrics = gather_segmentation_metrics(exp_path)
            ce_map: Dict[str, float] = {}
            for entry in seg_metrics.get("per_video", []):
                video = entry.get("video")
                metrics = entry.get("metrics") or {}
                ce_val = metrics.get("centroid_mean") if isinstance(metrics, dict) else None
                if video is not None and ce_val is not None:
                    # map both basename and stem for robustness
                    video_name = os.path.basename(str(video))
                    ce_map[video_name] = ce_val
                    ce_map[Path(video_name).stem] = ce_val
            per_frame_cache: Dict[Path, Dict[str, Dict]] = {}
            for item in items:
                label = item.get("label", "")
                path_obj = Path(label)
                stem = path_obj.stem
                video_dir = None
                metrics_for_video: Optional[Dict[str, Dict]] = None
                parts = path_obj.parts
                if "visualizations_roi" in parts:
                    idx = parts.index("visualizations_roi")
                    if idx >= 1:
                        video_dir = parts[idx - 1]
                        video_dir_path = exp_path / Path(*parts[: idx])  # path up to video dir
                        metrics_path = video_dir_path / "metrics_per_frame.json"
                        if metrics_path.exists():
                            metrics_for_video = per_frame_cache.get(metrics_path)
                            if metrics_for_video is None:
                                metrics_for_video = load_json(metrics_path) or {}
                                per_frame_cache[metrics_path] = metrics_for_video
                # strip common suffixes
                for suffix in ["_overlay", "_error"]:
                    if stem.endswith(suffix):
                        stem = stem[: -len(suffix)]
                        break
                # Try per-frame centroid error if available
                frame_idx = None
                if stem.startswith("frame_"):
                    try:
                        frame_idx = int(stem.split("_", 1)[1])
                    except Exception:
                        frame_idx = None
                if frame_idx is None:
                    match = re.search(r"(\d+)$", stem)
                    if match:
                        try:
                            frame_idx = int(match.group(1))
                        except Exception:
                            frame_idx = None
                ce_val = None
                if metrics_for_video and frame_idx is not None:
                    frame_entry = metrics_for_video.get(str(frame_idx))
                    if isinstance(frame_entry, dict):
                        ce_val = frame_entry.get("centroid")
                if ce_val is None:
                    ce_val = ce_map.get(video_dir) or ce_map.get(stem) or ce_map.get(path_obj.name)
                if ce_val is not None:
                    item["ce_px"] = ce_val

        else:
            raise HTTPException(status_code=400, detail="Unknown category")

        return {"items": items}

    @app.get("/media")
    def media(exp_id: str = Query(...), resource: str = Query(...)):
        try:
            exp_path = index.get_path(exp_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Experiment not found")
        target = (exp_path / resource).resolve()
        if not str(target).startswith(str(exp_path)) or not target.exists():
            raise HTTPException(status_code=404, detail="Resource not found")
        return FileResponse(target)

    @app.get("/")
    def index_page():
        index_html = static_dir / "index.html"
        if not index_html.exists():
            raise HTTPException(status_code=500, detail="index.html missing")
        return FileResponse(index_html)

    return app


def main():
    parser = argparse.ArgumentParser(description="Interactive schedule results viewer")
    parser.add_argument("--results", default="results", help="Path to schedule root or single experiment folder")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (for development)")
    args = parser.parse_args()
    results_root = Path(args.results)
    app = create_app(results_root)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
