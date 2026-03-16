"""
debug_clf_replay.py — 最小分類重現腳本

在不重新訓練偵測器的情況下，直接使用已儲存的偵測預測結果
重現 classification 階段，用以取得完整 traceback。

執行方式:
    python tools/debug_clf_replay.py \
        --result_dir results/2026-03-06_15-10-16_schedule_4exp/2026-03-06_15-10-16_mednext_clahe_global_mts_rf \
        --schedule schedules/mednext_noloso_clahe_clf3.yaml \
        --queue_label 01_mednext_mts_rf
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml


def _load_predictions_from_dir(predictions_dir: str) -> Dict[str, List]:
    """從 predictions_by_video 目錄載入 FramePrediction 列表。
    支援舊版 (predictions_by_video/{vid_stem}/*.json)
    和新版 (predictions_by_video/{subject_id}/{vid_stem}/*.json) 兩種目錄結構。"""
    from tracking.core.interfaces import FramePrediction

    predictions: Dict[str, List[FramePrediction]] = {}
    if not os.path.isdir(predictions_dir):
        raise FileNotFoundError(f"prediction dir not found: {predictions_dir}")

    def _load_vid_dir(vid_stem: str, vid_dir: str):
        for json_file in os.listdir(vid_dir):
            if not json_file.endswith(".json"):
                continue
            model_name = json_file.replace(".json", "")
            full_path = os.path.join(vid_dir, json_file)
            with open(full_path, encoding="utf-8") as f:
                raw = json.load(f)
            fps: List[FramePrediction] = []
            for entry in raw:
                try:
                    fps.append(
                        FramePrediction(
                            frame_index=int(entry["frame_index"]),
                            bbox=tuple(float(x) for x in entry["bbox"]),
                            score=float(entry.get("score", 1.0)),
                        )
                    )
                except Exception as exc:
                    print(f"  WARN: skip frame entry {entry}: {exc}")
            if not predictions.get(model_name):
                predictions[model_name] = {}
            if fps:
                predictions[model_name][vid_stem] = fps

    for entry1 in os.listdir(predictions_dir):
        path1 = os.path.join(predictions_dir, entry1)
        if not os.path.isdir(path1):
            continue
        sub_entries = os.listdir(path1)
        has_json = any(e.endswith(".json") for e in sub_entries)
        if has_json:
            # Legacy layout: {vid_stem}/{model}.json
            _load_vid_dir(entry1, path1)
        else:
            # New layout: {subject_id}/{vid_stem}/{model}.json
            for entry2 in sub_entries:
                path2 = os.path.join(path1, entry2)
                if os.path.isdir(path2):
                    _load_vid_dir(entry2, path2)

    return predictions


def _resolve_video_paths(
    model_preds: Dict[str, List],
    dataset_root: str,
) -> Dict[str, List]:
    """將 vid_stem → full video path。"""
    import glob
    result = {}
    for vid_stem, fps in model_preds.items():
        pattern = os.path.join(dataset_root, "**", vid_stem + ".*")
        matches = glob.glob(pattern, recursive=True)
        if not matches:
            print(f"  WARN: cannot find video for stem '{vid_stem}' in {dataset_root}")
            result[vid_stem] = fps  # keep as stem
        else:
            result[matches[0]] = fps  # use first match
    return result


class SimpleDataset:
    def __init__(self, video_paths: List[str], ann_by_video: dict):
        self._items = []
        for vp in video_paths:
            self._items.append({"video_path": vp, "annotation": ann_by_video.get(vp)})

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


def main():
    parser = argparse.ArgumentParser(description="Replay classification stage for debugging")
    parser.add_argument("--result_dir", required=True, help="Experiment result directory")
    parser.add_argument("--schedule", required=True, help="Schedule YAML file")
    parser.add_argument("--queue_label", required=True, help="Queue item label (e.g. 01_mednext_mts_rf)")
    parser.add_argument("--dataset_root", default=None, help="Dataset root (auto from metadata if not given)")
    args = parser.parse_args()

    result_dir = args.result_dir
    print(f"Result dir: {result_dir}")

    # --- Load metadata ---
    meta_path = os.path.join(result_dir, "metadata.json")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    dataset_root = args.dataset_root or meta.get("dataset", {}).get("root")
    print(f"Dataset root: {dataset_root}")

    # --- Load schedule ---
    with open(args.schedule, encoding="utf-8") as f:
        schedule = yaml.safe_load(f)

    # Find queue item
    clf_cfg = None
    for item in schedule.get("queue", []):
        if item.get("label") == args.queue_label:
            clf_cfg = item.get("config", {}).get("classification")
            break
    if clf_cfg is None:
        print(f"ERROR: queue_label '{args.queue_label}' not found or has no 'classification' key")
        sys.exit(1)

    print(f"Classification config: {json.dumps(clf_cfg, indent=2, ensure_ascii=False)}")

    import copy
    clf_cfg = copy.deepcopy(clf_cfg)
    # Ensure segmentation is enabled (mirrors original run)
    if "segmentation" not in clf_cfg:
        clf_cfg["segmentation"] = {"enabled": True, "params": {}}
        print("NOTE: Added default segmentation config (enabled=True, params={})")

    # --- Load detector predictions ---
    pred_root = os.path.join(result_dir, "test", "detection", "predictions_by_video")
    print(f"\nLoading predictions from: {pred_root}")

    from tracking.core.interfaces import FramePrediction

    # Build test_predictions dict: {model_name: {video_path: [FramePrediction]}}
    test_predictions: Dict[str, Dict[str, List[FramePrediction]]] = {}

    # Walk predictions_by_video/ — supports both legacy (vid_stem/) and new (subject_id/vid_stem/) layouts
    import glob as _glob
    for entry1 in os.listdir(pred_root):
        path1 = os.path.join(pred_root, entry1)
        if not os.path.isdir(path1):
            continue
        # Detect layout: if path1 contains JSON files directly → legacy (entry1 = vid_stem)
        # If path1 contains subdirectories → new layout (entry1 = subject_id)
        sub_entries = os.listdir(path1)
        has_json = any(e.endswith(".json") for e in sub_entries)
        if has_json:
            # Legacy layout: predictions_by_video/{vid_stem}/{model}.json
            level2_dirs = [(entry1, path1)]
        else:
            # New layout: predictions_by_video/{subject_id}/{vid_stem}/{model}.json
            level2_dirs = [(e, os.path.join(path1, e)) for e in sub_entries if os.path.isdir(os.path.join(path1, e))]
        for vid_stem, vid_dir_path in level2_dirs:
            for json_fname in os.listdir(vid_dir_path):
                if not json_fname.endswith(".json"):
                    continue
                model_name = json_fname.replace(".json", "")
                json_fpath = os.path.join(vid_dir_path, json_fname)
                with open(json_fpath, encoding="utf-8") as ff:
                    raw = json.load(ff)
                fps = []
                for entry in raw:
                    fps.append(
                        FramePrediction(
                            frame_index=int(entry["frame_index"]),
                            bbox=tuple(float(x) for x in entry["bbox"]),
                            score=float(entry.get("score", 1.0)),
                        )
                    )
                if model_name not in test_predictions:
                    test_predictions[model_name] = {}

                # Resolve full video path
                pattern = os.path.join(dataset_root, "**", vid_stem + ".*")
                matches = _glob.glob(pattern, recursive=True)
                if matches:
                    test_predictions[model_name][matches[0]] = fps
                else:
                    print(f"  WARN: no video file found for stem '{vid_stem}'")
                    test_predictions[model_name][vid_stem] = fps

    print(f"Models found in predictions: {list(test_predictions.keys())}")
    for model_name, vpreds in test_predictions.items():
        print(f"  {model_name}: {len(vpreds)} test videos")
        for vp, fps in vpreds.items():
            print(f"    {os.path.basename(vp)}: {len(fps)} frames")

    # --- Build train dataset from metadata ---
    # Get train videos from dataset split info in metadata
    dataset_info = meta.get("dataset", {})
    train_preview = dataset_info.get("train_preview", [])
    print(f"\nTrain preview ({len(train_preview)}): {train_preview[:5]}")

    # Use COCOJsonDatasetManager to get all videos
    from tracking.data.dataset_manager import COCOJsonDatasetManager
    dm = COCOJsonDatasetManager(dataset_root)
    print(f"Total dataset videos: {len(dm.videos)}")
    print(f"Annotated videos: {len(dm.ann_by_video)}")

    # Reconstruct split
    split_cfg = schedule.get("queue", [{}])[0].get("config", {}).get("dataset", {}).get("split", {})
    # Find the correct queue item's split
    for item in schedule.get("queue", []):
        if item.get("label") == args.queue_label:
            split_cfg_candidate = item.get("config", {}).get("dataset", {}).get("split", {})
            if split_cfg_candidate:
                split_cfg = split_cfg_candidate
                break
    # If not found, use the 00_defs split
    if not split_cfg:
        for item in schedule.get("queue", []):
            split_cfg = (item.get("config") or {}).get("dataset", {}).get("split", {})
            if split_cfg:
                break

    print(f"\nSplit config: {split_cfg}")
    method = split_cfg.get("method", "video_level")
    ratios = split_cfg.get("ratios", [0.8, 0.2])
    seed = 42
    import random

    if method == "subject_level":
        split_tt = dm.split(method="subject_level", seed=seed, ratios=(ratios[0], 0.0, ratios[1]))
    else:
        split_tt = dm.split(method=method, seed=seed, ratios=(ratios[0], 0.0, ratios[1]))

    train_ds = split_tt["train"]
    test_ds = split_tt["test"]
    print(f"Train videos: {len(train_ds)}, Test videos: {len(test_ds)}")

    # --- Run classification ---
    from tracking.classification.engine import run_subject_classification

    print("\n" + "=" * 60)
    print("Running classification...")
    print("=" * 60)

    def logger(msg: str) -> None:
        print(msg)

    try:
        result = run_subject_classification(
            clf_cfg,
            dataset_root,
            train_ds,
            test_predictions,
            os.path.join(result_dir, "_debug_clf"),
            logger,
            split_method=method,
        )
        print(f"\nClassification SUCCEEDED: {result}")
    except Exception as exc:
        print(f"\n[ERROR] Classification FAILED: {exc}")
        print("\nFull traceback:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
