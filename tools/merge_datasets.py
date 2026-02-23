#!/usr/bin/env python
"""合併兩個超音波資料集 (有病組 + 正常組) 為一個統一的訓練用資料集。

使用方式:
    python tools/merge_datasets.py

此腳本會：
1. 將舊資料集 (超音波檔案, 有病組) 的 flat 結構轉換為 per-subject 資料夾
2. 將新資料集 (CTS_US_正常組) 的完整受試者複製過來 (跳過不完整的)
3. 更新所有 JSON 中的 mask_path 為相對於合併後 root 的路徑
4. 自動產生 ann.txt 標籤檔 (1=有病, 0=正常)
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ---- 設定 ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"

OLD_DATASET = DATASET_DIR / "超音波檔案-20251126T083302Z-1-001"
NEW_DATASET = DATASET_DIR / "CTS_US_正常組-20260223T045150Z-3-001" / "CTS_US_正常組"
MERGED_DIR = DATASET_DIR / "merged"

# 舊資料集受試者前綴正則  (e.g. "001Grasp.avi" → subject="001", action="Grasp")
_OLD_VIDEO_RE = re.compile(r"^(\d+)(.*)\.(avi|mp4|mov|mkv)$", re.IGNORECASE)

# 新資料集中需跳過的受試者 (不完整: 沒有 JSON / seg_masks)
SKIP_NEW_SUBJECTS = {"n002"}

# 疾病標籤
LABEL_DISEASED = 1
LABEL_HEALTHY = 0


def _parse_old_video(filename: str) -> Tuple[str, str] | None:
    """解析舊資料集的檔名, 返回 (subject_id, action_name) 或 None."""
    m = _OLD_VIDEO_RE.match(filename)
    if not m:
        return None
    subject = m.group(1)  # e.g. "001"
    action = m.group(2)   # e.g. "Grasp", "doppler", "Rest post"
    return subject, action


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(str(src), str(dst))


def _update_json_mask_paths(json_path: Path, subject_prefix: str) -> None:
    """更新 JSON 中所有 mask_path, 加入 subject 前綴使其相對於合併根目錄。"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    modified = False
    for ann in data.get("annotations", []):
        mp = ann.get("mask_path")
        if mp and not mp.startswith(subject_prefix):
            ann["mask_path"] = f"{subject_prefix}/{mp}"
            modified = True

    if modified:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def _copy_seg_masks(src_masks_dir: Path, dst_masks_dir: Path) -> int:
    """複製 seg_masks 目錄下的所有 PNG, 返回複製數量."""
    count = 0
    if not src_masks_dir.exists():
        return 0
    for item in src_masks_dir.rglob("*.png"):
        rel = item.relative_to(src_masks_dir)
        dst = dst_masks_dir / rel
        _copy_file(item, dst)
        count += 1
    return count


def merge_old_dataset() -> Dict[str, str]:
    """合併舊資料集 → 返回 {subject_id: "diseased"}."""
    subjects_found: Dict[str, List[str]] = {}  # subject → [action, ...]
    print(f"\n=== 處理舊資料集 (有病組): {OLD_DATASET} ===")

    if not OLD_DATASET.exists():
        print(f"  [警告] 舊資料集不存在: {OLD_DATASET}")
        return {}

    # 1. 收集所有影片並按 subject 分組
    for f in sorted(OLD_DATASET.iterdir()):
        if f.is_dir() or f.suffix.lower() not in {".avi", ".mp4", ".mov", ".mkv"}:
            continue
        parsed = _parse_old_video(f.name)
        if parsed is None:
            print(f"  [略過] 無法解析: {f.name}")
            continue
        subject, action = parsed
        subjects_found.setdefault(subject, []).append(action)

    # 2. 複製每個受試者的資料
    subject_labels = {}
    for subject in sorted(subjects_found.keys()):
        actions = subjects_found[subject]
        dst_subject_dir = MERGED_DIR / subject
        dst_subject_dir.mkdir(parents=True, exist_ok=True)

        # labels.txt
        labels_src = OLD_DATASET / "labels.txt"
        if labels_src.exists():
            _copy_file(labels_src, dst_subject_dir / "labels.txt")

        for action in actions:
            # 組合原始檔名 (e.g. "001Grasp")
            original_stem = f"{subject}{action}"

            # 複製影片
            for ext in [".avi", ".mp4", ".mov", ".mkv"]:
                src_video = OLD_DATASET / f"{original_stem}{ext}"
                if src_video.exists():
                    dst_video = dst_subject_dir / f"{action}{ext}"
                    _copy_file(src_video, dst_video)
                    break

            # 複製 JSON (動作名不含 subject prefix)
            src_json = OLD_DATASET / f"{original_stem}.json"
            dst_json = dst_subject_dir / f"{action}.json"
            if src_json.exists():
                _copy_file(src_json, dst_json)
                # 更新 mask_path: "seg_masks/001Grasp/..." → "001/seg_masks/Grasp/..."
                # 先需要更新 JSON 中的 mask_path
                _rewrite_old_json_mask_paths(dst_json, subject, original_stem, action)

            # 複製 seg_masks
            src_mask_dir = OLD_DATASET / "seg_masks" / original_stem
            dst_mask_dir = dst_subject_dir / "seg_masks" / action
            if src_mask_dir.exists():
                count = _copy_seg_masks(src_mask_dir, dst_mask_dir)
                print(f"  {subject}/{action}: 複製 {count} 個 mask 檔案")

        subject_labels[subject] = LABEL_DISEASED
        print(f"  受試者 {subject}: {len(actions)} 個動作 (有病)")

    return subject_labels


def _rewrite_old_json_mask_paths(
    json_path: Path, subject: str, original_stem: str, action: str,
) -> None:
    """重寫舊資料集 JSON 中的 mask_path。
    
    原始: seg_masks/001Grasp/frame_xxx.png
    目標: 001/seg_masks/Grasp/frame_xxx.png
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    modified = False
    for ann in data.get("annotations", []):
        mp = ann.get("mask_path")
        if not mp:
            continue
        # 替換 "seg_masks/<original_stem>/" → "<subject>/seg_masks/<action>/"
        old_prefix = f"seg_masks/{original_stem}/"
        if mp.startswith(old_prefix):
            frame_part = mp[len(old_prefix):]
            ann["mask_path"] = f"{subject}/seg_masks/{action}/{frame_part}"
            modified = True

    if modified:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def merge_new_dataset() -> Dict[str, int]:
    """合併新資料集 → 返回 {subject_id: label}."""
    print(f"\n=== 處理新資料集 (正常組): {NEW_DATASET} ===")
    subject_labels = {}

    if not NEW_DATASET.exists():
        print(f"  [警告] 新資料集不存在: {NEW_DATASET}")
        return {}

    for subject_dir in sorted(NEW_DATASET.iterdir()):
        if not subject_dir.is_dir():
            continue
        subject = subject_dir.name
        if subject in SKIP_NEW_SUBJECTS:
            print(f"  [略過] {subject}: 不完整 (無 JSON / masks)")
            continue

        # 檢查此受試者是否有 JSON 標註
        jsons = list(subject_dir.glob("*.json"))
        if not jsons:
            print(f"  [略過] {subject}: 無 JSON 標註")
            continue

        dst_subject_dir = MERGED_DIR / subject
        dst_subject_dir.mkdir(parents=True, exist_ok=True)

        video_count = 0
        for f in sorted(subject_dir.iterdir()):
            if f.is_dir():
                if f.name == "seg_masks":
                    # 複製 seg_masks
                    for mask_folder in f.iterdir():
                        if mask_folder.is_dir():
                            count = _copy_seg_masks(
                                mask_folder,
                                dst_subject_dir / "seg_masks" / mask_folder.name,
                            )
                continue

            if f.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv"}:
                _copy_file(f, dst_subject_dir / f.name)
                video_count += 1
            elif f.suffix.lower() == ".json":
                _copy_file(f, dst_subject_dir / f.name)
                # 更新 mask_path: "seg_masks/D/..." → "n001/seg_masks/D/..."
                _update_json_mask_paths(dst_subject_dir / f.name, subject)
            elif f.name == "labels.txt":
                _copy_file(f, dst_subject_dir / f.name)

        subject_labels[subject] = LABEL_HEALTHY
        print(f"  受試者 {subject}: {video_count} 個影片 (正常)")

    return subject_labels


def create_ann_txt(labels: Dict[str, int]) -> None:
    """產生 ann.txt 標籤檔案。"""
    ann_path = MERGED_DIR / "ann.txt"
    lines = []
    for subject in sorted(labels.keys()):
        lines.append(f"{subject} {labels[subject]}")

    with open(ann_path, "w", encoding="utf-8") as f:
        f.write("# Subject labels: 1=diseased (CTS), 0=healthy\n")
        f.write("\n".join(lines) + "\n")

    print(f"\n已產生 ann.txt ({len(labels)} 個受試者):")
    for line in lines:
        print(f"  {line}")


def create_labels_txt() -> None:
    """產生 labels.txt (類別名稱)."""
    labels_path = MERGED_DIR / "labels.txt"
    with open(labels_path, "w", encoding="utf-8") as f:
        f.write("median_nerve\n")


def main():
    print(f"合併目標: {MERGED_DIR}")

    if MERGED_DIR.exists():
        print(f"\n[警告] 目標目錄已存在: {MERGED_DIR}")
        resp = input("是否清除並重新合併? (y/N): ").strip().lower()
        if resp != "y":
            print("取消合併。")
            return
        shutil.rmtree(str(MERGED_DIR))

    MERGED_DIR.mkdir(parents=True, exist_ok=True)

    # 合併兩個資料集
    old_labels = merge_old_dataset()
    new_labels = merge_new_dataset()

    # 合併標籤
    all_labels = {}
    all_labels.update(old_labels)
    all_labels.update(new_labels)

    # 產生標籤檔案
    create_ann_txt(all_labels)
    create_labels_txt()

    # 統計
    diseased = sum(1 for v in all_labels.values() if v == LABEL_DISEASED)
    healthy = sum(1 for v in all_labels.values() if v == LABEL_HEALTHY)
    print(f"\n=== 合併完成 ===")
    print(f"  有病組: {diseased} 位受試者")
    print(f"  正常組: {healthy} 位受試者")
    print(f"  合計: {len(all_labels)} 位受試者")
    print(f"  輸出路徑: {MERGED_DIR}")


if __name__ == "__main__":
    main()
