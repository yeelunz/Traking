#!/usr/bin/env python
"""將 dataset/2026_深度1.5/ 的資料合併到 dataset/merged_extend/。

結構說明：
  2026_深度1.5/c1_Image05_伸握伸.wmv   → merged_extend/c1/Image05_伸握伸.wmv
  2026_深度1.5/c1_Image05_伸握伸.json  → merged_extend/c1/Image05_伸握伸.json  (mask_path 更新)
  2026_深度1.5/seg_masks/c1_Image05_伸握伸/ → merged_extend/c1/seg_masks/Image05_伸握伸/

受試者標籤：
  c1 = 1  (女性, CTS 有症狀)
  c3 = 0  (女性, 正常)
  c5 = 0  (女性, 正常)

執行方式：
    python tools/merge_2026_dep15.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
SOURCE_DIR = DATASET_DIR / "2026_深度1.5"
TARGET_DIR = DATASET_DIR / "merged_extend"

# 從檔名解析 subject 前綴與 video stem
# e.g. "c1_Image05_伸握伸.wmv" → subject="c1", stem="Image05_伸握伸"
_FNAME_RE = re.compile(r"^(c\d+)_(.+)\.(wmv|avi|mp4|mov|mkv)$", re.IGNORECASE)

# 受試者標籤對照表
SUBJECT_LABELS: dict[str, int] = {
    "c1": 1,   # 女性 CTS 有症狀
    "c3": 0,   # 女性正常
    "c5": 0,   # 女性正常
}

# 受試者備註 (寫入 ann.txt)
SUBJECT_NOTES: dict[str, str] = {
    "c1": "female CTS",
    "c3": "female normal",
    "c5": "female normal",
}


def _copy(src: Path, dst: Path) -> bool:
    """複製檔案；父目錄不存在時自動建立。回傳是否有實際複製。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return False
    shutil.copy2(str(src), str(dst))
    return True


def _process_json(
    src_json: Path,
    dst_json: Path,
    subject: str,
    orig_stem: str,   # e.g. "c1_Image05_伸握伸"
    video_stem: str,  # e.g. "Image05_伸握伸"
    dry_run: bool,
) -> int:
    """複製 JSON 並更新 mask_path / video name。回傳更新的 annotation 數量。"""
    with open(src_json, encoding="utf-8") as f:
        data = json.load(f)

    # 更新 videos[] 的 name / file_name 欄位 (去掉 subject 前綴)
    for vid in data.get("videos", []):
        for key in ("name", "file_name"):
            val: str = vid.get(key, "")
            if val.startswith(f"{subject}_"):
                vid[key] = val[len(subject) + 1:]

    # 更新 annotations[].mask_path
    #   原始: "seg_masks/c1_Image05_伸握伸/frame_00000002.png"
    #   目標: "seg_masks/Image05_伸握伸/frame_00000002.png"  (相對於 c1/ 資料夾)
    old_prefix = f"seg_masks/{orig_stem}/"
    new_prefix = f"seg_masks/{video_stem}/"
    updated = 0
    for ann in data.get("annotations", []):
        mp: str | None = ann.get("mask_path")
        if mp and mp.startswith(old_prefix):
            ann["mask_path"] = new_prefix + mp[len(old_prefix):]
            updated += 1

    if not dry_run:
        dst_json.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    return updated


def _copy_seg_masks(
    src_masks_dir: Path,  # e.g. 2026_深度1.5/seg_masks/c1_Image05_伸握伸/
    dst_masks_dir: Path,  # e.g. merged_extend/c1/seg_masks/Image05_伸握伸/
    dry_run: bool,
) -> int:
    """複製 seg_masks 子目錄下的 PNG。回傳複製數量。"""
    count = 0
    if not src_masks_dir.exists():
        print(f"  [警告] 遮罩來源不存在: {src_masks_dir}")
        return 0
    for png in sorted(src_masks_dir.glob("*.png")):
        if dry_run:
            count += 1
        else:
            dst = dst_masks_dir / png.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(str(png), str(dst))
            count += 1
    return count


def _update_ann_txt(
    target_dir: Path,
    new_labels: dict[str, int],
    notes: dict[str, str],
    dry_run: bool,
) -> None:
    """在 ann.txt 末尾追加新受試者標籤 (若尚未存在)。"""
    ann_path = target_dir / "ann.txt"
    existing: set[str] = set()

    if ann_path.exists():
        with open(ann_path, encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    existing.add(stripped.split()[0])

    to_add: list[str] = []
    for subj, label in sorted(new_labels.items()):
        if subj not in existing:
            note = notes.get(subj, "")
            to_add.append(f"{subj} {label}  # {note}\n" if note else f"{subj} {label}\n")

    if not to_add:
        print("  ann.txt 已包含所有新受試者，無需更新。")
        return

    if dry_run:
        print("  [dry-run] 將在 ann.txt 追加:")
        for line in to_add:
            print(f"    {line.rstrip()}")
        return

    with open(ann_path, "a", encoding="utf-8") as f:
        for line in to_add:
            f.write(line)
    print(f"  已追加 {len(to_add)} 筆受試者標籤到 ann.txt")


def merge(dry_run: bool = False) -> None:
    if not SOURCE_DIR.exists():
        print(f"[錯誤] 來源目錄不存在: {SOURCE_DIR}")
        sys.exit(1)

    print(f"來源: {SOURCE_DIR}")
    print(f"目標: {TARGET_DIR}")
    if dry_run:
        print("[DRY-RUN 模式] 不實際寫入檔案\n")

    subjects_processed: dict[str, list[str]] = {}

    for wmv in sorted(SOURCE_DIR.glob("*.wmv")):
        m = _FNAME_RE.match(wmv.name)
        if not m:
            print(f"  [略過] 無法解析: {wmv.name}")
            continue

        subject = m.group(1)   # e.g. "c1"
        orig_stem = wmv.stem   # e.g. "c1_Image05_伸握伸"
        video_stem = m.group(2)  # e.g. "Image05_伸握伸"

        if subject not in SUBJECT_LABELS:
            print(f"  [略過] 未知 subject '{subject}': {wmv.name}")
            continue

        print(f"\n處理: {wmv.name}  →  {subject}/{video_stem}.wmv")

        dst_subject = TARGET_DIR / subject
        dst_video = dst_subject / f"{video_stem}.wmv"
        dst_json = dst_subject / f"{video_stem}.json"
        dst_masks = dst_subject / "seg_masks" / video_stem

        # 1. 複製 WMV
        if not dry_run:
            copied = _copy(wmv, dst_video)
            print(f"  video: {'複製' if copied else '已存在 (略過)'}")
        else:
            print(f"  [dry-run] 複製 video → {dst_video.relative_to(PROJECT_ROOT)}")

        # 2. 複製 + 更新 JSON
        src_json = SOURCE_DIR / f"{orig_stem}.json"
        if src_json.exists():
            n_updated = _process_json(src_json, dst_json, subject, orig_stem, video_stem, dry_run)
            print(f"  json: {n_updated} 個 mask_path 已更新")
        else:
            print(f"  [警告] JSON 不存在: {src_json}")

        # 3. 複製 seg_masks (位於 SOURCE_DIR/seg_masks/{orig_stem}/)
        src_mask_dir = SOURCE_DIR / "seg_masks" / orig_stem
        n_masks = _copy_seg_masks(src_mask_dir, dst_masks, dry_run)
        print(f"  seg_masks: {n_masks} 個 PNG {'(dry-run)' if dry_run else '已複製'}")

        # 4. 複製 labels.txt
        src_labels = SOURCE_DIR / "labels.txt"
        if src_labels.exists() and not dry_run:
            _copy(src_labels, dst_subject / "labels.txt")

        subjects_processed.setdefault(subject, []).append(video_stem)

    print("\n--- 合併摘要 ---")
    for subj, videos in sorted(subjects_processed.items()):
        label = SUBJECT_LABELS.get(subj, "?")
        print(f"  {subj} (label={label}): {len(videos)} 部影片 → {', '.join(videos)}")

    print("\n更新 ann.txt ...")
    _update_ann_txt(TARGET_DIR, SUBJECT_LABELS, SUBJECT_NOTES, dry_run)

    print("\n合併完成。" if not dry_run else "\n[dry-run] 完成，未寫入任何檔案。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合併 2026_深度1.5 資料集")
    parser.add_argument("--dry-run", action="store_true", help="僅顯示將執行的操作，不實際寫入")
    args = parser.parse_args()
    merge(dry_run=args.dry_run)
