from __future__ import annotations
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, Any, Iterable, List, Tuple, Optional

from ..core.interfaces import DatasetManager


class SimpleDataset:
    """
    A minimal dataset reading per-video JSON annotations produced by the label tool.
    Each video has a JSON next to it: <video>.json with COCO-VID-like content.
    This dataset yields frame records for a given split list of videos.
    """
    def __init__(self, videos: List[str], annotations: Dict[str, Any]):
        self.items: List[Tuple[str, Dict[str, Any]]] = []
        for v in videos:
            ann = annotations.get(v)
            if ann is not None:
                self.items.append((v, ann))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        video_path, ann = self.items[idx]
        return {"video_path": video_path, "annotation": ann}


_SUBJECT_PREFIX_RE = re.compile(r"^(\d+)")


class COCOJsonDatasetManager(DatasetManager):
    def __init__(self, root: str):
        self.root = root
        # index videos and their jsons
        self.videos: List[str] = []
        self.ann_by_video: Dict[str, Any] = {}
        self.missing_annotations: List[str] = []
        self.video_subjects: Dict[str, str] = {}
        self.last_split_method: str = "video_level"
        self._scan()

    def _scan(self):
        exts = {".mp4", ".avi", ".mov", ".mkv"}
        videos: List[str] = []
        missing: List[str] = []
        root_dir = os.path.abspath(self.root)
        if not os.path.exists(root_dir):
            return
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if os.path.splitext(filename)[1].lower() not in exts:
                    continue
                video_path = os.path.join(dirpath, filename)
                videos.append(video_path)
                subject = self._derive_subject(video_path)
                self.video_subjects[video_path] = subject
                json_path = os.path.splitext(video_path)[0] + ".json"
                if os.path.exists(json_path):
                    try:
                        with open(json_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        self.ann_by_video[video_path] = data
                    except Exception:
                        pass
                else:
                    missing.append(video_path)
        videos.sort()
        self.videos = videos
        self.missing_annotations = missing

    def _derive_subject(self, video_path: str) -> str:
        try:
            rel = os.path.relpath(video_path, self.root)
        except Exception:
            rel = os.path.basename(video_path)
        if rel.startswith(".."):
            rel = os.path.basename(video_path)
        parts = Path(rel).parts
        if len(parts) > 1:
            subject = parts[0]
        else:
            subject = Path(parts[0]).stem
        primary = self._normalise_subject_token(subject)
        if primary:
            return primary
        fallback_stem = Path(video_path).stem
        fallback = self._normalise_subject_token(fallback_stem)
        if fallback:
            return fallback
        return subject or fallback_stem

    def _normalise_subject_token(self, token: Optional[str]) -> Optional[str]:
        if token is None:
            return None
        cleaned = str(token).strip()
        if not cleaned:
            return None
        match = _SUBJECT_PREFIX_RE.match(cleaned)
        if match:
            return match.group(1)
        return cleaned if cleaned else None

    def _group_by_subject(self) -> Dict[str, List[str]]:
        groups: Dict[str, List[str]] = {}
        for video_path in self.videos:
            subject = self.video_subjects.get(video_path) or self._derive_subject(video_path)
            groups.setdefault(subject, []).append(video_path)
        return groups

    def split(self, method: str = "video_level", seed: int = 0, ratios=(0.7, 0.2, 0.1)) -> Dict[str, Any]:
        self.last_split_method = method
        if method == "subject_level":
            train, val, test = self._split_subject_level(seed, ratios)
        elif method == "loso":
            # For LOSO we still return a single split if caller expects split(); prefer k_fold/iterators for full CV.
            folds = list(self.loso())
            if folds:
                train, val, test = folds[0]["train"], [], folds[0]["test"]
            else:
                train, val, test = [], [], []
        else:
            train, val, test = self._split_video_level(seed, ratios)
        return {
            "train": SimpleDataset(train, self.ann_by_video),
            "val": SimpleDataset(val, self.ann_by_video),
            "test": SimpleDataset(test, self.ann_by_video),
        }

    def loso(self) -> Iterable[Dict[str, Any]]:
        """
        Leave-One-Subject-Out splits: each subject's videos become the test set once.
        """
        groups = self._group_by_subject()
        subjects = sorted(groups.keys())
        for subject in subjects:
            test_videos = groups.get(subject, [])
            train_videos = [v for s, vids in groups.items() if s != subject for v in vids]
            yield {
                "subject": subject,
                "train": train_videos,
                "test": test_videos,
            }

    def _split_video_level(self, seed: int, ratios: Tuple[float, float, float]) -> Tuple[List[str], List[str], List[str]]:
        vids = self.videos[:]
        random.Random(seed).shuffle(vids)
        n = len(vids)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        train = vids[:n_train]
        val = vids[n_train:n_train + n_val]
        test = vids[n_train + n_val:]
        return train, val, test

    def _split_subject_level(self, seed: int, ratios: Tuple[float, float, float]) -> Tuple[List[str], List[str], List[str]]:
        groups = self._group_by_subject()
        subjects = list(groups.keys())
        random.Random(seed).shuffle(subjects)
        total_videos = sum(len(groups[s]) for s in subjects)
        train_ratio, val_ratio, test_ratio = ratios
        target_train = int(total_videos * max(0.0, train_ratio))
        target_val = int(total_videos * max(0.0, val_ratio))
        train: List[str] = []
        val: List[str] = []
        test: List[str] = []
        counts = [0, 0, 0]
        for subject in subjects:
            vids = groups[subject]
            if counts[0] < target_train:
                train.extend(vids)
                counts[0] += len(vids)
            elif counts[1] < target_val:
                val.extend(vids)
                counts[1] += len(vids)
            else:
                test.extend(vids)
                counts[2] += len(vids)
        return train, val, test

    def k_fold(self, k: int, seed: int = 0) -> Iterable[Dict[str, Any]]:
        method = getattr(self, "last_split_method", "video_level")
        if method == "subject_level":
            groups = self._group_by_subject()
            subjects = list(groups.keys())
            random.Random(seed).shuffle(subjects)
            fold_size = max(1, len(subjects) // k)
            for i in range(k):
                val_subjects = subjects[i * fold_size:(i + 1) * fold_size]
                val = [v for s in val_subjects for v in groups[s]]
                train = [v for s in subjects if s not in val_subjects for v in groups[s]]
                yield {
                    "train": SimpleDataset(train, self.ann_by_video),
                    "val": SimpleDataset(val, self.ann_by_video),
                }
        else:
            vids = self.videos[:]
            random.Random(seed).shuffle(vids)
            fold_size = max(1, len(vids) // k)
            for i in range(k):
                val = vids[i * fold_size:(i + 1) * fold_size]
                train = [v for v in vids if v not in val]
                yield {
                    "train": SimpleDataset(train, self.ann_by_video),
                    "val": SimpleDataset(val, self.ann_by_video),
                }

    def to_pytorch_dataloader(self, subset_name: str, batch_size: int, transforms=None):
        raise NotImplementedError("Hook up with torch.utils.data when needed.")
