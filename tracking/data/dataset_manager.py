from __future__ import annotations
import json
import os
import random
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


class COCOJsonDatasetManager(DatasetManager):
    def __init__(self, root: str):
        self.root = root
        # index videos and their jsons
        self.videos: List[str] = []
        self.ann_by_video: Dict[str, Any] = {}
        self._scan()

    def _scan(self):
        exts = {".mp4", ".avi", ".mov", ".mkv"}
        for name in os.listdir(self.root):
            p = os.path.join(self.root, name)
            if os.path.splitext(name)[1].lower() in exts:
                self.videos.append(p)
                j = os.path.splitext(p)[0] + ".json"
                if os.path.exists(j):
                    try:
                        with open(j, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        self.ann_by_video[p] = data
                    except Exception:
                        pass
        self.videos.sort()

    def split(self, method: str = "video_level", seed: int = 0, ratios=(0.7, 0.2, 0.1)) -> Dict[str, Any]:
        random.Random(seed).shuffle(self.videos)
        n = len(self.videos)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        train = self.videos[:n_train]
        val = self.videos[n_train:n_train + n_val]
        test = self.videos[n_train + n_val:]
        return {
            "train": SimpleDataset(train, self.ann_by_video),
            "val": SimpleDataset(val, self.ann_by_video),
            "test": SimpleDataset(test, self.ann_by_video),
        }

    def k_fold(self, k: int, seed: int = 0) -> Iterable[Dict[str, Any]]:
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
