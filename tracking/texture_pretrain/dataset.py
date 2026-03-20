from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
except Exception as exc:  # noqa: BLE001
    torch = None  # type: ignore
    Dataset = object  # type: ignore
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


@dataclass(frozen=True)
class ROISample:
    image_path: str
    label: int


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for ROIPretrainDataset."
            + (f" Import error: {_TORCH_IMPORT_ERROR}" if _TORCH_IMPORT_ERROR else "")
        )


def _to_sample(item: Union[ROISample, Dict[str, Any], Tuple[str, int], List[Any]]) -> ROISample:
    if isinstance(item, ROISample):
        return item
    if isinstance(item, dict):
        image_path = item.get("image_path", item.get("path", item.get("image")))
        label = item.get("label", item.get("y", item.get("target")))
        if image_path is None or label is None:
            raise ValueError(f"Invalid manifest item: {item}")
        return ROISample(str(image_path), int(label))
    if isinstance(item, (tuple, list)) and len(item) >= 2:
        return ROISample(str(item[0]), int(item[1]))
    raise ValueError(f"Unsupported ROI sample format: {type(item)}")


def load_roi_manifest(manifest_path: str) -> List[ROISample]:
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"ROI manifest not found: {manifest_path}")

    suffix = path.suffix.lower()
    samples: List[ROISample] = []

    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("samples", payload.get("data", []))
        if not isinstance(payload, list):
            raise ValueError(f"JSON manifest must be a list or contain 'samples': {manifest_path}")
        samples = [_to_sample(item) for item in payload]
    elif suffix in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                samples.append(_to_sample(json.loads(raw)))
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                samples.append(_to_sample(row))
    else:
        raise ValueError(
            "Unsupported ROI manifest format. Use .json, .jsonl/.ndjson, or .csv"
        )

    if not samples:
        raise ValueError(f"No ROI samples found in manifest: {manifest_path}")
    return samples


class ROIPretrainDataset(Dataset):
    """Stage-1 ROI supervised dataset.

    Each sample contains:
    - image_path: path to ROI image
    - label: integer class id
    """

    def __init__(
        self,
        samples: Sequence[Union[ROISample, Dict[str, Any], Tuple[str, int], List[Any]]],
        transform: Optional[Any] = None,
        strict: bool = True,
    ):
        _require_torch()
        self.samples: List[ROISample] = [_to_sample(item) for item in samples]
        self.transform = transform
        self.strict = bool(strict)

        if not self.samples:
            raise ValueError("ROIPretrainDataset received empty samples.")

        if self.strict:
            missing = [s.image_path for s in self.samples if not Path(s.image_path).exists()]
            if missing:
                raise FileNotFoundError(
                    f"Found {len(missing)} missing ROI images. First missing: {missing[0]}"
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        path = Path(sample.image_path)
        if not path.exists():
            raise FileNotFoundError(f"ROI image not found: {sample.image_path}")

        with Image.open(path) as img:
            img = img.convert("RGB")
            if self.transform is not None:
                x = self.transform(img)
            else:
                x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

        y = torch.tensor(int(sample.label), dtype=torch.long)
        return x, y
