from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np


@dataclass
class FeatureVectoriser:
    """Deterministically maps feature dictionaries to numpy vectors."""

    keys: Sequence[str]

    def transform(self, features: Iterable[Dict[str, float]]) -> np.ndarray:
        ordered_keys: List[str] = list(self.keys)
        matrix: List[List[float]] = []
        for feat in features:
            row = [float(feat.get(k, 0.0)) for k in ordered_keys]
            matrix.append(row)
        if not matrix:
            return np.zeros((0, len(ordered_keys)), dtype=np.float32)
        return np.asarray(matrix, dtype=np.float32)
