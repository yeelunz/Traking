from __future__ import annotations
import os
import random
from typing import Optional

try:
    import numpy as np  # type: ignore
except Exception:
    np = None

try:
    import torch  # type: ignore
except Exception:
    torch = None


def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True  # type: ignore
            torch.backends.cudnn.benchmark = False  # type: ignore
