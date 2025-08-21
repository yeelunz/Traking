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
    """Set seeds for common RNGs to improve reproducibility.

    Note: setting PYTHONHASHSEED at runtime does not affect the interpreter's
    hash randomization for objects created before startup. For full
    reproducibility involving hash(), set the PYTHONHASHSEED environment
    variable before launching Python (e.g. `set PYTHONHASHSEED=42` on Windows).
    """
    # Python built-in RNG
    random.seed(seed)
    # Note: changing PYTHONHASHSEED at runtime won't retroactively affect
    # hash() behavior for already-started interpreter; it's still useful
    # to record it and to set for subprocesses.
    os.environ["PYTHONHASHSEED"] = str(seed)

    # numpy RNG
    if np is not None:
        try:
            np.random.seed(seed)
        except Exception:
            # older/newer numpy variants should support this, but ignore if not
            pass

    # torch RNG + deterministic options
    if torch is not None:
        try:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
        if deterministic:
            # Prefer use_deterministic_algorithms when available (PyTorch 1.8+)
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                # Fallback to cudnn flags
                try:
                    torch.backends.cudnn.deterministic = True  # type: ignore
                    torch.backends.cudnn.benchmark = False  # type: ignore
                except Exception:
                    pass


def get_torch_generator(seed: int):
    """Return a torch.Generator seeded with `seed`, or None if torch missing.

    Use this with DataLoader(..., generator=get_torch_generator(seed)) to
    make shuffling reproducible across runs.
    """
    if torch is None:
        return None
    try:
        g = torch.Generator()
        g.manual_seed(seed)
        return g
    except Exception:
        return None
