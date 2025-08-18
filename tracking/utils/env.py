from __future__ import annotations
import os
import platform
import subprocess
from typing import Dict


def capture_env() -> Dict[str, str]:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        out = subprocess.check_output(["pip", "freeze"], text=True)
        info["pip_freeze"] = out
    except Exception:
        pass
    return info
