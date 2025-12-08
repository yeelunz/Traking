from __future__ import annotations

import json
import os
from typing import Any, Dict


def resolve_path(path: str) -> str:
    """Expand user/home markers and return an absolute path."""
    candidate = os.path.expanduser(str(path))
    return os.path.abspath(candidate)


def load_json_file(path: str) -> Dict[str, Any]:
    resolved = resolve_path(path)
    with open(resolved, "r", encoding="utf-8") as f:
        return json.load(f)
