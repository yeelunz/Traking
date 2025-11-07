from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _resolve_launch():
    try:
        from .main_window import launch  # type: ignore  # pragma: no cover
        return launch
    except ImportError:
        package_dir = Path(__file__).resolve().parent
        project_root = package_dir.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        module = importlib.import_module("segmentation_annotator.main_window")
        return getattr(module, "launch")


def main() -> None:
    launch = _resolve_launch()
    launch()


if __name__ == "__main__":
    main()
