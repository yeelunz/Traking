from __future__ import annotations

try:
    from .main_window import launch
except ImportError:  # pragma: no cover - executed when run as a script
    from main_window import launch  # type: ignore


if __name__ == "__main__":
    launch()
