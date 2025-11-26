"""Unified desktop UI hosting multiple tracking related tools.

This module provides a single entry point exposing existing utility widgets
such as the schedule results viewer, and stubs for upcoming diagnostics
features. Future tools can be registered here by adding new tabs to the
``ToolsWorkbench`` window.
"""
from __future__ import annotations

import sys

TOOL_NAME = "tools_workbench"

RETIRE_MESSAGE = (
    f"The '{TOOL_NAME}' entry point has been retired.\n"
    "Please use the consolidated experiment viewer instead:\n"
    "  python -m tools.experiment_viewer --help\n"
)


def main() -> int:
    sys.stderr.write(RETIRE_MESSAGE)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
