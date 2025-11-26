"""Simple UI tool to inspect schedule experiment results.

This utility lets users enter a results schedule folder (e.g.
``2025-10-31_17-30-25_schedule_10exp``) and visualise all experiment
metrics in a sortable table. Data is gathered from ``metadata.json`` and
``test/metrics/summary.json`` inside each experiment directory.
"""
from __future__ import annotations

import sys

TOOL_NAME = "schedule_results_viewer"

RETIRE_MESSAGE = (
    f"The '{TOOL_NAME}' viewer has been retired.\n"
    "Please use the consolidated experiment viewer instead:\n"
    "  python -m tools.experiment_viewer --help\n"
)


def main() -> int:
    sys.stderr.write(RETIRE_MESSAGE)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
