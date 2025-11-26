from __future__ import annotations

import sys

TOOL_NAME = "upgrade_summary_metrics"

RETIRE_MESSAGE = (
    f"The '{TOOL_NAME}' helper has been retired.\n"
    "Please use the consolidated experiment viewer instead:\n"
    "  python -m tools.experiment_viewer --help\n"
)


def main() -> int:
    sys.stderr.write(RETIRE_MESSAGE)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
