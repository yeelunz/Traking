"""Debug script for FasterRCNN first-epoch anomaly analysis.

Usage (Windows cmd):
  python -m tools.debug_frcnn --root "PATH/TO/DATASET" --max-batches 10 --yaml def_frcnn.yaml

It will:
  1. Load the dataset manager like pipeline.
  2. Build FasterRCNN model (respecting YAML params if provided).
  3. Iterate over the training DataLoader for N batches (no weight update by default unless --train-step).
  4. For each batch, compute forward loss dict, record per-loss components, image/box stats.
  5. Detect outlier batch (very large total loss vs median).
  6. Optionally dump visualization (draw GT boxes) for outlier batch with --viz-dir.

Outputs:
  - Printed table.
  - JSON summary with batch stats (losses, per-image box stats) at --out-json.
"""
from __future__ import annotations

import sys

TOOL_NAME = "debug_frcnn"

RETIRE_MESSAGE = (
    f"The '{TOOL_NAME}' diagnostics script has been retired.\n"
    "Please use the consolidated experiment viewer instead:\n"
    "  python -m tools.experiment_viewer --help\n"
)


def main() -> int:
    sys.stderr.write(RETIRE_MESSAGE)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
