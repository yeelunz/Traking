from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Tuple


def _rename_metric_keys(data: Any) -> Tuple[Any, bool]:
    """Recursively rename legacy mAP keys to success rate keys.

    Returns the possibly modified object and a flag indicating whether a change was made.
    """
    changed = False
    if isinstance(data, dict):
        updated_items = {}
        for key, value in data.items():
            new_value, value_changed = _rename_metric_keys(value)
            changed |= value_changed
            new_key = key
            if "mAP_50" in key:
                new_key = key.replace("mAP_50", "success_rate_50")
            elif "mAP_75" in key:
                new_key = key.replace("mAP_75", "success_rate_75")
            if new_key != key:
                changed = True
            updated_items[new_key] = new_value
        data.clear()
        data.update(updated_items)
        if "frames_count" in data:
            # Ensure new metrics exist even for historical runs where they couldn't be computed.
            data.setdefault("fps", 0.0)
            data.setdefault("drift_rate", 0.0)
        return data, changed
    if isinstance(data, list):
        for idx, item in enumerate(data):
            new_item, item_changed = _rename_metric_keys(item)
            if item_changed:
                data[idx] = new_item
                changed = True
        return data, changed
    return data, changed


def upgrade_file(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc
    updated_payload, changed = _rename_metric_keys(payload)
    if not changed:
        return False
    path.write_text(json.dumps(updated_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Upgrade tracking summary metrics to success rate naming.")
    parser.add_argument("files", nargs="+", type=Path, help="JSON summary files to upgrade in-place")
    args = parser.parse_args()
    any_changes = False
    for file_path in args.files:
        if not file_path.exists():
            parser.error(f"File not found: {file_path}")
        if upgrade_file(file_path):
            print(f"Updated {file_path}")
            any_changes = True
        else:
            print(f"No changes needed for {file_path}")
    if not any_changes:
        print("All files already up-to-date.")


if __name__ == "__main__":  # pragma: no cover - direct CLI usage
    main()
