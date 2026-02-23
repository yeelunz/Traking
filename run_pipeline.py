from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple


def _iter_config_variants(cfg: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Yield one-config variants.

    UI Queue 匯入時會把 experiments>1 拆成多個 config；CLI 也沿用相同行為。
    """
    experiments = cfg.get("experiments")
    if not isinstance(experiments, list) or len(experiments) <= 1:
        yield cfg
        return
    base = dict(cfg)
    for exp in experiments:
        if not isinstance(exp, dict):
            continue
        variant = dict(base)
        variant["experiments"] = [dict(exp)]
        yield variant

def _iter_schedule_items(
    data: Any,
    *,
    queue_index: Optional[int] = None,
    queue_label: Optional[str] = None,
) -> Iterator[Tuple[Optional[str], Dict[str, Any]]]:
    """Yield (label, config_dict) from either a single config or a queue schedule."""
    if not isinstance(data, dict):
        raise SystemExit("Config must be a dict (single config) or a dict with 'queue: [...]'.")

    if isinstance(data.get("queue"), list):
        queue = data.get("queue") or []
        selected_item: Optional[Dict[str, Any]] = None
        if queue_label:
            for item in queue:
                if isinstance(item, dict) and item.get("label") == queue_label:
                    selected_item = item
                    break
            if selected_item is None:
                raise SystemExit(f"queue_label not found: {queue_label}")
        elif queue_index is not None:
            if queue_index < 0 or queue_index >= len(queue):
                raise SystemExit(f"queue_index out of range: {queue_index}")
            selected_item = queue[queue_index] if isinstance(queue[queue_index], dict) else None

        items = [selected_item] if selected_item is not None else queue
        for item in items:
            if not isinstance(item, dict):
                continue
            cfg_obj = item.get("config") if isinstance(item.get("config"), dict) else None
            if cfg_obj is None:
                # allow shorthand where queue element itself is the config
                cfg_obj = item
            if not isinstance(cfg_obj, dict):
                continue

            # Skip defs/empty configs to make CLI queue runs convenient
            experiments = cfg_obj.get("experiments")
            if isinstance(experiments, list) and len(experiments) == 0:
                continue

            label = item.get("label") if isinstance(item.get("label"), str) else None
            variants = list(_iter_config_variants(cfg_obj))
            if label and len(variants) > 1:
                for idx, v in enumerate(variants, start=1):
                    yield f"{label} #{idx}", v
            else:
                for v in variants:
                    yield label, v
        return

    # not a queue schedule -> treat as single config
    for v in _iter_config_variants(data):
        yield None, v


def main():
    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from tracking.orchestrator.runner import PipelineRunner

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to pipeline YAML/JSON config")
    ap.add_argument(
        "--queue-index",
        type=int,
        default=None,
        help="If config contains `queue:`, run only the Nth item (0-based).",
    )
    ap.add_argument(
        "--queue-label",
        type=str,
        default=None,
        help="If config contains `queue:`, run only the item with matching label.",
    )
    args = ap.parse_args()

    # Support JSON first; YAML optional if pyyaml installed
    cfg: Any = None
    if args.config.lower().endswith((".yml", ".yaml")):
        try:
            import yaml  # type: ignore

            with open(args.config, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        except Exception as e:
            raise SystemExit(f"Failed to read YAML config: {e}")
    else:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)

    ran_any = False
    detector_reuse_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for label, one_cfg in _iter_schedule_items(cfg, queue_index=args.queue_index, queue_label=args.queue_label):
        ran_any = True
        if label:
            print(f"\n=== Running schedule item: {label} ===")
        runner = PipelineRunner(one_cfg, detector_reuse_cache=detector_reuse_cache)
        runner.run()

    if not ran_any:
        print("No runnable items found (all empty?)", file=sys.stderr)
        return


if __name__ == "__main__":
    main()
