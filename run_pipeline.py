from __future__ import annotations
import argparse
import json
import os

from tracking.orchestrator.runner import PipelineRunner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to pipeline YAML/JSON config")
    args = ap.parse_args()

    # Support JSON first; YAML optional if pyyaml installed
    cfg = None
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

    runner = PipelineRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
