from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import shutil
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

try:
    import torch
    from torch.utils.data import DataLoader
except Exception as exc:  # noqa: BLE001
    torch = None  # type: ignore
    DataLoader = None  # type: ignore
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from tracking.texture_pretrain.dataset import ROIPretrainDataset, load_roi_manifest
    from tracking.texture_pretrain.model import TexturePretrainModel
    from tracking.texture_pretrain.trainer import (
        TexturePretrainConfig,
        TexturePretrainTrainer,
        build_eval_transform,
        build_train_transform,
    )
else:
    from .dataset import ROIPretrainDataset, load_roi_manifest
    from .model import TexturePretrainModel
    from .trainer import (
        TexturePretrainConfig,
        TexturePretrainTrainer,
        build_eval_transform,
        build_train_transform,
    )


def _require_runtime() -> None:
    if torch is None or DataLoader is None:
        raise RuntimeError(
            "PyTorch is required for train_texture.py."
            + (f" Import error: {_TORCH_IMPORT_ERROR}" if _TORCH_IMPORT_ERROR else "")
        )


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _build_cache_identity(cfg: TexturePretrainConfig) -> Dict[str, Any]:
    train_manifest = Path(str(cfg.train_manifest)).resolve()
    val_manifest = Path(str(cfg.val_manifest)).resolve() if cfg.val_manifest else None
    identity: Dict[str, Any] = {
        "schema": "texture_pretrain_cache_v1",
        "backbone": str(cfg.backbone),
        "num_classes": int(cfg.num_classes),
        "embedding_dim": int(cfg.embedding_dim),
        "input_size": int(cfg.input_size),
        "head_type": str(cfg.head_type),
        "hidden_dim": int(cfg.hidden_dim),
        "dropout": float(cfg.dropout),
        "pretrained_imagenet": bool(cfg.pretrained_imagenet),
        "train_manifest_sha256": _sha256_file(train_manifest),
        "val_manifest_sha256": _sha256_file(val_manifest) if val_manifest else None,
    }
    return identity


def _build_cache_key(identity: Dict[str, Any]) -> str:
    packed = json.dumps(identity, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()[:24]


def _resolve_last_ckpt_path(save_path: Path) -> Path:
    if save_path.suffix:
        return save_path.with_name(f"{save_path.stem}.last{save_path.suffix}")
    return save_path.with_name(f"{save_path.name}.last")


def _parse_config(config_path: str) -> TexturePretrainConfig:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    section: Dict[str, Any]
    if isinstance(payload, dict) and "texture_pretrain" in payload:
        section = dict(payload.get("texture_pretrain") or {})
    elif isinstance(payload, dict):
        section = dict(payload)
    else:
        raise ValueError("Config root must be a mapping.")

    if "enable" in section and not bool(section.get("enable")):
        raise RuntimeError("texture_pretrain.enable is false; nothing to run.")

    known = {f.name for f in fields(TexturePretrainConfig)}
    kwargs = {k: v for k, v in section.items() if k in known}

    cfg = TexturePretrainConfig(**kwargs)
    if not cfg.train_manifest:
        raise ValueError("texture_pretrain.train_manifest is required.")
    return cfg


def _build_loader(dataset, batch_size: int, num_workers: int, shuffle: bool):
    return DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=bool(shuffle),
        num_workers=max(0, int(num_workers)),
        pin_memory=torch.cuda.is_available(),
    )


def run_from_config(config_path: str) -> Dict[str, Any]:
    _require_runtime()
    cfg = _parse_config(config_path)

    if bool(cfg.force_official_pretrained) and not bool(cfg.pretrained_imagenet):
        raise ValueError(
            "Texture Stage-1 requires official pretrained initialization for fine-tuning. "
            "Set texture_pretrain.pretrained_imagenet=true (or disable force_official_pretrained)."
        )

    identity = _build_cache_identity(cfg)
    cache_key = _build_cache_key(identity)
    cfg.cache_key = cache_key

    if bool(cfg.cache_enabled):
        cache_dir = Path(str(cfg.cache_dir)).resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_ckpt = cache_dir / f"{cache_key}.pth"
        cache_meta = cache_dir / f"{cache_key}.json"

        if cache_ckpt.exists() and cache_meta.exists():
            try:
                meta_payload = json.loads(cache_meta.read_text(encoding="utf-8"))
            except Exception:
                meta_payload = {}

            saved_identity = meta_payload.get("identity") if isinstance(meta_payload, dict) else None
            if isinstance(saved_identity, dict) and saved_identity == identity:
                save_path = Path(str(cfg.save_path)).resolve()
                save_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cache_ckpt, save_path)
                last_path = _resolve_last_ckpt_path(save_path)
                shutil.copy2(cache_ckpt, last_path)

                logging.getLogger(__name__).info(
                    "Texture pretrain cache hit: key=%s, ckpt=%s",
                    cache_key,
                    cache_ckpt,
                )
                return {
                    "cached": True,
                    "cache_key": cache_key,
                    "best_ckpt": str(save_path),
                    "last_ckpt": str(last_path),
                    "cache_ckpt": str(cache_ckpt),
                    "identity": identity,
                }

    train_samples = load_roi_manifest(str(cfg.train_manifest))
    val_samples = load_roi_manifest(str(cfg.val_manifest)) if cfg.val_manifest else None

    train_tf = build_train_transform(int(cfg.input_size), enable_augmentation=True)
    eval_tf = build_eval_transform(int(cfg.input_size))

    train_ds = ROIPretrainDataset(train_samples, transform=train_tf, strict=True)
    val_ds = ROIPretrainDataset(val_samples, transform=eval_tf, strict=True) if val_samples else None

    train_loader = _build_loader(train_ds, cfg.batch_size, cfg.num_workers, shuffle=True)
    val_loader = _build_loader(val_ds, cfg.batch_size, cfg.num_workers, shuffle=False) if val_ds else None

    model = TexturePretrainModel(
        backbone=cfg.backbone,
        num_classes=cfg.num_classes,
        pretrained=bool(cfg.pretrained_imagenet),
        embedding_dim=cfg.embedding_dim,
        head_type=cfg.head_type,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
        input_size=cfg.input_size,
    )

    trainer = TexturePretrainTrainer(model, cfg)
    summary = trainer.fit(train_loader, val_loader)

    if bool(cfg.cache_enabled):
        cache_dir = Path(str(cfg.cache_dir)).resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_ckpt = cache_dir / f"{cache_key}.pth"
        cache_meta = cache_dir / f"{cache_key}.json"

        best_ckpt_path = Path(str(summary.get("best_ckpt", cfg.save_path))).resolve()
        if best_ckpt_path.exists():
            shutil.copy2(best_ckpt_path, cache_ckpt)
            cache_meta.write_text(
                json.dumps(
                    {
                        "cache_key": cache_key,
                        "identity": identity,
                        "best_ckpt": str(best_ckpt_path),
                        "cache_ckpt": str(cache_ckpt),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            summary["cache_key"] = cache_key
            summary["cache_ckpt"] = str(cache_ckpt)
            summary["cached"] = False

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Texture Stage-1 pretraining")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()

    _setup_logging(args.log_level)
    _set_seed(int(args.seed))

    summary = run_from_config(args.config)
    logging.getLogger(__name__).info("Pretraining finished: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
