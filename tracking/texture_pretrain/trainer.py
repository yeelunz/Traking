from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
except Exception as exc:  # noqa: BLE001
    torch = None  # type: ignore
    nn = None  # type: ignore
    DataLoader = None  # type: ignore
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

try:
    from torchvision import transforms
except Exception as exc:  # noqa: BLE001
    transforms = None  # type: ignore
    _TV_IMPORT_ERROR = exc
else:
    _TV_IMPORT_ERROR = None


logger = logging.getLogger(__name__)


def _require_runtime() -> None:
    if torch is None or nn is None or DataLoader is None:
        raise RuntimeError(
            "PyTorch is required for texture pretrain trainer."
            + (f" Import error: {_TORCH_IMPORT_ERROR}" if _TORCH_IMPORT_ERROR else "")
        )
    if transforms is None:
        raise RuntimeError(
            "torchvision is required for transforms."
            + (f" Import error: {_TV_IMPORT_ERROR}" if _TV_IMPORT_ERROR else "")
        )


@dataclass
class TexturePretrainConfig:
    backbone: str = "convnext_tiny"
    num_classes: int = 2
    embedding_dim: int = 32
    input_size: int = 224
    batch_size: int = 32
    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-4
    save_path: str = "ckpt/convnext_tex_pretrain.pth"
    save_backbone_only: bool = True
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    num_workers: int = 4
    device: str = "auto"
    amp: bool = True
    head_type: str = "linear"
    hidden_dim: int = 256
    dropout: float = 0.2
    pretrained_imagenet: bool = True
    train_manifest: Optional[str] = None
    val_manifest: Optional[str] = None
    cache_enabled: bool = True
    cache_dir: str = "ckpt/texture_pretrain_cache"
    cache_key: Optional[str] = None
    force_official_pretrained: bool = True


def build_train_transform(input_size: int, enable_augmentation: bool = True):
    _require_runtime()
    ops = [transforms.Resize((input_size, input_size))]
    if enable_augmentation:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ]
        )
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(ops)


def build_eval_transform(input_size: int):
    _require_runtime()
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class TexturePretrainTrainer:
    def __init__(
        self,
        model: Any,
        cfg: TexturePretrainConfig,
    ):
        _require_runtime()
        self.model = model
        self.cfg = cfg
        if cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(cfg.device)

        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        if str(cfg.optimizer).lower() != "adamw":
            raise ValueError("Only AdamW is currently supported.")
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(cfg.lr),
            weight_decay=float(cfg.weight_decay),
        )

        sch = str(cfg.scheduler).strip().lower()
        if sch in {"", "none", "off"}:
            self.scheduler = None
            self._scheduler_mode = "none"
        elif sch == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(1, int(cfg.epochs))
            )
            self._scheduler_mode = "cosine"
        elif sch == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", factor=0.5, patience=3
            )
            self._scheduler_mode = "plateau"
        else:
            raise ValueError("scheduler must be one of: none, cosine, plateau")

        self.use_amp = bool(cfg.amp and self.device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _run_epoch(self, loader: Any, training: bool) -> Dict[str, float]:
        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            batch_size = int(labels.shape[0])

            with torch.set_grad_enabled(training):
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)

                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            pred = torch.argmax(logits, dim=1)
            total_correct += int((pred == labels).sum().item())
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size

        if total_count <= 0:
            return {"loss": 0.0, "acc": 0.0}
        return {
            "loss": float(total_loss / total_count),
            "acc": float(total_correct / total_count),
        }

    def _save_checkpoint(
        self,
        path: Path,
        *,
        epoch: int,
        is_best: bool,
        save_backbone_only: bool,
        history: Dict[str, Any],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: Dict[str, Any] = {
            "stage": "texture_pretrain_stage1",
            "epoch": int(epoch),
            "is_best": bool(is_best),
            "backbone_name": getattr(self.model, "backbone_name", self.cfg.backbone),
            "num_classes": int(self.cfg.num_classes),
            "embedding_dim": int(self.cfg.embedding_dim),
            "input_size": int(self.cfg.input_size),
            "pretrained_imagenet": bool(self.cfg.pretrained_imagenet),
            "head_type": str(self.cfg.head_type),
            "cache_key": self.cfg.cache_key,
            "history": history,
            "backbone_state_dict": {
                k: v.detach().cpu() for k, v in self.model.backbone.state_dict().items()
            },
            "projection_state_dict": {
                k: v.detach().cpu() for k, v in self.model.proj.state_dict().items()
            },
        }

        if not save_backbone_only:
            payload["model_state_dict"] = {
                k: v.detach().cpu() for k, v in self.model.state_dict().items()
            }
            payload["head_state_dict"] = {
                k: v.detach().cpu() for k, v in self.model.head.state_dict().items()
            }

        torch.save(payload, str(path))

    def fit(self, train_loader: Any, val_loader: Optional[Any] = None) -> Dict[str, Any]:
        epochs = int(self.cfg.epochs)
        if epochs <= 0:
            raise ValueError("epochs must be > 0")

        best_metric = float("-inf")
        best_state = None
        history: Dict[str, Any] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        for epoch in range(1, epochs + 1):
            train_stats = self._run_epoch(train_loader, training=True)
            val_stats = self._run_epoch(val_loader, training=False) if val_loader is not None else None

            history["train_loss"].append(train_stats["loss"])
            history["train_acc"].append(train_stats["acc"])
            if val_stats is not None:
                history["val_loss"].append(val_stats["loss"])
                history["val_acc"].append(val_stats["acc"])

            metric = val_stats["acc"] if val_stats is not None else train_stats["acc"]
            if metric > best_metric:
                best_metric = metric
                best_state = copy.deepcopy(self.model.state_dict())
                self._save_checkpoint(
                    Path(self.cfg.save_path),
                    epoch=epoch,
                    is_best=True,
                    save_backbone_only=bool(self.cfg.save_backbone_only),
                    history=history,
                )

            if self.scheduler is not None:
                if self._scheduler_mode == "plateau":
                    self.scheduler.step(metric)
                else:
                    self.scheduler.step()

            lr_now = float(self.optimizer.param_groups[0].get("lr", 0.0))
            if val_stats is None:
                logger.info(
                    "[Epoch %03d/%03d] train_loss=%.4f train_acc=%.4f lr=%.6g",
                    epoch,
                    epochs,
                    train_stats["loss"],
                    train_stats["acc"],
                    lr_now,
                )
            else:
                logger.info(
                    "[Epoch %03d/%03d] train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f lr=%.6g",
                    epoch,
                    epochs,
                    train_stats["loss"],
                    train_stats["acc"],
                    val_stats["loss"],
                    val_stats["acc"],
                    lr_now,
                )

        if best_state is not None:
            self.model.load_state_dict(best_state)

        last_path = Path(self.cfg.save_path)
        if last_path.suffix:
            last_path = last_path.with_name(f"{last_path.stem}.last{last_path.suffix}")
        else:
            last_path = last_path.with_name(f"{last_path.name}.last")

        self._save_checkpoint(
            last_path,
            epoch=epochs,
            is_best=False,
            save_backbone_only=bool(self.cfg.save_backbone_only),
            history=history,
        )

        return {
            "best_metric": float(best_metric),
            "best_ckpt": str(Path(self.cfg.save_path)),
            "last_ckpt": str(last_path),
            "device": str(self.device),
            "epochs": epochs,
            "history": history,
        }
