from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracking.classification.metrics import summarise_classification


VIDEO_EXTS = (".avi", ".mp4", ".mov", ".mkv", ".wmv")


@dataclass
class FrameSample:
    video_path: str
    frame_index: int
    bbox: Tuple[float, float, float, float]
    subject: str
    label: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_subject_labels(label_file: Path) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    with label_file.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            subject = parts[0].strip()
            label = int(parts[1])
            mapping[subject] = 1 if label == 1 else 0
    if not mapping:
        raise ValueError(f"No valid labels found in: {label_file}")
    return mapping


def find_video_for_json(json_path: Path) -> Optional[Path]:
    base = json_path.with_suffix("")
    for ext in VIDEO_EXTS:
        vp = base.with_suffix(ext)
        if vp.exists():
            return vp
    return None


def sample_indices_evenly(total: int, max_count: int) -> List[int]:
    if max_count <= 0 or total <= max_count:
        return list(range(total))
    positions = np.linspace(0, total - 1, max_count, dtype=int)
    return sorted(set(int(v) for v in positions))


def build_samples(
    data_root: Path,
    subject_labels: Dict[str, int],
    max_frames_per_video: int,
) -> List[FrameSample]:
    all_samples: List[FrameSample] = []

    for subject_dir in sorted([p for p in data_root.iterdir() if p.is_dir()]):
        subject = subject_dir.name
        if subject not in subject_labels:
            continue
        label = subject_labels[subject]

        json_files = sorted(subject_dir.glob("*.json"))
        for json_path in json_files:
            video_path = find_video_for_json(json_path)
            if video_path is None:
                continue

            try:
                with json_path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                continue

            images = payload.get("images", []) or []
            annotations = payload.get("annotations", []) or []
            img_to_frame: Dict[int, int] = {}
            for img in images:
                img_id = img.get("id")
                frame_idx = img.get("frame_index")
                if img_id is None or frame_idx is None:
                    continue
                img_to_frame[int(img_id)] = int(frame_idx)

            local_samples: List[FrameSample] = []
            for ann in annotations:
                image_id = ann.get("image_id")
                bbox = ann.get("bbox")
                if image_id is None or not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                if int(image_id) not in img_to_frame:
                    continue
                x, y, w, h = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
                if w <= 1.0 or h <= 1.0:
                    continue
                local_samples.append(
                    FrameSample(
                        video_path=str(video_path),
                        frame_index=int(img_to_frame[int(image_id)]),
                        bbox=(x, y, w, h),
                        subject=subject,
                        label=label,
                    )
                )

            local_samples.sort(key=lambda s: s.frame_index)
            if max_frames_per_video > 0 and len(local_samples) > max_frames_per_video:
                keep_idx = sample_indices_evenly(len(local_samples), max_frames_per_video)
                local_samples = [local_samples[i] for i in keep_idx]

            all_samples.extend(local_samples)

    if not all_samples:
        raise RuntimeError(f"No ROI frame samples were found under: {data_root}")
    return all_samples


def split_subjects(
    subject_labels: Dict[str, int],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    subjects = sorted(subject_labels.keys())
    y = [subject_labels[s] for s in subjects]

    if len(subjects) < 4:
        n = len(subjects)
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(0, int(round(n * val_ratio)))
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)
        rng = random.Random(seed)
        shuffled = subjects[:]
        rng.shuffle(shuffled)
        train_subjects = shuffled[:n_train]
        val_subjects = shuffled[n_train:n_train + n_val]
        test_subjects = shuffled[n_train + n_val:]
        if not test_subjects:
            test_subjects = [shuffled[-1]]
            if shuffled[-1] in train_subjects:
                train_subjects.remove(shuffled[-1])
        return sorted(train_subjects), sorted(val_subjects), sorted(test_subjects)

    try:
        train_subjects, temp_subjects, _, temp_y = train_test_split(
            subjects,
            y,
            test_size=max(0.05, 1.0 - train_ratio),
            random_state=seed,
            stratify=y,
        )
    except Exception:
        rng = random.Random(seed)
        shuffled = subjects[:]
        rng.shuffle(shuffled)
        n_train = max(1, int(round(len(shuffled) * train_ratio)))
        train_subjects = shuffled[:n_train]
        temp_subjects = shuffled[n_train:]
        temp_y = [subject_labels[s] for s in temp_subjects]

    if not temp_subjects:
        fallback = [s for s in subjects if s not in train_subjects]
        temp_subjects = fallback if fallback else [subjects[-1]]
        temp_y = [subject_labels[s] for s in temp_subjects]

    rem_ratio = max(1e-6, (1.0 - train_ratio))
    val_share = min(0.9, max(0.0, val_ratio / rem_ratio))

    if len(temp_subjects) < 2 or val_share <= 0.0:
        val_subjects = []
        test_subjects = temp_subjects
    else:
        try:
            val_subjects, test_subjects = train_test_split(
                temp_subjects,
                test_size=max(0.1, 1.0 - val_share),
                random_state=seed,
                stratify=temp_y,
            )
        except Exception:
            rng = random.Random(seed)
            shuffled = temp_subjects[:]
            rng.shuffle(shuffled)
            n_val = int(round(len(shuffled) * val_share))
            n_val = max(0, min(n_val, len(shuffled) - 1))
            val_subjects = shuffled[:n_val]
            test_subjects = shuffled[n_val:]

    if not test_subjects:
        test_subjects = val_subjects[-1:] if val_subjects else train_subjects[-1:]
        val_subjects = [s for s in val_subjects if s not in test_subjects]

    train_subjects = sorted(set(train_subjects) - set(test_subjects))
    val_subjects = sorted(set(val_subjects) - set(test_subjects))
    test_subjects = sorted(set(test_subjects))
    return train_subjects, val_subjects, test_subjects


def split_train_val_subjects_for_loso(
    subject_labels: Dict[str, int],
    val_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    subjects = sorted(subject_labels.keys())
    if not subjects:
        return [], []
    if len(subjects) == 1:
        return subjects, []

    val_ratio = max(0.0, min(0.8, float(val_ratio)))
    if val_ratio <= 0.0:
        return subjects, []

    y = [subject_labels[s] for s in subjects]
    n_val = max(1, int(round(len(subjects) * val_ratio)))
    n_val = min(n_val, len(subjects) - 1)
    test_size = n_val / float(len(subjects))

    try:
        train_subjects, val_subjects = train_test_split(
            subjects,
            test_size=test_size,
            random_state=seed,
            stratify=y,
        )
    except Exception:
        rng = random.Random(seed)
        shuffled = subjects[:]
        rng.shuffle(shuffled)
        val_subjects = shuffled[:n_val]
        train_subjects = shuffled[n_val:]

    train_subjects = sorted(set(train_subjects))
    val_subjects = sorted(set(val_subjects) - set(train_subjects))
    if not train_subjects and val_subjects:
        train_subjects = [val_subjects.pop()]
    return train_subjects, val_subjects


def filter_samples_by_subject(samples: List[FrameSample], subjects: List[str]) -> List[FrameSample]:
    subject_set = set(subjects)
    return [s for s in samples if s.subject in subject_set]


def safe_crop_roi(
    frame: np.ndarray,
    bbox: Tuple[float, float, float, float],
    pad_ratio: float = 0.0,
) -> np.ndarray:
    h_img, w_img = frame.shape[:2]
    x, y, w, h = bbox
    pad_ratio = max(0.0, float(pad_ratio))
    pad_w = w * pad_ratio
    pad_h = h * pad_ratio

    x1 = int(max(0, min(w_img - 1, round(x - pad_w))))
    y1 = int(max(0, min(h_img - 1, round(y - pad_h))))
    x2 = int(max(x1 + 1, min(w_img, round(x + w + pad_w))))
    y2 = int(max(y1 + 1, min(h_img, round(y + h + pad_h))))
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    return roi


def read_video_frame(video_path: str, frame_index: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        return frame
    finally:
        cap.release()


class ROIFrameDataset(Dataset):
    def __init__(self, samples: List[FrameSample], transform: transforms.Compose, roi_pad_ratio: float = 0.0):
        self.samples = samples
        self.transform = transform
        self.roi_pad_ratio = max(0.0, float(roi_pad_ratio))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        frame = read_video_frame(sample.video_path, sample.frame_index)
        if frame is None:
            roi = np.zeros((64, 64, 3), dtype=np.uint8)
        else:
            roi = safe_crop_roi(frame, sample.bbox, pad_ratio=self.roi_pad_ratio)
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(roi_rgb)
        tensor = self.transform(image)
        target = torch.tensor(float(sample.label), dtype=torch.float32)
        return tensor, target


def make_model(pretrained: bool) -> nn.Module:
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    return model


def select_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if y_true.size == 0 or y_prob.size == 0:
        return 0.5
    best_thr = 0.5
    best_j = -1e9
    for thr in np.linspace(0.05, 0.95, 91):
        y_pred = (y_prob >= thr).astype(np.int64)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tpr = tp / max(1, tp + fn)
        tnr = tn / max(1, tn + fp)
        youden_j = tpr + tnr - 1.0
        if youden_j > best_j:
            best_j = youden_j
            best_thr = float(thr)
    return best_thr


def run_inference(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs: List[float] = []
    labels: List[int] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x).squeeze(1)
            p = torch.sigmoid(logits).cpu().numpy().tolist()
            probs.extend(float(v) for v in p)
            labels.extend(int(v) for v in y.numpy().tolist())
    return np.asarray(labels, dtype=np.int64), np.asarray(probs, dtype=np.float32)


def aggregate_subject_level(samples: List[FrameSample], probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    subj_probs: Dict[str, List[float]] = {}
    subj_label: Dict[str, int] = {}
    for sample, prob in zip(samples, probs.tolist()):
        subj_probs.setdefault(sample.subject, []).append(float(prob))
        subj_label[sample.subject] = int(sample.label)

    subjects = sorted(subj_probs.keys())
    y_true: List[int] = []
    y_prob: List[float] = []
    for s in subjects:
        y_true.append(subj_label[s])
        y_prob.append(float(np.mean(subj_probs[s])))
    return np.asarray(y_true, dtype=np.int64), np.asarray(y_prob, dtype=np.float32), subjects


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    count = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x).squeeze(1)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        batch_size = x.shape[0]
        total_loss += float(loss.item()) * batch_size
        count += batch_size
    return total_loss / max(1, count)


def evaluate_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x).squeeze(1)
            loss = criterion(logits, y)
            batch_size = x.shape[0]
            total_loss += float(loss.item()) * batch_size
            count += batch_size
    return total_loss / max(1, count)


def main() -> None:
    parser = argparse.ArgumentParser(description="EfficientNet ROI end-to-end test (single-frame, subject-level split).")
    parser.add_argument("--data-root", type=Path, default=Path("dataset/merged_extend"))
    parser.add_argument("--label-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/efficientnet_roi_e2e"))
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--max-frames-per-video", type=int, default=24)
    parser.add_argument("--roi-pad-ratio", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--invert-labels", action="store_true", help="Flip label mapping: 1<->0 (for sanity check).")
    parser.add_argument("--loso", action="store_true", help="Run Leave-One-Subject-Out evaluation.")
    parser.add_argument("--loso-max-folds", type=int, default=0, help="Limit LOSO fold count (0 means all folds).")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    data_root = args.data_root.resolve()
    label_file = args.label_file.resolve() if args.label_file else (data_root / "ann.txt").resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_root.exists():
        raise FileNotFoundError(f"data root not found: {data_root}")
    if not label_file.exists():
        raise FileNotFoundError(f"label file not found: {label_file}")

    subject_labels = read_subject_labels(label_file)
    if args.invert_labels:
        subject_labels = {k: 1 - int(v) for k, v in subject_labels.items()}

    samples = build_samples(data_root, subject_labels, args.max_frames_per_video)
    all_subjects = sorted(set(s.subject for s in samples))

    labels_in_data = {k: v for k, v in subject_labels.items() if k in all_subjects}
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_run_dir = output_dir / (f"run_{ts}_loso" if args.loso else f"run_{ts}")
    root_run_dir.mkdir(parents=True, exist_ok=True)

    def run_single_split(
        train_subjects: List[str],
        val_subjects: List[str],
        test_subjects: List[str],
        run_dir: Path,
        *,
        fold_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        train_samples = filter_samples_by_subject(samples, train_subjects)
        val_samples = filter_samples_by_subject(samples, val_subjects)
        test_samples = filter_samples_by_subject(samples, test_subjects)

        if not train_samples or not test_samples:
            raise RuntimeError("Insufficient samples after subject-level split; check dataset and split ratios.")

        train_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        eval_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        train_ds = ROIFrameDataset(train_samples, transform=train_transform, roi_pad_ratio=args.roi_pad_ratio)
        val_ds = ROIFrameDataset(val_samples, transform=eval_transform, roi_pad_ratio=args.roi_pad_ratio) if val_samples else None
        test_ds = ROIFrameDataset(test_samples, transform=eval_transform, roi_pad_ratio=args.roi_pad_ratio)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True) if val_ds else None
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        device = torch.device(args.device)
        model = make_model(pretrained=not args.no_pretrained).to(device)

        train_y = np.asarray([s.label for s in train_samples], dtype=np.int64)
        pos = int((train_y == 1).sum())
        neg = int((train_y == 0).sum())
        pos_weight = torch.tensor([neg / max(1, pos)], dtype=torch.float32, device=device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_state: Optional[Dict[str, Any]] = None
        best_val_loss = float("inf")
        history: List[Dict[str, float]] = []

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            if val_loader is not None:
                val_loss = evaluate_loss(model, val_loader, criterion, device)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            else:
                val_loss = float("nan")
                if train_loss < best_val_loss:
                    best_val_loss = train_loss
                    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

            history.append({"epoch": float(epoch), "train_loss": float(train_loss), "val_loss": float(val_loss)})
            prefix = f"[{fold_name}] " if fold_name else ""
            print(f"{prefix}[Epoch {epoch:02d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if best_state is not None:
            model.load_state_dict(best_state)

        val_threshold = 0.5
        if val_loader is not None and len(val_samples) > 0:
            y_val_true, y_val_prob = run_inference(model, val_loader, device)
            val_threshold = select_best_threshold(y_val_true, y_val_prob)

        y_test_true, y_test_prob = run_inference(model, test_loader, device)
        y_test_pred = (y_test_prob >= val_threshold).astype(np.int64)
        frame_metrics = summarise_classification(
            y_true=y_test_true.tolist(),
            y_pred=y_test_pred.tolist(),
            y_prob=y_test_prob.tolist(),
            positive_label=1,
        )

        y_sub_true, y_sub_prob, sub_order = aggregate_subject_level(test_samples, y_test_prob)
        y_sub_pred = (y_sub_prob >= val_threshold).astype(np.int64)
        subject_metrics = summarise_classification(
            y_true=y_sub_true.tolist(),
            y_pred=y_sub_pred.tolist(),
            y_prob=y_sub_prob.tolist(),
            positive_label=1,
        )

        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), run_dir / "efficientnet_roi_binary.pt")

        frame_pred_rows = []
        for sample, prob, pred in zip(test_samples, y_test_prob.tolist(), y_test_pred.tolist()):
            frame_pred_rows.append({
                "subject": sample.subject,
                "video_path": sample.video_path,
                "frame_index": sample.frame_index,
                "label_true": int(sample.label),
                "prob_positive": float(prob),
                "label_pred": int(pred),
            })

        subject_pred_rows = []
        for s, yt, yp, yhat in zip(sub_order, y_sub_true.tolist(), y_sub_prob.tolist(), y_sub_pred.tolist()):
            subject_pred_rows.append({
                "subject": s,
                "label_true": int(yt),
                "prob_positive": float(yp),
                "label_pred": int(yhat),
            })

        report = {
            "config": {
                "data_root": str(data_root),
                "label_file": str(label_file),
                "img_size": args.img_size,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "train_ratio": args.train_ratio,
                "val_ratio": args.val_ratio,
                "max_frames_per_video": args.max_frames_per_video,
                "roi_pad_ratio": args.roi_pad_ratio,
                "seed": args.seed,
                "pretrained": not args.no_pretrained,
                "threshold": float(val_threshold),
                "invert_labels": bool(args.invert_labels),
                "loso": bool(args.loso),
            },
            "split": {
                "train_subjects": train_subjects,
                "val_subjects": val_subjects,
                "test_subjects": test_subjects,
                "n_train_samples": len(train_samples),
                "n_val_samples": len(val_samples),
                "n_test_samples": len(test_samples),
            },
            "history": history,
            "metrics": {
                "frame_level": frame_metrics,
                "subject_level": subject_metrics,
            },
            "predictions": {
                "frame_level": frame_pred_rows,
                "subject_level": subject_pred_rows,
            },
        }

        with (run_dir / "report.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print("\n=== EfficientNet ROI E2E (single-frame) ===")
        print(f"Run dir: {run_dir}")
        print(f"Threshold: {val_threshold:.4f}")
        print("\n[Frame-level metrics]")
        for k in ["accuracy", "balanced_accuracy", "precision_positive", "recall_positive", "f1_positive", "roc_auc", "brier_score"]:
            v = frame_metrics.get(k, float("nan"))
            print(f"- {k}: {v}")

        print("\n[Subject-level metrics]")
        for k in ["accuracy", "balanced_accuracy", "precision_positive", "recall_positive", "f1_positive", "roc_auc", "brier_score"]:
            v = subject_metrics.get(k, float("nan"))
            print(f"- {k}: {v}")
        return report

    if not args.loso:
        train_subjects, val_subjects, test_subjects = split_subjects(
            labels_in_data,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        run_single_split(train_subjects, val_subjects, test_subjects, root_run_dir)
        return

    folds = sorted(labels_in_data.keys())
    if args.loso_max_folds > 0:
        folds = folds[: args.loso_max_folds]

    loso_reports: List[Dict[str, Any]] = []
    pooled_frame_preds: List[Dict[str, Any]] = []
    pooled_subject_preds: List[Dict[str, Any]] = []
    for fold_subject in folds:
        rem_labels = {k: v for k, v in labels_in_data.items() if k != fold_subject}
        train_subjects, val_subjects = split_train_val_subjects_for_loso(
            rem_labels,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        test_subjects = [fold_subject]
        fold_dir = root_run_dir / f"fold_{fold_subject}"
        report = run_single_split(
            train_subjects,
            val_subjects,
            test_subjects,
            fold_dir,
            fold_name=f"LOSO:{fold_subject}",
        )
        loso_reports.append(report)
        pooled_frame_preds.extend(report.get("predictions", {}).get("frame_level", []))
        pooled_subject_preds.extend(report.get("predictions", {}).get("subject_level", []))

    def _summarise_pooled(pred_rows: List[Dict[str, Any]]) -> Dict[str, float]:
        y_true = [int(r["label_true"]) for r in pred_rows]
        y_pred = [int(r["label_pred"]) for r in pred_rows]
        y_prob = [float(r["prob_positive"]) for r in pred_rows]
        return summarise_classification(y_true=y_true, y_pred=y_pred, y_prob=y_prob, positive_label=1)

    loso_summary = {
        "config": {
            "data_root": str(data_root),
            "label_file": str(label_file),
            "img_size": args.img_size,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "val_ratio": args.val_ratio,
            "max_frames_per_video": args.max_frames_per_video,
            "roi_pad_ratio": args.roi_pad_ratio,
            "seed": args.seed,
            "pretrained": not args.no_pretrained,
            "invert_labels": bool(args.invert_labels),
            "loso": True,
            "n_folds": len(loso_reports),
        },
        "metrics": {
            "frame_level_pooled": _summarise_pooled(pooled_frame_preds) if pooled_frame_preds else {},
            "subject_level_pooled": _summarise_pooled(pooled_subject_preds) if pooled_subject_preds else {},
        },
        "folds": [
            {
                "test_subjects": rep.get("split", {}).get("test_subjects", []),
                "frame_level": rep.get("metrics", {}).get("frame_level", {}),
                "subject_level": rep.get("metrics", {}).get("subject_level", {}),
            }
            for rep in loso_reports
        ],
    }
    with (root_run_dir / "loso_summary.json").open("w", encoding="utf-8") as f:
        json.dump(loso_summary, f, ensure_ascii=False, indent=2)

    print("\n=== LOSO summary ===")
    print(f"Run dir: {root_run_dir}")
    print("[Subject-level pooled metrics]")
    for k in ["accuracy", "balanced_accuracy", "precision_positive", "recall_positive", "f1_positive", "roc_auc", "brier_score"]:
        v = loso_summary["metrics"].get("subject_level_pooled", {}).get(k, float("nan"))
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
