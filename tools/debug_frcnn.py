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
import argparse
import json
import os
import statistics
from typing import Any, Dict, List

import torch
import numpy as np

# Import project modules (assumes running from repo root or PYTHONPATH set)
from tracking.data.dataset_manager import COCOJsonDatasetManager, SimpleDataset
from tracking.models.faster_rcnn import FasterRCNNModel
from tracking.utils.seed import set_seed

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


def load_yaml_params(yaml_path: str) -> Dict[str, Any]:
    if not yaml or not os.path.exists(yaml_path):
        return {}
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    # navigate to first experiment model params if present
    try:
        exps = data.get('experiments', [])
        if exps:
            pipe = exps[0].get('pipeline', [])
            for step in pipe:
                if step.get('type') == 'model' and step.get('name') == 'FasterRCNN':
                    return step.get('params', {})
    except Exception:
        pass
    return {}


def build_train_index_preview(model: FasterRCNNModel, dataset: SimpleDataset, limit_videos: int = 5, limit_frames: int = 5):
    """Peek at first few expanded frame records to verify frame_index / bbox sanity."""
    expanded = model._build_train_index(dataset)  # type: ignore (accessing internal for debug)
    preview = expanded[: min(len(expanded), limit_videos * limit_frames)]
    rows = []
    for r in preview:
        rows.append({
            'video': os.path.basename(r['video_path']),
            'frame_index': r['frame_index'],
            'n_bboxes': len(r.get('bboxes', [])),
            'first_bbox': r.get('bboxes', [None])[0]
        })
    return rows, len(expanded)


def visualize_batch(viz_dir: str, batch_idx: int, images: List[torch.Tensor], targets: List[Dict[str, Any]]):
    try:
        import cv2  # type: ignore
    except Exception:
        print('[WARN] OpenCV not available, skip visualization.')
        return
    os.makedirs(viz_dir, exist_ok=True)
    for i, (img_t, tgt) in enumerate(zip(images, targets)):
        # img_t is CxHxW in [0,1]
        arr = (img_t.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        arr = np.transpose(arr, (1, 2, 0))  # HWC RGB
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        h, w = arr.shape[:2]
        for b in tgt.get('boxes', []):
            if not torch.is_tensor(b):
                continue
        boxes = tgt.get('boxes')
        if torch.is_tensor(boxes):
            for b in boxes.detach().cpu().numpy():
                x1, y1, x2, y2 = map(float, b)
                cv2.rectangle(arr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        out_path = os.path.join(viz_dir, f'batch{batch_idx}_img{i}.jpg')
        cv2.imwrite(out_path, arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='Dataset root (videos + json)')
    ap.add_argument('--yaml', default='def_frcnn.yaml', help='YAML config to read default params')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-batches', type=int, default=5)
    ap.add_argument('--train-step', action='store_true', help='Actually run backward/optimizer (default: no)')
    ap.add_argument('--out-json', default='debug_frcnn_batches.json')
    ap.add_argument('--viz-dir', default=None, help='If set, save GT overlay images for each batch')
    ap.add_argument('--device', default=None, help='Force device, override YAML (cuda/cpu)')
    args = ap.parse_args()

    set_seed(args.seed, deterministic=False)

    dm = COCOJsonDatasetManager(args.root)
    split = dm.split(method='video_level', seed=args.seed, ratios=(0.8, 0.0, 0.2))
    train_ds: SimpleDataset = split['train']

    params = load_yaml_params(args.yaml)
    if args.device:
        params['device'] = args.device
    print('[INFO] Model params:', params)
    model = FasterRCNNModel(params)

    # quick expand preview
    prev_rows, total_frames = build_train_index_preview(model, train_ds)
    print(f'[INFO] Expanded frame-level samples: {total_frames}. Preview:')
    for r in prev_rows[:10]:
        print('   ', r)

    # reconstruct the frame dataset like train() does
    train_index = model._build_train_index(train_ds)  # type: ignore

    class _FrameDataset(torch.utils.data.Dataset):  # type: ignore
        def __init__(self, outer, index):
            self.outer = outer
            self.index = index
        def __len__(self):
            return len(self.index)
        def __getitem__(self, i):
            rec = self.index[i]
            frame = self.outer._read_frame(rec['video_path'], rec['frame_index'])
            if frame is None:
                import numpy as _np
                frame = _np.zeros((32, 32, 3), dtype=_np.uint8)
                bbs = []
            else:
                bbs = rec.get('bboxes', [])
            import cv2 as _cv2
            frame = self.outer._apply_preprocs_np(frame)
            rgb = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
            tensor = self.outer._to_tensor(rgb)
            h, w = rgb.shape[:2]
            tgt = self.outer._targets_from_bboxes(bbs, img_h=h, img_w=w)
            tgt['meta_video'] = rec['video_path']
            tgt['meta_frame_index'] = rec['frame_index']
            return tensor, tgt

    ds = _FrameDataset(model, train_index)
    if len(ds) == 0:
        print('[ERROR] No training samples found.')
        return

    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=params.get('batch_size', 2),
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: (list(list(zip(*b))[0]), list(list(zip(*b))[1]))
    )

    model.model.train()
    device = torch.device(params.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    model.model.to(device)

    optimizer = torch.optim.AdamW([p for p in model.model.parameters() if p.requires_grad], lr=params.get('lr', 0.0005))

    batch_stats: List[Dict[str, Any]] = []
    for bi, (images, targets) in enumerate(dl):
        if bi >= args.max_batches:
            break
        images = [im.to(device) for im in images]
        targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()} for t in targets]
        loss_dict = model.model(images, targets)
        total_loss = sum(loss_dict.values())
        # gather per-image stats
        img_shapes = [tuple(im.shape) for im in images]
        per_img_boxes = [int(t['boxes'].shape[0]) for t in targets]
        all_boxes = torch.cat([t['boxes'] for t in targets], dim=0) if sum(per_img_boxes) > 0 else torch.zeros((0,4))
        box_w = (all_boxes[:,2] - all_boxes[:,0]).cpu().numpy() if all_boxes.numel() else []
        box_h = (all_boxes[:,3] - all_boxes[:,1]).cpu().numpy() if all_boxes.numel() else []
        stat = {
            'batch_index': bi,
            'loss_total': float(total_loss.item()),
            **{k: float(v.item()) for k, v in loss_dict.items()},
            'n_images': len(images),
            'img_shapes': img_shapes,  # (C,H,W)
            'per_image_box_counts': per_img_boxes,
            'box_w_mean': float(np.mean(box_w)) if len(box_w) else None,
            'box_w_max': float(np.max(box_w)) if len(box_w) else None,
            'box_h_mean': float(np.mean(box_h)) if len(box_h) else None,
            'box_h_max': float(np.max(box_h)) if len(box_h) else None,
            'videos': [os.path.basename(t.get('meta_video','')) for t in targets],
            'frame_indices': [int(t.get('meta_frame_index',-1)) for t in targets],
        }
        print(f"[B{bi}] total={stat['loss_total']:.4f} cls={stat.get('loss_classifier',0):.4f} box={stat.get('loss_box_reg',0):.4f} obj={stat.get('loss_objectness',0):.4f} rpn={stat.get('loss_rpn_box_reg',0):.4f} boxes={sum(per_img_boxes)} w_mean={stat['box_w_mean']} h_mean={stat['box_h_mean']}")
        batch_stats.append(stat)

        if args.viz_dir:
            visualize_batch(args.viz_dir, bi, images, targets)

        if args.train_step:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    # outlier detection
    if batch_stats:
        losses = [b['loss_total'] for b in batch_stats]
        med = statistics.median(losses)
        mad = statistics.median([abs(x - med) for x in losses]) if len(losses) > 1 else 0.0
        for b in batch_stats:
            if mad > 0 and (b['loss_total'] - med) / (mad + 1e-9) > 10:
                b['outlier_mad10'] = True
            else:
                b['outlier_mad10'] = False
        print('[INFO] Median loss:', med, 'MAD:', mad)
        print('[INFO] Outlier batches:', [b['batch_index'] for b in batch_stats if b['outlier_mad10']])

    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(batch_stats, f, ensure_ascii=False, indent=2)
    print('[INFO] Saved batch stats to', args.out_json)


if __name__ == '__main__':
    main()
