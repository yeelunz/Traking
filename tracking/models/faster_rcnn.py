from __future__ import annotations
from typing import Dict, Any, List, Optional
import os

import numpy as np

# Capture detailed import error to surface it later in UI/logs
_TORCH_TV_IMPORT_ERROR: Optional[Exception] = None
try:
    import torch
    import torchvision
except Exception as e:
    _TORCH_TV_IMPORT_ERROR = e
    torch = None  # type: ignore
    torchvision = None  # type: ignore

from ..core.interfaces import TrackingModel, FramePrediction, PreprocessingModule, Dataset
from ..core.registry import register_model
from ..utils.annotations import load_coco_vid


@register_model("FasterRCNN")
class FasterRCNNModel(TrackingModel):
    name = "FasterRCNN"
    DEFAULT_CONFIG = {
        "score_thresh": 0.5,
        "device": "cuda",
        "pretrained": True,
        "num_classes": 2,  # background + 1 default class
        # training params
        "epochs": 1,
        "batch_size": 2,
        "lr": 0.0025,
        "weight_decay": 0.0005,
        "momentum": 0.9,
        "step_size": 3,
        "gamma": 0.1,
        "num_workers": 0,  # Windows 建議 0 以避免多程序問題
    "pin_memory": False,
    # optimizer selection
    "optimizer": "AdamW",  # SGD | AdamW
    # AdamW-specific
    "adamw_betas": [0.9, 0.999],
    "adamw_eps": 1e-08,
    # train stability
    "grad_clip": 5.0,
    "detect_anomaly": False,
    # data handling
    "include_empty_frames": False,
    # prediction behavior
    "fallback_last_prediction": True,
    # inference efficiency
    "inference_batch": 4,
    }

    def __init__(self, config: Dict[str, Any]):
        if torch is None or torchvision is None:
            detail = f" underlying import error: {_TORCH_TV_IMPORT_ERROR!r}" if _TORCH_TV_IMPORT_ERROR else ""
            raise RuntimeError(f"PyTorch and torchvision are required for FasterRCNN model.{detail}")
        self.score_thresh = float(config.get("score_thresh", self.DEFAULT_CONFIG["score_thresh"]))
        self.device = str(config.get("device", self.DEFAULT_CONFIG["device"]))
        self.pretrained = bool(config.get("pretrained", self.DEFAULT_CONFIG["pretrained"]))
        self.num_classes = int(config.get("num_classes", self.DEFAULT_CONFIG["num_classes"]))
        # training params
        self.epochs = int(config.get("epochs", self.DEFAULT_CONFIG["epochs"]))
        self.batch_size = int(config.get("batch_size", self.DEFAULT_CONFIG["batch_size"]))
        self.lr = float(config.get("lr", self.DEFAULT_CONFIG["lr"]))
        self.weight_decay = float(config.get("weight_decay", self.DEFAULT_CONFIG["weight_decay"]))
        self.momentum = float(config.get("momentum", self.DEFAULT_CONFIG["momentum"]))
        self.step_size = int(config.get("step_size", self.DEFAULT_CONFIG["step_size"]))
        self.gamma = float(config.get("gamma", self.DEFAULT_CONFIG["gamma"]))
        self.num_workers = int(config.get("num_workers", self.DEFAULT_CONFIG["num_workers"]))
        self.pin_memory = bool(config.get("pin_memory", self.DEFAULT_CONFIG["pin_memory"]))
        # optimizer
        self.optimizer_name = str(config.get("optimizer", self.DEFAULT_CONFIG["optimizer"]))
        betas_val = config.get("adamw_betas", self.DEFAULT_CONFIG["adamw_betas"])  # type: ignore
        try:
            if isinstance(betas_val, (list, tuple)) and len(betas_val) >= 2:
                self.adamw_betas = (float(betas_val[0]), float(betas_val[1]))
            else:
                self.adamw_betas = (0.9, 0.999)
        except Exception:
            self.adamw_betas = (0.9, 0.999)
        self.adamw_eps = float(config.get("adamw_eps", self.DEFAULT_CONFIG["adamw_eps"]))
        self.grad_clip = float(config.get("grad_clip", self.DEFAULT_CONFIG["grad_clip"]))
        self.detect_anomaly = bool(config.get("detect_anomaly", self.DEFAULT_CONFIG["detect_anomaly"]))
        self.include_empty_frames = bool(config.get("include_empty_frames", self.DEFAULT_CONFIG["include_empty_frames"]))
        self.fallback_last_prediction = bool(config.get("fallback_last_prediction", self.DEFAULT_CONFIG["fallback_last_prediction"]))
        self.preprocs = []
        # build model
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT if self.pretrained else None
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        if self.num_classes != 91:  # default COCO classes; allow override if user sets num_classes
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, self.num_classes)
        self.model.eval()
        self._device = torch.device(self.device if (self.device == "cpu" or torch.cuda.is_available()) else "cpu")
        self.model.to(self._device)
        # Lightweight per-video frame reader cache to avoid reopening files for every frame during training
        self._vfcache = self._VideoFrameCache(max_open=4)

    class _VideoFrameCache:
        """LRU-style cache for cv2.VideoCapture to minimize open/close overhead.

        Not process-safe; intended for single-process dataloading (num_workers=0 on Windows).
        """
        def __init__(self, max_open: int = 4):
            self.max_open = max(1, int(max_open))
            self._caps: Dict[str, Any] = {}
            self._order: List[str] = []  # MRU at end

        def get_frame(self, video_path: str, frame_index: int):
            import cv2
            cap = self._caps.get(video_path)
            if cap is None or not getattr(cap, 'isOpened', lambda: False)():
                # open new capture, evict LRU if needed
                try:
                    cap = cv2.VideoCapture(video_path)
                except Exception:
                    cap = None
                if cap is None or not cap.isOpened():
                    return None
                self._caps[video_path] = cap
                if video_path in self._order:
                    self._order.remove(video_path)
                self._order.append(video_path)
                # evict
                while len(self._order) > self.max_open:
                    old = self._order.pop(0)
                    try:
                        c = self._caps.pop(old, None)
                        if c is not None:
                            c.release()
                    except Exception:
                        pass
            else:
                # mark as MRU
                try:
                    if video_path in self._order:
                        self._order.remove(video_path)
                    self._order.append(video_path)
                except Exception:
                    pass
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame = cap.read()
                if not ok:
                    return None
                return frame
            except Exception:
                return None

        def close_all(self):
            for k, c in list(self._caps.items()):
                try:
                    c.release()
                except Exception:
                    pass
            self._caps.clear()
            self._order.clear()

    def _apply_preprocs_np(self, frame_bgr: np.ndarray) -> np.ndarray:
        if not self.preprocs:
            return frame_bgr
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        for p in self.preprocs:
            rgb = p.apply_to_frame(rgb)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _to_tensor(self, img_rgb: np.ndarray):
        """Convert HxWxC uint8 RGB numpy array to torch.FloatTensor CxHxW in [0,1]."""
        if torch is None:
            raise RuntimeError("Torch not available")
        # Ensure contiguous array
        if not isinstance(img_rgb, np.ndarray):
            img_rgb = np.array(img_rgb)
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).contiguous()
        if tensor.dtype != torch.float32:
            tensor = tensor.to(torch.float32)
        return tensor.div(255.0)

    def _build_train_index(self, dataset: Dataset) -> List[Dict[str, Any]]:
        """
        將 SimpleDataset 形式的 per-video 項目展開為 frame 級索引。
        回傳列表中每個元素包含 {video_path, frame_index, bboxes(list[(x,y,w,h)])}
        僅使用標註 JSON 中存在的 frame_index；若該 frame 無標註，仍會產生空框樣本。
        """
        index: List[Dict[str, Any]] = []
        for i in range(len(dataset)):
            item = dataset[i]
            vp: str = item["video_path"]
            frames = {}
            # 優先使用 Dataset 提供的 annotation（避免重讀與路徑問題）
            ann_raw = item.get("annotation") if isinstance(item, dict) else None
            if isinstance(ann_raw, dict) and ann_raw:
                try:
                    # 解析與 load_coco_vid 相同邏輯
                    img_to_frame = {}
                    for img in ann_raw.get("images", []):
                        fi = img.get("frame_index")
                        if fi is None:
                            fi = img.get("id", 0) - 1
                        img_to_frame[img["id"]] = fi
                        # 不先建立空 frame，避免誤將無 GT 幀加入；後續按 include_empty_frames 控制
                    for ann in ann_raw.get("annotations", []):
                        img_id = ann.get("image_id")
                        bbox = ann.get("bbox")
                        if img_id is None or bbox is None:
                            continue
                        fi = img_to_frame.get(img_id)
                        if fi is None:
                            continue
                        frames.setdefault(fi, []).append(tuple(bbox))
                except Exception:
                    frames = {}
            # 若無法從記憶體取得，退回磁碟讀取
            if not frames:
                json_path = os.path.splitext(vp)[0] + ".json"
                if not os.path.exists(json_path):
                    # 無標註則略過此影片
                    continue
                ann = load_coco_vid(json_path)
                frames = ann.get("frames", {})
            # 使用已知 frame 索引；可選擇是否包含無 GT 幀
            for fi, bboxes in frames.items():
                if not self.include_empty_frames and (not bboxes or len(bboxes) == 0):
                    continue
                index.append({
                    "video_path": vp,
                    "frame_index": int(fi),
                    "bboxes": list(bboxes or []),
                })
        # 依 (video, frame_index) 排序，確保可重現
        index.sort(key=lambda x: (x["video_path"], x["frame_index"]))
        return index

    def _collate_fn(self, batch):
        images, targets = zip(*batch)
        return list(images), list(targets)

    def _read_frame(self, video_path: str, frame_index: int) -> Optional[np.ndarray]:
        # Use reusable cache to avoid repeatedly opening the same video during training
        return self._vfcache.get_frame(video_path, frame_index)

    def _targets_from_bboxes(self, bboxes: List[Any], img_h: Optional[int] = None, img_w: Optional[int] = None, image_id: Optional[int] = None) -> Dict[str, Any]:
        # 轉換為 FasterRCNN 期望的 target 結構
        if torch is None:
            raise RuntimeError("Torch not available")
        if len(bboxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
        else:
            # bbox 格式 (x,y,w,h) -> (x1,y1,x2,y2)
            xyxy = []
            areas = []
            # 嘗試偵測是否為 0~1 歸一化座標（全部框皆 <=1.0）
            normalized = False
            try:
                if img_w is not None and img_h is not None and len(bboxes) > 0:
                    normalized = all((0.0 <= float(b[0]) <= 1.0 and 0.0 <= float(b[1]) <= 1.0 and 0.0 < float(b[2]) <= 1.0 and 0.0 < float(b[3]) <= 1.0) for b in bboxes)
            except Exception:
                normalized = False
            for b in bboxes:
                try:
                    x, y, w, h = [float(v) for v in b]
                except Exception:
                    # 跳過不可轉 float 的框
                    continue
                # 過濾非有限值
                if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(w) and np.isfinite(h)):
                    continue
                if normalized and img_w is not None and img_h is not None:
                    # 將 0~1 座標轉為像素
                    x *= float(img_w)
                    y *= float(img_h)
                    w *= float(img_w)
                    h *= float(img_h)
                # 確保正尺寸
                w = max(1.0, float(w))
                h = max(1.0, float(h))
                x1, y1, x2, y2 = x, y, x + w, y + h
                # 依影像大小裁切（若提供）
                if img_w is not None and img_h is not None:
                    # 先粗略過濾極端越界（>10x 邊長）
                    if (x2 < -float(img_w) * 10) or (y2 < -float(img_h) * 10) or (x1 > float(img_w) * 10) or (y1 > float(img_h) * 10):
                        continue
                    x1 = float(max(0.0, min(float(img_w - 1), x1)))
                    y1 = float(max(0.0, min(float(img_h - 1), y1)))
                    x2 = float(max(0.0, min(float(img_w), x2)))
                    y2 = float(max(0.0, min(float(img_h), y2)))
                    # 確保 x2>x1, y2>y1
                    if x2 <= x1:
                        x2 = min(float(img_w), x1 + 1.0)
                    if y2 <= y1:
                        y2 = min(float(img_h), y1 + 1.0)
                # 再次確認有限且有效
                if not (np.isfinite(x1) and np.isfinite(y1) and np.isfinite(x2) and np.isfinite(y2)):
                    continue
                if (x2 - x1) < 1.0 or (y2 - y1) < 1.0:
                    continue
                xyxy.append([x1, y1, x2, y2])
                areas.append(max(1.0, (x2 - x1)) * max(1.0, (y2 - y1)))
            if len(xyxy) == 0:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
                area = torch.zeros((0,), dtype=torch.float32)
            else:
                boxes = torch.tensor(xyxy, dtype=torch.float32)
                labels = torch.ones((boxes.shape[0],), dtype=torch.int64)  # 單一前景類別 id=1
                area = torch.tensor(areas, dtype=torch.float32)
        target: Dict[str, Any] = {
            "boxes": boxes,
            "labels": labels,
            "area": area,
            "iscrowd": torch.zeros((labels.shape[0],), dtype=torch.int64),
        }
        if image_id is not None:
            target["image_id"] = torch.tensor([int(image_id)])
        return target

    def train(self, train_dataset: Dataset, val_dataset: Optional[Dataset] = None, seed: int = 0, output_dir: Optional[str] = None):
        if torch is None or torchvision is None:
            detail = f" underlying import error: {_TORCH_TV_IMPORT_ERROR!r}" if _TORCH_TV_IMPORT_ERROR else ""
            raise RuntimeError(f"PyTorch and torchvision are required for FasterRCNN training.{detail}")

        # 建立 frame-level 索引
        train_index = self._build_train_index(train_dataset)
        if len(train_index) == 0:
            return {"status": "no_data"}

        # 建立 torch Dataset
        class FrameDataset(torch.utils.data.Dataset):  # type: ignore
            def __init__(self, outer: "FasterRCNNModel", index: List[Dict[str, Any]]):
                self.outer = outer
                self.index = index

            def __len__(self):
                return len(self.index)

            def __getitem__(self, i):
                rec = self.index[i]
                frame = self.outer._read_frame(rec["video_path"], rec["frame_index"])
                if frame is None:
                    # 回傳一個空白樣本以避免中斷；讓 collate 仍可處理
                    img = (np.zeros((32, 32, 3), dtype=np.uint8))
                    bbs: List[Any] = []
                else:
                    img = frame
                    bbs = rec.get("bboxes", [])
                # 套用預處理（保持與預測一致）
                img = self.outer._apply_preprocs_np(img)
                # 轉 RGB -> tensor
                import cv2
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                tensor = self.outer._to_tensor(img_rgb)
                h, w = img_rgb.shape[0], img_rgb.shape[1]
                target = self.outer._targets_from_bboxes(bbs, img_h=h, img_w=w, image_id=i)
                # 方便除錯：攜帶來源資訊（非 Tensor，不會搬到 GPU）
                try:
                    target["meta_video"] = rec.get("video_path")
                    target["meta_frame_index"] = rec.get("frame_index")
                except Exception:
                    pass
                return tensor, target

        ds_train = FrameDataset(self, train_index)
        dl_train = torch.utils.data.DataLoader(
            ds_train,
            batch_size=max(1, self.batch_size),
            shuffle=True,
            num_workers=max(0, self.num_workers),
            pin_memory=bool(self.pin_memory) if self._device.type == "cuda" else False,
            collate_fn=self._collate_fn,
        )

        # 驗證資料（可選）
        dl_val = None
        if val_dataset is not None:
            val_index = self._build_train_index(val_dataset)
            if len(val_index) > 0:
                ds_val = FrameDataset(self, val_index)
                dl_val = torch.utils.data.DataLoader(
                    ds_val,
                    batch_size=max(1, self.batch_size),
                    shuffle=False,
                    num_workers=max(0, self.num_workers),
                    pin_memory=bool(self.pin_memory) if self._device.type == "cuda" else False,
                    collate_fn=self._collate_fn,
                )

        # Optimizer & Scheduler
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt_name = (self.optimizer_name or "SGD").strip().lower()
        if opt_name == "adamw":
            optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay, betas=self.adamw_betas, eps=self.adamw_eps)
        else:
            optimizer = torch.optim.SGD(params, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        # Scheduler（防呆：step_size<=0 會造成 ZeroDivisionError）
        try:
            safe_step = int(self.step_size) if isinstance(self.step_size, (int, float)) else 1
        except Exception:
            safe_step = 1
        if safe_step < 1:
            try:
                print("[FasterRCNN] Invalid step_size<=0 provided; defaulting to 1 to avoid ZeroDivisionError.")
            except Exception:
                pass
        safe_step = max(1, int(safe_step))
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=safe_step, gamma=self.gamma)

        history = {"train_loss": [], "val_loss": []}

        self.model.train()
        self.model.to(self._device)
        # 可選的 autograd anomaly 偵測（會變慢，但有助定位 NaN 來源）
        if bool(self.detect_anomaly):
            try:
                torch.autograd.set_detect_anomaly(True)
            except Exception:
                pass

        for epoch in range(max(1, self.epochs)):
            try:
                cb = getattr(self, 'progress_callback', None)
                if callable(cb):
                    cb('train_epoch_start', epoch+1, int(self.epochs))
            except Exception:
                pass
            epoch_loss = 0.0
            n_batches = 0
            for bi, (images, targets) in enumerate(dl_train):
                images = [img.to(self._device) for img in images]
                targets = [{k: v.to(self._device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
                # 早期批次顯示 box 統計資訊
                if epoch == 0 and bi < 2:
                    try:
                        n_boxes = [int(t["boxes"].shape[0]) for t in targets]
                        all_boxes = torch.cat([t["boxes"] for t in targets], dim=0) if sum(n_boxes) > 0 else None
                        if all_boxes is not None:
                            mn = torch.min(all_boxes).item()
                            mx = torch.max(all_boxes).item()
                            print(f"[FasterRCNN][Debug] batch={bi} boxes_total={int(sum(n_boxes))} range=({mn:.2f},{mx:.2f})")
                        else:
                            print(f"[FasterRCNN][Debug] batch={bi} boxes_total=0")
                    except Exception:
                        pass
                loss_dict = self.model(images, targets)
                # 記錄各項 loss 並檢測 NaN
                def _to_safe_float(x: Any) -> float:
                    try:
                        if torch.is_tensor(x):
                            x = x.detach()
                            return float(x.item())
                        return float(x)
                    except Exception:
                        return float("nan")
                l_cls = _to_safe_float(loss_dict.get("loss_classifier", 0.0))
                l_box = _to_safe_float(loss_dict.get("loss_box_reg", 0.0))
                l_obj = _to_safe_float(loss_dict.get("loss_objectness", 0.0))
                l_rpn = _to_safe_float(loss_dict.get("loss_rpn_box_reg", 0.0))

                losses = sum(loss for loss in loss_dict.values())
                # NaN/Inf 防呆
                if not torch.isfinite(losses):
                    try:
                        print(f"[FasterRCNN][Epoch {epoch+1}] Non-finite loss detected: total={_to_safe_float(losses)} cls={l_cls} box={l_box} obj={l_obj} rpn={l_rpn}. Skipping batch {bi}.")
                        # 額外檢查輸入是否為有限值
                        imgs_ok = all(torch.isfinite(im).all().item() for im in images)
                        boxes_ok = True
                        try:
                            for t in targets:
                                if torch.is_tensor(t.get("boxes")) and not torch.isfinite(t["boxes"]).all().item():
                                    boxes_ok = False
                                    break
                        except Exception:
                            pass
                        if not imgs_ok or not boxes_ok:
                            print(f"[FasterRCNN][Epoch {epoch+1}] Input anomaly: images_finite={imgs_ok} boxes_finite={boxes_ok}")
                        # 列印此批資料來源，協助定位
                        try:
                            for ti, t in enumerate(targets):
                                mv = t.get("meta_video", "?")
                                mfi = t.get("meta_frame_index", "?")
                                nbox = int(t.get("boxes").shape[0]) if torch.is_tensor(t.get("boxes")) else -1
                                print(f"  - sample {ti}: video={os.path.basename(str(mv))} frame={mfi} n_boxes={nbox}")
                        except Exception:
                            pass
                    except Exception:
                        pass
                    # 略過此批次
                    continue
                optimizer.zero_grad()
                losses.backward()
                # 梯度裁剪避免梯度爆炸（可配置）
                try:
                    if self.grad_clip and self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(self.grad_clip))
                except Exception:
                    pass
                optimizer.step()
                epoch_loss += float(losses.item())
                n_batches += 1
            lr_scheduler.step()
            epoch_loss = epoch_loss / max(1, n_batches)
            history["train_loss"].append(epoch_loss)

            # 簡易驗證（以 loss 為度量）
            val_epoch_loss = None
            if dl_val is not None:
                self.model.train()  # 計算 loss 需在 train 模式
                with torch.no_grad():
                    v_loss_sum, v_n = 0.0, 0
                    for images, targets in dl_val:
                        images = [img.to(self._device) for img in images]
                        targets = [{k: v.to(self._device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
                        loss_dict = self.model(images, targets)
                        losses = sum(loss for loss in loss_dict.values())
                        v_loss_sum += float(losses.item())
                        v_n += 1
                    val_epoch_loss = v_loss_sum / max(1, v_n)
                    history["val_loss"].append(val_epoch_loss)

            # 簡易列印
            try:
                print(f"[FasterRCNN][Epoch {epoch+1}/{self.epochs}] train_loss={epoch_loss:.4f} val_loss={val_epoch_loss if val_epoch_loss is not None else 'NA'}")
            except Exception:
                pass
            try:
                cb = getattr(self, 'progress_callback', None)
                if callable(cb):
                    cb('train_epoch_end', epoch+1, int(self.epochs))
            except Exception:
                pass

        # 儲存 checkpoint
        if output_dir:
            try:
                os.makedirs(output_dir, exist_ok=True)
                ckpt_path = os.path.join(output_dir, f"{self.name}.pth")
                torch.save(self.model.state_dict(), ckpt_path)
                # 也存訓練紀錄
                import json
                with open(os.path.join(output_dir, f"{self.name}_history.json"), "w", encoding="utf-8") as f:
                    json.dump(history, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

        # 切回 eval 模式
        self.model.eval()
        # Close any cached video handles
        try:
            if hasattr(self, "_vfcache"):
                self._vfcache.close_all()
        except Exception:
            pass
        return {"status": "ok", "history": history}

    def load_checkpoint(self, ckpt_path: str):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(ckpt_path)
        state = torch.load(ckpt_path, map_location=self._device)
        self.model.load_state_dict(state)
        self.model.to(self._device)
        self.model.eval()

    def predict(self, video_path: str) -> List[FramePrediction]:
        if torch is None or torchvision is None:
            detail = f" underlying import error: {_TORCH_TV_IMPORT_ERROR!r}" if _TORCH_TV_IMPORT_ERROR else ""
            raise RuntimeError(f"PyTorch/torchvision not available. Install torch & torchvision to use FasterRCNN.{detail}")
        # detection-based: 對每幀取最高分框作為單目標預測
        import cv2
        # Ensure eval mode for inference
        if hasattr(self, "model"):
            try:
                self.model.eval()
            except Exception:
                pass
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        preds: List[FramePrediction] = []
        idx = 0
        last_bbox = None  # type: Optional[tuple]
        # small-batch inference to reduce per-call overhead
        try:
            batch_size = int(getattr(self, "inference_batch", 4) or 4)
        except Exception:
            batch_size = 4
        frames_buf: List[Any] = []
        indices_buf: List[int] = []
        def _flush_batch():
            nonlocal last_bbox
            if not frames_buf:
                return
            with torch.no_grad():
                tensors = []
                for fr in frames_buf:
                    img = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                    tensors.append(self._to_tensor(img).to(self._device))
                outputs_list = self.model(tensors)
                for out, fidx in zip(outputs_list, indices_buf):
                    scores = out.get("scores")
                    boxes = out.get("boxes")
                    if scores is not None and boxes is not None and len(scores) > 0:
                        try:
                            thresh = float(getattr(self, "score_thresh", 0.0))
                        except Exception:
                            thresh = 0.0
                        best_idx = None
                        try:
                            above = (scores >= thresh).nonzero(as_tuple=False)
                            if above is not None and above.numel() > 0:
                                cand_idx = above.view(-1)
                                cand_scores = scores[cand_idx]
                                rel = int(torch.argmax(cand_scores).item())
                                best_idx = int(cand_idx[rel].item())
                        except Exception:
                            best_idx = None
                        if best_idx is None:
                            try:
                                best_idx = int(torch.argmax(scores).item())
                            except Exception:
                                best_idx = None
                        if best_idx is not None:
                            x1, y1, x2, y2 = boxes[best_idx].tolist()
                            bbox = (float(x1), float(y1), float(max(1.0, x2 - x1)), float(max(1.0, y2 - y1)))
                            last_bbox = bbox
                            sc = float(scores[best_idx].item()) if hasattr(scores, "device") else float(scores[best_idx])
                            preds.append(FramePrediction(int(fidx), bbox, sc))
                        else:
                            if self.fallback_last_prediction and last_bbox is not None:
                                preds.append(FramePrediction(int(fidx), last_bbox, None))
                    else:
                        if self.fallback_last_prediction and last_bbox is not None:
                            preds.append(FramePrediction(int(fidx), last_bbox, None))
            frames_buf.clear()
            indices_buf.clear()
        with torch.no_grad():
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = self._apply_preprocs_np(frame)
                frames_buf.append(frame)
                indices_buf.append(idx)
                if len(frames_buf) >= max(1, batch_size):
                    _flush_batch()
                idx += 1
            # flush remaining
            _flush_batch()
        cap.release()
        return preds

    # Optional sparse inference for evaluation on specific frames
    def predict_frames(self, video_path: str, frame_indices: List[int]) -> List[FramePrediction]:
        if torch is None or torchvision is None:
            detail = f" underlying import error: {_TORCH_TV_IMPORT_ERROR!r}" if _TORCH_TV_IMPORT_ERROR else ""
            raise RuntimeError(f"PyTorch/torchvision not available. Install torch & torchvision to use FasterRCNN.{detail}")
        import cv2
        if hasattr(self, "model"):
            try:
                self.model.eval()
            except Exception:
                pass
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        preds: List[FramePrediction] = []
        last_bbox = None  # type: Optional[tuple]
        try:
            batch_size = int(getattr(self, "inference_batch", 4) or 4)
        except Exception:
            batch_size = 4
        buf_frames: List[Any] = []
        buf_indices: List[int] = []
        def _flush():
            nonlocal last_bbox
            if not buf_frames:
                return
            with torch.no_grad():
                tensors = []
                for fr in buf_frames:
                    img = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                    tensors.append(self._to_tensor(img).to(self._device))
                outs = self.model(tensors)
                for out, fidx in zip(outs, buf_indices):
                    scores = out.get("scores")
                    boxes = out.get("boxes")
                    if scores is not None and boxes is not None and len(scores) > 0:
                        try:
                            thresh = float(getattr(self, "score_thresh", 0.0))
                        except Exception:
                            thresh = 0.0
                        best_idx = None
                        try:
                            above = (scores >= thresh).nonzero(as_tuple=False)
                            if above is not None and above.numel() > 0:
                                cand_idx = above.view(-1)
                                cand_scores = scores[cand_idx]
                                rel = int(torch.argmax(cand_scores).item())
                                best_idx = int(cand_idx[rel].item())
                        except Exception:
                            best_idx = None
                        if best_idx is None:
                            try:
                                best_idx = int(torch.argmax(scores).item())
                            except Exception:
                                best_idx = None
                        if best_idx is not None:
                            x1, y1, x2, y2 = boxes[best_idx].tolist()
                            bbox = (float(x1), float(y1), float(max(1.0, x2 - x1)), float(max(1.0, y2 - y1)))
                            last_bbox = bbox
                            sc = float(scores[best_idx].item()) if hasattr(scores, "device") else float(scores[best_idx])
                            preds.append(FramePrediction(int(fidx), bbox, sc))
                        else:
                            if self.fallback_last_prediction and last_bbox is not None:
                                preds.append(FramePrediction(int(fidx), last_bbox, None))
                    else:
                        if self.fallback_last_prediction and last_bbox is not None:
                            preds.append(FramePrediction(int(fidx), last_bbox, None))
            buf_frames.clear()
            buf_indices.clear()
        with torch.no_grad():
            for idx in sorted(set(int(i) for i in frame_indices)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if not ok:
                    continue
                frame = self._apply_preprocs_np(frame)
                buf_frames.append(frame)
                buf_indices.append(int(idx))
                if len(buf_frames) >= max(1, batch_size):
                    _flush()
            _flush()
        cap.release()
        return preds
