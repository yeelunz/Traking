from __future__ import annotations
from typing import Dict, Any, List, Optional
import os

import numpy as np

# Lazy/safe import of Ultralytics
_ULTRA_IMPORT_ERROR: Optional[Exception] = None
try:
    from ultralytics import YOLO  # type: ignore
except Exception as e:  # pragma: no cover - environment without ultralytics
    _ULTRA_IMPORT_ERROR = e
    YOLO = None  # type: ignore

from ..core.interfaces import TrackingModel, FramePrediction, PreprocessingModule
from ..core.registry import register_model
from ..utils.init_bbox import resolve_weights_path


@register_model("YOLOv11")
class YOLOv11Model(TrackingModel):
    """Single-object tracking via per-frame detection using Ultralytics YOLOv11.

    Strategy: run detector on each frame and take the highest-confidence bbox.
    When no detection, optionally re-use previous bbox (fallback).
    """
    name = "YOLOv11"
    DEFAULT_CONFIG = {
        # Inference
        "weights": "yolo11n.pt",
        "conf": 0.25,
        "iou": 0.5,
        "imgsz": 640,
        "device": "cuda",     # "cuda" | "cpu"
        "classes": None,        # e.g., [0] for person; None for all
        "max_det": 100,
        "fallback_last_prediction": True,
        "train_enabled": True,
        # Training
        "epochs": 5,
        "batch": 8,
        "lr0": 0.01,
        "patience": 50,
        "workers": 0,           # Windows 建議 0 以避免多程序問題
        "include_empty_frames": False,
    }

    def __init__(self, config: Dict[str, Any]):
        if YOLO is None:
            detail = f" underlying import error: {_ULTRA_IMPORT_ERROR!r}" if _ULTRA_IMPORT_ERROR else ""
            raise RuntimeError(f"Ultralytics 'ultralytics' package is required for YOLOv11 model.{detail}")

        # --- Config params ---
        self.weights = str(config.get("weights", self.DEFAULT_CONFIG["weights"]))
        self._weights_path = resolve_weights_path(self.weights)
        self.conf = float(config.get("conf", self.DEFAULT_CONFIG["conf"]))
        self.iou = float(config.get("iou", self.DEFAULT_CONFIG["iou"]))
        self.imgsz = int(config.get("imgsz", self.DEFAULT_CONFIG["imgsz"]))
        self.device = str(config.get("device", self.DEFAULT_CONFIG["device"]))
        self.classes = config.get("classes", self.DEFAULT_CONFIG["classes"])  # type: ignore
        self.max_det = int(config.get("max_det", self.DEFAULT_CONFIG["max_det"]))
        self.fallback_last_prediction = bool(config.get("fallback_last_prediction", self.DEFAULT_CONFIG["fallback_last_prediction"]))
        self.include_empty_frames = bool(config.get("include_empty_frames", self.DEFAULT_CONFIG.get("include_empty_frames", False)))

        # --- Training-specific (之前未保存 -> 導致 epochs 等使用預設值) ---
        # 保留在實例屬性上，train() 透過 getattr 可取到覆寫值
        self.epochs = int(config.get("epochs", self.DEFAULT_CONFIG["epochs"]))
        self.batch = int(config.get("batch", self.DEFAULT_CONFIG["batch"]))
        self.lr0 = float(config.get("lr0", self.DEFAULT_CONFIG["lr0"]))
        self.patience = int(config.get("patience", self.DEFAULT_CONFIG["patience"]))
        self.workers = int(config.get("workers", self.DEFAULT_CONFIG["workers"]))
        self.train_enabled = bool(config.get("train_enabled", self.DEFAULT_CONFIG["train_enabled"]))

        # --- Runtime-injected preproc chain ---
        self.preprocs: List[PreprocessingModule] = []

        # --- Build / load weights with corruption fallback ---
        try:
            self.model = YOLO(self._weights_path)
        except Exception as e:
            try:
                if os.path.exists(self._weights_path) and os.path.isfile(self._weights_path):
                    size = -1
                    try:
                        size = os.path.getsize(self._weights_path)
                    except Exception:
                        pass
                    if 0 <= size < 1024:
                        try: os.remove(self._weights_path)
                        except Exception: pass
                        try:
                            self.model = YOLO(self._weights_path)
                            e = None
                        except Exception:
                            pass
            except Exception:
                pass
            if e is not None:
                raise RuntimeError(
                    f"Failed to load YOLOv11 weights '{self.weights}': {e}.\n"
                    "Common causes: corrupted or partial .pt file. Remove it and retry, or set params.weights to a valid checkpoint."
                )

        # --- Resolve device (fallback to CPU if CUDA not available) ---
        try:
            import torch  # type: ignore
            if self.device != "cpu" and not torch.cuda.is_available():
                self._device_str = "cpu"
            else:
                self._device_str = self.device
        except Exception:
            self._device_str = "cpu"

    def _apply_preprocs_np(self, frame_bgr: np.ndarray) -> np.ndarray:
        if not self.preprocs:
            return frame_bgr
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        for p in self.preprocs:
            if getattr(p, "train_only", False):
                continue
            rgb = p.apply_to_frame(rgb)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def train(self, train_dataset, val_dataset=None, seed: int = 0, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Export dataset to YOLO format (images+labels) and run Ultralytics training.

        - Converts per-video COCO-like annotations to YOLO txt labels.
        - Applies preproc chain to saved training images (to match inference distribution).
        - If val_dataset is None, uses train split for validation as a fallback.
        """
        if not getattr(self, "train_enabled", True):
            return {"status": "skipped", "reason": "train_disabled"}
        if YOLO is None:
            detail = f" underlying import error: {_ULTRA_IMPORT_ERROR!r}" if _ULTRA_IMPORT_ERROR else ""
            raise RuntimeError(f"Ultralytics not available.{detail}")

        # Resolve paths
        base_out = output_dir or os.path.join(os.getcwd(), "results", "yolo_train")
        os.makedirs(base_out, exist_ok=True)
        ds_root = os.path.join(base_out, "yolo_dataset")
        train_img_dir = os.path.join(ds_root, "train", "images")
        train_lbl_dir = os.path.join(ds_root, "train", "labels")
        val_img_dir = os.path.join(ds_root, "val", "images")
        val_lbl_dir = os.path.join(ds_root, "val", "labels")
        for d in (train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir):
            os.makedirs(d, exist_ok=True)

        # Helpers
        def _apply_preprocs_np(frame_bgr: np.ndarray) -> np.ndarray:
            return self._apply_preprocs_np(frame_bgr)

        def _apply_preprocs_np_with_bboxes(
            frame_bgr: np.ndarray,
            bboxes: List[tuple],
        ) -> tuple[np.ndarray, List[tuple]]:
            if not self.preprocs:
                return frame_bgr, list(bboxes or [])
            import cv2
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            out_bboxes: List[tuple] = list(bboxes or [])
            for p in self.preprocs:
                if hasattr(p, "apply_to_frame_and_bboxes"):
                    rgb, out_bboxes = p.apply_to_frame_and_bboxes(rgb, out_bboxes)
                elif hasattr(p, "apply_to_frame_and_bbox"):
                    # fallback: apply to image only (bbox left unchanged)
                    rgb = p.apply_to_frame(rgb)
                else:
                    rgb = p.apply_to_frame(rgb)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), out_bboxes

        def _coco_to_yolo(bbox, w: int, h: int):
            # bbox: (x,y,w,h) in pixels -> YOLO normalized (xc,yc,w,h)
            try:
                x, y, bw, bh = [float(v) for v in bbox]
            except Exception:
                return None
            bw = max(1.0, bw)
            bh = max(1.0, bh)
            xc = (x + bw / 2.0) / max(1.0, float(w))
            yc = (y + bh / 2.0) / max(1.0, float(h))
            nw = bw / max(1.0, float(w))
            nh = bh / max(1.0, float(h))
            # clamp
            xc = float(min(1.0, max(0.0, xc)))
            yc = float(min(1.0, max(0.0, yc)))
            nw = float(min(1.0, max(0.0, nw)))
            nh = float(min(1.0, max(0.0, nh)))
            return (xc, yc, nw, nh)

        def _iter_video_frames(video_path: str, frame_indices: List[int]):
            import cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return []
            out = []
            try:
                for fi in sorted(set(int(i) for i in frame_indices)):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    out.append((fi, frame))
            finally:
                cap.release()
            return out

        def _export_split(dataset, img_dir: str, lbl_dir: str) -> int:
            from ..utils.annotations import load_coco_vid
            import cv2
            count = 0
            include_empty = bool(getattr(self, "include_empty_frames", False))
            for i in range(len(dataset)):
                item = dataset[i]
                vp: str = item["video_path"]
                # Load annotations (prefer in-memory)
                frames = {}
                ann_raw = item.get("annotation") if isinstance(item, dict) else None
                if isinstance(ann_raw, dict) and ann_raw:
                    try:
                        img_to_frame = {}
                        for img in ann_raw.get("images", []):
                            fi = img.get("frame_index")
                            if fi is None:
                                fi = img.get("id", 0) - 1
                            img_to_frame[img["id"]] = fi
                            if include_empty:
                                frames.setdefault(fi, [])
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
                if not frames:
                    j = os.path.splitext(vp)[0] + ".json"
                    if not os.path.exists(j):
                        continue
                    ann = load_coco_vid(j)
                    frames = ann.get("frames", {})
                    if not include_empty:
                        frames = {int(k): v for k, v in frames.items() if v}

                # Extract and write
                if not frames:
                    continue
                # Collect desired frame indices
                fids = list(frames.keys())
                # Read frames
                read = _iter_video_frames(vp, fids)
                for (fi, frame) in read:
                    # apply preproc and save image (+ update bboxes if needed)
                    bboxes = frames.get(fi) or []
                    frame, bboxes = _apply_preprocs_np_with_bboxes(frame, list(bboxes))
                    h, w = frame.shape[:2]
                    stem = f"{os.path.splitext(os.path.basename(vp))[0]}_f{int(fi):06d}"
                    img_path = os.path.join(img_dir, stem + ".jpg")
                    try:
                        cv2.imwrite(img_path, frame)
                    except Exception:
                        continue
                    # labels
                    lbl_path = os.path.join(lbl_dir, stem + ".txt")
                    lines = []
                    for b in (bboxes or []):
                        yb = _coco_to_yolo(b, w, h)
                        if yb is None:
                            continue
                        xc, yc, bw, bh = yb
                        # single-class id = 0
                        lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                    try:
                        with open(lbl_path, "w", encoding="utf-8") as f:
                            f.write("\n".join(lines))
                    except Exception:
                        pass
                    count += 1
            return count

        n_train = _export_split(train_dataset, train_img_dir, train_lbl_dir)
        if val_dataset is None:
            # fallback: use train as val to keep training loop valid
            val_dataset = train_dataset
        n_val = _export_split(val_dataset, val_img_dir, val_lbl_dir)

        # Build data.yaml for Ultralytics
        data_yaml = os.path.join(ds_root, "data.yaml")
        data_cfg = (
            f"path: {ds_root}\n"
            f"train: train/images\n"
            f"val: val/images\n"
            f"nc: 1\n"
            f"names: ['object']\n"
        )
        try:
            with open(data_yaml, "w", encoding="utf-8") as f:
                f.write(data_cfg)
        except Exception:
            pass

        # Training args
        epochs = int(getattr(self, "epochs", self.DEFAULT_CONFIG["epochs"]))
        batch = int(getattr(self, "batch", self.DEFAULT_CONFIG["batch"]))
        lr0 = float(getattr(self, "lr0", self.DEFAULT_CONFIG["lr0"]))
        patience = int(getattr(self, "patience", self.DEFAULT_CONFIG["patience"]))
        workers = int(getattr(self, "workers", self.DEFAULT_CONFIG["workers"]))

        # Hook progress callbacks (per-epoch) if UI injected one
        cb = getattr(self, 'progress_callback', None)
        def _emit(stage: str, trainer):
            try:
                if callable(cb):
                    cur = int(getattr(trainer, 'epoch', 0)) + 1
                    tot = int(getattr(trainer, 'epochs', epochs))
                    cb(stage, cur, tot)
            except Exception:
                pass
        try:
            if callable(cb) and hasattr(self.model, 'add_callback'):
                # start/end events (names per ultralytics callback system)
                self.model.add_callback('on_train_epoch_start', lambda trainer: _emit('train_epoch_start', trainer))
                # Some versions use on_train_epoch_end, others may allow on_fit_epoch_end; register both safely
                self.model.add_callback('on_train_epoch_end', lambda trainer: _emit('train_epoch_end', trainer))
                try:
                    self.model.add_callback('on_fit_epoch_end', lambda trainer: _emit('train_epoch_end', trainer))
                except Exception:
                    pass
        except Exception:
            pass

        # Run training
        results = None
        try:
            results = self.model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=self.imgsz,
                device=self._device_str,
                batch=batch,
                lr0=lr0,
                patience=patience,
                workers=workers,
                project=base_out,
                name="YOLOv11",
                exist_ok=True,
                verbose=True,
            )
        except Exception as e:
            return {"status": "error", "error": str(e)}

        # Try to collect best weights
        best_ckpt = None
        try:
            # Ultralytics saves to {project}/{name}/weights/best.pt
            run_dir = os.path.join(base_out, "YOLOv11")
            best_path = os.path.join(run_dir, "weights", "best.pt")
            if os.path.exists(best_path):
                best_ckpt = best_path
        except Exception:
            pass

        return {
            "status": "ok",
            "train_images": n_train,
            "val_images": n_val,
            "best_ckpt": best_ckpt,
        }

    def should_train(self, *_args, **_kwargs) -> bool:
        return bool(getattr(self, "train_enabled", True))

    def load_checkpoint(self, ckpt_path: str):
        if YOLO is None:
            detail = f" underlying import error: {_ULTRA_IMPORT_ERROR!r}" if _ULTRA_IMPORT_ERROR else ""
            raise RuntimeError(f"Ultralytics not available.{detail}")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(ckpt_path)
        self.model = YOLO(ckpt_path)

    def predict(self, video_path: str) -> List[FramePrediction]:
        if YOLO is None:
            detail = f" underlying import error: {_ULTRA_IMPORT_ERROR!r}" if _ULTRA_IMPORT_ERROR else ""
            raise RuntimeError(f"Ultralytics not available.{detail}")

        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        preds: List[FramePrediction] = []
        idx = 0
        last_bbox: Optional[tuple] = None
        # small-batch prediction to reduce per-call overhead
        try:
            batch_size = int(getattr(self, "inference_batch", 4) or 4)
        except Exception:
            batch_size = 4
        frames_buf = []
        idx_buf: List[int] = []
        def _flush():
            nonlocal last_bbox
            if not frames_buf:
                return
            try:
                results_list = self.model.predict(
                    source=frames_buf,
                    conf=self.conf,
                    iou=self.iou,
                    imgsz=self.imgsz,
                    device=self._device_str,
                    classes=self.classes,
                    verbose=False,
                    max_det=self.max_det,
                )
            except Exception:
                results_list = []
            for res, fidx in zip(results_list or [], idx_buf):
                bbox_added = False
                try:
                    boxes = getattr(res, "boxes", None)
                    if boxes is not None and len(boxes) > 0:
                        confs = boxes.conf.cpu().numpy().astype(float)
                        best = int(np.argmax(confs))
                        xyxy = boxes.xyxy.cpu().numpy()[best].tolist()
                        x1, y1, x2, y2 = map(float, xyxy)
                        w = max(1.0, x2 - x1)
                        h = max(1.0, y2 - y1)
                        score = float(confs[best])
                        bbox = (float(x1), float(y1), float(w), float(h))
                        preds.append(FramePrediction(int(fidx), bbox, score))
                        last_bbox = bbox
                        bbox_added = True
                except Exception:
                    bbox_added = False
                if not bbox_added and self.fallback_last_prediction and last_bbox is not None:
                    preds.append(
                        FramePrediction(
                            frame_index=int(fidx),
                            bbox=last_bbox,
                            score=None,
                            is_fallback=True,
                        )
                    )
            frames_buf.clear()
            idx_buf.clear()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = self._apply_preprocs_np(frame)
            frames_buf.append(frame)
            idx_buf.append(idx)
            if len(frames_buf) >= max(1, batch_size):
                _flush()
            idx += 1
        _flush()

        cap.release()
        return preds

    def predict_frames(self, video_path: str, frame_indices: List[int]) -> List[FramePrediction]:
        if YOLO is None:
            detail = f" underlying import error: {_ULTRA_IMPORT_ERROR!r}" if _ULTRA_IMPORT_ERROR else ""
            raise RuntimeError(f"Ultralytics not available.{detail}")
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        preds: List[FramePrediction] = []
        last_bbox: Optional[tuple] = None
        try:
            batch_size = int(getattr(self, "inference_batch", 4) or 4)
        except Exception:
            batch_size = 4
        buf_frames = []
        buf_indices: List[int] = []
        def _flush():
            nonlocal last_bbox
            if not buf_frames:
                return
            try:
                results_list = self.model.predict(
                    source=buf_frames,
                    conf=self.conf,
                    iou=self.iou,
                    imgsz=self.imgsz,
                    device=self._device_str,
                    classes=self.classes,
                    verbose=False,
                    max_det=self.max_det,
                )
            except Exception:
                results_list = []
            for res, fidx in zip(results_list or [], buf_indices):
                bbox_added = False
                try:
                    boxes = getattr(res, "boxes", None)
                    if boxes is not None and len(boxes) > 0:
                        confs = boxes.conf.cpu().numpy().astype(float)
                        best = int(np.argmax(confs))
                        xyxy = boxes.xyxy.cpu().numpy()[best].tolist()
                        x1, y1, x2, y2 = map(float, xyxy)
                        w = max(1.0, x2 - x1)
                        h = max(1.0, y2 - y1)
                        score = float(confs[best])
                        bbox = (float(x1), float(y1), float(w), float(h))
                        preds.append(FramePrediction(int(fidx), bbox, score))
                        last_bbox = bbox
                        bbox_added = True
                except Exception:
                    bbox_added = False
                if not bbox_added and self.fallback_last_prediction and last_bbox is not None:
                    preds.append(
                        FramePrediction(
                            frame_index=int(fidx),
                            bbox=last_bbox,
                            score=None,
                            is_fallback=True,
                        )
                    )
            buf_frames.clear(); buf_indices.clear()
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
