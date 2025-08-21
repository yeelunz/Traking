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

        self.weights = str(config.get("weights", self.DEFAULT_CONFIG["weights"]))
        self.conf = float(config.get("conf", self.DEFAULT_CONFIG["conf"]))
        self.iou = float(config.get("iou", self.DEFAULT_CONFIG["iou"]))
        self.imgsz = int(config.get("imgsz", self.DEFAULT_CONFIG["imgsz"]))
        self.device = str(config.get("device", self.DEFAULT_CONFIG["device"]))
        self.classes = config.get("classes", self.DEFAULT_CONFIG["classes"])  # type: ignore
        self.max_det = int(config.get("max_det", self.DEFAULT_CONFIG["max_det"]))
        self.fallback_last_prediction = bool(config.get("fallback_last_prediction", self.DEFAULT_CONFIG["fallback_last_prediction"]))

        # allow runner to inject preprocessing chain
        self.preprocs: List[PreprocessingModule] = []

        # Build model (weights path or model name) with safe retry for corrupted local files
        try:
            self.model = YOLO(self.weights)
        except Exception as e:
            # If a local file with this name exists and appears corrupted, try removing it and retry once
            try:
                if os.path.exists(self.weights) and os.path.isfile(self.weights):
                    # only attempt cleanup for small/invalid files
                    try:
                        size = os.path.getsize(self.weights)
                    except Exception:
                        size = -1
                    # heuristic: if file size is unusually small (< 1024 bytes) or loading failed, remove to allow redownload
                    if size >= 0 and size < 1024:
                        try:
                            os.remove(self.weights)
                        except Exception:
                            pass
                        try:
                            self.model = YOLO(self.weights)
                            e = None
                        except Exception:
                            pass
            except Exception:
                pass
            if e is not None:
                raise RuntimeError(
                    f"Failed to load YOLOv11 weights '{self.weights}': {e}.\n"
                    "Common causes: the local .pt file is corrupted (partial download) or not a valid torch checkpoint.\n"
                    "Fixes: remove the corrupted file and retry, or set params.weights to a valid checkpoint name/path (e.g. 'yolov8n.pt' or full path to your best.pt)."
                )

        # Resolve device string (use CPU if CUDA not available)
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
            rgb = p.apply_to_frame(rgb)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def train(self, train_dataset, val_dataset=None, seed: int = 0, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Export dataset to YOLO format (images+labels) and run Ultralytics training.

        - Converts per-video COCO-like annotations to YOLO txt labels.
        - Applies preproc chain to saved training images (to match inference distribution).
        - If val_dataset is None, uses train split for validation as a fallback.
        """
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
            include_empty = bool(self.DEFAULT_CONFIG.get("include_empty_frames", False))
            try:
                include_empty = bool(getattr(self, "include_empty_frames", include_empty))
            except Exception:
                pass
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
                    # apply preproc and save image
                    frame = _apply_preprocs_np(frame)
                    h, w = frame.shape[:2]
                    stem = f"{os.path.splitext(os.path.basename(vp))[0]}_f{int(fi):06d}"
                    img_path = os.path.join(img_dir, stem + ".jpg")
                    try:
                        cv2.imwrite(img_path, frame)
                    except Exception:
                        continue
                    # labels
                    lbl_path = os.path.join(lbl_dir, stem + ".txt")
                    bboxes = frames.get(fi) or []
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
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = self._apply_preprocs_np(frame)

            # Run YOLO inference on the frame (BGR is acceptable)
            try:
                results = self.model.predict(
                    source=frame,
                    conf=self.conf,
                    iou=self.iou,
                    imgsz=self.imgsz,
                    device=self._device_str,
                    classes=self.classes,
                    verbose=False,
                    max_det=self.max_det,
                )
            except Exception as e:
                # If inference fails on this frame, skip but keep index advancing
                results = []

            bbox_added = False
            if results:
                r0 = results[0]
                try:
                    boxes = getattr(r0, "boxes", None)
                    if boxes is not None and len(boxes) > 0:
                        # Choose highest-confidence box
                        confs = boxes.conf.cpu().numpy().astype(float)
                        best = int(np.argmax(confs))
                        xyxy = boxes.xyxy.cpu().numpy()[best].tolist()  # [x1,y1,x2,y2]
                        x1, y1, x2, y2 = map(float, xyxy)
                        w = max(1.0, x2 - x1)
                        h = max(1.0, y2 - y1)
                        score = float(confs[best])
                        bbox = (float(x1), float(y1), float(w), float(h))
                        preds.append(FramePrediction(idx, bbox, score))
                        last_bbox = bbox
                        bbox_added = True
                except Exception:
                    pass
            if not bbox_added and self.fallback_last_prediction and last_bbox is not None:
                preds.append(FramePrediction(idx, last_bbox, None))
            idx += 1

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
        for idx in sorted(set(int(i) for i in frame_indices)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            frame = self._apply_preprocs_np(frame)
            try:
                results = self.model.predict(
                    source=frame,
                    conf=self.conf,
                    iou=self.iou,
                    imgsz=self.imgsz,
                    device=self._device_str,
                    classes=self.classes,
                    verbose=False,
                    max_det=self.max_det,
                )
            except Exception as e:
                results = []
            bbox_added = False
            if results:
                r0 = results[0]
                try:
                    boxes = getattr(r0, "boxes", None)
                    if boxes is not None and len(boxes) > 0:
                        confs = boxes.conf.cpu().numpy().astype(float)
                        best = int(np.argmax(confs))
                        xyxy = boxes.xyxy.cpu().numpy()[best].tolist()
                        x1, y1, x2, y2 = map(float, xyxy)
                        w = max(1.0, x2 - x1)
                        h = max(1.0, y2 - y1)
                        score = float(confs[best])
                        bbox = (float(x1), float(y1), float(w), float(h))
                        preds.append(FramePrediction(int(idx), bbox, score))
                        last_bbox = bbox
                        bbox_added = True
                except Exception:
                    pass
            if not bbox_added and self.fallback_last_prediction and last_bbox is not None:
                preds.append(FramePrediction(int(idx), last_bbox, None))
        cap.release()
        return preds
