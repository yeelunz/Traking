from __future__ import annotations
from typing import Dict, Any, List, Optional
import os

import numpy as np

try:
    import torch
    import torchvision
    from torchvision.transforms import functional as F
except Exception:
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
    }

    def __init__(self, config: Dict[str, Any]):
        if torch is None or torchvision is None:
            raise RuntimeError("PyTorch and torchvision are required for FasterRCNN model.")
        self.score_thresh = float(config.get("score_thresh", self.DEFAULT_CONFIG["score_thresh"]))
        self.device = str(config.get("device", self.DEFAULT_CONFIG["device"]))
        self.pretrained = bool(config.get("pretrained", self.DEFAULT_CONFIG["pretrained"]))
        self.num_classes = int(config.get("num_classes", self.DEFAULT_CONFIG["num_classes"]))
        self.preprocs: List[PreprocessingModule] = []
        # build model
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT if self.pretrained else None
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        if self.num_classes != 91:  # default COCO classes; allow override if user sets num_classes
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, self.num_classes)
        self.model.eval()
        self._device = torch.device(self.device if (self.device == "cpu" or torch.cuda.is_available()) else "cpu")
        self.model.to(self._device)

    def _apply_preprocs_np(self, frame_bgr: np.ndarray) -> np.ndarray:
        if not self.preprocs:
            return frame_bgr
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        for p in self.preprocs:
            rgb = p.apply_to_frame(rgb)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def train(self, train_dataset: Dataset, val_dataset: Optional[Dataset] = None, seed: int = 0, output_dir: Optional[str] = None):
        # Minimal stub: training loop not implemented here
        return {"status": "no_training"}

    def load_checkpoint(self, ckpt_path: str):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(ckpt_path)
        state = torch.load(ckpt_path, map_location=self._device)
        self.model.load_state_dict(state)
        self.model.to(self._device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, video_path: str) -> List[FramePrediction]:
        # detection-based: 對每幀取最高分框作為單目標預測
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        preds: List[FramePrediction] = []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = self._apply_preprocs_np(frame)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = F.to_tensor(img).to(self._device)
            outputs = self.model([tensor])[0]
            scores = outputs.get("scores")
            boxes = outputs.get("boxes")
            if scores is not None and boxes is not None and len(scores) > 0:
                # 取最高分框
                best = int(torch.argmax(scores).item())
                if float(scores[best].item()) >= self.score_thresh:
                    x1, y1, x2, y2 = boxes[best].tolist()
                    bbox = (float(x1), float(y1), float(max(1.0, x2 - x1)), float(max(1.0, y2 - y1)))
                    preds.append(FramePrediction(idx, bbox, float(scores[best].item())))
            idx += 1
        cap.release()
        return preds
