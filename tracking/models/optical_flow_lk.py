from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore
import numpy as np

from ..core.interfaces import TrackingModel, FramePrediction, PreprocessingModule
from ..core.registry import register_model
from ..utils.annotations import load_coco_vid


def _median_displacement(pts0: np.ndarray, pts1: np.ndarray) -> Tuple[float, float]:
    dx = np.median(pts1[:, 0] - pts0[:, 0])
    dy = np.median(pts1[:, 1] - pts0[:, 1])
    return float(dx), float(dy)


@register_model("OpticalFlowLK")
class OpticalFlowLK(TrackingModel):
    name = "OpticalFlowLK"
    DEFAULT_CONFIG = {
        "init_box": [64, 64],
        "max_corners": 50,
        "quality_level": 0.01,
        "min_distance": 7,
        "win_size": 21,
    # 新增：長影片避免耗時與“卡住”錯覺的控制參數
    # 若為 True，當超過最後一個有 GT 的 frame 後即停止（評估 restrict_to_gt_frames 時不需跑整部片）
    "stop_at_last_gt": True,
    # 強制最多處理多少 frame（None=不限）
    "max_eval_frames": None,
    # 每處理多少 frame 列印一次除錯訊息（0/None=不列印）
    "log_every": 500,
    }

    def __init__(self, config: Dict[str, Any]):
        if cv2 is None:
            raise RuntimeError("OpenCV is required for OpticalFlowLK.")
        self.init_w, self.init_h = tuple(config.get("init_box", [64, 64]))
        self.max_corners = int(config.get("max_corners", 50))
        self.quality_level = float(config.get("quality_level", 0.01))
        self.min_distance = int(config.get("min_distance", 7))
        self.win_size = int(config.get("win_size", 21))
        self.preprocs: List[PreprocessingModule] = []
        self.stop_at_last_gt = bool(config.get("stop_at_last_gt", self.DEFAULT_CONFIG["stop_at_last_gt"]))
        self.max_eval_frames = config.get("max_eval_frames", self.DEFAULT_CONFIG["max_eval_frames"])  # 可能是 None or int
        try:
            if self.max_eval_frames is not None:
                self.max_eval_frames = int(self.max_eval_frames)
        except Exception:
            self.max_eval_frames = None
        try:
            self.log_every = int(config.get("log_every", self.DEFAULT_CONFIG["log_every"]) or 0)
        except Exception:
            self.log_every = 0

    def train(self, train_dataset, val_dataset=None, seed: int = 0, output_dir: str | None = None):
        cb = getattr(self, 'progress_callback', None)
        try:
            if callable(cb):
                cb('train_epoch_start', 1, 1)
                cb('train_epoch_end', 1, 1)
        except Exception:
            pass
        return {"status": "no_training"}

    def load_checkpoint(self, ckpt_path: str):
        pass

    def _apply_preprocs(self, frame_bgr):
        if not self.preprocs:
            return frame_bgr
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        for p in self.preprocs:
            rgb = p.apply_to_frame(rgb)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _first_gt_bbox(self, video_path: str) -> Optional[Tuple[float, float, float, float]]:
        import os
        j = os.path.splitext(video_path)[0] + ".json"
        if not os.path.exists(j):
            return None
        try:
            gt = load_coco_vid(j)
            frames = gt.get("frames", {})
            if not frames:
                return None
            first_idx = sorted(int(k) for k in frames.keys() if frames.get(k))[0]
            bboxes = frames.get(first_idx) or []
            if not bboxes:
                return None
            x, y, w, h = bboxes[0]
            return float(x), float(y), float(w), float(h)
        except Exception:
            return None

    def predict(self, video_path: str) -> List[FramePrediction]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        preds: List[FramePrediction] = []
        prev_gray = None
        bbox = None
        pts_prev = None
        t = 0
        last_gt_frame: Optional[int] = None
        gt_frames_sorted: List[int] = []
        if self.stop_at_last_gt:
            # 輕量載入一次 JSON 拿到所有有標註的 frame index
            import os as _os
            j = _os.path.splitext(video_path)[0] + ".json"
            if _os.path.exists(j):
                try:
                    gt = load_coco_vid(j)
                    gt_frames_sorted = sorted(int(fi) for fi, boxes in (gt.get("frames", {}) or {}).items() if boxes)
                    if gt_frames_sorted:
                        last_gt_frame = gt_frames_sorted[-1]
                except Exception:
                    pass
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = self._apply_preprocs(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is None:
                h, w = gray.shape
                gtbb = self._first_gt_bbox(video_path)
                if gtbb is None:
                    cap.release()
                    import os as _os
                    raise RuntimeError(f"Missing GT first-frame bbox for video: {_os.path.basename(video_path)}. Provide GT JSON.")
                gx, gy, gw, gh = gtbb
                x = max(0.0, min(float(w - 1), float(gx)))
                y = max(0.0, min(float(h - 1), float(gy)))
                iw = float(min(gw, w - x))
                ih = float(min(gh, h - y))
                bbox = (float(x), float(y), float(iw), float(ih))
                # detect features inside bbox
                mask = np.zeros_like(gray)
                mask[int(y):int(y+ih), int(x):int(x+iw)] = 255
                pts_prev = cv2.goodFeaturesToTrack(gray, maxCorners=self.max_corners,
                                                   qualityLevel=self.quality_level,
                                                   minDistance=self.min_distance,
                                                   mask=mask)
                if pts_prev is not None:
                    pts_prev = pts_prev.reshape(-1, 2)
                preds.append(FramePrediction(t, bbox, 1.0))
                prev_gray = gray
                t += 1
                continue
            if pts_prev is None or len(pts_prev) == 0:
                preds.append(FramePrediction(t, bbox, 1.0))
                prev_gray = gray
                t += 1
                continue
            pts_next, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts_prev.astype(np.float32), None,
                                                         winSize=(self.win_size, self.win_size), maxLevel=3)
            good_old = pts_prev[st.reshape(-1) == 1]
            good_new = pts_next.reshape(-1, 2)[st.reshape(-1) == 1]
            if len(good_old) >= 3 and len(good_new) >= 3:
                dx, dy = _median_displacement(good_old, good_new)
                x, y, w, h = bbox
                x = max(0.0, x + dx)
                y = max(0.0, y + dy)
                bbox = (x, y, w, h)
            preds.append(FramePrediction(t, bbox, 1.0))
            prev_gray = gray
            pts_prev = good_new if len(good_new) else pts_prev
            t += 1
            # 早停條件：達到最後 GT frame 且再多補 1~2 frame 已足夠（保持 bbox 最終定位）
            if last_gt_frame is not None and t > last_gt_frame + 2:
                break
            if self.max_eval_frames is not None and t >= self.max_eval_frames:
                break
            if self.log_every and (t % self.log_every == 0):
                try:
                    print(f"[OpticalFlowLK] processed {t} frames (video={video_path})")
                except Exception:
                    pass
        cap.release()
        return preds
