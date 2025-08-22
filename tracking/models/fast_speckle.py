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


@register_model("FASTSpeckle")
class FASTSpeckle(TrackingModel):
    """
    FAST corner-based speckle tracking:
    - Initialize ROI from first GT bbox.
    - Detect FAST keypoints within ROI.
    - Track with LK optical flow; update bbox by median displacement.
    - Re-detect when features get too few or periodically.
    """

    name = "FASTSpeckle"
    DEFAULT_CONFIG = {
        # FAST detector
        "fast_threshold": 20,
        "fast_nonmax": True,
        "max_features": 200,
        # Lucas-Kanade
        "lk_win_size": 21,
        "lk_max_level": 3,
        # maintenance
        "min_features": 30,
        "reinit_interval": 10,  # frames
    # fallback
    "use_gftt_fallback": True,
    # debug
    "debug": False,
    }

    def __init__(self, config: Dict[str, Any]):
        if cv2 is None:
            raise RuntimeError("OpenCV is required for FASTSpeckle.")
        self.fast_threshold = int(config.get("fast_threshold", self.DEFAULT_CONFIG["fast_threshold"]))
        self.fast_nonmax = bool(config.get("fast_nonmax", self.DEFAULT_CONFIG["fast_nonmax"]))
        self.max_features = int(config.get("max_features", self.DEFAULT_CONFIG["max_features"]))
        self.lk_win = int(config.get("lk_win_size", self.DEFAULT_CONFIG["lk_win_size"]))
        self.lk_max_level = int(config.get("lk_max_level", self.DEFAULT_CONFIG["lk_max_level"]))
        self.min_features = int(config.get("min_features", self.DEFAULT_CONFIG["min_features"]))
        self.reinit_interval = int(config.get("reinit_interval", self.DEFAULT_CONFIG["reinit_interval"]))
        self.use_gftt_fallback = bool(config.get("use_gftt_fallback", self.DEFAULT_CONFIG["use_gftt_fallback"]))
        self.debug = bool(config.get("debug", self.DEFAULT_CONFIG["debug"]))
        self.preprocs = []  # type: List[PreprocessingModule]

        # build detector
        try:
            self.det = cv2.FastFeatureDetector_create(threshold=self.fast_threshold, nonmaxSuppression=self.fast_nonmax)
        except Exception:
            # OpenCV older variants
            try:
                self.det = cv2.FastFeatureDetector_create(self.fast_threshold)
            except Exception:
                self.det = None

    def train(self, train_dataset, val_dataset=None, seed: int = 0, output_dir: str | None = None):
        # Emit a dummy one-epoch progress so UI can show completion instantly
        cb = getattr(self, 'progress_callback', None)
        try:
            if callable(cb):
                cb('train_epoch_start', 1, 1)
        except Exception:
            pass
        try:
            if callable(cb):
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

    def _detect_in_roi(self, gray: np.ndarray, bbox: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
        x, y, w, h = bbox
        H, W = gray.shape
        x0 = int(max(0, min(W - 1, x)))
        y0 = int(max(0, min(H - 1, y)))
        x1 = int(min(W, x0 + max(1, int(w))))
        y1 = int(min(H, y0 + max(1, int(h))))
        if x1 <= x0 or y1 <= y0:
            return None
        roi = gray[y0:y1, x0:x1]
        # detect corners via FAST; if detector is unavailable, fall back later
        kps = []
        try:
            if self.det is not None:
                kps = self.det.detect(roi)
        except Exception:
            kps = []
        if (not kps) and self.use_gftt_fallback:
            try:
                gftt = cv2.GFTTDetector_create(maxCorners=max(50, self.max_features), qualityLevel=0.01, minDistance=3)
                kps = gftt.detect(roi)
            except Exception:
                kps = []
        # final fallback: cv2.goodFeaturesToTrack
        if (not kps) and self.use_gftt_fallback:
            try:
                mask = np.zeros_like(roi)
                mask[:, :] = 255
                pts = cv2.goodFeaturesToTrack(roi, maxCorners=max(50, self.max_features), qualityLevel=0.01, minDistance=3, mask=mask)
                if pts is not None and len(pts) > 0:
                    pts = pts.reshape(-1, 2)
                    return np.array([[float(px + x0), float(py + y0)] for (px, py) in pts], dtype=np.float32)
            except Exception:
                pass
        if not kps:
            return None
        # Normalize keypoints: some OpenCV builds may return tuples (x,y)
        try:
            # object keypoints path
            kps_sorted = sorted(
                kps,
                key=lambda kp: float(getattr(kp, 'response', 1.0)),
                reverse=True,
            )[: self.max_features]
            pts = np.array([[float(kp.pt[0] + x0), float(kp.pt[1] + y0)] for kp in kps_sorted], dtype=np.float32)
        except Exception:
            # tuple/list path
            try:
                arr = np.asarray(kps, dtype=np.float32)
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    # sort by nothing; just clip count
                    arr = arr[: self.max_features]
                    pts = np.stack([arr[:, 0] + x0, arr[:, 1] + y0], axis=1).astype(np.float32)
                else:
                    pts = None
            except Exception:
                pts = None
        if pts is None:
            return None
        if pts.size == 0:
            return None
        return pts

    def predict(self, video_path: str) -> List[FramePrediction]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        preds: List[FramePrediction] = []
        prev_gray = None
        bbox: Optional[Tuple[float, float, float, float]] = None
        pts_prev: Optional[np.ndarray] = None
        t = 0
        last_reinit = -10**9
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
                iw = float(max(1.0, min(gw, w - x)))
                ih = float(max(1.0, min(gh, h - y)))
                bbox = (x, y, iw, ih)
                pts_prev = self._detect_in_roi(gray, bbox)
                if self.debug:
                    try:
                        n = 0 if pts_prev is None else len(pts_prev)
                        print(f"[FASTSpeckle] init roi=({int(x)},{int(y)},{int(iw)},{int(ih)}) features={n}")
                    except Exception:
                        pass
                preds.append(FramePrediction(t, bbox, 1.0))
                prev_gray = gray
                last_reinit = t
                t += 1
                continue
            # decide if we should (re)detect
            need_reinit = (
                pts_prev is None or len(pts_prev) < self.min_features or (t - last_reinit) >= self.reinit_interval
            )
            if need_reinit and bbox is not None:
                # use current frame for re-detect to adapt to appearance changes
                new_pts = self._detect_in_roi(gray, bbox)
                if new_pts is not None and len(new_pts) > 0:
                    pts_prev = new_pts
                    last_reinit = t
                if self.debug:
                    try:
                        n = 0 if new_pts is None else len(new_pts)
                        print(f"[FASTSpeckle] reinit@{t}: features={n}")
                    except Exception:
                        pass
            if pts_prev is None or len(pts_prev) == 0 or bbox is None:
                # keep previous bbox if tracking lost
                if bbox is not None:
                    preds.append(FramePrediction(t, bbox, 1.0))
                prev_gray = gray
                t += 1
                continue
            # track with LK
            pts_next, st, _err = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, pts_prev.astype(np.float32), None,
                winSize=(self.lk_win, self.lk_win), maxLevel=self.lk_max_level
            )
            status = st.reshape(-1) == 1 if st is not None else np.zeros((len(pts_prev),), dtype=bool)
            good_old = pts_prev[status]
            good_new = (pts_next.reshape(-1, 2) if pts_next is not None else pts_prev)[status]
            if len(good_old) >= 3 and len(good_new) >= 3:
                dx, dy = _median_displacement(good_old, good_new)
                x, y, w, h = bbox
                x = max(0.0, x + dx)
                y = max(0.0, y + dy)
                # keep size fixed (speckle drift only translates)
                bbox = (x, y, w, h)
            if self.debug:
                try:
                    print(f"[FASTSpeckle] frame={t} tracked={len(good_new)} bbox=({int(bbox[0])},{int(bbox[1])},{int(bbox[2])},{int(bbox[3])})")
                except Exception:
                    pass
            preds.append(FramePrediction(t, bbox, 1.0))
            prev_gray = gray
            pts_prev = good_new if len(good_new) else pts_prev
            t += 1
        cap.release()
        return preds
