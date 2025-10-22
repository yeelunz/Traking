from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore
import numpy as np

from ..core.interfaces import TrackingModel, FramePrediction, PreprocessingModule
from ..core.registry import register_model
from ..utils.init_bbox import detect_bbox_on_frame, resolve_first_frame_bbox


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
    # 對外顯示名稱改為 NCC（Normalized Cross-Correlation 取向的簡稱）
    name = "NCC"
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
        # first-frame init
        "first_frame_source": "gt",
        "first_frame_fallback": "gt",
        "init_detector_weights": "best.pt",
        "init_detector_conf": 0.25,
        "init_detector_iou": 0.5,
        "init_detector_imgsz": 640,
        "init_detector_device": "auto",
        "init_detector_classes": None,
        "init_detector_max_det": 50,
        # low-confidence recovery
        "low_confidence_reinit": {
            "enabled": False,
            "threshold": 0.3,
            "min_interval": 15,
            "detector": {},
            "detector_min_conf": None,
        },
    }

    def __init__(self, config: Dict[str, Any]):
        if cv2 is None:
            raise RuntimeError("OpenCV is required for NCC (FASTSpeckle).")
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

        self.first_frame_source = str(config.get("first_frame_source", self.DEFAULT_CONFIG["first_frame_source"]) or "gt").lower()
        raw_fallback = config.get("first_frame_fallback", self.DEFAULT_CONFIG["first_frame_fallback"])
        if raw_fallback is None:
            self.first_frame_fallback: Optional[str] = None
        else:
            fb = str(raw_fallback).strip().lower()
            self.first_frame_fallback = fb if fb not in ("", "none", "null") else None
        self.init_detector_params = {
            "weights": str(config.get("init_detector_weights", self.DEFAULT_CONFIG["init_detector_weights"])),
            "conf": float(config.get("init_detector_conf", self.DEFAULT_CONFIG["init_detector_conf"])),
            "iou": float(config.get("init_detector_iou", self.DEFAULT_CONFIG["init_detector_iou"])),
            "imgsz": int(config.get("init_detector_imgsz", self.DEFAULT_CONFIG["init_detector_imgsz"])),
            "device": str(config.get("init_detector_device", self.DEFAULT_CONFIG["init_detector_device"])),
            "classes": config.get("init_detector_classes", self.DEFAULT_CONFIG["init_detector_classes"]),
            "max_det": int(config.get("init_detector_max_det", self.DEFAULT_CONFIG["init_detector_max_det"])),
        }

        lc_default = self.DEFAULT_CONFIG["low_confidence_reinit"]
        lc_cfg = config.get("low_confidence_reinit", lc_default)
        if isinstance(lc_cfg, bool):
            lc_cfg = {"enabled": bool(lc_cfg)}
        lc_cfg = dict(lc_cfg or {})
        self.low_conf_reinit_enabled = bool(lc_cfg.get("enabled", lc_default["enabled"]))
        self.low_conf_threshold = float(lc_cfg.get("threshold", lc_default["threshold"]))
        self.low_conf_min_interval = max(1, int(lc_cfg.get("min_interval", lc_default["min_interval"])))
        detector_override = lc_cfg.get("detector") or {}
        if not isinstance(detector_override, dict):
            detector_override = {}
        self.low_conf_detector_params = {**self.init_detector_params, **detector_override}
        min_conf_val = lc_cfg.get("detector_min_conf", lc_default.get("detector_min_conf"))
        self.low_conf_detector_min_conf = None if min_conf_val in (None, "none", "") else float(min_conf_val)

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

    def _resolve_first_bbox(self, video_path: str) -> Optional[Tuple[float, float, float, float]]:
        fallback = self.first_frame_fallback
        return resolve_first_frame_bbox(
            video_path,
            mode=self.first_frame_source,
            detector=self.init_detector_params,
            fallback=fallback,
        )

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
        prev_gray: Optional[np.ndarray] = None
        bbox: Optional[Tuple[float, float, float, float]] = None
        pts_prev: Optional[np.ndarray] = None
        t = 0
        last_reinit = -10**9
        last_detector_reinit = -10**9
        base_feature_count = 0

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_bgr = self._apply_preprocs(frame_bgr)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            if prev_gray is None:
                h, w = gray.shape
                initbb = self._resolve_first_bbox(video_path)
                if initbb is None:
                    cap.release()
                    import os as _os

                    raise RuntimeError(
                        f"Failed to obtain first-frame bbox for video: {_os.path.basename(video_path)} "
                        f"(mode={self.first_frame_source})."
                    )
                gx, gy, gw, gh = initbb
                x = max(0.0, min(float(w - 1), float(gx)))
                y = max(0.0, min(float(h - 1), float(gy)))
                iw = float(max(1.0, min(gw, w - x)))
                ih = float(max(1.0, min(gh, h - y)))
                bbox = (x, y, iw, ih)
                pts_prev = self._detect_in_roi(gray, bbox)
                base_feature_count = len(pts_prev) if pts_prev is not None else 0
                if self.debug:
                    try:
                        n = 0 if pts_prev is None else len(pts_prev)
                        print(f"[NCC] init roi=({int(x)},{int(y)},{int(iw)},{int(ih)}) features={n}")
                    except Exception:
                        pass
                confidence_init = 1.0 if base_feature_count > 0 else 0.0
                preds.append(FramePrediction(t, bbox, confidence_init))
                prev_gray = gray
                last_reinit = t
                t += 1
                continue

            need_reinit = (
                pts_prev is None
                or len(pts_prev) < self.min_features
                or (t - last_reinit) >= self.reinit_interval
            )
            if need_reinit and bbox is not None:
                new_pts = self._detect_in_roi(gray, bbox)
                if new_pts is not None and len(new_pts) > 0:
                    pts_prev = new_pts
                    base_feature_count = len(new_pts)
                    last_reinit = t
                if self.debug:
                    try:
                        n = 0 if new_pts is None else len(new_pts)
                        print(f"[NCC] reinit@{t}: features={n}")
                    except Exception:
                        pass

            if pts_prev is None or len(pts_prev) == 0 or bbox is None:
                confidence = 0.0
                if (
                    self.low_conf_reinit_enabled
                    and (t - last_detector_reinit) >= self.low_conf_min_interval
                ):
                    det_bbox, det_conf = detect_bbox_on_frame(
                        frame_bgr,
                        self.low_conf_detector_params,
                        self.low_conf_detector_min_conf,
                    )
                    if det_bbox is not None:
                        bbox = det_bbox
                        pts_prev = self._detect_in_roi(gray, bbox)
                        base_feature_count = len(pts_prev) if pts_prev is not None else 0
                        prev_gray = gray
                        last_detector_reinit = t
                        det_conf_val = float(det_conf) if det_conf is not None else None
                        conf_out = 1.0
                        if self.debug:
                            try:
                                det_msg = (
                                    f" det_conf={det_conf_val:.3f}" if det_conf_val is not None else ""
                                )
                                print(
                                    f"[NCC] detector reinit@{t}: conf={conf_out:.3f}{det_msg} bbox="
                                    f"({int(bbox[0])},{int(bbox[1])},{int(bbox[2])},{int(bbox[3])})"
                                )
                            except Exception:
                                pass
                        preds.append(FramePrediction(t, bbox, max(0.0, min(1.0, conf_out))))
                        t += 1
                        continue

                if bbox is not None:
                    preds.append(FramePrediction(t, bbox, confidence))
                prev_gray = gray
                t += 1
                continue

            pts_prev_np = np.asarray(pts_prev, dtype=np.float32)
            pts_next, st, err = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                gray,
                pts_prev_np,
                None,
                winSize=(self.lk_win, self.lk_win),
                maxLevel=self.lk_max_level,
            )
            status = st.reshape(-1) == 1 if st is not None else np.zeros((pts_prev_np.shape[0],), dtype=bool)
            good_old = pts_prev_np[status] if status.any() else pts_prev_np
            next_points = pts_next.reshape(-1, 2) if pts_next is not None else pts_prev_np
            good_new = next_points[status] if status.any() else next_points

            if len(good_old) >= 3 and len(good_new) >= 3:
                dx, dy = _median_displacement(good_old, good_new)
                x, y, w, h = bbox
                x = max(0.0, x + dx)
                y = max(0.0, y + dy)
                bbox = (x, y, w, h)

            valid_count = int(status.sum()) if status.size else len(good_new)
            denom_reference = max(1, base_feature_count if base_feature_count > 0 else pts_prev_np.shape[0])
            ratio = valid_count / float(denom_reference)

            err_factor = 0.0
            if err is not None:
                try:
                    err_vals = err.reshape(-1)[status] if status.any() else err.reshape(-1)
                except Exception:
                    err_vals = None
                if err_vals is not None and err_vals.size > 0:
                    median_err = float(np.median(err_vals))
                    norm_denom = float((self.lk_win ** 2) * max(1, self.lk_max_level + 1))
                    err_factor = float(np.clip(median_err / norm_denom, 0.0, 1.0))

            confidence = float(np.clip(ratio * (1.0 - err_factor), 0.0, 1.0))

            low_confidence = (
                self.low_conf_reinit_enabled
                and confidence < self.low_conf_threshold
                and (t - last_detector_reinit) >= self.low_conf_min_interval
            )
            if low_confidence:
                det_bbox, det_conf = detect_bbox_on_frame(
                    frame_bgr,
                    self.low_conf_detector_params,
                    self.low_conf_detector_min_conf,
                )
                if det_bbox is not None:
                    bbox = det_bbox
                    pts_prev = self._detect_in_roi(gray, bbox)
                    base_feature_count = len(pts_prev) if pts_prev is not None else 0
                    prev_gray = gray
                    last_detector_reinit = t
                    det_conf_val = float(det_conf) if det_conf is not None else None
                    conf_out = 1.0
                    if self.debug:
                        try:
                            det_msg = (
                                f" det_conf={det_conf_val:.3f}" if det_conf_val is not None else ""
                            )
                            print(
                                f"[NCC] detector reinit@{t}: conf={conf_out:.3f}{det_msg} bbox="
                                f"({int(bbox[0])},{int(bbox[1])},{int(bbox[2])},{int(bbox[3])})"
                            )
                        except Exception:
                            pass
                    preds.append(FramePrediction(t, bbox, max(0.0, min(1.0, conf_out))))
                    t += 1
                    continue

            if self.debug:
                try:
                    print(
                        f"[NCC] frame={t} tracked={len(good_new)} bbox="
                        f"({int(bbox[0])},{int(bbox[1])},{int(bbox[2])},{int(bbox[3])}) conf={confidence:.3f}"
                    )
                except Exception:
                    pass

            preds.append(FramePrediction(t, bbox, confidence))
            prev_gray = gray
            pts_prev = good_new if len(good_new) else pts_prev_np
            if pts_prev is not None and len(pts_prev) > 0:
                base_feature_count = max(base_feature_count, len(pts_prev))
            t += 1

        cap.release()
        return preds

# 兼容性：允許以 "NCC" 名稱引用此模型
try:
    from ..core.registry import MODEL_REGISTRY  # type: ignore
    MODEL_REGISTRY.setdefault("NCC", FASTSpeckle)
except Exception:
    pass
