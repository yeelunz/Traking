from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Sequence, Optional, List

import numpy as np
import cv2
import torch
from torch import nn
from torchvision import models, transforms
from sklearn.random_projection import GaussianRandomProjection
import warnings

from ..core.interfaces import FramePrediction
from ..core.registry import register_feature_extractor
from .interfaces import TrajectoryFeatureExtractor


def _safe_std(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.std(arr, ddof=0))


@register_feature_extractor("basic")
class BasicTrajectoryFeatureExtractor(TrajectoryFeatureExtractor):
    """Compute simple kinematic statistics from a trajectory."""

    name = "BasicTrajectoryFeatures"

    def __init__(self, params: Dict[str, Any] | None = None):
        cfg = params or {}
        self._epsilon = float(cfg.get("epsilon", 1e-6))
        self._video_keys = [
            "num_points",
            "duration_frames",
            "span_x",
            "span_y",
            "path_length",
            "mean_speed",
            "max_speed",
            "std_speed",
            "median_speed",
            "p95_speed",
            "mean_acc",
            "max_acc",
            "std_acc",
            "mean_area",
            "std_area",
            "start_x",
            "start_y",
            "end_x",
            "end_y",
        ]
        subj_stats = cfg.get("aggregate_stats", ["mean", "std", "min", "max"])
        self._subject_stats = [str(s) for s in subj_stats]
        ordered_subject_keys = ["video_count"]
        for stat in self._subject_stats:
            for k in self._video_keys:
                ordered_subject_keys.append(f"{stat}__{k}")
        self._subject_keys = ordered_subject_keys

    def feature_order(self, level: str = "video") -> Sequence[str]:
        if str(level).lower() == "subject":
            return self._subject_keys
        return self._video_keys

    def extract_video(
        self,
        samples: Sequence[FramePrediction],
        video_path: Optional[str] = None,
    ) -> Dict[str, float]:
        if not samples:
            return {k: 0.0 for k in self._video_keys}
        frames = np.asarray([float(s.frame_index) for s in samples], dtype=np.float32)
        centers = np.asarray([s.center for s in samples], dtype=np.float32)
        bboxes = np.asarray([s.bbox for s in samples], dtype=np.float32)
        widths = bboxes[:, 2]
        heights = bboxes[:, 3]
        areas = widths * heights
        duration = float(frames[-1] - frames[0]) if frames.size > 1 else 0.0
        span_x = float(np.max(centers[:, 0]) - np.min(centers[:, 0])) if centers.size else 0.0
        span_y = float(np.max(centers[:, 1]) - np.min(centers[:, 1])) if centers.size else 0.0
        path_length = 0.0
        speeds: np.ndarray
        if centers.shape[0] > 1:
            diffs = np.diff(centers, axis=0)
            frame_deltas = np.diff(frames)
            frame_deltas[frame_deltas == 0] = 1.0
            step_dist = np.linalg.norm(diffs, axis=1)
            path_length = float(np.sum(step_dist))
            speeds = step_dist / np.maximum(frame_deltas, self._epsilon)
        else:
            speeds = np.zeros((0,), dtype=np.float32)
        if speeds.size > 1:
            accels = np.diff(speeds)
        else:
            accels = np.zeros((0,), dtype=np.float32)
        feature_map: Dict[str, float] = OrderedDict()
        feature_map["num_points"] = float(len(samples))
        feature_map["duration_frames"] = duration
        feature_map["span_x"] = span_x
        feature_map["span_y"] = span_y
        feature_map["path_length"] = path_length
        feature_map["mean_speed"] = float(np.mean(speeds)) if speeds.size else 0.0
        feature_map["max_speed"] = float(np.max(speeds)) if speeds.size else 0.0
        feature_map["std_speed"] = _safe_std(speeds)
        feature_map["median_speed"] = float(np.median(speeds)) if speeds.size else 0.0
        feature_map["p95_speed"] = float(np.percentile(speeds, 95)) if speeds.size else 0.0
        feature_map["mean_acc"] = float(np.mean(accels)) if accels.size else 0.0
        feature_map["max_acc"] = float(np.max(np.abs(accels))) if accels.size else 0.0
        feature_map["std_acc"] = _safe_std(accels)
        feature_map["mean_area"] = float(np.mean(areas)) if areas.size else 0.0
        feature_map["std_area"] = _safe_std(areas)
        feature_map["start_x"] = float(centers[0, 0]) if centers.size else 0.0
        feature_map["start_y"] = float(centers[0, 1]) if centers.size else 0.0
        feature_map["end_x"] = float(centers[-1, 0]) if centers.size else 0.0
        feature_map["end_y"] = float(centers[-1, 1]) if centers.size else 0.0
        return feature_map

    def aggregate_subject(self, video_features: Sequence[Dict[str, float]]) -> Dict[str, float]:
        if not video_features:
            return {key: 0.0 for key in self._subject_keys}
        agg: Dict[str, float] = OrderedDict()
        agg["video_count"] = float(len(video_features))
        feature_matrix = {
            key: np.asarray([vf.get(key, 0.0) for vf in video_features], dtype=np.float32)
            for key in self._video_keys
        }
        for stat in self._subject_stats:
            for key, values in feature_matrix.items():
                full_key = f"{stat}__{key}"
                if stat == "mean":
                    agg[full_key] = float(np.mean(values)) if values.size else 0.0
                elif stat == "std":
                    agg[full_key] = _safe_std(values)
                elif stat == "min":
                    agg[full_key] = float(np.min(values)) if values.size else 0.0
                elif stat == "max":
                    agg[full_key] = float(np.max(values)) if values.size else 0.0
                elif stat == "median":
                    agg[full_key] = float(np.median(values)) if values.size else 0.0
                else:
                    # fallback: mean
                    agg[full_key] = float(np.mean(values)) if values.size else 0.0
        # ensure missing keys filled
        for key in self._subject_keys:
            agg.setdefault(key, 0.0)
        return agg


def _choose_interpolation(name: str) -> int:
    lookup = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    return lookup.get(name.lower(), cv2.INTER_AREA)


def _resolve_device(device_hint: str | None) -> torch.device:
    if not device_hint:
        return torch.device("cpu")
    hint = str(device_hint).lower()
    if hint.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(hint)
        warnings.warn("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    if hint in ("mps", "metal"):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        warnings.warn("MPS requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device("cpu")


def _build_backbone(backbone_name: str, pretrained: bool) -> tuple[nn.Module, int, int]:
    name = str(backbone_name).replace("-", "").replace("_", "").lower()

    def _resolve_weights(attr: str) -> tuple[Any, bool]:
        weights_attr = getattr(models, attr, None)
        if weights_attr is None:
            return None, False
        if not pretrained:
            return None, True
        return getattr(weights_attr, "DEFAULT", None), True

    if name == "mobilenetv2":
        weights, has_enum = _resolve_weights("MobileNet_V2_Weights")
        if has_enum:
            model = models.mobilenet_v2(weights=weights)
        else:
            model = models.mobilenet_v2(pretrained=pretrained)
        feature_dim = model.classifier[1].in_features
        extractor = nn.Sequential(
            model.features,
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        input_size = 224
    elif name in {"resnet34"}:
        weights, has_enum = _resolve_weights("ResNet34_Weights")
        if has_enum:
            model = models.resnet34(weights=weights)
        else:
            model = models.resnet34(pretrained=pretrained)
        feature_dim = model.fc.in_features
        modules = list(model.children())[:-1]
        extractor = nn.Sequential(*modules)
        input_size = 224
    elif name in {"densenet121"}:
        weights, has_enum = _resolve_weights("DenseNet121_Weights")
        if has_enum:
            model = models.densenet121(weights=weights)
        else:
            model = models.densenet121(pretrained=pretrained)
        feature_dim = model.classifier.in_features
        extractor = nn.Sequential(
            model.features,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        input_size = 224
    elif name in {"efficientnetb2", "efficientnetv2b2"}:
        weights, has_enum = _resolve_weights("EfficientNet_B2_Weights")
        if has_enum:
            model = models.efficientnet_b2(weights=weights)
        else:
            model = models.efficientnet_b2(pretrained=pretrained)
        classifier = getattr(model, "classifier", None)
        if isinstance(classifier, nn.Sequential):
            last_linear = next((m for m in classifier.modules() if isinstance(m, nn.Linear)), None)
            feature_dim = last_linear.in_features if last_linear else 1408
        else:
            feature_dim = 1408
        extractor = nn.Sequential(
            model.features,
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        input_size = 260
    else:
        raise ValueError(
            "Unsupported backbone '{}'. Choose from: MobileNetV2, ResNet34, DenseNet121, EfficientNetB2.".format(
                backbone_name
            )
        )

    extractor.eval()
    extractor.requires_grad_(False)
    return extractor, int(feature_dim), int(input_size)


@register_feature_extractor("texture_hybrid")
class TextureHybridFeatureExtractor(TrajectoryFeatureExtractor):
    """Combine kinematic features with texture descriptors extracted from bounding boxes.

    動態特徵沿用 BasicTrajectoryFeatureExtractor；紋理部分會擷取指定影格的 bbox patch，
    經過灰階化與 resize 後計算統計量、梯度與灰階直方圖，再與動態特徵串接。
    """

    name = "TextureHybridFeatures"

    def __init__(self, params: Dict[str, Any] | None = None):
        cfg = params or {}

        self._dynamic = BasicTrajectoryFeatureExtractor(cfg.get("dynamic_params"))

        # Texture configuration
        self._patch_size = int(cfg.get("texture_patch_size", 96))
        self._hist_bins = max(4, int(cfg.get("texture_hist_bins", 16)))
        self._max_texture_frames = max(1, int(cfg.get("max_texture_frames", 3)))
        interp_name = str(cfg.get("resize_interpolation", "area"))
        self._resize_interpolation = _choose_interpolation(interp_name)
        self._normalise_histogram = bool(cfg.get("normalise_histogram", True))
        self._aggregate_stats = [str(s) for s in cfg.get("aggregate_stats", ["mean", "std", "min", "max"])]

        self._dynamic_keys = list(self._dynamic.feature_order("video"))
        base_texture_keys = [
            "tex_mean",
            "tex_std",
            "tex_min",
            "tex_max",
            "tex_energy",
            "tex_entropy",
            "tex_grad_mean",
            "tex_grad_std",
        ]
        hist_keys = [f"tex_hist_bin_{i:02d}" for i in range(self._hist_bins)]
        self._texture_keys = base_texture_keys + hist_keys

        self._video_keys = self._dynamic_keys + self._texture_keys

        ordered_subject_keys = ["video_count"]
        for stat in self._aggregate_stats:
            for key in self._video_keys:
                ordered_subject_keys.append(f"{stat}__{key}")
        self._subject_keys = ordered_subject_keys

    def feature_order(self, level: str = "video") -> Sequence[str]:
        if str(level).lower() == "subject":
            return self._subject_keys
        return self._video_keys

    def extract_video(
        self,
        samples: Sequence[FramePrediction],
        video_path: Optional[str] = None,
    ) -> Dict[str, float]:
        dynamic_features = self._dynamic.extract_video(samples, video_path=video_path)
        texture_features = self._compute_texture_features(samples, video_path)

        ordered = OrderedDict()
        for key in self._dynamic_keys:
            ordered[key] = dynamic_features.get(key, 0.0)
        for key in self._texture_keys:
            ordered[key] = texture_features.get(key, 0.0)
        return ordered

    def aggregate_subject(self, video_features: Sequence[Dict[str, float]]) -> Dict[str, float]:
        if not video_features:
            return {key: 0.0 for key in self._subject_keys}

        agg: Dict[str, float] = OrderedDict()
        agg["video_count"] = float(len(video_features))

        feature_matrix = {
            key: np.asarray([vf.get(key, 0.0) for vf in video_features], dtype=np.float32)
            for key in self._video_keys
        }

        for stat in self._aggregate_stats:
            for key, values in feature_matrix.items():
                full_key = f"{stat}__{key}"
                if stat == "mean":
                    agg[full_key] = float(np.mean(values)) if values.size else 0.0
                elif stat == "std":
                    agg[full_key] = _safe_std(values)
                elif stat == "min":
                    agg[full_key] = float(np.min(values)) if values.size else 0.0
                elif stat == "max":
                    agg[full_key] = float(np.max(values)) if values.size else 0.0
                elif stat == "median":
                    agg[full_key] = float(np.median(values)) if values.size else 0.0
                else:
                    agg[full_key] = float(np.mean(values)) if values.size else 0.0

        for key in self._subject_keys:
            agg.setdefault(key, 0.0)
        return agg

    # -------- internal helpers --------
    def _compute_texture_features(
        self,
        samples: Sequence[FramePrediction],
        video_path: Optional[str],
    ) -> Dict[str, float]:
        zeros = OrderedDict((key, 0.0) for key in self._texture_keys)
        if not samples or not video_path:
            return zeros

        frame_samples = self._select_samples(samples)
        if not frame_samples:
            return zeros

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return zeros

        patch_stats: Dict[str, List[float]] = {k: [] for k in self._texture_keys if not k.startswith("tex_hist_bin_")}
        hist_accumulator: List[np.ndarray] = []

        try:
            for sample in frame_samples:
                patch = self._read_patch(cap, sample)
                if patch is None:
                    continue
                stats, hist = self._describe_patch(patch)
                for key, value in stats.items():
                    patch_stats[key].append(value)
                hist_accumulator.append(hist)
        finally:
            cap.release()

        if not hist_accumulator:
            return zeros

        combined = OrderedDict()
        for key in patch_stats:
            values = patch_stats[key]
            combined[key] = float(np.mean(values)) if values else 0.0

        hist_stack = np.stack(hist_accumulator, axis=0)
        hist_mean = hist_stack.mean(axis=0)
        for idx, value in enumerate(hist_mean):
            combined[f"tex_hist_bin_{idx:02d}"] = float(value)

        # Ensure all keys present
        for key in self._texture_keys:
            combined.setdefault(key, 0.0)
        return combined

    def _select_samples(self, samples: Sequence[FramePrediction]) -> Sequence[FramePrediction]:
        total = len(samples)
        if total == 0:
            return []
        if total <= self._max_texture_frames:
            return samples
        positions = np.linspace(0, total - 1, self._max_texture_frames, dtype=int)
        uniq = sorted(set(int(idx) for idx in positions))
        return [samples[i] for i in uniq]

    def _read_patch(self, cap: cv2.VideoCapture, sample: FramePrediction):
        frame_idx = max(0, int(round(sample.frame_index)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None

        h, w = frame.shape[:2]
        x, y, bw, bh = sample.bbox
        x0 = max(0, int(np.floor(x)))
        y0 = max(0, int(np.floor(y)))
        x1 = min(w, int(np.ceil(x + bw)))
        y1 = min(h, int(np.ceil(y + bh)))
        if x1 <= x0 or y1 <= y0:
            return None

        patch = frame[y0:y1, x0:x1]
        if patch.size == 0:
            return None

        if patch.ndim == 3:
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

        if self._patch_size > 0:
            patch = cv2.resize(
                patch,
                (self._patch_size, self._patch_size),
                interpolation=self._resize_interpolation,
            )

        return patch

    def _describe_patch(self, patch: np.ndarray) -> tuple[Dict[str, float], np.ndarray]:
        patch_f = patch.astype(np.float32)
        mean_val = float(np.mean(patch_f))
        std_val = float(np.std(patch_f))
        min_val = float(np.min(patch_f))
        max_val = float(np.max(patch_f))

        norm_patch = patch_f / 255.0
        energy = float(np.mean(norm_patch ** 2))

        hist, _ = np.histogram(patch_f, bins=self._hist_bins, range=(0, 255))
        hist = hist.astype(np.float32)
        if self._normalise_histogram:
            hist_sum = float(np.sum(hist))
            if hist_sum > 0:
                hist /= hist_sum
        entropy = float(-np.sum(hist * np.log2(hist + 1e-12)))

        grad_x = cv2.Sobel(patch_f, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(patch_f, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        grad_mean = float(np.mean(grad_mag))
        grad_std = float(np.std(grad_mag))

        stats = {
            "tex_mean": mean_val,
            "tex_std": std_val,
            "tex_min": min_val,
            "tex_max": max_val,
            "tex_energy": energy,
            "tex_entropy": entropy,
            "tex_grad_mean": grad_mean,
            "tex_grad_std": grad_std,
        }
        return stats, hist


@register_feature_extractor("backbone_texture")
class BackboneTextureFeatureExtractor(TrajectoryFeatureExtractor):
    """Extract texture embeddings via configurable CNN backbones and combine with kinematics.

    The extractor crops trajectory bounding boxes, feeds them through a backbone network,
    optionally reduces dimensionality with random projection, and aggregates the embeddings.
    """

    name = "BackboneTextureFeatures"

    _POOLING_FUNCS = {
        "mean": np.mean,
        "std": np.std,
        "max": np.max,
        "min": np.min,
    }

    def __init__(self, params: Dict[str, Any] | None = None):
        cfg = params or {}

        # Dynamic (kinematic) branch reuses the basic extractor
        self._dynamic = BasicTrajectoryFeatureExtractor(cfg.get("dynamic_params"))
        self._dynamic_keys = list(self._dynamic.feature_order("video"))

        # Backbone configuration
        backbone_name = cfg.get("backbone", "mobilenetv2")
        pretrained = bool(cfg.get("pretrained", False))
        self._device = _resolve_device(cfg.get("device", "cpu"))
        self._backbone, feature_dim, input_size = _build_backbone(backbone_name, pretrained)
        self._backbone = self._backbone.to(self._device)
        self._input_size = int(cfg.get("input_size", input_size))

        # Frame sampling + preprocessing config
        self._max_texture_frames = max(1, int(cfg.get("max_texture_frames", 3)))
        interp_name = str(cfg.get("resize_interpolation", "area"))
        self._resize_interpolation = _choose_interpolation(interp_name)
        self._zscore_patch = bool(cfg.get("zscore_patch", False))

        # Embedding post-processing
        self._pool_stats = [str(s).lower() for s in cfg.get("pool_stats", ["mean"])]
        self._pool_stats = [s for s in self._pool_stats if s in self._POOLING_FUNCS]
        if not self._pool_stats:
            self._pool_stats = ["mean"]

        reduction_method = str(cfg.get("reduction_method", "random_projection"))
        target_dim = int(cfg.get("reduced_dim", 64))
        target_dim = max(1, target_dim)
        self._feature_reducer: GaussianRandomProjection | None = None
        self._reducer_fitted = False
        if reduction_method.lower() == "random_projection" and target_dim < feature_dim:
            random_state = cfg.get("random_state", 1337)
            self._feature_reducer = GaussianRandomProjection(
                n_components=target_dim,
                random_state=random_state,
            )
            self._output_dim = target_dim
        else:
            self._output_dim = feature_dim

        stat_prefix = cfg.get("texture_prefix", "tex_backbone")
        self._texture_keys: List[str] = []
        for stat in self._pool_stats:
            for idx in range(self._output_dim):
                self._texture_keys.append(f"{stat_prefix}_{stat}_{idx:03d}")

        self._video_keys = self._dynamic_keys + self._texture_keys

        # Subject-level aggregation mirrors other extractors
        self._aggregate_stats = [str(s) for s in cfg.get("aggregate_stats", ["mean", "std", "min", "max"])]
        ordered_subject_keys = ["video_count"]
        for stat in self._aggregate_stats:
            for key in self._video_keys:
                ordered_subject_keys.append(f"{stat}__{key}")
        self._subject_keys = ordered_subject_keys

        self._normalise_mean = cfg.get("normalise_mean", (0.485, 0.456, 0.406))
        self._normalise_std = cfg.get("normalise_std", (0.229, 0.224, 0.225))
        self._normalise = transforms.Normalize(self._normalise_mean, self._normalise_std)

    def feature_order(self, level: str = "video") -> Sequence[str]:
        if str(level).lower() == "subject":
            return self._subject_keys
        return self._video_keys

    def extract_video(
        self,
        samples: Sequence[FramePrediction],
        video_path: Optional[str] = None,
    ) -> Dict[str, float]:
        dynamic_features = self._dynamic.extract_video(samples, video_path=video_path)
        texture_features = self._compute_backbone_features(samples, video_path)

        combined = OrderedDict()
        for key in self._dynamic_keys:
            combined[key] = dynamic_features.get(key, 0.0)
        for key in self._texture_keys:
            combined[key] = texture_features.get(key, 0.0)
        return combined

    def aggregate_subject(self, video_features: Sequence[Dict[str, float]]) -> Dict[str, float]:
        if not video_features:
            return {key: 0.0 for key in self._subject_keys}

        agg: Dict[str, float] = OrderedDict()
        agg["video_count"] = float(len(video_features))

        feature_matrix = {
            key: np.asarray([vf.get(key, 0.0) for vf in video_features], dtype=np.float32)
            for key in self._video_keys
        }

        for stat in self._aggregate_stats:
            for key, values in feature_matrix.items():
                full_key = f"{stat}__{key}"
                if stat == "mean":
                    agg[full_key] = float(np.mean(values)) if values.size else 0.0
                elif stat == "std":
                    agg[full_key] = _safe_std(values)
                elif stat == "min":
                    agg[full_key] = float(np.min(values)) if values.size else 0.0
                elif stat == "max":
                    agg[full_key] = float(np.max(values)) if values.size else 0.0
                elif stat == "median":
                    agg[full_key] = float(np.median(values)) if values.size else 0.0
                else:
                    agg[full_key] = float(np.mean(values)) if values.size else 0.0

        for key in self._subject_keys:
            agg.setdefault(key, 0.0)
        return agg

    # -------- internal helpers --------
    def _compute_backbone_features(
        self,
        samples: Sequence[FramePrediction],
        video_path: Optional[str],
    ) -> Dict[str, float]:
        zeros = OrderedDict((key, 0.0) for key in self._texture_keys)
        if not samples or not video_path:
            return zeros

        selected_samples = self._select_samples(samples)
        if not selected_samples:
            return zeros

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return zeros

        embeddings: List[np.ndarray] = []

        try:
            for sample in selected_samples:
                patch = self._read_patch(cap, sample)
                if patch is None:
                    continue
                tensor = self._preprocess_patch(patch)
                with torch.no_grad():
                    embedding = self._backbone(tensor)
                embedding = embedding.view(-1)
                embeddings.append(embedding.cpu().numpy())
        finally:
            cap.release()

        if not embeddings:
            return zeros

        features = np.stack(embeddings, axis=0)

        if self._feature_reducer is not None:
            if not self._reducer_fitted:
                self._feature_reducer.fit(features)
                self._reducer_fitted = True
            features = self._feature_reducer.transform(features)

        combined = OrderedDict()
        for stat in self._pool_stats:
            func = self._POOLING_FUNCS[stat]
            if stat == "std":
                pooled = func(features, axis=0, ddof=0)
            else:
                pooled = func(features, axis=0)
            start_idx = self._pool_stats.index(stat) * self._output_dim
            for offset, value in enumerate(pooled):
                key = self._texture_keys[start_idx + offset]
                combined[key] = float(value)

        for key in self._texture_keys:
            combined.setdefault(key, 0.0)
        return combined

    def _select_samples(self, samples: Sequence[FramePrediction]) -> Sequence[FramePrediction]:
        total = len(samples)
        if total == 0:
            return []
        if total <= self._max_texture_frames:
            return samples
        positions = np.linspace(0, total - 1, self._max_texture_frames, dtype=int)
        uniq = sorted(set(int(idx) for idx in positions))
        return [samples[i] for i in uniq]

    def _read_patch(self, cap: cv2.VideoCapture, sample: FramePrediction):
        frame_idx = max(0, int(round(sample.frame_index)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None

        h, w = frame.shape[:2]
        x, y, bw, bh = sample.bbox
        x0 = max(0, int(np.floor(x)))
        y0 = max(0, int(np.floor(y)))
        x1 = min(w, int(np.ceil(x + bw)))
        y1 = min(h, int(np.ceil(y + bh)))
        if x1 <= x0 or y1 <= y0:
            return None

        patch = frame[y0:y1, x0:x1]
        if patch.size == 0:
            return None
        return patch

    def _preprocess_patch(self, patch: np.ndarray) -> torch.Tensor:
        if patch.ndim == 2:
            patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        if self._input_size > 0:
            patch = cv2.resize(
                patch,
                (self._input_size, self._input_size),
                interpolation=self._resize_interpolation,
            )
        tensor = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
        if self._zscore_patch:
            tensor = (tensor - tensor.mean()) / (tensor.std() + 1e-6)
        tensor = self._normalise(tensor)
        return tensor.unsqueeze(0).to(self._device)
