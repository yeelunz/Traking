import os
import tempfile

import cv2
import numpy as np
import pytest

from torchvision import models

from tracking.classification.feature_extractors import (
    TextureHybridFeatureExtractor,
    BackboneTextureFeatureExtractor,
)
from tracking.core.interfaces import FramePrediction


def _make_temp_video(path: str, frame_count: int = 8) -> str:
    height, width = 72, 96
    video_path = os.path.join(path, "temp_video.avi")
    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"XVID"),
        15.0,
        (width, height),
        True,
    )
    if not writer.isOpened():
        raise RuntimeError("Unable to create temporary test video")
    for i in range(frame_count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.rectangle(
            frame,
            (10 + i, 15),
            (10 + i + 30, 15 + 20),
            (20 * i % 255, 120, 255 - 15 * i % 255),
            -1,
        )
        writer.write(frame)
    writer.release()
    return video_path


def _dummy_predictions() -> list[FramePrediction]:
    boxes = [
        (12.0, 14.0, 32.0, 24.0),
        (18.0, 16.0, 26.0, 22.0),
        (20.0, 18.0, 24.0, 20.0),
    ]
    frames = [0, 3, 6]
    return [FramePrediction(frame_index=fi, bbox=bb) for fi, bb in zip(frames, boxes)]


def test_texture_hybrid_video_features(tmp_path):
    video_path = _make_temp_video(tmp_path)
    samples = _dummy_predictions()

    extractor = TextureHybridFeatureExtractor(
        {
            "texture_hist_bins": 8,
            "texture_patch_size": 48,
            "max_texture_frames": 3,
            "dynamic_params": {"aggregate_stats": ["mean", "std"]},
        }
    )

    features = extractor.extract_video(samples, video_path=video_path)

    video_keys = extractor.feature_order("video")
    assert set(video_keys) == set(features.keys())
    assert all(np.isfinite(list(features.values())))

    hist_keys = [k for k in video_keys if k.startswith("tex_hist_bin_")]
    if hist_keys:
        hist_sum = sum(features[k] for k in hist_keys)
        assert pytest.approx(hist_sum, rel=1e-3, abs=1e-3) == 1.0


def test_texture_hybrid_subject_aggregation(tmp_path):
    video_path = _make_temp_video(tmp_path)
    samples = _dummy_predictions()
    extractor = TextureHybridFeatureExtractor(
        {
            "texture_hist_bins": 8,
            "texture_patch_size": 48,
            "max_texture_frames": 2,
            "aggregate_stats": ["mean", "max"],
        }
    )

    features = extractor.extract_video(samples, video_path=video_path)
    agg = extractor.aggregate_subject([features])

    assert agg["video_count"] == pytest.approx(1.0)
    subject_keys = extractor.feature_order("subject")
    assert set(subject_keys) == set(agg.keys())
    assert all(np.isfinite(list(agg.values())))


@pytest.mark.parametrize(
    "backbone_name, attr_name",
    [
        ("MobileNetV2", "mobilenet_v2"),
        ("ResNet34", "resnet34"),
        ("DenseNet121", "densenet121"),
        ("EfficientNetB2", "efficientnet_b2"),
    ],
)
def test_backbone_texture_feature_dimensions(tmp_path, backbone_name, attr_name):
    if not hasattr(models, attr_name):
        pytest.skip(f"torchvision does not provide {attr_name}")

    video_path = _make_temp_video(tmp_path, frame_count=5)
    samples = _dummy_predictions()
    extractor = BackboneTextureFeatureExtractor(
        {
            "backbone": backbone_name,
            "pretrained": False,
            "max_texture_frames": 1,
            "reduction_method": "random_projection",
            "reduced_dim": 8,
            "pool_stats": ["mean", "std"],
        }
    )

    features = extractor.extract_video(samples, video_path=video_path)
    video_keys = extractor.feature_order("video")

    assert set(video_keys) == set(features.keys())
    assert all(np.isfinite(list(features.values())))

    texture_keys = [k for k in video_keys if k.startswith("tex_backbone_")]
    assert len(texture_keys) == 16  # 8 dims * 2 pooling statistics


def test_backbone_texture_missing_video_returns_zero(tmp_path):
    extractor = BackboneTextureFeatureExtractor(
        {
            "backbone": "mobilenet_v2",
            "pretrained": False,
            "max_texture_frames": 1,
            "reduction_method": "random_projection",
            "reduced_dim": 4,
            "pool_stats": ["mean"],
        }
    )

    features = extractor.extract_video(_dummy_predictions(), video_path=None)

    texture_keys = [k for k in extractor.feature_order("video") if k.startswith("tex_backbone_")]
    assert texture_keys
    assert all(features[key] == pytest.approx(0.0) for key in texture_keys)
