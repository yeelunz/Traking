import cv2
import numpy as np
import pytest

from tracking.segmentation.utils import BoundingBox
from tracking.segmentation.workflow import (
    AUTO_MASK_RUNTIME_AVAILABLE,
    SegmentationWorkflow,
)


@pytest.mark.skipif(not AUTO_MASK_RUNTIME_AVAILABLE, reason="Auto-mask helpers unavailable")
def test_auto_mask_generate_full_simple_shape():
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    cv2.circle(frame, (64, 64), 24, (255, 255, 255), -1)
    bbox = BoundingBox(40.0, 40.0, 48.0, 48.0)

    workflow = SegmentationWorkflow(
        config={
            "model_name": "auto_mask",
            "model_params": {
                "margin": 0.1,
                "num_iter": 80,
                "canny_low": 20,
                "canny_high": 60,
            },
        }
    )

    mask = workflow._auto_mask_generate_full(frame, bbox)

    assert mask is not None, "Auto-mask should return a mask for a simple synthetic target"
    assert mask.shape == frame.shape[:2]
    assert mask.dtype == np.uint8
    assert np.count_nonzero(mask) > 0, "Mask should contain foreground pixels"

    roi_mask = mask[int(bbox.y) : int(bbox.y + bbox.h), int(bbox.x) : int(bbox.x + bbox.w)]
    assert np.count_nonzero(roi_mask) > 0, "ROI region should include the detected mask"
