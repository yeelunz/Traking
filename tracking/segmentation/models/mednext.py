from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

import torch.nn as nn

from ...core.registry import register_segmentation_model

_MEDNEXT_IMPORT_ERROR: Optional[Exception] = None
MedNeXt = None


def _try_import_mednext() -> None:
    global MedNeXt, _MEDNEXT_IMPORT_ERROR
    if MedNeXt is not None:
        return
    try:
        from nnunet_mednext.network_architecture.mednextv1.MedNextV1 import MedNeXt as _MedNeXt  # type: ignore
        MedNeXt = _MedNeXt
        return
    except Exception as exc:
        _MEDNEXT_IMPORT_ERROR = exc

    # Try local vendor path: libs/MedNeXt
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        vendor_root = os.path.join(project_root, "libs", "MedNeXt")
        if os.path.isdir(vendor_root):
            sys.path.insert(0, vendor_root)
        from nnunet_mednext.network_architecture.mednextv1.MedNextV1 import MedNeXt as _MedNeXt  # type: ignore
        MedNeXt = _MedNeXt
        _MEDNEXT_IMPORT_ERROR = None
    except Exception as exc:
        _MEDNEXT_IMPORT_ERROR = exc


_MEDNEXT_DEFAULTS = {
    "S": {
        "exp_r": 2,
        "block_counts": [2, 2, 2, 2, 2, 2, 2, 2, 2],
        "checkpoint_style": None,
    },
    "B": {
        "exp_r": [2, 3, 4, 4, 4, 4, 4, 3, 2],
        "block_counts": [2, 2, 2, 2, 2, 2, 2, 2, 2],
        "checkpoint_style": None,
    },
    "M": {
        "exp_r": [2, 3, 4, 4, 4, 4, 4, 3, 2],
        "block_counts": [3, 4, 4, 4, 4, 4, 4, 4, 3],
        "checkpoint_style": "outside_block",
    },
    "L": {
        "exp_r": [3, 4, 8, 8, 8, 8, 8, 4, 3],
        "block_counts": [3, 4, 8, 8, 8, 8, 8, 4, 3],
        "checkpoint_style": "outside_block",
    },
}


@register_segmentation_model("mednext")
class MedNeXtSegmenter(nn.Module):
    """MedNeXt segmentation wrapper (2D).

    Params:
        - in_channels (int, default=3)
        - classes (int, default=1)
        - model_id (str: S/B/M/L, default=B)
        - kernel_size (int, default=3)
        - deep_supervision (bool, default=False)
        - base_channels (int, default=32)
        - exp_r (int|list, optional)
        - block_counts (list, optional)
        - do_res (bool, default=True)
        - do_res_up_down (bool, default=True)
        - norm_type (str, default="group")
        - checkpoint_style (str|None, optional)
        - grn (bool, default=False)
        - dim (str, default="2d")
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__()
        _try_import_mednext()
        if MedNeXt is None:
            raise ImportError(
                "MedNeXt is required for 'mednext'. "
                "Ensure libs/MedNeXt exists or install via `pip install -e libs/MedNeXt`. "
                f"Original error: {_MEDNEXT_IMPORT_ERROR}"
            )

        cfg = dict(params or {})
        in_channels = int(cfg.pop("in_channels", cfg.pop("channels", 3)))
        classes = int(cfg.pop("classes", cfg.pop("num_classes", 1)))
        model_id = str(cfg.pop("model_id", cfg.pop("size", "B"))).strip().upper() or "B"
        kernel_size = int(cfg.pop("kernel_size", 3))
        deep_supervision = bool(cfg.pop("deep_supervision", False))
        n_channels = int(cfg.pop("base_channels", 32))
        do_res = bool(cfg.pop("do_res", True))
        do_res_up_down = bool(cfg.pop("do_res_up_down", True))
        norm_type = str(cfg.pop("norm_type", "group"))
        grn = bool(cfg.pop("grn", False))
        dim = str(cfg.pop("dim", "2d")).lower()

        defaults = _MEDNEXT_DEFAULTS.get(model_id, _MEDNEXT_DEFAULTS["B"])
        exp_r = cfg.pop("exp_r", defaults["exp_r"])
        block_counts = cfg.pop("block_counts", defaults["block_counts"])
        checkpoint_style = cfg.pop("checkpoint_style", defaults["checkpoint_style"])

        self.model = MedNeXt(
            in_channels=in_channels,
            n_channels=n_channels,
            n_classes=classes,
            exp_r=exp_r,
            kernel_size=kernel_size,
            deep_supervision=deep_supervision,
            do_res=do_res,
            do_res_up_down=do_res_up_down,
            checkpoint_style=checkpoint_style,
            block_counts=block_counts,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

    def forward(self, x):  # type: ignore[override]
        out = self.model(x)
        if isinstance(out, (list, tuple)) and out:
            return out[0]
        return out


__all__ = ["MedNeXtSegmenter"]