"""Compatibility façade for extended classifiers.

This module remains the public import path used across the codebase:
    from tracking.classification.classifiers_ext import set_progress_logger
"""

from .legacy_ext import set_progress_logger  # re-export

from . import xgboost as _load_xgboost  # noqa: F401
from . import tabpfn as _load_tabpfn  # noqa: F401
from . import multirocket as _load_multirocket  # noqa: F401
from . import patchtst as _load_patchtst  # noqa: F401
from . import timesnet as _load_timesnet  # noqa: F401
from . import timemachine as _load_timemachine  # noqa: F401

__all__ = ["set_progress_logger"]
