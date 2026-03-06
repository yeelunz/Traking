"""Classification utilities for subject-level diagnosis using tracking outputs."""

# Import default implementations so they register themselves
from . import feature_extractors  # noqa: F401
from . import classifiers  # noqa: F401
from . import feature_vector  # noqa: F401
from . import metrics  # noqa: F401

# Extended implementations (motion_texture_static, time_series, xgboost, tabpfn_v2,
# multirocket, patchtst, timemachine)
from . import feature_extractors_ext  # noqa: F401
from . import classifiers_ext  # noqa: F401

__all__ = [
    "feature_extractors",
    "classifiers",
    "feature_vector",
    "metrics",
    "feature_extractors_ext",
    "classifiers_ext",
]
