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

# V3-Lite implementations (motion_static_lite, time_series_v3lite — no deep learning)
from . import feature_extractors_v3lite  # noqa: F401

# Dimensionality reduction utilities (UMAP, LDA, Autoencoder, Learned Projection)
from . import dim_reduction  # noqa: F401

__all__ = [
    "feature_extractors",
    "classifiers",
    "feature_vector",
    "metrics",
    "feature_extractors_ext",
    "classifiers_ext",
    "feature_extractors_v3lite",
    "dim_reduction",
]
