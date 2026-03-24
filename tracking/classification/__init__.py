"""Classification utilities for subject-level diagnosis using tracking outputs."""

# Import default implementations so they register themselves
from . import feature_extractors  # noqa: F401
from . import classifiers  # noqa: F401
from . import feature_vector  # noqa: F401
from . import metrics  # noqa: F401

# Extended implementations (tab_v2, tsc_v2, xgboost, tabpfn_v2/tabpfn_2_5,
# multirocket, patchtst, timemachine)
from . import feature_extractors_ext  # noqa: F401
from . import classifiers_ext  # noqa: F401

# V3-Lite implementations (tab_v3_lite, tsc_v3_lite — no deep learning)
from . import feature_extractors_v3lite  # noqa: F401

# V3-Pro implementations (texture branch + fusion_mlp classifier; v3pro_fusion kept as alias)
from . import feature_extractors_v3pro  # noqa: F401
from . import feature_extractors_v4  # noqa: F401
from . import feature_extractors_v3pro_tsc  # noqa: F401
from . import classifiers_v3pro  # noqa: F401
from . import classifiers_limix  # noqa: F401

# Differentiable tabular heads (MLP / Linear head)
from . import classifiers_nn  # noqa: F401

# Independent fusion modules (concat / gating / attention)
from . import fusion_modules  # noqa: F401

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
    "feature_extractors_v3pro",
    "feature_extractors_v4",
    "feature_extractors_v3pro_tsc",
    "classifiers_v3pro",
    "classifiers_limix",
    "classifiers_nn",
    "fusion_modules",
    "dim_reduction",
]
