from .base import *  # noqa: F401,F403

from . import ext as _load_ext  # noqa: F401
from . import nn as _load_nn  # noqa: F401
from . import limix as _load_limix  # noqa: F401

# `v3pro` used to live under this package in older layouts.
# Keep a soft import for compatibility, but do not fail initialization
# when the module is not present.
try:  # pragma: no cover - compatibility shim
	from . import v3pro as _load_v3pro  # noqa: F401
except Exception:  # pragma: no cover - missing legacy module
	_load_v3pro = None
