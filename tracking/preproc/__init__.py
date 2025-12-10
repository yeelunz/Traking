"""Preprocessing modules package.
Import modules here to ensure registry population when package imported."""

from . import clahe  # noqa: F401
from . import srad   # noqa: F401
from . import logdr  # noqa: F401
from . import tgc    # noqa: F401
from . import augment  # noqa: F401

__all__ = [
	"clahe",
	"srad",
	"logdr",
	"tgc",
	"augment",
]
