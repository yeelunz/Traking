"""Compatibility bridge for legacy `tracking.classification.classifiers.v3pro`.

Some historical layouts exposed V3-Pro classifier symbols from
`tracking.classification.classifiers.v3pro`. Newer layouts may not ship that
module. Keep import-time behavior non-fatal so package initialization can
continue.
"""

try:  # pragma: no cover - compatibility shim
	from .classifiers.v3pro import *  # type: ignore # noqa: F401,F403
except Exception:  # pragma: no cover - missing legacy module
	# No legacy v3pro module in current layout; nothing to export.
	pass
