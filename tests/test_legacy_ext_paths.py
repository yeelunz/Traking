from __future__ import annotations

from pathlib import Path

import tracking.classification.classifiers.legacy_ext as legacy_ext


def test_legacy_ext_libs_dir_resolves_from_repo_root():
    repo_root = Path(__file__).resolve().parents[1]
    assert legacy_ext._LIBS_DIR == repo_root / "libs"


def test_legacy_ext_multirocket_dir_uses_repo_libs():
    repo_root = Path(__file__).resolve().parents[1]
    assert Path(legacy_ext._MR_DIR) == repo_root / "libs" / "MultiRocket"
