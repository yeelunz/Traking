"""
Basic regression for UI signal order bug.

When a parameter widget emits a change event before the raw text editor has
been created, the builder-changed callback used to try to write the raw text
and would crash with an AttributeError because `self.txt_cfg` did not yet
exist.  The fix adds guards in `_on_builder_changed` and
`_set_raw_text_programmatically`.

This test simply exercises those methods both with and without the widget
present.
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.getcwd())
from ui import SimpleRunnerUI


def test_builder_changed_guard():
    # ensure QApplication exists (required by Qt widgets)
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    ui = SimpleRunnerUI()
    # artificially remove the editor before invoking
    if hasattr(ui, 'txt_cfg'):
        delattr(ui, 'txt_cfg')
    # should not raise
    ui._on_builder_changed()
    ui._set_raw_text_programmatically('foo')

    # now restore a dummy attribute and verify functionality
    class Dummy:
        def blockSignals(self, x): pass
        def setPlainText(self, t): pass
    ui.txt_cfg = Dummy()
    ui._on_builder_changed()
    ui._set_raw_text_programmatically('bar')

    # cleanup
    ui.close()
    app.quit()
