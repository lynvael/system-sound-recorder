"""PySide6 GUI layer for Live Recorder.

Only this package (``app/gui/``) is owned by the frontend engineer. It is
built strictly against the shared contract documented in
``kind-hatching-allen.md`` (see the "### GUI (`app/gui/`)" section) and in
the module docstrings of :mod:`app.gui.worker` and :mod:`app.gui.main_window`.
"""

from app.gui.main_window import main

__all__ = ["main"]
