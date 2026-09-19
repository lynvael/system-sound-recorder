"""Entry point: launches the PySide6 GUI.

The GUI (app/gui/*) is built in parallel by another engineer; this module only
wires the entrypoint. Run with `python -m app` or the `live-recorder` script.
"""

from __future__ import annotations

from app.config import load_config
from app.log import setup as setup_logging


def _run() -> None:
    config = load_config()
    setup_logging(config.app.log_level)

    # Imported lazily so `python -m app.audio.devices` and other tools don't
    # pull in Qt. The GUI module is owned by another engineer.
    from app.gui.main_window import main

    main()


if __name__ == "__main__":
    _run()
