"""Single lazy PyAudioWPatch (PortAudio) backend for the whole process.

`PyAudioWPatch` ships Windows-only wheels (PyPI: win32/win_amd64), and
`Pa_Initialize()` is refcounted, so the process must share ONE `PyAudio()`
instance: device enumeration and both capture threads open their streams
through it. The instance is created LAZILY on first use (in the GUI that is
after `QApplication` exists — see the COM note in app/gui/main_window.py),
never at import time.

Importing `pyaudiowpatch` itself is also deferred to first use: on the Linux
dev box the package is not installed, and modules that only need the
file/batch pipeline (CLI `transcribe`, tests) must stay importable without it.

This module is the ONLY place the `PyAudio()` instance is created. Other
modules may import `pyaudiowpatch` (e.g. `app.audio.capture` does, inside
`run()`, for the `paFloat32` constant), but only function-level — never at
module scope.
"""

from __future__ import annotations

import atexit
import threading

from app.log import get_logger

logger = get_logger("backend")

_instance = None
_lock = threading.Lock()


def get_backend():
    """Return the shared `PyAudio()` instance, creating it on first call.

    Thread-safe (double-checked locking). Raises RuntimeError with a Russian
    message when `pyaudiowpatch` cannot be imported (i.e. not on Windows).
    A PortAudio init failure (e.g. COM error) propagates as the OSError the
    C layer raises (code in `errno`).
    """
    global _instance
    if _instance is not None:
        return _instance
    with _lock:
        if _instance is not None:
            return _instance
        try:
            import pyaudiowpatch as pyaudio
        except ImportError as exc:
            raise RuntimeError(
                "PyAudioWPatch доступен только на Windows: пакет не "
                f"установлен в этом окружении (import pyaudiowpatch: {exc})."
            ) from exc
        _instance = pyaudio.PyAudio()
        logger.debug("PortAudio initialized (lazy PyAudio() singleton)")
        return _instance


def shutdown() -> None:
    """Terminate PortAudio and drop the instance. Idempotent.

    Registered with `atexit`; a second call (or a call before any use) is a
    no-op. After a shutdown, a subsequent `get_backend()` creates a fresh
    instance — PortAudio's refcounted Pa_Initialize/Pa_Terminate pair allows
    re-initialization, and the process only shuts down once anyway.
    """
    global _instance
    with _lock:
        inst, _instance = _instance, None
    if inst is None:
        return
    try:
        inst.terminate()  # closes all streams, then Pa_Terminate()
    except Exception:  # noqa: BLE001 - best effort at process shutdown
        logger.exception("Error terminating PortAudio")
    logger.debug("PortAudio terminated")


atexit.register(shutdown)
