"""Qt bridge between :class:`app.pipeline.session.Session` and the GUI.

`Session` invokes its callbacks (``on_segment``, ``on_status``, ``on_backlog``,
``on_error``, ``on_finished``) from BACKGROUND threads (capture / VAD / STT
worker threads). :class:`SessionWorker` re-emits every callback as a Qt
signal so the main window only ever touches widgets on the Qt event-loop
thread.

Threading model used by ``app/gui/main_window.py``:
    - A single :class:`SessionWorker` instance is moved to its own
      :class:`~PySide6.QtCore.QThread` (started once, for the app's
      lifetime).
    - The main window talks to it exclusively through the
      ``request_start``/``request_stop`` signals declared on the window
      (connected to :meth:`SessionWorker.start_session` /
      :meth:`SessionWorker.stop_session`). Because the worker object lives on
      a different thread than the emitting (UI) thread, Qt's default
      ``AutoConnection`` automatically becomes a queued connection, so both
      slots run on the worker thread -- never on the UI thread. This is what
      keeps ``stop()`` (which can block for a while running the batch pass)
      from freezing the UI.
    - Signals emitted *from* ``Session``'s own background threads
      (``segment_ready`` etc.) are, symmetrically, auto-queued back onto
      whatever thread the connected slot (a MainWindow method) lives on --
      the Qt GUI thread -- because signal/slot dispatch is decided by
      comparing the *emitting* thread to the *receiving object*'s thread,
      not by which thread owns the signal's declaring QObject.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal, Slot

# --- shared contract with the backend (app/pipeline/, app/config.py) -------
# TODO(reconcile-with-backend): these imports assume the backend engineer's
# final module/class names exactly match kind-hatching-allen.md. If a name
# changes, this file (plus `load_app_config` below) is the single place to
# fix.
from app.pipeline.session import Session
from app.pipeline.transcript import Segment  # noqa: F401  (re-exported for type hints/consumers)

try:
    from app.config import load_config
except Exception:  # pragma: no cover - backend module not available yet
    load_config = None  # type: ignore[assignment]


def load_app_config():
    """Single centralized point of config access for the GUI.

    Confirmed against `app/config.py` (already built by the backend
    engineer): `load_config()` returns a `Config` aggregate exposing
    `.app`, `.capture`, `.stt`, `.vad`, `.session` sub-settings objects
    (each a `pydantic_settings.BaseSettings`). This helper is the ONLY place
    in the GUI that constructs it, and `SessionWorker.start_session` is the
    only place that mutates it (`config.stt.engine`, `config.stt.language`,
    `config.session.mode`) before handing it to `Session(config, mode, ...)`.

    TODO(reconcile-with-backend): `app/pipeline/session.py` does not exist
    yet, so it is still unverified that `Session.__init__` reads config
    overrides from these exact attribute paths (vs., e.g., taking
    `mode`/`engine`/`language` as explicit constructor args instead of
    mutating `config` in place). If `Session` turns out to ignore mutated
    `config.stt.engine` / `config.stt.language` / `config.session.mode`,
    only `SessionWorker.start_session` needs to change.
    """
    if load_config is None:
        raise RuntimeError(
            "app.config.load_config is not importable yet -- backend module "
            "not built. See load_app_config() TODO in app/gui/worker.py."
        )
    return load_config()


@dataclass
class SessionParams:
    """Everything the GUI collects from its selectors to start a session."""

    mode: str  # "live" | "batch" | "file"
    mic_id: Optional[str] = None
    loopback_id: Optional[str] = None
    import_path: Optional[Path] = None
    engine: str = "whisper"  # "whisper" | "gigaam"
    # "russian" | "english" | "auto" -- matches app.config.STTSettings.language
    # verbatim (not ISO codes); "auto" is mapped to None below for the
    # whisper pipeline's generate_kwargs={"language": ...}.
    language: str = "auto"


class SessionWorker(QObject):
    """Owns the `Session` instance and re-emits its callbacks as Qt signals."""

    segment_ready = Signal(object)  # Segment
    status_changed = Signal(str)
    backlog_changed = Signal(int)  # STT queue depth
    error = Signal(str)
    finished = Signal(object)  # Path to session dir
    session_dir_ready = Signal(object)  # Path to session dir, known at session start

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._session: Optional[Session] = None

    @Slot(object)
    def start_session(self, params: SessionParams) -> None:
        """Build and start a `Session`. Runs on the worker thread.

        `Session.start()` is documented as non-blocking (it only spins up
        background threads), so calling it directly here is safe even
        though this slot itself already runs off the UI thread.
        """
        if self._session is not None:
            self.error.emit("Сессия уже запущена.")
            return
        try:
            config = load_app_config()

            # TODO(reconcile-with-backend): attribute paths below assume
            # nested settings objects per kind-hatching-allen.md. Adjust here
            # only if the backend's AppSettings shape differs.
            config.stt.engine = params.engine
            config.stt.language = None if params.language == "auto" else params.language
            config.session.mode = params.mode

            self._session = Session(
                config,
                params.mode,
                mic_id=params.mic_id,
                loopback_id=params.loopback_id,
                import_path=params.import_path,
                on_segment=self._emit_segment,
                on_status=self.status_changed.emit,
                on_backlog=self.backlog_changed.emit,
                on_error=self._emit_error,
                on_finished=self._emit_finished,
            )
            # `Session.__init__` computes `session_dir` synchronously (a
            # timestamped path under the configured output dir) before any
            # capture/recording starts, so the destination is known right
            # here -- well before `on_finished` fires. Surface it to the GUI
            # immediately rather than waiting for session end.
            self.session_dir_ready.emit(self._session.session_dir)
            self._session.start()
        except Exception as exc:  # noqa: BLE001 - surface any failure to the GUI
            self.error.emit(f"Не удалось запустить сессию: {exc}")
            self._session = None

    @Slot()
    def stop_session(self) -> None:
        """Stop the current `Session`. Runs on the worker thread.

        For `mode == "batch"` (and file-import batch pass), `Session.stop()`
        is documented as potentially blocking while it runs the full
        segmentation+transcription pass over the recorded WAV. Because this
        slot only ever executes on the dedicated worker `QThread` (see the
        module docstring), that blocking never reaches the Qt UI thread.
        """
        if self._session is None:
            return
        try:
            self._session.stop()
        except Exception as exc:  # noqa: BLE001
            self.error.emit(f"Ошибка при остановке сессии: {exc}")
        finally:
            self._session = None

    # -- callback adapters --------------------------------------------------
    # Session invokes these from its own background threads; emitting a Qt
    # signal from a foreign thread is safe and is auto-delivered as a queued
    # connection to whatever thread the connected slot lives on.
    def _emit_segment(self, segment: Segment) -> None:
        self.segment_ready.emit(segment)

    def _emit_finished(self, session_dir: Path) -> None:
        # The pipeline reached a natural end on its own (file/batch pass
        # completed) without stop_session() ever being called, so this is
        # the only place that will clear `_session` for that path. Clearing
        # it here (rather than relying solely on `stop_session`) is what
        # lets a second Start be accepted after a file import completes.
        self._session = None
        self.finished.emit(session_dir)

    def _emit_error(self, message: str) -> None:
        # `on_error` is now only used by the backend for non-terminal
        # conditions (e.g. a single segment failing to transcribe) while
        # the session keeps running. It must NOT clear `_session`: doing so
        # would make `stop_session` early-return (orphaning the still-live
        # session/threads/devices) and would let `main_window` re-enable
        # Start with `_session` already None, launching a duplicate Session
        # onto the same devices. Every genuinely terminal condition resets
        # `_session` through its own path instead (`start_session`'s
        # `except`, `_emit_finished`, or `stop_session`'s `finally`).
        self.error.emit(message)
