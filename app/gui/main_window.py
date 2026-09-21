"""PySide6 main window for Live Recorder.

Built strictly against the shared contract in
``kind-hatching-allen.md`` (section "### GUI (`app/gui/`)") and the
`Session`/`Segment` dataclasses in `app/pipeline/`. See module docstring of
`app.gui.worker` for the threading model.

TODO(reconcile-with-backend) markers below call out every place this file
assumes something about a backend module it does not own.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# --- shared contract with the backend --------------------------------------
# TODO(reconcile-with-backend): device enumeration lives in app/audio/devices.py
# and is expected to return List[Tuple[str, Any]] (name, id) pairs for both
# helpers.
#
# NOTE(COM apartment ordering): the audio backend is PyAudioWPatch (a
# PortAudio fork). Importing it does NOT touch COM -- the COM initialization
# (`CoInitialize`, STA) happens when the `PyAudio()` instance is CREATED, and
# that instance is a lazy module singleton in `app.audio.backend`
# (`get_backend()`), created on first real use (device enumeration / capture
# start), never at import time. PortAudio's STA apartment is compatible with
# Qt (the previous backend claimed the main thread's COM apartment as MTA at
# import time and used to lose the apartment race, crashing launch with
# "OleInitialize() failed: RPC_E_CHANGED_MODE"). The lazy-import convention
# is kept anyway: `app.audio.devices` (directly) and
# `app.gui.worker` (transitively, via app.pipeline.session ->
# app.audio.capture) are imported only inside `_populate_devices` and
# `MainWindow.__init__`/`_on_start_clicked` below, so no audio code runs
# before `main()` has constructed the `QApplication`.

# Segment.speaker labels. TODO(reconcile-with-backend): the spec says speaker
# labels come from config ("метки говорящих" in CaptureSettings) and default
# to "Я" / "Собеседники"; a neutral label is used for imported mono files.
# These two are only used here to pick a text color, so an unrecognized
# label (e.g. a customized config label, or the neutral mono-import label)
# simply falls back to the neutral style below rather than erroring.
SPEAKER_ME = "Я"
SPEAKER_OTHERS = "Собеседники"

MODE_ITEMS = [
    ("Live", "live"),
    ("После записи (batch)", "batch"),
    ("Файл", "file"),
]
MODEL_ITEMS = [
    ("GigaAM", "gigaam"),
]

AUDIO_FILE_FILTER = "Audio files (*.wav *.mp3 *.flac *.m4a *.ogg);;All files (*.*)"


def _format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    mm, ss = divmod(total, 60)
    return f"{mm:02d}:{ss:02d}"


class MainWindow(QMainWindow):
    """Main Live Recorder window: device/model/mode selectors, live
    transcript view, a right-hand "Сеанс" (session) panel for browsing past
    sessions and running on-demand summarization, and Start/Stop control --
    all wired to a background `SessionWorker`.
    """

    # Emitted to the worker thread; SessionWorker lives on its own QThread,
    # so connecting these to its slots auto-queues delivery there (see
    # app/gui/worker.py docstring) -- the UI thread is never blocked.
    request_start = Signal(object)  # SessionParams
    request_stop = Signal()
    request_summarize = Signal(object)  # Path to session dir

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Live Recorder")
        self.resize(1150, 700)

        self._session_dir: Optional[Path] = None
        self._recording = False

        # The session currently recording (or just finished/started), as
        # opposed to whatever the user has picked in the session combo --
        # these are the same session right after Start/Stop, but the user
        # is free to browse older sessions in the picker while one is live.
        self._active_session_dir: Optional[Path] = None
        self._active_session_params = None  # SessionParams, set on Start

        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._on_elapsed_tick)
        self._elapsed_start: Optional[datetime] = None

        self._summarizing_dir: Optional[Path] = None
        self._current_backlog = 0

        # `app.config` does not touch the audio backend (unlike app.gui.worker
        # / app.audio.devices), so it's safe to import at module scope -- but
        # kept as a local, best-effort import here in case the backend
        # config module isn't importable yet (mirrors the graceful
        # degradation used elsewhere in this file).
        try:
            from app.config import load_config

            self._recordings_root = Path(load_config().session.output_dir)
        except Exception:  # noqa: BLE001 - degrade gracefully
            self._recordings_root = Path("recordings")

        self._build_ui()
        self._populate_devices()
        self._update_mode_dependent_widgets()
        self._refresh_session_list()

        # Imported here (lazily), not at module load time: this pulls in
        # app.pipeline.session -> app.audio.capture -> app.audio.backend.
        # The audio backend itself (the PyAudio() instance and with it the
        # PortAudio COM init) is created lazily on first use, never at import
        # time -- see the COM-apartment-ordering note near the top of this
        # module. `__init__` only runs after `main()` has already constructed
        # `QApplication`.
        from app.gui.worker import SessionWorker

        self._worker_thread = QThread(self)
        self._worker = SessionWorker()
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.start()

        self.request_start.connect(self._worker.start_session)
        self.request_stop.connect(self._worker.stop_session)
        self.request_summarize.connect(self._worker.summarize)
        self._worker.segment_ready.connect(self._on_segment_ready)
        self._worker.status_changed.connect(self._on_status_changed)
        self._worker.backlog_changed.connect(self._on_backlog_changed)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._on_finished)
        self._worker.session_dir_ready.connect(self._on_session_dir_ready)
        self._worker.summarize_status.connect(self._on_summarize_status)
        self._worker.summarize_done.connect(self._on_summarize_done)
        self._worker.summarize_failed.connect(self._on_summarize_failed)

    # -- UI construction ------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        controls_group = QGroupBox("Настройки сеанса", central)
        grid = QGridLayout(controls_group)

        self.mic_combo = QComboBox(controls_group)
        self.loopback_combo = QComboBox(controls_group)
        self.model_combo = QComboBox(controls_group)
        self.mode_combo = QComboBox(controls_group)
        # Live/batch only: unchecking skips the mic ("Я") capture entirely
        # (silence-padded channel) so a mic that just echoes the system audio
        # played through the speakers can't duplicate the transcription.
        self.mic_check = QCheckBox("Записывать микрофон", controls_group)
        self.mic_check.setChecked(True)

        for label, value in MODEL_ITEMS:
            self.model_combo.addItem(label, value)
        for label, value in MODE_ITEMS:
            self.mode_combo.addItem(label, value)

        grid.addWidget(QLabel("Режим:"), 0, 0)
        grid.addWidget(self.mode_combo, 0, 1)
        grid.addWidget(QLabel("Микрофон:"), 1, 0)
        grid.addWidget(self.mic_combo, 1, 1)
        grid.addWidget(self.mic_check, 2, 1, 1, 2)
        grid.addWidget(QLabel("Системный звук:"), 3, 0)
        grid.addWidget(self.loopback_combo, 3, 1)
        grid.addWidget(QLabel("Модель:"), 4, 0)
        grid.addWidget(self.model_combo, 4, 1)

        self.open_file_button = QPushButton("Открыть аудиофайл…", controls_group)
        self.open_file_button.setEnabled(False)
        grid.addWidget(self.open_file_button, 0, 2, 1, 1)

        # Persistent (non-transient) indicator of the file picked in File
        # mode -- unlike the status-bar message this used to rely on, it
        # doesn't get overwritten by later status updates. Only shown in
        # File mode; see `_update_mode_dependent_widgets`.
        self._import_file_label = QLabel("Файл не выбран", controls_group)
        self._import_file_label.setStyleSheet("color: #5f6368;")
        grid.addWidget(self._import_file_label, 0, 3, 1, 1)

        self.session_name_edit = QLineEdit(controls_group)
        self.session_name_edit.setPlaceholderText("необязательно — дата/время")
        grid.addWidget(QLabel("Название сеанса:"), 5, 0)
        grid.addWidget(self.session_name_edit, 5, 1, 1, 3)

        root.addWidget(controls_group)

        buttons_row = QHBoxLayout()
        self.start_button = QPushButton("Старт", central)
        self.stop_button = QPushButton("Стоп", central)
        self.stop_button.setEnabled(False)
        buttons_row.addWidget(self.start_button)
        buttons_row.addWidget(self.stop_button)
        buttons_row.addStretch(1)
        root.addLayout(buttons_row)

        # Transcript (left, wide) + "Сеанс" panel (right, narrow) side by side.
        body_row = QHBoxLayout()

        self.transcript_view = QTextEdit(central)
        self.transcript_view.setReadOnly(True)
        self.transcript_view.setPlaceholderText("Живой транскрипт появится здесь…")
        body_row.addWidget(self.transcript_view, stretch=4)

        body_row.addWidget(self._build_session_panel(central), stretch=0)

        root.addLayout(body_row, stretch=3)

        log_group = QGroupBox("Журнал", central)
        log_layout = QVBoxLayout(log_group)
        self.log_view = QPlainTextEdit(log_group)
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText(
            "Здесь появятся статус сеанса и путь к папке с записью…"
        )
        # Bound growth over long sessions; oldest lines are dropped first.
        self.log_view.setMaximumBlockCount(2000)
        self.log_view.setFixedHeight(120)
        log_layout.addWidget(self.log_view)
        root.addWidget(log_group, stretch=1)

        self._device_label = QLabel("Устройства: —")
        self._model_state_label = QLabel("Модель: не загружена")
        for lbl in (self._device_label, self._model_state_label):
            self.statusBar().addPermanentWidget(lbl)
        self.statusBar().showMessage("Готово")

        self.mode_combo.currentIndexChanged.connect(self._update_mode_dependent_widgets)
        self.mic_check.toggled.connect(self._update_mode_dependent_widgets)
        self.open_file_button.clicked.connect(self._on_open_file_clicked)
        self.start_button.clicked.connect(self._on_start_clicked)
        self.stop_button.clicked.connect(self._on_stop_clicked)

        self._import_path: Optional[Path] = None

    def _build_session_panel(self, parent: QWidget) -> QGroupBox:
        """The right-hand "Сеанс" group: past-session picker, info, and
        on-demand summarization / housekeeping actions for the SELECTED
        session (which may differ from the currently recording session).
        """
        panel = QGroupBox("Сеанс", parent)
        panel.setFixedWidth(300)
        layout = QVBoxLayout(panel)

        # -- picker ---------------------------------------------------------
        picker_row = QHBoxLayout()
        self.session_combo = QComboBox(panel)
        self.session_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.refresh_sessions_button = QPushButton("Обновить", panel)
        picker_row.addWidget(self.session_combo, stretch=1)
        picker_row.addWidget(self.refresh_sessions_button)
        layout.addLayout(picker_row)

        self.session_state_badge = QLabel("—")
        self.session_state_badge.setStyleSheet("font-weight: bold; padding: 2px 0;")
        layout.addWidget(self.session_state_badge)

        # -- info -------------------------------------------------------
        info_grid = QGridLayout()
        info_grid.setColumnStretch(1, 1)
        self._session_name_label = QLabel("—")
        self._session_path_label = QLabel("—")
        self._session_path_label.setWordWrap(True)
        self._session_duration_label = QLabel("—")
        self._session_counts_label = QLabel("—")
        self._session_speakers_label = QLabel("—")
        self._session_engine_label = QLabel("—")
        self._session_backlog_label = QLabel("0")
        self._session_elapsed_label = QLabel("—")

        rows = [
            ("Название:", self._session_name_label),
            ("Папка:", self._session_path_label),
            ("Длительность:", self._session_duration_label),
            ("Сегментов/слов:", self._session_counts_label),
            ("Реплики:", self._session_speakers_label),
            ("STT:", self._session_engine_label),
            ("Очередь STT:", self._session_backlog_label),
            ("Идёт:", self._session_elapsed_label),
        ]
        for row_idx, (caption, widget) in enumerate(rows):
            info_grid.addWidget(QLabel(caption), row_idx, 0)
            info_grid.addWidget(widget, row_idx, 1)
        layout.addLayout(info_grid)

        # -- summarization progress -----------------------------------------
        self.summarize_progress = QProgressBar(panel)
        self.summarize_progress.setRange(0, 0)  # indeterminate
        self.summarize_progress.setVisible(False)
        layout.addWidget(self.summarize_progress)
        self.summarize_status_label = QLabel("")
        self.summarize_status_label.setWordWrap(True)
        self.summarize_status_label.setVisible(False)
        layout.addWidget(self.summarize_status_label)

        # -- actions ----------------------------------------------------
        self.summarize_button = QPushButton("Саммаризация", panel)
        self.summarize_button.setEnabled(False)
        layout.addWidget(self.summarize_button)

        self.open_report_button = QPushButton("Открыть отчёт", panel)
        self.open_report_button.setEnabled(False)
        layout.addWidget(self.open_report_button)

        self.open_folder_button = QPushButton("Открыть папку сеанса", panel)
        self.open_folder_button.setEnabled(False)
        layout.addWidget(self.open_folder_button)

        self.open_transcript_button = QPushButton("Открыть транскрипт", panel)
        self.open_transcript_button.setEnabled(False)
        layout.addWidget(self.open_transcript_button)

        self.copy_transcript_button = QPushButton("Копировать транскрипт", panel)
        self.copy_transcript_button.setEnabled(False)
        layout.addWidget(self.copy_transcript_button)

        self.delete_session_button = QPushButton("Удалить сеанс", panel)
        self.delete_session_button.setEnabled(False)
        layout.addWidget(self.delete_session_button)

        layout.addStretch(1)

        self.session_combo.currentIndexChanged.connect(self._on_session_selected)
        self.refresh_sessions_button.clicked.connect(self._on_refresh_sessions_clicked)
        self.summarize_button.clicked.connect(self._on_summarize_clicked)
        self.open_report_button.clicked.connect(self._on_open_report_clicked)
        self.open_folder_button.clicked.connect(self._on_open_folder_clicked)
        self.open_transcript_button.clicked.connect(self._on_open_transcript_clicked)
        self.copy_transcript_button.clicked.connect(self._on_copy_transcript_clicked)
        self.delete_session_button.clicked.connect(self._on_delete_session_clicked)

        return panel

    def _populate_devices(self) -> None:
        # TODO(reconcile-with-backend): if app/audio/devices.py isn't built
        # yet, degrade gracefully instead of crashing the GUI at import time.
        #
        # Imported here (lazily), not at module load time: this is what
        # eventually reaches the audio backend (app.audio.devices ->
        # app.audio.backend), and the PyAudio() instance is created on first
        # use -- which must be after `QApplication` exists. This method only
        # runs from `MainWindow.__init__`, which `main()` only calls after
        # constructing `QApplication` -- see the COM-apartment-ordering note
        # near the top of this module.
        try:
            from app.audio.devices import list_loopbacks, list_microphones
        except Exception:  # pragma: no cover - backend module not available yet
            self.mic_combo.addItem("(модуль устройств недоступен)", None)
            self.loopback_combo.addItem("(модуль устройств недоступен)", None)
            return
        try:
            mics = list_microphones()
            loopbacks = list_loopbacks()
        except Exception as exc:  # noqa: BLE001
            self._show_error(f"Не удалось получить список устройств: {exc}")
            mics, loopbacks = [], []

        if not mics:
            self.mic_combo.addItem("(микрофоны не найдены)", None)
        for name, _device_id in mics:
            # Device identity IS the name (PortAudio exports no stable
            # endpoint id): the combo data must be the exact name, so
            # capture resolution matches what `list-devices` prints.
            self.mic_combo.addItem(name, name)

        if not loopbacks:
            self.loopback_combo.addItem("(системный звук не найден)", None)
        for name, _device_id in loopbacks:
            self.loopback_combo.addItem(name, name)

    # -- helpers ----------------------------------------------------------

    def _current_mode(self) -> str:
        return self.mode_combo.currentData()

    def _active_session_is_file_mode(self) -> bool:
        # Display-only check: whether the CURRENTLY ACTIVE session (the one
        # `self._recording` gates, if any) is a File-mode import/transcribe
        # run rather than a live/batch capture. `self._recording` itself
        # keeps its existing "session is active" meaning everywhere (button
        # enable/disable, elapsed timer, delete guard, teardown) -- only the
        # strings rendered from it are mode-aware, per
        # `_update_status_bar`/`_update_session_info` below.
        return (
            self._active_session_params is not None
            and self._active_session_params.mode == "file"
        )

    def _update_mode_dependent_widgets(self) -> None:
        is_file_mode = self._current_mode() == "file"
        self.open_file_button.setEnabled(is_file_mode)
        # The mic selector is usable only in live/batch AND only while the mic
        # channel is actually being recorded.
        self.mic_combo.setEnabled(not is_file_mode and self.mic_check.isChecked())
        self.mic_check.setEnabled(not is_file_mode)
        self.loopback_combo.setEnabled(not is_file_mode)
        # Only relevant in File mode -- hide it in live/batch so it doesn't
        # clutter the controls grid with a label nobody can act on there.
        # The remembered `_import_path` (if any) is kept regardless of mode,
        # so switching back to File mode still shows the previous pick.
        self._import_file_label.setVisible(is_file_mode)
        self._update_import_file_label()

    def _update_import_file_label(self) -> None:
        if self._import_path is None:
            self._import_file_label.setText("Файл не выбран")
            self._import_file_label.setToolTip("")
        else:
            self._import_file_label.setText(self._import_path.name)
            self._import_file_label.setToolTip(str(self._import_path))

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Live Recorder", message)

    def _append_segment_line(self, timestamp: str, speaker: str, text: str) -> None:
        if speaker == SPEAKER_ME:
            color = "#1a73e8"  # blue
            align = "left"
        elif speaker == SPEAKER_OTHERS:
            color = "#188038"  # green
            align = "left"
        else:
            # Neutral label (e.g. imported mono file with no diarization).
            color = "#5f6368"  # grey
            align = "left"

        safe_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html = (
            f'<div style="text-align:{align}; margin:2px 0;">'
            f'<span style="color:#888;">[{timestamp}]</span> '
            f'<b style="color:{color};">{speaker}:</b> '
            f"<span>{safe_text}</span>"
            f"</div>"
        )
        cursor = self.transcript_view.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.transcript_view.setTextCursor(cursor)
        if not self.transcript_view.document().isEmpty():
            cursor.insertBlock()
        cursor.insertHtml(html)
        self.transcript_view.moveCursor(QTextCursor.End)

    def _log(self, message: str) -> None:
        """Append a timestamped, persistent line to the log panel.

        Unlike `statusBar().showMessage(...)`, lines here never auto-clear --
        this is the durable record of status/path information for the
        session (the status bar is still used for quick, transient hints).
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_view.appendPlainText(f"[{timestamp}] {message}")

    def _update_status_bar(self) -> None:
        mic_name = self.mic_combo.currentText()
        loop_name = self.loopback_combo.currentText()
        if not self._recording:
            state = "остановлено"
        elif self._active_session_is_file_mode():
            # File mode never records audio -- an existing file is being
            # transcribed -- so "запись" (recording) would be misleading.
            state = "обработка файла"
        else:
            state = "запись"
        self._device_label.setText(f"Устройства: {mic_name} / {loop_name} ({state})")

    # -- session panel helpers ------------------------------------------

    def _selected_session_dir(self) -> Optional[Path]:
        data = self.session_combo.currentData()
        return data if isinstance(data, Path) else None

    def _refresh_session_list(self, select: Optional[Path] = None) -> None:
        """Repopulate the session picker from `recordings/` (newest first).

        Preserves/sets the selection to `select` if given (falls back to the
        newest session), without emitting spurious intermediate
        `currentIndexChanged` signals while the combo is being rebuilt.
        """
        dirs: list[Path] = []
        if self._recordings_root.exists():
            try:
                # Dot-prefixed dirs are app-internal helpers (e.g. the STT
                # engine's .stt_tmp temp-WAV dir) — never sessions.
                dirs = [
                    d
                    for d in self._recordings_root.iterdir()
                    if d.is_dir() and not d.name.startswith(".")
                ]
            except OSError:
                dirs = []
        dirs.sort(key=lambda d: d.name, reverse=True)

        self.session_combo.blockSignals(True)
        self.session_combo.clear()
        for d in dirs:
            self.session_combo.addItem(d.name, d)
        target_index = 0
        if select is not None:
            for i in range(self.session_combo.count()):
                if self.session_combo.itemData(i) == select:
                    target_index = i
                    break
        if self.session_combo.count() > 0:
            self.session_combo.setCurrentIndex(target_index)
        self.session_combo.blockSignals(False)

        # setCurrentIndex above never emitted (signals were blocked), and it
        # may be a no-op even when unblocked (index unchanged) -- so always
        # refresh info/buttons for whatever ended up selected, explicitly.
        self._on_session_selected()

    def _load_transcript_segments(self, session_dir: Path) -> Optional[list[dict]]:
        """Best-effort read of transcript.json; None on missing/corrupt file."""
        path = session_dir / "transcript.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, list):
            return None
        # Keep only well-shaped records: a structurally-valid JSON file of the
        # wrong shape (e.g. `[1, 2, 3]`) must degrade to "—" like a corrupt one,
        # not raise out of the currentIndexChanged slot that calls us.
        return [s for s in data if isinstance(s, dict)]

    def _update_session_info(self) -> None:
        session_dir = self._selected_session_dir()
        if session_dir is None:
            self._session_name_label.setText("—")
            self._session_path_label.setText("—")
            self._session_duration_label.setText("—")
            self._session_counts_label.setText("—")
            self._session_speakers_label.setText("—")
            self._session_engine_label.setText("—")
            self._session_backlog_label.setText("—")
            self._session_elapsed_label.setText("—")
            self.session_state_badge.setText("—")
            return

        is_active = (
            self._active_session_dir is not None
            and session_dir == self._active_session_dir
        )

        self._session_name_label.setText(session_dir.name)
        self._session_path_label.setText(str(session_dir))

        segments = self._load_transcript_segments(session_dir)
        if segments is None:
            self._session_duration_label.setText("—")
            self._session_counts_label.setText("—")
            self._session_speakers_label.setText("—")
        else:

            def _end(s: dict) -> float:
                try:
                    return float(s.get("end", 0.0))
                except (TypeError, ValueError):
                    return 0.0

            duration = max((_end(s) for s in segments), default=0.0)
            self._session_duration_label.setText(_format_timestamp(duration))
            word_count = sum(len(str(s.get("text", "")).split()) for s in segments)
            self._session_counts_label.setText(f"{len(segments)} / {word_count}")
            me = sum(1 for s in segments if s.get("speaker") == SPEAKER_ME)
            others = sum(1 for s in segments if s.get("speaker") == SPEAKER_OTHERS)
            self._session_speakers_label.setText(
                f"{SPEAKER_ME}: {me} / {SPEAKER_OTHERS}: {others}"
            )

        if is_active and self._active_session_params is not None:
            p = self._active_session_params
            self._session_engine_label.setText(p.engine)
        else:
            self._session_engine_label.setText("—")

        if is_active:
            self._session_backlog_label.setText(str(self._current_backlog))
        else:
            self._session_backlog_label.setText("—")

        if is_active and self._recording and self._elapsed_start is not None:
            elapsed = (datetime.now() - self._elapsed_start).total_seconds()
            self._session_elapsed_label.setText(_format_timestamp(elapsed))
        else:
            self._session_elapsed_label.setText("—")

        if self._summarizing_dir is not None and session_dir == self._summarizing_dir:
            badge = "Саммаризация…"
        elif is_active and self._recording and self._active_session_is_file_mode():
            # File mode: an existing file is being transcribed, nothing is
            # being recorded -- see `_active_session_is_file_mode`.
            badge = "Обработка файла"
        elif is_active and self._recording:
            badge = "Идёт запись"
        else:
            badge = "Готово"
        self.session_state_badge.setText(badge)

    def _update_action_buttons(self) -> None:
        session_dir = self._selected_session_dir()
        is_active_recording = (
            self._recording
            and self._active_session_dir is not None
            and session_dir == self._active_session_dir
        )
        any_summarizing = self._summarizing_dir is not None
        has_transcript = (
            session_dir is not None and (session_dir / "transcript.json").exists()
        )
        has_report = session_dir is not None and (session_dir / "summary.docx").exists()
        has_transcript_txt = (
            session_dir is not None and (session_dir / "transcript.txt").exists()
        )

        self.summarize_button.setEnabled(
            has_transcript and not is_active_recording and not any_summarizing
        )
        self.open_report_button.setEnabled(has_report)
        self.open_folder_button.setEnabled(
            session_dir is not None and session_dir.exists()
        )
        self.open_transcript_button.setEnabled(has_transcript_txt)
        self.copy_transcript_button.setEnabled(has_transcript_txt)
        is_being_summarized = (
            session_dir is not None and session_dir == self._summarizing_dir
        )
        self.delete_session_button.setEnabled(
            session_dir is not None
            and not is_active_recording
            and not is_being_summarized
        )

    # -- slots: UI actions --------------------------------------------------

    def _on_open_file_clicked(self) -> None:
        # DontUseNativeDialog: the native Win32 file-open dialog is a known
        # freeze culprit here -- it enumerates shell namespace extensions
        # (OneDrive/cloud-sync overlay icons, network drives in Quick
        # access/Recent) synchronously on the UI thread while showing, which
        # can hang for many seconds or indefinitely depending on what's
        # mounted. Qt's own dialog avoids that shell integration entirely.
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Открыть аудиофайл",
            "",
            AUDIO_FILE_FILTER,
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if path_str:
            self._import_path = Path(path_str)
            self._update_import_file_label()
            self.statusBar().showMessage(f"Выбран файл: {self._import_path.name}")

    def _on_start_clicked(self) -> None:
        # Local import: by this point `app.gui.worker` is already cached in
        # `sys.modules` (imported in `__init__`), so this is just a cheap
        # name lookup, not a re-import. Kept local for symmetry/clarity with
        # the COM-apartment-ordering constraint documented above.
        from app.gui.worker import SessionParams

        mode = self._current_mode()
        if mode == "file" and self._import_path is None:
            self._show_error("Сначала выберите аудиофайл («Открыть аудиофайл…»).")
            return

        # Optional user-chosen session name; blank/whitespace-only means "use
        # the default timestamp naming" -- normalized to None here so the
        # backend doesn't have to special-case an empty string vs. missing
        # value (the `session_name` field on `SessionParams` defaults to None).
        session_name = self.session_name_edit.text().strip() or None

        params = SessionParams(
            mode=mode,
            mic_id=self.mic_combo.currentData(),
            loopback_id=self.loopback_combo.currentData(),
            import_path=self._import_path if mode == "file" else None,
            engine=self.model_combo.currentData(),
            session_name=session_name,
            record_mic=self.mic_check.isChecked(),
        )

        self.transcript_view.clear()
        self.log_view.clear()
        self._session_dir = None
        self._active_session_dir = None
        self._active_session_params = params
        self._current_backlog = 0
        self._elapsed_start = datetime.now()
        self._elapsed_timer.start()
        self._recording = True
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._model_state_label.setText("Модель: загрузка…")
        self._update_status_bar()
        self._update_action_buttons()
        self._log("Обработка файла запущена…" if mode == "file" else "Запуск сеанса…")

        self.request_start.emit(params)

    def _on_stop_clicked(self) -> None:
        self.stop_button.setEnabled(False)
        self.statusBar().showMessage("Остановка…")
        self.request_stop.emit()

    def _on_open_folder_clicked(self) -> None:
        session_dir = self._selected_session_dir()
        if session_dir is None or not session_dir.exists():
            self.statusBar().showMessage("Папка сеанса не найдена.", 3000)
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(session_dir)))

    def _on_open_report_clicked(self) -> None:
        session_dir = self._selected_session_dir()
        if session_dir is None:
            return
        report = session_dir / "summary.docx"
        if not report.exists():
            self.statusBar().showMessage("Отчёт ещё не создан.", 3000)
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(report)))

    def _on_open_transcript_clicked(self) -> None:
        session_dir = self._selected_session_dir()
        if session_dir is None:
            return
        txt_path = session_dir / "transcript.txt"
        if not txt_path.exists():
            self.statusBar().showMessage("Транскрипт не найден.", 3000)
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(txt_path)))

    def _on_copy_transcript_clicked(self) -> None:
        session_dir = self._selected_session_dir()
        if session_dir is None:
            return
        txt_path = session_dir / "transcript.txt"
        if not txt_path.exists():
            self.statusBar().showMessage("Транскрипт не найден.", 3000)
            return
        try:
            content = txt_path.read_text(encoding="utf-8")
        except OSError as exc:
            self._show_error(f"Не удалось прочитать транскрипт: {exc}")
            return
        QApplication.clipboard().setText(content)
        self.statusBar().showMessage("Транскрипт скопирован в буфер обмена.", 3000)

    def _on_delete_session_clicked(self) -> None:
        session_dir = self._selected_session_dir()
        if session_dir is None:
            return
        if self._recording and self._active_session_dir == session_dir:
            # Also enforced by the button's enabled state; defense in depth.
            verb = (
                "обработка файла" if self._active_session_is_file_mode() else "запись"
            )
            self._show_error(f"Нельзя удалить сеанс, пока идёт {verb}.")
            return
        if session_dir == self._summarizing_dir:
            # rmtree would race the summarize daemon writing summary.docx here.
            self._show_error("Нельзя удалить сеанс, пока идёт саммаризация.")
            return
        reply = QMessageBox.question(
            self,
            "Удалить сеанс",
            f"Удалить сеанс «{session_dir.name}» без возможности восстановления?\n\n{session_dir}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            shutil.rmtree(session_dir)
        except OSError as exc:
            self._show_error(f"Не удалось удалить папку сеанса: {exc}")
            return
        self._log(f"Сеанс удалён: {session_dir}")
        if self._session_dir == session_dir:
            self._session_dir = None
        self._refresh_session_list()

    def _on_refresh_sessions_clicked(self) -> None:
        self._refresh_session_list(select=self._selected_session_dir())

    def _on_session_selected(self, _index: int = -1) -> None:
        self._update_session_info()
        self._update_action_buttons()

    def _on_summarize_clicked(self) -> None:
        session_dir = self._selected_session_dir()
        if session_dir is None:
            return
        self._summarizing_dir = session_dir
        self.summarize_progress.setVisible(True)
        self.summarize_status_label.setVisible(True)
        self.summarize_status_label.setText("Запуск саммаризации…")
        self._update_action_buttons()
        self._update_session_info()
        self._log(f"Саммаризация запущена: {session_dir}")
        self.request_summarize.emit(session_dir)

    def _on_elapsed_tick(self) -> None:
        if self._elapsed_start is None or self._active_session_dir is None:
            return
        if self._selected_session_dir() != self._active_session_dir:
            return
        elapsed = (datetime.now() - self._elapsed_start).total_seconds()
        self._session_elapsed_label.setText(_format_timestamp(elapsed))

    # -- slots: worker signals -----------------------------------------------

    def _on_segment_ready(self, segment) -> None:
        # `segment` is a `app.pipeline.transcript.Segment`; kept untyped here
        # (duck-typed access) so the GUI module doesn't hard-fail if the
        # backend's Segment shape drifts slightly before final reconciliation.
        timestamp = _format_timestamp(getattr(segment, "start", 0.0))
        speaker = getattr(segment, "speaker", "")
        text = getattr(segment, "text", "")
        self._append_segment_line(timestamp, speaker, text)

    def _on_session_dir_ready(self, session_dir) -> None:
        # Fired by SessionWorker right after `Session` is constructed --
        # before capture/recording actually starts -- so the destination is
        # known up front rather than only once the session finishes.
        self._session_dir = Path(session_dir) if session_dir is not None else None
        self._active_session_dir = self._session_dir
        if self._session_dir is not None:
            self._log(f"Папка сеанса: {self._session_dir}")
            # Bring the just-started session into the picker and select it
            # so its (still mostly empty) info/state is visible immediately.
            self._refresh_session_list(select=self._session_dir)

    def _on_status_changed(self, status: str) -> None:
        self._model_state_label.setText(f"Статус: {status}")
        self.statusBar().showMessage(status, 5000)
        self._log(status)

    def _on_backlog_changed(self, depth: int) -> None:
        self._current_backlog = depth
        if (
            self._active_session_dir is not None
            and self._selected_session_dir() == self._active_session_dir
        ):
            self._session_backlog_label.setText(str(depth))

    def _on_error(self, message: str) -> None:
        self._recording = False
        self._elapsed_timer.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._update_status_bar()
        self._update_action_buttons()
        self._log(f"Ошибка: {message}")
        self._show_error(message)

    def _on_finished(self, session_dir) -> None:
        if session_dir is not None:
            self._session_dir = Path(session_dir)
            self._active_session_dir = self._session_dir
        self._recording = False
        self._elapsed_timer.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._model_state_label.setText("Модель: сеанс завершён")
        self._update_status_bar()
        if self._session_dir is not None:
            self.statusBar().showMessage(f"Сеанс сохранён: {self._session_dir}", 8000)
            self._log(f"Сеанс сохранён: {self._session_dir}")
        # Refresh the picker (a brand-new dir may not have been listed yet)
        # and auto-select the just-finished session so its info/actions --
        # notably "Саммаризация", now that transcript.json exists -- show up
        # immediately without the user hunting for it in the list.
        self._refresh_session_list(select=self._session_dir)

    def _on_summarize_status(self, status: str) -> None:
        self.summarize_status_label.setText(status)
        self._log(status)

    def _on_summarize_done(self, path) -> None:
        self._log(f"Саммаризация завершена: {path}")
        self._summarizing_dir = None
        self.summarize_progress.setVisible(False)
        self.summarize_status_label.setVisible(False)
        self._update_action_buttons()
        self._update_session_info()

    def _on_summarize_failed(self, message: str) -> None:
        self._log(f"Ошибка саммаризации: {message}")
        self._summarizing_dir = None
        self.summarize_progress.setVisible(False)
        self.summarize_status_label.setVisible(False)
        self._update_action_buttons()
        self._update_session_info()
        # Non-terminal: the transcript/recording is intact, only the
        # on-demand summarization attempt failed. Warn, don't crash.
        QMessageBox.warning(
            self, "Саммаризация", f"Не удалось выполнить саммаризацию:\n\n{message}"
        )

    # -- lifecycle ------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        # Route through the same queued request_stop signal used everywhere
        # else, so `Session.stop()` (which can block for a while running the
        # batch/file pass) executes on the worker QThread, not here on the
        # UI thread. Never call a `Session`/`SessionWorker` method directly
        # from the UI thread.
        try:
            if self._recording:
                self.request_stop.emit()
        finally:
            self._worker_thread.quit()
            self._worker_thread.wait(3000)
        super().closeEvent(event)


def main() -> None:
    """Entrypoint called by `app/__main__.py` (owned by the backend engineer)
    via ``from app.gui.main_window import main; main()``.
    """
    existing_app = QApplication.instance()
    app = existing_app or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    if existing_app is None:
        sys.exit(app.exec())


if __name__ == "__main__":
    main()
