"""Session orchestration: capture -> VAD -> STT, for modes live / batch / file.

SHARED INTERFACE CONTRACT (the GUI is built against this — keep it stable):

    class Session:
        def __init__(self, config, mode: str, *,
                     mic_id=None, loopback_id=None, import_path=None,
                     on_segment=None,      # Callable[[Segment], None]
                     on_status=None,       # Callable[[str], None]
                     on_backlog=None,      # Callable[[int], None]  STT queue depth
                     on_error=None,        # Callable[[str], None]
                     on_finished=None):    # Callable[[Path], None] session dir
            ...
        def start(self) -> None: ...   # non-blocking; spins up threads
        def stop(self) -> None: ...    # signals stop; for batch runs the pass;
                                       # joins; writes transcript

IMPORTANT — threading: every callback (on_segment, on_status, on_backlog,
on_error, on_finished) may be invoked from a background thread. The GUI worker
must marshal these onto the UI thread (e.g. via Qt signals). The backend never
imports Qt.

Modes:
  - "live":  capture mic + loopback -> AlignedRecorder streams stereo session.wav
             AND forwards aligned per-channel blocks to two ContinuousSegmenters
             (each in its own thread); finalized segments go to a single FIFO
             transcription queue, transcribed by one worker as they are ready.
  - "batch": during the session only session.wav is streamed to disk. On stop(),
             both tracks are re-read from disk in blocks (wav_source, channel
             0/1) through two VADs -> full "Я"/"Собеседники" attribution.
  - "file":  import a file. ALWAYS transcribed as a single speaker (the neutral
             label, config default "Speaker") regardless of channel count — no
             stereo split, no channel attribution, no correlation detection.
             Multi-channel files are downmixed to mono. Resampled to the target
             rate on import if needed. (Dual "Я"/"Собеседники" attribution is
             reserved for genuine live-recorded session.wav in batch mode.)

batch and file share `_run_disk_pass`.

Design notes (from spec):
  - Single FIFO transcription queue ordered by segment ready-time; ONE worker
    guarded by a single Lock (inference is not reentrant).
  - The queue NEVER drops segments (correctness over latency). Backlog depth is
    reported via on_backlog so the GUI can show it.
  - Disk is the single source of truth; the whole session is never held in RAM.
"""

from __future__ import annotations

import itertools
import queue
import re
import threading
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.audio import wav_source
from app.audio.capture import DEFAULT_NATIVE_RATE, CaptureThread
from app.audio.recorder import AlignedRecorder
from app.config import Config
from app.log import get_logger
from app.pipeline.transcript import Segment, export_all
from app.stt.factory import build_engine
from app.vad.segmenter import ContinuousSegmenter, RawSegment, load_silero_vad

logger = get_logger("session")

# Block size (samples) for streaming disk passes (batch/file). ~0.25 s at 16 kHz.
_DISK_BLOCKSIZE = 4096

# Bounded wait for capture/feeder/segmenter threads to exit on stop(). A wedged
# soundcard.record() (e.g. loopback blocking on system silence) can otherwise
# hang teardown indefinitely; we log and abandon the thread instead.
_JOIN_TIMEOUT = 5.0

_SENTINEL = object()

# Max length (chars) of the sanitized user name portion of a session folder.
# The full folder name is "<timestamp>_<sanitized>", so the timestamp prefix
# (15 chars) plus this cap keeps the segment well under Windows' per-component
# limit (255) and leaves plenty of headroom for the parent output_dir path.
_MAX_NAME_LEN = 80

# Chars illegal in a Windows path SEGMENT. Besides the path separators / and \,
# Windows reserves < > : " | ? *. We strip these rather than substitute so a
# name like "a/b" collapses cleanly instead of leaving stray placeholder glyphs.
# Control/format chars (ASCII controls, DEL, C1, zero-width/bidi) are handled
# separately below via a unicodedata category filter -- see
# _sanitize_session_name -- so they are intentionally NOT listed here. Unicode
# letters/digits (incl. Cyrillic) are deliberately NOT touched -- users name
# sessions in Russian.
_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*]')

# Runs of any whitespace (spaces, tabs, newlines) collapse to a single "_" so
# the folder name has no internal spaces -- easier to type in a shell / less
# ambiguous in logs. (Chosen over collapsing to a single space; see report.)
_WHITESPACE_RUN = re.compile(r"\s+")

# Windows reserved device names (case-insensitive, with or without extension).
# A folder literally named e.g. "CON" or "nul" is unusable, so we suffix a
# sanitized result that matches one of these.
_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def _sanitize_session_name(raw: Optional[str]) -> Optional[str]:
    """Turn raw user text into a safe Windows folder-name segment.

    Returns the sanitized segment, or None when the input is empty/None/
    whitespace-only OR when sanitization reduces it to nothing -- in which case
    the caller falls back to the timestamp-only folder name (today's behavior).

    Preserves Unicode letters/digits (Cyrillic included); only strips/collapses
    characters that are unsafe or awkward in a Windows path component.
    """
    if raw is None:
        return None
    # Drop chars illegal in a Windows path segment.
    cleaned = _UNSAFE_CHARS.sub("", raw)
    # Drop invisible/dangerous control and format characters the regex above
    # does not cover: ASCII C0 controls and DEL (0x7F), the C1 controls
    # (0x80-0x9F), and Unicode format chars such as the zero-width space
    # (U+200B) and bidi overrides (e.g. U+202E RIGHT-TO-LEFT OVERRIDE, which can
    # spoof how a folder name renders). unicodedata categories Cc (control) and
    # Cf (format) capture ALL of these while leaving Cyrillic and other real
    # letters/digits (categories L*/N*) untouched.
    cleaned = "".join(
        ch for ch in cleaned if unicodedata.category(ch) not in ("Cc", "Cf")
    )
    # Collapse whitespace runs to single underscores.
    cleaned = _WHITESPACE_RUN.sub("_", cleaned)
    # Windows silently trims trailing dots/spaces from path segments, which can
    # turn a name into something unexpected (or empty); strip them from both
    # ends. Also strip underscores we may have introduced at the edges.
    cleaned = cleaned.strip(" ._")
    if not cleaned:
        return None
    # Cap length before the reserved-name check so a truncation can't re-create
    # a reserved name at the boundary.
    cleaned = cleaned[:_MAX_NAME_LEN].strip(" ._")
    if not cleaned:
        return None
    # A folder named after a DOS device is unusable. Windows matches the reserved
    # set against the STEM -- the portion before the FIRST dot -- so "CON.txt" is
    # just as reserved as "CON". Suffix the underscore onto the stem (not the
    # whole string) so the mangled name's stem no longer matches: "CON.txt" ->
    # "CON_.txt", "CON" -> "CON_". Splitting on the first "." preserves any
    # remainder verbatim.
    stem, dot, remainder = cleaned.partition(".")
    if stem.upper() in _RESERVED_NAMES:
        cleaned = f"{stem}_{dot}{remainder}"
    return cleaned


class Session:
    def __init__(
        self,
        config: Config,
        mode: str,
        *,
        mic_id=None,
        loopback_id=None,
        import_path=None,
        name: Optional[str] = None,
        on_segment=None,
        on_status=None,
        on_backlog=None,
        on_error=None,
        on_finished=None,
    ) -> None:
        if mode not in ("live", "batch", "file"):
            raise ValueError(f"Unknown mode {mode!r} (live|batch|file)")
        self.config = config
        self.mode = mode
        self.mic_id = mic_id
        self.loopback_id = loopback_id
        self.import_path = Path(import_path) if import_path else None

        self.on_segment = on_segment
        self.on_status = on_status
        self.on_backlog = on_backlog
        self.on_error = on_error
        self.on_finished = on_finished

        self.rate = config.capture.target_sample_rate
        self.frame_size = config.capture.frame_size
        self.mic_label = config.capture.mic_label
        self.loop_label = config.capture.loopback_label
        self.neutral_label = config.capture.neutral_label

        # Session directory (created at start). session.wav lives here for
        # live/batch; for file the import stays where it is.
        #
        # Folder name: a timestamp, optionally suffixed with a sanitized user
        # name. With no (usable) name we keep the historical timestamp-only
        # name verbatim. With a name we prefix the timestamp so sessions stay
        # sortable AND collision-safe across seconds -- the picker sorts folder
        # names descending, so timestamp-first keeps chronological order
        # regardless of the name. (Two sessions started within the same 1-second
        # timestamp with the same/no name can still collide; this is pre-existing
        # behavior and out of scope here.)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = _sanitize_session_name(name)
        dir_name = f"{timestamp}_{safe_name}" if safe_name else timestamp
        self.session_dir = Path(config.session.output_dir) / dir_name
        self.wav_path = self.session_dir / "session.wav"
        # Per-channel mono outputs (live/batch). Derived in lockstep from the
        # same aligned blocks as session.wav, so they stay sample-for-sample
        # aligned with it; session.wav remains the single source of truth for the
        # disk re-read and VAD timeline. mic -> "Я", loopback -> "Собеседники".
        self.mic_wav_path = self.session_dir / "mic.wav"
        self.loopback_wav_path = self.session_dir / "loopback.wav"

        self._engine = None
        self._tqueue: "queue.Queue" = queue.Queue()
        self._infer_lock = threading.Lock()  # inference is not reentrant
        self._lock = threading.Lock()  # guards _segments
        self._segments: list[Segment] = []
        # Defense-in-depth against a systemic all-segments-fail transcription
        # bug (e.g. the torchcodec class of failure). Both are touched only on the
        # single STT worker thread, and read at end-of-pass AFTER _stop_worker()
        # joins that thread, so no lock is needed. attempted = segments sent to
        # the engine; failed = segments whose transcribe() raised.
        self._attempted = 0
        self._failed = 0
        self._cancel = threading.Event()
        self._finished = False
        self._finish_lock = threading.Lock()
        # Ensures a mid-run recorder-writer failure drives teardown only once.
        self._teardown_started = threading.Event()

        # Live/batch capture machinery.
        self._recorder: AlignedRecorder | None = None
        self._captures: list[CaptureThread] = []
        self._cap_queues: list[queue.Queue] = []
        self._feeders: list[threading.Thread] = []
        self._seg_queues: list[queue.Queue] = []
        self._seg_threads: list[threading.Thread] = []
        self._worker_thread: threading.Thread | None = None
        # file-mode processing thread.
        self._disk_thread: threading.Thread | None = None

    # --- public API --------------------------------------------------------
    def start(self) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        if self.mode == "live":
            self._start_live()
        elif self.mode == "batch":
            self._start_recording_only()
        else:  # file
            self._start_file()

    def stop(self) -> None:
        # _cancel is set PER-MODE, not unconditionally. Its only job is to
        # abort a running file-mode disk pass so the join() below returns
        # promptly. In batch mode stop() is what TRIGGERS the disk pass, so
        # setting _cancel here would make _run_disk_pass break on its first
        # block and transcribe nothing. Live never checks _cancel.
        if self.mode == "live":
            self._stop_live()
        elif self.mode == "batch":
            self._stop_batch()
        else:  # file
            self._cancel.set()
            if self._disk_thread is not None:
                self._disk_thread.join()

    # --- callback helpers --------------------------------------------------
    def _set_status(self, text: str) -> None:
        logger.info("%s", text)
        if self.on_status:
            self.on_status(text)

    def _emit_error(self, text: str) -> None:
        logger.error("%s", text)
        if self.on_error:
            self.on_error(text)

    def _report_backlog(self) -> None:
        if self.on_backlog:
            self.on_backlog(self._tqueue.qsize())

    def _emit_systemic_failure_if_all_failed(self) -> None:
        """Surface an all-attempted-all-failed transcription pass as terminal.

        A SINGLE (or partial) segment failure stays non-terminal — it goes to
        on_status ("Пропущен сегмент"), never on_error — per the MEMORY contract.
        But the torchcodec bug was EVERY segment failing under that same swallow,
        which yields an empty transcript and NO on_error: the exact "recorded but
        nothing transcribed, no error" symptom. This is defense-in-depth for a
        future systemic failure: when every attempted segment failed, report it
        via on_error.

        Must be called at end-of-pass, AFTER _stop_worker() has joined the worker
        (so the counts are final and race-free), and BEFORE _finish() so the
        terminal error precedes on_finished. on_error is non-resetting for the GUI
        worker; on_finished (via _finish) is what resets _session — so ordering
        error-then-finished keeps the worker's reset paths correct and does not
        double-report.

        Guarded by attempted > 0 so a zero-speech recording (VAD found nothing ->
        nothing enqueued -> nothing attempted) is a legitimately empty session and
        does NOT false-positive. failed == attempted excludes normal partial
        failures.
        """
        if self._attempted > 0 and self._failed == self._attempted:
            self._emit_error(
                "Не удалось расшифровать ни один из "
                f"{self._attempted} сегментов — запись сохранена, но "
                "транскрипция пуста (системный сбой распознавания)."
            )

    def _on_recorder_failure(self, text: str) -> None:
        """Terminal: the recorder writer thread died mid-run.

        Surface the failure via on_error (the session is genuinely ending), then
        drive a full teardown from a SAFE context. This handler runs on the
        recorder's own writer thread, so it must NOT run the blocking stop()
        sequence inline — stop() joins that very writer thread (self-join
        deadlock). Instead spawn a short-lived daemon that runs the normal
        stop/teardown path: captures/feeders/segmenters/STT-worker stopped via
        the existing bounded joins, session.wav closed by _recorder.stop(),
        transcript written, and on_finished fired exactly once (guarded by the
        idempotent _finish). The guard ensures teardown is initiated only once.
        """
        self._emit_error(text)
        if self._teardown_started.is_set():
            return
        self._teardown_started.set()
        threading.Thread(
            target=self.stop, name="recorder-failure-teardown", daemon=True
        ).start()

    def _finish(self) -> None:
        with self._finish_lock:
            if self._finished:
                return
            self._finished = True
        with self._lock:
            segments = sorted(self._segments, key=lambda s: s.start)
        export_all(segments, self.session_dir)
        self._set_status(f"Готово: {self.session_dir}")
        if self.on_finished:
            self.on_finished(self.session_dir)

    # --- transcription queue / worker --------------------------------------
    def _enqueue(self, raw: RawSegment, speaker: str) -> None:
        self._tqueue.put((raw, speaker))
        self._report_backlog()

    def _worker(self) -> None:
        while True:
            item = self._tqueue.get()
            if item is _SENTINEL:
                self._tqueue.task_done()
                break
            raw, speaker = item
            self._attempted += 1
            try:
                with self._infer_lock:
                    text = self._engine.transcribe(raw.audio)
            except Exception as exc:  # keep the queue moving (non-terminal)
                # A single segment failing to transcribe is recoverable: the STT
                # worker keeps running. Report via the non-terminal status channel
                # (NOT on_error, which now strictly means the session is ending).
                # We still COUNT the failure so an all-attempted-all-failed pass
                # can be surfaced as terminal at session end (see
                # _emit_systemic_failure_if_all_failed).
                self._failed += 1
                logger.warning("Пропущен сегмент (ошибка транскрипции): %s", exc)
                self._set_status(f"Пропущен сегмент (ошибка транскрипции): {exc}")
                self._tqueue.task_done()
                self._report_backlog()
                continue
            text = (text or "").strip()
            if text:
                seg = Segment(
                    start=raw.start / self.rate,
                    end=raw.end / self.rate,
                    speaker=speaker,
                    text=text,
                )
                with self._lock:
                    self._segments.append(seg)
                if self.on_segment:
                    self.on_segment(seg)
            self._tqueue.task_done()
            self._report_backlog()

    def _start_worker(self) -> None:
        self._worker_thread = threading.Thread(
            target=self._worker, name="stt-worker", daemon=True
        )
        self._worker_thread.start()

    def _stop_worker(self) -> None:
        self._tqueue.put(_SENTINEL)
        if self._worker_thread is not None:
            self._worker_thread.join()
            self._worker_thread = None

    # --- segmenters --------------------------------------------------------
    def _new_segmenter(self) -> ContinuousSegmenter:
        return ContinuousSegmenter(
            load_silero_vad(),
            sample_rate=self.rate,
            threshold=self.config.vad.threshold,
            silence_timeout=self.config.vad.silence_timeout,
            max_duration=self.config.vad.max_duration,
            frame_size=self.frame_size,
        )

    # --- capture / recorder plumbing (live + batch) ------------------------
    def _feeder_loop(self, cap_q: queue.Queue, submit) -> None:
        while True:
            frame = cap_q.get()
            if frame is None:  # capture-thread sentinel
                break
            submit(frame)

    def _segmenter_loop(
        self, seg_q: queue.Queue, segmenter: ContinuousSegmenter, speaker: str
    ) -> None:
        while True:
            block = seg_q.get()
            if block is None:
                for seg in segmenter.flush():
                    self._enqueue(seg, speaker)
                break
            for seg in segmenter.feed(block):
                self._enqueue(seg, speaker)

    def _build_recorder(self, forward: bool) -> None:
        """Create AlignedRecorder. If forward, aligned blocks feed segmenters."""
        on_left = on_right = None
        if forward:
            self._seg_queues = [queue.Queue(), queue.Queue()]
            on_left = self._seg_queues[0].put
            on_right = self._seg_queues[1].put
        self._recorder = AlignedRecorder(
            self.wav_path,
            self.rate,
            left_path=self.mic_wav_path,
            right_path=self.loopback_wav_path,
            on_left=on_left,
            on_right=on_right,
            on_error=self._on_recorder_failure,
        )

    def _start_captures(self) -> None:
        self._cap_queues = [queue.Queue(), queue.Queue()]
        submits = [self._recorder.submit_left, self._recorder.submit_right]
        device_ids = [self.mic_id, self.loopback_id]
        names = ["capture-mic", "capture-loop"]
        for dev_id, cap_q, submit, name in zip(
            device_ids, self._cap_queues, submits, names
        ):
            ct = CaptureThread(
                dev_id,
                cap_q,
                frame_size=self.frame_size,
                target_sample_rate=self.rate,
                native_sample_rate=DEFAULT_NATIVE_RATE,
                chunk_frames=self.config.capture.chunk_frames,
                name=name,
            )
            self._captures.append(ct)
            feeder = threading.Thread(
                target=self._feeder_loop,
                args=(cap_q, submit),
                name=f"{name}-feeder",
                daemon=True,
            )
            self._feeders.append(feeder)

    def _validate_capture_devices(self) -> None:
        """Fail fast (before opening anything) if a device id is missing.

        A null device id otherwise reaches the capture thread and produces a
        silent, empty session; raising here lets start() propagate a clear error
        to the GUI worker so it can reset.
        """
        if self.mic_id is None:
            raise ValueError("Не выбрано устройство микрофона (mic_id)")
        if self.loopback_id is None:
            raise ValueError(
                "Не выбрано устройство системного звука (loopback_id)"
            )

    def _join_captures_and_feeders(self) -> None:
        """Join capture + feeder threads with a bounded timeout.

        A wedged soundcard.record() never returns, so its capture thread also
        never emits the sentinel that unblocks its feeder. We wait a bounded
        time, then log and abandon rather than hang stop()/app close.
        """
        for ct in self._captures:
            ct.join(timeout=_JOIN_TIMEOUT)
            if ct.is_alive():
                logger.warning(
                    "Capture thread %s did not exit within %.1fs; abandoning it "
                    "(device may be wedged in record())",
                    ct.name,
                    _JOIN_TIMEOUT,
                )
        for f in self._feeders:
            f.join(timeout=_JOIN_TIMEOUT)  # drain remaining frames into recorder
            if f.is_alive():
                logger.warning(
                    "Feeder thread %s did not exit within %.1fs; abandoning it",
                    f.name,
                    _JOIN_TIMEOUT,
                )

    def _surface_capture_errors(self) -> bool:
        """Report any capture-thread exception via on_error. Returns True if any."""
        had_error = False
        for ct in self._captures:
            if ct.error is not None:
                had_error = True
                self._emit_error(f"Ошибка захвата ({ct.name}): {ct.error}")
        return had_error

    def _abort_startup(self) -> None:
        """Best-effort teardown of anything opened by a failed start().

        Ensures no capture threads, recorder writer thread / SoundFile, segmenter
        or worker threads are leaked when start() raises partway through.
        """
        for ct in self._captures:
            ct.stop()
        for ct in self._captures:
            if ct.ident is not None:  # only threads that actually started
                ct.join(timeout=_JOIN_TIMEOUT)
        for f in self._feeders:
            if f.ident is not None:
                f.join(timeout=_JOIN_TIMEOUT)
        if self._recorder is not None:
            try:
                self._recorder.stop()  # closes the SoundFile
            except Exception:
                logger.exception("Error stopping recorder during startup abort")
            self._recorder = None
        for seg_q in self._seg_queues:
            seg_q.put(None)
        for t in self._seg_threads:
            if t.ident is not None:
                t.join(timeout=_JOIN_TIMEOUT)
        if self._worker_thread is not None and self._worker_thread.ident is not None:
            self._stop_worker()

    # --- live --------------------------------------------------------------
    def _start_live(self) -> None:
        self._validate_capture_devices()
        self._set_status("Загрузка модели…")
        try:
            self._engine = build_engine(self.config)
        except Exception as exc:
            # Surface, then propagate so start() raises and the GUI worker resets
            # instead of being left with a half-initialized session.
            self._emit_error(f"Не удалось загрузить модель STT: {exc}")
            raise

        try:
            self._build_recorder(forward=True)
            # Two segmenter threads consuming the aligned per-channel blocks.
            segmenters = [self._new_segmenter(), self._new_segmenter()]
            labels = [self.mic_label, self.loop_label]
            for seg_q, segmenter, label in zip(self._seg_queues, segmenters, labels):
                t = threading.Thread(
                    target=self._segmenter_loop,
                    args=(seg_q, segmenter, label),
                    name="segmenter",
                    daemon=True,
                )
                self._seg_threads.append(t)

            self._start_worker()
            self._recorder.start()
            self._start_captures()
            for t in self._seg_threads:
                t.start()
            for ct in self._captures:
                ct.start()
            for f in self._feeders:
                f.start()
        except Exception as exc:
            # Clean up any partially opened threads/handles, then propagate.
            self._emit_error(f"Не удалось запустить запись: {exc}")
            self._abort_startup()
            raise
        self._set_status("Идёт запись…")

    def _stop_live(self) -> None:
        self._set_status("Остановка…")
        for ct in self._captures:
            ct.stop()
        self._join_captures_and_feeders()
        capture_failed = self._surface_capture_errors()
        if self._recorder is not None:
            self._recorder.stop()  # final drain -> forwards last aligned blocks
        # Signal segmenters end (after all aligned blocks were forwarded).
        for seg_q in self._seg_queues:
            seg_q.put(None)
        for t in self._seg_threads:
            t.join(timeout=_JOIN_TIMEOUT)
            if t.is_alive():
                logger.warning(
                    "Segmenter thread %s did not exit within %.1fs; abandoning it",
                    t.name,
                    _JOIN_TIMEOUT,
                )
        self._set_status("Завершение транскрипции…")
        self._stop_worker()
        if capture_failed:
            self._emit_error(
                "Запись завершена с ошибкой захвата — результат может быть неполным."
            )
        self._emit_systemic_failure_if_all_failed()
        self._finish()

    # --- batch (record now, transcribe on stop) ----------------------------
    def _start_recording_only(self) -> None:
        self._validate_capture_devices()
        try:
            self._build_recorder(forward=False)
            self._recorder.start()
            self._start_captures()
            for ct in self._captures:
                ct.start()
            for f in self._feeders:
                f.start()
        except Exception as exc:
            self._emit_error(f"Не удалось запустить запись: {exc}")
            self._abort_startup()
            raise
        self._set_status("Идёт запись…")

    def _stop_batch(self) -> None:
        self._set_status("Остановка записи…")
        for ct in self._captures:
            ct.stop()
        self._join_captures_and_feeders()
        capture_failed = self._surface_capture_errors()
        if self._recorder is not None:
            self._recorder.stop()
        if capture_failed:
            self._emit_error(
                "Запись завершена с ошибкой захвата — расшифровка может быть неполной."
            )
        # Now transcribe both tracks from disk.
        self._run_disk_pass(self.wav_path, stereo=True)

    # --- file (import) -----------------------------------------------------
    def _start_file(self) -> None:
        if self.import_path is None:
            # Emit, then raise so start() propagates and the GUI worker's
            # start_session `except` resets `_session` (matching how
            # _start_live/_start_recording_only surface load failures). A bare
            # return here would leave start() completing normally with no reset
            # path, wedging the worker ("Сессия уже запущена" forever). Do NOT
            # call _finish() — that would write an empty transcript / fire
            # on_finished for a session that never actually started.
            self._emit_error("Не указан файл для импорта")
            raise ValueError("Не указан файл для импорта")
        try:
            # Validate the file is openable before spawning the pass; the
            # channel count no longer matters (imports are always mono/neutral).
            wav_source.read_info(self.import_path)
        except Exception as exc:
            # Same contract: surface the error, then re-raise so start()
            # propagates and the worker resets instead of staying wedged.
            self._emit_error(f"Не удалось открыть файл: {exc}")
            raise RuntimeError(f"Не удалось открыть файл: {exc}") from exc
        # An imported file is ALWAYS transcribed as a single speaker (the neutral
        # "Speaker" label) via the mono disk pass, regardless of channel count. We
        # do not split stereo, attribute channels, or detect correlation for
        # imports — dual "Я"/"Собеседники" attribution is reserved for genuine
        # live-recorded session.wav (batch mode). iter_blocks(channel=None)
        # downmixes any multi-channel file to mono. read_info above still runs to
        # validate the file is openable before we spawn the pass.
        self._disk_thread = threading.Thread(
            target=self._run_disk_pass,
            args=(self.import_path, False),
            name="file-pass",
            daemon=True,
        )
        self._disk_thread.start()

    # --- shared disk pass (batch + file) -----------------------------------
    def _run_disk_pass(self, path: Path, stereo: bool) -> None:
        if self._engine is None:
            self._set_status("Загрузка модели…")
            try:
                self._engine = build_engine(self.config)
            except Exception as exc:
                self._emit_error(f"Не удалось загрузить модель STT: {exc}")
                self._finish()
                return

        self._set_status("Расшифровка…")
        self._start_worker()
        rate = self.rate
        bs = _DISK_BLOCKSIZE
        try:
            if stereo:
                seg_mic = self._new_segmenter()
                seg_loop = self._new_segmenter()
                it0 = wav_source.iter_blocks(
                    path, bs, channel=0, target_sample_rate=rate
                )
                it1 = wav_source.iter_blocks(
                    path, bs, channel=1, target_sample_rate=rate
                )
                for b0, b1 in itertools.zip_longest(it0, it1):
                    if self._cancel.is_set():
                        break
                    if b0 is not None:
                        for seg in seg_mic.feed(b0):
                            self._enqueue(seg, self.mic_label)
                    if b1 is not None:
                        for seg in seg_loop.feed(b1):
                            self._enqueue(seg, self.loop_label)
                for seg in seg_mic.flush():
                    self._enqueue(seg, self.mic_label)
                for seg in seg_loop.flush():
                    self._enqueue(seg, self.loop_label)
            else:
                seg = self._new_segmenter()
                for block in wav_source.iter_blocks(
                    path, bs, channel=None, target_sample_rate=rate
                ):
                    if self._cancel.is_set():
                        break
                    for s in seg.feed(block):
                        self._enqueue(s, self.neutral_label)
                for s in seg.flush():
                    self._enqueue(s, self.neutral_label)
        except Exception as exc:
            self._emit_error(f"Ошибка чтения аудио: {exc}")

        self._stop_worker()
        self._emit_systemic_failure_if_all_failed()
        self._finish()
