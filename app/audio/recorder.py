"""AlignedRecorder — stream a single stereo WAV aligned to a common clock.

Disk is the single source of truth. Both channels are written continuously to
one stereo `session.wav` (target rate, PCM16; L = mic / "Я", R = loopback /
"Собеседники"), block-appended via soundfile, never buffering the whole session
in RAM.

On each tick the recorder aligns both channels to
`expected = round(elapsed * sample_rate)` using a shared monotonic clock:
  - a channel that is short is padded with silence up to `expected`,
  - a channel that ran ahead is trimmed (only while the excess is silence).
This corrects both loopback silence gaps (loopback often yields no data when the
system is quiet) and clock drift between the two capture devices.

Because both channels are emitted in lockstep and gaplessly, the position
"sample N" is one instant in both channels: a VAD segment on either channel maps
directly onto the shared timeline. The aligned per-channel blocks are forwarded
to `on_left` / `on_right` so the segmenters see exactly what is on disk, and a
segmenter's own running sample counter equals the absolute session sample index.

Optionally, two derived mono files (`left_path` = mic / "Я", `right_path` =
loopback / "Собеседники") are streamed alongside the stereo file. They are
written from the SAME aligned blocks in the same tick, so all three files stay
sample-for-sample aligned; the stereo `session.wav` remains the single source of
truth (batch/file re-read and the VAD timeline are unchanged). This costs roughly
one extra copy of the audio on disk.

Callbacks (`on_left`, `on_right`) are invoked from the recorder's writer thread.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import soundfile as sf

from app.log import get_logger

logger = get_logger("recorder")

# A channel may run at most this far ahead of the write cursor before its excess
# is trimmed (only when that excess is silence) — the drift-correction bound.
_DRIFT_TRIM_SECONDS = 1.0
# Samples counted as silence for trim decisions.
_SILENCE_EPS = 1e-4


class _ChannelBuffer:
    """Thread-safe FIFO of float32 samples with front-popping."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buf = np.empty(0, dtype=np.float32)

    def append(self, samples: np.ndarray) -> None:
        with self._lock:
            self._buf = np.concatenate(
                [self._buf, np.asarray(samples, dtype=np.float32).reshape(-1)]
            )

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    def pop(self, n: int) -> np.ndarray:
        """Pop up to n samples from the front; pad with silence if short."""
        with self._lock:
            take = min(n, len(self._buf))
            out = self._buf[:take]
            self._buf = self._buf[take:]
        if take < n:
            out = np.concatenate([out, np.zeros(n - take, dtype=np.float32)])
        return out.astype(np.float32, copy=False)

    def trim_silence(self, keep: int) -> int:
        """Trim excess beyond `keep` samples if that excess is silence.

        Returns the number of samples dropped (0 if the excess wasn't silence).
        """
        with self._lock:
            excess = len(self._buf) - keep
            if excess <= 0:
                return 0
            head = self._buf[keep:]
            if head.size and float(np.max(np.abs(head))) < _SILENCE_EPS:
                self._buf = self._buf[:keep]
                return excess
            return 0


class AlignedRecorder:
    def __init__(
        self,
        path: str | Path,
        sample_rate: int,
        *,
        left_path: str | Path | None = None,
        right_path: str | Path | None = None,
        on_left: Callable[[np.ndarray], None] | None = None,
        on_right: Callable[[np.ndarray], None] | None = None,
        on_error: Callable[[str], None] | None = None,
        tick_interval: float = 0.05,
    ) -> None:
        self.path = Path(path)
        self.sample_rate = int(sample_rate)
        # Optional per-channel mono outputs. When set, the SAME aligned left /
        # right blocks written into the stereo file are also streamed into these
        # two mono files, so all three stay sample-for-sample aligned (VAD sample
        # index == position in every file). The stereo file remains the single
        # source of truth (batch/file re-read still reads it); these are derived
        # convenience tracks (L = mic / "Я", R = loopback / "Собеседники").
        self.left_path = Path(left_path) if left_path is not None else None
        self.right_path = Path(right_path) if right_path is not None else None
        self.on_left = on_left
        self.on_right = on_right
        self.on_error = on_error
        self.tick_interval = tick_interval

        self._left = _ChannelBuffer()
        self._right = _ChannelBuffer()
        self._file: sf.SoundFile | None = None
        self._left_file: sf.SoundFile | None = None
        self._right_file: sf.SoundFile | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._start_time = 0.0
        self._written = 0  # samples written per channel (equal for both)

    # --- input -------------------------------------------------------------
    def submit_left(self, samples: np.ndarray) -> None:
        self._left.append(samples)

    def submit_right(self, samples: np.ndarray) -> None:
        self._right.append(samples)

    # --- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        # Open the (up to three) SoundFiles transactionally: the writer thread's
        # `finally` in _run() is the only other place that closes them, and it
        # only runs once the thread has started. So if the 2nd or 3rd open raises
        # (disk full, bad path, permissions) BEFORE the thread starts, the
        # already-opened handles would leak — and on Windows an open handle blocks
        # deleting/reopening the partial session.wav. Guard the opens so any
        # failure closes whatever we already opened and resets them to None.
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = sf.SoundFile(
                str(self.path),
                mode="w",
                samplerate=self.sample_rate,
                channels=2,
                subtype="PCM_16",
            )
            if self.left_path is not None:
                self.left_path.parent.mkdir(parents=True, exist_ok=True)
                self._left_file = sf.SoundFile(
                    str(self.left_path),
                    mode="w",
                    samplerate=self.sample_rate,
                    channels=1,
                    subtype="PCM_16",
                )
            if self.right_path is not None:
                self.right_path.parent.mkdir(parents=True, exist_ok=True)
                self._right_file = sf.SoundFile(
                    str(self.right_path),
                    mode="w",
                    samplerate=self.sample_rate,
                    channels=1,
                    subtype="PCM_16",
                )
        except Exception:
            # Close whatever opened before the failure (each guarded), reset to
            # None, then re-raise so the caller's start() abort path runs.
            for attr in ("_file", "_left_file", "_right_file"):
                f = getattr(self, attr)
                if f is not None:
                    try:
                        f.close()
                    except Exception:
                        logger.exception("Error closing %s during start abort", attr)
                    setattr(self, attr, None)
            raise
        self._start_time = time.monotonic()
        self._written = 0
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="aligned-recorder", daemon=True
        )
        self._thread.start()
        logger.debug("AlignedRecorder writing %s", self.path)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    # --- writer thread -----------------------------------------------------
    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self._tick()
                time.sleep(self.tick_interval)
            # Final drain: flush everything still buffered, padding to equal len.
            self._tick(final=True)
        except Exception as exc:
            logger.exception("AlignedRecorder writer failed")
            if self.on_error is not None:
                try:
                    self.on_error(f"Ошибка записи аудио: {exc}")
                except Exception:
                    logger.exception("AlignedRecorder on_error callback failed")
        finally:
            for attr in ("_file", "_left_file", "_right_file"):
                f = getattr(self, attr)
                if f is not None:
                    try:
                        f.close()
                    except Exception:
                        logger.exception("Error closing %s", attr)
                    setattr(self, attr, None)

    def _tick(self, final: bool = False) -> None:
        elapsed = time.monotonic() - self._start_time
        expected = round(elapsed * self.sample_rate)

        if final:
            # Emit whatever remains on both channels, aligned to equal length.
            expected = self._written + max(len(self._left), len(self._right))

        n = expected - self._written
        if n <= 0:
            return

        left = self._left.pop(n)
        right = self._right.pop(n)
        self._write_block(left, right)
        self._written += n

        if not final:
            # Drift correction: if a channel ran ahead, trim its silent excess.
            cap = int(_DRIFT_TRIM_SECONDS * self.sample_rate)
            self._left.trim_silence(cap)
            self._right.trim_silence(cap)

    def _write_block(self, left: np.ndarray, right: np.ndarray) -> None:
        # Stereo is written before the two mono files. If a write raises mid-block
        # the three files could differ by one block — cosmetic only: a write error
        # takes the terminal path (_run's except -> on_error) which tears the whole
        # session down, so the one-block skew is never consumed.
        if self._file is not None:
            stereo = np.stack([left, right], axis=1)
            self._file.write(stereo)
        # Derived per-channel mono files, written from the same aligned blocks so
        # they stay sample-for-sample in step with the stereo source of truth.
        if self._left_file is not None:
            self._left_file.write(left)
        if self._right_file is not None:
            self._right_file.write(right)
        if self.on_left is not None:
            self.on_left(left)
        if self.on_right is not None:
            self.on_right(right)
