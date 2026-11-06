"""Live capture of a single device into exact target-rate mono frames.

`CaptureThread` runs one background thread per device (one for the mic, one
for the loopback). Each loop iteration:
  1. pulls a native-rate block from the PyAudioWPatch stream (`stream.read`),
  2. decodes the float32 bytes and downmixes to mono,
  3. streaming-resamples native -> target rate (`StreamingResampler`),
  4. buffers into exact `frame_size`-sample frames using the `pending` pattern
     ported from Chisa's orchestrator/audio.go,
  5. pushes each frame onto an output queue.

The stream is opened at the device's NATIVE rate (`defaultSampleRate`,
fallback `DEFAULT_NATIVE_RATE`) in `paFloat32`, so WASAPI does no extra
sample-rate conversion and soxr takes the audio to the target rate.

The single, common timeline is owned by `AlignedRecorder` (which aligns both
channels to a shared monotonic clock); individual frames are not timestamped.

`pyaudiowpatch` is imported lazily (in `backend.get_backend()`, and inside
`run()` for the `paFloat32` constant), so this module imports cleanly on
non-Windows dev boxes where the package is not installed.
"""

from __future__ import annotations

import queue
import threading

import numpy as np

from app.audio import backend
from app.audio.devices import get_device
from app.audio.resample import StreamingResampler
from app.log import get_logger

logger = get_logger("capture")

# Fallback native rate, used only when a device's `defaultSampleRate` is
# missing or <= 0. WASAPI shared mode commonly runs at 48 kHz; the device's
# own mix-format rate is preferred (read in run() after resolution) so the
# driver does no extra SRC.
DEFAULT_NATIVE_RATE = 48000


def _downmix_mono(block: np.ndarray) -> np.ndarray:
    """Average multichannel audio down to a 1-D float32 mono signal."""
    block = np.asarray(block, dtype=np.float32)
    if block.ndim == 1:
        return block
    if block.shape[1] == 1:
        return block[:, 0]
    return block.mean(axis=1).astype(np.float32)


def _pa_phrase(code: int) -> str | None:
    """Short Russian phrase for a PortAudio error code (None if unmapped).

    The table is built from the `pyaudiowpatch` constants (imported lazily so
    this module stays importable on non-Windows; after the first import this
    is a `sys.modules` hit). This runs only on the error path.
    """
    try:
        import pyaudiowpatch as pyaudio
    except ImportError:
        return None
    phrases = {
        pyaudio.paDeviceUnavailable: "устройство отключено",
        pyaudio.paInputOverflowed: "переполнение буфера ввода",
        pyaudio.paOutputUnderflowed: "недополнение буфера вывода",
        pyaudio.paInvalidDevice: "устройство недоступно",
        pyaudio.paInvalidSampleRate: "устройство не поддерживает "
        "запрошенную частоту дискретизации",
        pyaudio.paInvalidChannelCount: "устройство не поддерживает "
        "запрошенное число каналов",
        pyaudio.paSampleFormatNotSupported: "формат сэмплов не поддерживается",
        pyaudio.paUnanticipatedHostError: "непредвиденная ошибка "
        "аудио-подсистемы",
        pyaudio.paInternalError: "внутренняя ошибка аудио-подсистемы",
        pyaudio.paNotInitialized: "аудио-подсистема не инициализирована",
        pyaudio.paTimedOut: "таймаут аудио-операции",
        pyaudio.paBadStreamPtr: "аудиопоток захвата закрыт",
        pyaudio.paStreamIsStopped: "аудиопоток захвата остановлен",
        pyaudio.paStreamIsNotStopped: "аудиопоток захвата не остановлен",
        pyaudio.paInsufficientMemory: "недостаточно памяти",
    }
    return phrases.get(code)


def _translate_backend_error(exc: Exception) -> Exception:
    """Map a PyAudioWPatch/PortAudio failure to a compact Russian error.

    PortAudio raises OSError with the numeric error CODE in `errno`
    (== args[0]) and the English PortAudio text in `strerror` (see
    _portaudiomodule.c: PyErr_SetObject(PyExc_IOError,
    Py_BuildValue("(i,s)", err, Pa_GetErrorText(err)))). Mapped codes become
    short Russian phrases (the code is appended for field diagnostics);
    unmapped codes and non-OSError exceptions keep their original message so
    nothing is lost.
    """
    if isinstance(exc, OSError) and isinstance(exc.errno, int):
        phrase = _pa_phrase(exc.errno)
        if phrase is not None:
            err = RuntimeError(f"{phrase} (PortAudio {exc.errno})")
            err.__cause__ = exc
            return err
    return exc


class CaptureThread(threading.Thread):
    def __init__(
        self,
        device_id: str,
        out_queue: "queue.Queue[np.ndarray | None]",
        *,
        frame_size: int,
        target_sample_rate: int,
        expect_loopback: bool,
        chunk_frames: int = 1024,
        name: str = "capture",
    ) -> None:
        super().__init__(name=name, daemon=True)
        self.device_id = device_id
        self.out_queue = out_queue
        self.expect_loopback = expect_loopback
        self.frame_size = frame_size
        self.target_sample_rate = target_sample_rate
        self.chunk_frames = chunk_frames

        self._stop_event = threading.Event()
        # Created in run() AFTER the device is resolved, because the native
        # rate is only known then (per-device defaultSampleRate).
        self._resampler: StreamingResampler | None = None
        self._pending = np.empty(0, dtype=np.float32)
        self.error: Exception | None = None
        # Set once the device has been resolved+validated (the first thing run()
        # does). Lets Session distinguish "this channel is live" from "still
        # resolving / failed" so an all-devices-dead start can be detected fast.
        # A thread that fails resolution sets `.error` and exits WITHOUT setting
        # this; a thread that succeeds sets it and goes on recording.
        self.resolved = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def _emit_frames(self, samples: np.ndarray) -> None:
        """Buffer samples and emit exact frame_size chunks (the pending pattern)."""
        if samples.size:
            self._pending = np.concatenate([self._pending, samples])
        while len(self._pending) >= self.frame_size:
            frame = self._pending[: self.frame_size].copy()
            self._pending = self._pending[self.frame_size :]
            self.out_queue.put(frame)

    def run(self) -> None:
        stream = None
        try:
            p = backend.get_backend()
            # Exact name + isLoopbackDevice resolution, fail-fast (no fuzzy
            # fallbacks): a mic channel can never open a loopback endpoint.
            device = get_device(self.device_id, expect_loopback=self.expect_loopback)
            # Per-device native rate: capture at the device's mix-format rate
            # so WASAPI does no extra SRC; soxr takes it to the target rate.
            native = int(device.get("defaultSampleRate") or 0)
            if native <= 0:
                native = DEFAULT_NATIVE_RATE
            channels = int(device.get("maxInputChannels") or 0)
            self._resampler = StreamingResampler(native, self.target_sample_rate)
            self.resolved.set()
            logger.debug(
                "Capture start: %s (device=%r, native=%d -> target=%d)",
                self.name,
                device.get("name"),
                native,
                self.target_sample_rate,
            )
            # Local import: after the first one this is a sys.modules hit.
            # The constant is only needed here, never at module scope.
            import pyaudiowpatch as pyaudio

            stream = p.open(
                format=pyaudio.paFloat32,
                channels=channels,
                rate=native,
                input=True,
                input_device_index=device["index"],
                frames_per_buffer=self.chunk_frames,
            )
            while not self._stop_event.is_set():
                # PyAudioWPatch's stream.read() returns bytes (the C layer
                # builds a PyBytes of frames*channels*4); tolerate a
                # (bytes, n) tuple in case a future fork changes the type.
                data = stream.read(self.chunk_frames)
                if isinstance(data, tuple):
                    # Validate the payload so an unexpected shape (e.g.
                    # (n, bytes) or an empty tuple) fails here with a clear
                    # error instead of a cryptic frombuffer crash.
                    if not data or not isinstance(
                        data[0], (bytes, bytearray, memoryview)
                    ):
                        shape = ", ".join(type(x).__name__ for x in data)
                        raise TypeError(
                            "Unexpected stream.read() result: expected bytes "
                            f"or (bytes, n); got tuple({shape or 'empty'})"
                        )
                    data = data[0]
                block = np.frombuffer(data, dtype=np.float32)
                if block.size:
                    block = block.reshape(-1, channels)
                    mono = _downmix_mono(block)
                    resampled = self._resampler.resample_chunk(mono)
                    self._emit_frames(resampled)
            # Flush the resampler tail and any leftover partial frame.
            tail = self._resampler.flush()
            self._emit_frames(tail)
            if len(self._pending):
                pad = self.frame_size - len(self._pending)
                last = np.concatenate(
                    [self._pending, np.zeros(pad, dtype=np.float32)]
                )
                self.out_queue.put(last)
                self._pending = np.empty(0, dtype=np.float32)
        except Exception as exc:  # surfaced to Session via .error
            self.error = _translate_backend_error(exc)
            logger.exception("Capture thread %s failed", self.name)
        finally:
            if stream is not None:
                try:
                    stream.close()
                except Exception:  # noqa: BLE001 - may already be closed
                    logger.debug(
                        "Error closing capture stream %s", self.name, exc_info=True
                    )
            # Sentinel so consumers can drain and stop.
            self.out_queue.put(None)
            logger.debug("Capture stop: %s", self.name)
