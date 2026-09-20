"""Live capture of a single device into exact target-rate mono frames.

`CaptureThread` runs one background thread per device (one for the mic, one for
the loopback). Each loop iteration:
  1. pulls a native-rate block from `soundcard` (`recorder.record`),
  2. downmixes to mono,
  3. streaming-resamples native -> target rate (`StreamingResampler`),
  4. buffers into exact `frame_size`-sample frames using the `pending` pattern
     ported from Chisa's orchestrator/audio.go,
  5. pushes each frame onto an output queue.

The single, common timeline is owned by `AlignedRecorder` (which aligns both
channels to a shared monotonic clock); individual frames are not timestamped.
"""

from __future__ import annotations

import queue
import threading

import numpy as np
import soundcard as sc

from app.audio.resample import StreamingResampler
from app.log import get_logger

logger = get_logger("capture")

# WASAPI shared mode commonly runs at 48 kHz; used when a native rate isn't
# otherwise known. soundcard honours the requested rate (resampling in the
# driver if the device differs), and our soxr stage takes it to the target rate.
DEFAULT_NATIVE_RATE = 48000


def _downmix_mono(block: np.ndarray) -> np.ndarray:
    """Average multichannel audio down to a 1-D float32 mono signal."""
    block = np.asarray(block, dtype=np.float32)
    if block.ndim == 1:
        return block
    if block.shape[1] == 1:
        return block[:, 0]
    return block.mean(axis=1).astype(np.float32)


def _resolve_device(device_id: str, expect_loopback: bool):
    """Resolve a soundcard device by id and VALIDATE it is the endpoint we asked.

    This guards against soundcard's FUZZY id resolution silently opening the
    WRONG endpoint. `soundcard.get_microphone(id, include_loopback=...)` (see
    mediafoundation.py `_match_device`) tries, in order: exact id match, then a
    NAME-substring match (`id in name`), then a FUZZY match (`re.match` of the id
    with `.*` between every character). If a device's exact id no longer resolves
    — e.g. the physical microphone is unplugged — the substring/fuzzy fallbacks
    can return a completely different endpoint.

    Two defenses:
      1. `include_loopback` is passed through as `expect_loopback`. The mic thread
         passes False, so `all_microphones(include_loopback=False)` doesn't even
         contain loopback (system-audio) endpoints — a loopback can never be a
         fallback candidate for the mic channel. (This is the root-cause fix for
         the duplicate-transcription bug: previously the mic thread hardcoded
         include_loopback=True, so a disconnected mic fuzzy-fell-back onto a
         loopback endpoint and captured the same system audio as the loop thread.)
      2. We assert the resolved device is the exact endpoint requested (id equal)
         AND that its `.isloopback` flag matches expectation. If either fails we
         FAIL FAST with a clear Russian message rather than silently capturing the
         wrong device. This turns the previously-silent wrong-device capture into a
         surfaced capture error. Session applies a partial-vs-total policy to it
         (see `Session._handle_capture_errors`): a single failed channel is
         non-terminal (the other channel still records), all channels failing is
         terminal.
    """
    kind = "системного звука" if expect_loopback else "микрофона"
    try:
        device = sc.get_microphone(device_id, include_loopback=expect_loopback)
    except Exception as exc:  # IndexError('no device with id ...') and friends
        raise RuntimeError(
            f"Устройство {kind} не найдено (id={device_id!r}). "
            "Возможно, оно отключено или недоступно."
        ) from exc
    # Reject soundcard's substring/fuzzy fallback: only the exact endpoint is OK.
    if device.id != device_id:
        raise RuntimeError(
            f"Устройство {kind} не удалось однозначно определить: "
            f"запрошен id={device_id!r}, а выбрано {device.name!r} "
            f"(id={device.id!r}). Захват прекращён, чтобы не записать не то "
            "устройство."
        )
    # The decisive guard against capturing system audio on the mic channel.
    if bool(device.isloopback) != expect_loopback:
        if expect_loopback:
            raise RuntimeError(
                f"Ожидалось устройство системного звука (loopback), но {device.name!r} "
                "им не является."
            )
        raise RuntimeError(
            f"Устройство микрофона {device.name!r} оказалось устройством "
            "системного звука (loopback). Захват прекращён, чтобы не дублировать "
            "системный звук на канале «Я»."
        )
    return device


class CaptureThread(threading.Thread):
    def __init__(
        self,
        device_id: str,
        out_queue: "queue.Queue[np.ndarray | None]",
        *,
        frame_size: int,
        target_sample_rate: int,
        expect_loopback: bool,
        native_sample_rate: int = DEFAULT_NATIVE_RATE,
        chunk_frames: int = 1024,
        name: str = "capture",
    ) -> None:
        super().__init__(name=name, daemon=True)
        self.device_id = device_id
        self.out_queue = out_queue
        self.expect_loopback = expect_loopback
        self.frame_size = frame_size
        self.target_sample_rate = target_sample_rate
        self.native_sample_rate = native_sample_rate
        self.chunk_frames = chunk_frames

        self._stop_event = threading.Event()
        self._resampler = StreamingResampler(native_sample_rate, target_sample_rate)
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
        try:
            mic = _resolve_device(self.device_id, self.expect_loopback)
            self.resolved.set()
            logger.debug(
                "Capture start: %s (native=%d -> target=%d)",
                self.name,
                self.native_sample_rate,
                self.target_sample_rate,
            )
            with mic.recorder(
                samplerate=self.native_sample_rate,
                channels=None,
                blocksize=self.chunk_frames,
            ) as rec:
                while not self._stop_event.is_set():
                    block = rec.record(numframes=self.chunk_frames)
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
            self.error = exc
            logger.exception("Capture thread %s failed", self.name)
        finally:
            # Sentinel so consumers can drain and stop.
            self.out_queue.put(None)
            logger.debug("Capture stop: %s", self.name)
