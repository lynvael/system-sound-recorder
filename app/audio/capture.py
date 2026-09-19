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


class CaptureThread(threading.Thread):
    def __init__(
        self,
        device_id: str,
        out_queue: "queue.Queue[np.ndarray | None]",
        *,
        frame_size: int,
        target_sample_rate: int,
        native_sample_rate: int = DEFAULT_NATIVE_RATE,
        chunk_frames: int = 1024,
        name: str = "capture",
    ) -> None:
        super().__init__(name=name, daemon=True)
        self.device_id = device_id
        self.out_queue = out_queue
        self.frame_size = frame_size
        self.target_sample_rate = target_sample_rate
        self.native_sample_rate = native_sample_rate
        self.chunk_frames = chunk_frames

        self._stop_event = threading.Event()
        self._resampler = StreamingResampler(native_sample_rate, target_sample_rate)
        self._pending = np.empty(0, dtype=np.float32)
        self.error: Exception | None = None

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
            mic = sc.get_microphone(self.device_id, include_loopback=True)
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
