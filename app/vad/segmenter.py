"""ContinuousSegmenter — Silero VAD speech segmentation over an endless stream.

Based on Chisa's StreamingSegmenter (512-sample frames, PREROLL_FRAMES so the
first word isn't clipped), with two changes for a recorder:

  1. After a segment finalizes (silence or max_duration) it auto-resets and
     keeps running — it is continuous, not one-shot. There is no global
     start-timeout that would end the session on initial silence.
  2. It tracks a running sample position over the whole stream. Because both
     channels are fed the aligned stream from AlignedRecorder (live) or read
     from the same file positions (batch/file), a segment's sample indices map
     directly onto the shared session timeline.

Each channel owns its own Silero model instance (`load_silero_vad()` per
channel) since the model is stateful.

`feed(samples)` returns a list of finalized `RawSegment`s. Call `flush()` at end
of stream to finalize any in-progress segment.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from app.log import get_logger

logger = get_logger("vad")

# Frames kept before a detected onset so the first word isn't clipped (~256 ms).
PREROLL_FRAMES = 8


@dataclass
class RawSegment:
    """A finalized speech span in the shared stream (audio still un-transcribed)."""

    start: int  # sample index in the aligned stream
    end: int  # sample index in the aligned stream (exclusive)
    audio: np.ndarray  # 16 kHz mono float32 speech samples


def load_silero_vad():
    """Load a fresh Silero VAD model instance (one per channel)."""
    from silero_vad import load_silero_vad as _load

    return _load()


class ContinuousSegmenter:
    def __init__(
        self,
        model,
        *,
        sample_rate: int,
        threshold: float,
        silence_timeout: float,
        max_duration: float,
        frame_size: int = 512,
    ) -> None:
        self.model = model
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.window = frame_size  # Silero VAD needs 512-sample frames at 16 kHz
        frame_ms = self.window / self.sample_rate * 1000

        self.silence_frames_to_stop = int(silence_timeout * 1000 / frame_ms)
        self.max_frames = int(max_duration * 1000 / frame_ms)

        # Stream-level state (persists across segments).
        self.residual = np.empty(0, dtype=np.float32)
        self.stream_pos = 0  # samples consumed from the stream so far
        self._reset_segment()

    def _reset_segment(self) -> None:
        """Clear per-segment state and reset the model; keep stream position."""
        self.model.reset_states()
        self.preroll: deque[np.ndarray] = deque(maxlen=PREROLL_FRAMES)
        self.collected: list[np.ndarray] = []
        self.speech_started = False
        self.silence_frames = 0
        self.seg_frames = 0
        self.seg_start = 0

    def feed(self, samples: np.ndarray) -> list[RawSegment]:
        out: list[RawSegment] = []
        self.residual = np.concatenate(
            [self.residual, np.asarray(samples, dtype=np.float32).reshape(-1)]
        )

        while len(self.residual) >= self.window:
            frame = self.residual[: self.window]
            self.residual = self.residual[self.window :]
            frame_start = self.stream_pos

            prob = self.model(torch.from_numpy(frame), self.sample_rate).item()

            if prob >= self.threshold:
                if not self.speech_started:
                    self.speech_started = True
                    num_pre = len(self.preroll)
                    self.seg_start = frame_start - num_pre * self.window
                    self.collected.extend(self.preroll)
                self.silence_frames = 0
                self.collected.append(frame)
            elif self.speech_started:
                self.silence_frames += 1
                self.collected.append(frame)  # keep trailing silence in segment
            else:
                self.preroll.append(frame)

            self.stream_pos += self.window
            if self.speech_started:
                self.seg_frames += 1

            if (
                self.speech_started
                and self.silence_frames >= self.silence_frames_to_stop
            ):
                seg = self._finalize()
                if seg is not None:
                    out.append(seg)
                self._reset_segment()
            elif self.speech_started and self.seg_frames >= self.max_frames:
                logger.debug("Max duration reached, splitting segment")
                seg = self._finalize()
                if seg is not None:
                    out.append(seg)
                self._reset_segment()

        return out

    def flush(self) -> list[RawSegment]:
        """Finalize any in-progress segment at end of stream."""
        if self.speech_started:
            seg = self._finalize()
            self._reset_segment()
            if seg is not None:
                return [seg]
        return []

    def _finalize(self) -> RawSegment | None:
        if not self.collected:
            return None
        audio = np.concatenate(self.collected).astype(np.float32)
        end = self.seg_start + len(audio)
        return RawSegment(start=max(0, self.seg_start), end=end, audio=audio)
