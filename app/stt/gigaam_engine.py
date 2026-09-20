"""GigaAM-v3 STT engine (Russian only).

Loaded via `AutoModel.from_pretrained("ai-sage/GigaAM-v3", revision=...,
trust_remote_code=True)`. `model.transcribe(path)` takes a FILE PATH, so each
numpy segment is written to a temporary 16 kHz WAV, transcribed, then deleted.

GigaAM-v3's short-form `model.transcribe()` rejects clips over ~30 s ("Too long
wav file, use 'transcribe_longform' method."). `transcribe_longform` is NOT used
here (product decision). Instead, over-length segments are sliced by us with the
project's own Silero VAD (`ContinuousSegmenter`) into sub-<=limit chunks, each
transcribed via the SHORT-form `transcribe()` and joined in chronological order.

GigaAM needs extra dependencies (install with `uv sync --extra gigaam`). If they
are missing, constructing this engine raises a clear, catchable error.
"""

from __future__ import annotations

import os
import tempfile
import time

import numpy as np
import soundfile as sf

from app.config import STTSettings, VADSettings
from app.log import get_logger
from app.vad.segmenter import ContinuousSegmenter, load_silero_vad

logger = get_logger("stt.gigaam")

# Routing threshold: clips shorter than this go straight to short-form
# `transcribe()`; clips at/above it are sliced by VAD first. Sits safely below
# GigaAM-v3's ~30 s hard limit so a marginally-over clip (resampling jitter,
# off-by-a-frame) never trips the "Too long wav file" error.
_MAX_TRANSCRIBE_SECONDS = 24.0

# `max_duration` handed to the internal slicing segmenter. ContinuousSegmenter
# hard-splits continuous speech at `max_frames = int(max_duration*1000/frame_ms)`
# (frame_ms = 512/16000*1000 = 32 ms), so a sub-chunk is at most
# `max_frames + PREROLL_FRAMES` frames long (preroll is prepended before onset).
# With 20 s: max_frames = int(20000/32) = 625, so the longest possible sub-chunk
# is (625 + 8) * 32 ms = 20.256 s -- ~9.7 s under GigaAM's ~30 s limit. This is
# the guarantee that no sub-chunk can be rejected as too long. Also reused as the
# fixed-window size for the zero-sub-segment fallback (each window <= 20 s).
_SPLIT_MAX_DURATION = 20.0


class GigaAMDependencyError(RuntimeError):
    """Raised when GigaAM extra dependencies are not installed."""


class GigaAMEngine:
    def __init__(
        self,
        settings: STTSettings,
        sample_rate: int,
        vad: VADSettings,
    ) -> None:
        self.s = settings
        self.sample_rate = sample_rate
        # VAD params are threaded in (not re-read from .env inside the engine) to
        # stay consistent with the app's "construct one Config at startup and
        # thread it through engines" convention (see config.py / Session, which
        # reads self.config.vad). The factory passes config.vad here.
        self.vad = vad
        # Silero model is built lazily on the first over-length segment, so
        # short-only sessions never pay the VAD load cost.
        self._vad_model = None

        try:
            from transformers import AutoModel
        except Exception as exc:  # pragma: no cover - transformers is a base dep
            raise GigaAMDependencyError(
                f"transformers is required for GigaAM: {exc}"
            ) from exc

        logger.info(
            "Loading GigaAM %s (revision=%s)",
            self.s.gigaam_model_id,
            self.s.gigaam_revision,
        )
        started = time.perf_counter()
        try:
            self.model = AutoModel.from_pretrained(
                self.s.gigaam_model_id,
                revision=self.s.gigaam_revision,
                trust_remote_code=True,
            )
        except ImportError as exc:
            # GigaAM's remote code imports its extra stack (pyannote, hydra,
            # omegaconf, sentencepiece, torchcodec) at load time.
            raise GigaAMDependencyError(
                "GigaAM requires extra dependencies. Install them with "
                "`uv sync --extra gigaam`. Original error: " + str(exc)
            ) from exc
        logger.info("GigaAM loaded in %.2fs", time.perf_counter() - started)

    def transcribe(self, audio: np.ndarray) -> str:
        # GigaAM-v3 is Russian only.
        audio = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)
        started = time.perf_counter()
        duration = len(audio) / float(self.sample_rate)
        if duration < _MAX_TRANSCRIBE_SECONDS:
            text = self._transcribe_short(audio)
        else:
            text = self._transcribe_sliced(audio)
        logger.debug(
            "Transcribed %d samples (%.2fs) in %.2fs: %r",
            len(audio),
            duration,
            time.perf_counter() - started,
            text,
        )
        return text

    def _transcribe_short(self, audio: np.ndarray) -> str:
        """Short-form transcribe of a <= limit clip via a temp WAV.

        A genuine `model.transcribe()` error is allowed to propagate: the session
        worker's per-segment except handles it as NON-terminal (see MEMORY). We
        never swallow it into empty text.
        """
        fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="gigaam_")
        os.close(fd)
        try:
            sf.write(tmp_path, audio, self.sample_rate, subtype="PCM_16")
            result = self.model.transcribe(tmp_path)
            return self._extract_text(result).strip()
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def _transcribe_sliced(self, audio: np.ndarray) -> str:
        """Slice an over-length clip with VAD, transcribe each sub-chunk short-form.

        A FRESH ContinuousSegmenter is built per long segment: the segmenter is
        stateful (keeps `residual`/`stream_pos`/in-progress collection across
        `feed` calls), so a new instance guarantees no residual from a previous
        segment bleeds in. The cached Silero model is reused (its per-segment
        state is reset in the segmenter constructor).
        """
        splitter = self._new_splitter()
        sub_segments = splitter.feed(audio)
        sub_segments.extend(splitter.flush())
        chunks = [seg.audio for seg in sub_segments]

        if not chunks:
            # The parent segment came from VAD, so we KNOW it is speech; if the
            # sub-VAD pass yields nothing (e.g. quieter re-analysis), never drop
            # it -- fall back to a plain fixed-window time split into <= limit
            # windows so every part still gets transcribed.
            logger.warning(
                "Sub-VAD found no speech in a %.2fs segment known to be speech; "
                "falling back to fixed-window split",
                len(audio) / float(self.sample_rate),
            )
            chunks = self._fixed_windows(audio)

        parts = [self._transcribe_short(chunk) for chunk in chunks]
        return " ".join(p for p in parts if p)

    def _new_splitter(self) -> ContinuousSegmenter:
        return ContinuousSegmenter(
            self._get_vad_model(),
            sample_rate=self.sample_rate,
            threshold=self.vad.threshold,
            silence_timeout=self.vad.silence_timeout,
            max_duration=_SPLIT_MAX_DURATION,
        )

    def _get_vad_model(self):
        if self._vad_model is None:
            self._vad_model = load_silero_vad()
        return self._vad_model

    def _fixed_windows(self, audio: np.ndarray) -> list[np.ndarray]:
        """Split raw audio into consecutive <= _SPLIT_MAX_DURATION windows."""
        win = int(_SPLIT_MAX_DURATION * self.sample_rate)
        return [audio[i : i + win] for i in range(0, len(audio), win)]

    @staticmethod
    def _extract_text(result) -> str:
        """GigaAM may return a str or a dict/obj carrying the transcription."""
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            for key in ("transcription", "text"):
                if key in result:
                    return str(result[key])
            logger.warning(
                "Unrecognized transcription dict shape (keys=%r); ignoring",
                sorted(result.keys()),
            )
            return ""
        # Attribute-carrying object (e.g. a dataclass-like segment).
        for attr in ("transcription", "text"):
            value = getattr(result, attr, None)
            if value is not None:
                return str(value)
        logger.warning(
            "Unrecognized transcription result type %s; ignoring",
            type(result).__name__,
        )
        return ""
