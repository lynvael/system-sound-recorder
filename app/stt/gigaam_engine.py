"""GigaAM-v3 STT engine (Russian only).

Loaded via `AutoModel.from_pretrained("ai-sage/GigaAM-v3", revision=...,
trust_remote_code=True)`. `model.transcribe(path)` takes a FILE PATH, so each
numpy segment is written to a temporary 16 kHz WAV, transcribed, then deleted.

GigaAM needs extra dependencies (install with `uv sync --extra gigaam`). If they
are missing, constructing this engine raises a clear, catchable error.
"""

from __future__ import annotations

import os
import tempfile
import time

import numpy as np
import soundfile as sf

from app.config import STTSettings
from app.log import get_logger

logger = get_logger("stt.gigaam")


class GigaAMDependencyError(RuntimeError):
    """Raised when GigaAM extra dependencies are not installed."""


class GigaAMEngine:
    def __init__(self, settings: STTSettings, sample_rate: int) -> None:
        self.s = settings
        self.sample_rate = sample_rate

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

    def transcribe(self, audio: np.ndarray, language: str | None) -> str:
        # GigaAM-v3 is Russian only; `language` is ignored by design.
        audio = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)
        started = time.perf_counter()
        fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="gigaam_")
        os.close(fd)
        try:
            sf.write(tmp_path, audio, self.sample_rate, subtype="PCM_16")
            result = self.model.transcribe(tmp_path)
            text = self._extract_text(result).strip()
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        logger.debug(
            "Transcribed %d samples in %.2fs: %r",
            len(audio),
            time.perf_counter() - started,
            text,
        )
        return text

    @staticmethod
    def _extract_text(result) -> str:
        """GigaAM may return a str or a dict/obj carrying the transcription."""
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            for key in ("transcription", "text"):
                if key in result:
                    return str(result[key])
        return str(result)
