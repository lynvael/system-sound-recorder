"""Application settings, loaded from .env via pydantic-settings.

Adapted from Chisa's app/common/config.py. Each settings group has its own env
prefix; defaults match the values decided in the tech spec, so an empty .env
behaves like the documented defaults.

`CaptureSettings.target_sample_rate` is the single source of truth for the audio
pipeline sample rate: the VAD segmenter and the streaming resampler both read it
from here rather than hardcoding 16000, so the two can never silently diverge.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class _Base(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class AppSettings(_Base):
    log_level: str = "INFO"


class CaptureSettings(_Base):
    model_config = SettingsConfigDict(
        env_prefix="CAPTURE_", env_file=".env", extra="ignore"
    )

    # Silero VAD processes fixed 512-sample frames at 16 kHz. This is the frame
    # size handed to the segmenter, and the granularity CaptureThread emits at.
    frame_size: int = 512
    # Single source of truth for the pipeline sample rate (VAD + resampler).
    target_sample_rate: int = 16000
    # How many native-rate frames to pull per stream.read() call.
    chunk_frames: int = 1024

    # Speaker labels. mic -> left channel ("Я"), loopback -> right ("Собеседники").
    # neutral_label is used for arbitrary mono imports (no diarization).
    mic_label: str = "Я"
    loopback_label: str = "Собеседники"
    neutral_label: str = "Speaker"


class STTSettings(_Base):
    model_config = SettingsConfigDict(
        env_prefix="STT_", env_file=".env", extra="ignore"
    )

    # Which engine to build. GigaAM is the only engine.
    engine: str = "gigaam"

    # GigaAM-v3 (Russian only). revision: "e2e_rnnt" | "e2e_ctc" (with punctuation).
    gigaam_model_id: str = "ai-sage/GigaAM-v3"
    gigaam_revision: str = "e2e_rnnt"

    # "auto" -> cuda:0 if available else cpu.
    device: str = "auto"


class VADSettings(_Base):
    model_config = SettingsConfigDict(
        env_prefix="VAD_", env_file=".env", extra="ignore"
    )

    threshold: float = 0.5
    silence_timeout: float = 0.8
    # Cap on a single segment length; long monologues are split at this bound.
    max_duration: float = 60.0


class LLMSettings(_Base):
    model_config = SettingsConfigDict(
        env_prefix="LLM_", env_file=".env", extra="ignore"
    )

    # OpenAI-compatible endpoint. `url` becomes the openai client `base_url`,
    # so it must include the /v1 suffix if the server expects one.
    url: str = "http://localhost:8080/v1"
    api_key: str = "not-needed"
    model: str = "local-model"

    # Map-reduce tunables. chunk_chars is the character budget per map chunk;
    # chunk_overlap keeps context across chunk boundaries. Sizes are in
    # characters (not tokens) — offline/CPU-friendly, no tokenizer needed.
    chunk_chars: int = 8000
    chunk_overlap: int = 400
    temperature: float = 0.3
    # Upper bound on the model's response length per LLM call.
    max_tokens: int = 16384
    # Per-request timeout in seconds (whole call, incl. retries by the client).
    request_timeout: float = 120.0


class SessionSettings(_Base):
    model_config = SettingsConfigDict(
        env_prefix="SESSION_", env_file=".env", extra="ignore"
    )

    # "live" | "batch" | "file".
    mode: str = "live"
    output_dir: str = "recordings"


class Config:
    """Aggregate of all settings groups, passed around the pipeline.

    Instantiating this reads .env once per group. Construct a single Config at
    startup and thread it through Session / engines.
    """

    def __init__(self) -> None:
        self.app = AppSettings()
        self.capture = CaptureSettings()
        self.stt = STTSettings()
        self.vad = VADSettings()
        self.session = SessionSettings()
        self.llm = LLMSettings()


def load_config() -> Config:
    """Load the full application configuration from environment / .env."""
    return Config()
