"""Build the configured STT engine, lazily (at session start).

Engines are imported inside `build_engine` so importing this module doesn't pull
torch/transformers, and so a missing GigaAM extra only errors when GigaAM is
actually selected.
"""

from __future__ import annotations

from app.config import Config
from app.log import get_logger
from app.stt.base import ASREngine

logger = get_logger("stt.factory")


def build_engine(config: Config) -> ASREngine:
    """Construct the ASREngine named by config.stt.engine ("whisper"|"gigaam")."""
    engine = config.stt.engine.lower()
    sample_rate = config.capture.target_sample_rate
    if engine == "gigaam":
        # GigaAMEngine.__init__ already raises GigaAMDependencyError when its
        # extra dependencies are missing, so no extra ImportError handling here.
        from app.stt.gigaam_engine import GigaAMEngine

        return GigaAMEngine(config.stt, sample_rate)

    raise ValueError(
        f"Unknown STT engine {config.stt.engine!r} (expected 'whisper' or 'gigaam')"
    )
