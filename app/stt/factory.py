"""Build the configured STT engine, lazily (at session start).

Engines are imported inside `build_engine` so importing this module doesn't pull
torch/transformers, and so a missing GigaAM extra only errors when GigaAM is
actually selected.
"""

from __future__ import annotations

from pathlib import Path

from app.config import Config
from app.log import get_logger
from app.stt.base import ASREngine

logger = get_logger("stt.factory")


def build_engine(
    config: Config,
    work_dir: str | Path | None = None,
) -> ASREngine:
    """Construct the ASREngine named by config.stt.engine ("gigaam").

    work_dir: optional directory for the engine's temp files. GigaAM writes
    one temp WAV per segment and its remote code cannot open non-ASCII paths
    on Windows, so the session passes a dedicated ASCII-only dir under the
    output dir (NEVER session_dir — user session names may be Cyrillic). The
    engine validates the dir and falls back to the system tempdir when it is
    missing/unusable; engines that keep no temp files ignore it.
    """
    engine = config.stt.engine.lower()
    sample_rate = config.capture.target_sample_rate
    if engine == "gigaam":
        # GigaAMEngine.__init__ already raises GigaAMDependencyError when its
        # extra dependencies are missing, so no extra ImportError handling here.
        from app.stt.gigaam_engine import GigaAMEngine

        return GigaAMEngine(config.stt, sample_rate, config.vad, work_dir=work_dir)

    raise ValueError(
        f"Unknown STT engine {config.stt.engine!r} (expected 'gigaam')"
    )
