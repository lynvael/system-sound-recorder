"""Logging setup for Live Recorder.

Adapted from Chisa's app/common/log.py. Root logger stays at WARNING so
libraries (transformers, torch, ...) keep quiet; only the "recorder" tree
follows LOG_LEVEL.
"""

import logging

PREFIX = "recorder"
FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# Libraries that raise their own level and would otherwise slip through.
NOISY = ("transformers", "torch", "torchaudio", "numba", "soundcard", "urllib3")


class _TrimPrefix(logging.Formatter):
    """Drops the "recorder." prefix so the log shows plain module names."""

    def format(self, record: logging.LogRecord) -> str:
        record.name = record.name.removeprefix(PREFIX + ".")
        return super().format(record)


def setup(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(_TrimPrefix(FORMAT, datefmt="%H:%M:%S"))

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.WARNING)

    logging.getLogger(PREFIX).setLevel(level.upper())
    for name in NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Logger for one module, e.g. get_logger("capture") -> "recorder.capture"."""
    return logging.getLogger(f"{PREFIX}.{name}")
