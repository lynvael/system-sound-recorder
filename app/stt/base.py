"""Common STT engine interface.

Both engines take a 16 kHz mono float32 numpy array and return recognized text.
`language` is a whisper-style label ("russian" / "english" / None for auto);
engines that support only one language (GigaAM) ignore it.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ASREngine(Protocol):
    def transcribe(self, audio: np.ndarray, language: str | None) -> str: ...
