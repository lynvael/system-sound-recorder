"""Common STT engine interface.

An engine takes a 16 kHz mono float32 numpy array and returns recognized text.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ASREngine(Protocol):
    def transcribe(self, audio: np.ndarray) -> str: ...
