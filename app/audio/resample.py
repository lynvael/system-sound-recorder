"""Streaming resampler on top of soxr.ResampleStream.

Stateful: the filter state is carried between chunks (`resample_chunk(x)`), so
joins between chunks have no edge artifacts and no manual overlap logic is
needed. Use one instance per channel (native device rate -> target rate).

If in_rate == out_rate the resampler is a pass-through (soxr still works, but we
short-circuit to avoid needless work and any identity-transform quirks).
"""

from __future__ import annotations

import numpy as np
import soxr


class StreamingResampler:
    def __init__(self, in_rate: int, out_rate: int) -> None:
        self.in_rate = int(in_rate)
        self.out_rate = int(out_rate)
        self._passthrough = self.in_rate == self.out_rate
        if self._passthrough:
            self._stream = None
        else:
            self._stream = soxr.ResampleStream(
                self.in_rate, self.out_rate, 1, dtype="float32"
            )

    def resample_chunk(self, x: np.ndarray, last: bool = False) -> np.ndarray:
        """Resample one mono float32 chunk, keeping filter state between calls.

        `last=True` flushes the filter tail at end of stream. Returns a
        (possibly empty) 1-D float32 array at the output rate.
        """
        x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
        if self._passthrough:
            return x.copy()
        return self._stream.resample_chunk(x, last=last)

    def flush(self) -> np.ndarray:
        """Flush any remaining samples from the filter at end of stream."""
        empty = np.empty(0, dtype=np.float32)
        return self.resample_chunk(empty, last=True)
