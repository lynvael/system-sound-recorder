"""Stream a WAV (or other libsndfile-readable) file from disk in blocks.

Used by batch and file modes so an hour of audio is never loaded into RAM at
once. For a stereo session file, `channel=0/1` extracts one track (0 = mic /
"Я", 1 = loopback / "Собеседники") so each channel goes to its own VAD with
attribution preserved. For a mono/foreign file use `channel=None`.

If the file's sample rate differs from `target_sample_rate`, blocks are
streaming-resampled per channel on the way out.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import soundfile as sf

from app.audio.resample import StreamingResampler
from app.log import get_logger

logger = get_logger("wav_source")


def read_info(path: str | Path) -> sf._SoundFileInfo:
    """Return soundfile metadata (samplerate, channels, frames, ...)."""
    return sf.info(str(path))


def iter_blocks(
    path: str | Path,
    blocksize: int,
    *,
    channel: int | None = None,
    target_sample_rate: int | None = None,
) -> Iterator[np.ndarray]:
    """Yield mono float32 blocks from `path`.

    channel: 0/1 selects a track from a stereo file; None downmixes to mono
             (single-channel files pass through). Out-of-range channel raises.
    target_sample_rate: if set and the file rate differs, blocks are resampled.
    Blocks are `blocksize` samples except possibly the last; the resampler may
    make emitted block sizes vary, which downstream consumers (VAD) tolerate.
    """
    info = read_info(path)
    src_rate = info.samplerate
    resampler: StreamingResampler | None = None
    if target_sample_rate is not None and src_rate != target_sample_rate:
        resampler = StreamingResampler(src_rate, target_sample_rate)
        logger.debug("Resampling %s: %d -> %d", path, src_rate, target_sample_rate)

    with sf.SoundFile(str(path)) as f:
        n_channels = f.channels
        if channel is not None and channel >= n_channels:
            raise ValueError(
                f"Requested channel {channel} but file has {n_channels} channel(s)"
            )
        for block in f.blocks(blocksize=blocksize, dtype="float32", always_2d=True):
            if channel is not None:
                mono = block[:, channel]
            elif n_channels == 1:
                mono = block[:, 0]
            else:
                mono = block.mean(axis=1).astype(np.float32)
            if resampler is not None:
                mono = resampler.resample_chunk(mono)
            if mono.size:
                yield np.ascontiguousarray(mono, dtype=np.float32)

    if resampler is not None:
        tail = resampler.flush()
        if tail.size:
            yield np.ascontiguousarray(tail, dtype=np.float32)
