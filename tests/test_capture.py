"""Tests for app/audio/capture.py: CaptureThread with a fake PyAudioWPatch.

No audio hardware: `backend.get_backend()` and `capture.get_device` are
monkeypatched with fakes, and `CaptureThread.run()` is called synchronously
(the fake stream never blocks). The conftest installs a `pyaudiowpatch` stub
on non-Windows, so the `paFloat32`/error-code constants resolve to the real
PortAudio v19 values.
"""

from __future__ import annotations

import queue
import threading

import numpy as np
import pytest

import pyaudiowpatch as pyaudio  # conftest stub on Linux, real module on Windows

from app.audio import backend, capture
from app.audio.capture import CaptureThread, _downmix_mono


# --- fakes -------------------------------------------------------------------

class FakeStream:
    """Fake PyAudioWPatch stream: serves prepared `read()` results.

    `blocks` is a list of read() results: bytes, (bytes, n), or an Exception
    to raise. When the list is exhausted, read() sets the capture thread's
    stop event and returns empty data, so run() exits the loop cleanly.
    """

    def __init__(self, blocks: list, stop_event: threading.Event) -> None:
        self._blocks = list(blocks)
        self._stop_event = stop_event
        self.closed = False
        self.read_calls: list[int] = []

    def read(self, num_frames, exception_on_overflow=True):
        self.read_calls.append(num_frames)
        if self._blocks:
            item = self._blocks.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        self._stop_event.set()
        return b""

    def close(self) -> None:
        self.closed = True


class FakePyAudio:
    """Fake PyAudio() singleton: open() returns the prepared stream."""

    def __init__(self, stream: FakeStream, open_error: Exception | None = None):
        self._stream = stream
        self.open_error = open_error
        self.open_kwargs: dict | None = None

    def open(self, **kwargs):
        self.open_kwargs = kwargs
        if self.open_error is not None:
            raise self.open_error
        return self._stream


@pytest.fixture()
def fake_backend(monkeypatch: pytest.MonkeyPatch):
    """Wire `backend.get_backend()` and `capture.get_device` to fakes.

    `expect_loopback` is the value `CaptureThread.run()` is expected to pass
    through (invariant #1 wiring): a mismatch fails the test loudly instead
    of silently resolving the wrong endpoint kind.
    """

    def _install(stream: FakeStream, device: dict,
                 open_error: Exception | None = None,
                 expect_loopback: bool = False) -> FakePyAudio:
        fake_p = FakePyAudio(stream, open_error)
        monkeypatch.setattr(backend, "get_backend", lambda: fake_p)
        expected = expect_loopback

        def _get_device(device_id, *, expect_loopback):
            if expect_loopback is not expected:
                raise AssertionError(
                    f"CaptureThread passed expect_loopback={expect_loopback!r}, "
                    f"expected {expected!r}"
                )
            return device

        monkeypatch.setattr(capture, "get_device", _get_device)
        return fake_p

    return _install


def _device(name: str = "Jabra", *, loopback: bool = False, in_ch: int = 2,
            rate: int = 16000, index: int = 3) -> dict:
    return {
        "index": index,
        "name": name,
        "maxInputChannels": in_ch,
        "defaultSampleRate": rate,
        "isLoopbackDevice": loopback,
    }


def _float32_bytes(frames: int, channels: int = 1,
                   value: float = 0.25) -> bytes:
    return np.full(frames * channels, value, dtype=np.float32).tobytes()


def _new_thread(q: "queue.Queue") -> CaptureThread:
    return CaptureThread(
        "Jabra",
        q,
        frame_size=512,
        target_sample_rate=16000,
        expect_loopback=False,
        chunk_frames=1024,
        name="t",
    )


def _drain(q: "queue.Queue") -> list[np.ndarray]:
    """Collect frames until the None sentinel; assert it is present."""
    frames: list[np.ndarray] = []
    while True:
        item = q.get()
        if item is None:
            return frames
        frames.append(item)


# --- _downmix_mono -------------------------------------------------------------

def test_downmix_mono_1d_passthrough():
    x = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    out = _downmix_mono(x)
    assert out.shape == (3,)
    assert out.dtype == np.float32
    assert np.array_equal(out, x)


def test_downmix_mono_stereo_mean():
    x = np.array([[0.0, 0.5], [0.2, 0.6]], dtype=np.float32)
    out = _downmix_mono(x)
    assert out.shape == (2,)
    assert out.dtype == np.float32
    assert np.allclose(out, [0.25, 0.4])


# --- CaptureThread.run() -------------------------------------------------------

def test_run_emits_exact_frames_and_sentinel(fake_backend):
    # native == target (16 kHz): the resampler is a pass-through, so the
    # output is exactly the input sample count.
    q: "queue.Queue" = queue.Queue()
    ct = _new_thread(q)
    stream = FakeStream([_float32_bytes(1024, 1)] * 4, ct._stop_event)
    fake_backend(stream, _device(rate=16000, in_ch=1))

    ct.run()  # synchronous: the fake stream never blocks

    assert ct.error is None
    assert ct.resolved.is_set()
    frames = _drain(q)
    assert len(frames) == 4096 // 512
    assert all(f.shape == (512,) and f.dtype == np.float32 for f in frames)
    assert stream.closed
    # 4 prepared blocks + 1 final read (the fake's "block until stop").
    assert len(stream.read_calls) == 5
    assert all(c == 1024 for c in stream.read_calls)


def test_run_downmixes_stereo_to_mono(fake_backend):
    q = queue.Queue()
    ct = _new_thread(q)
    l = np.full(1024, 0.5, dtype=np.float32)
    r = np.full(1024, 0.1, dtype=np.float32)
    stereo = np.stack([l, r], axis=1).tobytes()
    stream = FakeStream([stereo], ct._stop_event)
    fake_backend(stream, _device(rate=16000, in_ch=2))

    ct.run()

    frames = _drain(q)
    total = np.concatenate(frames)
    assert len(total) == 1024
    assert np.allclose(total, 0.3, atol=1e-6)


def test_run_pads_partial_tail_frame(fake_backend):
    # 1500 samples = 2*512 + 476 -> 3 frames, the last padded with 36 zeros.
    q = queue.Queue()
    ct = _new_thread(q)
    stream = FakeStream(
        [_float32_bytes(1024, 1), _float32_bytes(476, 1)], ct._stop_event
    )
    fake_backend(stream, _device(rate=16000, in_ch=1))

    ct.run()

    frames = _drain(q)
    assert len(frames) == 3
    assert all(f.shape == (512,) for f in frames)
    assert np.all(frames[2][-36:] == 0)


def test_run_accepts_tuple_read_result(fake_backend):
    # A future fork may return (bytes, n) instead of bytes; both must work.
    q = queue.Queue()
    ct = _new_thread(q)
    data = _float32_bytes(1024, 1)
    stream = FakeStream([(data, 1024)] * 2, ct._stop_event)
    fake_backend(stream, _device(rate=16000, in_ch=1))

    ct.run()

    assert ct.error is None
    frames = _drain(q)
    assert len(frames) == 2048 // 512


def test_run_sets_error_on_read_failure(fake_backend):
    # Hot-unplug: OSError on read() -> .error (mapped Russian phrase) + the
    # None sentinel is still emitted so feeders drain.
    q = queue.Queue()
    ct = _new_thread(q)
    err = OSError(pyaudio.paDeviceUnavailable, "Device unavailable")
    stream = FakeStream([_float32_bytes(1024, 1), err], ct._stop_event)
    fake_backend(stream, _device(rate=16000, in_ch=1))

    ct.run()

    assert ct.error is not None
    assert isinstance(ct.error, RuntimeError)
    assert "устройство отключено" in str(ct.error)
    assert ct.resolved.is_set()  # device resolved OK; the stream died later
    _drain(q)  # sentinel present
    assert stream.closed


def test_run_unmapped_error_keeps_original_message(fake_backend):
    q = queue.Queue()
    ct = _new_thread(q)
    err = OSError(-12345, "some exotic portaudio failure")
    stream = FakeStream([err], ct._stop_event)
    fake_backend(stream, _device(rate=16000, in_ch=1))

    ct.run()

    assert ct.error is not None
    # Unmapped code: the original message is preserved, not swallowed.
    assert "some exotic portaudio failure" in str(ct.error)
    _drain(q)


def test_run_sets_error_on_open_failure(fake_backend):
    q = queue.Queue()
    ct = _new_thread(q)
    err = OSError(pyaudio.paInvalidSampleRate, "Invalid sample rate")
    stream = FakeStream([], ct._stop_event)
    fake_backend(stream, _device(rate=44100, in_ch=1), open_error=err)

    ct.run()

    assert ct.error is not None
    assert "частоту дискретизации" in str(ct.error)
    # resolved == "device resolved+validated" (set before the stream is
    # opened), same semantics as pre-migration: an open() failure is a
    # partial failure handled at stop time, not an all-dead start.
    assert ct.resolved.is_set()
    assert q.get() is None  # sentinel still emitted


def test_open_kwargs(fake_backend):
    q = queue.Queue()
    ct = _new_thread(q)
    stream = FakeStream([_float32_bytes(1024, 2)], ct._stop_event)
    fake_p = fake_backend(stream, _device(rate=48000, in_ch=2, index=7))

    ct.run()

    assert fake_p.open_kwargs == {
        "format": pyaudio.paFloat32,
        "channels": 2,
        "rate": 48000,
        "input": True,
        "input_device_index": 7,
        "frames_per_buffer": 1024,
    }


# --- native rate: per-device defaultSampleRate + fallback ---------------------

def _spy_resampler(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    seen: list[tuple[int, int]] = []
    real = capture.StreamingResampler

    class SpyResampler:
        def __init__(self, native, target):
            seen.append((native, target))
            self._inner = real(native, target)

        def resample_chunk(self, x, last=False):
            return self._inner.resample_chunk(x, last)

        def flush(self):
            return self._inner.flush()

    monkeypatch.setattr(capture, "StreamingResampler", SpyResampler)
    return seen


def test_native_rate_from_device(fake_backend, monkeypatch):
    seen = _spy_resampler(monkeypatch)
    q = queue.Queue()
    ct = _new_thread(q)
    stream = FakeStream([_float32_bytes(1024, 1)], ct._stop_event)
    fake_backend(stream, _device(rate=44100, in_ch=1))

    ct.run()

    assert seen == [(44100, 16000)]


def test_native_rate_fallback_when_zero(fake_backend, monkeypatch):
    seen = _spy_resampler(monkeypatch)
    q = queue.Queue()
    ct = _new_thread(q)
    stream = FakeStream([_float32_bytes(1024, 1)], ct._stop_event)
    fake_backend(stream, _device(rate=0, in_ch=1))

    ct.run()

    assert seen == [(capture.DEFAULT_NATIVE_RATE, 16000)]


def test_native_rate_fallback_when_missing(fake_backend, monkeypatch):
    seen = _spy_resampler(monkeypatch)
    q = queue.Queue()
    ct = _new_thread(q)
    stream = FakeStream([_float32_bytes(1024, 1)], ct._stop_event)
    dev = _device(rate=48000, in_ch=1)
    del dev["defaultSampleRate"]
    fake_backend(stream, dev)

    ct.run()

    assert seen == [(capture.DEFAULT_NATIVE_RATE, 16000)]
