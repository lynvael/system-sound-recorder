"""Tests for app/audio/devices.py: enumeration filters + fail-fast resolution.

Device identity is the NAME (PortAudio exports no stable WASAPI endpoint id
to Python), so `get_device` matches on exact name + `isLoopbackDevice` flag
with NO fuzzy/substring fallbacks. The mic channel must never resolve a
loopback endpoint (invariant #1 of the ADR).
"""

from __future__ import annotations

import pytest

from app.audio import backend, devices


class _FakeBackend:
    """Stand-in for the PyAudio() singleton: a fixed device info dict list."""

    def __init__(self, devs: list[dict]) -> None:
        self._devs = devs

    def get_device_info_generator(self):
        yield from self._devs


@pytest.fixture()
def fake_backend(monkeypatch: pytest.MonkeyPatch):
    """Point `backend.get_backend()` at a fake; call with a device list."""

    def _install(devs: list[dict]) -> _FakeBackend:
        fake = _FakeBackend(devs)
        monkeypatch.setattr(backend, "get_backend", lambda: fake)
        return fake

    return _install


def _dev(name: str, *, loopback: bool = False, in_ch: int = 2,
         rate: int = 48000, index: int = 0) -> dict:
    return {
        "index": index,
        "name": name,
        "maxInputChannels": in_ch,
        "maxOutputChannels": 0 if loopback else 2,
        "defaultSampleRate": rate,
        "isLoopbackDevice": loopback,
    }


# --- enumeration -------------------------------------------------------------

def test_list_microphones_filters(fake_backend):
    fake_backend([
        _dev("Jabra Evolve2 30 SE", loopback=False, index=0),
        _dev("Realtek Mic", loopback=False, index=1),
        _dev("Speakers [Loopback]", loopback=True, index=2),
        _dev("Output-only", loopback=False, in_ch=0, index=3),
    ])
    assert devices.list_microphones() == [
        ("Jabra Evolve2 30 SE", "Jabra Evolve2 30 SE"),
        ("Realtek Mic", "Realtek Mic"),
    ]


def test_list_loopbacks_filters(fake_backend):
    fake_backend([
        _dev("Jabra", loopback=False, index=0),
        _dev("Speakers [Loopback]", loopback=True, index=1),
        _dev("HDMI [Loopback]", loopback=True, in_ch=0, index=2),
    ])
    assert devices.list_loopbacks() == [
        ("Speakers [Loopback]", "Speakers [Loopback]"),
    ]


# --- resolution: success / not found / ambiguous -----------------------------

def test_get_device_exact_match(fake_backend):
    fake_backend([
        _dev("Jabra", loopback=False, index=0),
        _dev("Jabra [Loopback]", loopback=True, index=1),
    ])
    dev = devices.get_device("Jabra", expect_loopback=False)
    assert dev["index"] == 0
    assert dev["isLoopbackDevice"] is False

    dev = devices.get_device("Jabra [Loopback]", expect_loopback=True)
    assert dev["index"] == 1
    assert dev["isLoopbackDevice"] is True


def test_get_device_not_found(fake_backend):
    fake_backend([_dev("Jabra", loopback=False, index=0)])
    with pytest.raises(RuntimeError, match="не найдено"):
        devices.get_device("Nope", expect_loopback=False)
    # Name exists but the flag doesn't match — still "not found", never a
    # fuzzy fallback onto the other endpoint kind.
    with pytest.raises(RuntimeError, match="не найдено"):
        devices.get_device("Jabra", expect_loopback=True)


def test_get_device_ambiguous(fake_backend):
    fake_backend([
        _dev("Dup", loopback=False, index=0),
        _dev("Dup", loopback=False, index=1),
    ])
    with pytest.raises(RuntimeError, match="неоднозначно"):
        devices.get_device("Dup", expect_loopback=False)


def test_get_device_ignores_output_only(fake_backend):
    fake_backend([_dev("Out", loopback=False, in_ch=0, index=0)])
    with pytest.raises(RuntimeError, match="не найдено"):
        devices.get_device("Out", expect_loopback=False)


# --- invariant #1: mic never resolves a loopback, and vice versa -------------

def test_mic_channel_never_resolves_loopback(fake_backend):
    # A loopback endpoint with the SAME name as the mic: the mic channel must
    # resolve only the real mic, and the loopback channel only the loopback.
    fake_backend([
        _dev("Jabra", loopback=True, index=0),
        _dev("Jabra", loopback=False, index=1),
    ])
    dev = devices.get_device("Jabra", expect_loopback=False)
    assert dev["index"] == 1
    assert dev["isLoopbackDevice"] is False

    dev = devices.get_device("Jabra", expect_loopback=True)
    assert dev["index"] == 0
    assert dev["isLoopbackDevice"] is True
