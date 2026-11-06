"""Tests for app/audio/backend.py (the lazy shared PyAudio() singleton).

The real `pyaudiowpatch` is Windows-only, so these tests install a fake
module into `sys.modules` and assert on the singleton/shutdown semantics
only — no audio hardware is touched.
"""

from __future__ import annotations

import sys
import types

import pytest

from app.audio import backend


@pytest.fixture()
def reset_backend(monkeypatch: pytest.MonkeyPatch):
    """Reset the module-level singleton before each test."""
    monkeypatch.setattr(backend, "_instance", None)
    yield


class _FakePyAudio:
    """Records construction/termination; one class per test run."""

    def __init__(self) -> None:
        self.terminated = 0
        type(self).instances.append(self)

    def terminate(self) -> None:
        self.terminated += 1

    instances: list = []


def _install_fake(monkeypatch: pytest.MonkeyPatch, pyaudio_cls) -> None:
    _FakePyAudio.instances = []
    mod = types.ModuleType("pyaudiowpatch")
    mod.PyAudio = pyaudio_cls
    monkeypatch.setitem(sys.modules, "pyaudiowpatch", mod)


def test_get_backend_is_singleton(monkeypatch, reset_backend):
    _install_fake(monkeypatch, _FakePyAudio)
    b1 = backend.get_backend()
    b2 = backend.get_backend()
    assert b1 is b2
    assert len(_FakePyAudio.instances) == 1  # created exactly once


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="real pyaudiowpatch is installed on Windows; the import would "
    "succeed and create a real PyAudio() (real Pa_Initialize/COM)",
)
def test_get_backend_raises_on_non_windows(monkeypatch, reset_backend):
    # No pyaudiowpatch in sys.modules and not installed on this box.
    monkeypatch.delitem(sys.modules, "pyaudiowpatch", raising=False)
    with pytest.raises(RuntimeError, match="PyAudioWPatch доступен только на Windows"):
        backend.get_backend()
    assert backend._instance is None


def test_init_failure_propagates_and_is_not_cached(monkeypatch, reset_backend):
    class _Boom:
        def __init__(self):
            raise OSError(-9999, "Unanticipated host error")

    _install_fake(monkeypatch, _Boom)
    with pytest.raises(OSError):
        backend.get_backend()
    assert backend._instance is None  # a failed init must not be cached


def test_shutdown_idempotent(monkeypatch, reset_backend):
    _install_fake(monkeypatch, _FakePyAudio)
    b = backend.get_backend()
    backend.shutdown()
    backend.shutdown()  # second call: no-op, no error
    assert b.terminated == 1  # terminate() called exactly once
    assert backend._instance is None


def test_shutdown_before_init_is_noop(reset_backend):
    backend.shutdown()  # must not raise


def test_get_backend_reinitializes_after_shutdown(monkeypatch, reset_backend):
    _install_fake(monkeypatch, _FakePyAudio)
    b1 = backend.get_backend()
    backend.shutdown()
    b2 = backend.get_backend()
    assert b2 is not b1
    assert len(_FakePyAudio.instances) == 2
