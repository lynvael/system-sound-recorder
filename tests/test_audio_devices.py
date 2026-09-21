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

    def __init__(self, devs: list[dict], *,
                 default_input: dict | None = None,
                 default_output: dict | None = None) -> None:
        self._devs = devs
        self._default_input = default_input
        self._default_output = default_output

    def get_device_info_generator(self):
        yield from self._devs

    def get_default_input_device_info(self):
        return self._default_input

    def get_default_output_device_info(self):
        return self._default_output


@pytest.fixture()
def fake_backend(monkeypatch: pytest.MonkeyPatch):
    """Point `backend.get_backend()` at a fake; call with a device list."""

    def _install(devs: list[dict], *,
                 default_input: dict | None = None,
                 default_output: dict | None = None) -> _FakeBackend:
        fake = _FakeBackend(devs, default_input=default_input,
                            default_output=default_output)
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


# --- default_device_names ----------------------------------------------------

def test_default_device_names_both_present(fake_backend):
    # Loopback name is derived from the default OUTPUT name ("Speakers" ->
    # "Speakers [Loopback]"), not read from the input device list.
    fake_backend(
        [
            _dev("Jabra Evolve2 30 SE", loopback=False, index=0),
            _dev("Speakers [Loopback]", loopback=True, index=1),
        ],
        default_input=_dev("Jabra Evolve2 30 SE", loopback=False, index=0),
        default_output=_dev("Speakers", in_ch=0, index=2),
    )
    assert devices.default_device_names() == (
        "Jabra Evolve2 30 SE",
        "Speakers [Loopback]",
    )


def test_default_device_names_no_default_input(fake_backend):
    fake_backend(
        [
            _dev("Jabra", loopback=False, index=0),
            _dev("Speakers [Loopback]", loopback=True, index=1),
        ],
        default_input=None,
        default_output=_dev("Speakers", in_ch=0, index=2),
    )
    assert devices.default_device_names() == (None, "Speakers [Loopback]")


def test_default_device_names_default_input_is_loopback(fake_backend):
    # A loopback endpoint as the default input is NOT preselected as the mic
    # (invariant #1): mic_name stays None.
    fake_backend(
        [
            _dev("Jabra", loopback=False, index=0),
            _dev("Speakers [Loopback]", loopback=True, index=1),
        ],
        default_input=_dev("Speakers [Loopback]", loopback=True, index=1),
        default_output=_dev("Speakers", in_ch=0, index=2),
    )
    assert devices.default_device_names() == (None, "Speakers [Loopback]")


def test_default_device_names_default_input_without_input_channels(fake_backend):
    # A default input that cannot capture (maxInputChannels == 0) is not a
    # usable mic either.
    fake_backend(
        [_dev("Jabra", loopback=False, index=0)],
        default_input=_dev("Weird", loopback=False, in_ch=0, index=1),
        default_output=None,
    )
    assert devices.default_device_names() == (None, None)


def test_default_device_names_no_matching_loopback(fake_backend):
    # Default output exists but its "[Loopback]" endpoint is absent -> no
    # fuzzy fallback onto a different loopback; loopback_name stays None.
    fake_backend(
        [
            _dev("Jabra", loopback=False, index=0),
            _dev("HDMI [Loopback]", loopback=True, index=1),
        ],
        default_input=_dev("Jabra", loopback=False, index=0),
        default_output=_dev("Speakers", in_ch=0, index=2),
    )
    assert devices.default_device_names() == ("Jabra", None)


def test_default_device_names_loopback_without_input_channels(fake_backend):
    # The matching loopback endpoint exists by name but cannot capture
    # (filtered out of list_loopbacks) -> loopback_name stays None.
    fake_backend(
        [
            _dev("Jabra", loopback=False, index=0),
            _dev("Speakers [Loopback]", loopback=True, in_ch=0, index=1),
        ],
        default_input=_dev("Jabra", loopback=False, index=0),
        default_output=_dev("Speakers", in_ch=0, index=2),
    )
    assert devices.default_device_names() == ("Jabra", None)


def test_default_device_names_missing_methods_api_drift(monkeypatch):
    # A backend without the get_default_*_device_info methods (API drift):
    # (None, None), no exception.
    class _DriftedBackend:
        def get_device_info_generator(self):
            yield from []

    monkeypatch.setattr(backend, "get_backend", lambda: _DriftedBackend())
    assert devices.default_device_names() == (None, None)


def test_default_device_names_default_method_raises(monkeypatch):
    # A default-info call that raises is swallowed (soft UX feature); the
    # other slot still works.
    class _FlakyBackend:
        def get_device_info_generator(self):
            yield from []

        def get_default_input_device_info(self):
            raise RuntimeError("paUnanticipatedHostError")

        def get_default_output_device_info(self):
            return {"name": "Speakers"}

    monkeypatch.setattr(backend, "get_backend", lambda: _FlakyBackend())
    assert devices.default_device_names() == (None, None)


def test_default_device_names_propagates_backend_error(monkeypatch):
    # Only what get_backend() raises may propagate (e.g. non-Windows).
    def _raise():
        raise RuntimeError("PyAudioWPatch доступен только на Windows")

    monkeypatch.setattr(backend, "get_backend", _raise)
    with pytest.raises(RuntimeError, match="Windows"):
        devices.default_device_names()


# --- __main__ runner -----------------------------------------------------------

def test_main_runner_prints_lists_and_defaults(fake_backend, capsys):
    fake_backend(
        [
            _dev("Jabra", loopback=False, index=0),
            _dev("Realtek Mic", loopback=False, index=1),
            _dev("Speakers [Loopback]", loopback=True, index=2),
        ],
        default_input=_dev("Jabra", loopback=False, index=0),
        default_output=_dev("Speakers", in_ch=0, index=3),
    )
    devices._main()
    out = capsys.readouterr().out
    assert "=== Microphones ===" in out
    assert "Jabra  ← по умолчанию" in out
    assert "Realtek Mic\n" in out
    assert "=== Loopbacks (system audio) ===" in out
    assert "Speakers [Loopback]  ← по умолчанию" in out


def test_main_runner_backend_error(monkeypatch, capsys):
    # Non-Windows: backend creation fails -> stderr message, exit code 1.
    def _raise():
        raise RuntimeError("PyAudioWPatch доступен только на Windows")

    monkeypatch.setattr(backend, "get_backend", _raise)
    with pytest.raises(SystemExit) as excinfo:
        devices._main()
    assert excinfo.value.code == 1
    assert "Не удалось получить список устройств" in capsys.readouterr().err
