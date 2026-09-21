"""Tests for app/gui/prefs.py (QSettings persistence of the device choice).

QSettings is redirected to a temporary store so the tests never touch the
developer's real settings: ``prefs.QSettings`` is monkeypatched to a
factory producing ``QSettings(<tmp_path>/LiveRecorder.ini, IniFormat)``.

The file-name-based constructor is used deliberately: the
``(IniFormat, UserScope)`` variant resolves its path via QStandardPaths,
which Qt CACHES after the first QSettings construction in the process
(verified on the dev box: changing ``XDG_CONFIG_HOME`` between tests is
ignored, so per-test tmp dirs would leak across tests). The file-name
constructor is per-instance and deterministic on every platform.

Only ``app.gui.prefs`` is imported here (PySide6 is a base dependency);
``app.gui.main_window`` is deliberately not exercised -- it needs a
QApplication.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QSettings

from app.gui import prefs


def _ini_settings(ini_path: Path) -> QSettings:
    return QSettings(str(ini_path), QSettings.Format.IniFormat)


@pytest.fixture()
def ini_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect prefs' QSettings to an ini file inside tmp_path."""
    ini_path = tmp_path / "LiveRecorder.ini"
    monkeypatch.setattr(
        prefs, "QSettings", lambda *a, **k: _ini_settings(ini_path)
    )
    return ini_path


# --- roundtrip ---------------------------------------------------------------

def test_save_load_roundtrip(ini_store: Path) -> None:
    prefs.save_device_prefs("Jabra Evolve2 30 SE", "Динамики [Loopback]")
    assert prefs.load_device_prefs() == ("Jabra Evolve2 30 SE", "Динамики [Loopback]")


def test_roundtrip_persists_to_disk(ini_store: Path) -> None:
    prefs.save_device_prefs("Mic", "Loop")
    assert ini_store.exists()
    # Qt's INI writer renders "devices/mic" as section [devices], key mic.
    content = ini_store.read_text(encoding="utf-8")
    assert "[devices]" in content
    assert "mic=Mic" in content and "loopback=Loop" in content


def test_load_partial_values(ini_store: Path) -> None:
    _ini_settings(ini_store).setValue("devices/mic", "Mic A")
    assert prefs.load_device_prefs() == ("Mic A", None)


# --- missing / empty / corrupted -> (None, None) ------------------------------

def test_load_without_file_returns_none(ini_store: Path) -> None:
    assert prefs.load_device_prefs() == (None, None)


def test_load_without_keys_returns_none(ini_store: Path) -> None:
    _ini_settings(ini_store).setValue("unrelated/key", "value")
    assert prefs.load_device_prefs() == (None, None)


def test_load_corrupt_values_returns_none(ini_store: Path) -> None:
    settings = _ini_settings(ini_store)
    settings.setValue("devices/mic", 42)  # wrong type (int)
    settings.setValue("devices/loopback", ["a", "b"])  # wrong type (list)
    assert prefs.load_device_prefs() == (None, None)


def test_load_empty_string_returns_none(ini_store: Path) -> None:
    settings = _ini_settings(ini_store)
    settings.setValue("devices/mic", "")
    settings.setValue("devices/loopback", "")
    assert prefs.load_device_prefs() == (None, None)


def test_load_garbage_file_returns_none(ini_store: Path) -> None:
    prefs.save_device_prefs("Mic", "Loop")
    ini_store.write_text("not [valid ini\n\x00garbage=]]\n", encoding="utf-8")
    assert prefs.load_device_prefs() == (None, None)


# --- never-raise ---------------------------------------------------------------

def test_save_never_raises_when_store_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    class _ExplodingQSettings:
        def __init__(self, *args, **kwargs):
            raise OSError("simulated: settings store unavailable")

    monkeypatch.setattr(prefs, "QSettings", _ExplodingQSettings)
    prefs.save_device_prefs("Mic", "Loop")  # must not raise
    assert prefs.load_device_prefs() == (None, None)  # must not raise


def test_save_writes_documented_keys_and_tolerates_sync_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[_FakeSettings] = []

    class _FakeSettings:
        def __init__(self, *args, **kwargs):
            self.values: dict = {}
            created.append(self)

        def setValue(self, key: str, value: object) -> None:
            self.values[key] = value

        def sync(self):
            return False  # force the failure path (PySide6 returns None on success)

    monkeypatch.setattr(prefs, "QSettings", _FakeSettings)
    prefs.save_device_prefs("Mic A", "Loop B")  # must not raise
    assert created, "QSettings was never constructed"
    assert created[-1].values == {"devices/mic": "Mic A", "devices/loopback": "Loop B"}
