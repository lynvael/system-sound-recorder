"""Tests for app/gui/prefs.py (QSettings persistence of the device choice
and the summarization options).

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

# The ADR-002 backend API may not be landed yet (it is written in parallel);
# the `resolve_options` tests below are skipped until it is importable.
try:
    from app.summarize import DEFAULT_STRATEGY_ID, SummarizationOptions  # noqa: F401

    _SUMMARIZE_BACKEND_READY = True
except ImportError:
    _SUMMARIZE_BACKEND_READY = False

needs_summarize_backend = pytest.mark.skipif(
    not _SUMMARIZE_BACKEND_READY,
    reason="app.summarize ADR-002 API not landed yet",
)


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


# --- summarization prefs -----------------------------------------------------

def test_summarize_prefs_defaults_without_file(ini_store: Path) -> None:
    assert prefs.load_summarize_prefs() == prefs.SummarizePrefs()


def test_summarize_prefs_roundtrip(ini_store: Path) -> None:
    p = prefs.SummarizePrefs(
        strategy="eacss",
        use_default_prompt=False,
        custom_prompt=(
            "Раздел «Риски»:\n"
            "- пункт = 1; и ещё\n"
            "# не комментарий, а часть текста\n"
        ),
    )
    prefs.save_summarize_prefs(p)
    assert prefs.load_summarize_prefs() == p


def test_summarize_prefs_roundtrip_persists_to_disk(ini_store: Path) -> None:
    prefs.save_summarize_prefs(
        prefs.SummarizePrefs(strategy="map_reduce", use_default_prompt=True)
    )
    # Qt's INI writer renders "summarize/strategy" as section [summarize].
    content = ini_store.read_text(encoding="utf-8")
    assert "[summarize]" in content
    assert "strategy=map_reduce" in content
    assert "use_default_prompt=true" in content


def test_summarize_prefs_partial_values(ini_store: Path) -> None:
    _ini_settings(ini_store).setValue("summarize/strategy", "hierarchical")
    loaded = prefs.load_summarize_prefs()
    assert loaded.strategy == "hierarchical"
    assert loaded.use_default_prompt is True  # missing key -> default
    assert loaded.custom_prompt == ""


def test_summarize_use_default_true_keeps_custom_text(ini_store: Path) -> None:
    prefs.save_summarize_prefs(
        prefs.SummarizePrefs(strategy=None, use_default_prompt=False, custom_prompt="мой текст")
    )
    # Toggling back to "use default" must NOT erase the stored text.
    prefs.save_summarize_prefs(
        prefs.SummarizePrefs(strategy=None, use_default_prompt=True, custom_prompt="мой текст")
    )
    loaded = prefs.load_summarize_prefs()
    assert loaded.use_default_prompt is True
    assert loaded.custom_prompt == "мой текст"


def test_summarize_bool_string_parsing(ini_store: Path) -> None:
    # QSettings renders stored bools as "true"/"false" strings in INI
    # (and sometimes in the Windows registry) -- both must parse.
    settings = _ini_settings(ini_store)
    settings.setValue("summarize/use_default_prompt", "true")
    assert prefs.load_summarize_prefs().use_default_prompt is True
    settings.setValue("summarize/use_default_prompt", "false")
    assert prefs.load_summarize_prefs().use_default_prompt is False


def test_summarize_bool_real_bool_parsing(ini_store: Path) -> None:
    settings = _ini_settings(ini_store)
    settings.setValue("summarize/use_default_prompt", False)
    assert prefs.load_summarize_prefs().use_default_prompt is False
    settings.setValue("summarize/use_default_prompt", True)
    assert prefs.load_summarize_prefs().use_default_prompt is True


def test_summarize_bool_garbage_defaults_true(ini_store: Path) -> None:
    settings = _ini_settings(ini_store)
    settings.setValue("summarize/use_default_prompt", "да")  # unrecognized
    assert prefs.load_summarize_prefs().use_default_prompt is True
    settings.setValue("summarize/use_default_prompt", 42)  # wrong type
    assert prefs.load_summarize_prefs().use_default_prompt is True


def test_summarize_corrupt_values_fall_back(ini_store: Path) -> None:
    settings = _ini_settings(ini_store)
    settings.setValue("summarize/strategy", 42)  # wrong type -> None
    settings.setValue("summarize/custom_prompt", ["a", "b"])  # wrong type -> ""
    loaded = prefs.load_summarize_prefs()
    assert loaded.strategy is None
    assert loaded.custom_prompt == ""


def test_summarize_unknown_strategy_preserved(ini_store: Path) -> None:
    # prefs is domain-agnostic: an unknown id is stored and returned as-is;
    # the fallback to DEFAULT_STRATEGY_ID is the GUI's/resolve_options' job.
    prefs.save_summarize_prefs(prefs.SummarizePrefs(strategy="no_such_method"))
    assert prefs.load_summarize_prefs().strategy == "no_such_method"


def test_summarize_prefs_never_raise_when_store_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ExplodingQSettings:
        def __init__(self, *args, **kwargs):
            raise OSError("simulated: settings store unavailable")

    monkeypatch.setattr(prefs, "QSettings", _ExplodingQSettings)
    prefs.save_summarize_prefs(
        prefs.SummarizePrefs(strategy="eacss", custom_prompt="текст")
    )  # must not raise
    assert prefs.load_summarize_prefs() == prefs.SummarizePrefs()  # must not raise


# --- resolve_options (pure helper; needs the ADR-002 backend API) -------------

@needs_summarize_backend
def test_resolve_options_use_default_drops_text() -> None:
    opts = prefs.resolve_options("map_reduce", True, "мой текст")
    assert opts.strategy == "map_reduce"
    assert opts.custom_prompt is None  # backend uses its built-in protocol


@needs_summarize_backend
def test_resolve_options_custom_prompt_passed_verbatim() -> None:
    text = "Разделы:\n1. TL;DR\n2. Задачи {с фигурными скобками}"
    opts = prefs.resolve_options("map_reduce", False, text)
    assert opts.custom_prompt == text


@needs_summarize_backend
def test_resolve_options_unknown_strategy_falls_back() -> None:
    assert prefs.resolve_options("no_such_method", True, "").strategy == DEFAULT_STRATEGY_ID
    assert prefs.resolve_options(None, True, "").strategy == DEFAULT_STRATEGY_ID


@needs_summarize_backend
def test_resolve_options_blank_custom_prompt_stays_blank() -> None:
    # Not None: the GUI refuses to start with a blank custom prompt, and the
    # backend guard is the second line of defense.
    opts = prefs.resolve_options("map_reduce", False, "   ")
    assert opts.custom_prompt == "   "
