"""QSettings persistence for the GUI device selection.

Remembers the user's last microphone / system-audio (loopback) device
choice so the next launch can preselect it. Keys: ``devices/mic`` and
``devices/loopback``. Uses the native format (on Windows:
``HKCU\\Software\\LiveRecorder\\LiveRecorder``).

GUI-layer only: imports PySide6, but has no dependency on MainWindow or
the audio backend, so it stays unit-testable in isolation. Both public
functions are total -- they never raise.
"""

from __future__ import annotations

from PySide6.QtCore import QSettings

from app.log import get_logger

logger = get_logger("gui.prefs")

ORGANIZATION = "LiveRecorder"
APPLICATION = "LiveRecorder"

_MIC_KEY = "devices/mic"
_LOOPBACK_KEY = "devices/loopback"


def _settings() -> QSettings:
    return QSettings(ORGANIZATION, APPLICATION)


def _read_name(settings: QSettings, key: str) -> str | None:
    """A single stored device name.

    None when the key is missing, empty, or holds a value of the wrong
    type (corruption) -- a device name is always stored as a plain string.
    Note: ``settings.value(key, type=str)`` is deliberately NOT used -- it
    would coerce a corrupt non-string (e.g. an int) into its string form
    instead of rejecting it.
    """
    value = settings.value(key)
    if not isinstance(value, str) or not value:
        return None
    return value


def load_device_prefs() -> tuple[str | None, str | None]:
    """Return ``(saved mic name, saved loopback name)``.

    ``(None, None)`` when the keys are missing, empty, or corrupted, or
    when QSettings itself fails. Never raises.
    """
    try:
        settings = _settings()
        return (_read_name(settings, _MIC_KEY), _read_name(settings, _LOOPBACK_KEY))
    except Exception as exc:  # noqa: BLE001 - must never break the GUI
        logger.warning("Не удалось прочитать сохранённые устройства: %s", exc)
        return (None, None)


def save_device_prefs(mic_name: str, loopback_name: str) -> None:
    """Persist both device names. Never raises."""
    try:
        settings = _settings()
        settings.setValue(_MIC_KEY, mic_name)
        settings.setValue(_LOOPBACK_KEY, loopback_name)
        # PySide6's sync() returns None (not a bool) on success, so only an
        # explicit False counts as a failure.
        if settings.sync() is False:
            logger.warning(
                "Не удалось записать сохранённые устройства (sync отклонён)"
            )
    except Exception as exc:  # noqa: BLE001 - must never break the GUI
        logger.warning("Не удалось сохранить выбор устройств: %s", exc)
