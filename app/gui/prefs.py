"""QSettings persistence for GUI preferences.

Remembers the user's last microphone / system-audio (loopback) device
choice (keys ``devices/mic`` and ``devices/loopback``) and the
summarization options (keys ``summarize/strategy``,
``summarize/use_default_prompt``, ``summarize/custom_prompt``) so the
next launch can preselect them. Uses the native format (on Windows:
``HKCU\\Software\\LiveRecorder\\LiveRecorder``).

GUI-layer only: imports PySide6, but has no dependency on MainWindow or
the audio backend, so it stays unit-testable in isolation. All public
functions are total -- they never raise.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QSettings

from app.log import get_logger

logger = get_logger("gui.prefs")

ORGANIZATION = "LiveRecorder"
APPLICATION = "LiveRecorder"

_MIC_KEY = "devices/mic"
_LOOPBACK_KEY = "devices/loopback"

_STRATEGY_KEY = "summarize/strategy"
_USE_DEFAULT_PROMPT_KEY = "summarize/use_default_prompt"
_CUSTOM_PROMPT_KEY = "summarize/custom_prompt"


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


# --- summarization options ---------------------------------------------------


@dataclass
class SummarizePrefs:
    """Stored summarization options.

    ``strategy`` is the raw stored id (``None`` = not saved / corrupt);
    validating it against ``STRATEGIES`` is the GUI's job, not prefs' --
    prefs knows nothing about the domain. ``custom_prompt`` is ALWAYS
    preserved, even when ``use_default_prompt`` is True (toggling the
    checkbox only decides whether the text is used, never deletes it).
    """

    strategy: str | None = None
    use_default_prompt: bool = True
    custom_prompt: str = ""


def _read_bool(settings: QSettings, key: str) -> bool:
    """Tolerant bool read: missing / wrong-type / unrecognized value -> True.

    QSettings renders a stored bool as the string "true"/"false" in the
    INI format (and sometimes in the Windows registry), so both a real
    bool and those strings are accepted; anything else falls back to the
    default (True = use the built-in prompt).
    """
    value = settings.value(key)
    if value is True:
        return True
    if value is False:
        return False
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return True


def load_summarize_prefs() -> SummarizePrefs:
    """Return the stored summarization options.

    Defaults to ``SummarizePrefs()`` (``None`` / True / ``""``) when keys
    are missing, corrupted, or QSettings itself fails. Never raises.
    """
    try:
        settings = _settings()
        strategy = settings.value(_STRATEGY_KEY)
        strategy = strategy if isinstance(strategy, str) and strategy else None
        custom_prompt = settings.value(_CUSTOM_PROMPT_KEY)
        custom_prompt = custom_prompt if isinstance(custom_prompt, str) else ""
        return SummarizePrefs(
            strategy=strategy,
            use_default_prompt=_read_bool(settings, _USE_DEFAULT_PROMPT_KEY),
            custom_prompt=custom_prompt,
        )
    except Exception as exc:  # noqa: BLE001 - must never break the GUI
        logger.warning("Не удалось прочитать настройки саммаризации: %s", exc)
        return SummarizePrefs()


def save_summarize_prefs(p: SummarizePrefs) -> None:
    """Persist the summarization options. Never raises.

    ``custom_prompt`` is written verbatim on every save (multi-line text
    round-trips: the INI writer escapes ``\\n``, the registry stores
    REG_SZ), so saving with ``use_default_prompt=True`` never erases it.
    A ``None`` strategy is not written (there is nothing meaningful to
    store).
    """
    try:
        settings = _settings()
        if p.strategy is not None:
            settings.setValue(_STRATEGY_KEY, p.strategy)
        settings.setValue(_USE_DEFAULT_PROMPT_KEY, p.use_default_prompt)
        settings.setValue(_CUSTOM_PROMPT_KEY, p.custom_prompt)
        if settings.sync() is False:
            logger.warning(
                "Не удалось записать настройки саммаризации (sync отклонён)"
            )
    except Exception as exc:  # noqa: BLE001 - must never break the GUI
        logger.warning("Не удалось сохранить настройки саммаризации: %s", exc)


def resolve_options(
    strategy_id: str | None,
    use_default: bool,
    custom_text: str,
):
    """Pure GUI-state -> backend-options mapping (unit-tested, no Qt).

    - unknown / missing ``strategy_id`` -> ``DEFAULT_STRATEGY_ID``;
    - ``use_default`` -> ``custom_prompt=None`` (the backend then uses its
      built-in ``DEFAULT_REPORT_PROMPT``); the stored text stays in prefs
      either way -- this only decides what one run receives; the report
      output is Markdown for both default and custom prompts;
    - otherwise the text is passed through verbatim (even blank: the GUI
      refuses to start with a blank custom prompt, and the backend guard
      is the second line of defense).
    """
    # Local import: keeps app.summarize off this module's import-time
    # surface (prefs.py is exercised by unit tests that must not require
    # the LLM backend to be importable).
    from app.summarize import DEFAULT_STRATEGY_ID, STRATEGIES, SummarizationOptions

    strategy = (
        strategy_id
        if isinstance(strategy_id, str) and strategy_id in STRATEGIES
        else DEFAULT_STRATEGY_ID
    )
    custom_prompt = None if use_default else custom_text
    return SummarizationOptions(strategy=strategy, custom_prompt=custom_prompt)
