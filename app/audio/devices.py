"""Enumerate audio input devices and resolve them by name.

`PyAudioWPatch` (the PortAudio fork with WASAPI loopback) exposes both real
microphones and WASAPI loopback endpoints through its device list; loopback
endpoints are INPUT devices with `isLoopbackDevice == True` and the name
suffix " [Loopback]". We split them into two lists so the GUI can offer a
microphone selector and a system-audio (loopback) selector separately.

Device identity is the device NAME: PortAudio keeps the stable WASAPI
endpoint id in its C struct but does not export it to Python. Therefore the
`id` in the returned `(name, id)` pairs equals the name, and resolution
(`get_device`) matches on exact name + `isLoopbackDevice` flag, fail-fast,
with no fuzzy/substring fallbacks.

`default_device_names()` reports the current Windows system defaults (the
default input device, and the loopback endpoint of the default output
device) so the GUI/CLI can preselect them. It is defensive: a missing
default, a missing backend method (API drift), or a failed backend call
yields `None` for that slot — it never raises for those, and only
propagates what `backend.get_backend()` itself raises (e.g. the
non-Windows RuntimeError).

This module does no `pyaudiowpatch` import of its own (the `PyAudio()`
instance is created lazily in `backend.get_backend()`), so it imports
cleanly on non-Windows dev boxes.

Runnable for verification: `python -m app.audio.devices` prints both lists,
marking the current system defaults (mic / loopback).
"""

from __future__ import annotations

import sys

from app.audio import backend
from app.log import get_logger

logger = get_logger("devices")


def _iter_input_devices():
    """Yield info dicts of all devices that can capture (maxInputChannels > 0)."""
    p = backend.get_backend()
    for dev in p.get_device_info_generator():
        if int(dev.get("maxInputChannels") or 0) > 0:
            yield dev


def list_microphones() -> list[tuple[str, str]]:
    """Real microphones (no loopback). Returns (name, id) pairs; id == name."""
    return [
        (dev["name"], dev["name"])
        for dev in _iter_input_devices()
        if not dev.get("isLoopbackDevice", False)
    ]


def list_loopbacks() -> list[tuple[str, str]]:
    """System-audio loopback endpoints. Returns (name, id) pairs; id == name."""
    return [
        (dev["name"], dev["name"])
        for dev in _iter_input_devices()
        if dev.get("isLoopbackDevice", False)
    ]


def get_device(device_id: str, *, expect_loopback: bool) -> dict:
    """Resolve a device by exact NAME and the expected `isLoopbackDevice` flag.

    This is the single source of truth for device resolution (shared by the
    GUI/CLI listing and `CaptureThread`). PortAudio does not export a stable
    WASAPI endpoint id to Python, so the name is the identity — and there is
    deliberately NO fuzzy/substring fallback:

      - exactly one device with `name == device_id` AND
        `isLoopbackDevice == expect_loopback` -> that device's info dict;
      - zero matches -> RuntimeError «устройство не найдено»;
      - more than one -> RuntimeError «неоднозначно».

    The flag check is the hard guard that a mic channel can never resolve to
    a loopback endpoint (and vice versa) — the root-cause protection against
    the duplicated-transcription bug (a disconnected mic silently falling
    back onto a system-audio endpoint).
    """
    kind = "системного звука" if expect_loopback else "микрофона"
    matches = [
        dev
        for dev in _iter_input_devices()
        if dev["name"] == device_id
        and bool(dev.get("isLoopbackDevice", False)) == expect_loopback
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise RuntimeError(
            f"Устройство {kind} не найдено (имя={device_id!r}). "
            "Возможно, оно отключено или переименовано."
        )
    raise RuntimeError(
        f"Устройство {kind} неоднозначно: найдено несколько устройств "
        f"с именем {device_id!r}. Захват прекращён, чтобы не записать "
        "не то устройство."
    )


def _default_info(p, method_name: str) -> dict | None:
    """Call `p.<method_name>()` defensively; return the info dict or None.

    None means "no default" for the caller: the method is missing (API
    drift, logged at debug) or the call failed / returned a non-dict
    (logged at warning). Default preselection is a soft UX feature and must
    never break its callers.
    """
    method = getattr(p, method_name, None)
    if method is None:
        logger.debug(
            "backend has no %s (API drift) — treating as no default",
            method_name,
        )
        return None
    try:
        info = method()
    except Exception as exc:  # noqa: BLE001 - preselection must never raise
        logger.warning("backend.%s() failed: %s", method_name, exc)
        return None
    return info if isinstance(info, dict) else None


def default_device_names() -> tuple[str | None, str | None]:
    """Return the current system default device names: (mic, loopback).

    - ``mic_name``: the name of the default INPUT device, if it is a real
      microphone — ``maxInputChannels > 0`` and ``isLoopbackDevice`` falsy.
      A loopback endpoint as the default input is NOT preselected (the mic
      channel must never land on a loopback — invariant #1).
    - ``loopback_name``: if the default OUTPUT device is named N and
      ``list_loopbacks()`` contains a device with the EXACT name
      ``N + " [Loopback]"`` — that name (loopback endpoints are named after
      their output endpoint). No fuzzy/substring matching.

    Either slot is ``None`` when the corresponding default is absent or
    cannot be determined (no default, no matching loopback, API drift,
    backend call failure). This function never raises for those cases; it
    only propagates what ``backend.get_backend()`` raises (e.g. the
    non-Windows RuntimeError) — the caller handles that.
    """
    p = backend.get_backend()

    mic_name: str | None = None
    info = _default_info(p, "get_default_input_device_info")
    if (
        info is not None
        and int(info.get("maxInputChannels") or 0) > 0
        and not info.get("isLoopbackDevice", False)
        and info.get("name")
    ):
        mic_name = info["name"]

    loopback_name: str | None = None
    info = _default_info(p, "get_default_output_device_info")
    if info is not None and info.get("name"):
        candidate = f"{info['name']} [Loopback]"
        if any(name == candidate for name, _ in list_loopbacks()):
            loopback_name = candidate

    return mic_name, loopback_name


def _main() -> None:
    try:
        mics = list_microphones()
        loopbacks = list_loopbacks()
        try:
            default_mic, default_loopback = default_device_names()
        except Exception:  # noqa: BLE001 - default markers are best-effort
            default_mic = default_loopback = None
    except Exception as exc:  # noqa: BLE001 - e.g. non-Windows: no backend
        print(f"[ошибка] Не удалось получить список устройств: {exc}",
              file=sys.stderr)
        sys.exit(1)
    print("=== Microphones ===")
    for name, dev_id in mics:
        marker = "  ← по умолчанию" if name == default_mic else ""
        print(f"  {name}{marker}\n    id: {dev_id}")
    print("\n=== Loopbacks (system audio) ===")
    for name, dev_id in loopbacks:
        marker = "  ← по умолчанию" if name == default_loopback else ""
        print(f"  {name}{marker}\n    id: {dev_id}")


if __name__ == "__main__":
    _main()
