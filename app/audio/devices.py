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

This module does no `pyaudiowpatch` import of its own (the `PyAudio()`
instance is created lazily in `backend.get_backend()`), so it imports
cleanly on non-Windows dev boxes.

Runnable for verification: `python -m app.audio.devices` prints both lists.
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


def _main() -> None:
    try:
        mics = list_microphones()
        loopbacks = list_loopbacks()
    except Exception as exc:  # noqa: BLE001 - e.g. non-Windows: no backend
        print(f"[ошибка] Не удалось получить список устройств: {exc}",
              file=sys.stderr)
        sys.exit(1)
    print("=== Microphones ===")
    for name, dev_id in mics:
        print(f"  {name}\n    id: {dev_id}")
    print("\n=== Loopbacks (system audio) ===")
    for name, dev_id in loopbacks:
        print(f"  {name}\n    id: {dev_id}")


if __name__ == "__main__":
    _main()
