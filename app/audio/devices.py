"""Enumerate audio input devices for the GUI dropdowns.

`soundcard` exposes both real microphones and WASAPI loopback endpoints through
`all_microphones`; loopback devices carry the `isloopback` flag. We split them
into two lists so the GUI can offer a microphone selector and a system-audio
(loopback) selector separately.

Runnable for verification: `python -m app.audio.devices` prints both lists.
"""

from __future__ import annotations

import soundcard as sc

from app.log import get_logger

logger = get_logger("devices")


def list_microphones() -> list[tuple[str, str]]:
    """Real microphones (no loopback). Returns (name, id) pairs for the GUI."""
    return [(m.name, m.id) for m in sc.all_microphones(include_loopback=False)]


def list_loopbacks() -> list[tuple[str, str]]:
    """System-audio loopback endpoints. Returns (name, id) pairs for the GUI."""
    return [
        (m.name, m.id)
        for m in sc.all_microphones(include_loopback=True)
        if m.isloopback
    ]


def get_microphone(device_id: str, *, include_loopback: bool = False):
    """Resolve a soundcard microphone/loopback object by its id.

    `include_loopback` MUST match the endpoint kind you expect: pass True only
    when resolving a system-audio (loopback) id, and leave it False for a real
    microphone. Passing True for a mic makes soundcard's fuzzy id fallback able
    to resolve onto a loopback endpoint (see `app/audio/capture._resolve_device`
    for why that duplicated the transcription). Raises if not found.
    """
    return sc.get_microphone(device_id, include_loopback=include_loopback)


def _main() -> None:
    print("=== Microphones ===")
    for name, dev_id in list_microphones():
        print(f"  {name}\n    id: {dev_id}")
    print("\n=== Loopbacks (system audio) ===")
    for name, dev_id in list_loopbacks():
        print(f"  {name}\n    id: {dev_id}")


if __name__ == "__main__":
    _main()
