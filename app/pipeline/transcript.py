"""Transcript model and exporters.

`Segment` is the shared unit produced by the pipeline and consumed by the GUI.
Times are seconds from session start; `speaker` is a config label ("Я" /
"Собеседники", or a neutral label for imported mono audio).

Exporters write into a session directory `recordings/<timestamp>/`:
  - transcript.txt  human-readable "[mm:ss] speaker: text"
  - transcript.json list of segments (start/end/speaker/text) — input for a
                    future summarization step
  - transcript.srt  subtitle format
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Segment:
    start: float  # seconds from session start
    end: float  # seconds from session start
    speaker: str  # config label, e.g. "Я" / "Собеседники" / neutral for mono
    text: str


def merge_consecutive(segments: list[Segment]) -> list[Segment]:
    """Merge ADJACENT segments sharing the same speaker into one.

    A run of consecutive same-speaker segments becomes a single Segment with
    start = first.start, end = last.end and the texts joined by a space. Blank
    texts are skipped when joining. Pure function: the input list and its
    Segments are not mutated (new Segments are returned).
    """
    merged: list[Segment] = []
    for seg in segments:
        if merged and merged[-1].speaker == seg.speaker:
            prev = merged[-1]
            parts = [p for p in (prev.text, seg.text) if p]
            merged[-1] = Segment(
                start=prev.start,
                end=seg.end,
                speaker=prev.speaker,
                text=" ".join(parts),
            )
        else:
            merged.append(
                Segment(
                    start=seg.start,
                    end=seg.end,
                    speaker=seg.speaker,
                    text=seg.text,
                )
            )
    return merged


def _fmt_clock(seconds: float) -> str:
    """mm:ss (or h:mm:ss past an hour)."""
    seconds = max(0.0, seconds)
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _fmt_srt_time(seconds: float) -> str:
    """HH:MM:SS,mmm."""
    seconds = max(0.0, seconds)
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def export_txt(segments: list[Segment], path: str | Path) -> None:
    path = Path(path)
    lines = [
        f"[{_fmt_clock(seg.start)}] {seg.speaker}: {seg.text}" for seg in segments
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def export_json(segments: list[Segment], path: str | Path) -> None:
    path = Path(path)
    data = [asdict(seg) for seg in segments]
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def export_srt(segments: list[Segment], path: str | Path) -> None:
    path = Path(path)
    blocks = []
    for i, seg in enumerate(segments, start=1):
        blocks.append(
            f"{i}\n"
            f"{_fmt_srt_time(seg.start)} --> {_fmt_srt_time(seg.end)}\n"
            f"{seg.speaker}: {seg.text}\n"
        )
    path.write_text("\n".join(blocks), encoding="utf-8")


def export_all(segments: list[Segment], session_dir: str | Path) -> None:
    """Write transcript.{txt,json,srt} into the session directory."""
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    export_txt(segments, session_dir / "transcript.txt")
    export_json(segments, session_dir / "transcript.json")
    export_srt(segments, session_dir / "transcript.srt")
