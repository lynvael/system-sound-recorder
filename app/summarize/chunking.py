"""Build the transcript text and split it into overlapping character chunks.

Character-budgeted (not token-budgeted) so the whole step stays offline and
CPU-friendly — no tokenizer download. Splitting prefers line boundaries so a
speaker turn is not cut mid-sentence when possible.
"""

from __future__ import annotations

from app.pipeline.transcript import Segment, _fmt_clock


def build_transcript_text(segments: list[Segment]) -> str:
    """Render merged, speaker-tagged segments as "[mm:ss] speaker: text" lines."""
    return "\n".join(
        f"[{_fmt_clock(seg.start)}] {seg.speaker}: {seg.text}" for seg in segments
    )


def chunk_text(text: str, chunk_chars: int, overlap: int) -> list[str]:
    """Split `text` into chunks of ~chunk_chars with `overlap` chars of context.

    Prefers to cut on the last newline inside the budget so speaker turns stay
    intact; falls back to a hard character cut when a single line exceeds the
    budget.
    """
    text = text.strip()
    if not text:
        return []
    if chunk_chars <= 0:
        return [text]
    overlap = max(0, min(overlap, chunk_chars - 1))

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_chars, n)
        if end < n:
            # Try to break on a newline within the window (after some content).
            nl = text.rfind("\n", start, end)
            if nl > start:
                end = nl
        chunks.append(text[start:end].strip())
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]
