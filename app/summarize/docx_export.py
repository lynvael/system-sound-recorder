"""Render the LLM's Markdown report into a .docx via python-docx.

Layout:
  - Heading: «Протокол встречи — <session>»
  - Meta block: date, segment count, speakers, duration (if derivable),
    method label («Метод: …») when set
  - The Markdown body, rendered with a FIXED subset the final prompt asks
    for: `#`..`###` headings, `-`/`*` bullets, `1.` numbered lists,
    `**bold**` inline, plain paragraphs.

Anything outside the subset (tables, code fences, unknown constructs)
degrades to plain text paragraphs — the renderer NEVER raises on strange
Markdown; the worst case is visible markup characters.

The `docx` import is local to the function so importing this module stays
cheap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Placeholder used when the model returned nothing renderable.
_EMPTY = "—"

_BOLD_RE = re.compile(r"(\*\*.+?\*\*)")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_BULLET_RE = re.compile(r"^(\s*)[-*+]\s+(.*)$")
_NUMBERED_RE = re.compile(r"^(\s*)\d+[.)]\s+(.*)$")
_HRULE_RE = re.compile(r"^(-{3,}|\*{3,}|_{3,})$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")

# Characters lxml rejects in XML text: C0 controls (all except \t \n \r),
# lone UTF-16 surrogates (UnicodeEncodeError) and U+FFFE/U+FFFF (ValueError).
# LLM output is untrusted; strip them at the boundary so the renderer keeps
# its "never raises on strange input" contract.
_ILLEGAL_XML_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]")


def _clean(text: str) -> str:
    """Strip XML-illegal control characters from untrusted text."""
    return _ILLEGAL_XML_RE.sub("", text)


@dataclass
class SummaryMeta:
    session_name: str
    date_str: str
    segment_count: int
    speakers: list[str]
    duration_str: str | None
    method: str | None = None


def _add_runs(paragraph, text: str) -> None:
    """Add `text` to `paragraph`, rendering `**bold**` spans as bold runs.

    `text` must already be `_clean`-ed (callers in `_render_line` do it);
    the bold split only produces substrings, so no second pass is needed.
    """
    for part in _BOLD_RE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            paragraph.add_run(part[2:-2]).bold = True
        else:
            paragraph.add_run(part)


def _render_line(document, line: str) -> None:
    """Render one Markdown line; unknown constructs degrade to a paragraph."""
    stripped = line.strip()
    if not stripped:
        return
    if _HRULE_RE.fullmatch(stripped):
        return  # horizontal rule: no docx equivalent we ask for — skip
    m = _HEADING_RE.match(stripped)
    if m:
        # `###` and deeper all map to Heading 3 (the prompt only allows 1..3).
        document.add_heading(_clean(m.group(2).strip()), level=min(len(m.group(1)), 3))
        return
    m = _BULLET_RE.match(line)
    if m:
        indent = m.group(1).expandtabs(4)
        style = "List Bullet 2" if len(indent) >= 2 else "List Bullet"
        _add_runs(document.add_paragraph(style=style), _clean(m.group(2).strip()))
        return
    m = _NUMBERED_RE.match(line)
    if m:
        _add_runs(document.add_paragraph(style="List Number"), _clean(m.group(2).strip()))
        return
    _add_runs(document.add_paragraph(), _clean(stripped))


def _add_header(document, meta: SummaryMeta) -> None:
    """Shared document header: title + meta block (+ «Метод: …» when set)."""
    document.add_heading(_clean(f"Протокол встречи — {meta.session_name}"), level=0)

    meta_para = document.add_paragraph()
    meta_para.add_run("Дата: ").bold = True
    meta_para.add_run(_clean(meta.date_str))
    meta_para.add_run("\nРеплик (после объединения): ").bold = True
    meta_para.add_run(str(meta.segment_count))
    meta_para.add_run("\nУчастники: ").bold = True
    meta_para.add_run(
        _clean(", ".join(meta.speakers)) if meta.speakers else _EMPTY
    )
    if meta.duration_str:
        meta_para.add_run("\nДлительность: ").bold = True
        meta_para.add_run(_clean(meta.duration_str))
    if meta.method:
        meta_para.add_run("\nМетод: ").bold = True
        meta_para.add_run(_clean(meta.method))


def write_markdown_docx(markdown: str, meta: SummaryMeta, out_path: str | Path) -> Path:
    """Write the Markdown report to `out_path` and return it as a Path."""
    from docx import Document

    out_path = Path(out_path)
    document = Document()
    _add_header(document, meta)

    if markdown is None or not markdown.strip():
        document.add_paragraph(_EMPTY)
        document.save(str(out_path))
        return out_path

    in_fence = False
    for line in markdown.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            # Code fence content: plain paragraphs, no Markdown interpretation.
            if line.strip():
                document.add_paragraph(_clean(line.strip()))
            continue
        _render_line(document, line)

    document.save(str(out_path))
    return out_path
