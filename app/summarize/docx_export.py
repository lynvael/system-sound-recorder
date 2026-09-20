"""Render the structured Russian summary into a .docx via python-docx.

Layout:
  - Heading: session name + date
  - Meta block: segment count, speakers, duration (if derivable)
  - One section per MeetingSummary field, rendered from the validated object
    (no more Markdown scraping): TL;DR as a paragraph, the rest as bulleted
    lists. Empty lists render «—» to preserve the old "no data" behaviour.

The `docx` import is local to the function so importing this module stays cheap.
Section headings stay in Russian and are owned here (single source of truth).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.summarize.schema import MeetingSummary

# The Russian docx headings — identical to what we rendered before. Order here
# is the render order of the list sections; TL;DR is handled separately as a
# paragraph rather than a bullet list.
_TLDR_HEADING = "TL;DR"
_LIST_SECTIONS: tuple[tuple[str, str], ...] = (
    ("Ключевые решения", "key_decisions"),
    ("Задачи", "tasks"),
    ("Обсуждённые темы", "topics"),
    ("Открытые вопросы", "open_questions"),
)

# Placeholder used when a section has no data (matches the old behaviour).
_EMPTY = "—"


@dataclass
class SummaryMeta:
    session_name: str
    date_str: str
    segment_count: int
    speakers: list[str]
    duration_str: str | None


def _task_line(task) -> str:
    """Render a Task as "text" or "text — assignee" when an assignee is named."""
    text = task.text.strip()
    if not text:
        # No task text: drop the line entirely rather than emit " — assignee",
        # which would slip past the `if line` filter as a stray "— Иван" bullet.
        return ""
    assignee = (task.assignee or "").strip()
    if assignee:
        return f"{text} — {assignee}"
    return text


def write_docx(summary: MeetingSummary, meta: SummaryMeta, out_path: str | Path) -> Path:
    """Write the structured summary to `out_path` and return it as a Path."""
    from docx import Document

    out_path = Path(out_path)
    document = Document()

    document.add_heading(f"Протокол встречи — {meta.session_name}", level=0)

    meta_para = document.add_paragraph()
    meta_para.add_run("Дата: ").bold = True
    meta_para.add_run(meta.date_str)
    meta_para.add_run("\nРеплик (после объединения): ").bold = True
    meta_para.add_run(str(meta.segment_count))
    meta_para.add_run("\nУчастники: ").bold = True
    meta_para.add_run(", ".join(meta.speakers) if meta.speakers else _EMPTY)
    if meta.duration_str:
        meta_para.add_run("\nДлительность: ").bold = True
        meta_para.add_run(meta.duration_str)

    # TL;DR: a plain paragraph (falls back to «—» when the model left it empty).
    document.add_heading(_TLDR_HEADING, level=1)
    tldr = summary.tldr.strip()
    document.add_paragraph(tldr if tldr else _EMPTY)

    # List sections: one bullet per item; «—» when the list is empty. Tasks get
    # their assignee appended when one was named.
    for heading, field in _LIST_SECTIONS:
        document.add_heading(heading, level=1)
        items = getattr(summary, field)
        if field == "tasks":
            lines = [_task_line(t) for t in items]
        else:
            lines = [str(item).strip() for item in items]
        lines = [line for line in lines if line]
        if not lines:
            document.add_paragraph(_EMPTY)
            continue
        for line in lines:
            document.add_paragraph(line, style="List Bullet")

    document.save(str(out_path))
    return out_path
