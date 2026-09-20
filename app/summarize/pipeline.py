"""Map-reduce summarization pipeline and the single public entrypoint.

`run_summarization` is the seam the GUI worker calls after a session finishes:
it loads a finished session's transcript.json, merges consecutive same-speaker
segments, summarizes via an OpenAI-compatible LLM (map per chunk, then reduce),
and writes a structured Russian `summary.docx` into the session directory.

Rules:
  - Raise a clear exception on any failure (missing/empty transcript, LLM error);
    the caller surfaces it. Errors are never swallowed.
  - Never overwrite or delete the existing transcript.* files (we only ever
    write summary.docx).
  - No Qt imports. `on_status` may be called from any thread; keep it best-effort.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Callable

from app.config import Config
from app.log import get_logger
from app.pipeline.transcript import Segment, _fmt_clock, merge_consecutive
from app.summarize import prompts
from app.summarize.chunking import build_transcript_text, chunk_text
from app.summarize.client import build_client, chat, chat_structured
from app.summarize.docx_export import SummaryMeta, write_docx
from app.summarize.schema import MeetingSummary

logger = get_logger("summarize.pipeline")


class SummarizationError(Exception):
    """Raised when summarization cannot complete (bad input or LLM failure)."""


def _notify(on_status: Callable[[str], None] | None, message: str) -> None:
    """Best-effort progress report; a failing callback never breaks the run."""
    logger.info(message)
    if on_status is None:
        return
    try:
        on_status(message)
    except Exception:  # pragma: no cover - callback is caller's responsibility
        logger.warning("on_status callback raised; ignoring", exc_info=True)


def _load_segments(transcript_path: Path) -> list[Segment]:
    if not transcript_path.exists():
        raise SummarizationError(f"Стенограмма не найдена: {transcript_path}")
    try:
        raw = json.loads(transcript_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SummarizationError(
            f"Не удалось прочитать {transcript_path.name}: {exc}"
        ) from exc
    if not isinstance(raw, list) or not raw:
        raise SummarizationError(f"Стенограмма пуста: {transcript_path}")
    segments: list[Segment] = []
    for item in raw:
        try:
            segments.append(
                Segment(
                    start=float(item["start"]),
                    end=float(item["end"]),
                    speaker=str(item["speaker"]),
                    text=str(item["text"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SummarizationError(
                f"Некорректная запись в стенограмме: {item!r} ({exc})"
            ) from exc
    if not any(seg.text.strip() for seg in segments):
        raise SummarizationError("В стенограмме нет текста для саммаризации.")
    return segments


def _build_meta(session_dir: Path, segments: list[Segment]) -> SummaryMeta:
    name = session_dir.name
    date_str = name
    try:
        dt = datetime.strptime(name, "%Y%m%d_%H%M%S")
        date_str = dt.strftime("%d.%m.%Y %H:%M")
    except ValueError:
        pass  # non-timestamp dir name: fall back to the raw name

    speakers: list[str] = []
    for seg in segments:
        if seg.speaker not in speakers:
            speakers.append(seg.speaker)

    duration_str: str | None = None
    if segments:
        span = max(seg.end for seg in segments) - min(seg.start for seg in segments)
        if span > 0:
            duration_str = _fmt_clock(span)

    return SummaryMeta(
        session_name=name,
        date_str=date_str,
        segment_count=len(segments),
        speakers=speakers,
        duration_str=duration_str,
    )


def run_summarization(
    session_dir: str | Path,
    config: Config,
    on_status: Callable[[str], None] | None = None,
) -> Path:
    """Summarize a finished session and write summary.docx into `session_dir`.

    Loads transcript.json, merges consecutive same-speaker segments, runs the
    map-reduce pipeline over the configured OpenAI-compatible LLM, and writes
    `summary.docx` into `session_dir`. Returns the path to the written file.

    Raises SummarizationError (or the LLM client's own exceptions) on failure;
    the existing transcript.* files are never touched.
    """
    session_dir = Path(session_dir)
    if not session_dir.is_dir():
        raise SummarizationError(f"Каталог сессии не найден: {session_dir}")

    _notify(on_status, "Саммаризация: чтение стенограммы…")
    segments = _load_segments(session_dir / "transcript.json")
    merged = merge_consecutive(segments)

    text = build_transcript_text(merged)
    llm = config.llm
    chunks = chunk_text(text, llm.chunk_chars, llm.chunk_overlap)
    if not chunks:
        raise SummarizationError("После обработки стенограммы не осталось текста.")

    client = build_client(llm)

    # Any transport/API failure (openai errors, timeouts) or an empty-response
    # RuntimeError from chat() is wrapped into a clean Russian SummarizationError
    # so the GUI surfaces a user-facing message, not a raw technical trace.
    try:
        if len(chunks) == 1:
            _notify(on_status, "Саммаризация: обработка стенограммы…")
            structured = chat_structured(
                client,
                llm,
                prompts.SYSTEM,
                prompts.SINGLE_INSTRUCTIONS.format(chunk=chunks[0]),
                MeetingSummary,
            )
        else:
            total = len(chunks)
            chunk_summaries: list[str] = []
            for i, chunk in enumerate(chunks, start=1):
                _notify(
                    on_status,
                    f"Саммаризация: обработка фрагмента {i}/{total}…",
                )
                result = chat(
                    client,
                    llm,
                    prompts.SYSTEM,
                    prompts.MAP_INSTRUCTIONS.format(index=i, total=total, chunk=chunk),
                )
                print(f"result: ${result}")
                chunk_summaries.append(result)
            _notify(on_status, "Сборка итогового отчёта…")
            joined = "\n\n---\n\n".join(chunk_summaries)
            structured = chat_structured(
                client,
                llm,
                prompts.SYSTEM,
                prompts.REDUCE_INSTRUCTIONS.format(summaries=joined),
                MeetingSummary,
            )
    except SummarizationError:
        raise
    except Exception as exc:  # noqa: BLE001 - unify LLM/transport errors
        raise SummarizationError(f"Ошибка обращения к LLM: {exc}") from exc

    _notify(on_status, "Саммаризация: запись summary.docx…")
    meta = _build_meta(session_dir, merged)
    # A locked/open summary.docx (common on Windows when it's open in Word) or
    # any other write failure becomes a clear Russian error, not a raw OSError.
    try:
        out_path = write_docx(structured, meta, session_dir / "summary.docx")
    except OSError as exc:
        raise SummarizationError(
            f"Не удалось сохранить summary.docx (возможно, файл открыт в Word): {exc}"
        ) from exc
    _notify(on_status, "Саммаризация завершена.")
    return out_path
