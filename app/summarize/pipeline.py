"""Summarization pipeline — orchestration only.

`run_summarization` is the single public entrypoint the GUI worker calls
after a session finishes: it validates the options, loads the finished
session's transcript.json, merges consecutive same-speaker segments, runs the
selected strategy (`STRATEGIES`) to condense the transcript, makes ONE final
LLM call (`_finalize`, Markdown for every strategy and every prompt), and
writes a timestamped `summary_<strategy>_<YYYYMMDD_HHMMSS>.docx` into the
session directory.

Rules:
  - All failures are funneled into SummarizationError (Russian message);
    cancellation raises SummarizationCancelled. Never swallowed.
  - Never overwrite or delete the existing transcript.* files.
  - No Qt imports. `on_status` may be called from any thread; best-effort.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from app.config import Config
from app.log import get_logger
from app.pipeline.transcript import Segment, _fmt_clock, merge_consecutive
from app.summarize import prompts
from app.summarize.chunking import build_transcript_text
from app.summarize.client import build_client, chat
from app.summarize.docx_export import SummaryMeta, write_markdown_docx
from app.summarize.errors import SummarizationCancelled, SummarizationError
from app.summarize.extractive import Embedder
from app.summarize.parallel import parallel_map
from app.summarize.strategies import (
    DEFAULT_STRATEGY_ID,
    STRATEGIES,
    StrategyInfo,
)
from app.summarize.strategies.base import FinalInput, StrategyContext

logger = get_logger("summarize.pipeline")

# Legacy filename written before per-method timestamped reports existed.
_LEGACY_SUMMARY_NAME = "summary.docx"


@dataclass(frozen=True)
class SummarizationOptions:
    """Per-run user choices (one object in the Qt signal; no Qt here).

    `custom_prompt=None` means "use the built-in DEFAULT_REPORT_PROMPT"; a
    non-None value must be non-blank (validated before any network call).
    """

    strategy: str = DEFAULT_STRATEGY_ID
    custom_prompt: str | None = None


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


def _build_meta(
    session_dir: Path, segments: list[Segment], method: str | None
) -> SummaryMeta:
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
        method=method,
    )


def _validate_options(
    options: SummarizationOptions, config: Config
) -> StrategyInfo:
    """Validate user options BEFORE any network call (ADR-002 D4/D6)."""
    info = STRATEGIES.get(options.strategy)
    if info is None:
        raise SummarizationError(
            f"Неизвестный метод саммаризации: {options.strategy!r}."
        )
    if options.custom_prompt is not None and not options.custom_prompt.strip():
        raise SummarizationError(
            "Пользовательский промпт пуст: заполните «Промпт отчёта…» или "
            "используйте промпт по умолчанию."
        )
    if info.requires_embeddings and not config.embed.is_configured:
        raise SummarizationError(
            f"Для метода «{info.label}» нужны настройки эмбеддингов: "
            "задайте EMBED_URL и EMBED_MODEL в .env."
        )
    return info


def _finalize(
    ctx: StrategyContext, final_input: FinalInput, custom_prompt: str | None
) -> str:
    """The ONE final LLM call, shared by all strategies; returns Markdown.

    Message layout: strategy framing + material + report-format requirements
    (built-in DEFAULT_REPORT_PROMPT or the user's prompt). The user's text is
    always concatenated as a VALUE — never `.format()`-ed, so braces in it
    are safe. Markdown output rules live in the fixed SYSTEM part, not in the
    user-editable prompt.
    """
    prompt = custom_prompt if custom_prompt is not None else prompts.DEFAULT_REPORT_PROMPT
    user = (
        final_input.framing
        + "\n\n"
        + final_input.material
        + "\n\n"
        + "Требования к формату отчёта:\n"
        + prompt
    )
    system = prompts.SYSTEM + "\n\n" + prompts.MARKDOWN_OUTPUT_RULES
    return chat(ctx.client, ctx.llm, system, user)


def run_summarization(
    session_dir: str | Path,
    config: Config,
    on_status: Callable[[str], None] | None = None,
    options: SummarizationOptions | None = None,
    cancel_event: threading.Event | None = None,
) -> Path:
    """Summarize a finished session and write a .docx report into `session_dir`.

    Output: `summary_<strategy_id>_<YYYYMMDD_HHMMSS>.docx` (a new file per
    run). Returns the path to the written file.

    Raises SummarizationCancelled when `cancel_event` is set (no report is
    written) or SummarizationError on any other failure; the existing
    transcript.* files are never touched.
    """
    session_dir = Path(session_dir)
    if not session_dir.is_dir():
        raise SummarizationError(f"Каталог сессии не найден: {session_dir}")

    options = options or SummarizationOptions()
    info = _validate_options(options, config)
    if cancel_event is not None and cancel_event.is_set():
        raise SummarizationCancelled()

    _notify(on_status, "Саммаризация: чтение стенограммы…")
    segments = _load_segments(session_dir / "transcript.json")
    llm = config.llm
    # Cap merged line length at the chunk budget so chunk_text splits long
    # monologues at turn boundaries instead of hard-cutting mid-sentence.
    merged = merge_consecutive(segments, max_chars=llm.chunk_chars)
    text = build_transcript_text(merged)

    client = build_client(llm)
    embedder: Embedder | None = None
    try:
        # The embeddings endpoint is only touched by requires_embeddings
        # strategies; the guard above guarantees it is configured for them.
        # Built inside the try so a constructor failure surfaces as
        # SummarizationError and the LLM client is closed in `finally`.
        embedder = Embedder(config.embed) if info.requires_embeddings else None
        ctx = StrategyContext(
            client=client,
            llm=llm,
            embedder=embedder,
            notify=lambda msg: _notify(on_status, msg),
            cancel_event=cancel_event,
        )
        final_input = info.condense(text, ctx)
        _notify(on_status, "Сборка итогового отчёта…")
        # The final call goes through parallel_map too, so cancellation stays
        # prompt even if the user cancels while it is in flight: we never
        # block on the in-flight HTTP call, its result is discarded.
        markdown = parallel_map(
            [None],
            lambda _item: _finalize(ctx, final_input, options.custom_prompt),
            max_workers=1,
            cancel_event=cancel_event,
        )[0]
    except SummarizationError:
        raise
    except Exception as exc:  # noqa: BLE001 - unify LLM/transport errors
        raise SummarizationError(f"Ошибка обращения к LLM: {exc}") from exc
    finally:
        # NOTE: close() does NOT abort in-flight requests — the openai SDK
        # keeps them running to completion (up to request_timeout per attempt,
        # with retries) in the abandoned daemon threads, and their results are
        # discarded. We close anyway to release the connection pool.
        try:
            client.close()
        except Exception:  # pragma: no cover - best-effort teardown
            logger.warning("Не удалось закрыть LLM-клиент", exc_info=True)
        if embedder is not None:
            try:
                embedder.close()
            except Exception:  # pragma: no cover - best-effort teardown
                logger.warning(
                    "Не удалось закрыть клиент эмбеддингов", exc_info=True
                )

    if cancel_event is not None and cancel_event.is_set():
        # Cancelled while the final call was in flight: discard, write nothing.
        raise SummarizationCancelled()

    _notify(on_status, "Саммаризация: запись отчёта…")
    meta = _build_meta(session_dir, merged, info.label)
    out_path = session_dir / (
        f"summary_{info.id}_{datetime.now():%Y%m%d_%H%M%S}.docx"
    )
    # A locked/open file (common on Windows when it's open in Word) or any
    # other write failure becomes a clear Russian error, not a raw exception.
    try:
        write_markdown_docx(markdown, meta, out_path)
    except OSError as exc:
        raise SummarizationError(
            f"Не удалось сохранить {out_path.name} "
            f"(возможно, файл открыт в Word): {exc}"
        ) from exc
    except Exception as exc:  # noqa: BLE001 - keep the SummarizationError contract
        raise SummarizationError(f"Не удалось сохранить {out_path.name}: {exc}") from exc
    _notify(on_status, "Саммаризация завершена.")
    return out_path


def find_latest_summary(session_dir: str | Path) -> Path | None:
    """Newest report in `session_dir` by mtime, or None if there is none.

    Considers both per-method files (`summary_*.docx`) and the legacy
    `summary.docx` written by older versions.
    """
    session_dir = Path(session_dir)
    candidates = [p for p in session_dir.glob("summary_*.docx") if p.is_file()]
    legacy = session_dir / _LEGACY_SUMMARY_NAME
    if legacy.is_file():
        candidates.append(legacy)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)
