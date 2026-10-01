"""Map-reduce strategy — the app's EXISTING map-reduce algorithm (ADR-002 Q1:
the "Map-Reduce" option is this algorithm, not summary_tests/map_reduce).

Map: one `chat()` per character chunk (chunks come from the shared
`chunking.chunk_text` with `llm.chunk_chars`/`chunk_overlap`), run in parallel
with at most `llm.concurrency` calls in flight, order preserved.
Reduce: a single final call, done once by the pipeline's `_finalize` — this
strategy only prepares its framing + material. With exactly one chunk there is
no map call at all: the final call gets the transcript directly (as before).
"""

from __future__ import annotations

from app.config import LLMSettings
from app.summarize import prompts
from app.summarize.chunking import chunk_text
from app.summarize.client import chat
from app.summarize.errors import SummarizationError
from app.summarize.parallel import parallel_map
from app.summarize.strategies.base import FinalInput, StrategyContext

# Map pass: condense one chunk into free-text notes (chunk notes stay
# free-text; only the FINAL report is Markdown).
MAP_INSTRUCTIONS = (
    "Ниже — фрагмент стенограммы встречи (часть {index} из {total}). "
    "Кратко законспектируй ЭТОТ фрагмент на русском языке, выделив:\n"
    "- принятые решения;\n"
    "- поставленные задачи (с ответственными, если они названы);\n"
    "- обсуждённые темы и важные детали;\n"
    "- открытые вопросы.\n"
    "Не придумывай то, чего нет в тексте. Если какой-то категории нет — "
    "просто пропусти её. Ответ дай простым маркированным списком.\n\n"
    "Фрагмент стенограммы:\n{chunk}"
)

# Framings for the final call (what the material is + method rules).
_SINGLE_FRAMING = (
    "Ниже — полная стенограмма встречи. Составь итоговый протокол на русском "
    "языке, опираясь исключительно на текст."
)
_REDUCE_FRAMING = (
    "Ниже — последовательные конспекты фрагментов одной встречи. "
    "Объедини их в ОДИН связный итоговый отчёт на русском языке, убрав "
    "повторы и противоречия, опираясь исключительно на конспекты."
)


def _map_chunk(
    pair: tuple[int, str],
    *,
    total: int,
    ctx: StrategyContext,
    llm: LLMSettings,
) -> str:
    index, chunk = pair
    return chat(
        ctx.client,
        llm,
        prompts.SYSTEM,
        MAP_INSTRUCTIONS.format(index=index, total=total, chunk=chunk),
    )


def condense(text: str, ctx: StrategyContext) -> FinalInput:
    """Compress `text` via map-reduce and return the final call's input."""
    llm = ctx.llm
    chunks = chunk_text(text, llm.chunk_chars, llm.chunk_overlap)
    if not chunks:
        raise SummarizationError("После обработки стенограммы не осталось текста.")

    if len(chunks) == 1:
        # Nothing to map: the final call works on the transcript directly.
        return FinalInput(
            framing=_SINGLE_FRAMING,
            material="Стенограмма:\n" + chunks[0],
        )

    total = len(chunks)

    def on_done(done: int, _total: int) -> None:
        ctx.notify(f"Саммаризация: обработано фрагментов {done}/{total}…")

    summaries = parallel_map(
        list(enumerate(chunks, start=1)),
        lambda pair: _map_chunk(pair, total=total, ctx=ctx, llm=llm),
        max_workers=llm.concurrency,
        cancel_event=ctx.cancel_event,
        on_done=on_done,
    )

    joined = "\n\n---\n\n".join(summaries)
    return FinalInput(
        framing=_REDUCE_FRAMING,
        material="Конспекты фрагментов:\n" + joined,
    )
