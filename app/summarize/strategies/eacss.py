"""EACSS strategy — extractive-then-abstractive summarization.

Port of summary_tests/eacss (AWS ML blog, "Simplify summarization of long
documents with LLMs", Dec 2023), adapted to the app:

  - the transcript keeps speakers/timestamps (ADR-002 A3);
  - chunking is the shared `chunk_text` with the method's own constants
    (32000 chars, 500 overlap — module constants, no new env settings);
  - every LLM call goes through `client.chat` (SDK retries, the experiment's
    custom retry loop is NOT ported);
  - the FINAL (report) call is the pipeline's `_finalize`: `condense`
    returns the `FinalInput` whose material is the selected sentences
    (stuff case) or, when they overflow one chunk, the per-part summaries
    (map calls run via `parallel_map`).

Extractive phase (per chunk, in parallel): sentence split → remote
embeddings → k-means → the sentence nearest to each centroid (original
order preserved).
"""

from __future__ import annotations

from app.log import get_logger
from app.summarize import prompts
from app.summarize.chunking import chunk_text
from app.summarize.client import chat
from app.summarize.errors import SummarizationError
from app.summarize.extractive import choose_k, kmeans, split_sentences
from app.summarize.parallel import parallel_map
from app.summarize.strategies.base import FinalInput, StrategyContext

logger = get_logger("summarize.strategies.eacss")

# Method constants (ADR-002 D2: module constants, not env settings).
CHUNK_CHARS = 32000
CHUNK_OVERLAP = 500

# Map prompt for the overflow case (fixed Russian prompt from the experiment).
MAP_PROMPT = (
    "Составь краткое саммари следующего текста. Сохрани все ключевые факты: "
    "имена, цифры, даты, сроки, решения. Отвечай на языке исходного текста.\n\n"
    "---\n"
    "{content}\n"
    "---\n"
)

# Framings for the final call (what the material is + method rules).
_STUFF_FRAMING = (
    "Ниже — ключевые предложения, извлечённые из стенограммы встречи методом "
    "EACSS (кластеризация эмбеддингов). Составь отчёт только на основе этих "
    "предложений. Если что-то важное кажется отсутствующим или "
    "неоднозначным — так и напиши. Не добавляй внешних знаний."
)
_OVERFLOW_FRAMING = (
    "Ниже — саммари ключевых частей стенограммы: извлечённые методом EACSS "
    "предложения не поместились в один фрагмент, поэтому были разбиты на "
    "части и саммаризированы по частям. Объедини их в ОДИН связный итоговый "
    "отчёт, опираясь исключительно на эти саммари. Не добавляй внешних "
    "знаний."
)


def _extract_from_chunk(sentences: list[str], ctx: StrategyContext) -> list[str]:
    """Pick the sentences closest to the k-means centroids of one chunk."""
    import numpy as np

    if not sentences:
        return []
    k = min(choose_k(len(sentences)), len(sentences))
    X = ctx.embedder.embed(sentences, cancel_event=ctx.cancel_event)
    # Normalize: the nearest-centroid choice is then a cosine similarity.
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms
    centers, labels = kmeans(Xn, k)

    chosen: set[int] = set()
    for c in range(k):
        mask = labels == c
        if not mask.any():
            continue
        indices = np.where(mask)[0]
        sims = Xn[indices] @ centers[c]
        chosen.add(int(indices[int(sims.argmax())]))

    # Preserve the original order of the chunk.
    return [s for i, s in enumerate(sentences) if i in chosen]


def condense(text: str, ctx: StrategyContext) -> FinalInput:
    """Compress `text` via EACSS and return the final call's input."""
    chunks = chunk_text(text, CHUNK_CHARS, CHUNK_OVERLAP)
    if not chunks:
        raise SummarizationError("После обработки стенограммы не осталось текста.")

    total = len(chunks)

    def extract(index: int) -> list[str]:
        return _extract_from_chunk(split_sentences(chunks[index]), ctx)

    def on_chunk_done(done: int, _total: int) -> None:
        ctx.notify(
            f"Саммаризация: извлечено предложений из фрагментов {done}/{total}…"
        )

    selected = parallel_map(
        list(range(total)),
        extract,
        max_workers=ctx.llm.concurrency,
        cancel_event=ctx.cancel_event,
        on_done=on_chunk_done,
    )

    extracted = "\n\n".join("\n".join(s) for s in selected if s)
    if not extracted:
        raise SummarizationError(
            "Экстрактивная фаза не выбрала ни одного предложения."
        )

    if len(extracted) <= CHUNK_CHARS:
        return FinalInput(
            framing=_STUFF_FRAMING,
            material="Ключевые предложения:\n" + extracted,
        )

    # Overflow: one level of map over the extracted content (parallel), and
    # the reduce call is the pipeline's `_finalize`.
    parts = chunk_text(extracted, CHUNK_CHARS, 0)
    llm = ctx.llm

    def map_part(part: str) -> str:
        return chat(
            ctx.client, llm, prompts.SYSTEM, MAP_PROMPT.format(content=part)
        )

    def on_part_done(done: int, _total: int) -> None:
        ctx.notify(f"Саммаризация: обработано частей {done}/{len(parts)}…")

    part_summaries = parallel_map(
        parts,
        map_part,
        max_workers=llm.concurrency,
        cancel_event=ctx.cancel_event,
        on_done=on_part_done,
    )
    material = "\n\n".join(
        f"Часть {i} из {len(part_summaries)}:\n{s}"
        for i, s in enumerate(part_summaries, start=1)
    )
    return FinalInput(
        framing=_OVERFLOW_FRAMING, material="Саммари частей:\n" + material
    )
