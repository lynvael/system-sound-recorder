"""Hierarchical strategy — Context-Aware Hierarchical Merging (Extract-Support).

Port of summary_tests/hierarchical_context (Ou & Lapata, "Context-Aware
Hierarchical Merging for Long Document Summarization", Findings of ACL 2025,
arXiv:2502.00977), adapted to the app:

  - the transcript keeps speakers/timestamps (ADR-002 A3);
  - chunking is the shared `chunk_text` with the method's own constants
    (32000 chars, 500 overlap — module constants, no new env settings);
  - every LLM call goes through `client.chat` with the shared system prompt
    (the experiment's own SYSTEM_PROMPT is superseded by `prompts.SYSTEM`);
  - the LAST merge is the pipeline's `_finalize`: `condense` merges levels
    until the remaining nodes fit in ONE group and returns the `FinalInput`
    whose material is that group's summaries + supporting source passages.

Mechanics (Support variant): at every merge the LLM gets the intermediate
summaries PLUS extractive passages re-selected (embeddings + k-means) from
the source text; contexts are used for proofreading only, never as a source
of new content.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.log import get_logger
from app.summarize import prompts
from app.summarize.chunking import chunk_text
from app.summarize.client import chat
from app.summarize.errors import SummarizationError
from app.summarize.extractive import choose_k, kmeans, split_sentences
from app.summarize.parallel import parallel_map
from app.summarize.strategies.base import FinalInput, StrategyContext

logger = get_logger("summarize.strategies.hierarchical")

# Method constants (ADR-002 D2: module constants, not env settings).
CHUNK_CHARS = 32000
CHUNK_OVERLAP = 500
# Max sentences kept in a support context (extractive cap).
MAX_CONTEXT_SENTENCES = 20
# Safety cap on merge levels: if the LLM keeps failing to compress, force a
# final merge of everything in one call once this level is reached.
MAX_MERGE_LEVELS = 10

# Prompts translated verbatim from Appendix A of Ou & Lapata
# (arXiv:2502.00977): Table 6 (level-1 chunk summary) and Table 8
# (Extract/Retrieve-Support merge).
LEVEL1_PROMPT = """\
Ниже приведён документ:

— {chunk} —

Составь саммари, содержащее все ключевые сведения. В саммари не должно быть явных упоминаний слов «документ» и «саммари». Отвечай на языке исходного текста."""

MERGE_PROMPT = """\
Ниже приведены саммари разных частей документа:

— {summaries} —

Ниже приведены поддерживающие контексты для показанных выше саммари:

— {contexts} —

Объедини приведённые саммари в одно саммари, содержащее все ключевые сведения, и используй поддерживающие контексты, чтобы убедиться, что в объединённом саммари нет фактических ошибок. Суть саммари должна базироваться исключительно на приведённых саммари, а поддерживающие контексты должны использоваться только для вычитки. В саммари не должно быть явных упоминаний слов «документ», «контекст» и «саммари». Отвечай на языке исходного текста."""


@dataclass(frozen=True)
class Node:
    """A summary together with its extractive support context."""

    summary: str
    context: str


def extractive_context(
    text: str,
    ctx: StrategyContext,
    max_sentences: int | None = None,
) -> str:
    """Select up to `max_sentences` representative sentences from `text`.

    `max_sentences=None` uses the module constant (resolved at call time, so
    the method constant stays tunable).

    When the text has at most `max_sentences` sentences, no embeddings are
    requested and all of them are returned. Otherwise sentences are embedded,
    clustered with k-means, and the sentences closest to their cluster
    centroids are kept (round-robin across clusters), re-ordered by their
    original position.
    """
    import numpy as np

    if max_sentences is None:
        max_sentences = MAX_CONTEXT_SENTENCES
    sents = split_sentences(text)
    if not sents:
        return ""
    if len(sents) <= max_sentences:
        return " ".join(sents)
    X = ctx.embedder.embed(sents, cancel_event=ctx.cancel_event)
    k = min(choose_k(len(sents)), len(sents))
    centers, assign = kmeans(X, k)
    dist = np.linalg.norm(X - centers[assign], axis=1)
    by_cluster: dict[int, list[int]] = {
        j: sorted(np.where(assign == j)[0].tolist(), key=lambda i: dist[i])
        for j in range(k)
    }
    chosen: list[int] = []
    while len(chosen) < max_sentences:
        progressed = False
        for j in range(k):
            if by_cluster[j] and len(chosen) < max_sentences:
                chosen.append(by_cluster[j].pop(0))
                progressed = True
        if not progressed:
            break
    chosen.sort()
    return " ".join(sents[i] for i in chosen)


def group_nodes(nodes: list[Node], limit: int) -> list[list[Node]]:
    """Greedy grouping so that sum(len(summary)+len(context)) per group <= limit."""
    groups: list[list[Node]] = []
    cur: list[Node] = []
    cur_size = 0
    for n in nodes:
        size = len(n.summary) + len(n.context)
        if cur and cur_size + size > limit:
            groups.append(cur)
            cur, cur_size = [], 0
        cur.append(n)
        cur_size += size
    if cur:
        groups.append(cur)
    return groups


def format_summaries(nodes: list[Node]) -> str:
    return "\n\n".join(
        f"Саммари {i}:\n{n.summary}" for i, n in enumerate(nodes, start=1)
    )


def _final_input(nodes: list[Node]) -> FinalInput:
    """The last merge: summaries + supporting passages for `_finalize`."""
    framing = (
        "Ниже — саммари частей стенограммы, полученные иерархическим "
        "объединением, и поддерживающие контексты — отрывки из исходной "
        "стенограммы. Суть отчёта извлекай из саммари; поддерживающие "
        "контексты используй только для вычитки фактов и не добавляй из них "
        "нового содержания."
    )
    material = (
        "Саммари частей:\n"
        + format_summaries(nodes)
        + "\n\nПоддерживающие контексты:\n"
        + "\n\n".join(n.context for n in nodes)
    )
    return FinalInput(framing=framing, material=material)


def _merge_level(groups: list[list[Node]], ctx: StrategyContext) -> list[Node]:
    """Merge every group (parallel): LLM merge + re-extracted support context."""
    llm = ctx.llm

    def merge(group: list[Node]) -> Node:
        prompt = MERGE_PROMPT.format(
            summaries=format_summaries(group),
            contexts="\n\n".join(n.context for n in group),
        )
        new_summary = chat(ctx.client, llm, prompts.SYSTEM, prompt)
        # The next level's support context is re-extracted from the joined
        # passages, so contexts always come from the source text, never from
        # intermediate summaries (the Support mechanic).
        new_context = extractive_context("\n".join(n.context for n in group), ctx)
        return Node(summary=new_summary, context=new_context)

    def on_done(done: int, total: int) -> None:
        ctx.notify(f"Саммаризация: объединено групп {done}/{total}…")

    return parallel_map(
        groups,
        merge,
        max_workers=llm.concurrency,
        cancel_event=ctx.cancel_event,
        on_done=on_done,
    )


def condense(text: str, ctx: StrategyContext) -> FinalInput:
    """Compress `text` via hierarchical merging and return the final input."""
    chunks = chunk_text(text, CHUNK_CHARS, CHUNK_OVERLAP)
    if not chunks:
        raise SummarizationError("После обработки стенограммы не осталось текста.")

    if len(chunks) == 1:
        # Nothing to merge: the final call works on the transcript directly
        # (equivalent to LEVEL1 with the report format, zero intermediate
        # calls).
        return FinalInput(
            framing=(
                "Ниже — полная стенограмма встречи. Составь итоговый протокол "
                "на русском языке, опираясь исключительно на текст."
            ),
            material="Стенограмма:\n" + chunks[0],
        )

    llm = ctx.llm
    total = len(chunks)

    def level1(index: int) -> Node:
        chunk = chunks[index]
        summary = chat(
            ctx.client, llm, prompts.SYSTEM, LEVEL1_PROMPT.format(chunk=chunk)
        )
        context = extractive_context(chunk, ctx)
        return Node(summary=summary, context=context)

    def on_chunk_done(done: int, _total: int) -> None:
        ctx.notify(f"Саммаризация: обработано чанков {done}/{total}…")

    nodes = parallel_map(
        list(range(total)),
        level1,
        max_workers=llm.concurrency,
        cancel_event=ctx.cancel_event,
        on_done=on_chunk_done,
    )

    # Merge levels: group the previous level's nodes so summary+context fit
    # CHUNK_CHARS, merge each group, until the nodes fit in ONE group — that
    # last merge is the pipeline's `_finalize`.
    level = 1
    while len(nodes) > 1:
        if level >= MAX_MERGE_LEVELS:
            logger.warning(
                "Иерархическое слияние: достигнут предел MAX_MERGE_LEVELS=%d "
                "без сходимости; все %d узлов передаются в финальный шаг "
                "одним пакетом (промпт может превысить окно контекста).",
                MAX_MERGE_LEVELS,
                len(nodes),
            )
            break
        groups = group_nodes(nodes, CHUNK_CHARS)
        if len(groups) == len(nodes):
            logger.warning(
                "Иерархическое слияние: ни одна пара узлов не влезает в "
                "лимит %d символов; все %d узлов передаются в финальный шаг "
                "одним пакетом (промпт может превысить окно контекста).",
                CHUNK_CHARS,
                len(nodes),
            )
            break
        if len(groups) == 1:
            break  # the last merge is the pipeline's `_finalize`
        nodes = _merge_level(groups, ctx)
        level += 1

    return _final_input(nodes)
