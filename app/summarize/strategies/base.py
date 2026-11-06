"""Strategy interface: shared types for summarization strategies.

A strategy is a plain function `condense(transcript_text, ctx) -> FinalInput`
that compresses the (already merged, speaker-tagged) transcript into the
material the FINAL LLM call will work on. The final call itself is NOT part of
the strategy: the pipeline's `_finalize` runs it exactly once for every
strategy, appending the user's report-format prompt (ADR-002).

No protocols / base classes on purpose — the strategies are stateless
functions, a class would be ceremony (ADR-002 D1).
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event
from typing import Any, Callable

from app.config import LLMSettings


@dataclass(frozen=True)
class FinalInput:
    """What the strategy hands to the final LLM call.

    `framing` is strategy-owned: it describes what `material` is and the
    method's rules for using it (Russian). `material` is the condensed content
    including its own caption line (e.g. "Конспекты фрагментов:").
    """

    framing: str
    material: str


@dataclass(frozen=True)
class StrategyContext:
    """Everything a strategy needs, injected by the pipeline.

    `client` is the sync OpenAI client from `client.build_client` (typed Any
    to keep this module importable without openai). `embedder` is only
    non-None for `requires_embeddings` strategies (part 2). `notify` is a
    best-effort progress callback; `cancel_event`, when set, must abort the
    strategy with `SummarizationCancelled` without issuing new LLM calls.
    """

    client: Any
    llm: LLMSettings
    embedder: Any | None
    notify: Callable[[str], None]
    cancel_event: Event | None = None


# (transcript_text, ctx) -> FinalInput
Condense = Callable[[str, StrategyContext], FinalInput]
