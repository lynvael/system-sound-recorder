"""On-demand meeting summarization.

Reads a FINISHED session's transcript.json from disk, merges consecutive
same-speaker segments, condenses it with the selected strategy
(`STRATEGIES`, default: the app's map-reduce), makes one final LLM call that
produces a Markdown report (built-in or user-defined format), and writes a
timestamped `summary_<strategy>_<YYYYMMDD_HHMMSS>.docx` into the session
directory.

This package is backend-only: no Qt imports, and no openai/docx/numpy at
import time (heavy deps are imported lazily inside the functions that use
them). The single public entrypoint is `run_summarization`; the GUI worker
calls it after a session finishes.
"""

from __future__ import annotations

from app.summarize.errors import SummarizationCancelled, SummarizationError
from app.summarize.pipeline import (
    SummarizationOptions,
    find_latest_summary,
    run_summarization,
)
from app.summarize.prompts import DEFAULT_REPORT_PROMPT
from app.summarize.strategies import (
    DEFAULT_STRATEGY_ID,
    STRATEGIES,
    StrategyInfo,
)

__all__ = [
    "DEFAULT_REPORT_PROMPT",
    "DEFAULT_STRATEGY_ID",
    "STRATEGIES",
    "StrategyInfo",
    "SummarizationCancelled",
    "SummarizationError",
    "SummarizationOptions",
    "find_latest_summary",
    "run_summarization",
]
