"""Summarization error types.

Kept in a separate module so strategies and helpers can raise
`SummarizationError` without importing `pipeline` (no import cycle).
"""

from __future__ import annotations


class SummarizationError(Exception):
    """Raised when summarization cannot complete (bad input or LLM failure).

    Always carries a user-facing Russian message; the GUI surfaces it as a
    non-terminal failure (transcript files are never touched).
    """


class SummarizationCancelled(SummarizationError):
    """Raised when the user cancels a running summarization.

    No report is written; partial results are discarded.
    """

    def __init__(self) -> None:
        super().__init__("Саммаризация отменена.")
