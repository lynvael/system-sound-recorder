"""On-demand meeting summarization.

Reads a FINISHED session's transcript.json from disk, merges consecutive
same-speaker segments, runs a map-reduce summarization over an OpenAI-compatible
LLM, and writes a structured Russian `summary.docx` into the session directory.

This package is backend-only: no Qt imports. The single public entrypoint is
`run_summarization`; the GUI worker calls it after a session finishes.
"""

from __future__ import annotations

from app.summarize.pipeline import run_summarization

__all__ = ["run_summarization"]
