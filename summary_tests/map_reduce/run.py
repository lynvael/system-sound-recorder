#!/usr/bin/env python3
"""Map-Reduce summarization of long meeting transcripts.

Fully self-contained experiment: has its own input loader and its own LLM
client. No imports from app/ or from other summary_tests/ folders.

Pipeline:
  1. Split the transcript into chunks of --chunk-size characters, preferring
     sentence boundaries, with a character overlap between chunks.
  2. Map: summarize every chunk in parallel via the LLM.
  3. Reduce: merge chunk summaries into one; if the merged text still exceeds
     the chunk size, batch and reduce recursively until a single summary
     remains. The final step produces a structured summary (title, intro,
     bullet list, conclusion).

Usage:
    uv run python summary_tests/map_reduce/run.py <transcript.txt|transcript.json>
"""

from __future__ import annotations

import argparse
import asyncio
import bisect
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI

DEFAULT_CHUNK_SIZE = 32000
DEFAULT_OVERLAP = 500
DEFAULT_CONCURRENCY = 4
MAX_REDUCE_ROUNDS = 10

# Transient network failures (connection dropped, request timed out) are
# retried with a small backoff. HTTP 4xx/5xx responses (APIStatusError and
# subclasses such as AuthenticationError) are NOT retried.
RETRY_DELAYS = (5.0, 15.0)

# Prompts are intentionally in Russian: the source transcripts are Russian,
# and the instruction "answer in the language of the source text" keeps the
# strategy language-agnostic.
#
# Sources:
# - MAP_PROMPT: after the LangChain default map-reduce map prompt
#   ("Write a concise summary of the following: ... CONCISE SUMMARY:"),
#   extended with explicit fact-preservation instructions.
# - REDUCE_PROMPT: after the merge prompt from Ou & Lapata, "Summarizing
#   the Summaries" (ACL 2025).
# - FINAL_PROMPT: structural format (title / intro / bullets / conclusion)
#   from Atef Ataya's map-reduce summarization article.
# - COLLAPSE_PROMPT: plain fact-preserving compression.
MAP_PROMPT = """Составь краткое саммари следующего текста.
Сохрани все ключевые факты: имена, цифры, даты, сроки, решения.
Отвечай на языке исходного текста.

---
{text}

САММАРИ:"""

REDUCE_PROMPT = """Ниже приведены саммари разных частей документа:

---
{text}
---

Объедини приведённые саммари в одно саммари, содержащее все ключевые факты.
В саммари не должно быть явных упоминаний слов «документ», «саммари», «часть».
Отвечай на языке исходного текста."""

# Used only for the final reduce step (when exactly one summary remains):
# produces a structured summary with a title, intro, bullet list and
# concluding paragraph.
FINAL_PROMPT = """Составь итоговое саммари по приведённым саммари. Включи следующие элементы:
* Заголовок, точно отражающий содержание.
* Вступительный абзац с обзором темы.
* Пунктирный список ключевых моментов (сохраняй точные цифры, даты, имена).
* Заключительный абзац с выводом.

---
{text}"""

# Used when a single summary alone exceeds chunk_size: it cannot be batched,
# so it is compressed first before the next reduce round.
COLLAPSE_PROMPT = """Сжми следующее саммари, сохранив все ключевые факты (имена, цифры, даты, решения):

---
{text}"""

# A "sentence" is a run of characters ending with sentence-final punctuation
# (or the end of the text). Non-greedy, DOTALL so it also spans newlines.
SENTENCE_RE = re.compile(r"(?s).+?(?:[.!?…]+|$)")


def load_transcript(path: Path) -> str:
    """Load a .txt transcript or a .json list of segments into plain text.

    JSON format: [{"start": float, "end": float, "speaker": str, "text": str}, ...]
    Speaker labels are intentionally not inserted into the text.
    """
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list of segments in {path}")
        parts = [
            str(seg.get("text", "")).strip()
            for seg in data
            if str(seg.get("text", "")).strip()
        ]
        return " ".join(parts)
    return path.read_text(encoding="utf-8")


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into chunks of at most chunk_size characters.

    Chunks end on sentence boundaries where possible. Each chunk (except the
    last) overlaps the next one by about `overlap` characters, so facts near
    a boundary are not lost in the map phase. A single sentence longer than
    the whole chunk is hard-split at chunk_size.
    """
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    spans = [m.span() for m in SENTENCE_RE.finditer(text)]
    starts = [s[0] for s in spans]

    chunks: list[str] = []
    pos = 0
    while pos < len(text):
        # Sentence containing `pos` (or the next one if pos is past all of them).
        i = bisect.bisect_right(starts, pos) - 1
        if i < 0:
            i = 0
        start = max(pos, spans[i][0])
        # Grow the chunk sentence by sentence while its length fits chunk_size.
        j = i
        while j + 1 < len(spans) and spans[j + 1][1] - start <= chunk_size:
            j += 1
        # Hard-cap a single sentence that is longer than the whole chunk.
        end = min(spans[j][1], start + chunk_size)
        chunks.append(text[start:end])
        if end >= len(text):
            break
        # Next chunk starts at the last sentence whose start is no later than
        # `end - overlap`; fall back to a hard position if that would not
        # make forward progress (e.g. one giant sentence).
        target = end - overlap
        k = bisect.bisect_right(starts, target) - 1
        next_pos = spans[k][0] if k >= 0 else target
        if next_pos <= start:
            # The overlap would not make forward progress. Either the current
            # chunk is a hard cut inside one giant sentence (target > start:
            # use the hard position, keeping the overlap), or the chunk is
            # shorter than the overlap itself (e.g. a short first sentence
            # followed by one giant unpunctuated run, common in STT output):
            # in that case `end` is a sentence boundary, so start right after
            # this chunk instead of crawling forward one character at a time.
            next_pos = target if target > start else end
        pos = next_pos
    return chunks


def build_client() -> tuple[AsyncOpenAI, str, float]:
    """Create the async LLM client from environment variables."""
    base_url = os.environ.get("LLM_URL")
    api_key = os.environ.get("LLM_API_KEY")
    model = os.environ.get("LLM_MODEL")
    timeout = float(os.environ.get("LLM_REQUEST_TIMEOUT", "600"))
    if not (base_url and api_key and model):
        raise RuntimeError(
            "LLM_URL, LLM_API_KEY and LLM_MODEL must be set in the environment (see .env)"
        )
    client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
    return client, model, timeout


async def llm_call(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    semaphore: asyncio.Semaphore,
    counter: dict[str, int],
) -> str:
    """Single chat completion, rate-limited by the semaphore, counted.

    Retries transient network errors (APIConnectionError / APITimeoutError)
    with a 5/15 s backoff; HTTP error responses are propagated as-is.
    """
    async with semaphore:
        attempt = 0
        while True:
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=1500,
                    # Required: the model is too slow without an explicit
                    # reasoning effort. Keep this in this single helper,
                    # never inline it.
                    reasoning_effort="medium",
                    extra_body={"allowed_openai_params": ["reasoning_effort"]},
                )
                break
            except (APIConnectionError, APITimeoutError):
                if attempt >= len(RETRY_DELAYS):
                    raise
                await asyncio.sleep(RETRY_DELAYS[attempt])
                attempt += 1
        counter["calls"] += 1
        content = response.choices[0].message.content or ""
        if not content.strip():
            raise RuntimeError(
                "LLM returned an empty response; the summary would be "
                "silently lost. Check the model/proxy configuration."
            )
        return content.strip()


async def map_phase(
    client: AsyncOpenAI,
    model: str,
    chunks: list[str],
    semaphore: asyncio.Semaphore,
    counter: dict[str, int],
) -> list[str]:
    """Summarize every chunk in parallel."""
    prompts = [MAP_PROMPT.format(text=chunk) for chunk in chunks]
    return list(
        await asyncio.gather(
            *(llm_call(client, model, p, semaphore, counter) for p in prompts)
        )
    )


async def reduce_phase(
    client: AsyncOpenAI,
    model: str,
    summaries: list[str],
    chunk_size: int,
    semaphore: asyncio.Semaphore,
    counter: dict[str, int],
) -> str:
    """Merge chunk summaries into one final summary.

    Intermediate rounds: if the joined summaries fit into chunk_size, the
    final step is reached. Otherwise they are re-chunked (no overlap needed:
    summaries are already compressed) and merged with REDUCE_PROMPT. A single
    summary that alone exceeds chunk_size cannot be batched, so it is
    compressed with COLLAPSE_PROMPT first. The final step (one summary
    remains) uses the structural FINAL_PROMPT.
    """
    joined = "\n\n".join(summaries)
    rounds = 0
    while len(joined) > chunk_size and rounds < MAX_REDUCE_ROUNDS:
        rounds += 1
        oversized = [s for s in summaries if len(s) > chunk_size]
        if oversized:
            # A single oversized summary would not fit into any batch;
            # compress it before the next round.
            prompts = [COLLAPSE_PROMPT.format(text=s) for s in oversized]
            compressed = list(
                await asyncio.gather(
                    *(llm_call(client, model, p, semaphore, counter) for p in prompts)
                )
            )
            # `compressed` holds only the oversized summaries, so consume it
            # with an iterator instead of zipping positionally against the
            # full `summaries` list (which would drop/misalign entries).
            comp_iter = iter(compressed)
            summaries = [next(comp_iter) if len(s) > chunk_size else s for s in summaries]
            joined = "\n\n".join(summaries)
            continue
        batches = split_text(joined, chunk_size, 0)
        prompts = [REDUCE_PROMPT.format(text=b) for b in batches]
        summaries = list(
            await asyncio.gather(
                *(llm_call(client, model, p, semaphore, counter) for p in prompts)
            )
        )
        joined = "\n\n".join(summaries)

    # Final step (falls back to one oversized call if the text never shrinks
    # below chunk_size, e.g. the LLM refuses to compress): structural summary.
    return await llm_call(
        client, model, FINAL_PROMPT.format(text=joined), semaphore, counter
    )


async def run(args: argparse.Namespace) -> int:
    load_dotenv()
    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        return 1

    text = load_transcript(input_path)
    if not text.strip():
        print("Error: transcript is empty", file=sys.stderr)
        return 1

    chunks = split_text(text, args.chunk_size, args.overlap)
    client, model, _timeout = build_client()
    semaphore = asyncio.Semaphore(args.concurrency)
    counter: dict[str, int] = {"calls": 0}

    started = time.monotonic()
    try:
        map_summaries = await map_phase(client, model, chunks, semaphore, counter)
        final_summary = await reduce_phase(
            client, model, map_summaries, args.chunk_size, semaphore, counter
        )
    finally:
        await client.close()
    elapsed = time.monotonic() - started

    # Output file: summary_tests/map_reduce/output/<input-name>.md
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{input_path.stem}.md"
    out_path.write_text(final_summary + "\n", encoding="utf-8")

    print(final_summary)
    print(
        f"\n--- stats ---\n"
        f"input chars:      {len(text)}\n"
        f"chunks:           {len(chunks)} (size={args.chunk_size}, overlap={args.overlap})\n"
        f"LLM calls:        {counter['calls']} (model={model})\n"
        f"elapsed:          {elapsed:.1f}s\n"
        f"saved to:         {out_path}"
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Map-Reduce transcript summarization")
    parser.add_argument("input", help="Path to .txt or .json transcript")
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help=f"Chunk size in characters (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--overlap", type=int, default=DEFAULT_OVERLAP,
        help=f"Overlap between chunks in characters (default: {DEFAULT_OVERLAP})",
    )
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY,
        help=f"Max parallel LLM calls (default: {DEFAULT_CONCURRENCY})",
    )
    args = parser.parse_args()
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    if args.overlap < 0:
        parser.error("--overlap must be >= 0")
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
