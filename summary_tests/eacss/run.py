#!/usr/bin/env python3
"""EACSS summarization CLI.

Usage:
  uv run python summary_tests/eacss/run.py INPUT [--chunk-size 32000]
                                           [--overlap 500]
                                           [--dry-run]
                                           [--extracted FILE]

INPUT:      .txt file or .json transcript (list of {start,end,speaker,text}).
--dry-run:  no network: synthetic embeddings + stub abstractive phase.
--extracted FILE: skip the extractive phase and run the abstractive phase
                  on the given pre-extracted content (LLM is still called).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from eacss import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    AsyncOpenAI,
    Stats,
    SyntheticEmbedder,
    abstractive_summarize,
    embedder_factory,
    load_text,
    run_extractive,
    split_sentences,
)

OUTPUT_DIR = Path(__file__).parent / "output"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EACSS long-document summarization")
    p.add_argument("input", type=Path, help="Path to .txt or .json transcript")
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    p.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="No network: synthetic embeddings, stub abstractive phase",
    )
    p.add_argument(
        "--extracted",
        type=Path,
        help="Pre-extracted content: skip extractive phase, run LLM only",
    )
    return p.parse_args()


async def main() -> int:
    args = parse_args()
    load_dotenv()  # root .env

    stats = Stats()
    started = time.monotonic()

    # --- LLM config ---------------------------------------------------------
    llm_url = os.environ.get("LLM_URL")
    llm_key = os.environ.get("LLM_API_KEY")
    llm_model = os.environ.get("LLM_MODEL")
    llm_timeout = float(os.environ.get("LLM_REQUEST_TIMEOUT", "600"))

    # --- Validate env up front, before any network call ---------------------
    # Abstractive phase needs the LLM unless we are in --dry-run (stub).
    if not args.dry_run:
        if not llm_url or not llm_key:
            print("Ошибка: LLM_URL / LLM_API_KEY не заданы.", file=sys.stderr)
            return 2
        if not llm_model:
            print("Ошибка: LLM_MODEL не задан.", file=sys.stderr)
            return 2
    # Extractive phase needs the embeddings endpoint unless --dry-run or
    # --extracted (which skips the extractive phase entirely).
    if not args.dry_run and not args.extracted:
        embed_url = os.environ.get("EMBED_URL")
        embed_model = os.environ.get("EMBED_MODEL")
        if not embed_url:
            print(
                "Ошибка: EMBED_URL не задан. Добавьте в .env: "
                "EMBED_URL (base_url с /v1), EMBED_MODEL, при необходимости EMBED_API_KEY.",
                file=sys.stderr,
            )
            return 2
        if not embed_model:
            print("Ошибка: EMBED_MODEL не задан.", file=sys.stderr)
            return 2

    # --- Extractive phase ---------------------------------------------------
    if args.extracted:
        extracted = args.extracted.read_text(encoding="utf-8")
        stats.chunks = 1
        stats.sentences = len(split_sentences(extracted))
        stats.selected = stats.sentences
    elif args.dry_run:
        text = load_text(args.input)
        sentences = split_sentences(text)
        embedder = SyntheticEmbedder()
        extracted = await run_extractive(sentences, args.chunk_size, args.overlap, embedder, stats)
        stats.embed_calls = embedder.calls
    else:
        embed_url = os.environ.get("EMBED_URL")
        embed_key = os.environ.get("EMBED_API_KEY", "not-needed")
        embed_model = os.environ.get("EMBED_MODEL")
        embedder = embedder_factory(embed_url, embed_key, embed_model)
        try:
            text = load_text(args.input)
            sentences = split_sentences(text)
            extracted = await run_extractive(sentences, args.chunk_size, args.overlap, embedder, stats)
        finally:
            await embedder.client.close()
        stats.embed_calls = embedder.calls

    # --- Abstractive phase --------------------------------------------------
    if args.dry_run:
        # Stub: no LLM call in dry-run mode.
        summary = (
            "[dry-run: abstractive-фаза пропущена]\n\n"
            + extracted[:2000]
            + ("\n... [обрезано]" if len(extracted) > 2000 else "")
        )
    else:
        client = AsyncOpenAI(base_url=llm_url, api_key=llm_key, timeout=llm_timeout)
        try:
            summary = await abstractive_summarize(
                extracted, client, llm_model, args.chunk_size, stats
            )
        finally:
            await client.close()

    stats.elapsed = time.monotonic() - started

    # --- Output -------------------------------------------------------------
    out_path = OUTPUT_DIR / f"{args.input.stem}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        f"# EACSS summary: {args.input.name}\n\n"
        f"## Statistics\n\n"
        f"- chunks: {stats.chunks}\n"
        f"- sentences: {stats.sentences}\n"
        f"- selected sentences: {stats.selected}\n"
        f"- LLM calls: {stats.llm_calls}\n"
        f"- embedding calls: {stats.embed_calls}\n"
        f"- elapsed: {stats.elapsed:.1f}s\n\n"
        f"## Summary\n\n{summary}\n",
        encoding="utf-8",
    )

    print(summary)
    print(
        f"\n--- stats: chunks={stats.chunks} sentences={stats.sentences} "
        f"selected={stats.selected} llm_calls={stats.llm_calls} "
        f"embed_calls={stats.embed_calls} elapsed={stats.elapsed:.1f}s "
        f"output={out_path} ---",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
