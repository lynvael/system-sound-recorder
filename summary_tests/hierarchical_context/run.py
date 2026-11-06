#!/usr/bin/env python
"""CLI entry point for Context-Aware Hierarchical Merging (Extract-Support).

Usage:
    uv run python summary_tests/hierarchical_context/run.py <transcript.txt|transcript.json> \
        [--chunk-size 32000] [--overlap 500] [--max-context-sentences 20] \
        [--max-chars N] [--concurrency 4] [--fake-embeddings]
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv

from pipeline import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_CONTEXT_SENTENCES,
    DEFAULT_OVERLAP,
    make_embedder,
    make_llm,
    render_report,
    run_pipeline,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
# summary_tests/hierarchical_context/run.py -> two levels up
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONCURRENCY = 4


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Context-Aware Hierarchical Merging summarization")
    p.add_argument("transcript", help="Path to a .txt or .json transcript")
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                   help="Max characters per chunk / per merge group (default: 32000)")
    p.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP,
                   help="Chunk overlap in characters (default: 500)")
    p.add_argument("--max-context-sentences", type=int, default=DEFAULT_MAX_CONTEXT_SENTENCES,
                   help="Max sentences in the extractive support context (default: 20)")
    p.add_argument("--max-chars", type=int, default=None,
                   help="Process only the first N characters (testing aid)")
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                   help=f"Max parallel LLM calls (default: {DEFAULT_CONCURRENCY})")
    p.add_argument("--fake-embeddings", action="store_true",
                   help="Use a deterministic fake embedder instead of EMBED_* endpoint (testing aid)")
    return p.parse_args(argv)


async def _run(args: argparse.Namespace) -> None:
    from pipeline import load_transcript

    text = load_transcript(args.transcript)
    if args.max_chars is not None:
        text = text[: args.max_chars]
    if not text.strip():
        raise SystemExit(f"Transcript is empty: {args.transcript}")

    llm = make_llm()
    embed = make_embedder(fake=args.fake_embeddings)

    try:
        summary, stats = await run_pipeline(
            text,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            max_context_sentences=args.max_context_sentences,
            llm=llm,
            embed=embed,
            concurrency=args.concurrency,
        )
    finally:
        await llm.close()

    print(summary)
    print(
        f"\n[stats] levels={stats.levels} chunks={stats.chunks} "
        f"llm_calls={stats.llm_calls} embed_calls={stats.embed_calls} "
        f"time={stats.elapsed:.1f}s",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{Path(args.transcript).stem}.md"
    out_path.write_text(render_report(Path(args.transcript).name, summary, stats), encoding="utf-8")
    print(f"[saved] {out_path}")


def main(argv: list[str] | None = None) -> None:
    # Load .env here (not at pipeline import time) to avoid import side effects.
    load_dotenv(_PROJECT_ROOT / ".env")
    args = parse_args(argv)
    try:
        asyncio.run(_run(args))
    except RuntimeError as e:
        raise SystemExit(f"Error: {e}")


if __name__ == "__main__":
    main()
