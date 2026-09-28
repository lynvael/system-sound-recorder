"""Context-Aware Hierarchical Merging (Extract-Support) summarization.

Implements the method from Ou & Lapata, "Context-Aware Hierarchical Merging
for Long Document Summarization" (Findings of ACL 2025):

  https://aclanthology.org/2025.findings-acl.289/

Plain hierarchical merging (chunk -> summary -> recursive merge of summaries)
accumulates hallucinations because at every merge level the LLM no longer sees
the source text. The fix used here ("Support" variant): at every merge step,
together with the intermediate summaries, the LLM is given *extractive*
passages taken from the original text as a factual support. The passages are
selected with an extractive procedure: sentence embeddings + k-means, then the
sentences closest to the cluster centroids are kept.

NOTE: the paper uses MemSum (a summarization-trained model) for the extractive
selection; we use generic embeddings instead. This is a known limitation, see
README.md.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, OpenAI

DEFAULT_CHUNK_SIZE = 32000
DEFAULT_OVERLAP = 500
DEFAULT_MAX_CONTEXT_SENTENCES = 20
EMBED_TIMEOUT = 60.0
EMBED_BATCH_SIZE = 64
KMEANS_SEED = 42

# Safety cap on merge levels: if the LLM keeps failing to compress, force a
# final merge of everything in a single call once this level is reached.
MAX_MERGE_LEVELS = 10

SYSTEM_PROMPT = (
    "Ты — опытный редактор, создающий точные и подробные саммари длинных "
    "документов. Ты никогда не выдумываешь фактов."
)

# Prompts are translated verbatim from Appendix A of Ou & Lapata
# (arXiv:2502.00977): Table 6 (level-1 chunk summary) and Table 8
# (Extract/Retrieve-Support merge). The summary must be written in the
# language of the source text.
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

_TIMESTAMP_RE = re.compile(r"^\s*\[\d{1,2}:\d{2}(?::\d{2})?\]\s*")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+|\n+")


# ---------------------------------------------------------------------------
# Text loading and sentence handling
# ---------------------------------------------------------------------------

def load_transcript(path: str | Path) -> str:
    """Load a transcript from .json (list of segments) or .txt.

    JSON format: [{"start": float, "end": float, "speaker": str, "text": str}].
    TXT format: plain lines, optionally prefixed with [mm:ss] timestamps,
    which are stripped.
    """
    p = Path(path)
    if p.suffix == ".json":
        segments = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(segments, list):
            raise ValueError(f"Expected a JSON list of segments in {p}")
        return " ".join(str(seg.get("text", "")).strip() for seg in segments)
    if p.suffix == ".txt":
        lines = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = _TIMESTAMP_RE.sub("", line)
            if line.strip():
                lines.append(line.strip())
        return " ".join(lines)
    raise ValueError(f"Unsupported transcript format: {p.suffix} (use .txt or .json)")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on [.!?…] boundaries and newlines."""
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def chunk_sentences(sentences: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Greedy sentence-level chunking with a character-based overlap.

    When a chunk would exceed `chunk_size`, it is closed and the next chunk
    starts with the trailing sentences of the previous chunk whose total
    length is at least `overlap` characters.
    """
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for s in sentences:
        if cur and cur_len + len(s) > chunk_size:
            chunks.append(" ".join(cur))
            ov: list[str] = []
            ov_len = 0
            for t in reversed(cur):
                ov.insert(0, t)
                ov_len += len(t)
                if ov_len >= overlap:
                    break
            cur = ov
            cur_len = sum(len(t) for t in cur)
        cur.append(s)
        cur_len += len(s)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


# ---------------------------------------------------------------------------
# k-means (numpy only, fixed seed)
# ---------------------------------------------------------------------------

def kmeans(X: np.ndarray, k: int, seed: int = KMEANS_SEED, iters: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Plain k-means with k-means++ initialization.

    Returns (centers, assignments).
    """
    n = X.shape[0]
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)

    # k-means++ initialization
    centers = [X[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min(((X[:, None, :] - np.stack(centers)[None, :, :]) ** 2).sum(-1), axis=1)
        total = d2.sum()
        probs = d2 / total if total > 0 else np.full(n, 1.0 / n)
        centers.append(X[rng.choice(n, p=probs)])
    centers = np.stack(centers)

    assign = np.zeros(n, dtype=int)
    for _ in range(iters):
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        assign = np.argmin(d2, axis=1)
        new_centers = np.array(
            [X[assign == j].mean(axis=0) if np.any(assign == j) else centers[j] for j in range(k)]
        )
        if np.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers
    return centers, assign


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

class Embedder(Protocol):
    requests: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class EmbeddingsClient:
    """OpenAI-compatible /v1/embeddings client (batched, 60 s timeout)."""

    def __init__(self, base_url: str, api_key: str, model: str,
                 timeout: float = EMBED_TIMEOUT, batch_size: int = EMBED_BATCH_SIZE):
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.batch_size = batch_size
        self.requests = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vecs: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            self.requests += 1
            vecs.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
        return vecs


class FakeEmbedder:
    """Deterministic hash-based embedder for offline testing (32-dim)."""

    def __init__(self, dim: int = 32):
        self.dim = dim
        self.requests = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.requests += 1
        vecs = []
        for t in texts:
            h = hashlib.md5(t.encode("utf-8")).digest()
            vec = np.frombuffer((h * self.dim)[: self.dim * 4], dtype=np.uint8).astype(np.float32)
            vecs.append((vec / 255.0).tolist())
        return vecs


def make_embedder(fake: bool = False) -> Embedder:
    """Build the real embeddings client from EMBED_* env vars, or a fake one."""
    if fake:
        return FakeEmbedder()
    base_url = _env("EMBED_URL")
    model = _env("EMBED_MODEL")
    api_key = _env("EMBED_API_KEY", default="not-needed")
    return EmbeddingsClient(base_url=base_url, api_key=api_key, model=model)


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def _reasoning_params() -> dict:
    """Extra kwargs required by the LLM endpoint.

    Without an explicit `reasoning_effort` the model spends too long
    reasoning and responses are unacceptably slow, so every
    `chat.completions.create` call must pass these parameters.
    """
    return {
        "reasoning_effort": "medium",
        "extra_body": {"allowed_openai_params": ["reasoning_effort"]},
    }


class LLMClient:
    """OpenAI-compatible chat client for the summarization LLM."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 600.0):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.calls = 0

    async def close(self) -> None:
        """Close the underlying async HTTP client (releases the connection pool)."""
        await self.client.close()

    async def complete(self, prompt: str) -> str:
        self.calls += 1
        last: Exception | None = None
        for attempt in range(2):
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=1500,
                    **_reasoning_params(),
                )
                content = resp.choices[0].message.content or ""
                if not content.strip():
                    raise RuntimeError(
                        "LLM returned an empty response; the summary would be "
                        "silently lost. Check the model/proxy configuration."
                    )
                return content.strip()
            except (APIConnectionError, APITimeoutError) as e:
                # Retry only transient network errors; other errors (400/404,
                # auth, ...) are pointless to repeat and propagate as-is.
                last = e
                if attempt == 0:
                    await asyncio.sleep(5)
        raise RuntimeError(f"LLM request failed after retries: {last}")


def make_llm() -> LLMClient:
    base_url = _env("LLM_URL")
    api_key = _env("LLM_API_KEY")
    model = _env("LLM_MODEL")
    timeout = float(_env("LLM_REQUEST_TIMEOUT", default="600"))
    return LLMClient(base_url=base_url, api_key=api_key, model=model, timeout=timeout)


def _env(name: str, default: str | None = None) -> str:
    import os
    value = os.environ.get(name, "").strip() or default
    if not value:
        raise RuntimeError(
            f"Environment variable {name} is not set. "
            f"Add it to the .env file in the project root."
        )
    return value


# ---------------------------------------------------------------------------
# Extractive context (Support)
# ---------------------------------------------------------------------------

def extractive_context(text: str, embed: Embedder, max_sentences: int = DEFAULT_MAX_CONTEXT_SENTENCES) -> str:
    """Select up to `max_sentences` representative sentences from `text`.

    Sentences are embedded, clustered with k-means (k = min(10, max(3, n//10))),
    and the sentences closest to their cluster centroids are kept, interleaved
    across clusters, then re-ordered by their original position.
    """
    sents = split_sentences(text)
    if not sents:
        return ""
    if len(sents) <= max_sentences:
        return " ".join(sents)
    X = np.array(embed.embed(sents), dtype=np.float32)
    k = min(10, max(3, len(sents) // 10))
    centers, assign = kmeans(X, k)
    dist = np.linalg.norm(X - centers[assign], axis=1)
    by_cluster: dict[int, list[int]] = {
        j: sorted(np.where(assign == j)[0].tolist(), key=lambda i: dist[i]) for j in range(k)
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


# ---------------------------------------------------------------------------
# Hierarchical merging
# ---------------------------------------------------------------------------

@dataclass
class Node:
    """A summary together with its extractive support context."""
    summary: str
    context: str


@dataclass
class Stats:
    chunks: int = 0
    levels: int = 0
    merge_groups: int = 0  # total number of merge groups across all merge levels
    llm_calls: int = 0
    embed_calls: int = 0
    elapsed: float = field(default=0.0)


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
    parts = []
    for i, n in enumerate(nodes, 1):
        parts.append(f"Саммари {i}:\n{n.summary}")
    return "\n\n".join(parts)


async def run_pipeline(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    max_context_sentences: int = DEFAULT_MAX_CONTEXT_SENTENCES,
    llm: LLMClient,
    embed: Embedder,
    concurrency: int = 4,
) -> tuple[str, Stats]:
    """Run the full Context-Aware Hierarchical Merging pipeline.

    Returns (final_summary, stats).
    """
    t0 = time.monotonic()
    stats = Stats()

    # Limit parallel LLM calls to avoid hammering the endpoint.
    sem = asyncio.Semaphore(concurrency)

    sentences = split_sentences(text)
    chunks = chunk_sentences(sentences, chunk_size, overlap)
    stats.chunks = len(chunks)

    # Level 1: per chunk, in parallel: (a) abstractive summary, (b) extractive context.
    async def process_chunk(chunk: str) -> Node:
        async with sem:
            summary = await llm.complete(LEVEL1_PROMPT.format(chunk=chunk))
        context = await asyncio.to_thread(extractive_context, chunk, embed, max_context_sentences)
        return Node(summary=summary, context=context)

    nodes = await asyncio.gather(*[process_chunk(c) for c in chunks])
    stats.levels = 1

    # Merge levels: group previous-level nodes (summary + context) so the total
    # size fits chunk_size; merge each group with LLM; the support context for
    # the next level is re-extracted (embeddings + k-means) from the joined
    # passages of the group, so contexts always come from the source text.
    #
    # Convergence guarantees (the LLM is not trusted to compress):
    #   * if grouping produced only single-node groups, nobody could be
    #     merged with anyone else -> force-merge everything in one group;
    #   * after MAX_MERGE_LEVELS levels, force a final merge of all nodes.
    while len(nodes) > 1:
        if stats.levels >= MAX_MERGE_LEVELS:
            print(
                f"[hierarchical_context] WARNING: reached MAX_MERGE_LEVELS="
                f"{MAX_MERGE_LEVELS} without converging to a single summary; "
                f"forcing a final merge of all {len(nodes)} nodes in one prompt "
                f"(it may exceed the provider's context window).",
                file=sys.stderr,
            )
            groups = [nodes]
        else:
            groups = group_nodes(nodes, chunk_size)
            if len(groups) == len(nodes) > 1:
                print(
                    f"[hierarchical_context] WARNING: no group of nodes fits the "
                    f"chunk limit ({chunk_size} chars); forcing a single merge of "
                    f"all {len(nodes)} nodes (the prompt may exceed the provider's "
                    f"context window).",
                    file=sys.stderr,
                )
                groups = [nodes]

        async def merge_group(group: list[Node]) -> Node:
            async with sem:
                prompt = MERGE_PROMPT.format(
                    summaries=format_summaries(group),
                    contexts="\n\n".join(n.context for n in group),
                )
                new_summary = await llm.complete(prompt)
            joined_passages = "\n".join(n.context for n in group)
            new_context = await asyncio.to_thread(
                extractive_context, joined_passages, embed, max_context_sentences
            )
            return Node(summary=new_summary, context=new_context)

        nodes = await asyncio.gather(*[merge_group(g) for g in groups])
        stats.levels += 1
        stats.merge_groups += len(groups)

    stats.llm_calls = llm.calls
    stats.embed_calls = embed.requests
    stats.elapsed = time.monotonic() - t0
    return nodes[0].summary, stats


def render_report(input_name: str, summary: str, stats: Stats) -> str:
    return (
        f"# Context-Aware Hierarchical Merging (Extract-Support)\n\n"
        f"Вход: `{input_name}`\n\n"
        f"## Статистика\n\n"
        f"- Уровней дерева: {stats.levels}\n"
        f"- Чанков (уровень 1): {stats.chunks}\n"
        f"- LLM-вызовов: {stats.llm_calls}\n"
        f"- Embedding-запросов: {stats.embed_calls}\n"
        f"- Время: {stats.elapsed:.1f} с\n\n"
        f"## Саммари\n\n{summary}\n"
    )
