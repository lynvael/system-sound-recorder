"""EACSS: Extractive-Abstractive Content Summarization Strategy.

Implementation based on the AWS Machine Learning Blog article
"Summarizing long documents with LLMs" (December 2023):
https://aws.amazon.com/blogs/machine-learning/simplify-summarization-of-long-documents-with-llms/

Pipeline:
  1. Split the document into ~32k-character chunks (sentence boundaries,
     500-char overlap).
  2. Extractive phase (per chunk, in parallel): embed sentences,
     k-means clustering, pick the sentence closest to each centroid.
  3. Abstractive phase: LLM summarizes the combined extractive content
     (single "stuff" call, or one level of map-reduce when it is too big).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI

# Transient network failures (connection dropped, request timed out) are
# retried with a small backoff. HTTP 4xx/5xx responses (APIStatusError and
# subclasses such as AuthenticationError) are NOT retried.
RETRY_DELAYS = (5.0, 15.0)

DEFAULT_CHUNK_SIZE = 32_000
DEFAULT_OVERLAP = 500
EMBED_BATCH_SIZE = 64
EMBED_TIMEOUT = 60.0
KMEANS_ITERATIONS = 10
KMEANS_SEED = 42
EMBED_CONCURRENCY = 4

# Anti-hallucination "cite-then-summarize" prompt for the abstractive phase.
# Source: bestprompts.sh "executive summary" prompt, adapted to summarize only
# the k-means-extracted sentences instead of the whole document.
# {content} is the extracted sentences (stuff) or the merged part summaries
# (reduce).
_SUMMARY_FORMAT = (
    "Формат ответа:\n"
    "1. **TL;DR** — самая важная мысль в 1-2 предложениях.\n"
    "2. **Ключевые моменты** — 3-6 пунктов. Сохраняй точные цифры, даты, "
    "имена; не округляй и не приближай.\n"
    "3. **Решения и действия** — только если в предложениях явно есть "
    "решения/дальнейшие шаги; иначе напиши «Не заявлено».\n"
    "4. **Чего не покрыто** — 1-3 вещи, которые читатель мог бы ожидать, "
    "но которые в извлечённых предложениях отсутствуют.\n\n"
    "Правила: используй только информацию из приведённых предложений. "
    "Если что-то важное кажется отсутствующим или неоднозначным — так и "
    "напиши. Не добавляй внешних знаний. Отвечай на языке исходного текста."
)

# "Stuff" prompt: content is the k-means-extracted key sentences.
SUMMARY_PROMPT = (
    "Ниже приведены ключевые предложения, извлечённые из более длинного "
    "документа (транскрипта встречи). Составь саммари только на основе этих "
    "предложений.\n\n"
    "---\n"
    "{content}\n"
    "---\n\n"
    + _SUMMARY_FORMAT
)

# "Reduce" prompt: content is the merged per-part summaries (map-reduce).
REDUCE_PROMPT = (
    "Ниже приведены саммари ключевых частей документа. Составь итоговое "
    "саммари только на основе этих саммари.\n\n"
    "---\n"
    "{content}\n"
    "---\n\n"
    + _SUMMARY_FORMAT
)

# "Map" prompt: plain extraction-preserving summary of one text chunk.
MAP_PROMPT = (
    "Составь краткое саммари следующего текста. Сохрани все ключевые факты: "
    "имена, цифры, даты, сроки, решения. Отвечай на языке исходного текста.\n\n"
    "---\n"
    "{content}\n"
    "---\n"
)

# Sentence boundary: whitespace after .!?… or a newline.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+|\n+")


@dataclass
class Stats:
    chunks: int = 0
    sentences: int = 0
    selected: int = 0
    llm_calls: int = 0
    embed_calls: int = 0
    elapsed: float = 0.0


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def load_text(path: Path) -> str:
    """Load plain text or a JSON transcript (list of segments with 'text')."""
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        segments = json.loads(raw)
        if not isinstance(segments, list):
            raise ValueError(f"Expected a JSON list of segments in {path}")
        return "\n".join(seg.get("text", "") for seg in segments)
    return raw


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> list[str]:
    """Split text into sentences (on .!?… and newlines)."""
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def make_chunks(
    sentences: list[str], chunk_size: int, overlap: int = DEFAULT_OVERLAP
) -> list[list[str]]:
    """Pack sentences into chunks of at most `chunk_size` characters.

    The next chunk starts at the first sentence that begins at or after
    (chunk_end - overlap), so consecutive chunks share ~`overlap` chars.
    A single sentence longer than `chunk_size` becomes its own chunk.
    """
    n = len(sentences)
    # prefix[i] = character offset where sentence i starts.
    prefix = [0] * (n + 1)
    for i, s in enumerate(sentences):
        prefix[i + 1] = prefix[i] + len(s) + (1 if i else 0)

    chunks: list[list[str]] = []
    i = 0
    while i < n:
        j = i
        # Include sentence j+1 only if its end still fits the chunk.
        while j + 1 < n and prefix[j + 2] - prefix[i] <= chunk_size:
            j += 1
        chunks.append(sentences[i : j + 1])
        if j + 1 == n:
            break
        # Next chunk starts at the first sentence that begins at or after
        # (chunk_end - overlap), i.e. re-including up to `overlap` chars.
        end = prefix[j + 1]
        limit = max(end - overlap, prefix[i] + 1)  # +1: always make progress
        k = j + 1
        while k > i and prefix[k - 1] >= limit:
            k -= 1
        # The overlap region may fall inside a single (too long) sentence;
        # in that case re-inclusion is impossible, so keep moving forward.
        i = max(k, i + 1)
    return chunks


# ---------------------------------------------------------------------------
# k-means (numpy implementation, no scikit-learn dependency)
# ---------------------------------------------------------------------------

def kmeans(
    X: np.ndarray, k: int, n_iter: int = KMEANS_ITERATIONS, seed: int = KMEANS_SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Plain k-means with k-means++ initialization.

    Returns (centers, labels). Deterministic for a given seed.
    """
    n, dim = X.shape
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)

    # k-means++ initialization
    centers = np.empty((k, dim), dtype=X.dtype)
    centers[0] = X[rng.integers(n)]
    for c in range(1, k):
        d2 = np.min(((X[:, None, :] - centers[None, :c, :]) ** 2).sum(-1), axis=1)
        total = d2.sum()
        probs = d2 / total if total > 0 else np.full(n, 1.0 / n)
        centers[c] = X[rng.choice(n, p=probs)]

    labels = np.zeros(n, dtype=np.int64)
    for _ in range(n_iter):
        d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        labels = d.argmin(axis=1)
        new_centers = centers.copy()
        for c in range(k):
            mask = labels == c
            if mask.any():
                new_centers[c] = X[mask].mean(axis=0)
        if np.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers
    return centers, labels


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

class Embedder:
    """AsyncOpenAI-compatible embeddings client, batches of 64."""

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.client = AsyncOpenAI(
            base_url=base_url, api_key=api_key, timeout=EMBED_TIMEOUT
        )
        self.model = model
        self.calls = 0
        self._sem = asyncio.Semaphore(EMBED_CONCURRENCY)

    async def embed(self, texts: list[str], batch_size: int = EMBED_BATCH_SIZE) -> np.ndarray:
        sem = self._sem

        async def _one(batch: list[str]) -> list[list[float]]:
            async with sem:
                resp = await self.client.embeddings.create(
                    model=self.model, input=batch
                )
                self.calls += 1
                return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]

        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        results = await asyncio.gather(*(_one(b) for b in batches))
        vecs = [v for batch in results for v in batch]
        return np.asarray(vecs, dtype=np.float32)


def embedder_factory(base_url: str, api_key: str, model: str) -> Embedder:
    return Embedder(base_url=base_url, api_key=api_key, model=model)


class SyntheticEmbedder:
    """Deterministic pseudo-embeddings for --dry-run (no network).

    Identical sentences produce identical vectors; different sentences
    produce independent random directions, which is enough to exercise
    the k-means / selection logic.
    """

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim
        self.calls = 0

    async def embed(self, texts: list[str], batch_size: int = EMBED_BATCH_SIZE) -> np.ndarray:
        vecs = []
        for t in texts:
            digest = hashlib.md5(t.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:4], "big")
            rng = np.random.default_rng(seed)
            v = rng.normal(size=self.dim)
            norm = np.linalg.norm(v)
            vecs.append(v / norm if norm else v)
        self.calls += (len(texts) + batch_size - 1) // batch_size
        return np.asarray(vecs, dtype=np.float32)


def _normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


# ---------------------------------------------------------------------------
# Extractive phase
# ---------------------------------------------------------------------------

def choose_k(n_sentences: int) -> int:
    return min(10, max(3, n_sentences // 10))


async def extractive_summarize(
    sentences: list[str], embedder, batch_size: int = EMBED_BATCH_SIZE
) -> list[str]:
    """Pick the sentences closest to the k-means centroids of the chunk."""
    if not sentences:
        return []
    k = min(choose_k(len(sentences)), len(sentences))
    X = await embedder.embed(sentences, batch_size=batch_size)
    Xn = _normalize(X)
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


# ---------------------------------------------------------------------------
# Abstractive phase
# ---------------------------------------------------------------------------

async def llm_summarize(
    client: AsyncOpenAI, model: str, prompt: str, stats: Stats
) -> str:
    attempt = 0
    while True:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                reasoning_effort="medium",
                extra_body={"allowed_openai_params": ["reasoning_effort"]},
            )
            break
        except (APIConnectionError, APITimeoutError):
            if attempt >= len(RETRY_DELAYS):
                raise
            await asyncio.sleep(RETRY_DELAYS[attempt])
            attempt += 1
    stats.llm_calls += 1
    content_out = resp.choices[0].message.content or ""
    if not content_out.strip():
        raise RuntimeError(
            "LLM returned an empty response; the summary would be "
            "silently lost. Check the model/proxy configuration."
        )
    return content_out.strip()


async def abstractive_summarize(
    content: str,
    client: AsyncOpenAI,
    model: str,
    chunk_size: int,
    stats: Stats,
) -> str:
    """Summarize the combined extractive content.

    Single "stuff" call when it fits in one chunk, otherwise one level of
    map-reduce. The abstractive phase is intentionally sequential.
    """
    if len(content) <= chunk_size:
        prompt = SUMMARY_PROMPT.format(content=content)
        return await llm_summarize(client, model, prompt, stats)

    sentences = split_sentences(content)
    parts = ["\n".join(c) for c in make_chunks(sentences, chunk_size, overlap=0)]
    part_summaries = []
    for part in parts:  # sequential on purpose (see README limitations)
        prompt = MAP_PROMPT.format(content=part)
        part_summaries.append(await llm_summarize(client, model, prompt, stats))
    merged = "\n\n".join(
        f"Часть {i + 1} из {len(part_summaries)}:\n{s}"
        for i, s in enumerate(part_summaries)
    )
    prompt = REDUCE_PROMPT.format(content=merged)
    return await llm_summarize(client, model, prompt, stats)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

async def run_extractive(
    sentences: list[str],
    chunk_size: int,
    overlap: int,
    embedder,
    stats: Stats,
) -> str:
    """Run the extractive phase over all chunks in parallel."""
    chunks = make_chunks(sentences, chunk_size, overlap)
    stats.chunks = len(chunks)
    stats.sentences = len(sentences)

    # One embedding call counter per embedder; parallelize across chunks.
    chunk_summaries = await asyncio.gather(
        *(extractive_summarize(c, embedder) for c in chunks)
    )
    stats.selected = sum(len(s) for s in chunk_summaries)
    return "\n\n".join("\n".join(s) for s in chunk_summaries)
