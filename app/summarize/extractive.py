"""Shared extractive utilities for the embeddings-based strategies.

`split_sentences`, `choose_k` and ONE deterministic `kmeans` (numpy, fixed
seed) are shared by EACSS and hierarchical so the two strategies never drift
apart (the experiments shipped two near-identical k-means copies).
`Embedder` is a sync OpenAI-compatible /v1/embeddings client: batches of 64,
response order restored via `.index`, failures wrapped in
`SummarizationError` (Russian).

numpy and openai are imported lazily inside the functions that use them:
`import app.summarize` must stay cheap (package contract, see
`app/summarize/__init__.py`).
"""

from __future__ import annotations

import re
from threading import Event
from typing import Any, Sequence

from app.config import EmbedSettings
from app.summarize.errors import SummarizationCancelled, SummarizationError

# Sentence boundary: whitespace after .!?… or a newline.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+|\n+")

# One HTTP request per batch of this many sentences.
EMBED_BATCH_SIZE = 64

# k-means defaults: fixed seed → deterministic clustering. n_iter=10 is a
# deliberate unification of both strategies (the hierarchical experiment
# used 20); early exit on convergence makes the exact cap immaterial.
KMEANS_ITERATIONS = 10
KMEANS_SEED = 42


def split_sentences(text: str) -> list[str]:
    """Split text into sentences (on .!?… and newlines)."""
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def choose_k(n_sentences: int) -> int:
    """Cluster count used by both strategies: min(10, max(3, n // 10))."""
    return min(10, max(3, n_sentences // 10))


def kmeans(
    X: Any,
    k: int,
    *,
    n_iter: int = KMEANS_ITERATIONS,
    seed: int = KMEANS_SEED,
) -> tuple[Any, Any]:
    """Plain k-means with k-means++ initialization.

    `X` is a 2-D numpy array of shape (n, dim). Returns (centers, labels).
    Deterministic for a given seed.
    """
    import numpy as np

    n = X.shape[0]
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)

    # k-means++ initialization
    centers = np.empty((k, X.shape[1]), dtype=X.dtype)
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


class Embedder:
    """Remote OpenAI-compatible embeddings client (batches of 64).

    Order-preserving: each response's `data` is sorted by `.index` before the
    vectors are concatenated, so row i of the result always corresponds to
    `texts[i]` even when the server returns entries out of order.
    """

    def __init__(self, settings: EmbedSettings) -> None:
        from openai import OpenAI

        self._client = OpenAI(
            base_url=settings.url,
            api_key=settings.api_key,
            timeout=settings.request_timeout,
        )
        self._model = settings.model

    def close(self) -> None:
        """Release the HTTP connection pool (best-effort teardown)."""
        self._client.close()

    def embed(
        self, texts: Sequence[str], *, cancel_event: Event | None = None
    ) -> Any:
        """Embed `texts`; returns a float32 numpy array of shape (len, dim).

        Cancellation: `cancel_event` is checked before every batch; when set,
        raises SummarizationCancelled without issuing further requests.
        """
        import numpy as np

        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        vecs: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            if cancel_event is not None and cancel_event.is_set():
                raise SummarizationCancelled()
            batch = list(texts[i : i + EMBED_BATCH_SIZE])
            try:
                resp = self._client.embeddings.create(
                    model=self._model, input=batch
                )
            except Exception as exc:  # noqa: BLE001 - unify endpoint errors
                raise SummarizationError(
                    f"Ошибка сервиса эмбеддингов: {exc}"
                ) from exc
            vecs.extend(
                d.embedding for d in sorted(resp.data, key=lambda d: d.index)
            )
        return np.asarray(vecs, dtype=np.float32)
