"""Unit tests for the EACSS pipeline (offline, no network)."""

import asyncio

import numpy as np
import pytest

from eacss import (
    EMBED_CONCURRENCY,
    Embedder,
    Stats,
    choose_k,
    kmeans,
    llm_summarize,
    make_chunks,
    split_sentences,
)


def test_split_sentences_basic():
    text = "Привет. Как дела?! Всё ок…\nНовое предложение."
    sents = split_sentences(text)
    assert sents == ["Привет.", "Как дела?!", "Всё ок…", "Новое предложение."]


def test_split_sentences_empty():
    assert split_sentences("   \n  ") == []


def test_make_chunks_respects_size():
    rng = np.random.default_rng(0)
    sentences = [
        "слово " * int(rng.integers(2, 30)) + "конец."
        for _ in range(200)
    ]
    chunks = make_chunks(sentences, chunk_size=1000, overlap=100)
    assert chunks, "no chunks produced"
    for chunk in chunks:
        assert len("\n".join(chunk)) <= 1000 or len(chunk) == 1
    # every sentence appears at least once
    seen = [s for c in chunks for s in c]
    for s in sentences:
        assert s in seen


def test_make_chunks_overlap():
    # 10 sentences of 100 chars each (101 with separator), chunk 500,
    # overlap 200. Chunk 0 fits s0..s3 (end 403; s4 ends at 504 > 500).
    # Next chunk starts at the first sentence whose start < 500 - 200 = 300:
    # s3 starts at 303? no -> s2 starts at 202 <= 300, s3 at 303 > 300,
    # so the scan stops at k=3 and chunk 1 starts at s3 (shared sentence).
    sentences = [f"{'x' * 99}." for _ in range(10)]
    chunks = make_chunks(sentences, chunk_size=500, overlap=200)
    assert chunks[0] == sentences[:4]
    assert chunks[1][0] == sentences[3]


def test_make_chunks_long_sentence_own_chunk():
    sentences = ["a" * 5000, "short one.", "another."]
    chunks = make_chunks(sentences, chunk_size=1000, overlap=100)
    assert chunks[0] == ["a" * 5000]
    assert sum(len(c) for c in chunks) >= 3  # all sentences covered


def test_make_chunks_progress_on_huge_sentence():
    # A sentence longer than chunk_size must not cause an infinite loop.
    sentences = ["a" * 10_000, "b" * 10_000, "c."]
    chunks = make_chunks(sentences, chunk_size=1000, overlap=100)
    assert len(chunks) == 3


def test_choose_k():
    assert choose_k(10) == 3
    assert choose_k(30) == 3
    assert choose_k(100) == 10
    assert choose_k(1000) == 10
    assert choose_k(10_000) == 10


def test_kmeans_separates_clusters():
    rng = np.random.default_rng(1)
    a = rng.normal(loc=0.0, scale=0.1, size=(50, 8))
    b = rng.normal(loc=5.0, scale=0.1, size=(50, 8))
    c = rng.normal(loc=-5.0, scale=0.1, size=(50, 8))
    X = np.vstack([a, b, c])
    centers, labels = kmeans(X, k=3)
    # Each synthetic cluster must map to a single label.
    for cluster_labels in [labels[:50], labels[50:100], labels[100:150]]:
        assert len(set(cluster_labels.tolist())) == 1
    assert len(set(labels.tolist())) == 3


def test_kmeans_deterministic():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(30, 4))
    c1, l1 = kmeans(X, k=4)
    c2, l2 = kmeans(X, k=4)
    assert np.allclose(c1, c2)
    assert np.array_equal(l1, l2)


def test_kmeans_k_larger_than_n():
    X = np.random.default_rng(3).normal(size=(2, 4))
    centers, labels = kmeans(X, k=10)
    assert centers.shape[0] == 2
    assert labels.shape == (2,)


def test_kmeans_convergence():
    # On well-separated data k-means should converge in a few iterations
    # and assign every point to its nearest center.
    rng = np.random.default_rng(4)
    X = np.vstack([rng.normal(scale=0.01, size=(20, 3)) for _ in range(4)])
    centers, labels = kmeans(X, k=4)
    for i, c in enumerate(centers):
        mask = labels == i
        if mask.any():
            assert np.allclose(c, X[mask].mean(axis=0), atol=1e-6)


def test_embedder_concurrency_limit():
    """The shared semaphore must cap parallel embedding requests."""
    state = {"active": 0, "max": 0}

    class _Data:
        def __init__(self, index, embedding):
            self.index = index
            self.embedding = embedding

    class _Resp:
        def __init__(self, data):
            self.data = data

    class _Embeddings:
        async def create(self, model, input):
            state["active"] += 1
            state["max"] = max(state["max"], state["active"])
            await asyncio.sleep(0.01)
            state["active"] -= 1
            return _Resp([_Data(i, [0.0, 1.0]) for i in range(len(input))])

    class _Client:
        def __init__(self):
            self.embeddings = _Embeddings()

    embedder = Embedder(base_url="http://x", api_key="k", model="m")
    embedder.client = _Client()

    # 8 batches of 1 -> 8 parallel candidates; the semaphore must cap at 4.
    vecs = asyncio.run(embedder.embed([f"t{i}" for i in range(8)], batch_size=1))
    assert vecs.shape == (8, 2)
    assert state["max"] <= EMBED_CONCURRENCY
    assert state["max"] == EMBED_CONCURRENCY


def test_llm_summarize_empty_response_raises():
    """A blank LLM reply must raise, not be swallowed into the summary."""

    class _Msg:
        content = ""

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    class _Completions:
        async def create(self, **kwargs):
            return _Resp()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    with pytest.raises(RuntimeError, match="empty response"):
        asyncio.run(llm_summarize(_Client(), "m", "content", Stats()))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
