"""Tests for the shared extractive utilities (kmeans, split_sentences,
Embedder) — no network: the openai client is faked at the `openai.OpenAI`
seam."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from app.config import EmbedSettings
from app.summarize.errors import SummarizationCancelled, SummarizationError
from app.summarize.extractive import (
    EMBED_BATCH_SIZE,
    Embedder,
    choose_k,
    kmeans,
    split_sentences,
)


def _matrix(n: int, dim: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, dim)).astype(np.float32)


# --- kmeans -----------------------------------------------------------------


class TestKMeans:
    def test_deterministic_for_fixed_seed(self):
        X = _matrix(50, 8)
        c1, l1 = kmeans(X, 5)
        c2, l2 = kmeans(X, 5)
        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_array_equal(l1, l2)

    def test_k_capped_at_n(self):
        X = _matrix(3, 4)
        centers, labels = kmeans(X, 10)
        assert centers.shape == (3, 4)
        assert labels.shape == (3,)

    def test_k_one_all_labels_zero(self):
        X = _matrix(10, 4)
        centers, labels = kmeans(X, 1)
        assert centers.shape == (1, 4)
        assert (labels == 0).all()

    def test_labels_within_k(self):
        X = _matrix(40, 6)
        _, labels = kmeans(X, 7)
        assert set(np.unique(labels)) <= set(range(7))


class TestChooseK:
    @pytest.mark.parametrize(
        ("n", "expected"),
        [(1, 3), (2, 3), (30, 3), (50, 5), (100, 10), (500, 10)],
    )
    def test_formula(self, n, expected):
        assert choose_k(n) == expected


# --- split_sentences ---------------------------------------------------------


class TestSplitSentences:
    def test_splits_on_punctuation_and_newlines(self):
        text = "Первое. Второе! Третье?\nЧетвёртое… Пятое."
        assert split_sentences(text) == [
            "Первое.",
            "Второе!",
            "Третье?",
            "Четвёртое…",
            "Пятое.",
        ]

    def test_strips_and_drops_empty(self):
        assert split_sentences("  А.   Б.  \n\n ") == ["А.", "Б."]

    def test_empty(self):
        assert split_sentences("") == []
        assert split_sentences("   \n  ") == []


# --- Embedder ----------------------------------------------------------------


def _fake_embedding(text: str) -> list[float]:
    """A per-text vector the fake API returns; row identity is checkable."""
    idx = int(text.rsplit("-", 1)[1])
    return [float(idx), idx * 0.5]


class FakeOpenAI:
    """Stands in for `openai.OpenAI`; returns data in REVERSED order to
    prove the embedder restores order via `.index`."""

    instances: list["FakeOpenAI"] = []

    def __init__(self, **_kwargs):
        FakeOpenAI.instances.append(self)
        self.batches: list[list[str]] = []
        api = self

        def create(**kwargs):
            batch = list(kwargs["input"])
            api.batches.append(batch)
            # Data comes back in reversed order, but each entry's `.index`
            # is the text's TRUE position in the batch.
            data = [
                SimpleNamespace(index=len(batch) - 1 - i, embedding=_fake_embedding(t))
                for i, t in enumerate(reversed(batch))
            ]
            return SimpleNamespace(data=data)

        self.embeddings = SimpleNamespace(create=create)


@pytest.fixture
def fake_openai(monkeypatch):
    import openai

    FakeOpenAI.instances = []
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    return FakeOpenAI


def _settings() -> EmbedSettings:
    return EmbedSettings(_env_file=None, url="http://e/v1", model="m")


class TestEmbedder:
    def test_batches_of_64_and_order_preserved(self, fake_openai):
        embedder = Embedder(_settings())
        texts = [f"текст-{i}" for i in range(130)]

        vecs = embedder.embed(texts)

        fake = fake_openai.instances[0]
        assert [len(b) for b in fake.batches] == [64, 64, 2]
        assert len(texts) == 130
        assert vecs.shape == (130, 2)
        assert vecs.dtype == np.float32
        # Row i corresponds to texts[i] even though data came back reversed.
        for i in range(130):
            np.testing.assert_array_equal(vecs[i], _fake_embedding(texts[i]))

    def test_single_batch(self, fake_openai):
        embedder = Embedder(_settings())
        vecs = embedder.embed(["текст-0", "текст-1"])
        assert [len(b) for b in fake_openai.instances[0].batches] == [2]
        assert vecs.shape == (2, 2)

    def test_empty_input(self, fake_openai):
        embedder = Embedder(_settings())
        vecs = embedder.embed([])
        assert vecs.shape == (0, 0)
        assert fake_openai.instances[0].batches == []

    def test_error_wrapped_in_summarization_error(self, monkeypatch):
        class BoomOpenAI:
            def __init__(self, **_kwargs):
                self.embeddings = SimpleNamespace(create=self._create)

            def _create(self, **_kwargs):
                raise RuntimeError("connection refused")

        import openai

        monkeypatch.setattr(openai, "OpenAI", BoomOpenAI)
        embedder = Embedder(_settings())
        with pytest.raises(SummarizationError, match="эмбеддингов"):
            embedder.embed(["текст-0"])

    def test_cancel_event_checked_between_batches(self, fake_openai):
        embedder = Embedder(_settings())
        cancel = threading.Event()
        cancel.set()
        with pytest.raises(SummarizationCancelled):
            embedder.embed([f"текст-{i}" for i in range(130)], cancel_event=cancel)
        # No request was issued at all.
        assert fake_openai.instances[0].batches == []

    def test_cancel_midway_stops_further_batches(self, fake_openai):
        embedder = Embedder(_settings())
        fake = fake_openai.instances[0]
        cancel = threading.Event()
        original_create = fake.embeddings.create

        def create(**kwargs):
            resp = original_create(**kwargs)
            cancel.set()  # cancel right after the first batch
            return resp

        fake.embeddings.create = create
        with pytest.raises(SummarizationCancelled):
            embedder.embed([f"текст-{i}" for i in range(130)], cancel_event=cancel)
        assert len(fake.batches) == 1

    def test_batch_size_constant(self):
        assert EMBED_BATCH_SIZE == 64
