"""Tests for the hierarchical (Extract-Support) strategy — fake LLM client +
fake embedder, no network.

Covers: group_nodes limit, extractive_context (no embeddings under the cap,
capped selection), the 1-chunk fast path (zero intermediate calls), the
N-chunk path (LEVEL1 + merge levels, final via `_finalize`, supporting
context drawn from the source), both forced-final branches, and the
EMBED_* backend guard / cancellation during level 1.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.config import Config, EmbedSettings, LLMSettings
from app.summarize import pipeline
from app.summarize.chunking import chunk_text
from app.summarize.errors import SummarizationCancelled, SummarizationError
from app.summarize.strategies import STRATEGIES
from app.summarize.strategies import hierarchical
from app.summarize.strategies.base import StrategyContext
from app.summarize.strategies.hierarchical import Node


# --- fakes ------------------------------------------------------------------


class FakeClient:
    """Stands in for the sync OpenAI client; records every call."""

    def __init__(self, responder=None, delay: float = 0.0):
        self.calls: list[dict] = []
        self.delay = delay
        self._responder = responder or (lambda _kw, _n: "# Отчёт\n")
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.delay:
            time.sleep(self.delay)
        content = self._responder(kwargs, len(self.calls))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    def close(self):
        pass


class FakeEmbedder:
    """Deterministic hash-based embedder with the real Embedder's interface."""

    def __init__(self, dim: int = 32):
        self.dim = dim
        self.calls = 0

    def close(self):
        pass

    def embed(self, texts, *, cancel_event=None):
        self.calls += 1
        vecs = []
        for t in texts:
            seed = int.from_bytes(hashlib.md5(t.encode()).digest()[:4], "big")
            v = np.random.default_rng(seed).normal(size=self.dim)
            vecs.append(v / np.linalg.norm(v))
        return np.asarray(vecs, dtype=np.float32)


def _ctx(client, embedder, cancel_event=None, **llm_kwargs) -> StrategyContext:
    return StrategyContext(
        client=client,
        llm=LLMSettings(_env_file=None, **llm_kwargs),
        embedder=embedder,
        notify=lambda _msg: None,
        cancel_event=cancel_event,
    )


def _text(n_sentences: int) -> str:
    """Transcript-like text: one sentence per line, ~40 chars each."""
    return "\n".join(
        f"[00:{i // 10:02d}] Я: Предложение номер {i} о проекте."
        for i in range(n_sentences)
    )


def _user(call: dict) -> str:
    return call["messages"][1]["content"]


def _write_session(tmp_path: Path) -> Path:
    session_dir = tmp_path / "20260523_144550_Тест"
    session_dir.mkdir()
    (session_dir / "transcript.json").write_text(
        json.dumps(
            [
                {"start": 0.0, "end": 2.0, "speaker": "Я", "text": "Привет."},
                {"start": 2.0, "end": 5.0, "speaker": "Собеседники", "text": "Релиз в пятницу."},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return session_dir


def _run_in_thread(fn):
    box: dict = {}

    def target():
        try:
            box["result"] = fn()
        except BaseException as exc:  # noqa: BLE001 - capture for assertion
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread, box


# --- registry ----------------------------------------------------------------


def test_registered_with_embeddings_required():
    info = STRATEGIES["hierarchical"]
    assert info.requires_embeddings is True
    assert info.label == "Иерархический (Extract-Support)"


# --- group_nodes --------------------------------------------------------------


class TestGroupNodes:
    def test_respects_limit_and_order(self):
        nodes = [Node(summary="s" * 100, context="c" * 100) for _ in range(5)]
        groups = hierarchical.group_nodes(nodes, 500)
        # Each node is 200 chars; limit 500 → 2 per group → [2, 2, 1].
        assert [len(g) for g in groups] == [2, 2, 1]
        for group in groups:
            assert sum(len(n.summary) + len(n.context) for n in group) <= 500
        # Node order is preserved across groups.
        assert [id(n) for g in groups for n in g] == [id(n) for n in nodes]

    def test_single_oversized_node_gets_own_group(self):
        nodes = [Node(summary="s" * 1000, context="")]
        groups = hierarchical.group_nodes(nodes, 100)
        assert [len(g) for g in groups] == [1]


# --- extractive_context -------------------------------------------------------


class TestExtractiveContext:
    def test_no_embedder_below_cap(self):
        embedder = FakeEmbedder()
        ctx = _ctx(FakeClient(), embedder)
        text = "Первое. Второе. Третье."
        assert hierarchical.extractive_context(text, ctx) == text
        assert embedder.calls == 0

    def test_caps_at_max_sentences(self):
        embedder = FakeEmbedder()
        ctx = _ctx(FakeClient(), embedder)
        text = " ".join(f"Предложение {i}." for i in range(60))
        out = hierarchical.extractive_context(text, ctx, max_sentences=20)
        assert embedder.calls == 1
        assert 1 <= out.count("Предложение") <= 20
        # Selected sentences keep their original relative order.
        positions = [text.find(s) for s in out.split(". ") if s]
        assert positions == sorted(positions)

    def test_empty_text(self):
        embedder = FakeEmbedder()
        ctx = _ctx(FakeClient(), embedder)
        assert hierarchical.extractive_context("", ctx) == ""
        assert embedder.calls == 0


# --- 1-chunk fast path ---------------------------------------------------------


class TestSingleChunk:
    def test_zero_intermediate_calls(self):
        client = FakeClient()
        embedder = FakeEmbedder()
        text = _text(30)  # well under the 32000-char chunk

        result = hierarchical.condense(text, _ctx(client, embedder))

        assert client.calls == []
        assert embedder.calls == 0
        assert result.material == "Стенограмма:\n" + text.strip()
        assert "полная стенограмма" in result.framing


# --- N-chunk path --------------------------------------------------------------


def _multi_chunk_responder():
    def responder(kwargs, n):
        user = kwargs["messages"][1]["content"]
        if "Ниже приведён документ" in user:
            return f"Саммари-уровень1-{n}"
        return f"Саммари-merge-{n}"

    return responder


class TestMultiChunk:
    @pytest.fixture
    def small_method(self, monkeypatch):
        # Small chunk/group limit + small support context so the tree
        # actually merges several levels within the test's text size.
        monkeypatch.setattr(hierarchical, "CHUNK_CHARS", 600)
        monkeypatch.setattr(hierarchical, "CHUNK_OVERLAP", 0)
        monkeypatch.setattr(hierarchical, "MAX_CONTEXT_SENTENCES", 3)

    def test_level1_and_merge_levels(self, small_method):
        client = FakeClient(responder=_multi_chunk_responder())
        embedder = FakeEmbedder()
        text = _text(300)  # ~12k chars → 20 chunks of 600

        result = hierarchical.condense(text, _ctx(client, embedder))

        n_chunks = len(chunk_text(text, 600, 0))
        level1 = [c for c in client.calls if "Ниже приведён документ" in _user(c)]
        merges = [
            c for c in client.calls
            if "Ниже приведены саммари разных частей" in _user(c)
        ]
        assert len(level1) == n_chunks
        assert len(merges) >= 1  # at least one real merge level
        assert len(client.calls) == n_chunks + len(merges)

        # Every call carries the reasoning kwargs.
        for call in client.calls:
            assert call["reasoning_effort"] == "medium"

        # Level-1 prompts contain the source chunk (speakers/timestamps kept).
        # Level-1 calls run in parallel, so the first APPENDED call is not
        # necessarily chunk 0 — check the set of calls instead.
        assert any("Предложение номер 0" in _user(c) for c in level1)

        # Merge prompts carry supporting context drawn from the SOURCE text,
        # not from intermediate summaries.
        for call in merges:
            user = _user(call)
            assert "Предложение номер" in user
            assert "Саммари-уровень1" not in user.split(
                "поддерживающие контексты"
            )[1]

        # Final input: summaries + supporting source passages.
        material = result.material
        assert "Саммари частей:" in material
        assert "Саммари 1:" in material
        assert "Поддерживающие контексты:" in material
        context_part = material.split("Поддерживающие контексты:")[1]
        assert "Предложение номер" in context_part
        assert "Саммари-" not in context_part
        assert "вычитки" in result.framing

    def test_forced_final_when_nothing_fits(self, monkeypatch):
        # Chunks of 300 chars hold <= 20 sentences, so the support context is
        # the whole chunk → each node is bigger than the group limit → the
        # forced-final branch must fire (no merge calls at all).
        monkeypatch.setattr(hierarchical, "CHUNK_CHARS", 300)
        monkeypatch.setattr(hierarchical, "CHUNK_OVERLAP", 0)
        client = FakeClient(responder=_multi_chunk_responder())
        text = _text(90)  # ~3.6k chars → 12 chunks of 300

        result = hierarchical.condense(text, _ctx(client, FakeEmbedder()))

        n_chunks = len(chunk_text(text, 300, 0))
        level1 = [c for c in client.calls if "Ниже приведён документ" in _user(c)]
        merges = [
            c for c in client.calls
            if "Ниже приведены саммари разных частей" in _user(c)
        ]
        assert len(level1) == n_chunks
        assert merges == []
        assert f"Саммари {n_chunks}:" in result.material

    def test_max_merge_levels_guard(self, small_method, monkeypatch):
        monkeypatch.setattr(hierarchical, "MAX_MERGE_LEVELS", 1)
        client = FakeClient(responder=_multi_chunk_responder())
        text = _text(300)

        result = hierarchical.condense(text, _ctx(client, FakeEmbedder()))

        n_chunks = len(chunk_text(text, 600, 0))
        merges = [
            c for c in client.calls
            if "Ниже приведены саммари разных частей" in _user(c)
        ]
        assert len(client.calls) == n_chunks  # guard fired before any merge
        assert merges == []
        assert f"Саммари {n_chunks}:" in result.material


# --- run_summarization integration ---------------------------------------------


class TestRunSummarization:
    def test_without_embed_config_fails_before_network(
        self, tmp_path, monkeypatch
    ):
        client = FakeClient()
        monkeypatch.setattr(pipeline, "build_client", lambda llm: client)
        cfg = Config()
        cfg.llm = LLMSettings(_env_file=None)
        cfg.embed = EmbedSettings(_env_file=None)  # not configured
        assert not cfg.embed.is_configured
        session = _write_session(tmp_path)

        with pytest.raises(SummarizationError, match="эмбеддингов"):
            pipeline.run_summarization(
                session,
                cfg,
                options=pipeline.SummarizationOptions(strategy="hierarchical"),
            )
        assert client.calls == []
        assert list(session.glob("summary*.docx")) == []

    def test_cancel_during_level1(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hierarchical, "CHUNK_CHARS", 400)
        monkeypatch.setattr(hierarchical, "CHUNK_OVERLAP", 0)
        client = FakeClient(delay=0.3)
        embedder = FakeEmbedder()
        monkeypatch.setattr(pipeline, "build_client", lambda llm: client)
        monkeypatch.setattr(pipeline, "Embedder", lambda settings: embedder)
        cfg = Config()
        cfg.llm = LLMSettings(
            _env_file=None, concurrency=2
        )
        cfg.embed = EmbedSettings(_env_file=None, url="http://e/v1", model="m")
        session = _write_session(tmp_path)
        # Several 400-char chunks so level 1 has parallel work in flight.
        (session / "transcript.json").write_text(
            json.dumps(
                [
                    {
                        "start": float(i),
                        "end": float(i) + 1.0,
                        "speaker": "Я",
                        "text": f"Реплика {i}: " + "слово " * 40,
                    }
                    for i in range(6)
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        cancel = threading.Event()
        started = threading.Event()

        def run():
            started.set()
            return pipeline.run_summarization(
                session,
                cfg,
                options=pipeline.SummarizationOptions(strategy="hierarchical"),
                cancel_event=cancel,
            )

        thread, box = _run_in_thread(run)
        started.wait(5)
        # Wait for a real level-1 call to be in flight (no fixed sleep).
        deadline = time.monotonic() + 5
        while len(client.calls) < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(client.calls) >= 1, "level-1 call did not start in time"
        n_at_cancel = len(client.calls)
        cancel.set()
        t0 = time.monotonic()
        thread.join(timeout=5)
        elapsed = time.monotonic() - t0

        assert not thread.is_alive(), "pipeline hung after cancel"
        assert isinstance(box.get("error"), SummarizationCancelled)
        assert elapsed < 2.0, "cancel was not prompt"
        # No new calls after cancel: at most the ones already in flight
        # (bounded by the concurrency) may still have started — the merge
        # level and the final call must never have run.
        assert len(client.calls) <= n_at_cancel + cfg.llm.concurrency
        assert list(session.glob("summary*.docx")) == []
