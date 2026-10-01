"""Tests for the EACSS strategy (fake LLM client + fake embedder, no
network).

Covers: the stuff branch (no intermediate LLM calls, material = selected
sentences in original order), the overflow branch (map calls via
parallel_map), the EMBED_* backend guard, cancellation during embeddings,
and a full `run_summarization` pass.
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
from app.summarize.errors import SummarizationCancelled, SummarizationError
from app.summarize.strategies import STRATEGIES
from app.summarize.strategies import eacss
from app.summarize.strategies.base import StrategyContext


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


class GatedEmbedder:
    """Blocks until released, then honors cancel_event — mimics the real
    Embedder's between-batches cancellation check (a long in-flight batch)."""

    def __init__(self, started: threading.Event, release: threading.Event):
        self.started = started
        self.release = release
        self.calls = 0

    def close(self):
        pass

    def embed(self, texts, *, cancel_event=None):
        self.calls += 1
        self.started.set()
        self.release.wait(5)
        if cancel_event is not None and cancel_event.is_set():
            raise SummarizationCancelled()
        return FakeEmbedder().embed(texts)


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
    info = STRATEGIES["eacss"]
    assert info.requires_embeddings is True
    assert info.label == "EACSS (экстрактивно-абстрактивный)"
    assert list(STRATEGIES) == ["map_reduce", "eacss", "hierarchical"]


# --- stuff branch ------------------------------------------------------------


class TestStuffBranch:
    def test_no_llm_calls_material_is_selected_sentences(self):
        client = FakeClient()
        embedder = FakeEmbedder()
        text = _text(40)  # ~1.7k chars → one chunk, extracted fits easily

        result = eacss.condense(text, _ctx(client, embedder))

        assert client.calls == []  # the final call is the pipeline's job
        assert embedder.calls == 1
        assert "ключевые предложения" in result.framing.lower()

        lines = result.material.splitlines()
        assert lines[0] == "Ключевые предложения:"
        selected = [line for line in lines[1:] if line]
        # k = min(10, max(3, 40 // 10)) = 4 → at most one sentence per cluster.
        assert 1 <= len(selected) <= 4
        # Every selected line is an original sentence, in original order.
        positions = [text.find(s) for s in selected]
        assert all(p >= 0 for p in positions)
        assert positions == sorted(positions)


# --- overflow branch ---------------------------------------------------------


class TestOverflowBranch:
    def test_map_calls_and_reduce_material(self, monkeypatch):
        monkeypatch.setattr(eacss, "CHUNK_CHARS", 400)
        monkeypatch.setattr(eacss, "CHUNK_OVERLAP", 0)
        client = FakeClient()
        embedder = FakeEmbedder()
        text = _text(120)  # ~5k chars → many 400-char chunks

        result = eacss.condense(text, _ctx(client, embedder))

        # Extracted content overflows one chunk → MAP calls, and the reduce
        # call is the pipeline's `_finalize` (not made here).
        n_parts = len(client.calls)
        assert n_parts >= 2
        for call in client.calls:
            user = call["messages"][1]["content"]
            assert "Составь краткое саммари" in user
            assert call["reasoning_effort"] == "medium"

        material = result.material
        assert material.startswith("Саммари частей:")
        assert f"Часть 1 из {n_parts}" in material
        assert f"Часть {n_parts} из {n_parts}" in material
        assert "саммари ключевых частей" in result.framing.lower()

    def test_map_preserves_part_order(self, monkeypatch):
        monkeypatch.setattr(eacss, "CHUNK_CHARS", 400)
        monkeypatch.setattr(eacss, "CHUNK_OVERLAP", 0)

        def responder(kwargs, _n):
            # Echo the part's first sentence: a marker unique per part.
            content = kwargs["messages"][1]["content"].split("---\n")[1]
            return content.splitlines()[0]

        client = FakeClient(responder=responder)
        embedder = FakeEmbedder()
        text = _text(120)
        result = eacss.condense(text, _ctx(client, embedder))

        # Recompute the expected parts (fakes are deterministic).
        from app.summarize.chunking import chunk_text
        from app.summarize.extractive import split_sentences

        ctx = _ctx(client, embedder)
        selected = [
            eacss._extract_from_chunk(split_sentences(c), ctx)
            for c in chunk_text(text, 400, 0)
        ]
        extracted = "\n\n".join("\n".join(s) for s in selected if s)
        first_lines = [
            p.splitlines()[0] for p in chunk_text(extracted, 400, 0)
        ]

        n_parts = len(client.calls)
        assert n_parts == len(first_lines)
        material = result.material
        positions = [material.find(fl) for fl in first_lines]
        assert all(p >= 0 for p in positions)
        assert positions == sorted(positions)
        # Each part's echo sits under its own "Часть i" header.
        for i, fl in enumerate(first_lines, start=1):
            header = material.find(f"Часть {i} из {n_parts}:")
            next_header = (
                material.find(f"Часть {i + 1} из {n_parts}:")
                if i < n_parts
                else len(material)
            )
            assert header < positions[i - 1] < next_header


# --- run_summarization integration -------------------------------------------


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
                options=pipeline.SummarizationOptions(strategy="eacss"),
            )
        assert client.calls == []
        assert list(session.glob("summary*.docx")) == []

    def test_success_writes_docx_with_one_final_call(self, tmp_path, monkeypatch):
        client = FakeClient()
        embedder = FakeEmbedder()
        monkeypatch.setattr(pipeline, "build_client", lambda llm: client)
        monkeypatch.setattr(pipeline, "Embedder", lambda settings: embedder)
        cfg = Config()
        cfg.llm = LLMSettings(_env_file=None)
        cfg.embed = EmbedSettings(_env_file=None, url="http://e/v1", model="m")
        session = _write_session(tmp_path)

        path = pipeline.run_summarization(
            session,
            cfg,
            options=pipeline.SummarizationOptions(strategy="eacss"),
        )
        assert path.name.startswith("summary_eacss_")
        # Stuff case: zero intermediate calls, exactly ONE final call.
        assert len(client.calls) == 1
        user = client.calls[0]["messages"][1]["content"]
        assert "Ключевые предложения:" in user
        assert "Требования к формату отчёта:" in user

    def test_cancel_during_embeddings(self, tmp_path, monkeypatch):
        client = FakeClient()
        embed_started = threading.Event()
        release = threading.Event()
        embedder = GatedEmbedder(embed_started, release)
        monkeypatch.setattr(pipeline, "build_client", lambda llm: client)
        monkeypatch.setattr(pipeline, "Embedder", lambda settings: embedder)
        cfg = Config()
        cfg.llm = LLMSettings(_env_file=None)
        cfg.embed = EmbedSettings(_env_file=None, url="http://e/v1", model="m")
        session = _write_session(tmp_path)

        cancel = threading.Event()
        started = threading.Event()

        def run():
            started.set()
            return pipeline.run_summarization(
                session,
                cfg,
                options=pipeline.SummarizationOptions(strategy="eacss"),
                cancel_event=cancel,
            )

        thread, box = _run_in_thread(run)
        started.wait(5)
        embed_started.wait(5)  # the first embed call is in flight now
        time.sleep(0.05)
        cancel.set()
        t0 = time.monotonic()
        thread.join(timeout=5)
        elapsed = time.monotonic() - t0
        release.set()  # release the blocked pool threads

        assert not thread.is_alive(), "pipeline hung after cancel"
        assert isinstance(box.get("error"), SummarizationCancelled)
        assert elapsed < 2.0, "cancel was not prompt"
        assert client.calls == []
        assert list(session.glob("summary*.docx")) == []
