"""Offline tests for the Context-Aware Hierarchical Merging pipeline.

Run with:
    uv run pytest summary_tests/hierarchical_context/ -v
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

from pipeline import (
    LEVEL1_PROMPT,
    MERGE_PROMPT,
    FakeEmbedder,
    LLMClient,
    Node,
    chunk_sentences,
    extractive_context,
    group_nodes,
    kmeans,
    load_transcript,
    run_pipeline,
    split_sentences,
)


class FakeLLM:
    """Mock LLM returning fixed-length summaries; counts calls."""

    def __init__(self, length: int = 120):
        self.calls = 0
        self.length = length

    async def complete(self, prompt: str) -> str:
        self.calls += 1
        return f"Саммари номер {self.calls}. " + "Факт. " * (self.length // 5)


# --- sentences / chunking ---------------------------------------------------

def test_split_sentences():
    sents = split_sentences("Привет. Как дела? Всё!… Хорошо\nВторой абзац.")
    assert sents == ["Привет.", "Как дела?", "Всё!…", "Хорошо", "Второй абзац."]


def test_chunking_respects_size_and_overlap():
    sentences = [f"Предложение номер {i} " + "x" * 80 + "." for i in range(200)]
    chunks = chunk_sentences(sentences, chunk_size=1000, overlap=150)
    assert len(chunks) > 1
    for c in chunks:
        # A single oversized sentence may exceed the limit; our sentences are small.
        assert len(c) <= 1000, f"chunk too long: {len(c)}"
    # Overlap: the first sentence of each next chunk must appear in the previous one.
    for prev, nxt in zip(chunks, chunks[1:]):
        first_sent = split_sentences(nxt)[0]
        assert first_sent in prev


def test_chunking_small_input_single_chunk():
    assert chunk_sentences(["Одно.", "Два."], 1000, 100) == ["Одно. Два."]


# --- k-means ----------------------------------------------------------------

def test_kmeans_finds_synthetic_clusters():
    rng = np.random.default_rng(0)
    blobs = [np.array(c) + rng.normal(0, 0.3, size=(50, 2)) for c in [(0, 0), (10, 10), (-10, 10)]]
    X = np.vstack(blobs)
    centers, assign = kmeans(X, 3)
    # Every true center must be close to some found center.
    for true in [(0, 0), (10, 10), (-10, 10)]:
        d = np.linalg.norm(centers - np.array(true), axis=1).min()
        assert d < 1.0, f"true center {true} not recovered"
    # Each cluster should be roughly homogeneous.
    for j in range(3):
        idx = np.where(assign == j)[0]
        assert len(idx) >= 40  # 50 points per blob, allow small splits


def test_kmeans_k_capped_by_n():
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    centers, assign = kmeans(X, k=10)
    assert centers.shape == (2, 2)
    assert set(assign.tolist()) <= {0, 1}


# --- merge tree grouping ----------------------------------------------------

def test_group_nodes_limit_and_order():
    nodes = [Node(summary="s" * 100, context="c" * 100) for _ in range(5)]  # 200 chars each
    groups = group_nodes(nodes, limit=500)
    assert sum(len(g) for g in groups) == 5
    for g in groups:
        assert sum(len(n.summary) + len(n.context) for n in g) <= 500
    # Order preserved.
    flat = [id(n) for g in groups for n in g]
    assert flat == [id(n) for n in nodes]


def test_group_nodes_single_oversized_node():
    nodes = [Node(summary="s" * 1000, context="")]
    assert group_nodes(nodes, limit=100) == [nodes]


# --- extractive context -----------------------------------------------------

def test_extractive_context_caps_and_subset():
    sents = [f"Предложение номер {i}. Уникальное слово {i}." for i in range(60)]
    text = " ".join(sents)
    ctx = extractive_context(text, FakeEmbedder(), max_sentences=20)
    chosen = split_sentences(ctx)
    assert len(chosen) <= 20
    assert len(chosen) > 0
    input_set = set(split_sentences(text))
    assert set(chosen) <= input_set  # extractive: only original sentences


def test_extractive_context_small_text_passthrough():
    text = "Первое. Второе."
    assert extractive_context(text, FakeEmbedder()) == text


# --- prompt templates -------------------------------------------------------

def test_level1_prompt_contains_chunk():
    rendered = LEVEL1_PROMPT.format(chunk="ЧАСТОТНЫЙ ФРАГМЕНТ")
    assert "ЧАСТОТНЫЙ ФРАГМЕНТ" in rendered
    assert "языке исходного текста" in rendered


def test_merge_prompt_contains_summaries_and_contexts():
    rendered = MERGE_PROMPT.format(summaries="С1\nС2", contexts="P1\nP2")
    assert "С1\nС2" in rendered
    assert "P1\nP2" in rendered
    # Paper semantics (Table 8): the gist comes solely from the summaries,
    # supporting contexts are used for proofreading only.
    assert "исключительно на приведённых саммари" in rendered
    assert "только для вычитки" in rendered


# --- full pipeline with mocks (merge tree) -----------------------------------

def test_pipeline_builds_tree_to_single_summary():
    # ~12k chars, chunk size 2000 -> ~6 chunks at level 1.
    text = " ".join(f"Предложение {i} " + "данные. " * 5 for i in range(300))
    llm = FakeLLM()
    embed = FakeEmbedder()
    summary, stats = asyncio.run(
        run_pipeline(text, chunk_size=2000, overlap=150, llm=llm, embed=embed)
    )
    assert summary  # non-empty final summary
    assert stats.chunks >= 3
    assert stats.levels >= 2
    # LLM calls = one per level-1 chunk + one per merge group at every level.
    assert llm.calls == stats.chunks + stats.merge_groups
    assert stats.llm_calls == llm.calls
    assert stats.merge_groups >= 1
    assert embed.requests > 0
    assert stats.embed_calls == embed.requests


def test_pipeline_converges_when_llm_does_not_compress():
    """Regression: a non-compressing LLM used to cause an infinite merge loop.

    The mock returns ~15000-char summaries (longer than chunk_size=5000), so
    after level 1 every node's (summary+context) size exceeds the grouping
    limit and group_nodes yields only single-node groups. The pipeline must
    force-merge all nodes in one group and terminate, not loop forever.
    """

    class NonCompressingLLM:
        def __init__(self):
            self.calls = 0

        async def complete(self, prompt: str) -> str:
            self.calls += 1
            return "Факт. " * 2500  # ~15000 chars, longer than chunk_size

    # ~20k chars, chunk_size 5000 -> ~4 chunks at level 1.
    text = " ".join(f"Предложение {i} " + "данные. " * 3 for i in range(500))
    llm = NonCompressingLLM()
    embed = FakeEmbedder()
    summary, stats = asyncio.run(
        asyncio.wait_for(
            run_pipeline(text, chunk_size=5000, overlap=200, llm=llm, embed=embed),
            timeout=60,
        )
    )
    assert summary
    assert stats.chunks >= 3
    # Level 2: nobody could be grouped, so all nodes were force-merged once.
    assert stats.merge_groups == 1
    assert llm.calls == stats.chunks + 1


def test_pipeline_raises_on_empty_llm_response():
    """Regression: an empty LLM response must abort the pipeline with
    RuntimeError, not silently converge to an empty final summary.

    The real LLMClient.complete() is exercised with a mocked OpenAI client
    whose endpoint returns an empty string.
    """

    class _Msg:
        content = ""

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    class _Completions:
        async def create(self, **kwargs):
            assert kwargs.get("max_tokens") == 1500
            return _Resp()

    class _Chat:
        completions = _Completions()

    class _FakeOpenAI:
        chat = _Chat()

        async def close(self):
            pass

    llm = LLMClient(base_url="http://127.0.0.1:9", api_key="k", model="m")
    llm.client = _FakeOpenAI()
    embed = FakeEmbedder()
    text = "Первое. Второе. Третье."
    with pytest.raises(RuntimeError, match="empty response"):
        asyncio.run(run_pipeline(text, chunk_size=2000, overlap=100, llm=llm, embed=embed))


# --- transcript loading ------------------------------------------------------

def test_load_transcript_txt_strips_timestamps(tmp_path: Path):
    p = tmp_path / "t.txt"
    p.write_text("[00:00] Привет.\n[01:02:03] Как дела?\n[12:34] Всё.\n", encoding="utf-8")
    assert load_transcript(p) == "Привет. Как дела? Всё."


def test_load_transcript_json_concatenates_text(tmp_path: Path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps([
        {"start": 0.0, "end": 1.0, "speaker": "A", "text": "Первое."},
        {"start": 1.0, "end": 2.0, "speaker": "B", "text": "Второе."},
    ]), encoding="utf-8")
    assert load_transcript(p) == "Первое. Второе."


def test_load_transcript_bad_extension(tmp_path: Path):
    p = tmp_path / "t.docx"
    p.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        load_transcript(p)
