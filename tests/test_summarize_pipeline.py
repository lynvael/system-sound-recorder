"""Tests for the strategy-based summarization pipeline (no network).

A FakeClient records every `chat.completions.create(**kwargs)` call and
returns canned Markdown, so tests can assert on the exact composition of
each LLM call (system/user messages, reasoning kwargs, absence of
response_format) and on the written docx.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from docx import Document

from app.config import Config, EmbedSettings, LLMSettings
from app.summarize import (
    DEFAULT_REPORT_PROMPT,
    SummarizationCancelled,
    SummarizationError,
    SummarizationOptions,
    find_latest_summary,
)
from app.summarize import pipeline
from app.summarize.chunking import build_transcript_text, chunk_text
from app.summarize.strategies import STRATEGIES, StrategyInfo
from app.pipeline.transcript import Segment


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


@pytest.fixture
def fake_client(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(pipeline, "build_client", lambda llm: client)
    return client


@pytest.fixture
def config():
    cfg = Config()
    # _env_file=None: tests must not depend on the repo's .env.
    cfg.llm = LLMSettings(_env_file=None, chunk_chars=300, chunk_overlap=0)
    cfg.embed = EmbedSettings(_env_file=None)
    return cfg


def _write_session(tmp_path: Path, segments: list[dict]) -> Path:
    session_dir = tmp_path / "20260523_144550_Тест"
    session_dir.mkdir()
    (session_dir / "transcript.json").write_text(
        json.dumps(segments, ensure_ascii=False), encoding="utf-8"
    )
    (session_dir / "transcript.txt").write_text(
        "текст стенограммы", encoding="utf-8"
    )
    return session_dir


def _seg(start: float, end: float, speaker: str, text: str) -> dict:
    return {"start": start, "end": end, "speaker": speaker, "text": text}


def _short_session(tmp_path: Path) -> Path:
    """A transcript that fits into a single chunk (no map phase)."""
    return _write_session(
        tmp_path,
        [
            _seg(0.0, 2.0, "Я", "Привет, начинаем планёрку по проекту."),
            _seg(2.0, 5.0, "Собеседники", "Решение: релиз в пятницу."),
        ],
    )


def _multi_chunk_session(tmp_path: Path) -> Path:
    """A transcript that needs several map chunks (300-char budget)."""
    segments = []
    t = 0.0
    for i in range(6):
        speaker = "Я" if i % 2 == 0 else "Собеседники"
        segments.append(_seg(t, t + 2.0, speaker, f"Реплика номер {i}: " + "слово " * 40))
        t += 2.0
    return _write_session(tmp_path, segments)


def _expected_chunk_count(session_dir: Path, llm: LLMSettings) -> int:
    segments = [
        Segment(start=s["start"], end=s["end"], speaker=s["speaker"], text=s["text"])
        for s in json.loads((session_dir / "transcript.json").read_text("utf-8"))
    ]
    from app.pipeline.transcript import merge_consecutive

    text = build_transcript_text(merge_consecutive(segments, max_chars=llm.chunk_chars))
    return len(chunk_text(text, llm.chunk_chars, llm.chunk_overlap))


def _user_message(call: dict) -> str:
    return call["messages"][1]["content"]


def _system_message(call: dict) -> str:
    return call["messages"][0]["content"]


# --- default prompt path ----------------------------------------------------


class TestDefaultPath:
    def test_single_chunk_one_markdown_call(self, tmp_path, fake_client, config):
        session_dir = _short_session(tmp_path)
        path = pipeline.run_summarization(session_dir, config)

        assert len(fake_client.calls) == 1
        call = fake_client.calls[0]
        # Plain Markdown call: no structured output, reasoning kwargs present.
        assert "response_format" not in call
        assert call["reasoning_effort"] == "medium"
        assert call["extra_body"] == {"allowed_openai_params": ["reasoning_effort"]}
        # Markdown rules live in the fixed system part.
        assert "Формат ответа: Markdown" in _system_message(call)
        # Default prompt + transcript material in the user message.
        user = _user_message(call)
        assert DEFAULT_REPORT_PROMPT in user
        assert "Требования к формату отчёта:" in user
        assert "релиз в пятницу" in user
        # Single chunk: the final call gets the transcript directly.
        assert "Стенограмма:" in user
        assert "Конспекты фрагментов:" not in user

        # Timestamped per-method filename in the session dir.
        assert path.parent == session_dir
        assert re.fullmatch(r"summary_map_reduce_\d{8}_\d{6}\.docx", path.name)
        assert path.is_file()

    def test_docx_header_and_method_line(self, tmp_path, fake_client, config):
        session_dir = _short_session(tmp_path)
        path = pipeline.run_summarization(session_dir, config)
        doc = Document(str(path))
        texts = [p.text for p in doc.paragraphs]
        assert texts[0] == "Протокол встречи — 20260523_144550_Тест"
        meta = texts[1]
        # Named session: the timestamp prefix is not the whole dir name, so
        # the date falls back to the raw name (existing behaviour).
        assert "Дата: 20260523_144550_Тест" in meta
        assert "Реплик (после объединения): 2" in meta
        assert "Участники: Я, Собеседники" in meta
        assert "Метод: Map-Reduce" in meta

    def test_multi_chunk_map_then_final(self, tmp_path, fake_client, config):
        session_dir = _multi_chunk_session(tmp_path)
        n = _expected_chunk_count(session_dir, config.llm)
        assert n >= 3  # the fixture must actually exercise the map phase

        path = pipeline.run_summarization(session_dir, config)

        assert len(fake_client.calls) == n + 1
        for call in fake_client.calls:
            assert "response_format" not in call
            assert call["reasoning_effort"] == "medium"
            assert call["extra_body"] == {"allowed_openai_params": ["reasoning_effort"]}

        # Map calls: one per chunk, numbered. Map calls run in parallel, so
        # their append order is not guaranteed — compare the SET of part
        # numbers over all non-final calls instead of positional access.
        part_numbers: set[int] = set()
        for call in fake_client.calls[:-1]:
            m = re.search(r"часть (\d+) из (\d+)", _user_message(call))
            assert m is not None
            assert m.group(2) == str(n)
            part_numbers.add(int(m.group(1)))
        assert part_numbers == set(range(1, n + 1))

        # Final call: joined chunk notes, in order.
        final_user = _user_message(fake_client.calls[-1])
        assert "Конспекты фрагментов:" in final_user
        assert "Требования к формату отчёта:" in final_user
        assert path.name.startswith("summary_map_reduce_")

    def test_map_results_preserve_chunk_order(self, tmp_path, fake_client, config):
        # Responder echoes the chunk index, so the final material must list
        # the notes in 1..N order even though map calls run in parallel.
        def responder(kwargs, _n):
            user = kwargs["messages"][1]["content"]
            m = re.search(r"часть (\d+) из", user)
            if m:
                return f"конспект-часть-{m.group(1)}"
            return "# Итог\n"

        fake_client._responder = responder
        session_dir = _multi_chunk_session(tmp_path)
        n = _expected_chunk_count(session_dir, config.llm)
        pipeline.run_summarization(session_dir, config)

        final_user = _user_message(fake_client.calls[-1])
        positions = [final_user.find(f"конспект-часть-{i}") for i in range(1, n + 1)]
        assert all(p >= 0 for p in positions)
        assert positions == sorted(positions)

    def test_on_status_progress(self, tmp_path, fake_client, config):
        session_dir = _multi_chunk_session(tmp_path)
        n = _expected_chunk_count(session_dir, config.llm)
        statuses: list[str] = []
        pipeline.run_summarization(session_dir, config, on_status=statuses.append)
        assert statuses[0] == "Саммаризация: чтение стенограммы…"
        assert "Саммаризация завершена." in statuses
        assert any(
            s == f"Саммаризация: обработано фрагментов {n}/{n}…" for s in statuses
        )

    def test_transcript_files_unchanged(self, tmp_path, fake_client, config):
        session_dir = _short_session(tmp_path)
        before = {
            name: (session_dir / name).read_bytes()
            for name in ("transcript.json", "transcript.txt")
        }
        pipeline.run_summarization(session_dir, config)
        for name, data in before.items():
            assert (session_dir / name).read_bytes() == data


# --- custom prompt ----------------------------------------------------------


class TestCustomPrompt:
    def test_custom_prompt_appended_after_material(self, tmp_path, fake_client, config):
        custom = "Оформи отчёт как сводку: сначала риски, потом решения."
        session_dir = _short_session(tmp_path)
        path = pipeline.run_summarization(
            session_dir,
            config,
            options=SummarizationOptions(custom_prompt=custom),
        )
        call = fake_client.calls[-1]
        user = _user_message(call)
        # Order: framing + material, THEN the format requirements, THEN the
        # user's prompt verbatim.
        material_pos = user.find("Стенограмма:")
        req_pos = user.find("Требования к формату отчёта:")
        prompt_pos = user.find(custom)
        assert 0 <= material_pos < req_pos < prompt_pos
        assert DEFAULT_REPORT_PROMPT not in user
        assert "Формат ответа: Markdown" in _system_message(call)
        assert path.is_file()

    def test_custom_prompt_with_braces_is_safe(self, tmp_path, fake_client, config):
        # Braces must survive: the user text is concatenated, never .format()-ed.
        custom = "Верни JSON-пример {summaries} {0} {x} {chunk} как есть."
        session_dir = _short_session(tmp_path)
        pipeline.run_summarization(
            session_dir, config, options=SummarizationOptions(custom_prompt=custom)
        )
        assert custom in _user_message(fake_client.calls[-1])

    def test_blank_custom_prompt_rejected_without_llm_calls(
        self, tmp_path, fake_client, config
    ):
        session_dir = _short_session(tmp_path)
        with pytest.raises(SummarizationError, match="Пользовательский промпт пуст"):
            pipeline.run_summarization(
                session_dir, config, options=SummarizationOptions(custom_prompt="   ")
            )
        assert fake_client.calls == []


# --- validation -------------------------------------------------------------


class TestValidation:
    def test_unknown_strategy_rejected_without_llm_calls(
        self, tmp_path, fake_client, config
    ):
        session_dir = _short_session(tmp_path)
        with pytest.raises(SummarizationError, match="Неизвестный метод"):
            pipeline.run_summarization(
                session_dir, config, options=SummarizationOptions(strategy="nope")
            )
        assert fake_client.calls == []

    def test_missing_session_dir(self, tmp_path, fake_client, config):
        with pytest.raises(SummarizationError, match="Каталог сессии не найден"):
            pipeline.run_summarization(tmp_path / "нет", config)
        assert fake_client.calls == []

    def test_missing_transcript(self, tmp_path, fake_client, config):
        session_dir = tmp_path / "20260523_144550"
        session_dir.mkdir()
        with pytest.raises(SummarizationError, match="Стенограмма не найдена"):
            pipeline.run_summarization(session_dir, config)
        assert fake_client.calls == []

    def test_embeddings_guard(self, tmp_path, fake_client, config, monkeypatch):
        # Register a fake embeddings-requiring strategy in the pipeline's
        # registry (part-2 strategies will do this for real).
        strategies = dict(STRATEGIES)
        strategies["fake_embed"] = StrategyInfo(
            id="fake_embed",
            label="Фейк (эмбеддинги)",
            requires_embeddings=True,
            condense=lambda text, ctx: SimpleNamespace(framing="f", material="m"),
        )
        monkeypatch.setattr(pipeline, "STRATEGIES", strategies)

        session_dir = _short_session(tmp_path)
        # Not configured → error before any network call.
        assert not config.embed.is_configured
        with pytest.raises(SummarizationError, match="эмбеддингов"):
            pipeline.run_summarization(
                session_dir, config, options=SummarizationOptions(strategy="fake_embed")
            )
        assert fake_client.calls == []

        # Configured → the guard passes (the fake strategy runs to completion).
        config.embed = EmbedSettings(_env_file=None, url="http://e/v1", model="m")
        path = pipeline.run_summarization(
            session_dir, config, options=SummarizationOptions(strategy="fake_embed")
        )
        assert path.is_file()
        assert path.name.startswith("summary_fake_embed_")


# --- error wrapping ---------------------------------------------------------


def test_client_error_wrapped_in_summarization_error(tmp_path, fake_client, config):
    def responder(_kwargs, _n):
        raise RuntimeError("boom: connection reset")

    fake_client._responder = responder
    session_dir = _short_session(tmp_path)
    with pytest.raises(SummarizationError, match="Ошибка обращения к LLM.*boom"):
        pipeline.run_summarization(session_dir, config)


def test_docx_write_failure_wrapped(tmp_path, fake_client, config, monkeypatch):
    # Any renderer failure (not only OSError) must respect the
    # SummarizationError contract, not escape as a raw exception.
    def boom(*_args, **_kwargs):
        raise ValueError("corrupted style")

    monkeypatch.setattr(pipeline, "write_markdown_docx", boom)
    session_dir = _short_session(tmp_path)
    with pytest.raises(SummarizationError, match="Не удалось сохранить.*corrupted style"):
        pipeline.run_summarization(session_dir, config)
    assert list(session_dir.glob("summary*.docx")) == []


def test_empty_llm_response_wrapped(tmp_path, fake_client, config):
    # The responder is called as (kwargs, n); a blank answer must surface the
    # client's own "пустой текст" error, not a signature TypeError.
    fake_client._responder = lambda _kw, _n: "   "
    session_dir = _short_session(tmp_path)
    with pytest.raises(SummarizationError, match="пустой текст"):
        pipeline.run_summarization(session_dir, config)


# --- cancellation -----------------------------------------------------------


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


class TestCancellation:
    def test_cancel_before_run(self, tmp_path, fake_client, config):
        session_dir = _short_session(tmp_path)
        cancel = threading.Event()
        cancel.set()
        with pytest.raises(SummarizationCancelled) as exc_info:
            pipeline.run_summarization(session_dir, config, cancel_event=cancel)
        assert str(exc_info.value) == "Саммаризация отменена."
        assert fake_client.calls == []
        assert list(session_dir.glob("summary*.docx")) == []

    def _wait_for_calls(self, client: FakeClient, n: int = 1) -> None:
        """Bounded wait until at least `n` LLM calls have been recorded."""
        deadline = time.monotonic() + 5
        while len(client.calls) < n and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(client.calls) >= n, "LLM call did not start in time"

    def test_cancel_during_map(self, tmp_path, config, monkeypatch):
        client = FakeClient(delay=0.2)
        monkeypatch.setattr(pipeline, "build_client", lambda llm: client)
        config.llm = LLMSettings(
            _env_file=None, chunk_chars=300, chunk_overlap=0, concurrency=2
        )
        session_dir = _multi_chunk_session(tmp_path)
        assert _expected_chunk_count(session_dir, config.llm) >= 3

        cancel = threading.Event()

        def run():
            return pipeline.run_summarization(
                session_dir, config, cancel_event=cancel
            )

        thread, box = _run_in_thread(run)
        self._wait_for_calls(client)  # the map phase is running now
        n_at_cancel = len(client.calls)
        cancel.set()
        t0 = time.monotonic()
        thread.join(timeout=5)
        elapsed = time.monotonic() - t0

        assert not thread.is_alive(), "pipeline hung after cancel"
        assert isinstance(box.get("error"), SummarizationCancelled)
        assert elapsed < 2.0, "cancel was not prompt"
        # No new calls after cancel: at most the ones already in flight
        # (bounded by the concurrency) may still have started.
        assert len(client.calls) <= n_at_cancel + config.llm.concurrency
        # The final call must never have been made.
        assert all(
            "Требования к формату отчёта" not in _user_message(c)
            for c in client.calls
        )
        assert list(session_dir.glob("summary*.docx")) == []

    def test_cancel_during_final_call(self, tmp_path, config, monkeypatch):
        client = FakeClient(delay=0.5)
        monkeypatch.setattr(pipeline, "build_client", lambda llm: client)
        session_dir = _short_session(tmp_path)  # single chunk → straight to final

        cancel = threading.Event()

        def run():
            return pipeline.run_summarization(
                session_dir, config, cancel_event=cancel
            )

        thread, box = _run_in_thread(run)
        self._wait_for_calls(client)  # the final call is in flight now
        cancel.set()
        t0 = time.monotonic()
        thread.join(timeout=5)
        elapsed = time.monotonic() - t0

        assert not thread.is_alive(), "pipeline hung after cancel"
        assert isinstance(box.get("error"), SummarizationCancelled)
        assert elapsed < 2.0, "cancel was not prompt"
        assert len(client.calls) == 1  # nothing was called after cancel
        assert list(session_dir.glob("summary*.docx")) == []


# --- find_latest_summary ----------------------------------------------------


class TestFindLatestSummary:
    def test_empty_dir_returns_none(self, tmp_path):
        assert find_latest_summary(tmp_path) is None

    def test_legacy_only(self, tmp_path):
        legacy = tmp_path / "summary.docx"
        legacy.write_text("x")
        assert find_latest_summary(tmp_path) == legacy

    def test_newest_by_mtime_among_method_files(self, tmp_path):
        older = tmp_path / "summary_map_reduce_20260101_000000.docx"
        newer = tmp_path / "summary_map_reduce_20260102_000000.docx"
        other = tmp_path / "summary_eacss_20260103_000000.docx"
        for p in (older, newer, other):
            p.write_text("x")
        now = time.time()
        os_utime(older, now - 300)
        os_utime(newer, now - 100)
        os_utime(other, now - 200)
        assert find_latest_summary(tmp_path) == newer

    def test_legacy_newer_wins(self, tmp_path):
        method = tmp_path / "summary_map_reduce_20260101_000000.docx"
        legacy = tmp_path / "summary.docx"
        method.write_text("x")
        legacy.write_text("x")
        now = time.time()
        os_utime(method, now - 500)
        os_utime(legacy, now - 10)
        assert find_latest_summary(tmp_path) == legacy

    def test_non_docx_and_other_files_ignored(self, tmp_path):
        (tmp_path / "summary_map_reduce.txt").write_text("x")
        (tmp_path / "notes.docx").write_text("x")
        assert find_latest_summary(tmp_path) is None


def os_utime(path: Path, ts: float):
    import os

    os.utime(path, (ts, ts))


# --- package import hygiene -------------------------------------------------


def test_package_import_is_lazy_and_complete():
    import subprocess
    import sys

    code = (
        "import sys; import app.summarize; "
        "from app.summarize import (run_summarization, SummarizationOptions, "
        "SummarizationError, SummarizationCancelled, STRATEGIES, "
        "DEFAULT_STRATEGY_ID, DEFAULT_REPORT_PROMPT, find_latest_summary); "
        "assert 'openai' not in sys.modules; "
        "assert 'docx' not in sys.modules; "
        "assert DEFAULT_STRATEGY_ID == 'map_reduce'; "
        "assert list(STRATEGIES) == ['map_reduce', 'eacss', 'hierarchical']; "
        "assert issubclass(SummarizationCancelled, SummarizationError)"
    )
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
