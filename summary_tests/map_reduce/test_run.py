"""Unit tests for summary_tests/map_reduce/run.py (no LLM calls)."""

import asyncio

import pytest

import run as run_module
from run import main, reduce_phase, split_text


def test_split_text_empty_and_short():
    assert split_text("", 32000, 500) == []
    assert split_text("короткий текст", 32000, 500) == ["короткий текст"]


def test_split_text_short_start_then_giant_sentence():
    """Regression: a short first chunk followed by one giant unpunctuated
    sentence (overlap longer than the chunk) used to loop forever in
    split_text: target = end - overlap went negative, next_pos fell back to
    it, and the next iteration restarted at the same position.

    Realistic for STT transcripts: a short intro sentence, then a long run
    without punctuation.
    """
    text = "abc. " + "x" * 40000
    chunks = split_text(text, chunk_size=32000, overlap=500)

    # SENTENCE_RE cuts the first sentence at "abc." (the space belongs to
    # the next sentence), so the chunks are:
    #   "abc." | " " + 31999 x's | 8501 x's
    assert len(chunks) == 3
    assert chunks[0] == "abc."
    assert chunks[1] == " " + "x" * 31999
    assert chunks[2] == "x" * 8501
    # The whole text is covered: the first chunk starts at the beginning,
    # the last one ends at the end, and no chunk exceeds the chunk size.
    assert text.startswith(chunks[0])
    assert text.endswith(chunks[2])
    assert all(len(c) <= 32000 for c in chunks)


def test_split_text_giant_sentence_alone():
    """One sentence longer than chunk_size: hard-split with overlap."""
    text = "x" * 40000
    chunks = split_text(text, chunk_size=32000, overlap=500)

    assert len(chunks) == 2
    assert chunks[0] == "x" * 32000
    assert chunks[1] == "x" * 8500
    assert text.endswith(chunks[1])


@pytest.mark.parametrize("argv", [
    ["run.py", "in.txt", "--chunk-size", "0"],
    ["run.py", "in.txt", "--chunk-size", "-5"],
    ["run.py", "in.txt", "--overlap", "-1"],
])
def test_main_rejects_bad_chunk_size_and_overlap(argv, monkeypatch, capsys):
    """parser.error -> SystemExit(2): chunk_size must be positive,
    overlap must be >= 0 (chunk_size=0 would hang split_text forever)."""
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "error:" in err


def test_reduce_phase_keeps_all_summaries_after_collapse(monkeypatch):
    """Regression: the collapse branch used to zip `summaries` positionally
    with `compressed`, but `compressed` contains only the oversized
    summaries. With [small, big] the big one was silently dropped from the
    final prompt (with [big, small] the small one was truncated by the zip).
    All summaries must reach the final LLM call."""
    small = "маленькое саммари"
    big = "B" * 200
    chunk_size = 100

    prompts: list[str] = []

    async def fake_llm_call(client, model, prompt, semaphore, counter):
        prompts.append(prompt)
        counter["calls"] += 1
        if prompt.startswith("Сжми"):
            return "B" * 50  # compressed, now fits into chunk_size
        return "FINAL"

    monkeypatch.setattr(run_module, "llm_call", fake_llm_call)
    semaphore = asyncio.Semaphore(4)
    counter = {"calls": 0}

    result = asyncio.run(
        reduce_phase(None, "model", [small, big], chunk_size, semaphore, counter)
    )

    # The loop terminated and the final step produced the structured summary.
    assert result == "FINAL"
    final_prompt = prompts[-1]
    # Both summaries are present in the final LLM call: the small one as-is
    # and the big one in its compressed form.
    assert small in final_prompt
    assert "B" * 50 in final_prompt
