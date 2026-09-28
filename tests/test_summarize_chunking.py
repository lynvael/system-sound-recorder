"""Tests for summarization chunking and segment merging.

Covers the degenerate-chunking regression: a single transcript line longer
than the chunk budget used to make `chunk_text` crawl forward one character
at a time, emitting O(overlap) nearly identical chunks. Now such lines are
hard-cut at the budget, and `merge_consecutive` can cap merged line length
via `max_chars` so long monologues break at turn boundaries.
"""

from app.pipeline.transcript import Segment, merge_consecutive
from app.summarize.chunking import build_transcript_text, chunk_text


def _seg(start: float, end: float, speaker: str, text: str) -> Segment:
    return Segment(start=start, end=end, speaker=speaker, text=text)


class TestChunkText:
    def test_empty_text_returns_empty_list(self):
        assert chunk_text("", 1000, 200) == []
        assert chunk_text("   \n  ", 1000, 200) == []

    def test_nonpositive_budget_returns_whole_text(self):
        assert chunk_text("abc", 0, 100) == ["abc"]
        assert chunk_text("abc", -5, 100) == ["abc"]

    def test_short_lines_split_on_line_boundaries_with_overlap(self):
        # Regression guard: behaviour on "normal" texts (all lines shorter
        # than the budget) must stay unchanged — line-boundary cuts with the
        # configured overlap.
        text = "A" * 50 + "\n" + "B" * 50 + "\n" + "C" * 50
        chunks = chunk_text(text, chunk_chars=80, overlap=20)
        # First chunk ends at the line boundary after 50 chars; each next
        # chunk starts `overlap` (20) chars before the previous cut.
        assert chunks == [
            "A" * 50,
            "A" * 20 + "\n" + "B" * 50,
            "B" * 20 + "\n" + "C" * 50,
        ]

    def test_single_line_longer_than_budget_hard_cuts(self):
        text = "short\n" + "x" * 25_000
        chunks = chunk_text(text, chunk_chars=1000, overlap=400)
        # O(len/budget) chunks (~41), not O(overlap) degenerate ones.
        assert len(chunks) <= 60
        # No degenerate slivers: every chunk keeps a meaningful size.
        assert all(len(c) >= 100 for c in chunks)
        # No text lost: the leading and trailing characters survive.
        assert chunks[0].startswith("short")
        assert chunks[-1].endswith("x")

    def test_long_single_line_covers_all_markers(self):
        # Overlapping chunks must cover the whole text: every 6-char marker
        # (longer than nothing, shorter than the overlap) appears verbatim.
        text = "".join(f"m{i:04d}" for i in range(2500))  # 15000 chars
        chunks = chunk_text(text, chunk_chars=1000, overlap=200)
        assert len(chunks) <= 30
        covered = "".join(chunks)
        for i in range(2500):
            assert f"m{i:04d}" in covered

    def test_line_exactly_at_budget_boundary(self):
        # A line exactly equal to the budget must not degenerate either.
        assert chunk_text("a" * 1000, chunk_chars=1000, overlap=200) == ["a" * 1000]

        text = "a" * 1000 + "\n" + "b" * 1000
        chunks = chunk_text(text, chunk_chars=1000, overlap=200)
        assert len(chunks) <= 4
        assert all(len(c) >= 100 for c in chunks)


class TestMergeConsecutive:
    def test_default_merges_all_consecutive_same_speaker(self):
        segs = [
            _seg(0, 1, "A", "one"),
            _seg(1, 2, "A", "two"),
            _seg(2, 3, "B", "bee"),
            _seg(3, 4, "A", "three"),
            _seg(4, 5, "A", "four"),
        ]
        merged = merge_consecutive(segs)
        assert [(s.speaker, s.text) for s in merged] == [
            ("A", "one two"),
            ("B", "bee"),
            ("A", "three four"),
        ]
        assert (merged[0].start, merged[0].end) == (0, 2)
        assert (merged[2].start, merged[2].end) == (3, 5)

    def test_max_chars_merges_until_limit_then_starts_new(self):
        segs = [
            _seg(0, 1, "A", "1" * 400),
            _seg(1, 2, "A", "2" * 400),
            _seg(2, 3, "A", "3" * 400),
        ]
        # 400 + 1 + 400 = 801 <= 900 -> first two merge; 801 + 1 + 400 = 1202
        # > 900 -> the third starts a new merged segment.
        merged = merge_consecutive(segs, max_chars=900)
        assert [len(s.text) for s in merged] == [801, 400]
        assert (merged[0].start, merged[0].end) == (0, 2)
        assert (merged[1].start, merged[1].end) == (2, 3)

    def test_max_chars_stricter_than_any_pair_never_merges(self):
        segs = [
            _seg(0, 1, "A", "1" * 400),
            _seg(1, 2, "A", "2" * 400),
            _seg(2, 3, "A", "3" * 400),
        ]
        # 801 > 800: no pair can merge.
        merged = merge_consecutive(segs, max_chars=800)
        assert [len(s.text) for s in merged] == [400, 400, 400]

    def test_single_segment_longer_than_limit_not_split(self):
        merged = merge_consecutive([_seg(0, 1, "A", "x" * 5000)], max_chars=1000)
        assert len(merged) == 1
        assert merged[0].text == "x" * 5000

    def test_input_not_mutated(self):
        segs = [
            _seg(0, 1, "A", "one"),
            _seg(1, 2, "A", "two"),
        ]
        snapshot = [(s.start, s.end, s.speaker, s.text) for s in segs]
        merge_consecutive(segs, max_chars=10)
        assert [(s.start, s.end, s.speaker, s.text) for s in segs] == snapshot

    def test_alternating_speakers_never_merge(self):
        segs = [
            _seg(0, 1, "A", "a"),
            _seg(1, 2, "B", "b"),
            _seg(2, 3, "A", "c"),
            _seg(3, 4, "B", "d"),
        ]
        merged = merge_consecutive(segs, max_chars=10_000)
        assert [s.speaker for s in merged] == ["A", "B", "A", "B"]
        assert [s.text for s in merged] == ["a", "b", "c", "d"]


def test_integration_long_monologue_no_degeneration():
    # 30 segments of one speaker (~400 chars each) interleaved with short
    # lines of a second speaker. With a merge limit the monologue is cut at
    # turn boundaries, and chunking must stay sane: few chunks, all of a
    # reasonable size (no degenerate slivers).
    segs = []
    t = 0.0
    for i in range(30):
        segs.append(_seg(t, t + 2, "Спикер", "м" * 400))
        t += 2
        if i % 6 == 5:
            segs.append(_seg(t, t + 1, "Гость", "короткая реплика"))
            t += 1

    merged = merge_consecutive(segs, max_chars=4000)
    text = build_transcript_text(merged)
    # Merged lines respect the limit (plus the small "[mm:ss] speaker: " prefix).
    assert all(len(line) <= 4200 for line in text.splitlines())

    chunks = chunk_text(text, chunk_chars=5000, overlap=400)
    assert len(chunks) <= 8
    assert all(len(c) >= 100 for c in chunks)
