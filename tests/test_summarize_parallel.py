"""Unit tests for the bounded-parallel fan-out helper (`parallel_map`)."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from app.summarize.errors import SummarizationCancelled
from app.summarize.parallel import parallel_map


def test_preserves_order_with_out_of_order_completions():
    # Later items finish FIRST (inverted sleep) — results must still come
    # back in input order.
    def fn(item: int):
        time.sleep(0.01 * (10 - item))
        return item * 10

    assert parallel_map(list(range(10)), fn, max_workers=4) == [i * 10 for i in range(10)]


def test_never_exceeds_max_workers():
    lock = threading.Lock()
    state = {"inflight": 0, "max": 0}

    def fn(_item):
        with lock:
            state["inflight"] += 1
            state["max"] = max(state["max"], state["inflight"])
        time.sleep(0.05)
        with lock:
            state["inflight"] -= 1

    results = parallel_map(list(range(8)), fn, max_workers=3)
    assert len(results) == 8
    assert state["max"] <= 3, "concurrency bound violated"
    assert state["max"] >= 2, "expected real parallelism, got sequential"


def test_max_workers_below_one_clamped_to_one():
    assert parallel_map([1, 2], lambda x: x + 1, max_workers=0) == [2, 3]


def test_task_error_propagates():
    def fn(item):
        if item == 2:
            raise ValueError("bad item")
        return item

    with pytest.raises(ValueError, match="bad item"):
        parallel_map([0, 1, 2, 3], fn, max_workers=2)


def test_task_error_stops_remaining_tasks():
    """Regression: a task error must NOT drain the rest of the queue.

    With max_workers=2 and item 0 raising, at most the in-flight item
    (item 1) may additionally have started — never the 8 queued ones.
    """
    calls: list[int] = []

    def fn(item: int):
        calls.append(item)
        if item == 0:
            raise ValueError("boom")
        time.sleep(0.05)
        return item

    with pytest.raises(ValueError, match="boom"):
        parallel_map(list(range(10)), fn, max_workers=2)
    time.sleep(0.3)  # give a (buggy) queue drain time to run
    assert len(calls) <= 1 + 2, f"queue was drained after the error: {calls}"


def test_on_done_reports_progress_in_calling_thread():
    events: list[tuple[int, int]] = []
    parallel_map(
        list(range(5)),
        lambda x: x,
        max_workers=2,
        on_done=lambda done, total: events.append((done, total)),
    )
    assert events[0] == (1, 5)
    assert events[-1] == (5, 5)
    assert [d for d, _ in events] == list(range(1, 6))


def test_cancel_before_any_call():
    cancel = threading.Event()
    cancel.set()
    calls = []
    with pytest.raises(SummarizationCancelled):
        parallel_map([1, 2, 3], calls.append, max_workers=2, cancel_event=cancel)
    assert calls == []


def test_cancel_during_run_is_prompt_and_discards_results():
    cancel = threading.Event()
    started = threading.Event()

    def fn(_item):
        started.set()
        time.sleep(0.5)
        return "result"

    def run():
        return parallel_map(list(range(6)), fn, max_workers=2, cancel_event=cancel)

    box: dict = {}
    thread = threading.Thread(target=lambda: _capture(run, box), daemon=True)
    thread.start()
    started.wait(5)
    time.sleep(0.05)
    cancel.set()
    t0 = time.monotonic()
    thread.join(timeout=5)
    elapsed = time.monotonic() - t0

    assert not thread.is_alive(), "parallel_map hung after cancel"
    assert isinstance(box.get("error"), SummarizationCancelled)
    assert elapsed < 2.0, "cancel was not prompt"


def _capture(fn, box):
    try:
        box["result"] = fn()
    except BaseException as exc:  # noqa: BLE001
        box["error"] = exc


def test_interpreter_exit_abandons_workers_without_draining_queue(tmp_path):
    """Regression: at interpreter shutdown the workers must be abandoned,
    NOT joined or queue-drained. With the old ThreadPoolExecutor, 6×1 s
    tasks at max_workers=2 all ran (~3 s of hidden traffic, i.e. hidden LLM
    calls in production) before the process exited."""
    code = (
        "import pathlib, sys, threading, time\n"
        "from app.summarize.parallel import parallel_map\n"
        "out = pathlib.Path(sys.argv[1])\n"
        "def slow(i):\n"
        "    time.sleep(1.0)\n"
        "    (out / f'task-{i}').write_text('done')\n"
        "    return i\n"
        "t = threading.Thread(\n"
        "    target=lambda: parallel_map(list(range(6)), slow, max_workers=2),\n"
        "    daemon=True,\n"
        ")\n"
        "t.start()\n"
        "time.sleep(0.3)  # 2 tasks in flight, 4 still queued\n"
    )
    t0 = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=15,
        cwd=Path(__file__).resolve().parent.parent,  # `app` package root
    )
    elapsed = time.monotonic() - t0

    assert proc.returncode == 0, proc.stderr
    # The main thread exits after 0.3 s; a joined/drain shutdown would take
    # ≥3 s (all 6 tasks) — the process must be long gone well before that.
    assert elapsed < 2.5, (
        f"interpreter took {elapsed:.1f}s to exit: workers were joined "
        f"and/or the queue was drained (stderr={proc.stderr!r})"
    )
    # The 4 queued tasks must NOT have run; at most the 2 in-flight ones did.
    assert len(list(tmp_path.glob("task-*"))) <= 2
