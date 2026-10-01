"""Bounded-parallel fan-out helper shared by summarization strategies.

The sync `openai` client (httpx under the hood) is thread-safe, so map-style
fan-out (one LLM call per chunk) runs on a small pool of DAEMON worker
threads that pull `(index, item)` pairs from a shared queue. No asyncio: the
whole summarization pipeline is sync by design (ADR-002 D3).

The workers are deliberately daemon threads: at interpreter shutdown Python
must NOT join them. The stdlib `ThreadPoolExecutor` uses non-daemon workers
and its atexit hook even drains the queue, which would keep calling the LLM
for minutes after the process is exiting (and later `submit` fails with
"cannot schedule new futures after interpreter shutdown").
"""

from __future__ import annotations

import queue
import threading
from typing import Any, Callable, Sequence, TypeVar

from app.summarize.errors import SummarizationCancelled

T = TypeVar("T")

# Seconds between cancellation checks while waiting for results. Small enough
# that cancellation is felt "promptly" (well under a second), large enough not
# to spin the CPU.
_POLL_TIMEOUT = 0.25

# Sentinel put into the work queue to tell a worker to stop without draining
# the rest of the queue.
_STOP = object()


def _guarded(fn: Callable[[T], Any], item: T, cancel_event: threading.Event | None) -> Any:
    """Task wrapper: re-checks cancellation right before calling `fn`.

    A task may sit in the queue for a while after cancellation was requested;
    checking here guarantees no new LLM call is issued once the event is set.
    """
    if cancel_event is not None and cancel_event.is_set():
        raise SummarizationCancelled()
    return fn(item)


def parallel_map(
    items: Sequence[T],
    fn: Callable[[T], Any],
    *,
    max_workers: int,
    cancel_event: threading.Event | None = None,
    on_done: Callable[[int, int], None] | None = None,
) -> list[Any]:
    """Apply `fn` to every item with at most `max_workers` calls in flight.

    Results are returned in the SAME order as `items`.

    Cancellation: when `cancel_event` is set, no new calls are issued (each
    task re-checks the event before calling `fn`) and `SummarizationCancelled`
    is raised promptly — the call does NOT block on in-flight tasks. Rationale:
    an in-flight HTTP call cannot be interrupted from here (force-disconnecting
    would make the SDK retry it as a connection error, masking the
    cancellation), so in-flight tasks are simply ignored and their results
    discarded. The workers are daemon threads, so at interpreter shutdown they
    are abandoned rather than joined — no queue drain, no hidden LLM traffic.

    A task error propagates to the caller and stops the remaining tasks.

    `on_done(done, total)` is called from the calling thread after each
    completed task (safe to use for GUI progress).
    """
    total = len(items)
    results: list[Any] = [None] * total
    work: queue.Queue = queue.Queue()
    for i, item in enumerate(items):
        work.put((i, item))
    events: queue.Queue = queue.Queue()

    def worker() -> None:
        while True:
            try:
                task = work.get_nowait()
            except queue.Empty:
                return
            if task is _STOP:
                return
            index, item = task
            try:
                results[index] = _guarded(fn, item, cancel_event)
                events.put(("ok", None))
            except BaseException as exc:  # noqa: BLE001 - report to the caller
                events.put(("error", exc))

    workers = [
        threading.Thread(target=worker, daemon=True)
        for _ in range(max(1, int(max_workers)))
    ]
    for w in workers:
        w.start()

    try:
        done_count = 0
        while done_count < total:
            if cancel_event is not None and cancel_event.is_set():
                raise SummarizationCancelled()
            try:
                kind, payload = events.get(timeout=_POLL_TIMEOUT)
            except queue.Empty:
                continue
            if kind == "error":
                raise payload
            done_count += 1
            if on_done is not None:
                on_done(done_count, total)
        return results
    finally:
        # Stop the workers without waiting: on cancel or a task error the
        # remaining work is abandoned (daemon threads die with the process).
        # Drain the pending items BEFORE the stop sentinels so no worker can
        # pick up unprocessed work in the meantime — otherwise a task error
        # would silently drain the whole queue (hidden LLM/embeddings calls).
        try:
            while True:
                work.get_nowait()
        except queue.Empty:
            pass
        for _ in workers:
            work.put(_STOP)
