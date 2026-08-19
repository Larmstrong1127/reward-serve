"""A minimal continuous batcher that honours the hook contract.

The real batcher belongs to the server-core agent (``reward_serve/batcher.py``).
This stub exists so the observability wiring can be tested and dry-run before
that lands, and so the hook call ordering the links depend on is pinned by
something executable rather than by a paragraph of prose.

It implements exactly what ``docs/contract.md`` section 3 promises and nothing
more:

* ``on_enqueue`` fires before the item can appear in a batch
* ``on_batch_start(batch_size, oldest_wait_ms)`` and
  ``on_batch_end(batch_size, forward_ms)`` fire once per forward pass,
  ``on_batch_end`` even when the forward pass raises
* ``on_complete(status)`` fires exactly once per accepted item
* items are packed FIFO, flushed at ``max_batch_size`` or ``max_wait_ms``
* a full queue rejects immediately with ``queue_full``
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

Forward = Callable[[Sequence[Any]], Sequence[float]]


def _noop(*_args: Any, **_kwargs: Any) -> None:
    return None


@dataclass
class _Item:
    payload: Any
    enqueued_at: float
    deadline: float
    future: asyncio.Future[float]


@dataclass
class StubBatcher:
    """FIFO batcher over an injected ``forward`` function."""

    forward: Forward
    max_batch_size: int = 8
    max_wait_ms: float = 10.0
    queue_depth: int = 256
    request_timeout_ms: float = 5000.0
    on_enqueue: Callable[[], None] = _noop
    on_batch_start: Callable[[int, float], None] = _noop
    on_batch_end: Callable[[int, float], None] = _noop
    on_complete: Callable[[str], None] = _noop

    _queue: list[_Item] = field(default_factory=list, repr=False)
    _task: asyncio.Task[None] | None = field(default=None, repr=False)
    _running: bool = field(default=False, repr=False)

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            await self._task
            self._task = None

    async def submit(self, payload: Any, timeout_ms: float | None = None) -> float:
        if len(self._queue) >= self.queue_depth:
            self.on_complete("queue_full")
            raise QueueFull("queue is at QUEUE_DEPTH")
        loop = asyncio.get_running_loop()
        now = time.perf_counter()
        item = _Item(
            payload=payload,
            enqueued_at=now,
            deadline=now + (timeout_ms or self.request_timeout_ms) / 1000.0,
            future=loop.create_future(),
        )
        self._queue.append(item)
        self.on_enqueue()
        return await item.future

    async def _loop(self) -> None:
        while self._running or self._queue:
            if not self._queue:
                await asyncio.sleep(0.001)
                continue
            oldest = self._queue[0]
            waited_ms = (time.perf_counter() - oldest.enqueued_at) * 1000.0
            if len(self._queue) < self.max_batch_size and waited_ms < self.max_wait_ms:
                await asyncio.sleep(0.001)
                continue
            self._flush()
            await asyncio.sleep(0)

    def _flush(self) -> None:
        now = time.perf_counter()
        expired = [i for i in self._queue if i.deadline <= now]
        for item in expired:
            self._queue.remove(item)
            self.on_complete("timeout")
            if not item.future.done():
                item.future.set_exception(DeadlineExceeded("deadline passed in queue"))
        if not self._queue:
            return

        batch = self._queue[: self.max_batch_size]
        del self._queue[: len(batch)]
        oldest_wait_ms = (now - batch[0].enqueued_at) * 1000.0
        self.on_batch_start(len(batch), oldest_wait_ms)

        started = time.perf_counter()
        error: Exception | None = None
        scores: Sequence[float] = ()
        try:
            scores = self.forward([i.payload for i in batch])
        except Exception as exc:  # forward failures still close the span
            error = exc
        forward_ms = (time.perf_counter() - started) * 1000.0
        self.on_batch_end(len(batch), forward_ms)

        for n, item in enumerate(batch):
            if error is not None:
                self.on_complete("cancelled")
                if not item.future.done():
                    item.future.set_exception(error)
            else:
                self.on_complete("ok")
                if not item.future.done():
                    item.future.set_result(scores[n])


class QueueFull(RuntimeError):
    """Raised instead of blocking when the queue is at QUEUE_DEPTH."""


class DeadlineExceeded(RuntimeError):
    """Raised when an item's deadline passed before a forward pass started."""
