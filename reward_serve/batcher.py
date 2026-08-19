"""Continuous batcher: a bounded queue plus one loop that packs waiting items.

Flush policy, deadline handling and cancellation are documented in
docs/design-batcher.md. The hook signatures are frozen by docs/contract.md.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

__all__ = [
    "BatcherHooks",
    "ContinuousBatcher",
    "DeadlineExceeded",
    "QueueFullError",
    "ScoreResult",
    "Scorer",
    "Status",
]


class Status(StrEnum):
    """Terminal state of an accepted item. Mirrors the contract."""

    OK = "ok"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    QUEUE_FULL = "queue_full"


class QueueFullError(Exception):
    """Raised by submit() when the queue is at capacity. Maps to HTTP 429."""


class DeadlineExceeded(Exception):
    """Raised when an item deadline passed before it was scored. Maps to 504."""


class Scorer(Protocol):
    """What the batcher needs from a model. See reward_serve.model.RewardModel."""

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """Score prompt/response pairs in one forward pass, in order."""
        ...


@dataclass(frozen=True)
class ScoreResult:
    """Per-item result plus the batch metadata the contract promises."""

    score: float
    batch_size: int
    queue_wait_ms: float
    forward_ms: float


def _noop() -> None:
    return None


def _noop_batch_start(batch_size: int, oldest_wait_ms: float) -> None:
    return None


def _noop_batch_end(batch_size: int, forward_ms: float) -> None:
    return None


def _noop_complete(status: str) -> None:
    return None


@dataclass
class BatcherHooks:
    """Observability seams. Every hook defaults to a no-op and must not block."""

    on_enqueue: Callable[[], None] = _noop
    on_batch_start: Callable[[int, float], None] = _noop_batch_start
    on_batch_end: Callable[[int, float], None] = _noop_batch_end
    on_complete: Callable[[str], None] = _noop_complete


@dataclass
class _Item:
    prompt: str
    response: str
    enqueued_at: float
    deadline: float
    future: asyncio.Future[ScoreResult]
    cancelled: bool = field(default=False)


class ContinuousBatcher:
    """One asyncio loop that drains a bounded queue into batched forward passes.

    A flush fires when either max_batch_size items are waiting or the oldest
    waiting item has waited max_wait_ms, whichever happens first.
    """

    def __init__(
        self,
        scorer: Scorer,
        *,
        max_batch_size: int = 8,
        max_wait_ms: float = 10.0,
        queue_depth: int = 256,
        request_timeout_ms: float = 5000.0,
        hooks: BatcherHooks | None = None,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if queue_depth < 1:
            raise ValueError("queue_depth must be >= 1")
        self._scorer = scorer
        self.max_batch_size = max_batch_size
        self.max_wait_s = max_wait_ms / 1000.0
        self.queue_depth = queue_depth
        self.request_timeout_s = request_timeout_ms / 1000.0
        self.hooks = hooks or BatcherHooks()

        self._queue: asyncio.Queue[_Item] = asyncio.Queue(maxsize=queue_depth)
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._accepting = False
        self._inflight = 0

    # -- lifecycle ---------------------------------------------------------

    @property
    def running(self) -> bool:
        """True while the batching loop is alive."""
        return self._task is not None and not self._task.done()

    @property
    def accepting(self) -> bool:
        """True while new items are accepted. False once a drain begins."""
        return self._accepting

    @property
    def depth(self) -> int:
        """Items currently waiting in the queue."""
        return self._queue.qsize()

    async def start(self) -> None:
        """Start the batching loop. Idempotent."""
        if self.running:
            return
        self._stop.clear()
        self._accepting = True
        self._task = asyncio.create_task(self._run(), name="batcher-loop")

    async def drain(self) -> None:
        """Stop accepting, let everything already queued finish, then stop."""
        self._accepting = False
        while self._queue.qsize() > 0 or self._inflight > 0:
            await asyncio.sleep(0.002)
        await self.stop()

    async def stop(self) -> None:
        """Stop the loop. Anything still queued is failed, never silently dropped."""
        self._accepting = False
        self._stop.set()
        task = self._task
        self._task = None
        if task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
            except (TimeoutError, asyncio.CancelledError):
                task.cancel()
        while not self._queue.empty():
            item = self._queue.get_nowait()
            self._fail(item, DeadlineExceeded("server stopped before scoring"), Status.TIMEOUT)

    # -- submission --------------------------------------------------------

    async def submit(
        self, prompt: str, response: str, timeout_ms: float | None = None
    ) -> ScoreResult:
        """Enqueue one pair and await its score.

        Raises QueueFullError at once when the queue is full, never blocking.
        Raises DeadlineExceeded when the deadline passes before a forward pass
        starts. Propagates CancelledError if the caller goes away.
        """
        if not self._accepting:
            raise QueueFullError("batcher is not accepting work")
        loop = asyncio.get_running_loop()
        now = loop.time()
        timeout_s = self.request_timeout_s if timeout_ms is None else timeout_ms / 1000.0
        item = _Item(
            prompt=prompt,
            response=response,
            enqueued_at=now,
            deadline=now + timeout_s,
            future=loop.create_future(),
        )
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            self.hooks.on_complete(Status.QUEUE_FULL.value)
            raise QueueFullError(f"queue is full at depth {self.queue_depth}") from None
        self.hooks.on_enqueue()
        try:
            return await item.future
        except asyncio.CancelledError:
            # The client went away. Mark it so the loop drops it before forwarding.
            item.cancelled = True
            self.hooks.on_complete(Status.CANCELLED.value)
            raise

    # -- the loop ----------------------------------------------------------

    async def _run(self) -> None:
        while not self._stop.is_set():
            first = await self._next_item()
            if first is None:
                continue
            batch = await self._fill(first)
            if batch:
                await self._forward(batch)

    async def _next_item(self) -> _Item | None:
        """Wait for one item while staying responsive to stop()."""
        getter: asyncio.Task[_Item] = asyncio.ensure_future(self._queue.get())
        stopper: asyncio.Task[bool] = asyncio.ensure_future(self._stop.wait())
        waiting: set[asyncio.Task[Any]] = {getter, stopper}
        try:
            await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
        finally:
            stopper.cancel()
        if getter.done():
            return getter.result()
        getter.cancel()
        return None

    async def _fill(self, first: _Item) -> list[_Item]:
        """Pack until max_batch_size items wait, or the oldest hits max_wait_ms."""
        loop = asyncio.get_running_loop()
        batch = [first]
        flush_at = first.enqueued_at + self.max_wait_s
        while len(batch) < self.max_batch_size:
            remaining = flush_at - loop.time()
            if remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(self._queue.get(), remaining))
            except TimeoutError:
                break
        return batch

    async def _forward(self, batch: list[_Item]) -> None:
        loop = asyncio.get_running_loop()
        now = loop.time()

        live: list[_Item] = []
        for item in batch:
            if item.cancelled or item.future.done():
                continue  # a cancelled item is never forwarded
            if now > item.deadline:
                self._fail(
                    item, DeadlineExceeded("deadline passed before scoring"), Status.TIMEOUT
                )
                continue
            live.append(item)
        if not live:
            return

        oldest_wait_ms = (now - min(i.enqueued_at for i in live)) * 1000.0
        self.hooks.on_batch_start(len(live), oldest_wait_ms)

        pairs = [(i.prompt, i.response) for i in live]
        self._inflight = len(live)
        started = loop.time()
        try:
            scores = await loop.run_in_executor(None, self._scorer.score_pairs, pairs)
        except Exception as exc:  # noqa: BLE001 - every waiter must learn about it
            forward_ms = (loop.time() - started) * 1000.0
            self._inflight = 0
            self.hooks.on_batch_end(len(live), forward_ms)
            for item in live:
                self._fail(item, exc, Status.TIMEOUT)
            return
        forward_ms = (loop.time() - started) * 1000.0
        self._inflight = 0
        self.hooks.on_batch_end(len(live), forward_ms)

        for item, score in zip(live, scores, strict=True):
            if item.future.done():
                continue
            item.future.set_result(
                ScoreResult(
                    score=float(score),
                    batch_size=len(live),
                    queue_wait_ms=(started - item.enqueued_at) * 1000.0,
                    forward_ms=forward_ms,
                )
            )
            self.hooks.on_complete(Status.OK.value)

    def _fail(self, item: _Item, exc: BaseException, status: Status) -> None:
        if not item.future.done():
            item.future.set_exception(exc)
            self.hooks.on_complete(status.value)
