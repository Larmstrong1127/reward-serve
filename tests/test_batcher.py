"""Batcher behaviour, against a stub scorer so the suite stays fast."""

from __future__ import annotations

import asyncio

import pytest
from conftest import StubScorer, wait_until

from reward_serve.batcher import (
    BatcherHooks,
    ContinuousBatcher,
    DeadlineExceeded,
    QueueFullError,
)


async def _started(scorer: StubScorer, **kw) -> ContinuousBatcher:
    b = ContinuousBatcher(scorer, **kw)
    await b.start()
    return b


async def test_flush_on_size_fires_before_the_wait_elapses(stub: StubScorer) -> None:
    """8 items submitted at once ride in one forward pass, not eight."""
    # max_wait is deliberately long: only the size trigger can flush this.
    b = await _started(stub, max_batch_size=8, max_wait_ms=10_000)
    try:
        results = await asyncio.gather(*(b.submit(f"p{i}", f"r{i}") for i in range(8)))
    finally:
        await b.stop()

    assert stub.batch_sizes == [8], f"expected one batch of 8, got {stub.batch_sizes}"
    assert all(r.batch_size == 8 for r in results)


async def test_flush_on_wait_fires_for_a_lone_item(stub: StubScorer) -> None:
    """A single item does not wait for a batch that will never fill."""
    # max_batch_size is unreachable: only the wait timer can flush this.
    b = await _started(stub, max_batch_size=64, max_wait_ms=15)
    try:
        result = await asyncio.wait_for(b.submit("lonely", "item"), timeout=2.0)
    finally:
        await b.stop()

    assert stub.batch_sizes == [1]
    assert result.batch_size == 1


async def test_expired_deadline_is_not_scored(stub: StubScorer) -> None:
    """An item whose deadline passes in the queue fails and never reaches the model."""
    slow = StubScorer(delay_s=0.15)
    b = await _started(slow, max_batch_size=1, max_wait_ms=1, request_timeout_ms=5000)
    try:
        # Occupy the single-item batcher so the next item must sit in the queue.
        blocker = asyncio.create_task(b.submit("blocker", "x"))
        await asyncio.sleep(0.02)
        with pytest.raises(DeadlineExceeded):
            await b.submit("doomed", "y", timeout_ms=10)
        await blocker
    finally:
        await b.stop()

    forwarded = [p for p, _ in slow.seen_pairs]
    assert "doomed" not in forwarded, f"expired item was scored anyway: {forwarded}"
    assert "blocker" in forwarded


async def test_cancelled_item_is_never_forwarded() -> None:
    """A client that disconnects has its item dropped before the forward pass."""
    slow = StubScorer(delay_s=0.15)
    b = await _started(slow, max_batch_size=1, max_wait_ms=1)
    try:
        blocker = asyncio.create_task(b.submit("blocker", "x"))
        await asyncio.sleep(0.02)
        victim = asyncio.create_task(b.submit("cancelled", "y"))
        await asyncio.sleep(0.02)  # let it land in the queue
        victim.cancel()
        with pytest.raises(asyncio.CancelledError):
            await victim
        await blocker
        await asyncio.sleep(0.1)
    finally:
        await b.stop()

    forwarded = [p for p, _ in slow.seen_pairs]
    assert "cancelled" not in forwarded, f"cancelled item was forwarded: {forwarded}"


async def test_full_queue_rejects_immediately_without_blocking() -> None:
    """Enqueue on a full queue raises at once rather than waiting for room."""
    slow = StubScorer(delay_s=0.3)
    b = await _started(slow, max_batch_size=1, max_wait_ms=1, queue_depth=2)
    pending: list[asyncio.Task[object]] = []
    try:
        # Occupy the loop first, so nothing is pulled while we fill the queue.
        pending.append(asyncio.create_task(b.submit("inflight", "x")))
        assert await wait_until(lambda: bool(slow.batches), timeout=1.0)

        for i in range(b.queue_depth):
            pending.append(asyncio.create_task(b.submit(f"fill{i}", "x")))
        assert await wait_until(lambda: b.depth == b.queue_depth, timeout=1.0)

        loop = asyncio.get_running_loop()
        before = loop.time()
        with pytest.raises(QueueFullError):
            await b.submit("rejected", "x")
        elapsed = loop.time() - before
        assert elapsed < 0.05, f"enqueue blocked for {elapsed:.3f}s instead of failing fast"
    finally:
        for t in pending:
            t.cancel()
        await b.stop()

    assert "rejected" not in [p for p, _ in slow.seen_pairs]


async def test_hooks_fire_with_the_contract_signatures(stub: StubScorer) -> None:
    """The observability seams fire with the arities docs/contract.md promises."""
    enqueued: list[None] = []
    starts: list[tuple[int, float]] = []
    ends: list[tuple[int, float]] = []
    completes: list[str] = []

    hooks = BatcherHooks(
        on_enqueue=lambda: enqueued.append(None),
        on_batch_start=lambda n, w: starts.append((n, w)),
        on_batch_end=lambda n, ms: ends.append((n, ms)),
        on_complete=completes.append,
    )
    b = await _started(stub, max_batch_size=4, max_wait_ms=10_000, hooks=hooks)
    try:
        await asyncio.gather(*(b.submit(f"p{i}", "r") for i in range(4)))
    finally:
        await b.stop()

    assert len(enqueued) == 4
    assert [n for n, _ in starts] == [4]
    assert [n for n, _ in ends] == [4]
    assert completes == ["ok"] * 4
    assert ends[0][1] >= 0.0


async def test_drain_completes_queued_work(stub: StubScorer) -> None:
    """Nothing already accepted is lost when a drain begins."""
    slow = StubScorer(delay_s=0.05)
    b = await _started(slow, max_batch_size=2, max_wait_ms=5)
    tasks = [asyncio.create_task(b.submit(f"p{i}", "r")) for i in range(6)]
    await wait_until(lambda: b.depth > 0 or slow.batches, timeout=1.0)

    await b.drain()
    results = await asyncio.gather(*tasks)

    assert len(results) == 6
    assert sum(len(x) for x in slow.batches) == 6
    assert not b.accepting
    with pytest.raises(QueueFullError):
        await b.submit("after", "drain")


async def test_backlog_is_batched_not_drained_one_at_a_time() -> None:
    """A queue that built up during a slow forward pass flushes as one batch.

    Regression: the wait window is anchored to the oldest item's enqueue time,
    so under a backlog that window has already expired. An implementation that
    only ever waits, and never sweeps what is already queued, forwards batches
    of one while a full queue sits behind it.
    """
    slow = StubScorer(delay_s=0.2)
    b = await _started(slow, max_batch_size=8, max_wait_ms=5, request_timeout_ms=30_000)
    try:
        # First item occupies the loop; the rest pile up behind it.
        head = asyncio.create_task(b.submit("head", "r"))
        assert await wait_until(lambda: bool(slow.batches), timeout=2.0)

        backlog = [asyncio.create_task(b.submit(f"q{i}", "r")) for i in range(6)]
        assert await wait_until(lambda: b.depth == 6, timeout=2.0)

        await head
        results = await asyncio.gather(*backlog)
    finally:
        await b.stop()

    assert slow.batch_sizes == [1, 6], (
        f"backlog was not batched: forward passes were {slow.batch_sizes}"
    )
    assert all(r.batch_size == 6 for r in results)


async def test_max_wait_survives_coarse_clock_deadline_arithmetic() -> None:
    """Coarse-clock deadline arithmetic must not collapse short wait windows.

    loop.time() advances in ~15.6 ms ticks on this box, so one tick after
    enqueue the computed remaining wait for a 5 ms window is already negative
    -- the arithmetic expires even though (measured against perf_counter)
    timers never fire early. Pre-fix, that flushed a lone item after ~0.18 ms
    mean and every small wait setting behaved identically. The batcher must
    re-check the deadline and hold the item; with the re-check, sub-resolution
    waits floor at ~one clock tick instead of collapsing to zero. We use a
    small window (5 ms) because that is where the bug lives, and average over
    trials because a single hold is quantised to the clock.
    """
    import time

    from reward_serve.batcher import ContinuousBatcher

    seen: list[float] = []

    class _Scorer:
        def score_pairs(self, pairs):  # type: ignore[no-untyped-def]
            seen.append(time.perf_counter())
            return [0.0 for _ in pairs]

    b = ContinuousBatcher(_Scorer(), max_batch_size=8, max_wait_ms=5, queue_depth=16)
    await b.start()
    holds: list[float] = []
    try:
        for _ in range(20):
            seen.clear()
            t0 = time.perf_counter()
            await b.submit("p", "r", timeout_ms=1000)
            holds.append((seen[0] - t0) * 1000.0)
    finally:
        await b.stop()
    mean_hold = sum(holds) / len(holds)
    # Collapsed batcher: ~0.05 ms. Correct batcher: >= a few ms (clock-quantised).
    assert mean_hold >= 2.0, f"mean hold {mean_hold:.2f} ms; the wait window collapsed"


async def test_stop_fails_an_abandoned_in_flight_batch_instead_of_hanging_it() -> None:
    """A forward that outlives the stop grace must fail its callers out loud.

    Before this fix, stop() cleared the queue but a batch already inside the
    scorer was invisible to it: cancelling the loop task left every caller in
    that batch awaiting a future nobody would ever resolve.
    """
    import asyncio

    from reward_serve.batcher import ContinuousBatcher, DeadlineExceeded

    release = asyncio.Event()

    class _StuckScorer:
        def score_pairs(self, pairs):  # type: ignore[no-untyped-def]
            # Block the (threaded) forward until told otherwise.
            import time

            while not release.is_set():
                time.sleep(0.01)
            return [0.0 for _ in pairs]

    b = ContinuousBatcher(_StuckScorer(), max_batch_size=4, max_wait_ms=5, queue_depth=8)
    b.stop_grace_s = 0.2  # do not wait 30 s in a test
    await b.start()
    caller = asyncio.ensure_future(b.submit("p", "r", timeout_ms=60_000))
    await asyncio.sleep(0.1)  # let it enter the forward
    try:
        await asyncio.wait_for(b.stop(), timeout=5.0)
        # The caller must resolve promptly with an error -- not hang.
        with pytest.raises(DeadlineExceeded):
            await asyncio.wait_for(caller, timeout=2.0)
    finally:
        release.set()
        if not caller.done():
            caller.cancel()
