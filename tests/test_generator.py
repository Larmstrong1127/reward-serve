"""The generator must offer the load it says it offers, and no more."""
from __future__ import annotations

import asyncio
import time
from typing import Any, Callable

import pytest

from loadgen.generator import run_closed_loop, run_open_loop, sleep_until, wait_ready
from loadgen.workload import BUCKETS, build_corpus


def test_corpus_is_deterministic_for_a_seed() -> None:
    a = build_corpus(7)
    b = build_corpus(7)
    c = build_corpus(8)
    assert [s.prompt for s in a] == [s.prompt for s in b]
    assert [s.prompt for s in a] != [s.prompt for s in c]


def test_corpus_respects_length_buckets() -> None:
    for sample in build_corpus(1, per_bucket=3):
        p_words, r_words = BUCKETS[sample.bucket]
        assert len(sample.prompt.split()) == p_words
        assert len(sample.response.split()) == r_words


def test_sleep_until_does_not_return_early_under_a_busy_loop() -> None:
    """Guards the Windows asyncio defect: the loop fires timers up to its clock
    resolution (15.625 ms here) early, which would make the generator send
    faster than its configured rate."""

    async def scenario() -> float:
        async def spin() -> None:
            end = time.perf_counter() + 0.5
            while time.perf_counter() < end:
                await asyncio.sleep(0)

        spinners = [asyncio.create_task(spin()) for _ in range(4)]
        target = 0.010
        start = time.perf_counter()
        await sleep_until(start + target)
        elapsed = time.perf_counter() - start
        for s in spinners:
            s.cancel()
        await asyncio.gather(*spinners, return_exceptions=True)
        return elapsed

    elapsed = asyncio.run(scenario())
    assert elapsed >= 0.010, f"slept only {elapsed * 1000:.3f} ms of a 10 ms deadline"


@pytest.mark.parametrize("target_rps", [40.0, 90.0])
def test_open_loop_arrival_rate_within_tolerance(stub_server: str, target_rps: float) -> None:
    result = asyncio.run(
        run_open_loop(stub_server, target_rps=target_rps, duration_s=4.0, warmup_s=0.5)
    )
    sent = len(result.records)
    expected = target_rps * 4.5
    assert result.dropped_no_slot == 0
    # Poisson arrivals over ~4.5 s: allow 15% either way.
    assert 0.85 * expected <= sent <= 1.15 * expected, (
        f"sent {sent}, expected ~{expected:.0f} at {target_rps} rps"
    )
    measured = result.measured()
    assert measured, "no measured records"
    achieved = len(measured) / (max(r.send_ts for r in measured) - min(r.send_ts for r in measured))
    assert 0.8 * target_rps <= achieved <= 1.2 * target_rps


def test_open_loop_does_not_wait_for_responses(stub_server: str) -> None:
    """Open loop means arrivals are independent of service time, so more than
    one request must be in flight at a time."""
    result = asyncio.run(run_open_loop(stub_server, target_rps=120.0, duration_s=2.0, warmup_s=0.2))
    assert result.max_inflight_observed > 1


@pytest.mark.parametrize("concurrency", [1, 4, 12])
def test_closed_loop_never_exceeds_its_concurrency(stub_server: str, concurrency: int) -> None:
    result = asyncio.run(
        run_closed_loop(stub_server, concurrency=concurrency, duration_s=2.0, warmup_s=0.3)
    )
    assert result.max_inflight_observed <= concurrency
    assert result.max_inflight_observed >= 1
    assert result.concurrency == concurrency
    assert all(r.status == 200 for r in result.records)


def test_warmup_records_are_marked_and_excluded(stub_server: str) -> None:
    result = asyncio.run(run_closed_loop(stub_server, concurrency=2, duration_s=1.0, warmup_s=0.5))
    warm = [r for r in result.records if r.is_warmup]
    measured = result.measured()
    assert warm and measured
    assert len(warm) + len(measured) == len(result.records)
    t0 = min(r.send_ts for r in result.records)
    # every warm-up record was sent before every measured record
    assert max(r.send_ts for r in warm) <= min(r.send_ts for r in measured) + 1e-6
    assert max(r.send_ts for r in warm) - t0 < 0.5 + 0.2


def test_batch_metadata_is_captured(stub_server: str) -> None:
    result = asyncio.run(run_closed_loop(stub_server, concurrency=6, duration_s=1.5, warmup_s=0.2))
    ok = [r for r in result.records if r.status == 200]
    assert ok
    assert all(r.batch_size is not None and r.batch_size >= 1 for r in ok)
    assert all(r.queue_wait_ms is not None and r.queue_wait_ms >= 0 for r in ok)
    assert all(r.forward_ms is not None and r.forward_ms > 0 for r in ok)
    assert all(r.first_byte_ts is not None and r.first_byte_ts <= r.done_ts for r in ok)


def test_rejections_are_recorded_not_raised(stub_server_factory: Callable[..., Any]) -> None:
    """A queue-full server returns 429; that is data, not a crash."""
    base = stub_server_factory(
        MAX_BATCH_SIZE=1, MAX_WAIT_MS=0, MAX_QUEUE=1,
        FORWARD_BASE_MS=20, FORWARD_PER_ITEM_MS=5,
    )
    result = asyncio.run(run_open_loop(base, target_rps=200.0, duration_s=2.0, warmup_s=0.2))
    statuses = {r.status for r in result.records}
    assert 429 in statuses, f"expected rejections, saw {statuses}"
    assert all(r.error is None for r in result.records if r.status == 429)


def test_wait_ready_times_out_on_a_dead_port() -> None:
    with pytest.raises(TimeoutError):
        asyncio.run(wait_ready("http://127.0.0.1:9", timeout_s=1.0, poll_s=0.1))


def test_corpus_interleaves_buckets_so_short_runs_see_the_full_mix() -> None:
    corpus = build_corpus(3, per_bucket=8)
    first_cycle = [s.bucket for s in corpus[: len(BUCKETS)]]
    assert sorted(first_cycle) == sorted(BUCKETS), (
        "the first requests of a run must cover every length bucket"
    )
    # any window one cycle long is balanced
    window = [s.bucket for s in corpus[4 : 4 + len(BUCKETS)]]
    assert sorted(window) == sorted(BUCKETS)
