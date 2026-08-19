"""Percentile math against arrays whose answers are known by hand."""
from __future__ import annotations

import pytest

from loadgen.generator import RequestRecord
from loadgen.stats import percentile, summarize_records


def test_percentile_nearest_rank_on_one_to_hundred() -> None:
    values = list(range(1, 101))  # 1..100, so p(q) == q by nearest rank
    assert percentile(values, 50) == 50
    assert percentile(values, 95) == 95
    assert percentile(values, 99) == 99
    assert percentile(values, 100) == 100


def test_percentile_small_arrays_by_hand() -> None:
    # n=10: rank = ceil(q/100*10). p50 -> rank 5 -> 5th smallest.
    values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    assert percentile(values, 50) == 50
    assert percentile(values, 51) == 60   # ceil(5.1) = 6
    assert percentile(values, 90) == 90
    assert percentile(values, 95) == 100  # ceil(9.5) = 10
    # single element
    assert percentile([7.0], 1) == 7.0
    assert percentile([7.0], 100) == 7.0


def test_percentile_ignores_input_order() -> None:
    assert percentile([9, 1, 5, 3, 7], 50) == 5
    assert percentile([3, 7, 1, 9, 5], 50) == 5


def test_percentile_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1.0], 0)
    with pytest.raises(ValueError):
        percentile([1.0], 101)


def _rec(idx: int, *, warmup: bool, status: int = 200, lat_ms: float = 10.0,
         batch: int | None = 4, qwait: float | None = 1.0) -> RequestRecord:
    return RequestRecord(
        index=idx, bucket="short", send_ts=float(idx), first_byte_ts=idx + lat_ms / 2000.0,
        done_ts=idx + lat_ms / 1000.0, status=status, is_warmup=warmup,
        batch_size=batch, queue_wait_ms=qwait, forward_ms=5.0, score=0.0,
    )


def test_warmup_records_are_excluded() -> None:
    rows = [_rec(i, warmup=True, lat_ms=1000.0) for i in range(10)]
    rows += [_rec(10 + i, warmup=False, lat_ms=10.0) for i in range(10)]
    out = summarize_records(rows)
    assert out["n_measured"] == 10
    # If warm-up leaked in, p99 would be near 1000 ms, not 10 ms.
    assert out["latency_p99_ms"] == pytest.approx(10.0, abs=0.01)
    assert out["latency_max_ms"] == pytest.approx(10.0, abs=0.01)
    with_warmup = summarize_records(rows, include_warmup=True)
    assert with_warmup["n_measured"] == 20
    assert with_warmup["latency_p99_ms"] == pytest.approx(1000.0, abs=0.01)


def test_rejections_are_not_errors_and_do_not_enter_latency() -> None:
    rows = [_rec(i, warmup=False, lat_ms=100.0) for i in range(8)]
    rows += [_rec(100 + i, warmup=False, status=429, lat_ms=1.0) for i in range(2)]
    rows += [_rec(200, warmup=False, status=503, lat_ms=1.0)]
    out = summarize_records(rows)
    assert out["n_measured"] == 11
    assert out["n_success"] == 8
    assert out["n_rejected_429"] == 2
    assert out["n_error"] == 1
    assert out["rejection_rate"] == pytest.approx(2 / 11)
    assert out["error_rate"] == pytest.approx(1 / 11)
    # a fast 429 must not drag the success latency down
    assert out["latency_p50_ms"] == pytest.approx(100.0, abs=0.01)
    assert out["status_counts"] == {"200": 8, "429": 2, "503": 1}


def test_mean_batch_and_queue_wait_use_reported_metadata_only() -> None:
    rows = [_rec(0, warmup=False, batch=2, qwait=1.0), _rec(1, warmup=False, batch=6, qwait=3.0)]
    rows.append(_rec(2, warmup=False, batch=None, qwait=None))
    out = summarize_records(rows)
    assert out["mean_batch_size"] == pytest.approx(4.0)
    assert out["mean_queue_wait_ms"] == pytest.approx(2.0)
    assert out["batch_size_reported"] is True


def test_missing_metadata_degrades_to_none() -> None:
    rows = [_rec(i, warmup=False, batch=None, qwait=None) for i in range(3)]
    out = summarize_records(rows)
    assert out["mean_batch_size"] is None
    assert out["mean_queue_wait_ms"] is None
    assert out["batch_size_reported"] is False
    assert out["n_success"] == 3
