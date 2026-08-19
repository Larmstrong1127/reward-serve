"""Exact statistics over a load-generator run.

Definitions used everywhere in this repo (state them once, then never drift):

* A record is *measured* if it was sent after the warm-up window. Warm-up
  records are excluded from every number below.
* `success`  = HTTP 200.
* `rejected` = HTTP 429 (server queue full). Rejections are not errors; they are
  the server shedding load on purpose, and they are reported separately.
* `error`    = anything else, including 503 (draining), 5xx, and transport
  failures (recorded with status 0).
* Latency percentiles are computed over SUCCESSES ONLY. Mixing in a 1 ms 429
  would make an overloaded server look fast.
* Percentiles are exact order statistics by the nearest-rank method:
  p(q) = sorted[ceil(q/100 * n) - 1]. No interpolation, no estimation, no
  streaming sketch.
* `window_s` is the wall span of the measured phase, from the first measured
  send to the last measured completion, so in-flight work at the end is
  counted where it landed.
"""
from __future__ import annotations

import math
import statistics
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:  # avoid a runtime import cycle with generator.py
    from loadgen.generator import RequestRecord, RunResult

PERCENTILES: tuple[float, ...] = (50.0, 95.0, 99.0)


def percentile(values: Sequence[float], q: float) -> float:
    """Exact nearest-rank percentile of `values` (0 < q <= 100)."""
    if not values:
        raise ValueError("percentile of an empty sequence")
    if not 0 < q <= 100:
        raise ValueError(f"q must be in (0, 100], got {q}")
    ordered = sorted(values)
    rank = math.ceil(q / 100.0 * len(ordered))
    return float(ordered[max(rank, 1) - 1])


def _mean_or_none(values: Sequence[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def summarize_records(
    records: Sequence["RequestRecord"],
    *,
    include_warmup: bool = False,
) -> dict[str, Any]:
    """Core statistics over a list of records (warm-up excluded by default)."""
    rows = list(records) if include_warmup else [r for r in records if not r.is_warmup]
    n = len(rows)
    successes = [r for r in rows if r.status == 200]
    rejected = [r for r in rows if r.status == 429]
    errors = [r for r in rows if r.status not in (200, 429)]

    lat = [r.latency_ms for r in successes]
    ttfb = [r.ttfb_ms for r in successes if r.ttfb_ms is not None]
    batch = [float(r.batch_size) for r in successes if r.batch_size is not None]
    qwait = [r.queue_wait_ms for r in successes if r.queue_wait_ms is not None]
    fwd = [r.forward_ms for r in successes if r.forward_ms is not None]

    if rows:
        window_s = max(r.done_ts for r in rows) - min(r.send_ts for r in rows)
        send_span_s = max(r.send_ts for r in rows) - min(r.send_ts for r in rows)
    else:
        window_s = 0.0
        send_span_s = 0.0

    out: dict[str, Any] = {
        "n_measured": n,
        "n_success": len(successes),
        "n_rejected_429": len(rejected),
        "n_error": len(errors),
        "rejection_rate": round(len(rejected) / n, 6) if n else None,
        "error_rate": round(len(errors) / n, 6) if n else None,
        "window_s": round(window_s, 4),
        "send_span_s": round(send_span_s, 4),
        "throughput_rps": round(len(successes) / window_s, 4) if window_s > 0 else None,
        "offered_rps_actual": round(n / send_span_s, 4) if send_span_s > 0 else None,
        "mean_batch_size": _mean_or_none(batch),
        "mean_queue_wait_ms": _mean_or_none(qwait),
        "mean_forward_ms": _mean_or_none(fwd),
        "batch_size_reported": bool(batch),
        "status_counts": _status_counts(rows),
    }
    for q in PERCENTILES:
        out[f"latency_p{int(q)}_ms"] = round(percentile(lat, q), 4) if lat else None
    out["latency_max_ms"] = round(max(lat), 4) if lat else None
    out["latency_mean_ms"] = _mean_or_none(lat)
    out["ttfb_p50_ms"] = round(percentile(ttfb, 50.0), 4) if ttfb else None
    out["ttfb_p99_ms"] = round(percentile(ttfb, 99.0), 4) if ttfb else None
    return out


def _status_counts(rows: Sequence["RequestRecord"]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in rows:
        key = str(r.status)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def summarize(result: "RunResult") -> dict[str, Any]:
    """Statistics for a RunResult, with the run's own configuration attached."""
    stats = summarize_records(result.records)
    stats.update(
        {
            "mode": result.mode,
            "target_rps": result.target_rps,
            "concurrency": result.concurrency,
            "warmup_s": result.warmup_s,
            "configured_duration_s": result.duration_s,
            "n_total_including_warmup": len(result.records),
            "n_warmup_excluded": len(result.records) - stats["n_measured"],
            "max_inflight_observed": result.max_inflight_observed,
            "dropped_no_slot": result.dropped_no_slot,
        }
    )
    return stats
