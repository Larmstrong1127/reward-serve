"""Exact statistics over a load-generator run.

Definitions used everywhere in this repo (state them once, then never drift):

* A record is *measured* if it was sent after the warm-up window. Warm-up
  records are excluded from every number below.
* `success`  = HTTP 200.
* `rejected` = HTTP 429 (`queue_full`). Rejections are not errors; they are the
  server shedding load on purpose, and they are reported separately.
* `timeout`  = HTTP 504 (`deadline_exceeded`): admitted, then the deadline
  passed before a forward pass started. Also load shedding, but shedding that
  already cost queue capacity, so it is counted apart from 429.
* `shed_rate` = (429 + 504) / measured -- the share of offered load the server
  declined to serve, whichever way it declined.
* `error`    = anything else: 503 (draining/not_ready), 422, other 5xx, and
  transport failures (recorded with status 0).
* Latency percentiles are computed over SUCCESSES ONLY. Mixing in a 1 ms 429
  would make an overloaded server look fast.
* Percentiles are exact order statistics by the nearest-rank method:
  p(q) = sorted[ceil(q/100 * n) - 1]. No interpolation, no estimation, no
  streaming sketch.
* `window_s` is the wall span of the measured phase, from the first measured
  send to the last measured completion, so in-flight work at the end is
  counted where it landed.

Repeats and uncertainty (added 2026-08-21):

* A *repeat* is an independent server start + warm-up + measured window for the
  same cell, with its own seed. Repeats are the only thing that can say how much
  a number would move if the cell ran again.
* `aggregate_repeats` combines them. Counts, rates, latency percentiles and the
  per-request means are computed over the POOLED measured records of every
  repeat, because those are properties of the latency/status distribution and
  pooling more samples estimates them better. The four span-derived quantities
  (`window_s`, `send_span_s`, `throughput_rps`, `offered_rps_actual`) are NOT
  pooled -- the repeats are separated in time, so a pooled span would include
  the gap between them. Those are computed per repeat and averaged, with the
  per-repeat min/max carried alongside.
* `bootstrap_percentile_ci` puts a percentile-bootstrap 95% confidence interval
  on a percentile of the pooled latencies. It answers exactly the question
  "how much would this number move if you ran it again", for the sampling noise
  component. It does NOT cover run-to-run drift of the box; the per-repeat
  min/max is what shows that, and when the repeats fall outside the interval
  the disagreement is reported (`repeats_disagree_beyond_ci`) rather than
  averaged away.
* Everything in this module is a pure function of the per-request records, which
  is why `scripts/recompute_sweep.py` can regenerate a whole summary from the
  committed `raw/*.jsonl` and nothing else.
"""
from __future__ import annotations

import hashlib
import heapq
import math
import random
import statistics
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

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
    records: Sequence[RequestRecord],
    *,
    include_warmup: bool = False,
) -> dict[str, Any]:
    """Core statistics over a list of records (warm-up excluded by default)."""
    rows = list(records) if include_warmup else [r for r in records if not r.is_warmup]
    n = len(rows)
    successes = [r for r in rows if r.status == 200]
    rejected = [r for r in rows if r.status == 429]
    timeouts = [r for r in rows if r.status == 504]
    errors = [r for r in rows if r.status not in (200, 429, 504)]

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
        "n_timeout_504": len(timeouts),
        "n_error": len(errors),
        "rejection_rate": round(len(rejected) / n, 6) if n else None,
        "timeout_rate": round(len(timeouts) / n, 6) if n else None,
        "shed_rate": round((len(rejected) + len(timeouts)) / n, 6) if n else None,
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
        "error_codes": _error_codes(rows),
    }
    for q in PERCENTILES:
        out[f"latency_p{int(q)}_ms"] = round(percentile(lat, q), 4) if lat else None
    out["latency_max_ms"] = round(max(lat), 4) if lat else None
    out["latency_mean_ms"] = _mean_or_none(lat)
    out["ttfb_p50_ms"] = round(percentile(ttfb, 50.0), 4) if ttfb else None
    out["ttfb_p99_ms"] = round(percentile(ttfb, 99.0), 4) if ttfb else None
    return out


def _status_counts(rows: Sequence[RequestRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in rows:
        key = str(r.status)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _error_codes(rows: Sequence[RequestRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in rows:
        if r.error_code:
            counts[r.error_code] = counts.get(r.error_code, 0) + 1
    return dict(sorted(counts.items()))


def summarize(result: RunResult) -> dict[str, Any]:
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


# --- repeats and uncertainty --------------------------------------------------

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_CONFIDENCE = 0.95

#: Keys that `summarize()` adds on top of `summarize_records()`. They describe the
#: generator's own configuration and internal counters, not the request records,
#: so `scripts/recompute_sweep.py` cannot rebuild them from `raw/*.jsonl` and
#: says so explicitly rather than pretending it checked them. `dropped_no_slot`
#: in particular counts requests that were never sent and therefore have no row.
GENERATOR_ONLY_KEYS: frozenset[str] = frozenset(
    {
        "mode",
        "target_rps",
        "concurrency",
        "warmup_s",
        "configured_duration_s",
        "n_total_including_warmup",
        "n_warmup_excluded",
        "max_inflight_observed",
        "dropped_no_slot",
    }
)

#: Span-derived quantities that must NOT be pooled across repeats: repeats are
#: separated in time, so a pooled span would swallow the gap between them and
#: divide a correct success count by a wall clock that includes a server restart.
SPAN_DERIVED_KEYS: tuple[str, ...] = (
    "window_s",
    "send_span_s",
    "throughput_rps",
    "offered_rps_actual",
)


def repeat_seed(base_seed: int, repeat_index: int) -> int:
    """Seed for repeat `repeat_index` (0-based), derived from the run's base seed.

    Distinct per repeat so the arrival process differs, and a pure function of
    the base seed so the whole run is reproducible from one number.
    """
    if repeat_index < 0:
        raise ValueError(f"repeat_index must be >= 0, got {repeat_index}")
    return int(base_seed) + 1_000_003 * int(repeat_index)


def bootstrap_seed_for(cell_id: str, repeat_seeds: Sequence[int]) -> int:
    """Deterministic bootstrap seed derived from data that lives in `raw/`.

    The confidence interval has to be reproducible by a reader who has only the
    raw per-request rows, so its seed cannot be an extra field somebody could
    forget to write down: it is derived from the cell id and the repeat seeds,
    both of which are on every raw row. `hashlib`, not `hash()`, because
    `hash()` of a str is salted per process.
    """
    material = cell_id + "|" + ",".join(str(int(s)) for s in sorted(repeat_seeds))
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _nearest_rank(sample: Sequence[float], q: float) -> float:
    """`percentile()` on an UNSORTED sample, avoiding a full sort for high q."""
    n = len(sample)
    idx = max(math.ceil(q / 100.0 * n), 1) - 1
    k = n - idx
    if k <= max(n // 4, 1):
        return float(heapq.nlargest(k, sample)[-1])
    return float(sorted(sample)[idx])


def bootstrap_percentile_ci(
    values: Sequence[float],
    q: float,
    *,
    seed: int,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = BOOTSTRAP_CONFIDENCE,
) -> dict[str, Any] | None:
    """Percentile-bootstrap confidence interval for the q-th percentile.

    Resample `values` with replacement `n_resamples` times, take the exact
    nearest-rank q-th percentile of each resample, and report the empirical
    (1-confidence)/2 and (1+confidence)/2 quantiles of that distribution.

    Deterministic: same values, same q, same seed, same interval, on any box.
    Returns None for an empty sample -- there is nothing to be uncertain about.
    """
    if not 0 < q <= 100:
        raise ValueError(f"q must be in (0, 100], got {q}")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
    pool = [float(v) for v in values]
    n = len(pool)
    if n == 0:
        return None
    point = percentile(pool, q)
    common: dict[str, Any] = {
        "q": q,
        "point_ms": round(point, 4),
        "n": n,
        "n_resamples": n_resamples,
        "confidence": confidence,
        "seed": seed,
        "method": "percentile bootstrap over pooled per-request latencies",
    }
    if n == 1:
        return {
            **common,
            "lo_ms": round(point, 4),
            "hi_ms": round(point, 4),
            "width_ms": 0.0,
            "note": "one sample: the interval is a point and means nothing",
        }
    rng = random.Random(seed)
    draws = rng.choices
    dist = [_nearest_rank(draws(pool, k=n), q) for _ in range(n_resamples)]
    alpha = (1.0 - confidence) / 2.0
    lo = percentile(dist, alpha * 100.0)
    hi = percentile(dist, (1.0 - alpha) * 100.0)
    return {
        **common,
        "lo_ms": round(lo, 4),
        "hi_ms": round(hi, 4),
        "width_ms": round(hi - lo, 4),
    }


def aggregate_repeats(
    repeats: Sequence[Sequence[RequestRecord]],
    *,
    cell_id: str,
    repeat_seeds: Sequence[int],
    ci_percentile: float = 99.0,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = BOOTSTRAP_CONFIDENCE,
) -> dict[str, Any]:
    """Combine the repeats of one cell into the cell's published statistics.

    `repeats[i]` is repeat i's full record list, warm-up rows included (they are
    excluded here, once, by the same rule as everywhere else).

    Pure function of the records plus the cell id and seeds, all of which are on
    every raw row -- so `scripts/recompute_sweep.py` reproduces this exactly.
    """
    if len(repeats) != len(repeat_seeds):
        raise ValueError(
            f"got {len(repeats)} repeats and {len(repeat_seeds)} seeds; they must pair up"
        )
    # A repeat that measured nothing carries no information and, more importantly,
    # leaves no rows in raw/ -- so including it here would make the published
    # aggregate impossible to reproduce from the raw files. Drop it in both places
    # by dropping it in the one function both paths call.
    kept = [
        (recs, seed)
        for recs, seed in zip(repeats, repeat_seeds, strict=True)
        if any(not r.is_warmup for r in recs)
    ]
    measured: list[list[RequestRecord]] = [
        [r for r in rep if not r.is_warmup] for rep, _ in kept
    ]
    repeat_seeds = [seed for _, seed in kept]
    pooled = [r for rep in measured for r in rep]
    out = summarize_records(pooled, include_warmup=True)  # already filtered above
    per_repeat = [summarize_records(rep, include_warmup=True) for rep in measured]

    # Span-derived quantities: per repeat, then averaged. Never pooled.
    for key in SPAN_DERIVED_KEYS:
        vals = [s[key] for s in per_repeat if s.get(key) is not None]
        out[key] = round(statistics.fmean(vals), 4) if vals else None
        out[f"{key}_repeat_min"] = round(min(vals), 4) if vals else None
        out[f"{key}_repeat_max"] = round(max(vals), 4) if vals else None
    out["span_derived_keys_are_repeat_means"] = list(SPAN_DERIVED_KEYS)

    lat = [r.latency_ms for r in pooled if r.status == 200]
    seed = bootstrap_seed_for(cell_id, repeat_seeds)
    out[f"latency_p{int(ci_percentile)}_ci_ms"] = bootstrap_percentile_ci(
        lat, ci_percentile, seed=seed, n_resamples=n_resamples, confidence=confidence
    )
    key = f"latency_p{int(ci_percentile)}_ms"
    per_rep_p = [s[key] for s in per_repeat if s.get(key) is not None]
    out[f"{key}_per_repeat"] = per_rep_p
    out[f"{key}_repeat_min"] = min(per_rep_p) if per_rep_p else None
    out[f"{key}_repeat_max"] = max(per_rep_p) if per_rep_p else None

    ci = out[f"latency_p{int(ci_percentile)}_ci_ms"]
    if ci is not None and per_rep_p:
        # A repeat landing outside the sampling-noise interval is a finding: it
        # means the box moved between repeats, not that the samples were noisy.
        out["repeats_disagree_beyond_ci"] = bool(
            min(per_rep_p) < ci["lo_ms"] or max(per_rep_p) > ci["hi_ms"]
        )
    else:
        out["repeats_disagree_beyond_ci"] = None

    out["repeats"] = len(measured)
    out["repeat_seeds"] = list(repeat_seeds)
    out["n_measured_per_repeat"] = [len(rep) for rep in measured]
    out["pooled_from_repeats"] = True
    return out
