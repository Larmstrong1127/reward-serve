"""The sweep's numbers must be checkable from raw, not merely assertable.

This file is the answer to a fair question from a cold-clone reader: *p99 =
1899 ms at batch 1, 4 rps -- how much would that move if you ran it again, and
how would I know?* Before per-request persistence the honest answer was "you
cannot know from what is committed". These tests hold the new answer in place:

* every statistic in `summary.json` is regenerated from `raw/*.jsonl` alone and
  must match (`test_the_committed_run_regenerates_exactly_from_raw`),
* the check has teeth -- a doctored summary is caught,
* the confidence interval is deterministic given a seed, so two readers get the
  same interval,
* warm-up rows are ON DISK and flagged, so the exclusion is verifiable rather
  than trusted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from loadgen.generator import RequestRecord  # noqa: E402
from loadgen.stats import (  # noqa: E402
    aggregate_repeats,
    bootstrap_percentile_ci,
    bootstrap_seed_for,
    percentile,
    repeat_seed,
    summarize_records,
)
from loadgen.sweep import write_raw_rows  # noqa: E402
from scripts import recompute_sweep  # noqa: E402

#: A two-cell, two-repeat run against tests/stub_server.py with the quiet gate
#: stood down. It is stamped `not_a_measurement` in every artifact it wrote: it
#: measures a sleep, on a box that was known to be busy, and exists only so this
#: suite has a real run to recompute. No number in it describes the model.
FIXTURE = REPO_ROOT / "results" / "2026-08-21-smoke-not-a-measurement"


def _record(
    idx: int, send: float, done: float, status: int = 200, *, warmup: bool = False,
    batch: int | None = 1, error_code: str | None = None,
) -> RequestRecord:
    return RequestRecord(
        index=idx, bucket="short", send_ts=send, first_byte_ts=done, done_ts=done,
        status=status, is_warmup=warmup, batch_size=batch,
        queue_wait_ms=1.0 if status == 200 else None,
        forward_ms=2.0 if status == 200 else None,
        error_code=error_code,
    )


# --- the fixture is unmistakably not a measurement -----------------------------


def test_the_committed_fixture_run_is_stamped_not_a_measurement() -> None:
    summary = json.loads((FIXTURE / "summary.json").read_text(encoding="utf-8"))
    assert summary["not_a_measurement"] is True
    assert summary["smoke"] is True
    assert json.loads((FIXTURE / "box_quiet.json").read_text())["gate_skipped"] is True
    assert "stub_server" in summary["target_under_test"]
    for cell in summary["cells"]:
        assert cell["not_a_measurement"] is True


# --- the headline property: recomputable from raw ------------------------------


def test_the_committed_run_regenerates_exactly_from_raw() -> None:
    """Every derivable statistic in summary.json falls out of raw/*.jsonl."""
    report = recompute_sweep.verify(FIXTURE, tol=1e-9)
    assert report["ok"], json.dumps(
        [c for c in report["cells"] if not c["ok"]], indent=2
    )[:4000]
    assert report["n_cells"] == report["n_cells_ok"] > 0
    # and it really checked the numbers a reader would quote, not two booleans
    agg = report["cells"][0]["aggregate"]
    assert agg["n_checked"] >= 25
    stored = json.loads((FIXTURE / "summary.json").read_text())["cells"][0]["stats"]
    for key in (
        "latency_p50_ms", "latency_p95_ms", "latency_p99_ms", "latency_max_ms",
        "throughput_rps", "shed_rate", "rejection_rate", "timeout_rate",
        "mean_batch_size", "n_success", "n_measured",
    ):
        assert key in stored, f"{key} is quoted in prose and must be in the summary"
        assert key not in {m["key"] for m in agg["mismatches"]}


def test_recompute_reads_only_the_raw_directory(tmp_path: Path) -> None:
    """`recompute_run` must not need -- or silently fall back to -- the summary."""
    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    for src in (FIXTURE / "raw").glob("*.jsonl"):
        (raw / src.name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    assert not (tmp_path / "summary.json").exists()
    out = recompute_sweep.recompute_run(tmp_path)
    assert out["n_raw_files"] == len(list(raw.glob("*.jsonl")))
    stored = json.loads((FIXTURE / "summary.json").read_text())["cells"][0]
    got = out["cells"][stored["cell_id"]]["stats"]
    assert got["latency_p99_ms"] == stored["stats"]["latency_p99_ms"]


def test_the_recompute_check_has_teeth(tmp_path: Path) -> None:
    """A summary whose p99 was edited by hand must fail verification."""
    run = tmp_path / "doctored"
    (run / "raw").mkdir(parents=True)
    for src in (FIXTURE / "raw").glob("*.jsonl"):
        (run / "raw" / src.name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    summary = json.loads((FIXTURE / "summary.json").read_text(encoding="utf-8"))
    summary["cells"][0]["stats"]["latency_p99_ms"] = 1.0
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    report = recompute_sweep.verify(run, tol=1e-9)
    assert report["ok"] is False
    bad = [c for c in report["cells"] if not c["ok"]]
    assert {m["key"] for m in bad[0]["aggregate"]["mismatches"]} == {"latency_p99_ms"}
    assert recompute_sweep.main([str(run)]) == 1


def test_a_run_without_raw_rows_is_refused_not_waved_through(tmp_path: Path) -> None:
    (tmp_path / "summary.json").write_text(json.dumps({"cells": []}), encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        recompute_sweep.recompute_run(tmp_path)
    assert "cannot be recomputed" in str(exc.value)


def test_generator_internal_keys_are_declared_unverifiable_not_skipped_quietly() -> None:
    report = recompute_sweep.verify(FIXTURE, tol=1e-9)
    skipped = report["cells"][0]["per_repeat"][0]["not_derivable_from_raw"]
    # dropped_no_slot counts requests that were never sent, so no row can exist.
    assert "dropped_no_slot" in skipped
    assert "max_inflight_observed" in skipped


# --- warm-up: on disk, flagged, and out of the statistics ----------------------


def test_warmup_rows_are_persisted_flagged_and_excluded_from_the_statistics() -> None:
    rows = [
        json.loads(line)
        for line in (FIXTURE / "raw" / "b1_w5_r30.jsonl").read_text().splitlines()
        if line.strip()
    ]
    warm = [r for r in rows if r["is_warmup"]]
    measured = [r for r in rows if not r["is_warmup"]]
    assert warm, "warm-up rows must stay in raw so the exclusion can be verified"
    assert measured

    stored = next(
        c for c in json.loads((FIXTURE / "summary.json").read_text())["cells"]
        if c["cell_id"] == "b1_w5_r30"
    )["stats"]
    assert stored["n_measured"] == len(measured)

    # and the exclusion is what actually moved the number: recompute including
    # the warm-up rows and the count changes.
    recs = [RequestRecord.from_json(r) for r in rows]
    with_warmup = summarize_records(recs, include_warmup=True)
    assert with_warmup["n_measured"] == len(rows) > stored["n_measured"]


def test_write_raw_rows_keeps_warmup_rows(tmp_path: Path) -> None:
    path = tmp_path / "raw" / "c.jsonl"
    recs = [_record(0, 0.0, 0.1, warmup=True), _record(1, 1.0, 1.1)]
    n = write_raw_rows(
        path, recs, cell_id="c", repeat=0, seed=7, batch_size=1, wait_ms=5.0,
        rps=10.0, truncate=True,
    )
    assert n == 2
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert [r["is_warmup"] for r in rows] == [True, False]
    assert all(r["cell_id"] == "c" and r["repeat"] == 0 and r["seed"] == 7 for r in rows)


def test_repeats_append_to_one_file_per_cell(tmp_path: Path) -> None:
    path = tmp_path / "raw" / "c.jsonl"
    for rep in (0, 1, 2):
        write_raw_rows(
            path, [_record(i, float(i), float(i) + 0.1) for i in range(4)],
            cell_id="c", repeat=rep, seed=100 + rep, batch_size=1, wait_ms=5.0,
            rps=10.0, truncate=(rep == 0),
        )
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert len(rows) == 12
    assert sorted({r["repeat"] for r in rows}) == [0, 1, 2]
    _, by_repeat, seeds = recompute_sweep.read_raw_cell(path)
    assert sorted(by_repeat) == [0, 1, 2]
    assert seeds == {0: 100, 1: 101, 2: 102}


# --- cells that measured nothing ----------------------------------------------


def test_a_cell_with_zero_successful_requests_still_writes_raw_and_aggregates(
    tmp_path: Path,
) -> None:
    """An all-shed cell is a result. It must not crash the writer or the stats."""
    path = tmp_path / "raw" / "shed.jsonl"
    recs = [
        _record(i, float(i), float(i) + 0.001, status=429, batch=None,
                error_code="queue_full")
        for i in range(20)
    ]
    assert write_raw_rows(
        path, recs, cell_id="shed", repeat=0, seed=1, batch_size=32, wait_ms=0.0,
        rps=100.0, truncate=True,
    ) == 20

    stats = aggregate_repeats([recs], cell_id="shed", repeat_seeds=[1], n_resamples=50)
    assert stats["n_success"] == 0
    assert stats["n_rejected_429"] == 20
    assert stats["rejection_rate"] == 1.0 and stats["shed_rate"] == 1.0
    assert stats["latency_p99_ms"] is None
    assert stats["latency_p99_ci_ms"] is None, "no successes means no interval to report"
    assert stats["repeats_disagree_beyond_ci"] is None
    assert stats["throughput_rps"] == 0.0

    got = recompute_sweep.recompute_cell(path, n_resamples=50)
    assert got["stats"]["n_rejected_429"] == 20
    assert got["stats"]["latency_p99_ci_ms"] is None


def test_a_repeat_that_produced_no_rows_at_all_is_handled(tmp_path: Path) -> None:
    """Server never came up: the file is created empty, and nothing explodes."""
    path = tmp_path / "raw" / "dead.jsonl"
    assert write_raw_rows(
        path, [], cell_id="dead", repeat=0, seed=1, batch_size=1, wait_ms=0.0,
        rps=1.0, truncate=True,
    ) == 0
    assert path.exists() and path.read_text() == ""
    got = recompute_sweep.recompute_cell(path, n_resamples=50)
    assert got["n_rows"] == 0
    assert got["stats"] == {}
    assert "measured nothing" in got["note"]


def test_an_empty_repeat_is_dropped_from_the_aggregate_not_counted(tmp_path: Path) -> None:
    good = [_record(i, float(i), float(i) + 0.01) for i in range(10)]
    stats = aggregate_repeats([good, []], cell_id="c", repeat_seeds=[1, 2], n_resamples=50)
    assert stats["repeats"] == 1, "a repeat with no rows cannot be reproduced from raw"
    assert stats["repeat_seeds"] == [1]


# --- the interval --------------------------------------------------------------


def test_bootstrap_ci_is_deterministic_under_a_fixed_seed() -> None:
    import random

    rng = random.Random(11)
    values = [rng.gauss(100.0, 25.0) for _ in range(500)]
    a = bootstrap_percentile_ci(values, 99.0, seed=12345, n_resamples=500)
    b = bootstrap_percentile_ci(values, 99.0, seed=12345, n_resamples=500)
    assert a == b
    assert a is not None
    assert a["lo_ms"] <= a["point_ms"] <= a["hi_ms"]
    assert a["width_ms"] > 0.0, "a zero-width interval would mean nothing resampled"
    # Determinism must not come from the RNG being ignored. Proving that needs a
    # deliberately under-resampled bootstrap: at the configured 2000 resamples the
    # 2.5/97.5 bounds have converged onto particular order statistics and land in
    # the same place for any seed, which is the bootstrap working, not the seed
    # being dead. At 9 resamples the seed is still visible in the answer.
    s1 = bootstrap_percentile_ci(values, 99.0, seed=1, n_resamples=9)
    s2 = bootstrap_percentile_ci(values, 99.0, seed=2, n_resamples=9)
    assert s1 is not None and s2 is not None
    assert (s1["lo_ms"], s1["hi_ms"]) != (s2["lo_ms"], s2["hi_ms"])
    assert s1 == bootstrap_percentile_ci(values, 99.0, seed=1, n_resamples=9)


def test_bootstrap_ci_matches_the_stored_interval_bit_for_bit() -> None:
    """The committed CI is reproducible by a reader holding only raw/."""
    stored = next(
        c for c in json.loads((FIXTURE / "summary.json").read_text())["cells"]
        if c["cell_id"] == "b1_w5_r30"
    )["stats"]["latency_p99_ci_ms"]
    got = recompute_sweep.recompute_cell(FIXTURE / "raw" / "b1_w5_r30.jsonl",
                                         n_resamples=stored["n_resamples"])
    assert got["stats"]["latency_p99_ci_ms"] == stored


def test_bootstrap_seed_is_derived_from_data_that_lives_in_raw() -> None:
    """Not a hidden field: cell id + repeat seeds, both on every raw row."""
    a = bootstrap_seed_for("b1_w5_r30", [1234, 1001237])
    assert a == bootstrap_seed_for("b1_w5_r30", [1001237, 1234]), "order must not matter"
    assert a != bootstrap_seed_for("b8_w5_r30", [1234, 1001237])
    assert a != bootstrap_seed_for("b1_w5_r30", [1234])


def test_a_single_sample_gets_a_degenerate_interval_that_says_so() -> None:
    ci = bootstrap_percentile_ci([42.0], 99.0, seed=1, n_resamples=100)
    assert ci is not None
    assert ci["lo_ms"] == ci["hi_ms"] == 42.0
    assert "one sample" in ci["note"]


def test_an_empty_sample_has_no_interval() -> None:
    assert bootstrap_percentile_ci([], 99.0, seed=1) is None


def test_the_interval_covers_the_point_estimate_and_narrows_with_more_data() -> None:
    import random

    rng = random.Random(4242)
    small = [rng.gauss(100.0, 20.0) for _ in range(60)]
    large = [rng.gauss(100.0, 20.0) for _ in range(6000)]
    a = bootstrap_percentile_ci(small, 99.0, seed=1, n_resamples=400)
    b = bootstrap_percentile_ci(large, 99.0, seed=1, n_resamples=400)
    assert a is not None and b is not None
    assert b["width_ms"] < a["width_ms"], (
        "more samples must buy a tighter interval, or the interval is not measuring "
        "sampling noise"
    )


def test_the_resampled_percentile_uses_the_same_nearest_rank_rule() -> None:
    """The bootstrap must not quietly switch percentile definitions."""
    from loadgen.stats import _nearest_rank

    for n in (1, 7, 50, 99, 100, 101, 400):
        vals = [float(i) for i in range(n)]
        shuffled = list(reversed(vals))
        for q in (50.0, 95.0, 99.0, 100.0):
            assert _nearest_rank(shuffled, q) == percentile(vals, q), (n, q)


# --- repeats -------------------------------------------------------------------


def test_repeat_seeds_are_distinct_and_derived_from_the_base_seed() -> None:
    seeds = [repeat_seed(1234, i) for i in range(5)]
    assert seeds[0] == 1234
    assert len(set(seeds)) == 5
    assert seeds == [repeat_seed(1234, i) for i in range(5)], "must be a pure function"
    with pytest.raises(ValueError):
        repeat_seed(1234, -1)


def test_the_committed_run_used_distinct_seeds_per_repeat() -> None:
    summary = json.loads((FIXTURE / "summary.json").read_text(encoding="utf-8"))
    assert summary["load"]["repeats_per_cell"] == 2
    for cell in summary["cells"]:
        seeds = [r["seed"] for r in cell["per_repeat"]]
        assert len(set(seeds)) == len(seeds) > 1
        assert seeds == summary["load"]["repeat_seeds"]


def test_span_derived_stats_are_per_repeat_means_never_pooled() -> None:
    """Pooling a span across repeats would divide by the gap between them."""
    rep_a = [_record(i, float(i), float(i) + 0.01) for i in range(10)]
    # second repeat happens a long time later, as a restarted server does
    rep_b = [_record(i, 1000.0 + i, 1000.0 + i + 0.01) for i in range(10)]
    stats = aggregate_repeats([rep_a, rep_b], cell_id="c", repeat_seeds=[1, 2],
                              n_resamples=50)
    assert stats["window_s"] < 20.0, (
        "a pooled span would be ~1009 s and would report a throughput near zero"
    )
    assert stats["n_success"] == 20, "counts ARE pooled"
    assert stats["window_s_repeat_min"] is not None
    assert "throughput_rps" in stats["span_derived_keys_are_repeat_means"]


def test_disagreement_between_repeats_is_reported_not_averaged_away() -> None:
    fast = [_record(i, float(i), float(i) + 0.010) for i in range(60)]
    slow = [_record(i, 100.0 + i, 100.0 + i + 0.500) for i in range(60)]
    stats = aggregate_repeats([fast, slow], cell_id="c", repeat_seeds=[1, 2],
                              n_resamples=400)
    assert stats["repeats_disagree_beyond_ci"] is True
    lo, hi = stats["latency_p99_ms_repeat_min"], stats["latency_p99_ms_repeat_max"]
    assert lo is not None and hi is not None and hi > 10 * lo
    assert len(stats["latency_p99_ms_per_repeat"]) == 2


def test_agreeing_repeats_are_not_flagged() -> None:
    a = [_record(i, float(i), float(i) + 0.010) for i in range(80)]
    b = [_record(i, 100.0 + i, 100.0 + i + 0.010) for i in range(80)]
    stats = aggregate_repeats([a, b], cell_id="c", repeat_seeds=[1, 2], n_resamples=400)
    assert stats["repeats_disagree_beyond_ci"] is False


def test_aggregate_refuses_mismatched_repeats_and_seeds() -> None:
    with pytest.raises(ValueError):
        aggregate_repeats([[], []], cell_id="c", repeat_seeds=[1], n_resamples=10)
