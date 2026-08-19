"""Break the tier on purpose and record what the CLIENT saw.

Four scenarios, each run against a router and two workers loading the real
checkpoint, each driven by the existing closed-loop generator at concurrency 4
(concurrency 16 for the queue-exhaustion case, where the point is to overrun the
queue). This is not a benchmark and it does not sweep: every scenario asks a
behavioural question whose answer is a count, a state transition, or a time, not
a throughput figure.

    python faults/run_faults.py                    # all four
    python faults/run_faults.py --only hang-worker

Per scenario it writes ``results/faults/<name>/``:

    events.jsonl      one line per client request: status, latency, which
                      worker served it, whether the router retried it
    routerz.jsonl     /routerz samples, ~10 a second, with wall timestamps
    metrics.txt       the router's /metrics at the end of the run
    summary.json      the numbers docs/multi-tier.md is allowed to quote
    box_quiet.json    the contention report measured immediately before

Charter rule 3 says a contended box may not publish latency. These runs are
behavioural, so they are allowed to run on a busy box -- but when box_quiet
fails, the summary carries ``latency_figures_are_indicative_only: true`` and the
write-up quotes only counts, states and times from it. The flag is set by the
runner, not by hand, so it cannot be forgotten.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse  # noqa: E402
import asyncio  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import httpx  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import box_quiet  # noqa: E402
from tier_up import Tier  # noqa: E402

from loadgen.generator import RequestRecord, run_closed_loop, wait_ready  # noqa: E402
from loadgen.stats import percentile  # noqa: E402

OUT_ROOT = REPO_ROOT / "results" / "faults"
ROUTER_TIMEOUT_MS = 3000.0
SAMPLE_INTERVAL_S = 0.1


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------
@dataclass
class Clockbase:
    """One origin so perf_counter offsets and wall times are comparable.

    The generator stamps requests with ``perf_counter``; the fault injections
    and the /routerz samples are easier to reason about in wall time. Everything
    written out is a *relative* second from this origin, so the two agree.
    """

    perf: float
    wall: float

    @classmethod
    def now(cls) -> Clockbase:
        return cls(perf=time.perf_counter(), wall=time.time())

    def rel_perf(self, t: float) -> float:
        return t - self.perf

    def rel_wall(self, t: float) -> float:
        return t - self.wall

    def mark(self) -> float:
        return time.time() - self.wall


class RouterzSampler:
    """Polls /routerz on an interval and keeps every sample."""

    def __init__(self, router_url: str, base: Clockbase) -> None:
        self.url = f"{router_url}/routerz"
        self.base = base
        self.samples: list[dict[str, Any]] = []
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    async def _loop(self) -> None:
        async with httpx.AsyncClient(timeout=2.0) as c:
            while not self._stop.is_set():
                at = self.base.mark()
                try:
                    body = (await c.get(self.url)).json()
                    self.samples.append({"t": round(at, 4), **body})
                except Exception as exc:  # noqa: BLE001 - a missed sample is data too
                    self.samples.append({"t": round(at, 4), "error": type(exc).__name__})
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._stop.wait(), SAMPLE_INTERVAL_S)

    def start(self) -> None:
        self._stop.clear()
        self._task = asyncio.create_task(self._loop(), name="routerz-sampler")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(self._task, 3.0)
            self._task = None

    def state_of(self, worker: str) -> list[tuple[float, str, str]]:
        """(t, health, breaker) for one worker across every sample."""
        out: list[tuple[float, str, str]] = []
        for s in self.samples:
            for w in s.get("workers", []):
                if w["name"] == worker:
                    out.append((float(s["t"]), str(w["state"]), str(w["breaker"])))
        return out

    def first_time(self, worker: str, *, state: str | None = None,
                   breaker: str | None = None, after: float = 0.0) -> float | None:
        for t, health, brk in self.state_of(worker):
            if t < after:
                continue
            if state is not None and health != state:
                continue
            if breaker is not None and brk != breaker:
                continue
            return t
        return None

    def transitions(self, worker: str) -> list[dict[str, Any]]:
        """Every observed change in (health, breaker) with the time it was seen.

        These are *observations at sampling resolution*, not the router's own
        transition log, so a change is dated to within SAMPLE_INTERVAL_S. The
        summary says so.
        """
        out: list[dict[str, Any]] = []
        prev: tuple[str, str] | None = None
        for t, health, brk in self.state_of(worker):
            cur = (health, brk)
            if cur != prev:
                out.append({"t": t, "state": health, "breaker": brk})
                prev = cur
        return out


def record_row(rec: RequestRecord, base: Clockbase) -> dict[str, Any]:
    return {
        "index": rec.index,
        "send_t": round(base.rel_perf(rec.send_ts), 4),
        "done_t": round(base.rel_perf(rec.done_ts), 4),
        "latency_ms": round(rec.latency_ms, 3),
        "status": rec.status,
        "error_code": rec.error_code,
        "error": rec.error,
        "served_by": rec.served_by,
        "retried": rec.retried,
        "batch_size": rec.batch_size,
        "is_warmup": rec.is_warmup,
    }


def latency_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """p50/p99 over successful requests only, as loadgen/stats.py defines it."""
    ok = [r["latency_ms"] for r in rows if r["status"] == 200]
    if not ok:
        return {"n_ok": 0, "p50_ms": None, "p99_ms": None, "max_ms": None}
    return {
        "n_ok": len(ok),
        "p50_ms": round(percentile(ok, 50), 3),
        "p99_ms": round(percentile(ok, 99), 3),
        "max_ms": round(max(ok), 3),
    }


def status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(r["status"]) for r in rows))


def served_split(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(r["served_by"]) for r in rows))


def share_by_bucket(
    rows: list[dict[str, Any]], worker: str, bucket_s: float = 2.0
) -> list[dict[str, Any]]:
    """Fraction of requests served by `worker`, in time buckets."""
    buckets: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        buckets.setdefault(int(r["send_t"] // bucket_s), []).append(r)
    out: list[dict[str, Any]] = []
    for b in sorted(buckets):
        rs = buckets[b]
        named = [r for r in rs if r["served_by"]]
        out.append(
            {
                "t_start": round(b * bucket_s, 2),
                "n": len(rs),
                "n_named": len(named),
                f"share_{worker}": (
                    round(sum(1 for r in named if r["served_by"] == worker) / len(named), 4)
                    if named else None
                ),
            }
        )
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


async def fetch_metrics(router_url: str) -> str:
    async with httpx.AsyncClient(timeout=5.0) as c:
        return (await c.get(f"{router_url}/metrics")).text


def metric_value(text: str, name: str) -> float | None:
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        if line.split("{")[0].split(" ")[0] == name:
            with contextlib.suppress(ValueError):
                return float(line.rsplit(" ", 1)[1])
    return None


# ---------------------------------------------------------------------------
# Scenario harness
# ---------------------------------------------------------------------------
@dataclass
class ScenarioConfig:
    name: str
    concurrency: int = 4
    duration_s: float = 24.0
    warmup_s: float = 4.0
    queue_depth: int = 256
    max_batch_size: int = 8


class ScenarioRun:
    """Owns the tier, the load, the sampler and the artifacts for one scenario."""

    def __init__(self, cfg: ScenarioConfig, out_dir: Path) -> None:
        self.cfg = cfg
        self.out_dir = out_dir
        self.tier = Tier(
            n_workers=2,
            synthetic=False,
            faults_enabled=True,
            queue_depth=cfg.queue_depth,
            max_batch_size=cfg.max_batch_size,
            router_timeout_ms=ROUTER_TIMEOUT_MS,
            trace="off",
        )
        self.base = Clockbase.now()
        self.sampler: RouterzSampler | None = None
        self.events: list[dict[str, Any]] = []
        self.injections: list[dict[str, Any]] = []
        self.metrics_text = ""

    def note(self, what: str, **extra: Any) -> float:
        at = self.base.mark()
        self.injections.append({"t": round(at, 4), "what": what, **extra})
        print(f"    t={at:7.2f}s  {what} {extra if extra else ''}", flush=True)
        return at

    async def drive(self, script: Any) -> None:
        """Run the load and `script` (the fault timeline) side by side."""
        self.base = Clockbase.now()
        sampler = RouterzSampler(self.tier.router_url, self.base)
        sampler.start()
        self.sampler = sampler
        load = asyncio.create_task(
            run_closed_loop(
                self.tier.router_url,
                concurrency=self.cfg.concurrency,
                duration_s=self.cfg.duration_s,
                warmup_s=self.cfg.warmup_s,
                seed=1234,
                buckets=("short",),
                timeout_s=30.0,
            )
        )
        await script(self)
        result = await load
        self.events = [record_row(r, self.base) for r in result.records]
        self.metrics_text = await fetch_metrics(self.tier.router_url)
        await sampler.stop()

    def measured(self) -> list[dict[str, Any]]:
        return [r for r in self.events if not r["is_warmup"]]

    def write(self, summary: dict[str, Any], quiet: dict[str, Any]) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(self.out_dir / "events.jsonl", self.events)
        write_jsonl(self.out_dir / "routerz.jsonl", self.sampler.samples if self.sampler else [])
        (self.out_dir / "metrics.txt").write_text(self.metrics_text, encoding="utf-8")
        (self.out_dir / "box_quiet.json").write_text(json.dumps(quiet, indent=2), encoding="utf-8")
        summary = {
            "scenario": self.cfg.name,
            "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": "DantheMan124/deberta-preference-reward",
            "tier": {
                "workers": 2,
                "router_timeout_ms": ROUTER_TIMEOUT_MS,
                "queue_depth": self.cfg.queue_depth,
                "max_batch_size": self.cfg.max_batch_size,
                "concurrency": self.cfg.concurrency,
                "duration_s": self.cfg.duration_s,
                "warmup_s": self.cfg.warmup_s,
            },
            "box_quiet_passed": bool(quiet.get("passed")),
            "latency_figures_are_indicative_only": not bool(quiet.get("passed")),
            "sampling_interval_s": SAMPLE_INTERVAL_S,
            "injections": self.injections,
            **summary,
        }
        (self.out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(f"    wrote {self.out_dir}")


async def set_fault(worker_url: str, spec: dict[str, Any]) -> None:
    async with httpx.AsyncClient(timeout=10.0) as c:
        r = await c.post(f"{worker_url}/admin/fault", json=spec)
        r.raise_for_status()


# ---------------------------------------------------------------------------
# The four scenarios
# ---------------------------------------------------------------------------
async def scenario_kill_worker(run: ScenarioRun) -> dict[str, Any]:
    """SIGTERM worker B while it has requests in flight."""

    async def script(r: ScenarioRun) -> None:
        await asyncio.sleep(10.0)
        b = r.tier.worker("B")
        r.note("terminate worker B", pid=b.pid)
        b.kill()
        await asyncio.sleep(r.cfg.duration_s + r.cfg.warmup_s - 10.0 + 1.0)

    await run.drive(script)
    rows = run.measured()
    killed_at = next(i["t"] for i in run.injections if "terminate" in i["what"])
    after = [r for r in rows if r["send_t"] >= killed_at]
    # Requests already on the wire when the process died. These are the ones the
    # client loses outright: the router cannot retry them, because bytes had
    # already been sent to a worker that no longer exists.
    in_flight = [
        r for r in rows if r["send_t"] < killed_at <= r["done_t"]
    ]
    failed = [r for r in after if r["status"] not in (200, 429)]
    retried = [r for r in rows if r["retried"]]
    sampler = run.sampler
    assert sampler is not None

    # Convergence: the first moment after the kill from which every later
    # request was served by A. Measured on send time, so it answers "when did
    # the router stop aiming at the dead worker", not "when did it stop failing".
    converged: float | None = None
    named_after = [r for r in after if r["served_by"]]
    for i, r in enumerate(named_after):
        if all(x["served_by"] == "A" for x in named_after[i:]):
            converged = r["send_t"]
            break

    return {
        "question": "what does the client see when a worker is killed mid-batch",
        "killed_worker": "B",
        "killed_at_s": round(killed_at, 3),
        "requests_measured": len(rows),
        "requests_after_kill": len(after),
        "requests_in_flight_at_kill": len(in_flight),
        "in_flight_at_kill_statuses": status_counts(in_flight),
        "in_flight_at_kill_served_by": served_split(in_flight),
        "failed_requests_after_kill": len(failed),
        "failure_statuses": status_counts(failed),
        "served_split_before_kill": served_split(
            [r for r in rows if r["send_t"] < killed_at]
        ),
        "served_split_after_kill": served_split(after),
        "first_request_served_only_by_A_at_s": (
            None if converged is None else round(converged, 3)
        ),
        "convergence_after_kill_s": (
            None if converged is None else round(converged - killed_at, 3)
        ),
        "router_state_transitions_B": sampler.transitions("B"),
        "B_first_seen_SUSPECT_at_s": sampler.first_time("B", state="SUSPECT", after=killed_at),
        "B_first_seen_DOWN_at_s": sampler.first_time("B", state="DOWN", after=killed_at),
        "retried_requests": len(retried),
        "retried_outcomes": status_counts(retried),
        "retried_latencies_ms": [r["latency_ms"] for r in retried],
        "router_retries_total": metric_value(run.metrics_text, "rs_router_retries_total"),
        "unsafe_retries": 0,
        "unsafe_retries_note": (
            "A retry is only issued for a pre-send connection failure, so no request "
            "that reached a worker can be retried. Enforced by "
            "tests/test_router.py::test_a_500_that_reached_a_worker_is_never_retried."
        ),
        "status_counts": status_counts(rows),
        "latency": latency_stats(rows),
    }


async def scenario_latency(run: ScenarioRun) -> dict[str, Any]:
    """2000 ms injected into worker B's forward pass, then removed."""
    inject_at = 8.0
    clear_at = 26.0

    async def script(r: ScenarioRun) -> None:
        await asyncio.sleep(inject_at)
        r.note("inject latency 2000 ms on B")
        await set_fault(r.tier.worker("B").url, {"kind": "latency", "ms": 2000})
        await asyncio.sleep(clear_at - inject_at)
        r.note("clear fault on B")
        await set_fault(r.tier.worker("B").url, {"kind": "clear"})
        await asyncio.sleep(r.cfg.duration_s + r.cfg.warmup_s - clear_at + 1.0)

    await run.drive(script)
    rows = run.measured()
    before = [r for r in rows if r["send_t"] < inject_at]
    during = [r for r in rows if inject_at <= r["send_t"] < clear_at]
    after = [r for r in rows if r["send_t"] >= clear_at]
    sampler = run.sampler
    assert sampler is not None

    def share_a(rs: list[dict[str, Any]]) -> float | None:
        named = [r for r in rs if r["served_by"]]
        if not named:
            return None
        return round(sum(1 for r in named if r["served_by"] == "A") / len(named), 4)

    return {
        "question": (
            "does least-outstanding move load off a slow worker, "
            "and does the router time out"
        ),
        "injected": {"worker": "B", "kind": "latency", "ms": 2000},
        "injected_at_s": inject_at,
        "cleared_at_s": clear_at,
        "phases": {
            "before": {"n": len(before), **latency_stats(before),
                       "statuses": status_counts(before), "share_A": share_a(before)},
            "during": {"n": len(during), **latency_stats(during),
                       "statuses": status_counts(during), "share_A": share_a(during)},
            "after": {"n": len(after), **latency_stats(after),
                      "statuses": status_counts(after), "share_A": share_a(after)},
        },
        "share_of_A_over_time": share_by_bucket(rows, "A", bucket_s=2.0),
        "router_timeouts_504": sum(1 for r in rows if r["status"] == 504),
        "router_timeouts_504_during": sum(1 for r in during if r["status"] == 504),
        "B_stayed_UP_throughout": all(
            s == "UP" for _, s, _ in sampler.state_of("B")
        ),
        "router_state_transitions_B": sampler.transitions("B"),
        "status_counts": status_counts(rows),
        "latency": latency_stats(rows),
    }


async def scenario_queue_exhaustion(run: ScenarioRun) -> dict[str, Any]:
    """Small queues, 16 clients: the workers should shed, not be ejected."""

    async def script(r: ScenarioRun) -> None:
        r.note(
            f"steady overload at concurrency {r.cfg.concurrency} against "
            f"QUEUE_DEPTH={r.cfg.queue_depth}, MAX_BATCH_SIZE={r.cfg.max_batch_size}"
        )
        await asyncio.sleep(r.cfg.duration_s + r.cfg.warmup_s + 1.0)

    await run.drive(script)
    rows = run.measured()
    sampler = run.sampler
    assert sampler is not None
    n429 = sum(1 for r in rows if r["status"] == 429)
    states = {w: sampler.state_of(w) for w in ("A", "B")}
    return {
        "question": "is a 429 treated as a worker failure (it must not be)",
        "queue_depth": run.cfg.queue_depth,
        "max_batch_size": run.cfg.max_batch_size,
        "concurrency": run.cfg.concurrency,
        "requests_measured": len(rows),
        "http_429": n429,
        "rate_429": round(n429 / len(rows), 4) if rows else None,
        "status_counts": status_counts(rows),
        "served_split": served_split(rows),
        "http_504": sum(1 for r in rows if r["status"] == 504),
        "breaker_openings_attributable_to_429": 0,
        "breaker_note": (
            "A 429 never increments a worker's failure counter, so no number of "
            "them can open a breaker; only the 504s in this run can. Asserted by "
            "tests/test_router.py::test_a_429_is_returned_untouched_and_is_not_a_failure."
        ),
        "share_served_by_busiest_worker": (
            round(max(served_split(rows).values()) / len(rows), 4) if rows else None
        ),
        "skew_note": (
            "Least-outstanding-requests prefers whichever worker answers fastest, "
            "and a 429 is the fastest answer there is: a worker refusing work "
            "drops to zero outstanding instantly and is picked again immediately. "
            "Under shed load the rule therefore concentrates traffic on the worker "
            "doing the least useful work. Named, not fixed -- see docs/multi-tier.md."
        ),
        "any_sample_where_a_worker_was_not_UP": any(
            s != "UP" for w in states for _, s, _ in states[w]
        ),
        "any_sample_where_a_breaker_was_not_CLOSED": any(
            b != "CLOSED" for w in states for _, _, b in states[w]
        ),
        "samples_per_worker": {w: len(states[w]) for w in states},
        "router_state_transitions_A": sampler.transitions("A"),
        "router_state_transitions_B": sampler.transitions("B"),
        "router_retries_total": metric_value(run.metrics_text, "rs_router_retries_total"),
        "decision": (
            "429 is not counted as an upstream failure. A worker answering 429 is "
            "healthy and refusing work it cannot queue; ejecting it would move the "
            "whole offered load onto the other worker and turn shed load into an "
            "outage. See Upstream.record_request_result."
        ),
        "latency": latency_stats(rows),
    }


async def scenario_hang(run: ScenarioRun) -> dict[str, Any]:
    """Worker B stops consuming its queue. Nothing may wait past the timeout."""
    inject_at = 8.0

    async def script(r: ScenarioRun) -> None:
        await asyncio.sleep(inject_at)
        r.note("inject hang on B")
        await set_fault(r.tier.worker("B").url, {"kind": "hang"})
        await asyncio.sleep(r.cfg.duration_s + r.cfg.warmup_s - inject_at + 1.0)
        r.note("clear hang on B so the drain can finish")
        await set_fault(r.tier.worker("B").url, {"kind": "clear"})

    await run.drive(script)
    rows = run.measured()
    sampler = run.sampler
    assert sampler is not None
    after = [r for r in rows if r["send_t"] >= inject_at]
    timeouts = [r for r in after if r["status"] == 504]
    over_budget = [
        r for r in rows if r["latency_ms"] > ROUTER_TIMEOUT_MS * 1.25
    ]
    down_at = sampler.first_time("B", state="DOWN", after=inject_at)
    open_at = sampler.first_time("B", breaker="OPEN", after=inject_at)
    return {
        "question": "how long does a wedged worker hurt the client, and what stops it",
        "injected": {"worker": "B", "kind": "hang"},
        "injected_at_s": inject_at,
        "requests_after_injection": len(after),
        "router_timeouts_504_after_injection": len(timeouts),
        "requests_over_1_25x_router_timeout": len(over_budget),
        "max_latency_ms": round(max((r["latency_ms"] for r in rows), default=0.0), 3),
        "router_timeout_ms": ROUTER_TIMEOUT_MS,
        "B_first_seen_DOWN_at_s": down_at,
        "time_to_DOWN_s": None if down_at is None else round(down_at - inject_at, 3),
        "B_first_seen_breaker_OPEN_at_s": open_at,
        "time_to_breaker_OPEN_s": None if open_at is None else round(open_at - inject_at, 3),
        "router_state_transitions_B": sampler.transitions("B"),
        "served_split_after_injection": served_split(after),
        "status_counts": status_counts(rows),
        "latency": latency_stats(rows),
    }


SCENARIOS: dict[str, tuple[ScenarioConfig, Any]] = {
    "kill-worker-mid-batch": (
        ScenarioConfig("kill-worker-mid-batch", concurrency=4, duration_s=24.0, warmup_s=4.0),
        scenario_kill_worker,
    ),
    "latency-2s-on-one-worker": (
        ScenarioConfig("latency-2s-on-one-worker", concurrency=4, duration_s=36.0, warmup_s=4.0),
        scenario_latency,
    ),
    # MAX_BATCH_SIZE=2, not the default 8, and that is the whole point of the
    # first attempt at this scenario. With a batch of 8, sixteen closed-loop
    # clients across two workers cannot overrun a queue of 4: the eight clients
    # aimed at one worker all fit in a single batch, so nothing is ever left
    # waiting and no request is refused. That run produced zero 429s and is kept
    # as results/faults/queue-exhaustion-batch8-never-sheds. Shrinking the batch
    # to 2 leaves six of the eight waiting on a queue of four, which is what
    # makes the queue -- rather than the model -- the binding constraint.
    "queue-exhaustion": (
        ScenarioConfig(
            "queue-exhaustion", concurrency=16, duration_s=16.0, warmup_s=3.0,
            queue_depth=4, max_batch_size=2,
        ),
        scenario_queue_exhaustion,
    ),
    "hang-worker": (
        ScenarioConfig("hang-worker", concurrency=4, duration_s=26.0, warmup_s=4.0),
        scenario_hang,
    ),
}


async def run_scenario(name: str, skip_quiet: bool) -> dict[str, Any]:
    cfg, fn = SCENARIOS[name]
    out_dir = OUT_ROOT / name
    print(f"\n=== {name} ===", flush=True)
    if skip_quiet:
        quiet = {"passed": False, "skipped": True,
                 "note": "box_quiet was not run; treat every latency figure as indicative"}
    else:
        print("    measuring box contention (30 s)...", flush=True)
        quiet = box_quiet.measure()
        print(f"    box_quiet passed={quiet['passed']} cpu_mean={quiet['cpu_mean_pct']}%",
              flush=True)

    run = ScenarioRun(cfg, out_dir)
    print("    starting tier (2 workers, real checkpoint)...", flush=True)
    table = run.tier.up()
    assert table["eligible_count"] == 2, table
    try:
        await wait_ready(run.tier.router_url, timeout_s=60.0)
        summary = await fn(run)
    finally:
        run.tier.down(drain_wait_s=2.0)
    run.write(summary, quiet)
    return summary


async def amain(names: list[str], skip_quiet: bool) -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    index: dict[str, Any] = {}
    for name in names:
        index[name] = await run_scenario(name, skip_quiet)
    (OUT_ROOT / "index.json").write_text(
        json.dumps(
            {
                "scenarios": list(index),
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "command": "python faults/run_faults.py",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nall scenarios done")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", action="append", choices=sorted(SCENARIOS), default=None)
    ap.add_argument(
        "--skip-box-quiet",
        action="store_true",
        help="skip the 30 s contention measurement; forces the indicative-only flag",
    )
    args = ap.parse_args(argv)
    names = args.only or list(SCENARIOS)
    return asyncio.run(amain(names, args.skip_box_quiet))


if __name__ == "__main__":
    raise SystemExit(main())
