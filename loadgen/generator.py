"""Hand-rolled asyncio load generator for the reward-serve scoring endpoint.

Two modes:
  * open loop   -- fixed offered arrival rate, Poisson (exponential) gaps. The
                   generator does NOT wait for a response before sending the
                   next request, so a slow server shows up as queueing, not as
                   a reduced offered load. This is the mode that can measure a
                   latency knee.
  * closed loop -- N concurrent clients; each fires its next request only when
                   its previous one completes. Offered load is therefore a
                   consequence of server speed, not an input.

Per request we record: send_ts, first_byte_ts, done_ts, HTTP status, and the
batch metadata the server returns. `first_byte_ts` is the moment the response
*headers* arrive (httpx streaming), which for this JSON endpoint is the first
byte of the response off the socket.

All timestamps are `time.perf_counter()` seconds, comparable within one run.

Contract (docs/contract.md in the server-core worktree, read 2026-08-18):
POST {base_url}/v1/score  {"prompt": str, "response": str}   (Form A, one pair)
  -> 200 {"results": [{"score", "batch_size", "queue_wait_ms", "forward_ms"}],
          "model", "max_length", "temperature"}
  -> 429 {"error": "queue_full"}        queue at QUEUE_DEPTH, returned immediately
  -> 504 {"error": "deadline_exceeded"} deadline passed before a forward pass
  -> 503 {"error": "draining" | "not_ready"}
  -> 422 malformed request
We send one pair per request, so `results` always has length 1. Missing or
unknown metadata fields degrade to None rather than failing the run, so the
harness keeps working against a simpler stub.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx

from loadgen.workload import Sample, build_corpus

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

Mode = Literal["open", "closed"]

SCORE_PATH = "/v1/score"
READY_PATH = "/readyz"


@dataclass
class RequestRecord:
    """One request's full timeline plus whatever the server told us about it."""

    index: int
    bucket: str
    send_ts: float
    first_byte_ts: float | None
    done_ts: float
    status: int
    is_warmup: bool
    # server batch metadata (None when the server did not report it)
    batch_size: int | None = None
    queue_wait_ms: float | None = None
    forward_ms: float | None = None
    score: float | None = None
    error: str | None = None
    # server-supplied code from an error body, e.g. "queue_full",
    # "deadline_exceeded", "draining", "not_ready" (contract section 2)
    error_code: str | None = None
    # Multi-tier only. Populated from the router's response headers, so a fault
    # run can say WHICH worker served a request and whether the router retried
    # it. Both are None against a single-tier server, which sends no such
    # headers -- the harness must keep working there unchanged.
    served_by: str | None = None
    retried: bool | None = None

    @property
    def latency_ms(self) -> float:
        return (self.done_ts - self.send_ts) * 1000.0

    @property
    def ttfb_ms(self) -> float | None:
        if self.first_byte_ts is None:
            return None
        return (self.first_byte_ts - self.send_ts) * 1000.0

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["latency_ms"] = self.latency_ms
        d["ttfb_ms"] = self.ttfb_ms
        return d


@dataclass
class RunResult:
    """Everything one generator invocation produced."""

    records: list[RequestRecord]
    mode: Mode
    warmup_s: float
    duration_s: float
    target_rps: float | None
    concurrency: int | None
    max_inflight_observed: int = 0
    dropped_no_slot: int = 0
    started_at_wall: float = field(default_factory=time.time)

    def measured(self) -> list[RequestRecord]:
        return [r for r in self.records if not r.is_warmup]


async def sleep_until(deadline: float, min_slice: float = 0.0005) -> None:
    """Sleep until `time.perf_counter() >= deadline`, resisting early wakeups.

    On Windows, asyncio's event loop fires a timer as soon as it is within
    `loop._clock_resolution` (15.625 ms here) of the scheduled time, so a busy
    loop makes `asyncio.sleep(0.005)` return in ~0.05 ms. Measured on this box:
    an idle loop slept 6.4/12.9/21.8 ms for 5/11/20 ms requested, but with four
    ready callbacks pending the same sleeps returned in 0.05/0.05/8.7 ms.
    A load generator that trusted that would silently send FASTER than its
    configured arrival rate. So: re-check the clock and sleep again.
    """
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return
        await asyncio.sleep(max(remaining, min_slice))


class _Inflight:
    """Counts concurrent in-flight requests and remembers the high-water mark."""

    def __init__(self) -> None:
        self.current = 0
        self.peak = 0

    def enter(self) -> None:
        self.current += 1
        self.peak = max(self.peak, self.current)

    def exit(self) -> None:
        self.current -= 1


async def _one_request(
    client: httpx.AsyncClient,
    url: str,
    sample: Sample,
    index: int,
    is_warmup: bool,
    inflight: _Inflight,
    records: list[RequestRecord],
) -> None:
    payload = {"prompt": sample.prompt, "response": sample.response}
    send_ts = time.perf_counter()
    first_byte_ts: float | None = None
    status = 0
    body: dict[str, Any] = {}
    err: str | None = None
    served_by: str | None = None
    retried_hdr: str | None = None
    inflight.enter()
    try:
        async with client.stream("POST", url, json=payload) as resp:
            first_byte_ts = time.perf_counter()
            status = resp.status_code
            served_by = resp.headers.get("x-rs-worker")
            retried_hdr = resp.headers.get("x-rs-retried")
            raw = await resp.aread()
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    body = parsed
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                err = f"decode: {exc}"
    except Exception as exc:  # network/timeout errors are results, not crashes
        err = f"{type(exc).__name__}: {exc}"
    finally:
        inflight.exit()
    done_ts = time.perf_counter()

    # Contract: 200 bodies are {"results": [ {...per pair...} ], "model", ...}.
    # We send one pair per request, so the metadata lives in results[0]. The
    # flat form is still accepted so the harness works against a simple stub.
    meta: dict[str, Any] = body
    results = body.get("results")
    if isinstance(results, list) and results and isinstance(results[0], dict):
        meta = results[0]

    def _num(key: str) -> Any:
        v = meta.get(key)
        return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    code = body.get("error")
    error_code = code if isinstance(code, str) else None
    bs = _num("batch_size")
    records.append(
        RequestRecord(
            index=index,
            bucket=sample.bucket,
            send_ts=send_ts,
            first_byte_ts=first_byte_ts,
            done_ts=done_ts,
            status=status,
            is_warmup=is_warmup,
            batch_size=int(bs) if bs is not None else None,
            queue_wait_ms=_num("queue_wait_ms"),
            forward_ms=_num("forward_ms"),
            score=_num("score"),
            error=err,
            error_code=error_code,
            served_by=served_by,
            retried=None if retried_hdr is None else retried_hdr == "1",
        )
    )


def _client(base_url: str, timeout_s: float, limit: int) -> httpx.AsyncClient:
    limits = httpx.Limits(max_connections=limit, max_keepalive_connections=limit)
    return httpx.AsyncClient(base_url=base_url, timeout=timeout_s, limits=limits)


async def wait_ready(base_url: str, timeout_s: float = 120.0, poll_s: float = 0.25) -> float:
    """Poll {base_url}/readyz until 200. Returns seconds waited. Raises on timeout."""
    start = time.perf_counter()
    deadline = start + timeout_s
    last = "no attempt"
    async with httpx.AsyncClient(base_url=base_url, timeout=5.0) as client:
        while time.perf_counter() < deadline:
            try:
                r = await client.get(READY_PATH)
                if r.status_code == 200:
                    return time.perf_counter() - start
                last = f"HTTP {r.status_code}"
            except Exception as exc:
                last = f"{type(exc).__name__}: {exc}"
            await asyncio.sleep(poll_s)
    raise TimeoutError(f"{base_url}{READY_PATH} not ready in {timeout_s}s (last: {last})")


async def run_open_loop(
    base_url: str,
    target_rps: float,
    duration_s: float,
    warmup_s: float = 0.0,
    seed: int = 1234,
    buckets: tuple[str, ...] | None = None,
    timeout_s: float = 60.0,
    max_inflight: int = 4096,
) -> RunResult:
    """Fire requests at `target_rps` with exponential inter-arrival gaps."""
    corpus = build_corpus(seed, buckets=buckets)
    rng = random.Random(seed)
    records: list[RequestRecord] = []
    inflight = _Inflight()
    tasks: set[asyncio.Task[None]] = set()
    dropped = 0
    index = 0
    async with _client(base_url, timeout_s, max_inflight) as client:
        t0 = time.perf_counter()
        end = t0 + warmup_s + duration_s
        next_at = t0
        while True:
            next_at += rng.expovariate(target_rps)
            if next_at >= end:
                break
            await sleep_until(next_at)
            if inflight.current >= max_inflight:
                dropped += 1
                index += 1
                continue
            sample = corpus[index % len(corpus)]
            is_warmup = (next_at - t0) < warmup_s
            task = asyncio.create_task(
                _one_request(client, SCORE_PATH, sample, index, is_warmup, inflight, records)
            )
            tasks.add(task)
            task.add_done_callback(tasks.discard)
            index += 1
        if tasks:
            await asyncio.gather(*list(tasks), return_exceptions=True)
    return RunResult(
        records=records, mode="open", warmup_s=warmup_s, duration_s=duration_s,
        target_rps=target_rps, concurrency=None,
        max_inflight_observed=inflight.peak, dropped_no_slot=dropped,
    )


async def run_closed_loop(
    base_url: str,
    concurrency: int,
    duration_s: float,
    warmup_s: float = 0.0,
    seed: int = 1234,
    buckets: tuple[str, ...] | None = None,
    timeout_s: float = 60.0,
) -> RunResult:
    """`concurrency` clients, each sending its next request on completion."""
    corpus = build_corpus(seed, buckets=buckets)
    records: list[RequestRecord] = []
    inflight = _Inflight()
    counter = {"n": 0}

    async with _client(base_url, timeout_s, concurrency) as client:
        t0 = time.perf_counter()
        end = t0 + warmup_s + duration_s

        async def worker() -> None:
            while True:
                now = time.perf_counter()
                if now >= end:
                    return
                idx = counter["n"]
                counter["n"] = idx + 1
                await _one_request(
                    client, SCORE_PATH, corpus[idx % len(corpus)], idx,
                    (now - t0) < warmup_s, inflight, records,
                )

        await asyncio.gather(*[asyncio.create_task(worker()) for _ in range(concurrency)])
    return RunResult(
        records=records, mode="closed", warmup_s=warmup_s, duration_s=duration_s,
        target_rps=None, concurrency=concurrency, max_inflight_observed=inflight.peak,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="reward-serve load generator")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--mode", choices=("open", "closed"), default="open")
    ap.add_argument("--rps", type=float, default=10.0, help="open-loop offered rate")
    ap.add_argument("--concurrency", type=int, default=8, help="closed-loop clients")
    ap.add_argument("--duration", type=float, default=20.0, help="measured seconds")
    ap.add_argument("--warmup", type=float, default=5.0, help="excluded seconds")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--buckets", default="short,medium,long")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--wait-ready", action="store_true", help="poll /readyz first")
    ap.add_argument("--raw-out", type=Path, default=None, help="write per-request JSONL")
    ap.add_argument("--json-out", type=Path, default=None, help="write summary JSON")
    return ap.parse_args(argv)


async def _amain(args: argparse.Namespace) -> int:
    from loadgen.stats import summarize  # local import keeps module deps flat

    if args.wait_ready:
        await wait_ready(args.base_url)
    buckets = tuple(b.strip() for b in args.buckets.split(",") if b.strip())
    if args.mode == "open":
        result = await run_open_loop(
            args.base_url, args.rps, args.duration, args.warmup, args.seed,
            buckets, args.timeout,
        )
    else:
        result = await run_closed_loop(
            args.base_url, args.concurrency, args.duration, args.warmup, args.seed,
            buckets, args.timeout,
        )
    summary = summarize(result)
    if args.raw_out:
        args.raw_out.parent.mkdir(parents=True, exist_ok=True)
        with args.raw_out.open("w", encoding="utf-8") as fh:
            for rec in result.records:
                fh.write(json.dumps(rec.to_json()) + "\n")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_amain(_parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
