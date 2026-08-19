"""Charter rule 10 for the router tier: ten requests before any scenario runs.

Brings up a router and two workers loading the real checkpoint, sends ten
concurrent requests through the router, and answers two questions with
artifacts rather than assertions:

1. Which worker served each request? Read off the ``x-rs-worker`` header the
   router copies from its chosen upstream.
2. Does one trace span both tiers? Both processes export spans to their own
   stdout, which this script captures and then searches for a trace id that
   appears in the router's ``router.request`` span, the worker's
   ``score.request`` span, and is linked from a ``batch.forward``.

    python scripts/tier_dry_run.py

Writes results/multi-tier-dry-run/dry_run.json. No latency figure is published
here: this is a structural check and the box is not required to be quiet for it.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse  # noqa: E402
import asyncio  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import httpx  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from tier_up import Tier, print_table  # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "multi-tier-dry-run"
PAIRS: list[tuple[str, str]] = [
    ("How do I reverse a list?", "Use the reversed() builtin."),
    ("How do I reverse a list?", "idk"),
    ("What is a bounded queue?", "A queue with a fixed capacity that rejects when full."),
    ("What is 2 + 2?", "4"),
    ("What is 2 + 2?", "Seventeen, probably."),
    ("Name a colour.", "Blue."),
    ("Define latency.", "Time between a request and its response."),
    ("Define throughput.", "Completed requests per unit of time."),
    ("Explain padding in a batch.", "Shorter sequences are filled to the longest length."),
    ("Explain a deadline.", "The time after which a queued request is no longer useful."),
]


def read_spans(path: Path) -> list[dict[str, Any]]:
    """Pull the JSON objects the console span exporter printed into `path`.

    The exporter pretty-prints one object per span with no separator, so the
    file is a concatenation of JSON values rather than JSON lines. `raw_decode`
    walks it.
    """
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    decoder = json.JSONDecoder()
    spans: list[dict[str, Any]] = []
    i = 0
    n = len(text)
    while i < n:
        while i < n and text[i] not in "{[":
            i += 1
        if i >= n:
            break
        try:
            obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(obj, dict) and "context" in obj:
            spans.append(obj)
        i = end
    return spans


def _trace_id(span: dict[str, Any]) -> str | None:
    ctx = span.get("context") or {}
    tid = ctx.get("trace_id")
    return str(tid) if tid else None


def find_cross_tier_trace(
    router_spans: list[dict[str, Any]], worker_spans: dict[str, list[dict[str, Any]]]
) -> dict[str, Any] | None:
    """One trace id carrying a router span and a worker span, with its shape."""
    router_by_trace: dict[str, list[dict[str, Any]]] = {}
    for s in router_spans:
        tid = _trace_id(s)
        if tid:
            router_by_trace.setdefault(tid, []).append(s)
    for worker, spans in worker_spans.items():
        for s in spans:
            tid = _trace_id(s)
            if tid is None or tid not in router_by_trace:
                continue
            if s.get("name") != "score.request":
                continue
            linked = [
                b for b in spans
                if b.get("name") == "batch.forward"
                and any(_trace_id_of_link(link) == tid for link in (b.get("links") or []))
            ]
            return {
                "trace_id": tid,
                "worker": worker,
                "router_spans": sorted({str(r.get("name")) for r in router_by_trace[tid]}),
                "worker_spans": sorted(
                    {str(x.get("name")) for x in spans if _trace_id(x) == tid}
                ),
                "parent_of_worker_span": str(s.get("parent_id")),
                "batch_forward_spans_linking_this_trace": len(linked),
                "batch_sizes_linked": [
                    (b.get("attributes") or {}).get("batch.size") for b in linked
                ],
            }
    return None


def _trace_id_of_link(link: dict[str, Any]) -> str | None:
    ctx = link.get("context") or {}
    tid = ctx.get("trace_id")
    return str(tid) if tid else None


async def drive(base_url: str) -> list[dict[str, Any]]:
    async with httpx.AsyncClient(base_url=base_url, timeout=120.0) as c:
        async def one(i: int, prompt: str, response: str) -> dict[str, Any]:
            r = await c.post("/v1/score", json={"prompt": prompt, "response": response})
            body: dict[str, Any] = {}
            # A non-JSON body is still a result worth recording.
            with contextlib.suppress(Exception):
                body = r.json()
            item = (body.get("results") or [{}])[0]
            return {
                "i": i,
                "status": r.status_code,
                "worker": r.headers.get("x-rs-worker"),
                "retried": r.headers.get("x-rs-retried"),
                "batch_size": item.get("batch_size"),
                "score": item.get("score"),
                "model": body.get("model"),
            }

        return list(await asyncio.gather(*(one(i, p, r) for i, (p, r) in enumerate(PAIRS))))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--synthetic", action="store_true", help="skip the checkpoint load")
    args = ap.parse_args(argv)

    raw = OUT_DIR / "raw"
    tier = Tier(
        n_workers=args.workers,
        synthetic=args.synthetic,
        trace="console",
        log_dir=raw,
    )
    table = tier.up()
    print_table(table)
    try:
        rows = asyncio.run(drive(tier.router_url))
    finally:
        tier.down(drain_wait_s=2.0)

    router_spans = read_spans(tier.log_path("router"))
    worker_spans = {
        w.name: read_spans(tier.log_path(f"worker-{w.name}")) for w in tier.workers
    }
    trace = find_cross_tier_trace(router_spans, worker_spans)

    served = Counter(str(r["worker"]) for r in rows)
    ok = sum(1 for r in rows if r["status"] == 200)
    report = {
        "requests": len(rows),
        "http_200": ok,
        "served_by": dict(served),
        "distinct_workers_used": len(served),
        "retried_any": any(r["retried"] == "1" for r in rows),
        "max_batch_size_observed": max(
            (int(r["batch_size"]) for r in rows if r["batch_size"]), default=0
        ),
        "model": next((r["model"] for r in rows if r["model"]), None),
        "state_table_at_start": table,
        "cross_tier_trace": trace,
        "rows": rows,
        "note": (
            "Structural dry run. No latency figure is published from it and the "
            "box was not required to be quiet."
        ),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "dry_run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{'req':>3} {'status':>7} {'worker':>7} {'batch':>6} {'score':>10}")
    for r in rows:
        score = r["score"]
        shown = f"{score:.4f}" if isinstance(score, (int, float)) else "-"
        print(f"{r['i']:>3} {r['status']:>7} {str(r['worker']):>7} "
              f"{str(r['batch_size']):>6} {shown:>10}")
    print(f"\nserved by: {dict(served)}")
    if trace is None:
        print("NO cross-tier trace found")
    else:
        print(f"cross-tier trace {trace['trace_id']} on worker {trace['worker']}")
        print(f"  router spans: {trace['router_spans']}")
        print(f"  worker spans: {trace['worker_spans']}")
        print(f"  batch.forward spans linking it: "
              f"{trace['batch_forward_spans_linking_this_trace']}")
    print(f"\nwrote {OUT_DIR / 'dry_run.json'}")
    return 0 if ok == len(rows) and trace is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
