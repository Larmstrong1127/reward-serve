"""Run the real server with obs wired in and capture the evidence.

This replaces scripts/forward_dry_run.py's stub path: the model is real, the
batcher is `reward_serve.batcher.ContinuousBatcher`, the HTTP surface is
`reward_serve.server.create_app`, and the only thing added is
`reward_serve.obs`. Twenty requests are fired concurrently so that batches
actually form, and three artefacts are written:

    results/obs_dry_run.json      the run's numbers, with the box_quiet report
    results/obs_dry_run.prom      a real /metrics scrape
    results/trace_batch_link.json one score.request trace and the batch.forward
                                  span it links to, with every sibling listed

Spans are captured through a file exporter rather than OTLP, because the
collector runs in Docker and Docker is not installed on this machine. The span
contents are identical either way; only the transport differs.

Run:

    cd D:\\ClaudeProjects\\reward-serve-wt-observability
    D:\\ClaudeProjects\\reward-serve\\.venv\\Scripts\\python.exe scripts/obs_dry_run.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HOME", r"D:\ClaudeCache\hf")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)

from reward_serve import obs  # noqa: E402
from reward_serve.server import Settings, create_app  # noqa: E402
from scripts.box_quiet import require_quiet  # noqa: E402

N_REQUESTS = 20

PROMPTS = [
    ("How do I reverse a list?", "Use the reversed() builtin or a [::-1] slice."),
    ("How do I reverse a list?", "You cannot reverse a list."),
    ("What is a hash map?", "A structure mapping keys to values in constant time."),
    ("What is a hash map?", "It is a kind of sandwich."),
    ("Explain recursion.", "A function defined in terms of a smaller call to itself."),
]


def _span_to_dict(span: Any) -> dict[str, Any]:
    ctx = span.context
    return {
        "name": span.name,
        "trace_id": format(ctx.trace_id, "032x"),
        "span_id": format(ctx.span_id, "016x"),
        "parent_span_id": format(span.parent.span_id, "016x") if span.parent else None,
        "start_time_unix_nano": span.start_time,
        "duration_ms": round((span.end_time - span.start_time) / 1e6, 3),
        "attributes": dict(span.attributes or {}),
        "links": [
            {
                "trace_id": format(link.context.trace_id, "032x"),
                "span_id": format(link.context.span_id, "016x"),
                "attributes": dict(link.attributes or {}),
            }
            for link in span.links
        ],
    }


def main() -> int:
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    quiet = require_quiet(results_dir / "box_quiet_obs_dry_run.json")

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    metrics = obs.setup_metrics()
    hooks = obs.ObsHooks(metrics, tracer=provider.get_tracer("reward_serve.obs"))

    settings = Settings.from_env()
    app = create_app(settings, hooks=obs.as_batcher_hooks(hooks))
    obs.instrument_app(app, hooks, metrics)

    from fastapi.testclient import TestClient

    load_started = time.perf_counter()
    latencies_ms: list[float] = []
    payloads: list[dict[str, Any]] = []

    with TestClient(app) as client:
        load_s = time.perf_counter() - load_started
        assert client.get("/readyz").status_code == 200, "server never became ready"

        # Concurrent, because a serial client would never let a batch form and
        # the whole point of this run is to see batching happen.
        import concurrent.futures as cf

        def one(n: int) -> tuple[int, float, dict[str, Any]]:
            prompt, response = PROMPTS[n % len(PROMPTS)]
            started = time.perf_counter()
            reply = client.post(
                "/v1/score", json={"prompt": prompt, "response": f"{response} ({n})"}
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return reply.status_code, elapsed_ms, reply.json()

        with cf.ThreadPoolExecutor(max_workers=N_REQUESTS) as pool:
            for status, elapsed_ms, body in pool.map(one, range(N_REQUESTS)):
                assert status == 200, body
                latencies_ms.append(elapsed_ms)
                payloads.append(body)

        scrape = client.get("/metrics")
        (results_dir / "obs_dry_run.prom").write_text(scrape.text, encoding="utf-8")

    spans = [_span_to_dict(s) for s in exporter.get_finished_spans()]
    forwards = [s for s in spans if s["name"] == "batch.forward"]
    requests = [s for s in spans if s["name"] == "score.request"]

    # One trace, end to end, with the batch it joined and its siblings.
    linked_forward = next((f for f in forwards if f["links"]), None)
    sample: dict[str, Any] = {}
    if linked_forward is not None:
        sibling_ids = {link["span_id"] for link in linked_forward["links"]}
        chosen = next((r for r in requests if r["span_id"] in sibling_ids), None)
        sample = {
            "what_this_shows": (
                "One score.request trace, the queue.wait span inside it, and the "
                "batch.forward span it links to. batch.size is how many requests "
                "shared that forward pass; the links enumerate them."
            ),
            "request": chosen,
            "queue_wait": next(
                (
                    s
                    for s in spans
                    if s["name"] == "queue.wait"
                    and chosen is not None
                    and s["parent_span_id"] == chosen["span_id"]
                ),
                None,
            ),
            "batch_forward": linked_forward,
            "siblings_in_that_batch": len(sibling_ids),
        }
        (results_dir / "trace_batch_link.json").write_text(
            json.dumps(sample, indent=2), encoding="utf-8"
        )

    server_forward_ms = [r["results"][0]["forward_ms"] for r in payloads]
    server_wait_ms = [r["results"][0]["queue_wait_ms"] for r in payloads]
    batch_sizes = [r["results"][0]["batch_size"] for r in payloads]
    info = payloads[0]

    report = {
        "model_id": info["model"],
        "max_length": info["max_length"],
        "temperature": info["temperature"],
        "device": "cpu",
        "n_requests": N_REQUESTS,
        "settings": {
            "max_batch_size": settings.max_batch_size,
            "max_wait_ms": settings.max_wait_ms,
            "queue_depth": settings.queue_depth,
            "request_timeout_ms": settings.request_timeout_ms,
        },
        "startup_and_model_load_s": round(load_s, 3),
        "forward_ms": {
            "n_distinct_passes": len({round(v, 3) for v in server_forward_ms}),
            "min": round(min(server_forward_ms), 3),
            "median": round(statistics.median(server_forward_ms), 3),
            "max": round(max(server_forward_ms), 3),
        },
        "queue_wait_ms": {
            "min": round(min(server_wait_ms), 3),
            "median": round(statistics.median(server_wait_ms), 3),
            "max": round(max(server_wait_ms), 3),
        },
        "client_latency_ms": {
            "min": round(min(latencies_ms), 3),
            "median": round(statistics.median(latencies_ms), 3),
            "p95": round(sorted(latencies_ms)[int(0.95 * (len(latencies_ms) - 1))], 3),
            "max": round(max(latencies_ms), 3),
        },
        "achieved_batch_sizes": sorted(set(batch_sizes)),
        "max_achieved_batch_size": max(batch_sizes),
        "n_forward_spans": len(forwards),
        "n_request_spans": len(requests),
        "forward_spans_with_links": sum(1 for f in forwards if f["links"]),
        "box_quiet": quiet,
        "command": "python scripts/obs_dry_run.py",
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (results_dir / "obs_dry_run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({k: report[k] for k in ("forward_ms", "queue_wait_ms",
                                             "client_latency_ms", "achieved_batch_sizes")},
                     indent=2))
    print(f"wrote {results_dir / 'obs_dry_run.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
