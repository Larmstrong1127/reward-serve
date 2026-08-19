"""obs.py wired into the real server and the real batcher, not the stub.

tests/test_obs.py pins the hook behaviour in isolation. This file proves the
same code survives contact with `reward_serve.server.create_app` and
`reward_serve.batcher.ContinuousBatcher`: the adapter matches the batcher's
hook dataclass, the middleware opens one root span per scoring request, the
metrics move, and a real batch produces a forward span linked to every request
that rode it.

A stub scorer is used so no model loads; the batching path is the real one.
"""

from __future__ import annotations

from typing import Any

import pytest
from conftest import StubScorer
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from reward_serve import obs
from reward_serve.batcher import BatcherHooks as RealBatcherHooks
from reward_serve.server import Settings, create_app


def _wire() -> tuple[Any, obs.Metrics, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    metrics = obs.setup_metrics()
    hooks = obs.ObsHooks(metrics, tracer=provider.get_tracer("test"))

    settings = Settings(max_batch_size=8, max_wait_ms=25)
    app = create_app(
        settings,
        scorer=StubScorer(delay_s=0.01),
        hooks=obs.as_batcher_hooks(hooks),
    )
    obs.instrument_app(app, hooks, metrics)
    return app, metrics, exporter


def test_adapter_produces_the_batchers_own_hook_type() -> None:
    """If the batcher's dataclass changes shape, this fails here, not in prod."""
    hooks = obs.ObsHooks(obs.setup_metrics())
    adapted = obs.as_batcher_hooks(hooks)
    assert isinstance(adapted, RealBatcherHooks)
    assert adapted.on_enqueue == hooks.on_enqueue
    assert adapted.on_batch_start == hooks.on_batch_start
    assert adapted.on_batch_end == hooks.on_batch_end
    assert adapted.on_complete == hooks.on_complete


def test_real_server_emits_metrics_and_a_linked_batch_span() -> None:
    app, metrics, exporter = _wire()

    with TestClient(app) as client:
        assert client.get("/readyz").status_code == 200
        body = {
            "pairs": [
                {"prompt": "How do I sort a list?", "response": f"Answer {n}"}
                for n in range(6)
            ]
        }
        response = client.post("/v1/score", json=body)
        assert response.status_code == 200
        results = response.json()["results"]
        assert len(results) == 6

        scrape = client.get("/metrics")
        assert scrape.status_code == 200

    # Six items scored, six items counted.
    assert metrics.registry.get_sample_value("rs_requests_total", {"status": "ok"}) == 6.0
    assert metrics.registry.get_sample_value("rs_queue_depth") == 0.0
    assert metrics.registry.get_sample_value("rs_queue_wait_seconds_count") == 6.0
    assert metrics.registry.get_sample_value("rs_request_latency_seconds_count") == 1.0

    text = scrape.text
    assert 'rs_requests_total{status="ok"} 6.0' in text
    assert "rs_batch_size_bucket" in text

    forwards = [s for s in exporter.get_finished_spans() if s.name == "batch.forward"]
    assert forwards, "the real batcher produced no forward span"
    total_batched = sum(int(s.attributes["batch.size"]) for s in forwards if s.attributes)
    assert total_batched == 6

    # The server-reported batch_size and the span attribute must agree; that
    # equality is the whole reason the batch link can be trusted.
    assert sorted(r["batch_size"] for r in results) == sorted(
        size
        for s in forwards
        if s.attributes
        for size in [int(s.attributes["batch.size"])] * int(s.attributes["batch.size"])
    )

    linked = {link.context.span_id for s in forwards for link in s.links}
    request_spans = [s for s in exporter.get_finished_spans() if s.name == "score.request"]
    assert len(request_spans) == 1
    assert request_spans[0].context is not None
    assert request_spans[0].context.span_id in linked


def test_probe_endpoints_are_not_traced_and_do_not_count_as_latency() -> None:
    """Probes must not bury real requests or skew the latency histogram."""
    app, metrics, exporter = _wire()
    with TestClient(app) as client:
        for _ in range(5):
            client.get("/healthz")
            client.get("/readyz")

    assert not [s for s in exporter.get_finished_spans() if s.name == "score.request"]
    assert metrics.registry.get_sample_value("rs_request_latency_seconds_count") == 0.0


def test_queue_full_and_drain_show_up_in_the_metrics() -> None:
    metrics = obs.setup_metrics()
    hooks = obs.ObsHooks(metrics)
    settings = Settings(max_batch_size=1, max_wait_ms=50, queue_depth=1)
    app = create_app(
        settings,
        scorer=StubScorer(delay_s=0.2),
        hooks=obs.as_batcher_hooks(hooks),
    )
    obs.instrument_app(app, hooks, metrics)

    with TestClient(app) as client:
        codes = set()
        for _ in range(6):
            codes.add(
                client.post(
                    "/v1/score",
                    json={"pairs": [{"prompt": "p", "response": "r"}] * 4},
                ).status_code
            )
        assert 429 in codes, f"expected backpressure, saw {codes}"
        assert client.post("/admin/drain").status_code == 200

    rejected = metrics.registry.get_sample_value("rs_rejected_total", {"reason": "queue_full"})
    assert rejected is not None and rejected > 0


@pytest.mark.parametrize("path", ["/metrics"])
def test_metrics_path_is_not_in_the_openapi_schema(path: str) -> None:
    app, _, _ = _wire()
    assert path not in app.openapi()["paths"]
