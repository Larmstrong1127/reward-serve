"""Tests for reward_serve.obs: hooks, spans, buckets, and the /metrics mount."""

from __future__ import annotations

import asyncio
import math
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from reward_serve import obs
from stub_batcher import DeadlineExceeded, QueueFull, StubBatcher


@pytest.fixture()
def metrics() -> obs.Metrics:
    return obs.setup_metrics()


@pytest.fixture()
def recorded() -> tuple[InMemorySpanExporter, Any]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter, provider.get_tracer("test")


def sample(metrics: obs.Metrics, name: str, **labels: str) -> float:
    value = metrics.registry.get_sample_value(name, labels or None)
    assert value is not None, f"{name}{labels} not present in the registry"
    return float(value)


# -- metric wiring ---------------------------------------------------------
def test_on_complete_counts_status_and_rejections(metrics: obs.Metrics) -> None:
    hooks = obs.ObsHooks(metrics)
    for status in ("ok", "ok", "timeout", "queue_full", "cancelled"):
        hooks.on_complete(status)

    assert sample(metrics, "rs_requests_total", status="ok") == 2.0
    assert sample(metrics, "rs_requests_total", status="timeout") == 1.0
    assert sample(metrics, "rs_rejected_total", reason="deadline_exceeded") == 1.0
    assert sample(metrics, "rs_rejected_total", reason="queue_full") == 1.0
    assert sample(metrics, "rs_rejected_total", reason="cancelled") == 1.0


def test_queue_depth_rises_on_enqueue_and_falls_on_batch(metrics: obs.Metrics) -> None:
    hooks = obs.ObsHooks(metrics)
    for _ in range(5):
        hooks.on_enqueue()
    assert sample(metrics, "rs_queue_depth") == 5.0

    hooks.on_batch_start(3, oldest_wait_ms=4.0)
    assert sample(metrics, "rs_queue_depth") == 2.0
    assert hooks.pending_count == 2

    hooks.on_batch_end(3, forward_ms=42.0)
    assert sample(metrics, "rs_forward_seconds_sum") == pytest.approx(0.042)
    assert sample(metrics, "rs_forward_seconds_count") == 1.0


def test_batch_size_histogram_uses_documented_buckets(metrics: obs.Metrics) -> None:
    hooks = obs.ObsHooks(metrics)
    for size in (1, 2, 8, 8, 33):
        hooks.on_batch_start(size, oldest_wait_ms=0.0)
        hooks.on_batch_end(size, forward_ms=1.0)

    assert sample(metrics, "rs_batch_size_bucket", le="1.0") == 1.0
    assert sample(metrics, "rs_batch_size_bucket", le="8.0") == 4.0
    assert sample(metrics, "rs_batch_size_bucket", le="64.0") == 5.0
    assert sample(metrics, "rs_batch_size_sum") == 52.0


def test_queue_wait_is_observed_once_per_item(metrics: obs.Metrics) -> None:
    hooks = obs.ObsHooks(metrics)
    for _ in range(4):
        hooks.on_enqueue()
    hooks.on_batch_start(4, oldest_wait_ms=6.0)
    assert sample(metrics, "rs_queue_wait_seconds_count") == 4.0


def test_drain_closes_abandoned_items(metrics: obs.Metrics) -> None:
    hooks = obs.ObsHooks(metrics)
    for _ in range(3):
        hooks.on_enqueue()
    assert hooks.drain_abandoned() == 3
    assert sample(metrics, "rs_queue_depth") == 0.0
    assert hooks.pending_count == 0


def test_request_span_records_latency(metrics: obs.Metrics) -> None:
    hooks = obs.ObsHooks(metrics)
    with hooks.request_span("score.request", **{"rs.pairs": 2}):
        pass
    assert sample(metrics, "rs_request_latency_seconds_count") == 1.0


def test_latency_is_recorded_even_when_the_request_raises(metrics: obs.Metrics) -> None:
    hooks = obs.ObsHooks(metrics)
    with pytest.raises(ValueError), hooks.request_span():
        raise ValueError("boom")
    assert sample(metrics, "rs_request_latency_seconds_count") == 1.0


# -- documented buckets ----------------------------------------------------
def test_buckets_are_the_documented_ones(metrics: obs.Metrics) -> None:
    """The buckets on the live collectors are the ones docs/slo.md reasons about."""
    documented = obs.documented_buckets()
    live = {
        "rs_request_latency_seconds": metrics.request_latency_seconds,
        "rs_forward_seconds": metrics.forward_seconds,
        "rs_queue_wait_seconds": metrics.queue_wait_seconds,
        "rs_batch_size": metrics.batch_size,
    }
    for name, collector in live.items():
        # prometheus_client keeps upper bounds without the +Inf edge.
        upper = tuple(collector._upper_bounds)
        assert upper == tuple(documented[name]), name

    assert obs.BATCH_SIZE_BUCKETS[:-1] == (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
    assert math.isinf(obs.LATENCY_BUCKETS[-1])
    # Deliberate, not the client defaults: the defaults have no edge between
    # 100 ms and 250 ms, and this service lives in 20-500 ms.
    assert 0.125 in obs.LATENCY_BUCKETS
    assert 0.175 in obs.LATENCY_BUCKETS
    # No bucket in the 20-500 ms band is more than 55% above the previous edge.
    band = [b for b in obs.LATENCY_BUCKETS if 0.02 <= b <= 0.5]
    for lo, hi in zip(band, band[1:], strict=False):
        assert hi / lo <= 1.55, f"bucket {lo}->{hi} is too coarse for the SLO"


# -- spans and links -------------------------------------------------------
def test_forward_span_carries_batch_size_and_links_every_request(
    metrics: obs.Metrics, recorded: tuple[InMemorySpanExporter, Any]
) -> None:
    exporter, tracer = recorded
    hooks = obs.ObsHooks(metrics, tracer=tracer)

    # The request spans stay open across the batch, which is the real ordering:
    # the HTTP handler is still awaiting its result while the forward pass runs.
    request_span_ids = []
    scopes = []
    for _ in range(3):
        scope = hooks.request_span()
        span = scope.__enter__()
        scopes.append(scope)
        request_span_ids.append(span.get_span_context().span_id)
        hooks.on_enqueue()

    hooks.on_batch_start(3, oldest_wait_ms=7.5)
    hooks.on_batch_end(3, forward_ms=40.0)
    for scope in scopes:
        scope.__exit__(None, None, None)

    spans = {s.name: s for s in exporter.get_finished_spans()}
    forward = spans["batch.forward"]
    assert forward.attributes is not None
    assert forward.attributes["batch.size"] == 3
    assert forward.attributes["batch.linked_requests"] == 3
    assert forward.attributes["batch.forward_ms"] == 40.0

    linked = {link.context.span_id for link in forward.links}
    assert linked == set(request_span_ids)

    # The forward span is deliberately parentless and in its own trace: the
    # requests reach it through links, not through a parent.
    assert forward.parent is None

    # And every request span links back, so one request's trace names its batch.
    for finished in exporter.get_finished_spans():
        if finished.name == "score.request":
            back = {link.context.span_id for link in finished.links}
            assert forward.context is not None
            assert forward.context.span_id in back


def test_back_link_is_skipped_when_the_request_span_already_ended(
    metrics: obs.Metrics, recorded: tuple[InMemorySpanExporter, Any]
) -> None:
    """A client that goes away before its item is batched must not raise here.

    The batch side keeps the relationship: batch.forward still links to the
    request span, so the batch is auditable even when the request is gone.
    """
    exporter, tracer = recorded
    hooks = obs.ObsHooks(metrics, tracer=tracer)
    with hooks.request_span() as span:
        hooks.on_enqueue()
        abandoned_id = span.get_span_context().span_id

    hooks.on_batch_start(1, oldest_wait_ms=1.0)
    hooks.on_batch_end(1, forward_ms=20.0)

    forward = next(s for s in exporter.get_finished_spans() if s.name == "batch.forward")
    assert {link.context.span_id for link in forward.links} == {abandoned_id}


def test_queue_wait_span_is_a_child_of_the_request(
    metrics: obs.Metrics, recorded: tuple[InMemorySpanExporter, Any]
) -> None:
    exporter, tracer = recorded
    hooks = obs.ObsHooks(metrics, tracer=tracer)
    with hooks.request_span() as span:
        hooks.on_enqueue()
        root_id = span.get_span_context().span_id
    hooks.on_batch_start(1, oldest_wait_ms=1.0)
    hooks.on_batch_end(1, forward_ms=20.0)

    wait = next(s for s in exporter.get_finished_spans() if s.name == "queue.wait")
    assert wait.parent is not None
    assert wait.parent.span_id == root_id


# -- /metrics endpoint -----------------------------------------------------
def test_metrics_endpoint_serves_the_registry(metrics: obs.Metrics) -> None:
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = fastapi.FastAPI()
    obs.mount_metrics(app, metrics)
    obs.ObsHooks(metrics).on_complete("ok")

    response = TestClient(app).get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    body = response.text
    for name in (
        "rs_requests_total",
        "rs_queue_depth",
        "rs_queue_wait_seconds_bucket",
        "rs_forward_seconds_bucket",
        "rs_batch_size_bucket",
        "rs_request_latency_seconds_bucket",
    ):
        assert name in body, name
    assert 'rs_requests_total{status="ok"} 1.0' in body


# -- end to end through the stub batcher ------------------------------------
def test_hooks_against_the_stub_batcher(
    metrics: obs.Metrics, recorded: tuple[InMemorySpanExporter, Any]
) -> None:
    exporter, tracer = recorded
    hooks = obs.ObsHooks(metrics, tracer=tracer)

    async def run() -> None:
        batcher = StubBatcher(
            forward=lambda items: [float(len(items))] * len(items),
            max_batch_size=4,
            max_wait_ms=5.0,
            on_enqueue=hooks.on_enqueue,
            on_batch_start=hooks.on_batch_start,
            on_batch_end=hooks.on_batch_end,
            on_complete=hooks.on_complete,
        )
        await batcher.start()

        async def one(n: int) -> float:
            with hooks.request_span():
                return await batcher.submit(("p", f"r{n}"))

        await asyncio.gather(*[one(n) for n in range(8)])
        await batcher.stop()

    asyncio.run(run())

    assert sample(metrics, "rs_requests_total", status="ok") == 8.0
    assert sample(metrics, "rs_request_latency_seconds_count") == 8.0
    assert sample(metrics, "rs_queue_wait_seconds_count") == 8.0
    assert sample(metrics, "rs_queue_depth") == 0.0

    forwards = [s for s in exporter.get_finished_spans() if s.name == "batch.forward"]
    assert forwards, "no forward span was recorded"
    assert sum(int(s.attributes["batch.size"]) for s in forwards if s.attributes) == 8
    # Batching actually happened: at least one pass carried more than one item.
    assert max(int(s.attributes["batch.size"]) for s in forwards if s.attributes) > 1


def test_queue_full_is_immediate(metrics: obs.Metrics) -> None:
    hooks = obs.ObsHooks(metrics)

    async def run() -> None:
        batcher = StubBatcher(
            forward=lambda items: [0.0] * len(items),
            queue_depth=1,
            on_enqueue=hooks.on_enqueue,
            on_complete=hooks.on_complete,
        )
        first = asyncio.create_task(batcher.submit(("p", "r")))
        await asyncio.sleep(0)
        with pytest.raises(QueueFull):
            await batcher.submit(("p", "r"))
        first.cancel()

    asyncio.run(run())
    assert sample(metrics, "rs_rejected_total", reason="queue_full") == 1.0


def test_deadline_exceeded_marks_a_rejection(metrics: obs.Metrics) -> None:
    hooks = obs.ObsHooks(metrics)

    async def run() -> None:
        batcher = StubBatcher(
            forward=lambda items: [0.0] * len(items),
            max_wait_ms=50.0,
            on_enqueue=hooks.on_enqueue,
            on_batch_start=hooks.on_batch_start,
            on_batch_end=hooks.on_batch_end,
            on_complete=hooks.on_complete,
        )
        await batcher.start()
        with pytest.raises(DeadlineExceeded):
            await batcher.submit(("p", "r"), timeout_ms=1.0)
        await batcher.stop()

    asyncio.run(run())
    assert sample(metrics, "rs_rejected_total", reason="deadline_exceeded") == 1.0
