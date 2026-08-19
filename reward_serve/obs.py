"""Observability for reward-serve: OpenTelemetry traces and Prometheus metrics.

This is the only module the server imports for telemetry. It owns three things:

1. ``setup_tracing`` -- a tracer whose spans go to an OTLP/gRPC collector, with
   a console exporter as the fallback when no collector is reachable.
2. ``setup_metrics`` -- a private Prometheus registry with the seven series the
   dashboards and the SLO in ``docs/slo.md`` are built on.
3. ``ObsHooks`` -- real implementations of the four no-op callables the
   batcher accepts (see ``docs/contract.md`` section 3). Wiring these in is what
   turns the batcher from a black box into something a trace can explain.

Scope: one server process on one machine. There is no second node to correlate
against, so every span and every series here is local.

Trace shape
-----------
Per HTTP request::

    score.request                (root, one trace per request)
      +- queue.wait              (enqueue -> the forward pass starting)

Per forward pass a separate, deliberately parentless span is created::

    batch.forward                (its own trace, attribute batch.size=N)
      links -> the score.request span of every item in that batch

and each of those request spans gets a link back to ``batch.forward``. That
two-way link is the point: opening the trace of one slow request shows which
batch it joined and how many siblings it was packed with, which is the first
question worth asking when tail latency moves.

Matching hook calls to items
----------------------------
The hook signatures in the contract carry no item identity -- ``on_batch_start``
reports only ``batch_size``. Correlation therefore relies on the batcher's FIFO
ordering, which the contract guarantees: ``on_enqueue`` fires before that item
can appear in a batch, and batches are packed from the head of the queue. This
module keeps a FIFO of pending items and takes ``batch_size`` of them off the
head at ``on_batch_start``. If the batcher ever packs out of order the links
become wrong while the metrics stay right.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.trace import Link, Span, SpanContext, Tracer
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from prometheus_client import generate_latest as _generate_latest
from prometheus_client.exposition import CONTENT_TYPE_LATEST

__all__ = [
    "BATCH_SIZE_BUCKETS",
    "FORWARD_BUCKETS",
    "LATENCY_BUCKETS",
    "QUEUE_WAIT_BUCKETS",
    "Metrics",
    "ObsHooks",
    "documented_buckets",
    "metrics_payload",
    "mount_metrics",
    "setup_metrics",
    "setup_tracing",
]

DEFAULT_OTLP_ENDPOINT = "localhost:4317"
SERVICE_NAME_KEY = "service.name"

# ---------------------------------------------------------------------------
# Histogram buckets
# ---------------------------------------------------------------------------
# Chosen for a service whose end-to-end request latency lives in roughly
# 20-500 ms on this machine: see results/obs_dry_run.json for the twenty-request
# real-server CPU dry run those numbers come from, and docs/slo.md for how the latency
# target is derived from them.
#
# Why not the prometheus_client defaults (.005 .01 .025 .05 .075 .1 .25 .5 .75
# 1 2.5 5 7.5 10): they put only two edges between 100 ms and 1 s, so a p99
# sitting near 400 ms gets interpolated across the 250-500 ms bucket -- a 250 ms
# wide guess about a 400 ms number. A histogram the SLO is judged against has to
# resolve finer than the margin the SLO is decided by.
#
# The rule used below: inside the 20-500 ms band no bucket edge is more than
# ~50% above the previous one, so a quantile read off it is accurate to well
# inside the gap between our target and our observed latency. Edges thin out
# above 1 s, where the only question is "how bad", not "how bad exactly", and
# the 5 s edge is the default REQUEST_TIMEOUT_MS so timed-out work lands in a
# bucket of its own instead of smearing across the tail.
LATENCY_BUCKETS: tuple[float, ...] = (
    0.010, 0.020, 0.030, 0.045, 0.065, 0.090, 0.125, 0.175,
    0.250, 0.350, 0.500, 0.750, 1.000, 2.500, 5.000, float("inf"),
)

# The forward pass is the floor under the latency above, so its buckets share
# the same edges but stop climbing sooner: a single CPU forward pass over a
# 512-token budget that takes more than a second means the box is contended,
# and one bucket is enough to say so.
FORWARD_BUCKETS: tuple[float, ...] = (
    0.010, 0.020, 0.030, 0.045, 0.065, 0.090, 0.125, 0.175,
    0.250, 0.350, 0.500, 1.000, 2.500, float("inf"),
)

# Queue wait is bounded by MAX_WAIT_MS (default 10 ms) whenever the batcher is
# keeping up, so resolution below 10 ms is what separates "healthy" from "the
# flush timer is the binding constraint". Above 100 ms the queue is backing up
# and the exact figure has stopped mattering.
QUEUE_WAIT_BUCKETS: tuple[float, ...] = (
    0.001, 0.002, 0.004, 0.006, 0.008, 0.010, 0.015, 0.025,
    0.050, 0.100, 0.250, 0.500, 1.000, float("inf"),
)

# Powers of two up to 64, as specified; MAX_BATCH_SIZE defaults to 8.
BATCH_SIZE_BUCKETS: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, float("inf"))


# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------
def _build_exporter(endpoint: str, force_console: bool) -> tuple[SpanExporter, bool]:
    """Return an exporter and whether it is the console fallback."""
    if force_console:
        return ConsoleSpanExporter(), True
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )

        return OTLPSpanExporter(endpoint=endpoint, insecure=True), False
    except Exception:  # pragma: no cover - only without the exporter installed
        return ConsoleSpanExporter(), True


def setup_tracing(
    service_name: str = "reward-serve",
    *,
    endpoint: str | None = None,
    console: bool | None = None,
) -> Tracer:
    """Install a tracer provider and return the tracer the server should use.

    The OTLP/gRPC endpoint comes from ``OTEL_EXPORTER_OTLP_ENDPOINT`` and
    defaults to ``localhost:4317``. Setting ``REWARD_SERVE_TRACE_CONSOLE=1`` (or
    passing ``console=True``) prints spans to stdout instead, which is what the
    tests and any no-collector run use. Building the OTLP exporter is lazy and a
    failure to import it falls back to the console rather than taking the server
    down: telemetry must never be the reason scoring stops.
    """
    if endpoint is None:
        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", DEFAULT_OTLP_ENDPOINT)
    if console is None:
        console = os.environ.get("REWARD_SERVE_TRACE_CONSOLE", "") not in ("", "0")

    resource = Resource.create({SERVICE_NAME_KEY: service_name})
    provider = TracerProvider(resource=resource)
    exporter, is_console = _build_exporter(endpoint, console)
    processor = (
        SimpleSpanProcessor(exporter) if is_console else BatchSpanProcessor(exporter)
    )
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    return provider.get_tracer("reward_serve.obs")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@dataclass
class Metrics:
    """The seven series every dashboard panel and SLO query is built on."""

    registry: CollectorRegistry
    requests_total: Counter
    queue_depth: Gauge
    queue_wait_seconds: Histogram
    forward_seconds: Histogram
    batch_size: Histogram
    request_latency_seconds: Histogram
    rejected_total: Counter


def setup_metrics(registry: CollectorRegistry | None = None) -> Metrics:
    """Build the metric set on a private registry.

    A private registry rather than the process-global default: tests build a
    clean set per case, and the server exposes exactly these series instead of
    whatever else happened to import prometheus_client. Counter names are given
    without the ``_total`` suffix because prometheus_client appends it.
    """
    reg = registry if registry is not None else CollectorRegistry()
    return Metrics(
        registry=reg,
        requests_total=Counter(
            "rs_requests",
            "Scoring items that reached a terminal state, by status.",
            ["status"],
            registry=reg,
        ),
        queue_depth=Gauge(
            "rs_queue_depth",
            "Items enqueued and not yet packed into a forward pass.",
            registry=reg,
        ),
        queue_wait_seconds=Histogram(
            "rs_queue_wait_seconds",
            "Enqueue until the forward pass carrying the item started.",
            buckets=QUEUE_WAIT_BUCKETS,
            registry=reg,
        ),
        forward_seconds=Histogram(
            "rs_forward_seconds",
            "Wall time of one batched forward pass.",
            buckets=FORWARD_BUCKETS,
            registry=reg,
        ),
        batch_size=Histogram(
            "rs_batch_size",
            "Achieved batch size per forward pass.",
            buckets=BATCH_SIZE_BUCKETS,
            registry=reg,
        ),
        request_latency_seconds=Histogram(
            "rs_request_latency_seconds",
            "End-to-end server-side latency of one scoring request.",
            buckets=LATENCY_BUCKETS,
            registry=reg,
        ),
        rejected_total=Counter(
            "rs_rejected",
            "Items the server refused or abandoned, by reason.",
            ["reason"],
            registry=reg,
        ),
    )


def metrics_payload(metrics: Metrics) -> tuple[bytes, str]:
    """Render the registry to the Prometheus text format, with its content type."""
    return _generate_latest(metrics.registry), CONTENT_TYPE_LATEST


def mount_metrics(app: Any, metrics: Metrics, path: str = "/metrics") -> None:
    """Add ``GET /metrics`` to a FastAPI app.

    FastAPI is imported lazily so this module stays usable, and testable,
    without it.
    """
    from fastapi import Response

    async def scrape() -> Response:
        body, content_type = metrics_payload(metrics)
        return Response(content=body, media_type=content_type)

    app.add_api_route(path, scrape, methods=["GET"], include_in_schema=False)


# ---------------------------------------------------------------------------
# Batcher hooks
# ---------------------------------------------------------------------------
@dataclass
class _Pending:
    """One enqueued item, waiting to be matched to a forward pass."""

    root_span: Span
    wait_span: Span
    enqueued_at: float


# Statuses the contract defines that mean the item was never scored. Anything
# listed here also increments rs_rejected_total, so the error budget reads one
# series instead of doing a subtraction between two.
REJECTION_REASONS: dict[str, str] = {
    "timeout": "deadline_exceeded",
    "queue_full": "queue_full",
    "cancelled": "cancelled",
}


@dataclass
class ObsHooks:
    """Real implementations of the four batcher hooks in ``docs/contract.md``.

    Pass ``hooks.on_enqueue``, ``hooks.on_batch_start``, ``hooks.on_batch_end``
    and ``hooks.on_complete`` to ``ContinuousBatcher``. The hooks run on the
    batcher's event loop, so everything here is bounded work: a deque operation,
    a span start or end, and a metric update. No I/O, and no lock is held across
    a span export.
    """

    metrics: Metrics
    tracer: Tracer | None = None
    _pending: deque[_Pending] = field(default_factory=deque, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _forward_span: Span | None = field(default=None, repr=False)

    def _get_tracer(self) -> Tracer:
        if self.tracer is None:
            self.tracer = trace.get_tracer("reward_serve.obs")
        return self.tracer

    # -- request scope ----------------------------------------------------
    @contextmanager
    def request_span(self, name: str = "score.request", **attrs: Any) -> Iterator[Span]:
        """Root span for one HTTP request, plus its end-to-end latency sample.

        The latency histogram is observed here rather than in ``on_complete``
        because this is the client-visible number: it covers request parsing and
        response assembly too, not only the queue and the forward pass.
        """
        started = time.perf_counter()
        with self._get_tracer().start_as_current_span(name) as span:
            for key, value in attrs.items():
                span.set_attribute(key, value)
            try:
                yield span
            finally:
                self.metrics.request_latency_seconds.observe(
                    time.perf_counter() - started
                )

    # -- hooks ------------------------------------------------------------
    def on_enqueue(self) -> None:
        """One item accepted onto the queue. Opens its ``queue.wait`` span."""
        root = trace.get_current_span()
        wait = self._get_tracer().start_span("queue.wait")
        with self._lock:
            self._pending.append(_Pending(root, wait, time.perf_counter()))
        self.metrics.queue_depth.inc()

    def on_batch_start(self, batch_size: int, oldest_wait_ms: float) -> None:
        """A forward pass is starting over the ``batch_size`` oldest items.

        Closes each item's ``queue.wait`` span, records its wait, and opens the
        one shared ``batch.forward`` span that links to all of them.
        """
        with self._lock:
            take = min(batch_size, len(self._pending))
            taken = [self._pending.popleft() for _ in range(take)]

        links: list[Link] = []
        now = time.perf_counter()
        for item in taken:
            self.metrics.queue_wait_seconds.observe(max(0.0, now - item.enqueued_at))
            item.wait_span.end()
            ctx = item.root_span.get_span_context()
            if ctx.is_valid:
                links.append(Link(ctx, {"link.kind": "batched_request"}))

        # Parentless on purpose: the forward pass belongs to no single request,
        # so it gets its own trace and the requests reach it through links.
        span = self._get_tracer().start_span(
            "batch.forward",
            context=Context(),
            links=links,
            attributes={
                "batch.size": batch_size,
                "batch.linked_requests": len(links),
                "batch.oldest_wait_ms": oldest_wait_ms,
            },
        )
        forward_ctx = span.get_span_context()
        for item in taken:
            _add_link(item.root_span, forward_ctx, "batch.forward")

        self.metrics.batch_size.observe(batch_size)
        if taken:
            self.metrics.queue_depth.dec(len(taken))
        with self._lock:
            self._forward_span = span

    def on_batch_end(self, batch_size: int, forward_ms: float) -> None:
        """The forward pass finished, or raised. Closes the shared span."""
        self.metrics.forward_seconds.observe(forward_ms / 1000.0)
        with self._lock:
            span, self._forward_span = self._forward_span, None
        if span is not None:
            span.set_attribute("batch.forward_ms", forward_ms)
            span.set_attribute("batch.size", batch_size)
            span.end()

    def on_complete(self, status: str) -> None:
        """One item reached a terminal state: ok, timeout, cancelled, queue_full.

        ``timeout`` and ``cancelled`` items are dropped by the batcher *before*
        the forward pass, so ``on_batch_start`` reports a batch size that
        excludes them. Without releasing them here the FIFO would keep entries
        that no batch will ever claim, the queue-depth gauge would drift upward
        forever, and every later batch would link to the wrong request spans.
        A dropped item is released from the head of the FIFO: exact for
        timeouts, which expire in enqueue order, and an approximation for a
        client that disconnects from the middle of the queue -- there the count
        stays right and one link may name the wrong sibling. ``queue_full``
        items were never enqueued, so there is nothing to release.
        """
        self.metrics.requests_total.labels(status=status).inc()
        reason = REJECTION_REASONS.get(status)
        if reason is not None:
            self.metrics.rejected_total.labels(reason=reason).inc()
        if status in ("timeout", "cancelled"):
            self._release_oldest(status)

    def _release_oldest(self, status: str) -> None:
        with self._lock:
            item = self._pending.popleft() if self._pending else None
        if item is None:
            return
        item.wait_span.set_attribute("queue.released_as", status)
        item.wait_span.end()
        self.metrics.queue_depth.dec()

    def drain_abandoned(self) -> int:
        """Close the spans of items that will never ride a batch, at shutdown.

        Returns how many were closed. Without it a drain leaves ``queue.wait``
        spans open forever and the queue-depth gauge stuck above zero.
        """
        with self._lock:
            leftover = list(self._pending)
            self._pending = deque()
        for item in leftover:
            item.wait_span.set_attribute("queue.abandoned", True)
            item.wait_span.end()
        if leftover:
            self.metrics.queue_depth.dec(len(leftover))
        return len(leftover)

    # -- introspection used by the tests and by the dry run ----------------
    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)


def _add_link(span: Span, ctx: SpanContext, kind: str) -> None:
    """Attach a link to an already-started span where the SDK allows it.

    ``Span.add_link`` landed in the OpenTelemetry Python SDK in 1.28. On older
    SDKs the back-link is recorded as attributes instead, so the batch is still
    findable from the request, just not clickable.

    A span that has already ended silently refuses both, which is correct: it
    happens when the client went away before its item was batched, and there is
    nothing left to annotate. The forward span still links to it, so the
    relationship survives from the batch side.
    """
    if not ctx.is_valid or not span.is_recording():
        return
    add_link = getattr(span, "add_link", None)
    if callable(add_link):
        add_link(ctx, {"link.kind": kind})
        return
    span.set_attribute("batch.trace_id", format(ctx.trace_id, "032x"))  # pragma: no cover
    span.set_attribute("batch.span_id", format(ctx.span_id, "016x"))  # pragma: no cover


def documented_buckets() -> dict[str, Sequence[float]]:
    """The bucket sets, so a test can assert the docs and the code agree."""
    return {
        "rs_request_latency_seconds": LATENCY_BUCKETS,
        "rs_forward_seconds": FORWARD_BUCKETS,
        "rs_queue_wait_seconds": QUEUE_WAIT_BUCKETS,
        "rs_batch_size": BATCH_SIZE_BUCKETS,
    }


# ---------------------------------------------------------------------------
# Wiring into the real server
# ---------------------------------------------------------------------------
def as_batcher_hooks(hooks: ObsHooks) -> Any:
    """Adapt ``ObsHooks`` to the batcher's own ``BatcherHooks`` dataclass.

    Imported lazily: obs.py must stay importable by tests, scripts, and tooling
    that never touch the batcher, and the batcher must never import obs.
    """
    from reward_serve.batcher import BatcherHooks

    return BatcherHooks(
        on_enqueue=hooks.on_enqueue,
        on_batch_start=hooks.on_batch_start,
        on_batch_end=hooks.on_batch_end,
        on_complete=hooks.on_complete,
    )


def instrument_app(
    app: Any,
    hooks: ObsHooks,
    metrics: Metrics,
    *,
    traced_paths: tuple[str, ...] = ("/v1/score",),
    metrics_path: str = "/metrics",
) -> None:
    """Mount ``/metrics`` and open a ``score.request`` root span per request.

    The middleware, rather than an edit to ``server.py``: the server owns the
    contract, this module owns the telemetry, and neither has to know the
    other's internals. Only the scoring paths get a root span -- tracing
    ``/healthz`` and ``/readyz`` would bury every real request under probe
    traffic, and the latency histogram would then measure the probes.
    """
    mount_metrics(app, metrics, metrics_path)

    async def _trace_requests(request: Any, call_next: Any) -> Any:
        if not request.url.path.startswith(traced_paths):
            return await call_next(request)
        with hooks.request_span("score.request", **{"http.route": request.url.path}) as span:
            response = await call_next(request)
            span.set_attribute("http.status_code", response.status_code)
            return response

    app.middleware("http")(_trace_requests)
