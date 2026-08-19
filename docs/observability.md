# Observability for reward-serve

One server process on one machine, watched by four containers. This document is
for the person who has just been handed the pager and has one browser tab: what
the traces mean, what each metric means, and which panel to open first.

Everything here is emitted by `reward_serve/obs.py`. The batcher itself knows
nothing about telemetry; it calls four no-op callables (`docs/contract.md`
section 3) and this module supplies the real implementations.

## The stack

    reward-serve --OTLP/gRPC--> otel-collector --OTLP--> jaeger      (traces)
    reward-serve <---scrape---- prometheus                           (metrics)
    grafana ------------------> prometheus, jaeger                   (panels)

`docker/docker-compose.yml` brings all five up. The server exports spans to the
collector rather than straight to Jaeger so that a collector restart, a change
of trace backend, or added sampling does not mean rebuilding the server image.
The collector also derives span metrics on `:8889`, which is how you tell "the
server stopped emitting" from "the collector stopped forwarding" — two very
different 3am problems that look identical on an empty graph.

## Wiring it in

Three lines at the app factory, and neither `server.py` nor `batcher.py` has to
know that telemetry exists:

```python
metrics = obs.setup_metrics()
hooks = obs.ObsHooks(metrics, tracer=obs.setup_tracing("reward-serve"))
app = create_app(settings, hooks=obs.as_batcher_hooks(hooks))
obs.instrument_app(app, hooks, metrics)   # mounts /metrics, opens score.request
```

`as_batcher_hooks` adapts to the batcher's own `BatcherHooks` dataclass, so a
change to those four signatures fails a test in `tests/test_obs_wiring.py`
rather than quietly leaving the hooks as no-ops. `instrument_app` adds an HTTP
middleware that traces only the scoring paths: tracing `/healthz` and `/readyz`
would bury every real request under probe traffic and drop probe timings into
the latency histogram, which is a quick way to build a dashboard that says
everything is fine.

## How to read a trace here

A single scoring request produces this:

    score.request                        root span, one trace per HTTP request
      └─ queue.wait                      enqueue → the forward pass starting
      ⇢ link: batch.forward

and the forward pass produces a separate, deliberately parentless span:

    batch.forward   batch.size=8  batch.linked_requests=8  batch.forward_ms=41.9
      ⇢ links: the score.request span of every item in that batch

The forward span is not a child of any request, because it does not belong to
one. Eight requests rode that pass; making it the child of whichever one
happened to arrive first would be a lie that shows up as a mysteriously slow
request in that trace and an invisible gap in the other seven. Instead the
relationship is expressed with span links in both directions, so:

- from **one request**, the link tells you which batch it joined and, through
  `batch.size`, how many siblings it was packed with;
- from **one batch**, the links enumerate every request that paid for it.

That is the single most useful thing in the whole setup. When a request is slow,
the first fork in the road is *did it wait, or did it compute*, and the trace
answers it directly: compare the `queue.wait` duration against
`batch.forward_ms`. A long wait and a short forward is a batching or admission
problem. A short wait and a long forward is a model, threading, or contention
problem. They have nothing in common except the number the client complained
about.

One caveat worth knowing before you trust a link: the hook signatures carry no
item identity, so `obs.py` matches hook calls to items by the batcher's FIFO
order, which `docs/contract.md` guarantees. If the batcher is ever changed to
pack out of order, the links go wrong while the metrics stay right. A test
(`tests/test_obs.py`) pins the ordering the links depend on.

## What each metric means

| metric | type | what it tells you |
|---|---|---|
| `rs_requests_total{status}` | counter | Items that reached a terminal state. `status` is `ok`, `timeout`, `cancelled`, or `queue_full`. Counted **per scored item**, so one request carrying eight pairs adds eight. |
| `rs_rejected_total{reason}` | counter | The subset that was refused or abandoned, split by `queue_full`, `deadline_exceeded`, `cancelled`. Redundant with the counter above by design: the error budget then reads one series instead of subtracting two. |
| `rs_queue_depth` | gauge | Items enqueued and not yet packed. Instantaneous, not a rate. Near `QUEUE_DEPTH` means saturation; near zero with high latency means the flush timer is the binding constraint, not the queue. |
| `rs_queue_wait_seconds` | histogram | Enqueue until the forward pass carrying that item started. Bounded by `MAX_WAIT_MS` whenever the batcher is keeping up, which is why the buckets are dense below 10 ms. |
| `rs_forward_seconds` | histogram | Wall time of one forward pass, once per pass rather than once per item. This is the floor under every latency number. |
| `rs_batch_size` | histogram | Achieved batch size per pass, bucketed 1,2,4,8,16,32,64. Configured batch size is an intention; this is the outcome. |
| `rs_request_latency_seconds` | histogram | End-to-end server-side latency, including parsing and response assembly. The client-visible number and the one the latency SLO is judged on. |

Bucket edges are chosen for a 20–500 ms service and justified in a comment in
`reward_serve/obs.py`; a test asserts the live collectors carry exactly those
edges, so the docs and the code cannot drift apart quietly.

Two of these deserve a warning. `rs_requests_total` counts items, not HTTP
requests, so it will not match a client-side request count whenever Form B is in
use — that is intentional, because the batcher's unit of work is the item.
And `rs_forward_seconds` is observed once per pass, so a mean taken against
`rs_requests_total` is meaningless; divide by `rs_batch_size_count` instead.

## Which panel to open first

**Error rate and rejections by reason**, top right of the reward-serve overview
dashboard. Not the latency panel, and not the trace list.

The reason is that it is the only panel that distinguishes the two failure modes
this server actually has, and they have opposite fixes. `queue_full` means
offered load exceeded what the queue will hold, and the fix is upstream: shed,
throttle, or raise capacity. `deadline_exceeded` means work was accepted and
then not delivered in time, and the fix is downstream: the forward pass, the
batch policy, or whatever else is eating the box. Latency graphs look similar in
both cases, so opening latency first costs you the first several minutes of the
incident guessing.

Once that panel has told you which of the two it is:

1. **`queue_full` moving** → queue depth panel. If depth is pinned at
   `QUEUE_DEPTH`, it is admission, and the mean-batch-size stat tells you
   whether the server was at least packing efficiently while it drowned.
2. **`deadline_exceeded` moving** → latency panel, then the batch size heatmap.
   If p99 rose while p50 held, requests are missing batches; if both rose and
   the `p95 forward only` series rose with them, the forward pass itself slowed
   and the cause is the box, not the batcher. If the heatmap has collapsed into
   the `le=1` row, batching stopped happening whatever the config says, and
   every pass is paying full cost for one item.
3. **Neither moving but the burn panels are red** → the errors are old. Look at
   the 6h window, not the 5m one, and treat it as a ticket.

Then, and only then, pull one trace: pick a slow request in Jaeger, read
`queue.wait` against `batch.forward_ms`, and follow the link to the batch to see
how many siblings it had. By that point the trace confirms a hypothesis rather
than supplying one, which is the right order.

## What this setup does not tell you

There is no profiling of the forward pass itself, so "the model got slower" is
as deep as the traces go — the next question, which layer or which thread, needs
a profiler and is not wired in. There is no per-tenant or per-client
attribution, because there are no tenants; one process serves whoever connects.
There is no correlation across machines, because there is only one machine. And
nothing here watches the host: CPU contention from other work on this box shows
up only indirectly, as a forward pass that got slower for no visible reason.
`scripts/box_quiet.py` exists precisely because that blind spot is real.

## Not done

- **The stack has never been brought up.** Docker is not installed on this
  machine — `docker` is absent from PATH and there is no Docker Desktop
  installation to start — so `docker compose config` has not been run, no
  container has been built, and no panel in the committed dashboard has ever
  rendered against live data. What has been checked is that every file parses
  and hangs together: the compose file declares the five services with the
  server's healthcheck on `/readyz`, the collector receives OTLP/gRPC and
  exports to Jaeger, Prometheus scrapes `server:8000/metrics` and loads the
  rules, the dashboard JSON is valid and every panel expression passes the
  PromQL checker, and every datasource UID the dashboard uses is one the
  provisioning file creates. Those are checks a test can make on this box;
  "the dashboard works" is not one of them, and it is not claimed.
- No log pipeline. Traces and metrics only; server logs stay on stdout.
- No exemplars linking histogram buckets to trace IDs. They would make the
  latency panel click straight through to a slow trace, and the wiring is
  supported by the collector already, but it is not built.
- No alertmanager. The burn-rate expressions exist as Prometheus rules and as
  dashboard panels; nothing routes them to a person.
