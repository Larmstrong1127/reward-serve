# reward-serve interface contract

This document is the interface the server core exposes to the load harness
(`loadgen/`) and to observability (`reward_serve/obs.py`, `docker/`). It is
written before the implementation and the implementation is held to it. If a
field name here disagrees with the code, the code is wrong.

Scope: one server process on one machine. There is no gateway, no sharding,
no second node.

## 1. Endpoints

### `POST /v1/score`

Accepts exactly one of two request forms.

Form A — one pair:

```json
{"prompt": "How do I sort a list?", "response": "Use sorted().", "timeout_ms": 5000}
```

Form B — many pairs:

```json
{"pairs": [{"prompt": "...", "response": "..."},
           {"prompt": "...", "response": "..."}],
 "timeout_ms": 5000}
```

`timeout_ms` is optional. When absent the server uses `REQUEST_TIMEOUT_MS`.
Supplying both form A and form B keys, or neither, is a `422`.

Every pair is enqueued as its own queue item. Pairs from one Form B request
may therefore ride in different batches and report different batch metadata.
This is deliberate: it keeps the batcher free to pack whatever is waiting.

Response `200`:

```json
{
  "results": [
    {"score": 0.8312, "batch_size": 4, "queue_wait_ms": 7.21, "forward_ms": 41.93}
  ],
  "model": "DantheMan124/deberta-preference-reward",
  "max_length": 512,
  "temperature": 1.1668
}
```

`results` is always a list and is always in the same order as the submitted
pairs, including for Form A (length 1).

Per-result batch metadata fields:

| field | type | meaning |
|---|---|---|
| `score` | float | raw logit divided by `config.reward_temperature`; ranking-only semantics |
| `batch_size` | int | number of items in the forward pass this item rode in |
| `queue_wait_ms` | float | enqueue until the forward pass started |
| `forward_ms` | float | wall time of the forward pass this item rode in |

Top-level fields `model`, `max_length`, `temperature` are read from the
checkpoint config at load time. `max_length` is never hardcoded.

### `GET /healthz`

`200 {"status": "ok"}` whenever the process is answering. It does not check
the model or the batcher.

### `GET /readyz`

`200 {"status": "ready", "model_loaded": true, "batcher_running": true}` only
when the model is loaded and the batching loop is running and the server is
not draining. Otherwise `503` with the same keys and `"status"` one of
`"loading"` or `"draining"`.

### `POST /admin/drain`

`200 {"status": "draining"}`. Starts a graceful drain: `/readyz` flips to
`503` at once, requests already in flight run to completion, and new
`/v1/score` requests get `503 {"error": "draining"}`. Drain is one-way for
the life of the process.

## 2. Error responses

| HTTP | body | when |
|---|---|---|
| 429 | `{"error": "queue_full", "detail": "..."}` | queue is at `QUEUE_DEPTH`; the server does not block waiting for room |
| 504 | `{"error": "deadline_exceeded", "detail": "..."}` | the item's deadline passed before a forward pass started; the item is never scored |
| 503 | `{"error": "draining", "detail": "..."}` | drain has begun |
| 503 | `{"error": "not_ready", "detail": "..."}` | model still loading |
| 422 | FastAPI validation body | malformed request |

A Form B request fails as a whole on 429, 504, and 503. Partial results are
not returned.

## 3. Batcher hooks

Callables passed to `ContinuousBatcher(...)`. Each defaults to a no-op.
They are called from the batcher's event loop, so they must not block; the
observability agent owns what goes inside them.

```python
on_enqueue() -> None
on_batch_start(batch_size: int, oldest_wait_ms: float) -> None
on_batch_end(batch_size: int, forward_ms: float) -> None
on_complete(status: str) -> None
```

`status` is one of `"ok"`, `"timeout"`, `"cancelled"`, `"queue_full"`.

Ordering guarantee per item: `on_enqueue` fires before that item can appear
in a batch, and `on_complete` fires exactly once per accepted item.
`on_batch_start` and `on_batch_end` fire once per forward pass, and
`on_batch_end` fires even when the forward pass raises.

## 4. Environment variables

| name | default | meaning |
|---|---|---|
| `MAX_BATCH_SIZE` | `8` | flush when this many items are waiting |
| `MAX_WAIT_MS` | `10` | flush when the oldest waiting item has waited this long |
| `QUEUE_DEPTH` | `256` | bounded queue size; beyond it, enqueue returns queue-full |
| `REQUEST_TIMEOUT_MS` | `5000` | default per-request deadline |
| `MODEL_ID` | `DantheMan124/deberta-preference-reward` | Hub id to load |
| `HF_HOME` | `D:\ClaudeCache\hf` | Hub cache root |

Every variable has a matching CLI flag on `python -m reward_serve`, in
lower-kebab form: `--max-batch-size`, `--max-wait-ms`, `--queue-depth`,
`--request-timeout-ms`, `--model-id`, plus `--host` and `--port`. A CLI flag
wins over the environment variable.

`CUDA_VISIBLE_DEVICES` is forced to empty by every entrypoint. The GPU on
this machine belongs to other work.

## 5. What the load harness can rely on

- The three timing fields above are measured inside the server, so the
  harness can compare its own client-side latency against them.
- `batch_size > 1` in a response is the only honest evidence that batching
  happened. The harness should record it per request.
- `429` is immediate, not a slow failure, so offered-load sweeps do not stall
  on a saturated queue.
