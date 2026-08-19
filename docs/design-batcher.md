# The batcher

`reward_serve/batcher.py` holds a bounded `asyncio.Queue` and one loop that
drains it into batched forward passes. One loop, one queue, one process.

## Flush policy

The loop waits for a first item, then keeps pulling until whichever of these
comes first:

- **Size trigger.** `max_batch_size` items are in hand.
- **Wait trigger.** The *oldest* item in the batch has waited `max_wait_ms`.

The deadline for the wait trigger is pinned to the first item's enqueue time,
not refreshed as later items arrive. A steady trickle of arrivals therefore
cannot keep a batch open indefinitely, which is the starvation bug this
arrangement exists to avoid. The tail latency a request can pick up from
batching is bounded by `max_wait_ms` plus one forward pass.

The forward pass runs in a thread executor, so the event loop keeps accepting
and enqueueing work while the model is busy. The loop itself awaits the
forward pass before assembling the next batch, so exactly one forward pass is
in flight at a time.

`tests/test_batcher.py::test_flush_on_size_fires_before_the_wait_elapses` sets
`max_wait_ms` to 10 seconds so only the size trigger can fire, and
`test_flush_on_wait_fires_for_a_lone_item` sets `max_batch_size` to 64 so only
the wait trigger can fire. Each proves one trigger in isolation.

## Deadlines

Every accepted item carries a deadline, from `timeout_ms` on the request or
from `REQUEST_TIMEOUT_MS`. The deadline is checked once more at the moment a
batch is assembled, immediately before the forward pass. An item past its
deadline is completed with `DeadlineExceeded` and is **removed from the
batch**, so the model never sees it. A request that has already given up costs
no compute.

The deadline is not enforced *during* a forward pass. Once an item is in a
batch it is scored, however long that batch takes. Cutting a forward pass in
half would waste the work done for every other item riding in it.

## Cancellation

`submit()` awaits a future. If the caller's task is cancelled, which is what a
client disconnect looks like by the time it reaches this layer, the handler
catches `CancelledError`, marks the item cancelled, and re-raises. The batch
assembly step skips cancelled items exactly as it skips expired ones.

The gap this leaves is honest and small: an item cancelled *after* batch
assembly has begun is still scored, and its result is discarded.
`test_cancelled_item_is_never_forwarded` covers the case that matters, an item
cancelled while it sits in the queue.

## Backpressure

`put_nowait` on a full queue raises at once and the caller gets
`QueueFullError`, which the server maps to HTTP 429. Enqueue never blocks
waiting for room. A server that blocks on a full queue converts a queueing
problem into a timeout problem and hides the saturation from the client; a
fast 429 lets the load harness see the knee where it actually is.

## Drain

`drain()` stops accepting, waits for the queue to empty and the in-flight
batch to finish, then stops the loop. Work already accepted is completed.
`stop()` fails anything still queued rather than dropping it silently, so no
caller waits forever on a future nobody will ever complete.

## Hooks

`on_enqueue`, `on_batch_start`, `on_batch_end`, `on_complete`. Signatures are
frozen in `docs/contract.md`. They default to no-ops and are called from the
event loop, so an implementation that blocks will stall batching.

## What this batcher does not do

There is no priority: the queue is strictly first-in-first-out and a small
urgent request queues behind whatever arrived before it. There is no
speculative or continuous token-level batching, because the model is a
single-pass encoder with one forward per item, not an autoregressive decoder;
"continuous" here means the loop keeps forming batches from whatever is
waiting, not that items join a batch mid-generation. There is no adaptive
tuning of `max_batch_size` or `max_wait_ms` from observed load. There is no
padding-aware packing or length-based bucketing, so one long sequence pads the
whole batch up to its length. It runs as one worker process on one machine
with an in-process queue, so there is no sharing of the queue across
processes, no replication, and no failover. Multiple workers would each need
their own queue and would not coordinate.
