# Multi-tier serving and fault injection

One router process in front of two worker processes, on one machine. This
document explains why the router makes the four decisions it makes, and then
reports what four deliberate failures actually did to the client.

Scope, said once so it is not implied later: there is no second host, no
service discovery, no sharding. Two worker processes and one router process
share a CPU with other work. Everything below is about reasoning across
processes and service tiers on one box.

Every number in this document comes from a committed `summary.json` under
`results/faults/`, produced by `faults/run_faults.py`. The command that
produced all of them is:

    set CUDA_VISIBLE_DEVICES=
    .venv/Scripts/python faults/run_faults.py

## The tiers

    client -> router (:8100) -> worker A (:8001)  model + batcher
                            \-> worker B (:8002)  model + batcher

A worker is `reward_serve/server.py` — the same `/v1/score`, `/healthz`,
`/readyz`, `/admin/drain` as before — plus a name it stamps on every response
in `x-rs-worker`, plus trace continuation, plus a fault surface that only
exists when `RS_FAULTS_ENABLED=1`. The router adds no scoring logic at all: it
chooses a worker, bounds the wait, and gets out of the way.

Bring it up with:

    .venv/Scripts/python scripts/tier_up.py --workers 2
    .venv/Scripts/python scripts/tier_up.py --down

`--down` drains the workers first, then stops them, then stops the router.
That order loses nothing: stopping the router first would strand requests
already forwarded to a worker with nowhere to return them.

## Health: the state machine

The router probes each worker's `/readyz` every 200 ms with a 500 ms timeout.
Probe results — nothing else — drive this machine:

                          probe fails
              ┌──────────────────────────────┐
              │                              v
          ┌───┴────┐                   ┌──────────┐
          │   UP   │                   │ SUSPECT  │
          └───┬────┘                   └────┬─────┘
              ^                             │  fail_threshold (3) consecutive
              │      one good probe         │  probe failures
              ├─────────────────────────────┤
              │                             v
              │   success_threshold   ┌──────────┐
              └───(2) good probes ────┤   DOWN   │
                                      └──────────┘

    start state: DOWN

Four choices in that picture are deliberate:

**Start at DOWN, not UP.** Nothing has been observed yet. Starting optimistic
means the first request after start-up is aimed at a process that may still be
loading a checkpoint. `scripts/tier_up.py` therefore waits for each worker's
`/readyz` before starting the router at all.

**Only UP is eligible for traffic.** SUSPECT and DOWN both mean "do not send".

**The two thresholds buy different things.** `fail_threshold` (3 probes at
200 ms = 600 ms of consistent silence) decides how much evidence is needed
before a worker is declared properly DOWN. `success_threshold` decides how much
it costs to come back. SUSPECT recovers on one good probe — cheap forgiveness
for a blip. DOWN needs two in a row — earned trust after a real outage.

**Active probing, not passive inference.** Passive health reacts faster but
goes blind exactly when a worker is idle *because* the router stopped sending
to it, so recovery never gets tested. A cheap GET every 200 ms keeps working on
a worker receiving no traffic.

The gauge `rs_router_upstream_state{worker}` encodes this as 0 DOWN, 1 SUSPECT,
2 UP, and `/routerz` returns the whole table as JSON.

## Routing: least outstanding requests

Among eligible workers, the router picks the one with the fewest requests
currently in flight; ties break by name so a quiet tier is deterministic.

Round-robin counts requests *sent*. Least-outstanding counts requests *not yet
finished*, which means a worker that has become slow accumulates in-flight
requests and stops being chosen without anyone having to declare it unhealthy.
That is the entire defence against a worker that is slow but answering — the
case no health probe can see, and the case the latency scenario below measures.

The rule has a cost, and the queue-exhaustion run below shows it: least
outstanding also prefers whichever worker answers *fastest*, and a rejection is
the fastest answer there is.

## Timeout: the router owns the client's latency bound

The router gives up on an attempt after `request_timeout_ms` (3000 ms by
default) and returns 504. The worker has its own `REQUEST_TIMEOUT_MS` of
5000 ms; these are not redundant.

The worker's deadline protects the worker's own queue and cannot fire at all if
the worker is wedged — the hang scenario is exactly that case. The router's
deadline is enforced in a different process, so it survives anything the worker
does. It is set *below* the worker's so the router always gives up first and
there is one number the client's latency is bounded by.

On expiry the router does not merely answer 504: `asyncio.wait_for` cancels the
attempt, which unwinds the `async with client.stream(...)` block and closes the
connection. The worker learns the client is gone rather than computing a result
nobody will read. `tests/test_router.py::test_router_timeout_returns_504_and_cancels_the_upstream`
proves this with a raw-socket worker that counts peer disconnects.

Connect gets its own much shorter budget, 250 ms, separate from the request
budget. That number is not a guess; see the kill-worker scenario.

## Retry: exactly one, and only before any bytes were sent

The router retries a request **only** when the connection failed before any
request byte reached a worker (`httpx.ConnectError`, `ConnectTimeout`,
`PoolTimeout`), **only once**, and **only onto a different eligible worker**.

Scoring is idempotent — the model has no state and re-running a pair yields the
same number — so a naive reading says retry everything. That is still wrong,
for reasons that have nothing to do with correctness:

* A request that came back 500 has already cost a forward pass, or is about to.
  Retrying it doubles load on a tier that just said it is in trouble, which is
  how a partial outage becomes a total one.
* A request that timed out may still be running on the worker. Retrying it puts
  two copies of the same work in a queue whose first copy nobody is waiting on
  any more.
* Retrying onto the *same* worker that refused the socket is just a slower
  failure.

A pre-send connection error is different in kind: the worker provably never saw
the request, so the retry adds no work anywhere and cannot double-count.
`tests/test_router.py::test_a_500_that_reached_a_worker_is_never_retried`
asserts the negative case, which is the one that matters.

## Circuit breaker

Per worker, driven by request outcomes rather than probes: five consecutive
failures open it for five seconds, then it half-opens and admits exactly one
probe request. That probe closing it returns the worker to service; that probe
failing reopens it immediately without spending the failure budget again.

A failure is a 5xx, a transport error, or a router timeout. **A 429 is not a
failure.** A worker answering 429 is healthy and refusing work it cannot queue.
Counting that as a failure would eject the worker that is behaving correctly
under overload and push the entire offered load onto the other one — turning a
shed-load event into an outage. The queue-exhaustion scenario exists to show
that decision holding under 1297 rejections.

The breaker matters most where health probing is useless: `/readyz` answers
fine while a worker returns 500s or hangs, because readiness is not the scoring
path. Health probing catches a *dead* worker; the breaker catches a *lying* one.

---

# What four deliberate failures did

Method: `faults/run_faults.py` brings up a router and two workers loading the
real checkpoint, drives steady closed-loop load through the router with the
existing generator at concurrency 4 (16 for queue exhaustion), injects one
fault at a known moment, and records every request as the *client* saw it —
status, latency, which worker served it, whether it was retried — alongside
`/routerz` samples at 10 Hz. This is not a benchmark and it does not sweep.

**The box was not quiet for any of these runs.** `scripts/box_quiet.py`
measured mean CPU between 40.9% and 50.6% before each one, and every
`summary.json` therefore carries `"latency_figures_are_indicative_only": true`
next to its `box_quiet.json`. Latency figures below are labelled indicative
wherever they appear and no SLO is derived from them. Counts, statuses, state
transitions and elapsed times are unaffected by contention in kind, though a
busy box makes every duration longer; that caveat is repeated in the threats to
validity.

## 1. kill-worker-mid-batch

`results/faults/kill-worker-mid-batch/summary.json`

**Injected:** worker B terminated (Windows `TerminateProcess`, the local
equivalent of a supervisor's SIGTERM — no chance to drain) at t = 10.003 s,
with requests in flight.

**What the client saw.** Of 249 measured requests, exactly **2 failed** — both
502, both of them requests that were already on the wire to B when the process
died. Four requests were in flight at that instant; the two aimed at A finished
normally. **Zero** of the 187 requests sent after the kill failed.

Four requests hit a refused connection on B and were **retried onto A**, all
four returning 200, at 464.7, 478.1, 508.4 and 515.7 ms (indicative).
`rs_router_retries_total` reads 4. No request that reached a worker was
retried, and none could be: the retry path is only reachable from a pre-send
connection error.

Traffic converged on A **93 ms** after the kill — the first request sent at
t = 10.096 s and every request after it was served by A. That is far faster
than the health machine, and it is the retry doing the work, not the state
machine.

**What the state table did** (observed at 100 ms sampling):

| observed at | Δ from kill | state | breaker |
| --- | --- | --- | --- |
| 10.644 s | +0.64 s | UP | OPEN |
| 10.746 s | +0.74 s | SUSPECT | OPEN |
| 12.182 s | +2.18 s | DOWN | OPEN |

The breaker beat the health machine to it, which is what it is for: five failed
attempts arrive faster than three failed probes when the load is real.

**The defect this run found.** The first version of this scenario left the
retry working but expensive: a connect to the port of a terminated worker took
about 2.3 s to be refused on this box, so the two retried requests cost 2.54 s
and 2.79 s against a p50 of 0.41 s. The evidence is kept at
`results/faults/kill-worker-mid-batch-before-connect-timeout/summary.json`, and
`rs_router_upstream_latency_seconds{worker="B"}` in its `metrics.txt` shows the
two failed attempts landing in the 2–3 s bucket. The fix was to give connect
its own 250 ms budget, separate from the 3 s request budget. After it, retried
requests cost 464–516 ms. A retry is only worth having if failing is cheap.

**Indicative latency:** p50 344.6 ms, p99 706.7 ms over 247 successes.

**First thing a responder should open:** `rs_router_upstream_state{worker="B"}`
— it is the one series that separates "the worker is gone" from "the worker is
slow", and the two have completely different next steps.

## 2. latency-2s-on-one-worker

`results/faults/latency-2s-on-one-worker/summary.json`

**Injected:** 2000 ms of delay in front of worker B's forward pass at
t = 8.0 s, cleared at t = 26.07 s.

**What the client saw** (all latency indicative):

| phase | n | p50 | p99 | statuses | share served by A |
| --- | --- | --- | --- | --- | --- |
| before | 17 | 748.9 ms | 1360.4 ms | all 200 | 0.647 |
| during | 134 | 351.3 ms | 2346.9 ms | 128 × 200, **6 × 504** | 0.940 |
| after | 116 | 458.1 ms | 971.3 ms | all 200 | 0.500 |

Load moved off B on its own. The per-2-second share of requests served by A
went 0.57 → 0.83 → 0.91 → 0.93 → **1.00** across the injection, and came back
to 0.47 within two seconds of the fault clearing. Nothing declared B unhealthy
to make that happen: least-outstanding did it, because a 2 s forward pass keeps
B's in-flight count high and B therefore stops winning the comparison.

**`B_stayed_UP_throughout` is true.** Every single `/routerz` sample had B in
state UP. This is the headline: a worker that is slow but answering is
completely invisible to health checking, because `/readyz` is not the scoring
path. Anyone relying on readiness to shed load off a degraded worker has no
defence here at all.

The router timeout did fire, six times, all during the injection. Six clients
paid 3 s and got a 504 rather than waiting behind a 2 s forward pass at the back
of a queue.

**What the state table did:** B's health never moved. Its breaker opened at
t = 17.40 s after five consecutive router timeouts, half-opened at t = 24.86 s,
and closed at t = 27.22 s once the fault was gone and its probe succeeded.

**The defect this run found.** In the first version, B's breaker opened at
t = 17.3 s and never closed — `share_A` stayed at 1.00 for the entire remainder
of the run even after the fault was cleared. `pick()` filtered on the raw
breaker state instead of calling the check that moves OPEN to HALF_OPEN, so
nothing ever advanced it and a tripped worker stayed ejected for the life of
the process. The unit tests passed, because they called the breaker directly.
Evidence:
`results/faults/latency-2s-on-one-worker-before-breaker-recovery/summary.json`.
The regression test now drives recovery through `pick`, not through the
breaker: `tests/test_router.py::test_an_open_breaker_recovers_and_pick_is_what_recovers_it`.

**First thing a responder should open:** the `batch.forward` span on worker B —
its `batch.forward_ms` attribute separates "the model got slow" from "the queue
got deep", and the router-side metrics cannot tell those apart.

## 3. queue-exhaustion

`results/faults/queue-exhaustion/summary.json`

**Injected:** nothing. `QUEUE_DEPTH=4`, `MAX_BATCH_SIZE=2`, sixteen closed-loop
clients for 16 s. The fault is the offered load.

**What the client saw.** Of 1363 measured requests, **1297 were 429**
(95.16%), 42 were 200 and 24 were 504.

**What the state table did: nothing.** Both workers were UP in every one of the
133 samples, and **both breakers were CLOSED in every sample**. 1297 rejections
opened no breaker and moved no state. That is the decision holding: a 429 never
increments a worker's failure counter, because a worker refusing work it cannot
queue is a healthy worker doing its job. Ejecting it would have moved the whole
offered load onto its neighbour and converted deliberate shedding into an
outage. Asserted independently by
`tests/test_router.py::test_a_429_is_returned_untouched_and_is_not_a_failure`.

**What this run also showed, and it is uncomfortable.** The busiest worker
served 66.8% of requests. Least-outstanding prefers whichever worker answers
fastest, and a 429 is the fastest answer there is: a rejecting worker drops to
zero outstanding instantly and is immediately picked again. Under shed load the
routing rule therefore concentrates traffic on the worker doing the *least*
useful work. This is named, not fixed — see "Not done".

**The first attempt at this scenario refused nothing at all.** At the default
`MAX_BATCH_SIZE=8`, sixteen closed-loop clients across two workers produced
zero 429s: the eight clients aimed at one worker all fit in a single batch, so
nothing was ever left waiting and the queue of four was never the binding
constraint. That run is kept as
`results/faults/queue-exhaustion-batch8-never-sheds/summary.json` — 284
requests, all 200. Shrinking the batch to 2 leaves six of the eight waiting on
a queue of four, which is what makes the queue bind. A closed-loop client
cannot overrun a queue it cannot get ahead of.

**Indicative latency:** p50 2299.4 ms over the 42 successes, which is a number
about a saturated tier and nothing else.

**First thing a responder should open:** `rs_rejected_total{reason="queue_full"}`
on the workers, next to `rs_router_requests_total{status="429"}` on the router.
If the router's count is much higher than the workers' sum, the router is
aiming badly rather than the workers being full.

## 4. hang-worker

`results/faults/hang-worker/summary.json`

**Injected:** worker B stops consuming its queue at t = 8.0 s. Its HTTP server
keeps answering; its batching loop never returns from a forward pass.

**What the client saw.** 126 requests were sent after the injection. Six of
them got a 504. **No request waited past the router's budget:** the longest
client-observed latency in the entire run was 3023.9 ms against a 3000 ms
router timeout, and zero requests exceeded 1.25× that budget. Requests queued
on B did not sit there — the router hung up on them at 3 s, in a different
process from the one that was stuck.

**What the state table did — and this is the finding.** B was **never marked
DOWN**. `time_to_DOWN_s` is `null`. `/readyz` on a hung worker returns 200,
because readiness checks that the model is loaded and the batching loop object
is alive, not that the loop is making progress. The batching loop is alive; it
is blocked inside a forward pass. A liveness probe that does not test progress
cannot see a wedged worker.

What actually removed B from rotation was the circuit breaker, at t = 16.910 s
— **8.91 s after the hang began**. It is that slow precisely because
least-outstanding was already routing around B: a hung worker's in-flight count
only rises, so B was rarely picked, so its five consecutive failures accumulated
slowly. Only 6 of the 126 post-injection requests reached B at all; the other
120 went to A. The two mechanisms partly cancel: the routing rule protects
almost every request, and by doing so it starves the breaker of the evidence it
needs to eject the worker properly. B's breaker then half-opened at t = 22.38 s,
spent its probe on the still-hung worker, and reopened at t = 25.40 s — the
half-open cycle behaving exactly as designed against a fault that had not gone
away.

**Indicative latency:** p50 510.5 ms, p99 1003.9 ms over 156 successes.

**First thing a responder should open:** `rs_queue_depth` on worker B. It is
the only series that distinguishes a hang from an idle worker — a wedged worker
shows a queue depth that rises and never falls, while `/readyz` and the worker's
own request-rate look like a worker with nothing to do.

---

## What the four runs say together

Three different faults produced three different detectors, and only one of them
was the health check:

| fault | health probe saw it | breaker saw it | routing rule handled it |
| --- | --- | --- | --- |
| worker killed | yes, DOWN at +2.18 s | yes, at +0.64 s | retry, converged at +0.093 s |
| worker slow | **no** — UP throughout | yes, at +9.4 s | yes, share 0.57 → 1.00 |
| queue full | no, and correctly not | no, and correctly not | shed, 95.2% 429 |
| worker hung | **no** — never DOWN | yes, at +8.91 s | yes, 120 of 126 to A |

The health machine is the *slowest* and *narrowest* of the three mechanisms. It
only catches the failure where the process is gone. Everything else was caught
by the timeout, the breaker, or the routing rule — all three of which live at
the router, and none of which need the worker to cooperate.

## Threats to validity

* **The box was not quiet for any run.** Mean CPU was 40.9%–50.6% during the
  measurement windows. Every latency figure above is labelled indicative and is
  quoted only as an order of magnitude. The behavioural facts — counts,
  statuses, which worker served what, state transitions — do not depend on the
  box being quiet, but the *durations* between them do: on an idle box every
  elapsed time above would likely be shorter. Treat "+8.91 s to breaker OPEN"
  as an upper bound of that shape, not a constant.
* **State transitions are observations at 100 ms sampling**, taken from
  `/routerz` polls, not from the router's internal transition log. Any timestamp
  in the tables above is late by up to one sample interval.
* **One run per scenario.** There is no repetition and no confidence interval.
  A count of 2 failed requests at the kill is a fact about that run; the
  underlying quantity is "the requests that happened to be in flight", which
  depends on concurrency and timing.
* **Closed-loop load hides overload.** Four clients that wait for a response
  cannot offer more load when the server slows down, so a degraded tier looks
  like reduced throughput rather than growing queues. That is why the
  queue-exhaustion scenario had to shrink the batch instead of raising the rate,
  and it is why none of these runs says anything about a latency knee.
* **The `x-rs-worker` attribution is the router's claim**, taken from the
  response header. For a retried request it names the second worker; the first
  attempt is only visible in `rs_router_retries_total` and the router's own
  upstream latency histogram.
* **Both workers load the same checkpoint on the same CPU.** They are not
  isolated from each other; a slow forward pass on one worker slows the other by
  competing for the same cores. Two processes on one machine share more than two
  machines would.

## Not done

* The routing rule is not corrected for fast-failing workers. Least-outstanding
  concentrates traffic on a worker that is refusing requests, measured at 66.8%
  in the queue-exhaustion run. The usual answer is to weight by recent latency
  or to penalise a worker's recent failures; neither is implemented.
* `/readyz` still cannot see a wedged batching loop. Making readiness depend on
  queue progress would have caught the hang scenario in under a second, but it
  would change the contract in `docs/contract.md`, which this work did not do.
* There is no request hedging, no priority, and no queue-depth signal from the
  worker back to the router. The router picks on its own view of in-flight
  count, not on anything the worker tells it.
* Worker count is static configuration. Nothing adds or removes a worker at
  runtime, and there is no supervisor that restarts a killed one — the
  kill-worker scenario ends with one worker, permanently.
* The fault surface covers latency, error rate and hang. It cannot corrupt a
  response, partially write one, or make a worker slow only for some inputs.
* No scenario was run with more than two workers, and none was run twice.
