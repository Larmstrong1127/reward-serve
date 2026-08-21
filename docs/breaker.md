# The router's circuit breaker

Scope: one router process in front of N worker processes on one machine. Each
worker has its own breaker; no breaker state is shared with anything outside
this process, and there is no second router.

The breaker is a separate mechanism from the health state machine, and the
split is deliberate. Health comes from active `/readyz` probes and answers "is
this worker answering at all". The breaker comes from real request outcomes and
answers "is sending work to this worker doing more harm than good". A worker
can be UP and have an OPEN breaker: it answers probes fine and fails scoring
requests. Both must be favourable for a worker to be eligible for traffic —
`Upstream.eligible` is `health is UP and breaker_ready()`.

## 1. The transition table

`k` is `breaker_failures` (default 5). `T` is `breaker_open_s` (default 5 s).
"probe" here means the one half-open trial request, not a `/readyz` probe.

| From | Event | To | Slot after | Notes |
|---|---|---|---|---|
| CLOSED | request success (2xx) | CLOSED | free | resets `breaker_consecutive_failures` to 0 |
| CLOSED | 429 | CLOSED | free | **not a failure**; a worker shedding load is behaving correctly |
| CLOSED | request failure (5xx, transport error, router timeout), count < k | CLOSED | free | counter increments |
| CLOSED | request failure, count reaches k | OPEN | free | `breaker_opened_at` set to now |
| OPEN | pick, less than T since it opened | OPEN | free | `breaker_ready()` is false, so the worker is not eligible and no request is sent |
| OPEN | pick, T or more since it opened | HALF_OPEN | **held** | the pick takes the one probe slot |
| HALF_OPEN | pick while the probe is in flight | HALF_OPEN | held | refused; a second request must not ride the probe slot |
| HALF_OPEN | probe success | CLOSED | free | `breaker_opened_at` cleared, counter reset |
| HALF_OPEN | probe 429 | CLOSED | free | 429 is not a failure here either |
| HALF_OPEN | probe failure | OPEN | free | reopens **immediately**, without spending the k-failure budget again |
| HALF_OPEN | probe cancelled (client disconnect) | HALF_OPEN | **free** | no outcome is recorded, but the slot is released |
| any | clock advance | unchanged | unchanged | only matters to OPEN, whose exit is time-based |

Three documented decisions are load-bearing and are preserved exactly:

1. **429 is not a failure.** Counting it would open the breaker on the worker
   that is correctly shedding load, push its whole offered load onto the other
   worker, and turn a shed-load event into an outage.
2. **A failed half-open probe reopens immediately.** It does not get to spend
   the k-failure budget again before backing off.
3. **Retries happen only on a pre-send connection error**, once, onto a
   different worker. The breaker does not change that rule; a retry simply
   asks the router to pick again, which takes a second slot if the second
   worker is half-open.

## 2. The invariants now enforced

`tests/test_router_breaker_model.py` asserts all of these after **every single
event** of every sequence it drives, not at the end of a scenario:

- **(a)** `breaker_probe_inflight` is never true while `breaker` is CLOSED.
  The same is asserted for OPEN: only HALF_OPEN may hold the slot.
- **(b)** No slot survives the event that took it. Every event either does not
  take the slot or ends the attempt that took it, so between events the slot
  is always free — as is `probe_epoch`, its identity. This is the invariant a
  new leak path breaks without anyone writing a test for it.
- **(c)** No permanent ejection. From any reachable state, a bounded run of
  good news — passing probes, succeeding requests, and enough clock for the
  open window to expire — always ends with the worker UP, CLOSED and eligible.
- **(d)** `breaker` is always one of CLOSED, OPEN, HALF_OPEN, and `health` one
  of UP, SUSPECT, DOWN. An OPEN breaker always knows when it opened, or it
  could never half-open again.
- **(e)** `breaker_consecutive_failures` is never negative, and neither is
  `outstanding`.

## 3. How the slot is guaranteed to be released

The slot used to be taken in `pick()` and given back on whichever later path
happened to run — a recorded request outcome, or a hand-written `except`
clause. Any path nobody thought of left the flag set, and a worker whose
breaker can never leave HALF_OPEN is ejected for good. A client disconnecting
mid-probe was that path.

It is now a lease:

- `Upstream.admit()` either refuses, or returns a `ProbeLease` — an object
  whose `__exit__` gives the slot back. A CLOSED breaker gets a lease that
  owns nothing, so the caller never has to know which case it got.
- `Router.pick_lease()` is what `forward()` uses, and it holds the lease in a
  `with` block that spans the attempt. If a later edit ever returns between
  the pick and the attempt, the slot still goes back.
- `Router.attempt()` collects the lease for its pick (`Upstream.take_lease()`,
  which returns a no-op lease rather than `None`, because a caller that has to
  check for `None` is a caller that can forget) and releases it in a `finally`.
  Every exit passes there: a result, a timeout, a transport error, or a
  cancellation from a client hanging up.
- Releases are idempotent and carry the epoch they were issued with, so a
  lease that outlives its attempt cannot clear a slot a later attempt has
  since taken, and the two guards above cannot double-release.
- `Upstream._clear_probe_slot()` is the single place the flag is cleared, and
  the transitions out of HALF_OPEN go through it too.

The evidence that this is enforced rather than merely intended: commenting out
both releases fails four tests, including all three model-based ones, and the
random-sequence failure shrinks to `probe_ok, probe_ok, request_failure ×3,
advance_clock_past_open_window, request_cancelled` — the historical bug,
rediscovered by a test that was never written for it.

## 4. What these tests still cannot catch

They run one router object, in one process, on one event loop, with a fake
clock and with `Router._send` replaced — so what they prove is that the
transition table and the slot protocol are internally consistent under every
event ordering the walk and the random sequences reach. They do not prove the
breaker behaves well against a real worker. Nothing here exercises a real
socket, so a partial write, a half-closed connection, a keep-alive connection
that dies between the pick and the send, or an `httpx` exception type not in
`PRE_SEND_ERRORS` is outside their reach; the socket-level cases live in the
stub-upstream tests and the fault runs instead. Nothing here runs two coroutines
against the same upstream at genuinely overlapping points inside `attempt`, so a
true interleaving race between an admit and a release is only argued for by the
epoch guard, not demonstrated. The fake clock means real time never moves
backwards, jumps, or drags under CPU contention. And because this is one router
process, there is no test at all of two routers agreeing about one worker —
that would need a second process holding its own copy of this state, which this
project does not have and does not claim to have.

## Not done

- No test drives two concurrent attempts that overlap inside `attempt` against
  the same upstream; the epoch guard that would make that safe is argued for,
  not measured.
- The walk is breadth-first to depth 8 over bucketed states. Behaviour that
  needs a longer causal chain than that is only reached by the random tails.
