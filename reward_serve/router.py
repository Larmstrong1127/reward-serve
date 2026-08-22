"""Router tier: one process in front of N worker processes on this machine.

The router owns four decisions, and each one is a place where a plausible
choice would be wrong:

1. **Who is eligible.** An active health loop probes ``/readyz`` on a fixed
   interval and moves each worker through UP -> SUSPECT -> DOWN -> UP. Passive
   health (infer from request failures) reacts faster but goes blind exactly
   when a worker is idle because the router stopped sending to it, so recovery
   never gets tested. Active probing costs one cheap GET per interval and keeps
   working on a worker receiving no traffic.

2. **Where a request goes.** Least-outstanding-requests, not round-robin.
   Round-robin keeps feeding a worker that has stopped answering, because it
   counts requests sent rather than requests finished. Least-outstanding is
   self-correcting: a worker that is slow accumulates outstanding requests and
   stops being chosen, without anyone having to declare it unhealthy. That is
   the whole defence against the ``latency`` fault, which no health probe sees.

3. **When to give up.** A per-request timeout at the router, independent of the
   worker's own ``REQUEST_TIMEOUT_MS``. The worker's deadline protects the
   worker's queue; it cannot fire at all if the worker is wedged. The router's
   deadline is the one the client's latency is actually bounded by, so it is
   the one the SLO is written against. On expiry the router returns 504 *and*
   cancels the upstream attempt, closing the connection, so the worker is not
   left computing a result nobody will read.

4. **When a retry is safe.** Exactly one retry, only for a connection failure
   that happened before any request byte was sent, and only onto a different
   eligible worker. See ``RETRY_RULE`` below for why the obvious wider policy
   is wrong.

Scope: one machine, several processes. There is no second host, no service
discovery, and the worker list is static configuration.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse  # noqa: E402
import asyncio  # noqa: E402
import contextlib  # noqa: E402
import time  # noqa: E402
from collections.abc import AsyncIterator, Awaitable, Callable  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from enum import StrEnum  # noqa: E402
from typing import Any  # noqa: E402

import httpx  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import JSONResponse, Response  # noqa: E402

from reward_serve import obs  # noqa: E402

__all__ = [
    "BreakerState",
    "Health",
    "ProbeLease",
    "Router",
    "RouterSettings",
    "Upstream",
    "create_router_app",
    "main",
]

WORKER_HEADER = "x-rs-worker"
RETRIED_HEADER = "x-rs-retried"
ATTEMPTS_HEADER = "x-rs-attempts"

# Why the retry policy is this narrow, spelled out because it is the part of a
# router most often written wrong:
#
# Scoring IS idempotent here -- the model has no state and re-running a pair
# yields the same number -- so a naive reading says "retry anything". That is
# still wrong, for two reasons that have nothing to do with correctness:
#
#   * A request that reached a worker and came back 500 has already cost a
#     forward pass, or is about to. Retrying it doubles the load on a tier that
#     just told you it is in trouble, which is how a partial outage becomes a
#     total one.
#   * A request that timed out may still be running on the worker. Retrying it
#     puts two copies of the same work in a queue the client is no longer
#     waiting on the first copy of.
#
# A connection error raised before any byte was written is different in kind:
# the worker provably never saw the request, so a retry adds no work anywhere
# and cannot double-count. That is the only case retried here, and it is capped
# at one attempt on a *different* worker -- retrying the same refused socket is
# just a slower failure.
RETRY_RULE = "one retry, connection failures only, before any bytes were sent"

# Exceptions httpx raises when the connection was never established, i.e. no
# request byte reached the worker. Anything else -- ReadTimeout, ReadError,
# RemoteProtocolError -- means bytes may have been sent, so it is not retried.
PRE_SEND_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)


class ProbeLease:
    """The right to hold one upstream's half-open probe slot for one attempt.

    The leak this exists to make impossible: the slot was taken in ``pick`` and
    given back on whichever later path happened to run -- a recorded result, or
    a hand-written ``except`` clause. Any path nobody thought of (a client
    disconnect cancelling the attempt was the one that shipped) left the flag
    set, and a worker whose breaker can never leave HALF_OPEN is ejected for
    good. A lease turns "remember to give it back" into "you are holding an
    object whose ``__exit__`` gives it back", which no new code path can skip.

    Releasing is idempotent and carries the epoch it was issued with, so a
    lease that outlives its attempt cannot clear a slot that a *later* attempt
    has since taken.
    """

    __slots__ = ("_epoch", "_upstream")

    def __init__(self, upstream: Upstream, epoch: int) -> None:
        self._upstream = upstream
        self._epoch = epoch

    @property
    def upstream(self) -> Upstream:
        return self._upstream

    def release(self) -> None:
        self._upstream.release_probe_slot(self._epoch)

    def __enter__(self) -> Upstream:
        return self._upstream

    def __exit__(self, *exc: object) -> None:
        self.release()


class Health(StrEnum):
    """Health as the router believes it, from probe results only."""

    UP = "UP"
    SUSPECT = "SUSPECT"
    DOWN = "DOWN"


class BreakerState(StrEnum):
    """Circuit breaker, driven by real request outcomes, not by probes."""

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


@dataclass
class RouterSettings:
    """Every knob, with the reason its default is what it is."""

    worker_urls: tuple[str, ...] = ()
    # 200 ms: five probes a second per worker is negligible load and puts the
    # detection floor (fail_threshold * interval = 600 ms) well under the
    # request timeout, so a wedged worker is ejected before many clients hit it.
    probe_interval_ms: float = 200.0
    probe_timeout_ms: float = 500.0
    # 250 ms to establish a TCP connection to a process on loopback, or give up.
    # This default was not a guess: the first kill-worker run left the retry
    # working but slow, because a connect to the port of a terminated worker
    # took ~2.3 s to be refused on this box, and the two retried requests cost
    # 2.54 s and 2.79 s against a p50 of 0.41 s. See
    # results/faults/kill-worker-mid-batch-before-connect-timeout/summary.json.
    # A connect budget far below the request budget is what makes a retry
    # cheap enough to be worth having.
    connect_timeout_ms: float = 250.0
    # Three consecutive misses, not one: a single dropped probe on a busy box is
    # noise. Three at 200 ms is 600 ms of consistent silence.
    fail_threshold: int = 3
    # Two consecutive hits to come back from DOWN, so a worker that is flapping
    # does not get traffic between two failures.
    success_threshold: int = 2
    # 3 s: above the worker's own p99 by a wide margin on this box, below the
    # worker's 5 s REQUEST_TIMEOUT_MS so the router gives up first and owns the
    # client's latency bound.
    request_timeout_ms: float = 3000.0
    # Five consecutive request failures on one worker, then stop trying for 5 s.
    breaker_failures: int = 5
    breaker_open_s: float = 5.0

    @classmethod
    def from_env(cls) -> RouterSettings:
        raw = os.environ.get("RS_WORKER_URLS", "")
        urls = tuple(u.strip().rstrip("/") for u in raw.split(",") if u.strip())
        return cls(
            worker_urls=urls,
            probe_interval_ms=float(os.environ.get("RS_PROBE_INTERVAL_MS", "200")),
            probe_timeout_ms=float(os.environ.get("RS_PROBE_TIMEOUT_MS", "500")),
            connect_timeout_ms=float(os.environ.get("RS_CONNECT_TIMEOUT_MS", "250")),
            fail_threshold=int(os.environ.get("RS_FAIL_THRESHOLD", "3")),
            success_threshold=int(os.environ.get("RS_SUCCESS_THRESHOLD", "2")),
            request_timeout_ms=float(os.environ.get("RS_ROUTER_TIMEOUT_MS", "3000")),
            breaker_failures=int(os.environ.get("RS_BREAKER_FAILURES", "5")),
            breaker_open_s=float(os.environ.get("RS_BREAKER_OPEN_S", "5")),
        )


@dataclass
class Upstream:
    """One worker as the router sees it: health, breaker, and load.

    The state machine and the breaker are pure functions of the events fed to
    them and an injectable clock, so ``tests/test_router.py`` drives them with a
    fake probe and a fake clock and never opens a socket.

    States and the moves between them::

                     probe fail                 fail_threshold consecutive
                  +--------------->             probe failures
        +----+    |            +---------+    +----------------> +------+
        | UP |----+            | SUSPECT |----+                  | DOWN |
        +----+                 +---------+                       +------+
           ^                        |                               |
           |     one good probe     |                               |
           +------------------------+                               |
           |                                                        |
           |        success_threshold consecutive good probes       |
           +--------------------------------------------------------+

    A router starts every worker at DOWN, not UP: nothing has been observed
    yet, and optimism here means the first request after start-up is aimed at a
    process that may still be loading a checkpoint.

    Only UP is eligible for traffic. SUSPECT and DOWN both mean "do not send",
    and the difference between them is the price of coming back: SUSPECT
    recovers on a single good probe, DOWN has to pass ``success_threshold`` in
    a row. That is what the two thresholds buy -- fast forgiveness for a blip,
    earned trust after a real outage.
    """

    name: str
    url: str
    fail_threshold: int = 3
    success_threshold: int = 2
    breaker_failures: int = 5
    breaker_open_s: float = 5.0
    clock: Callable[[], float] = time.monotonic

    health: Health = Health.DOWN
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    outstanding: int = 0
    last_probe_ok: bool | None = None
    last_transition_at: float | None = None
    transitions: list[dict[str, Any]] = field(default_factory=list)

    breaker: BreakerState = BreakerState.CLOSED
    breaker_consecutive_failures: int = 0
    breaker_opened_at: float | None = None
    breaker_probe_inflight: bool = False
    # Id of the half-open slot currently held. Zero means "not held". It exists
    # so a release can prove it is giving back its own slot and not one a
    # subsequent attempt has taken.
    probe_epoch: int = 0
    # The id generator behind it, which only ever counts up. It has to be a
    # separate field: clearing the slot sets `probe_epoch` back to 0 -- "not
    # held" -- so counting up from `probe_epoch` on the next admit re-issues an
    # id a lease from a dead attempt may still be holding, and that lease then
    # clears the live slot. Measured, not argued: the case is driven in
    # tests/test_router_epoch_guard.py.
    probe_epoch_seq: int = 0
    # The lease handed out by the last successful `admit`, waiting to be picked
    # up by the attempt it belongs to. `take_lease` is what collects it.
    pending_lease: ProbeLease | None = field(default=None, repr=False)

    # -- health state machine ------------------------------------------------
    def record_probe(self, ok: bool) -> Health:
        """Feed one probe result in. Returns the state after it."""
        self.last_probe_ok = ok
        before = self.health
        if ok:
            self.consecutive_failures = 0
            self.consecutive_successes += 1
            # SUSPECT forgives on one good probe; DOWN has to earn it.
            if self.health is Health.SUSPECT or (
                self.health is Health.DOWN
                and self.consecutive_successes >= self.success_threshold
            ):
                self._set(Health.UP)
        else:
            self.consecutive_successes = 0
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.fail_threshold:
                self._set(Health.DOWN)
            elif self.health is Health.UP:
                self._set(Health.SUSPECT)
        if self.health is not before:
            self.last_transition_at = self.clock()
            self.transitions.append(
                {
                    "from": str(before),
                    "to": str(self.health),
                    "at_monotonic": self.last_transition_at,
                    "at_wall": time.time(),
                    "consecutive_failures": self.consecutive_failures,
                    "consecutive_successes": self.consecutive_successes,
                }
            )
        return self.health

    def _set(self, health: Health) -> None:
        self.health = health

    # -- circuit breaker -----------------------------------------------------
    def breaker_ready(self) -> bool:
        """Would the breaker let one more request through? Changes nothing.

        Split from ``breaker_admit`` because routing has to ASK every candidate
        and TELL only the one it picked. The first version of this class had a
        single mutating check, ``pick`` filtered on the raw ``breaker is OPEN``
        instead of calling it, and nothing ever moved a breaker out of OPEN:
        one worker stayed ejected for the rest of the run. That is visible in
        results/faults/latency-2s-on-one-worker-before-breaker-recovery/summary.json,
        where worker B's breaker opens at t=17.3 s and never closes again even
        after the fault is cleared at t=26 s.
        """
        if self.breaker is BreakerState.CLOSED:
            return True
        if self.breaker is BreakerState.OPEN:
            opened = self.breaker_opened_at
            return opened is not None and self.clock() - opened >= self.breaker_open_s
        return not self.breaker_probe_inflight

    def admit(self) -> ProbeLease | None:
        """Take the slot for one request, or refuse. Mutates; call on the pick.

        OPEN becomes HALF_OPEN once ``breaker_open_s`` has passed, and HALF_OPEN
        admits exactly one request -- the probe. A second concurrent request
        while that probe is in flight is refused, because the point of the
        half-open state is to risk one request, not to reopen the floodgates
        before the answer comes back.

        Returns the lease that owns the slot, or ``None`` when the breaker
        refuses. A CLOSED breaker holds no slot, so its lease releases nothing;
        the caller does not have to know which case it got.
        """
        if self.breaker is BreakerState.CLOSED:
            return self._issue_lease(epoch=0)
        if self.breaker is BreakerState.OPEN:
            if not self.breaker_ready():
                return None
            self.breaker = BreakerState.HALF_OPEN
            self.breaker_probe_inflight = False
        if self.breaker_probe_inflight:
            return None
        self.probe_epoch_seq += 1
        self.probe_epoch = self.probe_epoch_seq
        self.breaker_probe_inflight = True
        return self._issue_lease(epoch=self.probe_epoch)

    def _issue_lease(self, *, epoch: int) -> ProbeLease:
        lease = ProbeLease(self, epoch)
        self.pending_lease = lease
        return lease

    def take_lease(self) -> ProbeLease:
        """Collect the lease ``admit`` left here, or a lease that owns nothing.

        An attempt that was handed an upstream rather than a lease still ends
        up holding the lease for that pick, so its ``finally`` releases the
        right slot. When no slot was taken this is a no-op object, never None,
        because a caller that has to check for None is a caller that can forget.
        """
        lease, self.pending_lease = self.pending_lease, None
        return lease if lease is not None else ProbeLease(self, 0)

    def release_probe_slot(self, epoch: int) -> None:
        """Give back slot ``epoch``. Idempotent, and a no-op for a stale epoch."""
        if epoch <= 0 or epoch != self.probe_epoch:
            return
        self._clear_probe_slot()

    def _clear_probe_slot(self) -> None:
        """The one place the slot is given back. Leaving a state clears it too."""
        self.probe_epoch = 0
        self.breaker_probe_inflight = False

    def breaker_admit(self) -> bool:
        """Boolean form of ``admit`` for driving the state machine directly.

        The lease is left pending on the upstream, so the slot stays held until
        an attempt collects and releases it -- which is exactly what the
        half-open tests assert. Routing calls ``admit`` and holds the lease.
        """
        return self.admit() is not None

    def record_request_result(self, *, failure: bool) -> None:
        """One completed attempt. ``failure`` excludes 429 on purpose.

        A 429 is a healthy worker refusing work it cannot queue. Counting it as
        a failure would open the breaker on the worker that is behaving
        correctly under overload, take it out of rotation, and push the whole
        offered load onto the other one -- turning a shed-load event into an
        outage. The queue-exhaustion scenario in ``docs/multi-tier.md`` exists
        to show this decision holding.
        """
        if failure:
            self.breaker_consecutive_failures += 1
            # A failed half-open probe reopens immediately -- it does not get
            # to spend the failure budget again before we back off.
            reopen = self.breaker is BreakerState.HALF_OPEN
            if reopen or self.breaker_consecutive_failures >= self.breaker_failures:
                self.breaker = BreakerState.OPEN
                self.breaker_opened_at = self.clock()
                self._clear_probe_slot()
        else:
            self.breaker_consecutive_failures = 0
            if self.breaker is BreakerState.HALF_OPEN:
                self.breaker = BreakerState.CLOSED
                self.breaker_opened_at = None
                self._clear_probe_slot()

    @property
    def eligible(self) -> bool:
        """UP, and the breaker would take one more request right now."""
        return self.health is Health.UP and self.breaker_ready()

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "url": self.url,
            "state": str(self.health),
            "eligible": self.eligible,
            "outstanding": self.outstanding,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "last_probe_ok": self.last_probe_ok,
            "breaker": str(self.breaker),
            "breaker_consecutive_failures": self.breaker_consecutive_failures,
            "transitions": len(self.transitions),
        }


@dataclass
class Attempt:
    """The outcome of one router -> worker try."""

    worker: str
    status: int
    body: bytes
    latency_s: float
    pre_send_failure: bool = False
    timed_out: bool = False
    error: str | None = None


class Router:
    """Health, routing, timeout, retry and breaker for a fixed worker list."""

    def __init__(
        self,
        settings: RouterSettings,
        *,
        metrics: obs.RouterMetrics | None = None,
        probe: Callable[[Upstream], Awaitable[bool]] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings
        self.metrics = metrics if metrics is not None else obs.setup_router_metrics()
        self.clock = clock
        self._probe_override = probe
        self.upstreams: list[Upstream] = [
            Upstream(
                name=_worker_name(i, url),
                url=url.rstrip("/"),
                fail_threshold=settings.fail_threshold,
                success_threshold=settings.success_threshold,
                breaker_failures=settings.breaker_failures,
                breaker_open_s=settings.breaker_open_s,
                clock=clock,
            )
            for i, url in enumerate(settings.worker_urls)
        ]
        self._client: httpx.AsyncClient | None = None
        self._health_task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        for up in self.upstreams:
            self._publish_state(up)

    # -- lifecycle -----------------------------------------------------------
    async def start(self) -> None:
        limits = httpx.Limits(max_connections=256, max_keepalive_connections=64)
        self._client = httpx.AsyncClient(limits=limits)
        self._stop.clear()
        self._health_task = asyncio.create_task(self._health_loop(), name="router-health")

    async def stop(self) -> None:
        self._stop.set()
        task, self._health_task = self._health_task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()

    # -- health --------------------------------------------------------------
    async def probe_once(self, up: Upstream) -> bool:
        """One ``/readyz`` probe. Any non-200, or any exception, is a failure."""
        if self._probe_override is not None:
            return await self._probe_override(up)
        client = self._client
        if client is None:
            return False
        try:
            r = await client.get(
                f"{up.url}/readyz", timeout=self.settings.probe_timeout_ms / 1000.0
            )
            return r.status_code == 200
        except Exception:  # noqa: BLE001 - a failed probe is a result, not a crash
            return False

    async def probe_all(self) -> None:
        results = await asyncio.gather(
            *(self.probe_once(up) for up in self.upstreams), return_exceptions=True
        )
        for up, ok in zip(self.upstreams, results, strict=True):
            up.record_probe(ok is True)
            self._publish_state(up)

    async def _health_loop(self) -> None:
        interval = self.settings.probe_interval_ms / 1000.0
        while not self._stop.is_set():
            await self.probe_all()
            try:
                await asyncio.wait_for(self._stop.wait(), interval)
            except TimeoutError:
                continue

    def _publish_state(self, up: Upstream) -> None:
        self.metrics.upstream_state.labels(worker=up.name).set(
            obs.UPSTREAM_STATE_CODES[str(up.health)]
        )

    # -- routing -------------------------------------------------------------
    def pick(self, exclude: str | None = None) -> Upstream | None:
        """Least outstanding among eligible workers; name breaks a tie.

        The tie-break is by name rather than random so that a quiet tier is
        deterministic and a test can assert which worker was chosen.
        """
        lease = self.pick_lease(exclude=exclude)
        return lease.upstream if lease is not None else None

    def pick_lease(self, exclude: str | None = None) -> ProbeLease | None:
        """``pick``, but handing back the lease that owns the probe slot.

        This is the form ``forward`` uses, because a lease is a thing you can
        put in a ``with`` block; an upstream is not. ``pick`` leaves the lease
        pending on the upstream instead, and ``attempt`` collects it, so the
        slot is still released exactly once whichever of the two a caller uses.
        """
        candidates = [
            up for up in self.upstreams if up.eligible and up.name != exclude
        ]
        while candidates:
            chosen = min(candidates, key=lambda u: (u.outstanding, u.name))
            # Only the chosen worker spends a half-open probe slot. Asking every
            # candidate would burn the slot of a worker we then did not use.
            lease = chosen.admit()
            if lease is not None:
                return lease
            candidates.remove(chosen)
        return None

    def state_table(self) -> dict[str, Any]:
        return {
            "wall_time": time.time(),
            "monotonic": self.clock(),
            "workers": [up.snapshot() for up in self.upstreams],
            "eligible_count": sum(1 for up in self.upstreams if up.eligible),
            "settings": {
                "probe_interval_ms": self.settings.probe_interval_ms,
                "fail_threshold": self.settings.fail_threshold,
                "success_threshold": self.settings.success_threshold,
                "request_timeout_ms": self.settings.request_timeout_ms,
                "connect_timeout_ms": self.settings.connect_timeout_ms,
                "breaker_failures": self.settings.breaker_failures,
                "breaker_open_s": self.settings.breaker_open_s,
                "retry_rule": RETRY_RULE,
            },
        }

    # -- forwarding ----------------------------------------------------------
    async def attempt(self, up: Upstream, payload: bytes, headers: dict[str, str]) -> Attempt:
        """One try against one worker, with the router's own timeout.

        The timeout is enforced with ``asyncio.wait_for`` around the whole
        send-and-read, not with httpx's per-phase timeouts, so there is a single
        number the client's latency is bounded by. Cancelling that task unwinds
        the ``async with client.stream(...)`` block, which closes the
        connection -- that close is what tells the worker to stop. httpx's own
        timeout is set higher so it never fires first and never becomes a second
        source of truth.
        """
        client = self._client
        if client is None:
            raise RuntimeError("router not started")
        budget = self.settings.request_timeout_ms / 1000.0
        started = self.clock()
        # Whatever probe slot the pick took for this attempt is now this
        # attempt's to give back, in the `finally` below.
        lease = up.take_lease()
        up.outstanding += 1
        try:
            return await asyncio.wait_for(
                self._send(client, up, payload, headers, budget, started), budget
            )
        except TimeoutError:
            latency = self.clock() - started
            self.metrics.upstream_latency_seconds.labels(worker=up.name).observe(latency)
            up.record_request_result(failure=True)
            return Attempt(
                worker=up.name,
                status=504,
                body=b"",
                latency_s=latency,
                timed_out=True,
                error="router timeout",
            )
        finally:
            up.outstanding -= 1
            # Every exit from this attempt passes here: a result, a timeout, a
            # transport error, or a client disconnect cancelling the task. That
            # is the whole guarantee -- the half-open slot is given back by the
            # shape of the code, not by a handler someone remembered to write
            # for the failure mode they happened to think of.
            lease.release()

    async def _send(
        self,
        client: httpx.AsyncClient,
        up: Upstream,
        payload: bytes,
        headers: dict[str, str],
        budget: float,
        started: float,
    ) -> Attempt:
        try:
            async with client.stream(
                "POST",
                f"{up.url}/v1/score",
                content=payload,
                headers={**headers, "content-type": "application/json"},
                # Connect gets its own short budget so a dead worker is refused
                # fast and the one retry is cheap. Read is set above the router
                # budget on purpose: asyncio.wait_for in `attempt` is the single
                # authority on when the client gives up.
                timeout=httpx.Timeout(
                    budget * 2, connect=self.settings.connect_timeout_ms / 1000.0
                ),
            ) as resp:
                status = resp.status_code
                body = await resp.aread()
        except PRE_SEND_ERRORS as exc:
            latency = self.clock() - started
            self.metrics.upstream_latency_seconds.labels(worker=up.name).observe(latency)
            up.record_request_result(failure=True)
            return Attempt(
                worker=up.name,
                status=0,
                body=b"",
                latency_s=latency,
                pre_send_failure=True,
                error=f"{type(exc).__name__}: {exc}",
            )
        except Exception as exc:  # noqa: BLE001 - transport failure after send
            latency = self.clock() - started
            self.metrics.upstream_latency_seconds.labels(worker=up.name).observe(latency)
            up.record_request_result(failure=True)
            return Attempt(
                worker=up.name,
                status=0,
                body=b"",
                latency_s=latency,
                error=f"{type(exc).__name__}: {exc}",
            )
        latency = self.clock() - started
        self.metrics.upstream_latency_seconds.labels(worker=up.name).observe(latency)
        # 429 is not a failure. See Upstream.record_request_result.
        up.record_request_result(failure=status >= 500)
        return Attempt(worker=up.name, status=status, body=body, latency_s=latency)

    async def forward(self, payload: bytes, headers: dict[str, str]) -> Response:
        """Pick, try, maybe retry once, and turn the outcome into a response."""
        first = self.pick_lease()
        if first is None:
            return self._no_upstream()
        # `with` on the lease, not a bare call to `attempt`: if a later edit
        # ever returns between the pick and the attempt, the slot still goes
        # back. `attempt` releasing the same lease first makes this a no-op.
        with first as first_up:
            attempt = await self.attempt(first_up, payload, headers)
            retried = False

            if attempt.pre_send_failure:
                second = self.pick_lease(exclude=first_up.name)
                if second is not None:
                    with second as second_up:
                        self.metrics.retries_total.inc()
                        retried = True
                        attempt = await self.attempt(second_up, payload, headers)

        return self._respond(attempt, retried)

    def _no_upstream(self) -> JSONResponse:
        body = {
            "error": "no_upstream",
            "detail": "no worker is eligible for traffic",
            "workers": [up.snapshot() for up in self.upstreams],
        }
        self.metrics.requests_total.labels(status="503").inc()
        return JSONResponse(body, status_code=503, headers={RETRIED_HEADER: "0"})

    def _respond(self, attempt: Attempt, retried: bool) -> Response:
        headers = {
            WORKER_HEADER: attempt.worker,
            RETRIED_HEADER: "1" if retried else "0",
            ATTEMPTS_HEADER: "2" if retried else "1",
        }
        if attempt.timed_out:
            self.metrics.requests_total.labels(status="504").inc()
            return JSONResponse(
                {
                    "error": "upstream_timeout",
                    "detail": (
                        f"worker {attempt.worker} did not answer within "
                        f"{self.settings.request_timeout_ms:.0f} ms; upstream cancelled"
                    ),
                },
                status_code=504,
                headers=headers,
            )
        if attempt.status == 0:
            self.metrics.requests_total.labels(status="502").inc()
            return JSONResponse(
                {"error": "upstream_unreachable", "detail": attempt.error or "no response"},
                status_code=502,
                headers=headers,
            )
        self.metrics.requests_total.labels(status=str(attempt.status)).inc()
        return Response(
            content=attempt.body,
            status_code=attempt.status,
            media_type="application/json",
            headers=headers,
        )


def _worker_name(index: int, url: str) -> str:
    """Name workers A, B, C... by position. The port is kept in the URL."""
    if index < 26:
        return chr(ord("A") + index)
    return f"W{index}"


def create_router_app(
    settings: RouterSettings | None = None,
    *,
    router: Router | None = None,
    trace_console: bool | None = None,
) -> FastAPI:
    """Build the router app. Pass `router` to inject a fake-probe Router."""
    cfg = settings or RouterSettings.from_env()
    tracer = obs.setup_tracing("reward-serve-router", console=trace_console)
    rt = router if router is not None else Router(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await rt.start()
        try:
            yield
        finally:
            await rt.stop()

    app = FastAPI(title="reward-serve-router", version="0.1.0", lifespan=lifespan)
    app.state.router = rt

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        n = sum(1 for up in rt.upstreams if up.eligible)
        body = {"status": "ready" if n else "no_upstream", "eligible_workers": n}
        return JSONResponse(body, status_code=200 if n else 503)

    @app.get("/routerz")
    async def routerz() -> JSONResponse:
        return JSONResponse(rt.state_table())

    @app.get("/metrics")
    async def metrics_endpoint() -> Response:
        body, ctype = obs.render_registry(rt.metrics.registry)
        return Response(content=body, media_type=ctype)

    @app.post("/v1/score")
    async def score(request: Request) -> Response:
        payload = await request.body()
        # One span per routed request, and the traceparent that carries it into
        # the worker. Without the injection the worker opens its own root span
        # and the two tiers show up as two unrelated traces -- which is exactly
        # the moment a multi-tier trace is worth having.
        with tracer.start_as_current_span("router.request") as span:
            span.set_attribute("http.route", "/v1/score")
            headers = obs.inject_trace_context({})
            response = await rt.forward(payload, headers)
            span.set_attribute("http.status_code", response.status_code)
            span.set_attribute("router.worker", response.headers.get(WORKER_HEADER, "none"))
            span.set_attribute("router.retried", response.headers.get(RETRIED_HEADER, "0"))
            return response

    return app


def build_parser() -> argparse.ArgumentParser:
    env = RouterSettings.from_env()
    p = argparse.ArgumentParser(prog="reward_serve.router", description=__doc__)
    p.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8100")))
    p.add_argument(
        "--workers",
        default=",".join(env.worker_urls),
        help="comma-separated worker base URLs",
    )
    p.add_argument("--probe-interval-ms", type=float, default=env.probe_interval_ms)
    p.add_argument("--fail-threshold", type=int, default=env.fail_threshold)
    p.add_argument("--success-threshold", type=int, default=env.success_threshold)
    p.add_argument("--request-timeout-ms", type=float, default=env.request_timeout_ms)
    p.add_argument("--connect-timeout-ms", type=float, default=env.connect_timeout_ms)
    p.add_argument("--breaker-failures", type=int, default=env.breaker_failures)
    p.add_argument("--breaker-open-s", type=float, default=env.breaker_open_s)
    p.add_argument("--log-level", default="warning")
    return p


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    args = build_parser().parse_args(argv)
    urls = tuple(u.strip().rstrip("/") for u in args.workers.split(",") if u.strip())
    if not urls:
        raise SystemExit("router needs at least one worker URL (--workers)")
    settings = RouterSettings(
        worker_urls=urls,
        probe_interval_ms=args.probe_interval_ms,
        fail_threshold=args.fail_threshold,
        success_threshold=args.success_threshold,
        request_timeout_ms=args.request_timeout_ms,
        connect_timeout_ms=args.connect_timeout_ms,
        breaker_failures=args.breaker_failures,
        breaker_open_s=args.breaker_open_s,
    )
    print(f"router on {args.host}:{args.port} workers={urls}", flush=True)
    uvicorn.run(
        create_router_app(settings), host=args.host, port=args.port, log_level=args.log_level
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
