"""Model-based tests for the breaker's transition table.

Why this file exists: the half-open probe slot has now leaked twice for the
same reason -- a code path that ended an attempt without going through the one
place that gives the slot back. Both times a test was written for the path that
was found. This file does not test paths. It enumerates the state space and the
event alphabet, drives random sequences of events at the real ``Upstream`` and
the real ``Router.attempt``, and asserts the invariants after every single
event. A third leak path has to break one of these without anyone writing a
test for it.

Tool: ``hypothesis``. It was chosen over a hand-rolled seeded driver for one
reason that matters here -- when a sequence fails it shrinks to the shortest
sequence that still fails, which is the difference between "some 30-event run
broke it" and "cancel a half-open probe". The runs are reproducible either
way: ``derandomize=True`` fixes the seeds, so this suite is deterministic.

What is real and what is a stub: the breaker, the routing, the lease and
``Router.attempt`` -- including its timeout handler and its ``finally`` -- are
the shipped code. Only ``Router._send`` is replaced, by a coroutine that
returns the scripted outcome without a socket. That is deliberate: the slot is
released in ``attempt``, so ``attempt`` is what has to be exercised.
"""

from __future__ import annotations

import asyncio
import contextlib

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from reward_serve.obs import setup_router_metrics
from reward_serve.router import Attempt, BreakerState, Health, Router, RouterSettings

# The event alphabet. Every one of these is something that can happen to a
# worker in the running router.
PROBE_OK = "probe_ok"
PROBE_FAIL = "probe_fail"
ADVANCE = "advance_clock_short"
ADVANCE_PAST = "advance_clock_past_open_window"
REQ_OK = "request_success"
REQ_429 = "request_429"
REQ_500 = "request_failure"
REQ_CONNECT = "request_connect_error"
REQ_TIMEOUT = "request_timeout"
REQ_CANCEL = "request_cancelled"

# The alphabet is weighted, not uniform, and the weights are the one piece of
# tuning in this file. Uniform events almost never produce the run that reaches
# the interesting corner -- health UP, then `breaker_failures` failures in a
# row, then enough clock for the open window, then a cancellation while the
# half-open probe is in flight. Measured: with a uniform alphabet, 1000
# sequences of 30 events did not reach a cancelled half-open probe at all.
# Weighting the failure, clock and cancellation events reaches it. The
# invariants are unchanged; only how often each state is visited is.
EVENT_WEIGHTS = {
    PROBE_OK: 3,
    PROBE_FAIL: 1,
    ADVANCE: 2,
    ADVANCE_PAST: 3,
    REQ_OK: 2,
    REQ_429: 1,
    REQ_500: 4,
    REQ_CONNECT: 2,
    REQ_TIMEOUT: 2,
    REQ_CANCEL: 3,
}
EVENTS = [event for event, weight in EVENT_WEIGHTS.items() for _ in range(weight)]
REPAIR_EVENTS = [PROBE_OK, REQ_OK, ADVANCE_PAST]

LEGAL_BREAKER_STATES = {BreakerState.CLOSED, BreakerState.OPEN, BreakerState.HALF_OPEN}
LEGAL_HEALTH_STATES = {Health.UP, Health.SUSPECT, Health.DOWN}


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _make_router(clock: FakeClock) -> Router:
    settings_ = RouterSettings(
        worker_urls=("http://127.0.0.1:1",),
        breaker_failures=3,
        breaker_open_s=5.0,
        fail_threshold=3,
        success_threshold=2,
        request_timeout_ms=3000.0,
    )
    rt = Router(settings_, metrics=setup_router_metrics(), clock=clock)
    # `attempt` only forwards the client to `_send`, which is stubbed below.
    rt._client = object()  # type: ignore[assignment]
    return rt


def _install_send(rt: Router, outcome: dict[str, str]) -> None:
    """Replace only the transport. Everything above it is the shipped code."""

    async def fake_send(client, up, payload, headers, budget, started):
        kind = outcome["kind"]
        if kind == REQ_TIMEOUT:
            # What `asyncio.wait_for` raises when the budget expires; raising
            # it here drives `attempt`'s real timeout handler.
            raise TimeoutError
        if kind == REQ_CANCEL:
            await asyncio.Event().wait()  # never resolves; the test cancels it
        if kind == REQ_CONNECT:
            up.record_request_result(failure=True)
            return Attempt(
                worker=up.name, status=0, body=b"", latency_s=0.0, pre_send_failure=True
            )
        status = {REQ_OK: 200, REQ_429: 429, REQ_500: 500}[kind]
        up.record_request_result(failure=status >= 500)
        return Attempt(worker=up.name, status=status, body=b"", latency_s=0.0)

    rt._send = fake_send


async def _apply(rt: Router, clock: FakeClock, event: str, outcome: dict[str, str]) -> None:
    if event == PROBE_OK:
        rt.upstreams[0].record_probe(True)
        return
    if event == PROBE_FAIL:
        rt.upstreams[0].record_probe(False)
        return
    if event == ADVANCE:
        clock.advance(2.6)  # time passes, but the 5 s open window has not
        return
    if event == ADVANCE_PAST:
        clock.advance(5.2)  # the open window has expired
        return

    # Requests go through `forward`, so the pick, the lease, the attempt, the
    # retry rule and the release are all the shipped code. The test never
    # touches a lease itself -- if it did, it would be testing its own release
    # instead of the router's, and a leak in the router would still pass.
    outcome["kind"] = event
    if event == REQ_CANCEL:
        task = asyncio.ensure_future(rt.forward(b"{}", {}))
        for _ in range(3):
            await asyncio.sleep(0)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        return
    await rt.forward(b"{}", {})


def _check_invariants(rt: Router, where: str) -> None:
    up = rt.upstreams[0]
    # (d) the state is always one of the three legal values.
    assert up.breaker in LEGAL_BREAKER_STATES, where
    assert up.health in LEGAL_HEALTH_STATES, where
    # (e) the failure counter never goes negative.
    assert up.breaker_consecutive_failures >= 0, where
    assert up.outstanding >= 0, where
    # (a) a CLOSED breaker never holds a probe slot.
    if up.breaker is BreakerState.CLOSED:
        assert not up.breaker_probe_inflight, f"{where}: CLOSED breaker holds a probe slot"
    if up.breaker is BreakerState.OPEN:
        # An OPEN breaker holds no slot either, and it always knows when it
        # opened, or it could never half-open again.
        assert not up.breaker_probe_inflight, f"{where}: OPEN breaker holds a probe slot"
        assert up.breaker_opened_at is not None, where
    # (b) no slot survives the event that took it. Every event either does not
    # take the slot or ends the attempt that took it, so between events the
    # slot is always free. This is the invariant a new leak path breaks.
    assert not up.breaker_probe_inflight, f"{where}: probe slot still held between events"
    assert up.probe_epoch == 0, f"{where}: probe slot epoch left set between events"


async def _drive(events: list[str]) -> None:
    clock = FakeClock()
    rt = _make_router(clock)
    outcome = {"kind": REQ_OK}
    _install_send(rt, outcome)
    _check_invariants(rt, "start")
    for i, event in enumerate(events):
        await _apply(rt, clock, event, outcome)
        _check_invariants(rt, f"after event {i} ({event})")


# --- exhaustive walk of the reachable state graph -----------------------------
#
# Random sequences alone were not enough, and that is worth recording rather
# than hiding: with the release removed, 1000 random 40-event sequences did NOT
# reach a cancelled half-open probe -- reaching it needs health UP, then
# `breaker_failures` failures in a row, then enough clock, then a cancellation,
# in that order, and blind sampling almost never lines those up. So the search
# is not left to chance. The walk below enumerates every reachable abstract
# state and applies every event to each one, which reaches that corner by
# construction and covers the whole transition table rather than a sample of it.


def _abstract_state(rt: Router) -> tuple[object, ...]:
    """The state as the transition table sees it: counters bucketed, clock relative.

    Bucketing is what makes the walk finite. The exact failure count past the
    threshold and the exact monotonic time do not change which transition fires;
    "at or past the threshold" and "the open window has expired" do.
    """
    up = rt.upstreams[0]
    if up.breaker_opened_at is None:
        window = "none"
    elif rt.clock() - up.breaker_opened_at >= up.breaker_open_s:
        window = "expired"
    else:
        window = "waiting"
    return (
        up.health,
        up.breaker,
        min(up.consecutive_failures, up.fail_threshold),
        min(up.consecutive_successes, up.success_threshold),
        min(up.breaker_consecutive_failures, up.breaker_failures),
        up.breaker_probe_inflight,
        window,
    )


async def _walk_state_graph(
    max_depth: int = 8,
) -> tuple[set[tuple[object, ...]], dict[tuple[object, ...], list[str]]]:
    """Breadth-first over reachable states, checking invariants at every step.

    Each node is replayed from a fresh router, so a state is always reached the
    way the running router would reach it, never by poking fields. Returns the
    set of (state, event) pairs exercised and one shortest event path to each
    reachable state.
    """
    reached: dict[tuple[object, ...], list[str]] = {}
    covered: set[tuple[object, ...]] = set()
    frontier: list[list[str]] = [[]]
    for _ in range(max_depth):
        nxt: list[list[str]] = []
        for path in frontier:
            for event in EVENT_WEIGHTS:
                seq = [*path, event]
                clock = FakeClock()
                rt = _make_router(clock)
                outcome = {"kind": REQ_OK}
                _install_send(rt, outcome)
                before = _abstract_state(rt)
                for i, ev in enumerate(seq):
                    before = _abstract_state(rt)
                    await _apply(rt, clock, ev, outcome)
                    _check_invariants(rt, f"{seq[: i + 1]}")
                covered.add((*before, event))
                state = _abstract_state(rt)
                if state not in reached:
                    reached[state] = seq
                    nxt.append(seq)
        frontier = nxt
        if not frontier:
            break
    return covered, reached


_PATH_CACHE: list[list[str]] = []


def _paths_to_every_state() -> list[list[str]]:
    """One event path to each reachable state, computed once and reused.

    The random sequences below start from these instead of from a fresh router.
    That is the fix for the honest weakness recorded above: a uniform or even a
    weighted random walk spends nearly all of its budget in the states that are
    cheap to reach, so the deep corners were sampled at a rate near zero.
    Starting from every reachable state makes the random part do what it is
    good at -- long unplanned event sequences -- from every starting point the
    machine actually has.
    """
    if not _PATH_CACHE:
        _PATH_CACHE.extend(asyncio.run(_walk_state_graph())[1].values())
    return _PATH_CACHE


def test_every_reachable_state_takes_every_event():
    covered, _reached = asyncio.run(_walk_state_graph())
    states = {key[:-1] for key in covered}
    assert len(states) >= 20, f"the walk only reached {len(states)} states"
    # The walk is only worth anything if it actually visits the corner the two
    # historical leaks lived in. Assert that, so the test cannot quietly stop
    # covering it after some future change to the state machine.
    # The half-open probe is taken by the pick *inside* a request event, so the
    # state a cancellation arrives at is OPEN with the window expired and the
    # worker UP. That is the corner both leaks lived in.
    corner = [
        key
        for key in covered
        if key[0] is Health.UP
        and key[1] is BreakerState.OPEN
        and key[-2] == "expired"
        and key[-1] == REQ_CANCEL
    ]
    assert corner, "the walk never cancelled a request that would take the half-open slot"
    for event in EVENT_WEIGHTS:
        assert any(key[-1] == event for key in covered), f"event never exercised: {event}"


@settings(
    max_examples=1000,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(
    prefix=st.deferred(lambda: st.sampled_from(_paths_to_every_state())),
    tail=st.lists(st.sampled_from(EVENTS), min_size=1, max_size=30),
)
def test_invariants_hold_after_every_event_in_any_sequence(prefix, tail):
    asyncio.run(_drive([*prefix, *tail]))


@settings(
    max_examples=300,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(
    prefix=st.deferred(lambda: st.sampled_from(_paths_to_every_state())),
    tail=st.lists(st.sampled_from(EVENTS), min_size=0, max_size=30),
)
def test_only_successes_always_make_a_worker_eligible_again(prefix, tail):
    """(c) no permanent ejection.

    Whatever state a random sequence leaves the worker in, a bounded run of
    good news -- probes that pass, requests that succeed, and enough clock for
    the open window to expire -- must make it eligible again. A worker that
    can be ejected for good is the bug this whole file was written for.
    """

    async def run() -> None:
        clock = FakeClock()
        rt = _make_router(clock)
        outcome = {"kind": REQ_OK}
        _install_send(rt, outcome)
        for event in [*prefix, *tail]:
            await _apply(rt, clock, event, outcome)
        up = rt.upstreams[0]
        # The bound: one long clock advance clears the open window, two good
        # probes clear DOWN, and one good request closes a half-open breaker.
        for i, event in enumerate(REPAIR_EVENTS * 4):
            await _apply(rt, clock, event, outcome)
            _check_invariants(rt, f"repair event {i} ({event})")
        assert up.health is Health.UP, f"health stuck at {up.health} after repair"
        assert up.breaker is BreakerState.CLOSED, f"breaker stuck at {up.breaker} after repair"
        assert up.eligible, "a worker that only sees successes must become eligible again"

    asyncio.run(run())
