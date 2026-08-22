"""Interleaving tests for the half-open probe slot's epoch guard.

``docs/breaker.md`` argued that ``ProbeLease.release`` carrying the epoch it
was issued with makes a stale lease harmless: a release that arrives after its
attempt has ended cannot clear a slot a *later* attempt has since taken. The
model-based suite could not measure that. It drives one request at a time and
checks its invariants *between* events, so no two attempts are ever live
against the same upstream at overlapping points inside ``Router.attempt``.
That is exactly the window the guard exists for.

This file measures it. Every test here drives the real ``Router``, the real
``Upstream``, the real ``pick_lease``/``attempt``/``forward`` -- including the
``finally`` that releases the lease and the outer ``with`` in ``forward`` that
releases it a second time. Only ``Router._send`` is replaced, by a stub that
parks on an ``asyncio.Event`` so the test decides, deterministically, where
each coroutine is suspended. Nothing is synchronised by sleeping: every wait is
on an Event that some other coroutine sets, so the tests neither race nor take
wall-clock time.

The overlap is not manufactured with a hook. It is the one the shipped code
already has: ``forward`` holds the first pick's lease across the *entire*
retried attempt against a second worker::

    with first as first_up:                # lease for worker A, epoch e
        attempt = await self.attempt(...)  # A fails pre-send -> breaker OPEN
        if attempt.pre_send_failure:
            second = self.pick_lease(exclude=first_up.name)
            with second as second_up:
                await self.attempt(second_up, ...)   # <-- long await here
    # first.release(epoch=e) runs only now

While that inner await is suspended, another request can reach worker A, find
its open window expired, and take a fresh half-open slot. Then the first
request's release fires against a slot it no longer owns. That is the race,
built entirely out of shipped code paths.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from reward_serve.obs import setup_router_metrics
from reward_serve.router import (
    Attempt,
    BreakerState,
    Health,
    ProbeLease,
    Router,
    RouterSettings,
)

# The invariant assertions are imported, not restated: if the model suite's
# notion of a consistent breaker changes, these tests must move with it.
from tests.test_router_breaker_model import _check_invariants

BREAKER_FAILURES = 3
BREAKER_OPEN_S = 5.0


class FakeClock:
    """The same injectable clock the model tests use. Real time never moves."""

    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class Transport:
    """A scripted stand-in for ``Router._send`` with a per-worker pause point.

    ``script[name]`` is the outcome the worker produces. If ``gate[name]`` is
    an Event, ``_send`` announces its arrival on ``entered[name]`` and then
    waits for that Event before producing the outcome -- so a test can hold a
    coroutine at a known point inside ``attempt``, with the probe slot taken
    and the ``finally`` not yet run.
    """

    def __init__(self) -> None:
        self.script: dict[str, str] = {}
        self.gate: dict[str, asyncio.Event] = {}
        self.entered: dict[str, asyncio.Event] = {}
        self.calls: list[str] = []

    def arm(self, name: str) -> None:
        """Make the next ``_send`` for ``name`` park until ``release`` is called."""
        self.gate[name] = asyncio.Event()
        self.entered[name] = asyncio.Event()

    async def wait_entered(self, name: str) -> None:
        await self.entered[name].wait()

    def release(self, name: str) -> None:
        self.gate.pop(name).set()

    def install(self, rt: Router) -> None:
        async def fake_send(
            client: Any,
            up: Any,
            payload: bytes,
            headers: dict[str, str],
            budget: float,
            started: float,
        ) -> Attempt:
            self.calls.append(up.name)
            gate = self.gate.get(up.name)
            if gate is not None:
                self.entered[up.name].set()
                await gate.wait()
            kind = self.script.get(up.name, "ok")
            if kind == "connect":
                up.record_request_result(failure=True)
                return Attempt(
                    worker=up.name,
                    status=0,
                    body=b"",
                    latency_s=0.0,
                    pre_send_failure=True,
                )
            status = {"ok": 200, "fail": 500, "shed": 429}[kind]
            up.record_request_result(failure=status >= 500)
            return Attempt(worker=up.name, status=status, body=b"", latency_s=0.0)

        rt._send = fake_send  # type: ignore[method-assign]


def make_router(clock: FakeClock, workers: int = 1) -> Router:
    settings_ = RouterSettings(
        worker_urls=tuple(f"http://127.0.0.1:{9000 + i}" for i in range(workers)),
        breaker_failures=BREAKER_FAILURES,
        breaker_open_s=BREAKER_OPEN_S,
        fail_threshold=3,
        success_threshold=2,
        request_timeout_ms=3000.0,
    )
    rt = Router(settings_, metrics=setup_router_metrics(), clock=clock)
    # `attempt` only passes the client through to `_send`, which is stubbed.
    rt._client = object()  # type: ignore[assignment]
    return rt


def upstream(rt: Router, name: str) -> Any:
    return next(u for u in rt.upstreams if u.name == name)


def bring_up(rt: Router, name: str) -> None:
    """Two good /readyz probes: DOWN -> UP, through the real state machine."""
    up = upstream(rt, name)
    up.record_probe(True)
    up.record_probe(True)
    assert up.health is Health.UP


async def open_breaker(rt: Router, tx: Transport, name: str) -> None:
    """Drive one worker's breaker to OPEN through real forwarded requests."""
    tx.script[name] = "fail"
    for _ in range(BREAKER_FAILURES):
        await rt.forward(b"{}", {})
    up = upstream(rt, name)
    assert up.breaker is BreakerState.OPEN
    assert not up.breaker_probe_inflight


# --- (a) two concurrent requests race for one half-open slot ------------------


def test_two_concurrent_forwards_share_one_half_open_upstream():
    """Exactly one of two racing requests gets the probe; the other is refused.

    The refusal is a 503 ``no_upstream``, not a queued wait and not a second
    request riding the probe slot: while the probe is in flight the upstream is
    not ``eligible`` at all (``breaker_ready`` is false for a HALF_OPEN
    upstream holding the slot), so ``pick_lease`` finds no candidate.
    """

    async def run() -> None:
        clock = FakeClock()
        rt = make_router(clock)
        tx = Transport()
        tx.install(rt)
        bring_up(rt, "A")
        await open_breaker(rt, tx, "A")
        clock.advance(BREAKER_OPEN_S + 0.2)

        up = upstream(rt, "A")
        tx.script["A"] = "ok"
        tx.arm("A")
        first = asyncio.ensure_future(rt.forward(b"{}", {}))
        await tx.wait_entered("A")

        # The probe is in flight and the slot is held by a live epoch.
        assert up.breaker is BreakerState.HALF_OPEN
        assert up.breaker_probe_inflight
        held_epoch = up.probe_epoch
        assert held_epoch > 0
        assert not up.breaker_ready()
        assert not up.eligible

        # The second request, started while the first is parked inside
        # `attempt`, is refused outright.
        second = await rt.forward(b"{}", {})
        assert second.status_code == 503
        assert tx.calls == ["A"] * (BREAKER_FAILURES + 1)
        assert up.probe_epoch == held_epoch, "the refused request disturbed the slot"

        tx.release("A")
        resp = await first
        assert resp.status_code == 200
        assert up.breaker is BreakerState.CLOSED
        _check_invariants(rt, "after the probe succeeded")

    asyncio.run(run())


# --- (b) THE stale release: a lease outliving its attempt ---------------------


async def _park_a_stale_first_lease(rt: Router, tx: Transport) -> Any:
    """Leave a request suspended on worker B while holding a dead A lease.

    Returns the still-running ``forward`` future. On return: worker A's breaker
    took a half-open slot for this request, the attempt failed pre-send so the
    breaker reopened and cleared that slot, and the retry onto worker B is
    parked inside ``attempt``. ``forward``'s outer ``with`` has not run yet, so
    the release of A's now-dead lease is still pending.
    """
    tx.script["A"] = "connect"
    tx.arm("B")
    task = asyncio.ensure_future(rt.forward(b"{}", {}))
    await tx.wait_entered("B")
    a = upstream(rt, "A")
    assert a.breaker is BreakerState.OPEN, "the failed probe should have reopened A"
    assert not a.breaker_probe_inflight
    return task


def test_stale_release_does_not_clear_a_later_attempts_slot():
    """(b) The case the epoch guard exists for, measured rather than argued.

    Request 1 takes worker A's half-open slot, fails pre-send, and retries onto
    worker B, where it parks -- still owing a release for A's slot. The clock
    then passes A's open window and request 2 takes a *fresh* half-open slot on
    A. Only then does request 1's stale release fire. It must not clear request
    2's slot, and request 2's probe must complete normally.
    """

    async def run() -> None:
        clock = FakeClock()
        rt = make_router(clock, workers=2)
        tx = Transport()
        tx.install(rt)
        bring_up(rt, "A")
        bring_up(rt, "B")
        await open_breaker(rt, tx, "A")
        clock.advance(BREAKER_OPEN_S + 0.2)

        first = await _park_a_stale_first_lease(rt, tx)

        # Request 2 reaches A after its window expires again and takes a new
        # half-open slot, while request 1 is still parked on B.
        clock.advance(BREAKER_OPEN_S + 0.2)
        a = upstream(rt, "A")
        tx.script["A"] = "ok"
        tx.arm("A")
        second = asyncio.ensure_future(rt.forward(b"{}", {}))
        await tx.wait_entered("A")
        assert a.breaker is BreakerState.HALF_OPEN
        assert a.breaker_probe_inflight
        live_epoch = a.probe_epoch
        assert live_epoch > 0

        # Now let request 1 finish. Its `finally` and its outer `with` both
        # fire, releasing the lease it took for A two attempts ago.
        tx.release("B")
        assert (await first).status_code == 200

        assert a.breaker_probe_inflight, "a stale release cleared a live probe slot"
        assert a.probe_epoch == live_epoch, "a stale release reset a live probe epoch"
        assert a.breaker is BreakerState.HALF_OPEN
        # And with the slot still held, nothing else can ride the probe.
        assert not a.eligible

        tx.release("A")
        assert (await second).status_code == 200
        assert a.breaker is BreakerState.CLOSED
        _check_invariants(rt, "after the stale release and the probe")

    asyncio.run(run())


def test_probe_epoch_is_never_reissued_after_the_slot_is_cleared():
    """A slot id is never handed out twice, which is what makes the guard work.

    An id that is recycled is an id a lease from a dead attempt can match. This
    pins the generator: across repeated open -> half-open -> open cycles, every
    slot taken has a strictly larger id than the one before it, even though
    ``probe_epoch`` itself is reset to 0 every time the slot is cleared.
    """

    async def run() -> None:
        clock = FakeClock()
        rt = make_router(clock)
        tx = Transport()
        tx.install(rt)
        bring_up(rt, "A")
        a = upstream(rt, "A")

        seen: list[int] = []
        for _ in range(4):
            await open_breaker(rt, tx, "A")
            clock.advance(BREAKER_OPEN_S + 0.2)
            tx.script["A"] = "ok"
            tx.arm("A")
            task = asyncio.ensure_future(rt.forward(b"{}", {}))
            await tx.wait_entered("A")
            seen.append(a.probe_epoch)
            tx.release("A")
            await task
            assert a.probe_epoch == 0

        assert seen == sorted(set(seen)), f"a slot id was reissued: {seen}"

    asyncio.run(run())


# --- (c) stale release arriving after the breaker closed ----------------------


def test_stale_release_after_close_does_not_perturb_closed_state():
    """(c) The same stale lease, but arriving once the breaker is CLOSED.

    This one is a regression guard rather than a teeth test: releasing into a
    CLOSED breaker clears an already-clear slot either way. What it pins is
    that the release path has no *other* effect -- it must not reopen the
    breaker, touch the failure counter, or move health.
    """

    async def run() -> None:
        clock = FakeClock()
        rt = make_router(clock, workers=2)
        tx = Transport()
        tx.install(rt)
        bring_up(rt, "A")
        bring_up(rt, "B")
        await open_breaker(rt, tx, "A")
        clock.advance(BREAKER_OPEN_S + 0.2)

        first = await _park_a_stale_first_lease(rt, tx)

        # Request 2 probes A and succeeds, closing the breaker outright.
        clock.advance(BREAKER_OPEN_S + 0.2)
        a = upstream(rt, "A")
        tx.script["A"] = "ok"
        assert (await rt.forward(b"{}", {})).status_code == 200
        assert a.breaker is BreakerState.CLOSED
        assert a.breaker_consecutive_failures == 0

        # Only now does the parked request give back its long-dead A lease.
        tx.release("B")
        assert (await first).status_code == 200

        assert a.breaker is BreakerState.CLOSED
        assert a.breaker_consecutive_failures == 0
        assert a.breaker_opened_at is None
        assert a.health is Health.UP
        assert a.eligible
        _check_invariants(rt, "after a stale release into a CLOSED breaker")

    asyncio.run(run())


# --- (d) rapid cycling with a firehose of releases from old epochs ------------


@pytest.mark.parametrize("probe_succeeds", [True, False])
def test_firehose_of_stale_releases_during_rapid_cycling(probe_succeeds: bool):
    """(d) Old-epoch releases fired at every point of an open/half-open cycle.

    Each cycle drives the breaker OPEN, waits out the window, parks a probe
    inside ``attempt``, then fires a release for every epoch id the upstream
    could plausibly have handed out. The live slot must survive all of them --
    proved by the concurrent request still being refused -- and the model
    suite's invariants must hold between cycles.
    """

    async def run() -> None:
        clock = FakeClock()
        rt = make_router(clock)
        tx = Transport()
        tx.install(rt)
        bring_up(rt, "A")
        a = upstream(rt, "A")
        _check_invariants(rt, "start")

        for cycle in range(4):
            await open_breaker(rt, tx, "A")
            _check_invariants(rt, f"cycle {cycle}: breaker open")
            clock.advance(BREAKER_OPEN_S + 0.2)

            tx.script["A"] = "ok" if probe_succeeds else "fail"
            tx.arm("A")
            task = asyncio.ensure_future(rt.forward(b"{}", {}))
            await tx.wait_entered("A")
            live_epoch = a.probe_epoch
            assert live_epoch > 0

            # The firehose: every epoch id an earlier attempt could be holding,
            # including the live one's neighbours and zero. The live epoch
            # itself is skipped -- releasing that one is the *correct* release,
            # not a stale one, and it is what ends the probe below.
            for epoch in range(0, live_epoch + 6):
                if epoch == live_epoch:
                    continue
                ProbeLease(a, epoch).release()
                assert a.breaker_probe_inflight, (
                    f"cycle {cycle}: release of epoch {epoch} cleared the live slot"
                )
                assert a.probe_epoch == live_epoch, (
                    f"cycle {cycle}: release of epoch {epoch} moved the live epoch"
                )
                # The observable consequence of a cleared slot: a second
                # request would be let through onto the half-open worker.
                assert not a.eligible, (
                    f"cycle {cycle}: epoch {epoch} release re-admitted traffic"
                )
                assert (await rt.forward(b"{}", {})).status_code == 503

            tx.release("A")
            await task
            _check_invariants(rt, f"cycle {cycle}: probe finished")
            assert a.breaker is (
                BreakerState.CLOSED if probe_succeeds else BreakerState.OPEN
            )
            if not probe_succeeds:
                clock.advance(BREAKER_OPEN_S + 0.2)
                tx.script["A"] = "ok"
                assert (await rt.forward(b"{}", {})).status_code == 200
            _check_invariants(rt, f"cycle {cycle}: end")

    asyncio.run(run())
