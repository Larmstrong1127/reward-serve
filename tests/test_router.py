"""Router behaviour: state machine, routing rule, timeout, retry, breaker.

The state machine and the breaker are tested as pure objects with a fake clock
and hand-fed probe results, because that is the only way to assert a transition
happened after exactly N probes rather than "eventually". The timeout, retry and
breaker-in-anger cases need real sockets, so they use tests/stub_upstream.py.
"""

from __future__ import annotations

import asyncio

import pytest

from reward_serve.obs import UPSTREAM_STATE_CODES, setup_router_metrics
from reward_serve.router import (
    BreakerState,
    Health,
    Router,
    RouterSettings,
    Upstream,
)
from tests.stub_upstream import StubUpstream, free_port, start_stubs, stop_stubs

PAYLOAD = b'{"prompt":"a","response":"b"}'


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def make_upstream(**kw) -> Upstream:
    clock = kw.pop("clock", FakeClock())
    up = Upstream(name="A", url="http://127.0.0.1:1", clock=clock, **kw)
    up.test_clock = clock  # type: ignore[attr-defined]
    return up


# --- health state machine -----------------------------------------------------


def test_starts_down_and_needs_success_threshold_to_come_up():
    up = make_upstream(fail_threshold=3, success_threshold=2)
    assert up.health is Health.DOWN
    assert up.record_probe(True) is Health.DOWN  # one is not enough
    assert up.record_probe(True) is Health.UP


def test_up_to_suspect_to_down_takes_exactly_fail_threshold_probes():
    up = make_upstream(fail_threshold=3, success_threshold=2)
    up.record_probe(True)
    up.record_probe(True)
    assert up.health is Health.UP
    assert up.record_probe(False) is Health.SUSPECT
    assert up.record_probe(False) is Health.SUSPECT
    assert up.record_probe(False) is Health.DOWN
    assert [t["to"] for t in up.transitions] == ["UP", "SUSPECT", "DOWN"]


def test_suspect_recovers_on_one_good_probe_but_down_does_not():
    up = make_upstream(fail_threshold=3, success_threshold=2)
    up.record_probe(True)
    up.record_probe(True)
    up.record_probe(False)
    assert up.health is Health.SUSPECT
    assert up.record_probe(True) is Health.UP

    for _ in range(3):
        up.record_probe(False)
    assert up.health is Health.DOWN
    assert up.record_probe(True) is Health.DOWN
    assert up.record_probe(True) is Health.UP


def test_a_success_resets_the_failure_run():
    up = make_upstream(fail_threshold=3, success_threshold=1)
    up.record_probe(True)
    up.record_probe(False)
    up.record_probe(False)
    up.record_probe(True)  # back to UP, counter cleared
    up.record_probe(False)
    up.record_probe(False)
    assert up.health is Health.SUSPECT  # not DOWN: only two in a row this time


def test_only_up_is_eligible():
    up = make_upstream(success_threshold=1)
    up.record_probe(True)
    assert up.eligible
    up.record_probe(False)
    assert up.health is Health.SUSPECT
    assert not up.eligible


# --- routing rule -------------------------------------------------------------


async def make_router(urls, **kw) -> Router:
    settings = RouterSettings(worker_urls=tuple(urls), **kw)

    async def always_ready(up: Upstream) -> bool:
        return True

    return Router(settings, probe=always_ready, metrics=setup_router_metrics())


async def bring_up(rt: Router) -> None:
    for _ in range(rt.settings.success_threshold):
        await rt.probe_all()


async def test_least_outstanding_picks_the_least_loaded():
    rt = await make_router(["http://a", "http://b", "http://c"])
    await bring_up(rt)
    a, b, c = rt.upstreams
    a.outstanding, b.outstanding, c.outstanding = 3, 1, 2
    assert rt.pick() is b
    b.outstanding = 9
    assert rt.pick() is c


async def test_ties_break_by_name_so_a_quiet_tier_is_deterministic():
    rt = await make_router(["http://a", "http://b"])
    await bring_up(rt)
    assert rt.pick().name == "A"


async def test_pick_skips_workers_that_are_not_up():
    rt = await make_router(["http://a", "http://b"])
    await bring_up(rt)
    a, b = rt.upstreams
    a.outstanding = 0
    b.outstanding = 5
    for _ in range(rt.settings.fail_threshold):
        a.record_probe(False)
    assert a.health is Health.DOWN
    assert rt.pick() is b


async def test_no_eligible_worker_gives_503_that_says_why():
    rt = await make_router(["http://a"])
    await rt.start()
    try:
        resp = await rt.forward(PAYLOAD, {})
        assert resp.status_code == 503
        body = resp.body.decode()
        assert "no_upstream" in body
        assert "DOWN" in body  # the table travels with the refusal
    finally:
        await rt.stop()


async def test_state_gauge_tracks_the_state_machine():
    rt = await make_router(["http://a"], success_threshold=1, fail_threshold=1)
    await bring_up(rt)
    assert _gauge(rt, "A") == UPSTREAM_STATE_CODES["UP"]
    rt.upstreams[0].record_probe(False)
    rt._publish_state(rt.upstreams[0])
    assert _gauge(rt, "A") == UPSTREAM_STATE_CODES["DOWN"]


def _gauge(rt: Router, worker: str) -> float:
    value = rt.metrics.registry.get_sample_value(
        "rs_router_upstream_state", {"worker": worker}
    )
    assert value is not None
    return float(value)


# --- timeout and cancellation -------------------------------------------------


async def test_router_timeout_returns_504_and_cancels_the_upstream():
    """The stub holds the request; the router must hang up, not just answer."""
    slow = StubUpstream("A", delay_s=5.0)
    await start_stubs(slow)
    rt = await make_router([slow.url], request_timeout_ms=250.0)
    await bring_up(rt)
    await rt.start()
    try:
        resp = await rt.forward(PAYLOAD, {})
        assert resp.status_code == 504
        assert b"upstream cancelled" in resp.body
        assert await _eventually(lambda: slow.cancelled == 1)
        assert slow.served == 0
    finally:
        await rt.stop()
        await stop_stubs(slow)


async def test_a_timeout_is_not_retried():
    slow = StubUpstream("A", delay_s=5.0)
    spare = StubUpstream("B")
    await start_stubs(slow, spare)
    rt = await make_router([slow.url, spare.url], request_timeout_ms=250.0)
    await bring_up(rt)
    await rt.start()
    try:
        resp = await rt.forward(PAYLOAD, {})
        assert resp.status_code == 504
        assert spare.requests == 0
        assert _counter(rt, "rs_router_retries_total") == 0.0
    finally:
        await rt.stop()
        await stop_stubs(slow, spare)


# --- retry policy -------------------------------------------------------------


async def test_connection_refused_before_send_is_retried_on_the_other_worker():
    dead = f"http://127.0.0.1:{free_port()}"
    live = StubUpstream("B")
    await start_stubs(live)
    rt = await make_router([dead, live.url])
    await bring_up(rt)
    await rt.start()
    try:
        rt.upstreams[0].outstanding = 0
        rt.upstreams[1].outstanding = 5  # force the dead one to be picked first
        resp = await rt.forward(PAYLOAD, {})
        assert resp.status_code == 200
        assert resp.headers["x-rs-worker"] == "B"
        assert resp.headers["x-rs-retried"] == "1"
        assert live.requests == 1
        assert _counter(rt, "rs_router_retries_total") == 1.0
    finally:
        await rt.stop()
        await stop_stubs(live)


async def test_a_500_that_reached_a_worker_is_never_retried():
    """The core of the retry rule: bytes were sent, so the request stays failed."""
    broken = StubUpstream("A", status=500)
    spare = StubUpstream("B")
    await start_stubs(broken, spare)
    rt = await make_router([broken.url, spare.url])
    await bring_up(rt)
    await rt.start()
    try:
        resp = await rt.forward(PAYLOAD, {})
        assert resp.status_code == 500
        assert resp.headers["x-rs-retried"] == "0"
        assert broken.requests == 1
        assert spare.requests == 0
        assert _counter(rt, "rs_router_retries_total") == 0.0
    finally:
        await rt.stop()
        await stop_stubs(broken, spare)


async def test_a_429_is_returned_untouched_and_is_not_a_failure():
    busy = StubUpstream("A", status=429)
    spare = StubUpstream("B")
    await start_stubs(busy, spare)
    rt = await make_router([busy.url, spare.url])
    await bring_up(rt)
    await rt.start()
    try:
        for _ in range(10):
            resp = await rt.forward(PAYLOAD, {})
            assert resp.status_code == 429
            rt.upstreams[1].outstanding = 5  # keep A the least-loaded choice
        assert rt.upstreams[0].breaker is BreakerState.CLOSED
        assert rt.upstreams[0].breaker_consecutive_failures == 0
        assert spare.requests == 0
    finally:
        await rt.stop()
        await stop_stubs(busy, spare)


async def test_retry_needs_a_second_eligible_worker_or_it_gives_up():
    dead = f"http://127.0.0.1:{free_port()}"
    rt = await make_router([dead])
    await bring_up(rt)
    await rt.start()
    try:
        resp = await rt.forward(PAYLOAD, {})
        assert resp.status_code == 502
        assert resp.headers["x-rs-retried"] == "0"
        assert _counter(rt, "rs_router_retries_total") == 0.0
    finally:
        await rt.stop()


# --- circuit breaker ----------------------------------------------------------


def test_breaker_opens_after_k_consecutive_failures_and_not_before():
    clock = FakeClock()
    up = make_upstream(clock=clock, breaker_failures=3, breaker_open_s=5.0)
    up.record_request_result(failure=True)
    up.record_request_result(failure=True)
    assert up.breaker is BreakerState.CLOSED
    up.record_request_result(failure=True)
    assert up.breaker is BreakerState.OPEN
    assert not up.breaker_admit()


def test_an_open_breaker_recovers_and_pick_is_what_recovers_it():
    """The regression for the bug the latency run found.

    `eligible` used to read the raw breaker state, so nothing ever called the
    method that moves OPEN to HALF_OPEN and a tripped worker stayed ejected
    forever. Driving it through `pick` -- not through the breaker directly --
    is what makes this test able to fail.
    """
    clock = FakeClock()
    settings = RouterSettings(worker_urls=("http://a",), breaker_failures=1,
                              breaker_open_s=5.0, success_threshold=1)
    rt = Router(settings, metrics=setup_router_metrics(), clock=clock)
    up = rt.upstreams[0]
    up.record_probe(True)
    assert rt.pick() is up

    up.record_request_result(failure=True)
    assert up.breaker is BreakerState.OPEN
    assert rt.pick() is None

    clock.advance(5.1)
    chosen = rt.pick()
    assert chosen is up, "an open breaker must eventually admit one probe"
    assert up.breaker is BreakerState.HALF_OPEN
    assert rt.pick() is None, "the second request must not ride the probe slot"

    up.record_request_result(failure=False)
    assert up.breaker is BreakerState.CLOSED
    assert rt.pick() is up


def test_a_success_before_the_threshold_resets_the_run():
    up = make_upstream(breaker_failures=3)
    up.record_request_result(failure=True)
    up.record_request_result(failure=True)
    up.record_request_result(failure=False)
    up.record_request_result(failure=True)
    up.record_request_result(failure=True)
    assert up.breaker is BreakerState.CLOSED


def test_breaker_half_opens_after_t_and_admits_exactly_one_probe():
    clock = FakeClock()
    up = make_upstream(clock=clock, breaker_failures=1, breaker_open_s=5.0)
    up.record_request_result(failure=True)
    assert up.breaker is BreakerState.OPEN
    clock.advance(4.9)
    assert not up.breaker_admit()
    clock.advance(0.2)
    assert up.breaker_admit()  # the one probe
    assert up.breaker is BreakerState.HALF_OPEN
    assert not up.breaker_admit()  # a second concurrent request is refused


def test_a_good_probe_closes_the_breaker_and_a_bad_one_reopens_it():
    clock = FakeClock()
    up = make_upstream(clock=clock, breaker_failures=1, breaker_open_s=5.0)
    up.record_request_result(failure=True)
    clock.advance(6.0)
    assert up.breaker_admit()
    up.record_request_result(failure=False)
    assert up.breaker is BreakerState.CLOSED
    assert up.breaker_admit()

    up.record_request_result(failure=True)
    clock.advance(6.0)
    assert up.breaker_admit()
    up.record_request_result(failure=True)
    assert up.breaker is BreakerState.OPEN
    assert not up.breaker_admit()


def test_an_open_breaker_makes_a_healthy_worker_ineligible():
    up = make_upstream(breaker_failures=1, success_threshold=1)
    up.record_probe(True)
    assert up.eligible
    up.record_request_result(failure=True)
    assert up.health is Health.UP
    assert not up.eligible


async def test_breaker_opens_against_a_worker_returning_500s():
    broken = StubUpstream("A", status=500)
    spare = StubUpstream("B")
    await start_stubs(broken, spare)
    rt = await make_router([broken.url, spare.url], breaker_failures=3)
    await bring_up(rt)
    await rt.start()
    try:
        for _ in range(3):
            rt.upstreams[1].outstanding = 5
            await rt.forward(PAYLOAD, {})
        assert rt.upstreams[0].breaker is BreakerState.OPEN
        rt.upstreams[1].outstanding = 5
        resp = await rt.forward(PAYLOAD, {})
        assert resp.headers["x-rs-worker"] == "B"  # traffic moved off A
        assert resp.status_code == 200
    finally:
        await rt.stop()
        await stop_stubs(broken, spare)


# --- trace propagation --------------------------------------------------------


async def test_the_router_sends_a_traceparent_to_the_worker():
    live = StubUpstream("A")
    await start_stubs(live)
    rt = await make_router([live.url])
    await bring_up(rt)
    await rt.start()
    try:
        from reward_serve import obs

        tracer = obs.setup_tracing("test-router", console=True)
        with tracer.start_as_current_span("router.request"):
            await rt.forward(PAYLOAD, obs.inject_trace_context({}))
        assert live.seen_traceparents
        tp = live.seen_traceparents[0]
        assert tp is not None and tp.startswith("00-")
    finally:
        await rt.stop()
        await stop_stubs(live)


# --- helpers ------------------------------------------------------------------


def _counter(rt: Router, name: str) -> float:
    value = rt.metrics.registry.get_sample_value(name)
    return 0.0 if value is None else float(value)


async def _eventually(predicate, timeout: float = 2.0) -> bool:
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while loop.time() < end:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return predicate()


@pytest.fixture(autouse=True)
def _no_gpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
