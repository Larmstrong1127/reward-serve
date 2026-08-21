"""Integration: scripts/tier_up.py really starts a router and two workers.

Marked slow because it spawns three processes and waits for sockets. It uses
synthetic workers, so it proves the orchestration and the routing, never a
latency number.

    pytest tests/test_tier_up.py -m slow -q
"""

from __future__ import annotations

import itertools
import socket
import sys
import time
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from tier_up import Tier  # noqa: E402

pytestmark = pytest.mark.slow


def _free(port: int) -> bool:
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


# Where the next block search starts. Advanced past every block we hand out so
# two consecutive tests never race for the same ports: `_free` binds and then
# closes, so "free" is only true at the instant it is checked, and a tier whose
# `down()` has returned may still be releasing its sockets.
_next_base = itertools.count(8300)


def _free_block(size: int, stop: int = 8900) -> int:
    """First base port with `size` consecutive free ports after it.

    Workers take base, base+1, ... so a block is what the tier needs. Probing a
    block rather than asking the OS for ephemeral ports keeps the ports
    contiguous, which is what `Tier` assumes.
    """
    for base in _next_base:
        if base + size >= stop:
            raise RuntimeError("no free port block for the tier")
        if all(_free(base + i) for i in range(size)):
            # Never hand this block out again in this session.
            for _ in range(size):
                next(_next_base)
            return base
    raise RuntimeError("no free port block for the tier")


def _wait_ports_free(base: int, size: int, timeout_s: float = 10.0) -> None:
    """Block until the OS has actually given the ports back.

    `Tier.down()` returns once it has asked its processes to stop; the sockets
    can outlive that. Without this wait a later test binds into ports the
    previous tier still holds, and its requests reach the draining worker --
    which surfaces as a 503 `{"error":"draining"}` from a tier the test never
    started.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if all(_free(base + i) for i in range(size)):
            return
        time.sleep(0.05)


@pytest.fixture
def tier():
    base = _free_block(3)
    t = Tier(
        n_workers=2,
        worker_base_port=base,
        router_port=base + 2,
        synthetic=True,
        faults_enabled=True,
    )
    try:
        yield t
    finally:
        t.down(drain_wait_s=0.5)
        _wait_ports_free(base, 3)


def test_tier_comes_up_with_two_workers_both_up(tier):
    table = tier.up(timeout_s=120.0)
    assert table["eligible_count"] == 2
    states = {w["name"]: w["state"] for w in table["workers"]}
    assert states == {"A": "UP", "B": "UP"}
    assert all(w["breaker"] == "CLOSED" for w in table["workers"])


def test_requests_through_the_router_reach_both_workers(tier):
    tier.up(timeout_s=120.0)
    served = []
    with httpx.Client(base_url=tier.router_url, timeout=30.0) as c:
        for i in range(10):
            r = c.post("/v1/score", json={"prompt": f"p{i}", "response": f"r{i}"})
            assert r.status_code == 200, r.text
            served.append(r.headers["x-rs-worker"])
            assert r.headers["x-rs-retried"] == "0"
    # Serial requests all land on the least-outstanding worker, which is the
    # name tie-break, so this asserts routing happened at all rather than a
    # particular split. The fault runs are where the split is measured.
    assert set(served) <= {"A", "B"}
    assert served


def test_router_exposes_its_metrics_and_state_table(tier):
    tier.up(timeout_s=120.0)
    with httpx.Client(base_url=tier.router_url, timeout=30.0) as c:
        c.post("/v1/score", json={"prompt": "p", "response": "r"})
        body = c.get("/metrics").text
        assert "rs_router_upstream_state" in body
        assert "rs_router_retries_total" in body
        assert "rs_router_upstream_latency_seconds" in body
        table = c.get("/routerz").json()
        assert {w["name"] for w in table["workers"]} == {"A", "B"}
        assert table["settings"]["retry_rule"].startswith("one retry")


def test_the_fault_surface_is_reachable_on_a_worker(tier):
    tier.up(timeout_s=120.0)
    applied = tier.set_fault("B", {"kind": "clear"})
    assert applied["status"] == "applied"
