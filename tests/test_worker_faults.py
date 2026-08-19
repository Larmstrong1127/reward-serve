"""The worker's fault surface: off by default, and each fault does its thing.

Everything here runs against a synthetic scorer through TestClient, so no
checkpoint loads. The faults are what is under test, not the model.
"""

from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

from reward_serve.faults import FaultSpec, FaultState, FaultyScorer
from reward_serve.server import Settings
from reward_serve.worker import SyntheticScorer, create_worker_app

PAIR = {"prompt": "hello", "response": "world"}


def build(monkeypatch, *, faults: bool, name: str = "A", **kw) -> tuple:
    monkeypatch.setenv("RS_FAULTS_ENABLED", "1" if faults else "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    state = FaultState()
    settings = Settings(max_batch_size=4, max_wait_ms=5, **kw)
    app = create_worker_app(
        settings, name=name, scorer=SyntheticScorer(base_ms=1, per_item_ms=0.5),
        fault_state=state, trace_console=True,
    )
    return app, state


# --- the flag -----------------------------------------------------------------


def test_fault_endpoint_is_404_when_the_flag_is_off(monkeypatch):
    app, _ = build(monkeypatch, faults=False)
    assert app.state.faults_installed is False
    with TestClient(app) as tc:
        assert tc.post("/admin/fault", json={"kind": "clear"}).status_code == 404
        assert tc.get("/admin/fault").status_code == 404
        # and the worker still works
        assert tc.post("/v1/score", json=PAIR).status_code == 200


def test_fault_endpoint_exists_only_with_the_explicit_flag(monkeypatch):
    app, _ = build(monkeypatch, faults=True)
    assert app.state.faults_installed is True
    with TestClient(app) as tc:
        r = tc.post("/admin/fault", json={"kind": "clear"})
        assert r.status_code == 200
        assert r.json()["status"] == "applied"


@pytest.mark.parametrize("value", ["", "0", "true", "yes", "TRUE"])
def test_only_the_literal_one_enables_faults(monkeypatch, value):
    monkeypatch.setenv("RS_FAULTS_ENABLED", value)
    state = FaultState()
    app = create_worker_app(
        Settings(), name="A", scorer=SyntheticScorer(), fault_state=state,
        trace_console=True,
    )
    assert app.state.faults_installed is False


# --- worker identity ----------------------------------------------------------


def test_every_response_names_the_worker(monkeypatch):
    app, _ = build(monkeypatch, faults=True, name="B")
    with TestClient(app) as tc:
        assert tc.post("/v1/score", json=PAIR).headers["x-rs-worker"] == "B"
        assert tc.get("/readyz").headers["x-rs-worker"] == "B"


# --- individual faults --------------------------------------------------------


def test_latency_fault_delays_the_forward_pass(monkeypatch):
    app, _ = build(monkeypatch, faults=True)
    with TestClient(app) as tc:
        started = time.perf_counter()
        tc.post("/v1/score", json=PAIR)
        baseline = time.perf_counter() - started

        tc.post("/admin/fault", json={"kind": "latency", "ms": 300})
        started = time.perf_counter()
        r = tc.post("/v1/score", json=PAIR)
        slow = time.perf_counter() - started
        assert r.status_code == 200
        assert slow >= 0.3
        assert slow > baseline + 0.25

        tc.post("/admin/fault", json={"kind": "clear"})
        started = time.perf_counter()
        tc.post("/v1/score", json=PAIR)
        assert time.perf_counter() - started < 0.3


def test_error_fault_returns_500s_at_about_the_configured_rate(monkeypatch):
    app, _ = build(monkeypatch, faults=True)
    with TestClient(app) as tc:
        tc.post("/admin/fault", json={"kind": "error", "rate": 1.0})
        r = tc.post("/v1/score", json=PAIR)
        assert r.status_code == 500
        assert r.json()["error"] == "injected_fault"
        assert r.headers["x-rs-worker"] == "A"

        tc.post("/admin/fault", json={"kind": "clear"})
        assert tc.post("/v1/score", json=PAIR).status_code == 200


def test_error_rate_is_a_rate_not_a_switch(monkeypatch):
    """A seeded draw, so a 0.5 rate is checked as a range, not as a coin flip."""
    state = FaultState(seed=7)
    state.apply(FaultSpec(kind="error", rate=0.5))
    hits = sum(state.should_error() for _ in range(400))
    assert 150 < hits < 250


def test_a_hang_stops_the_forward_pass_until_it_is_cleared():
    state = FaultState()
    scorer = FaultyScorer(SyntheticScorer(base_ms=0, per_item_ms=0), state)
    state.apply(FaultSpec(kind="hang"))
    done = threading.Event()

    def run() -> None:
        scorer.score_pairs([("a", "b")])
        done.set()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    assert not done.wait(0.4), "a hung worker must not complete the forward pass"
    state.apply(FaultSpec(kind="clear"))
    assert done.wait(5.0), "clearing the hang must release the forward pass"
    t.join(timeout=5)


def test_hang_backs_the_queue_up_and_clear_releases_it(monkeypatch):
    """End to end: with a hang installed, nothing comes back until it clears."""
    app, _ = build(monkeypatch, faults=True, queue_depth=16)
    with TestClient(app) as tc:
        tc.post("/admin/fault", json={"kind": "hang"})
        result: dict[str, int] = {}

        def call() -> None:
            result["status"] = tc.post("/v1/score", json=PAIR).status_code

        t = threading.Thread(target=call, daemon=True)
        t.start()
        time.sleep(0.4)
        assert "status" not in result
        assert tc.get("/admin/fault").json()["hanging"] is True
        tc.post("/admin/fault", json={"kind": "clear"})
        t.join(timeout=10)
        assert result["status"] == 200


# --- request validation -------------------------------------------------------


def test_a_fault_without_its_argument_is_rejected(monkeypatch):
    app, _ = build(monkeypatch, faults=True)
    with TestClient(app) as tc:
        assert tc.post("/admin/fault", json={"kind": "latency"}).status_code == 422
        assert tc.post("/admin/fault", json={"kind": "error"}).status_code == 422
        assert tc.post("/admin/fault", json={"kind": "nonsense"}).status_code == 422
        assert (
            tc.post("/admin/fault", json={"kind": "error", "rate": 2.0}).status_code == 422
        )


def test_clear_removes_every_fault_at_once(monkeypatch):
    state = FaultState()
    state.apply(FaultSpec(kind="latency", ms=100))
    state.apply(FaultSpec(kind="error", rate=1.0))
    state.apply(FaultSpec(kind="hang"))
    snap = state.apply(FaultSpec(kind="clear"))
    assert snap == {
        "latency_ms": 0.0,
        "error_rate": 0.0,
        "hanging": False,
        "applied_count": 4,
    }
