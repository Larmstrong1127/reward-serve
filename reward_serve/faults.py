"""Deliberate failure injection for a worker process. Off unless asked for.

Three faults, chosen because they are the three shapes an inference worker
actually fails in and they look different from the client:

* ``latency`` -- the worker still answers, just slowly. Health probes keep
  passing, so no state machine anywhere marks it bad; the only defence is the
  router's load-balancing rule and its per-request timeout.
* ``error`` -- the worker answers fast, with 500s, at a chosen rate. Probes
  still pass (``/readyz`` is not the scoring path), so this is the case where
  health checking is useless and the circuit breaker is the thing that helps.
* ``hang`` -- the worker stops consuming its queue. Requests already accepted
  never finish. Here the probe *does* eventually fail, and the interesting
  number is how long the client suffers before the state machine reacts.

The surface exists only when ``RS_FAULTS_ENABLED=1``. When the flag is off the
routes are never registered, so ``POST /admin/fault`` is a 404 from FastAPI's
own router rather than a guard inside a handler that could be bypassed by a
later refactor. ``tests/test_worker_faults.py`` asserts the 404.

Nothing here is a resilience feature. It is a way to make the failure happen on
purpose, at a known moment, so the client-side record of it can be trusted.
"""

from __future__ import annotations

import os
import random
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from reward_serve.batcher import Scorer

__all__ = [
    "FaultSpec",
    "FaultState",
    "FaultyScorer",
    "faults_enabled",
    "install_fault_admin",
]

# A hang holds a thread in the batcher's executor. Cap it so a run that forgets
# to clear the fault releases the process instead of wedging it forever. The
# cap is far longer than any router timeout, so from the client's point of view
# the hang is indistinguishable from permanent.
HANG_MAX_S = 300.0


def faults_enabled() -> bool:
    """True only for an explicit ``RS_FAULTS_ENABLED=1``."""
    return os.environ.get("RS_FAULTS_ENABLED", "") == "1"


class FaultSpec(BaseModel):
    """Body of ``POST /admin/fault``."""

    kind: Literal["latency", "error", "hang", "clear"]
    ms: float | None = Field(default=None, ge=0)
    rate: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _needs_its_argument(self) -> FaultSpec:
        if self.kind == "latency" and self.ms is None:
            raise ValueError("latency needs ms")
        if self.kind == "error" and self.rate is None:
            raise ValueError("error needs rate")
        return self


@dataclass
class FaultState:
    """What the worker is currently doing wrong. Shared across threads.

    The batcher's forward pass runs in an executor thread while the HTTP
    handlers run on the event loop, so every field here is read from both. They
    are plain floats and bools guarded by one lock; the hang uses an Event
    because a thread has to be woken when the fault is cleared.
    """

    latency_ms: float = 0.0
    error_rate: float = 0.0
    hanging: bool = False
    seed: int = 1234
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _release: threading.Event = field(default_factory=threading.Event, repr=False)
    _rng: random.Random = field(default_factory=random.Random, repr=False)
    _applied: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self._rng.seed(self.seed)
        self._release.set()

    def apply(self, spec: FaultSpec) -> dict[str, Any]:
        """Install one fault. ``clear`` removes all of them, including a hang."""
        with self._lock:
            if spec.kind == "clear":
                self.latency_ms = 0.0
                self.error_rate = 0.0
                self.hanging = False
                self._release.set()
            elif spec.kind == "latency":
                self.latency_ms = float(spec.ms or 0.0)
            elif spec.kind == "error":
                self.error_rate = float(spec.rate or 0.0)
            elif spec.kind == "hang":
                self.hanging = True
                self._release.clear()
            self._applied.append(
                {"kind": spec.kind, "ms": spec.ms, "rate": spec.rate, "at": time.time()}
            )
            return self._snapshot_locked()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> dict[str, Any]:
        return {
            "latency_ms": self.latency_ms,
            "error_rate": self.error_rate,
            "hanging": self.hanging,
            "applied_count": len(self._applied),
        }

    def should_error(self) -> bool:
        """Draw once against the configured error rate. Seeded, so runs repeat."""
        with self._lock:
            if self.error_rate <= 0.0:
                return False
            return self._rng.random() < self.error_rate

    def wait_if_hanging(self) -> bool:
        """Block the calling thread while a hang is installed.

        Returns True if it blocked at all. Called from the executor thread, so
        blocking here is what stops the batcher's queue from being consumed:
        the loop awaits this forward pass and never comes back for the next
        one, which is precisely the failure being simulated.
        """
        with self._lock:
            hanging = self.hanging
        if not hanging:
            return False
        self._release.wait(HANG_MAX_S)
        return True

    def sleep_latency(self) -> float:
        """Sleep the injected delay. Returns the seconds actually slept."""
        with self._lock:
            delay = self.latency_ms / 1000.0
        if delay > 0:
            time.sleep(delay)
        return delay


class FaultyScorer:
    """Wraps the real scorer and misbehaves on the way in.

    Latency and hang live here rather than in the HTTP handler on purpose: they
    have to delay the *forward pass*, because that is where a real inference
    worker gets slow, and because delaying inside the batcher is what makes a
    hang back the queue up instead of just making one request slow.
    """

    def __init__(self, inner: Scorer, state: FaultState) -> None:
        self._inner = inner
        self._state = state

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        self._state.wait_if_hanging()
        self._state.sleep_latency()
        return self._inner.score_pairs(pairs)


def install_fault_admin(app: FastAPI, state: FaultState) -> bool:
    """Register the fault routes if the flag is on. Returns whether it did."""
    if not faults_enabled():
        return False
    router = APIRouter()

    @router.post("/admin/fault")
    async def set_fault(spec: FaultSpec) -> JSONResponse:
        return JSONResponse({"status": "applied", **state.apply(spec)})

    @router.get("/admin/fault")
    async def get_fault() -> JSONResponse:
        return JSONResponse(state.snapshot())

    app.include_router(router)
    return True
