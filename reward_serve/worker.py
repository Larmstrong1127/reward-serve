"""One worker process: the scoring server, named, observed, and breakable.

A worker is `reward_serve.server` plus three things the single-tier server did
not need:

1. **A name.** Every response carries ``x-rs-worker``. Without it a client
   behind the router cannot say which process served it, and "the router moved
   the traffic" is unfalsifiable.
2. **Trace continuation.** ``obs.instrument_app`` extracts an inbound
   ``traceparent``, so the worker's ``score.request`` span is a child of the
   router's span instead of a new root.
3. **A fault surface**, registered only when ``RS_FAULTS_ENABLED=1``. See
   ``reward_serve/faults.py``.

Run it::

    python -m reward_serve.worker --port 8001 --name A

``--synthetic`` swaps the checkpoint for a sleep of a fixed shape. It exists so
tests and the tier-up smoke check do not pay two model loads, and any number
produced against it describes the sleep, never the model. The worker says so
in its own ``model`` field, which is how a reader can tell the runs apart.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse  # noqa: E402
import time  # noqa: E402
from collections.abc import Sequence  # noqa: E402
from typing import Any  # noqa: E402

from fastapi import FastAPI  # noqa: E402

from reward_serve import obs  # noqa: E402
from reward_serve.batcher import Scorer  # noqa: E402
from reward_serve.faults import (  # noqa: E402
    FaultState,
    FaultyScorer,
    faults_enabled,
    install_fault_admin,
)
from reward_serve.model import DEFAULT_MODEL_ID  # noqa: E402
from reward_serve.server import Settings, create_app  # noqa: E402

__all__ = ["SyntheticScorer", "build_parser", "create_worker_app", "main"]

WORKER_HEADER = "x-rs-worker"


class SyntheticScorer:
    """NOT the model. A sleep shaped like a batched forward pass.

    Cost is ``base_ms + per_item_ms * batch``, matching the stub the load
    harness was built against (``tests/stub_server.py``), so the two agree
    about what a fake forward pass costs.
    """

    def __init__(self, base_ms: float = 8.0, per_item_ms: float = 3.0) -> None:
        self.base_ms = base_ms
        self.per_item_ms = per_item_ms

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        time.sleep((self.base_ms + self.per_item_ms * len(pairs)) / 1000.0)
        return [float(len(p) + len(r)) for p, r in pairs]


def create_worker_app(
    settings: Settings | None = None,
    *,
    name: str = "worker",
    scorer: Scorer | None = None,
    fault_state: FaultState | None = None,
    trace_console: bool | None = None,
) -> FastAPI:
    """Build a named, instrumented, optionally breakable worker app."""
    cfg = settings or Settings.from_env()
    state = fault_state if fault_state is not None else FaultState()

    metrics = obs.setup_metrics()
    tracer = obs.setup_tracing(f"reward-serve-worker-{name}", console=trace_console)
    hooks = obs.ObsHooks(metrics, tracer=tracer)
    model_info: dict[str, Any] | None = None
    if scorer is not None:
        model_info = {
            "model": "synthetic (no model loaded)",
            "max_length": 0,
            "temperature": 1.0,
        }
    app = create_app(
        cfg,
        scorer=scorer,
        hooks=obs.as_batcher_hooks(hooks),
        model_info=model_info,
        scorer_wrapper=lambda inner: FaultyScorer(inner, state),
    )
    app.state.fault_state = state
    app.state.worker_name = name

    # Registered before instrument_app so the tracing middleware, added later,
    # ends up outside this one and records the injected 500 as the request's
    # status. An error fault is meant to be visible in the trace, not hidden
    # behind it.
    @app.middleware("http")
    async def _faults_and_identity(request: Any, call_next: Any) -> Any:
        from fastapi.responses import JSONResponse

        if request.url.path == "/v1/score" and state.should_error():
            response: Any = JSONResponse(
                {"error": "injected_fault", "detail": "fault: error"}, status_code=500
            )
        else:
            response = await call_next(request)
        response.headers[WORKER_HEADER] = name
        return response

    obs.instrument_app(app, hooks, metrics)
    app.state.faults_installed = install_fault_admin(app, state)
    return app


def build_parser() -> argparse.ArgumentParser:
    env = Settings.from_env()
    p = argparse.ArgumentParser(prog="reward_serve.worker", description=__doc__)
    p.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8001")))
    p.add_argument("--name", default=os.environ.get("RS_WORKER_NAME", "worker"))
    p.add_argument("--max-batch-size", type=int, default=env.max_batch_size)
    p.add_argument("--max-wait-ms", type=float, default=env.max_wait_ms)
    p.add_argument("--queue-depth", type=int, default=env.queue_depth)
    p.add_argument("--request-timeout-ms", type=float, default=env.request_timeout_ms)
    p.add_argument("--model-id", default=env.model_id or DEFAULT_MODEL_ID)
    p.add_argument(
        "--synthetic",
        action="store_true",
        default=os.environ.get("RS_WORKER_SYNTHETIC", "") == "1",
        help="sleep instead of loading the checkpoint; numbers describe the sleep",
    )
    p.add_argument("--log-level", default="warning")
    return p


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    args = build_parser().parse_args(argv)
    settings = Settings(
        max_batch_size=args.max_batch_size,
        max_wait_ms=args.max_wait_ms,
        queue_depth=args.queue_depth,
        request_timeout_ms=args.request_timeout_ms,
        model_id=args.model_id,
    )
    app = create_worker_app(
        settings,
        name=args.name,
        scorer=SyntheticScorer() if args.synthetic else None,
    )
    print(
        f"worker {args.name} on {args.host}:{args.port} "
        f"synthetic={args.synthetic} faults={faults_enabled()}",
        flush=True,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
