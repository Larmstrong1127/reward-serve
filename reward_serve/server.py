"""FastAPI surface for the reward server. Shapes are frozen by docs/contract.md."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import asyncio  # noqa: E402
from collections.abc import AsyncIterator  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from typing import Any  # noqa: E402

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from pydantic import BaseModel, Field, model_validator  # noqa: E402

from reward_serve.batcher import (  # noqa: E402
    BatcherHooks,
    ContinuousBatcher,
    DeadlineExceeded,
    QueueFullError,
    ScoreResult,
    Scorer,
)
from reward_serve.model import DEFAULT_MODEL_ID, RewardModel  # noqa: E402

__all__ = ["Settings", "create_app"]


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else float(raw)


@dataclass
class Settings:
    """Tunables. Every field has a matching env var and CLI flag."""

    max_batch_size: int = 8
    max_wait_ms: float = 10.0
    queue_depth: int = 256
    request_timeout_ms: float = 5000.0
    model_id: str = DEFAULT_MODEL_ID

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            max_batch_size=_env_int("MAX_BATCH_SIZE", 8),
            max_wait_ms=_env_float("MAX_WAIT_MS", 10.0),
            queue_depth=_env_int("QUEUE_DEPTH", 256),
            request_timeout_ms=_env_float("REQUEST_TIMEOUT_MS", 5000.0),
            model_id=os.environ.get("MODEL_ID", DEFAULT_MODEL_ID),
        )


class Pair(BaseModel):
    prompt: str
    response: str


class ScoreRequest(BaseModel):
    """Form A supplies prompt+response; form B supplies pairs. Exactly one."""

    prompt: str | None = None
    response: str | None = None
    pairs: list[Pair] | None = None
    timeout_ms: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _exactly_one_form(self) -> ScoreRequest:
        form_a = self.prompt is not None or self.response is not None
        form_b = self.pairs is not None
        if form_a and form_b:
            raise ValueError("supply either prompt/response or pairs, not both")
        if not form_a and not form_b:
            raise ValueError("supply either prompt/response or pairs")
        if form_a and (self.prompt is None or self.response is None):
            raise ValueError("form A needs both prompt and response")
        if form_b and not self.pairs:
            raise ValueError("pairs must not be empty")
        return self

    def as_pairs(self) -> list[tuple[str, str]]:
        if self.pairs is not None:
            return [(p.prompt, p.response) for p in self.pairs]
        assert self.prompt is not None and self.response is not None
        return [(self.prompt, self.response)]


class ServerState:
    """Holds the model, the batcher and the drain flag for one process."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model: RewardModel | None = None
        self.batcher: ContinuousBatcher | None = None
        self.hooks = BatcherHooks()
        self.draining = False

    @property
    def ready(self) -> bool:
        return (
            self.model is not None
            and self.batcher is not None
            and self.batcher.running
            and not self.draining
        )


def create_app(
    settings: Settings | None = None,
    *,
    scorer: Scorer | None = None,
    hooks: BatcherHooks | None = None,
    model_info: dict[str, Any] | None = None,
) -> FastAPI:
    """Build the app. Pass `scorer` to run against a stub instead of the model."""
    cfg = settings or Settings.from_env()
    state = ServerState(cfg)
    if hooks is not None:
        state.hooks = hooks

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if scorer is None:
            loop = asyncio.get_running_loop()
            state.model = await loop.run_in_executor(None, RewardModel, cfg.model_id)
            active: Scorer = state.model
        else:
            active = scorer
        state.batcher = ContinuousBatcher(
            active,
            max_batch_size=cfg.max_batch_size,
            max_wait_ms=cfg.max_wait_ms,
            queue_depth=cfg.queue_depth,
            request_timeout_ms=cfg.request_timeout_ms,
            hooks=state.hooks,
        )
        await state.batcher.start()
        try:
            yield
        finally:
            await state.batcher.stop()

    app = FastAPI(title="reward-serve", version="0.1.0", lifespan=lifespan)
    app.state.server_state = state

    def _info() -> dict[str, Any]:
        if state.model is not None:
            return {
                "model": state.model.model_id,
                "max_length": state.model.max_length,
                "temperature": state.model.temperature,
            }
        return model_info or {
            "model": "stub",
            "max_length": cfg.max_batch_size,
            "temperature": 1.0,
        }

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        batcher_running = state.batcher is not None and state.batcher.running
        body = {
            "status": "ready",
            "model_loaded": state.model is not None or scorer is not None,
            "batcher_running": batcher_running,
        }
        if state.draining:
            body["status"] = "draining"
            return JSONResponse(body, status_code=503)
        if not state.ready:
            body["status"] = "loading"
            return JSONResponse(body, status_code=503)
        return JSONResponse(body, status_code=200)

    @app.post("/admin/drain")
    async def drain() -> JSONResponse:
        if not state.draining:
            state.draining = True
            if state.batcher is not None:
                # Stop accepting at once; queued work still runs to completion.
                asyncio.create_task(state.batcher.drain())  # noqa: RUF006
        return JSONResponse({"status": "draining"})

    @app.post("/v1/score")
    async def score(req: ScoreRequest, request: Request) -> JSONResponse:
        if state.draining:
            return JSONResponse(
                {"error": "draining", "detail": "server is draining"}, status_code=503
            )
        batcher = state.batcher
        if batcher is None or not batcher.running:
            return JSONResponse(
                {"error": "not_ready", "detail": "model is still loading"}, status_code=503
            )

        pairs = req.as_pairs()
        tasks = [
            asyncio.create_task(batcher.submit(p, r, req.timeout_ms)) for p, r in pairs
        ]
        try:
            done = await asyncio.gather(*tasks)
        except QueueFullError as exc:
            for t in tasks:
                t.cancel()
            return JSONResponse(
                {"error": "queue_full", "detail": str(exc)}, status_code=429
            )
        except DeadlineExceeded as exc:
            for t in tasks:
                t.cancel()
            return JSONResponse(
                {"error": "deadline_exceeded", "detail": str(exc)}, status_code=504
            )
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()
            raise

        results: list[ScoreResult] = list(done)
        body = {
            "results": [
                {
                    "score": r.score,
                    "batch_size": r.batch_size,
                    "queue_wait_ms": round(r.queue_wait_ms, 3),
                    "forward_ms": round(r.forward_ms, 3),
                }
                for r in results
            ],
            **_info(),
        }
        return JSONResponse(body)

    return app
