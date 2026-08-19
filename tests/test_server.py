"""HTTP surface: score shapes, readiness semantics, backpressure and drain.

These run against a stub scorer through ASGI transport, so no model loads.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from conftest import StubScorer, wait_until

from reward_serve.server import Settings, create_app




async def test_healthz_and_readyz_semantics(stub: StubScorer) -> None:
    settings = Settings(max_batch_size=4, max_wait_ms=5)
    app = create_app(settings, scorer=stub)
    async with httpx.ASGITransport(app=app) as transport:  # noqa: SIM117
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            # Without lifespan the batcher has not started: healthz yes, readyz no.
            assert (await c.get("/healthz")).status_code == 200
            r = await c.get("/readyz")
            assert r.status_code == 503
            assert r.json()["status"] == "loading"


async def test_score_and_drain_end_to_end(stub: StubScorer) -> None:
    """Score both request forms, then prove a drain loses no in-flight request."""
    slow = StubScorer(delay_s=0.05)
    settings = Settings(max_batch_size=4, max_wait_ms=10)
    app = create_app(settings, scorer=slow)

    from fastapi.testclient import TestClient  # local import keeps startup cost off other tests

    with TestClient(app) as tc:
        assert tc.get("/readyz").status_code == 200
        assert tc.get("/readyz").json() == {
            "status": "ready",
            "model_loaded": True,
            "batcher_running": True,
        }

        r = tc.post("/v1/score", json={"prompt": "hello", "response": "world"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["results"]) == 1
        item = body["results"][0]
        assert set(item) == {"score", "batch_size", "queue_wait_ms", "forward_ms"}
        assert item["batch_size"] >= 1
        assert item["queue_wait_ms"] >= 0 and item["forward_ms"] > 0

        r = tc.post(
            "/v1/score",
            json={"pairs": [{"prompt": f"p{i}", "response": "r"} for i in range(4)]},
        )
        assert r.status_code == 200, r.text
        results = r.json()["results"]
        assert len(results) == 4
        # Four pairs submitted together should share a forward pass.
        assert max(x["batch_size"] for x in results) > 1

        # Malformed requests are rejected, not guessed at.
        assert tc.post("/v1/score", json={"prompt": "only"}).status_code == 422
        assert tc.post("/v1/score", json={}).status_code == 422
        assert (
            tc.post(
                "/v1/score", json={"prompt": "a", "response": "b", "pairs": []}
            ).status_code
            == 422
        )

        assert tc.post("/admin/drain").status_code == 200
        assert tc.get("/readyz").status_code == 503
        assert tc.get("/readyz").json()["status"] == "draining"
        assert tc.get("/healthz").status_code == 200  # process is still up
        r = tc.post("/v1/score", json={"prompt": "after", "response": "drain"})
        assert r.status_code == 503
        assert r.json()["error"] == "draining"


async def test_drain_does_not_lose_in_flight_requests() -> None:
    """Requests accepted before the drain all return a score."""
    slow = StubScorer(delay_s=0.03)
    settings = Settings(max_batch_size=2, max_wait_ms=5)
    app = create_app(settings, scorer=slow)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        # Drive lifespan by hand so we can interleave a drain with live requests.
        state = app.state.server_state
        from reward_serve.batcher import ContinuousBatcher

        state.batcher = ContinuousBatcher(
            slow, max_batch_size=2, max_wait_ms=5, request_timeout_ms=10_000
        )
        await state.batcher.start()

        inflight = [
            asyncio.create_task(
                c.post("/v1/score", json={"prompt": f"p{i}", "response": "r"})
            )
            for i in range(6)
        ]
        assert await wait_until(lambda: bool(slow.batches), timeout=2.0)

        drained = await c.post("/admin/drain")
        assert drained.status_code == 200

        responses = await asyncio.gather(*inflight)
        assert [r.status_code for r in responses] == [200] * 6, "a request was lost"
        assert sum(len(b) for b in slow.batches) == 6

        rejected = await c.post("/v1/score", json={"prompt": "new", "response": "r"})
        assert rejected.status_code == 503


async def test_full_queue_maps_to_http_429() -> None:
    """Backpressure surfaces as 429, immediately."""
    slow = StubScorer(delay_s=0.3)
    settings = Settings(max_batch_size=1, max_wait_ms=1, queue_depth=1)
    app = create_app(settings, scorer=slow)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        state = app.state.server_state
        from reward_serve.batcher import ContinuousBatcher

        state.batcher = ContinuousBatcher(
            slow, max_batch_size=1, max_wait_ms=1, queue_depth=1, request_timeout_ms=10_000
        )
        await state.batcher.start()

        occupy = asyncio.create_task(
            c.post("/v1/score", json={"prompt": "inflight", "response": "r"})
        )
        assert await wait_until(lambda: bool(slow.batches), timeout=2.0)
        fill = asyncio.create_task(
            c.post("/v1/score", json={"prompt": "queued", "response": "r"})
        )
        assert await wait_until(lambda: state.batcher.depth == 1, timeout=2.0)

        rejected = await c.post("/v1/score", json={"prompt": "over", "response": "r"})
        assert rejected.status_code == 429
        assert rejected.json()["error"] == "queue_full"

        await asyncio.gather(occupy, fill)
        assert "over" not in [p for p, _ in slow.seen_pairs]
        await state.batcher.stop()


async def test_expired_deadline_maps_to_http_504() -> None:
    slow = StubScorer(delay_s=0.2)
    settings = Settings(max_batch_size=1, max_wait_ms=1)
    app = create_app(settings, scorer=slow)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        state = app.state.server_state
        from reward_serve.batcher import ContinuousBatcher

        state.batcher = ContinuousBatcher(slow, max_batch_size=1, max_wait_ms=1)
        await state.batcher.start()

        occupy = asyncio.create_task(
            c.post("/v1/score", json={"prompt": "inflight", "response": "r"})
        )
        assert await wait_until(lambda: bool(slow.batches), timeout=2.0)

        r = await c.post(
            "/v1/score", json={"prompt": "doomed", "response": "r", "timeout_ms": 10}
        )
        assert r.status_code == 504
        assert r.json()["error"] == "deadline_exceeded"

        await occupy
        assert "doomed" not in [p for p, _ in slow.seen_pairs]
        await state.batcher.stop()


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"prompt": "a", "response": "b"}, 1),
        ({"pairs": [{"prompt": "a", "response": "b"}]}, 1),
        ({"pairs": [{"prompt": "a", "response": "b"}] * 3}, 3),
    ],
)
async def test_result_count_matches_submitted_pairs(
    stub: StubScorer, payload: dict, expected: int
) -> None:
    from fastapi.testclient import TestClient

    app = create_app(Settings(max_batch_size=4, max_wait_ms=5), scorer=stub)
    with TestClient(app) as tc:
        r = tc.post("/v1/score", json=payload)
        assert r.status_code == 200, r.text
        assert len(r.json()["results"]) == expected
