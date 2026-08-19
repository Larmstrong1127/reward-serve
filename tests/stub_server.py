"""A STUB scoring server. It is not reward-serve and it does not load a model.

It exists so the load harness can be built, tested and dry-run before the real
server core lands. It honours the same contract:

    GET  /readyz    -> 200 "ok" once the batcher loop is running
    POST /v1/score  {"prompt": str, "response": str}
                    -> 200 {"score", "batch_size", "queue_wait_ms", "forward_ms"}
                    -> 429 when the admission queue is full
                    -> 503 while draining (after SIGTERM/shutdown begins)

It implements a real queue and a real time-or-size flush policy, but the
"forward pass" is a sleep whose cost is FORWARD_BASE_MS + FORWARD_PER_ITEM_MS *
batch_size. Any number produced against this stub describes the STUB, never the
model. Sweep results collected against it must say so.

Env: PORT, MAX_BATCH_SIZE, MAX_WAIT_MS, MAX_QUEUE, FORWARD_BASE_MS,
     FORWARD_PER_ITEM_MS.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


@dataclass
class Config:
    port: int = int(os.environ.get("PORT", "8000"))
    max_batch_size: int = int(os.environ.get("MAX_BATCH_SIZE", "8"))
    max_wait_ms: float = float(os.environ.get("MAX_WAIT_MS", "5"))
    max_queue: int = int(os.environ.get("MAX_QUEUE", "256"))
    forward_base_ms: float = float(os.environ.get("FORWARD_BASE_MS", "8"))
    forward_per_item_ms: float = float(os.environ.get("FORWARD_PER_ITEM_MS", "3"))


async def sleep_at_least(seconds: float) -> None:
    """Sleep for at least `seconds`, resisting Windows asyncio's early timer
    wakeups (see the comment in Batcher.loop)."""
    deadline = time.perf_counter() + seconds
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return
        await asyncio.sleep(max(remaining, 0.0005))


@dataclass
class Item:
    enqueued_at: float
    future: asyncio.Future[dict[str, Any]] = field(default_factory=asyncio.Future)


class Batcher:
    """Time-or-size flush: fire when `max_batch_size` is reached or `max_wait_ms`
    has elapsed since the OLDEST waiting item, whichever comes first."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.queue: asyncio.Queue[Item] = asyncio.Queue()
        self.draining = False
        self.ready = False

    def depth(self) -> int:
        return self.queue.qsize()

    async def submit(self, item: Item) -> dict[str, Any]:
        self.queue.put_nowait(item)
        return await item.future

    async def loop(self) -> None:
        self.ready = True
        while True:
            first = await self.queue.get()
            batch = [first]
            deadline = time.perf_counter() + self.cfg.max_wait_ms / 1000.0
            while len(batch) < self.cfg.max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self.queue.get(), remaining))
                except asyncio.TimeoutError:
                    # Windows asyncio fires timers up to loop._clock_resolution
                    # (15.625 ms here) EARLY, so a TimeoutError does not prove
                    # the wait window elapsed. Re-check the wall clock; without
                    # this, MAX_WAIT_MS of 5 or 20 collapses to ~0 under load
                    # and the sweep would compare wait settings that never
                    # actually differed.
                    if time.perf_counter() >= deadline:
                        break
            flush_at = time.perf_counter()
            forward_s = (
                self.cfg.forward_base_ms + self.cfg.forward_per_item_ms * len(batch)
            ) / 1000.0
            await sleep_at_least(forward_s)
            forward_ms = (time.perf_counter() - flush_at) * 1000.0
            for it in batch:
                if not it.future.done():
                    it.future.set_result(
                        {
                            "score": 0.0,
                            "batch_size": len(batch),
                            "queue_wait_ms": (flush_at - it.enqueued_at) * 1000.0,
                            "forward_ms": forward_ms,
                        }
                    )


def _response(status: int, body: bytes, ctype: str = "application/json") -> bytes:
    reason = {200: "OK", 400: "Bad Request", 404: "Not Found",
              429: "Too Many Requests", 503: "Service Unavailable"}.get(status, "OK")
    head = (
        f"HTTP/1.1 {status} {reason}\r\n"
        f"Content-Type: {ctype}\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Connection: keep-alive\r\n\r\n"
    )
    return head.encode("ascii") + body


async def _read_request(
    reader: asyncio.StreamReader,
) -> tuple[str, str, bytes] | None:
    head = await reader.readuntil(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    method, path, _ = lines[0].split(" ", 2)
    length = 0
    for line in lines[1:]:
        if line.lower().startswith("content-length:"):
            length = int(line.split(":", 1)[1].strip())
    body = await reader.readexactly(length) if length else b""
    return method, path, body


def make_handler(batcher: Batcher, cfg: Config):  # noqa: ANN201 - asyncio callback
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                try:
                    req = await _read_request(reader)
                except (asyncio.IncompleteReadError, ConnectionResetError):
                    return
                if req is None:
                    return
                method, path, body = req
                if path == "/readyz":
                    ok = batcher.ready and not batcher.draining
                    writer.write(_response(200 if ok else 503, b"ok" if ok else b"draining", "text/plain"))
                elif path == "/v1/score" and method == "POST":
                    writer.write(await _score(batcher, cfg, body))
                else:
                    writer.write(_response(404, b'{"detail":"not found"}'))
                await writer.drain()
        finally:
            try:
                writer.close()
            except Exception:
                pass

    return handle


async def _score(batcher: Batcher, cfg: Config, body: bytes) -> bytes:
    if batcher.draining:
        return _response(503, b'{"detail":"draining"}')
    try:
        payload = json.loads(body)
        if not isinstance(payload, dict) or "prompt" not in payload or "response" not in payload:
            raise ValueError("prompt and response required")
    except Exception as exc:
        return _response(400, json.dumps({"detail": str(exc)}).encode())
    if batcher.depth() >= cfg.max_queue:
        return _response(429, b'{"detail":"queue full"}')
    result = await batcher.submit(Item(enqueued_at=time.perf_counter()))
    return _response(200, json.dumps(result).encode())


async def amain() -> None:
    cfg = Config()
    batcher = Batcher(cfg)
    asyncio.create_task(batcher.loop())
    server = await asyncio.start_server(make_handler(batcher, cfg), "127.0.0.1", cfg.port)
    async with server:
        await server.serve_forever()


def main() -> None:
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
