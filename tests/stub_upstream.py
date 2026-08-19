"""A raw-socket stand-in for a worker, used to test the router's edges.

It is deliberately not FastAPI and not `tests/stub_server.py`: the router tests
need to observe things an ASGI app hides, above all *whether the client hung up
before the response was written*. That is the only honest proof that the router
cancels an upstream attempt when its timeout fires, rather than returning 504 to
the client while the worker keeps computing.

Counters, all observed from the server side:
  requests   -- request lines fully read
  cancelled  -- connections the peer closed while this stub was still holding
                the request, i.e. attempts the router really did cancel
  served     -- responses actually written
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

_BODY = (
    b'{"results":[{"score":0.5,"batch_size":1,"queue_wait_ms":0.1,"forward_ms":1.0}],'
    b'"model":"stub-upstream","max_length":512,"temperature":1.0}'
)


def _http(status: int, body: bytes, headers: dict[str, str] | None = None) -> bytes:
    reason = {200: "OK", 429: "Too Many Requests", 500: "Internal Server Error",
              503: "Service Unavailable"}.get(status, "OK")
    head = [f"HTTP/1.1 {status} {reason}", "Content-Type: application/json",
            f"Content-Length: {len(body)}", "Connection: keep-alive"]
    for k, v in (headers or {}).items():
        head.append(f"{k}: {v}")
    return ("\r\n".join(head) + "\r\n\r\n").encode("ascii") + body


class StubUpstream:
    """One fake worker on a real socket."""

    def __init__(
        self,
        name: str = "S",
        *,
        status: int = 200,
        delay_s: float = 0.0,
        ready: bool = True,
    ) -> None:
        self.name = name
        self.status = status
        self.delay_s = delay_s
        self.ready = ready
        self.requests = 0
        self.served = 0
        self.cancelled = 0
        self.seen_traceparents: list[str | None] = []
        self._server: asyncio.AbstractServer | None = None
        self.port = 0

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def start(self) -> str:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = int(self._server.sockets[0].getsockname()[1])
        return self.url

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                try:
                    head = await reader.readuntil(b"\r\n\r\n")
                except (asyncio.IncompleteReadError, ConnectionResetError, OSError):
                    return
                lines = head.decode("latin-1").split("\r\n")
                method, path, _ = lines[0].split(" ", 2)
                length = 0
                traceparent: str | None = None
                for line in lines[1:]:
                    low = line.lower()
                    if low.startswith("content-length:"):
                        length = int(line.split(":", 1)[1].strip())
                    elif low.startswith("traceparent:"):
                        traceparent = line.split(":", 1)[1].strip()
                if length:
                    await reader.readexactly(length)

                if path == "/readyz":
                    body = b'{"status":"ready"}' if self.ready else b'{"status":"draining"}'
                    writer.write(_http(200 if self.ready else 503, body))
                    await writer.drain()
                    continue

                self.requests += 1
                self.seen_traceparents.append(traceparent)
                if self.delay_s and await self._peer_left(reader, self.delay_s):
                    self.cancelled += 1
                    return
                writer.write(
                    _http(self.status, _BODY, {"x-rs-worker": self.name})
                )
                await writer.drain()
                self.served += 1
        finally:
            # Closing an already-dead socket is the normal end of a cancelled
            # attempt, not an error.
            with contextlib.suppress(Exception):
                writer.close()

    async def _peer_left(self, reader: asyncio.StreamReader, hold_s: float) -> bool:
        """Hold the request for `hold_s`. True if the peer hung up meanwhile."""
        watcher: asyncio.Task[bytes] = asyncio.ensure_future(reader.read(1))
        done, _ = await asyncio.wait({watcher}, timeout=hold_s)
        if watcher in done:
            try:
                return watcher.result() == b""
            except Exception:  # noqa: BLE001 - a reset is also "the peer left"
                return True
        watcher.cancel()
        return False


async def start_stubs(*stubs: StubUpstream) -> tuple[str, ...]:
    urls: list[str] = []
    for s in stubs:
        urls.append(await s.start())
    return tuple(urls)


async def stop_stubs(*stubs: StubUpstream) -> None:
    for s in stubs:
        await s.stop()


def free_port() -> int:
    """A port nothing is listening on, for connection-refused tests."""
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port: Any = s.getsockname()[1]
    return int(port)
