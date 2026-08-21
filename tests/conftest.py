"""CPU-only test setup. The GPU on this machine belongs to other work.

Holds fixtures for both halves of the suite: the server core's in-process
StubScorer, and the load harness's out-of-process stub HTTP server.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import asyncio  # noqa: E402
import functools  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections.abc import Iterator, Sequence  # noqa: E402
from pathlib import Path  # noqa: E402

import httpx  # noqa: E402
import pytest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
STUB = REPO_ROOT / "tests" / "stub_server.py"

# `scripts/` is not an installed package, and several tests import from it.
# Putting the repo root on the path here rather than in individual test
# modules means the imports do not depend on collection order.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class StubScorer:
    """Deterministic, fast stand-in for the model in batcher and server tests.

    Records every batch it was handed so tests can assert on what was and was
    not forwarded.
    """

    def __init__(self, delay_s: float = 0.0) -> None:
        self.delay_s = delay_s
        self.batches: list[list[tuple[str, str]]] = []

    @property
    def seen_pairs(self) -> list[tuple[str, str]]:
        return [p for b in self.batches for p in b]

    @property
    def batch_sizes(self) -> list[int]:
        return [len(b) for b in self.batches]

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        self.batches.append(list(pairs))
        if self.delay_s:
            # Runs in the batcher's executor thread, so a plain sleep is fine.
            import time as _time

            _time.sleep(self.delay_s)
        return [float(len(p) + len(r)) for p, r in pairs]


@pytest.fixture
def stub() -> StubScorer:
    return StubScorer()


HUB_MODEL_ID = "DantheMan124/deberta-preference-reward"


@functools.cache
def hub_checkpoint_unavailable() -> str | None:
    """None when the published checkpoint can be reached, else a skip reason.

    CI has no warm Hub cache. The first run downloads the checkpoint; later
    runs restore it from the HF_HOME cache (see .github/workflows/ci.yml). When
    the Hub is unreachable -- offline, rate-limited, or the repo has moved --
    the model tests must say so out loud rather than fail the build for a
    reason that has nothing to do with the code under test.

    Only the config is fetched here (a few kB), so the probe itself is cheap.
    """
    try:
        from transformers import AutoConfig

        AutoConfig.from_pretrained(HUB_MODEL_ID)
    except Exception as exc:  # noqa: BLE001 - any failure to reach it is the same answer
        return (
            f"the Hub checkpoint {HUB_MODEL_ID} could not be loaded "
            f"({type(exc).__name__}: {exc}); this test needs the published "
            "checkpoint, which CI fetches from the Hub and caches under HF_HOME"
        )
    return None


@pytest.fixture(scope="session")
def hub_checkpoint() -> None:
    """Depend on this from any test that loads the real model."""
    reason = hub_checkpoint_unavailable()
    if reason:
        pytest.skip(reason)


@pytest.fixture(autouse=True)
def _fail_on_gpu() -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "tests must stay on CPU"


async def wait_until(predicate, timeout: float = 2.0, interval: float = 0.005) -> bool:
    """Poll a predicate; returns False on timeout instead of hanging the suite."""
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while loop.time() < end:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return predicate()


# --- load-harness fixtures: an out-of-process stub HTTP server -----------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_http_ready(base_url: str, timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if httpx.get(f"{base_url}/readyz", timeout=2.0).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.1)
    raise TimeoutError(f"{base_url} never became ready")


@pytest.fixture
def stub_server_factory() -> Iterator[object]:
    """Start stub servers with chosen batch settings; all are torn down after."""
    procs: list[subprocess.Popen[bytes]] = []

    def start(**env_overrides: object) -> str:
        port = _free_port()
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["PORT"] = str(port)
        env["PYTHONPATH"] = str(REPO_ROOT)
        for k, v in env_overrides.items():
            env[k] = str(v)
        procs.append(
            subprocess.Popen(
                [sys.executable, str(STUB)], env=env, cwd=str(REPO_ROOT),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        )
        base = f"http://127.0.0.1:{port}"
        _wait_http_ready(base)
        return base

    yield start
    for p in procs:
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()


@pytest.fixture
def stub_server(stub_server_factory: object) -> str:
    return stub_server_factory(  # type: ignore[operator]
        MAX_BATCH_SIZE=8, MAX_WAIT_MS=5, FORWARD_BASE_MS=2, FORWARD_PER_ITEM_MS=0.5
    )
