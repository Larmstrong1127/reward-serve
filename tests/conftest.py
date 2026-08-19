"""CPU-only test setup. The GPU on this machine belongs to other work."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HOME", r"D:\ClaudeCache\hf")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import asyncio  # noqa: E402
from collections.abc import Sequence  # noqa: E402

import pytest  # noqa: E402


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
            import time

            time.sleep(self.delay_s)
        return [float(len(p) + len(r)) for p, r in pairs]


@pytest.fixture
def stub() -> StubScorer:
    return StubScorer()


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
