"""Bring the two-tier setup up and down as subprocesses on this one machine.

    python scripts/tier_up.py --workers 2            # up, print the state table
    python scripts/tier_up.py --down                 # graceful stop

Up is: start N workers, wait for each ``/readyz``, start the router, wait until
``/routerz`` reports every worker eligible, print the table. Waiting for the
workers *before* starting the router is not cosmetic -- the router starts every
worker at DOWN and needs ``success_threshold`` good probes to admit it, so a
router started first spends its first second refusing traffic for no reason.

Down is: drain the workers, then stop them, then stop the router. That order is
the one that loses nothing. Stopping the router first would strand requests
already forwarded to a worker with nowhere to return; draining the workers
first flips their ``/readyz`` to 503, the router marks them DOWN, and new work
stops arriving while in-flight work finishes.

``faults/run_faults.py`` imports ``Tier`` and drives it directly rather than
shelling out, so a scenario can kill one named worker mid-run.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import httpx  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_FILE = REPO_ROOT / "results" / ".tier_state.json"
DEFAULT_WORKER_BASE_PORT = 8001
DEFAULT_ROUTER_PORT = 8100


@dataclass
class WorkerProc:
    """One worker subprocess and the facts needed to reach or kill it."""

    name: str
    port: int
    url: str
    proc: subprocess.Popen[bytes] | None = None
    pid: int | None = None
    killed_at: float | None = None

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def kill(self) -> float:
        """Terminate the process. Returns the wall time it was signalled at.

        ``Popen.terminate`` on Windows is ``TerminateProcess``: the process gets
        no chance to drain, which is the point -- this stands in for the SIGTERM
        a supervisor sends, and the question the scenario asks is what the
        client sees when a worker disappears with work in flight.
        """
        at = time.time()
        self.killed_at = at
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
        return at


@dataclass
class Tier:
    """One router plus N workers, started and stopped as a unit."""

    n_workers: int = 2
    worker_base_port: int = DEFAULT_WORKER_BASE_PORT
    router_port: int = DEFAULT_ROUTER_PORT
    synthetic: bool = False
    faults_enabled: bool = True
    queue_depth: int = 256
    max_batch_size: int = 8
    router_timeout_ms: float = 3000.0
    probe_interval_ms: float = 200.0
    # "off" silences the SDK entirely; "console" prints every span to the
    # subprocess stdout, which is what the dry run needs to show one trace id
    # crossing both tiers, and what a load run must never turn on.
    trace: str = "off"
    python: str = sys.executable
    workers: list[WorkerProc] = field(default_factory=list)
    router: subprocess.Popen[bytes] | None = None
    router_pid: int | None = None

    @property
    def router_url(self) -> str:
        return f"http://127.0.0.1:{self.router_port}"

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["PYTHONPATH"] = str(REPO_ROOT)
        env["PYTHONUNBUFFERED"] = "1"
        if self.trace == "console":
            env["REWARD_SERVE_TRACE_CONSOLE"] = "1"
            env.pop("OTEL_SDK_DISABLED", None)
        else:
            # Console span export would print a span per request onto the pipe
            # and dominate the worker's own cost. A behavioural run gets no
            # tracing rather than tracing that changes what it measures.
            env["OTEL_SDK_DISABLED"] = "true"
        return env

    def up(self, timeout_s: float = 300.0) -> dict[str, Any]:
        """Start everything and wait until the router says all workers are UP."""
        env = self._env()
        for i in range(self.n_workers):
            name = chr(ord("A") + i)
            port = self.worker_base_port + i
            wenv = dict(env)
            wenv["RS_FAULTS_ENABLED"] = "1" if self.faults_enabled else "0"
            wenv["QUEUE_DEPTH"] = str(self.queue_depth)
            wenv["MAX_BATCH_SIZE"] = str(self.max_batch_size)
            cmd = [self.python, "-m", "reward_serve.worker",
                   "--port", str(port), "--name", name]
            if self.synthetic:
                cmd.append("--synthetic")
            proc = subprocess.Popen(cmd, env=wenv, cwd=str(REPO_ROOT))
            self.workers.append(
                WorkerProc(name=name, port=port, url=f"http://127.0.0.1:{port}",
                           proc=proc, pid=proc.pid)
            )
        for w in self.workers:
            _wait_ready(f"{w.url}/readyz", timeout_s)

        renv = dict(env)
        renv["RS_WORKER_URLS"] = ",".join(w.url for w in self.workers)
        self.router = subprocess.Popen(
            [self.python, "-m", "reward_serve.router",
             "--port", str(self.router_port),
             "--workers", renv["RS_WORKER_URLS"],
             "--request-timeout-ms", str(self.router_timeout_ms),
             "--probe-interval-ms", str(self.probe_interval_ms)],
            env=renv, cwd=str(REPO_ROOT),
        )
        self.router_pid = self.router.pid
        _wait_ready(f"{self.router_url}/healthz", 60.0)
        _wait_all_eligible(f"{self.router_url}/routerz", self.n_workers, 60.0)
        self._save_state()
        return self.routerz()

    def routerz(self) -> dict[str, Any]:
        r = httpx.get(f"{self.router_url}/routerz", timeout=5.0)
        return dict(r.json())

    def worker(self, name: str) -> WorkerProc:
        for w in self.workers:
            if w.name == name:
                return w
        raise KeyError(f"no worker named {name!r}")

    def set_fault(self, name: str, spec: dict[str, Any]) -> dict[str, Any]:
        w = self.worker(name)
        r = httpx.post(f"{w.url}/admin/fault", json=spec, timeout=10.0)
        r.raise_for_status()
        return dict(r.json())

    def down(self, drain_wait_s: float = 3.0) -> None:
        """Drain the workers, stop them, then stop the router."""
        for w in self.workers:
            if w.alive():
                # A dead worker needs no drain; that is the normal case after
                # the kill scenario, not an error.
                with contextlib.suppress(Exception):
                    httpx.post(f"{w.url}/admin/drain", timeout=5.0)
        deadline = time.time() + drain_wait_s
        while time.time() < deadline:
            time.sleep(0.1)
        for w in self.workers:
            _stop(w.proc)
        _stop(self.router)
        self.router = None
        STATE_FILE.unlink(missing_ok=True)

    def _save_state(self) -> None:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(
            json.dumps(
                {
                    "router_port": self.router_port,
                    "router_pid": self.router_pid,
                    "workers": [
                        {"name": w.name, "port": w.port, "pid": w.pid} for w in self.workers
                    ],
                    "started_at": time.time(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def _stop(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()


def _wait_ready(url: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last = "no attempt"
    while time.time() < deadline:
        try:
            r = httpx.get(url, timeout=3.0)
            if r.status_code == 200:
                return
            last = f"HTTP {r.status_code}"
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}"
        time.sleep(0.2)
    raise TimeoutError(f"{url} not ready in {timeout_s}s (last: {last})")


def _wait_all_eligible(url: str, n: int, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last: Any = None
    while time.time() < deadline:
        try:
            body = httpx.get(url, timeout=3.0).json()
            last = body.get("eligible_count")
            if last == n:
                return
        except Exception as exc:  # noqa: BLE001
            last = type(exc).__name__
        time.sleep(0.2)
    raise TimeoutError(f"{url} never reported {n} eligible workers (last: {last})")


def print_table(table: dict[str, Any]) -> None:
    print(f"\nrouter state table  ({table['eligible_count']} eligible)")
    print(f"{'worker':>7} {'state':>8} {'breaker':>10} {'outstanding':>12} {'url':<28}")
    for w in table["workers"]:
        print(
            f"{w['name']:>7} {w['state']:>8} {w['breaker']:>10} "
            f"{w['outstanding']:>12} {w['url']:<28}"
        )
    print(f"\nretry rule: {table['settings']['retry_rule']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--worker-base-port", type=int, default=DEFAULT_WORKER_BASE_PORT)
    ap.add_argument("--router-port", type=int, default=DEFAULT_ROUTER_PORT)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--trace", choices=("off", "console"), default="off")
    ap.add_argument("--down", action="store_true", help="stop a tier started earlier")
    args = ap.parse_args(argv)

    if args.down:
        if not STATE_FILE.exists():
            print("no tier state file; nothing to stop")
            return 1
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        tier = Tier(router_port=int(state["router_port"]))
        tier.router_pid = state.get("router_pid")
        for w in state["workers"]:
            tier.workers.append(
                WorkerProc(name=w["name"], port=w["port"],
                           url=f"http://127.0.0.1:{w['port']}", pid=w["pid"])
            )
        # A CLI --down runs in a process that did not spawn these, so it cannot
        # use Popen handles. Drain over HTTP, then kill by pid.
        tier.down(drain_wait_s=3.0)
        for w in state["workers"]:
            _kill_pid(int(w["pid"]))
        _kill_pid(int(state["router_pid"]))
        STATE_FILE.unlink(missing_ok=True)
        print("tier stopped")
        return 0

    tier = Tier(
        n_workers=args.workers,
        worker_base_port=args.worker_base_port,
        router_port=args.router_port,
        synthetic=args.synthetic,
        trace=args.trace,
    )
    table = tier.up()
    print_table(table)
    print(f"\nrouter: {tier.router_url}")
    print(f"stop with: {sys.executable} scripts/tier_up.py --down")
    return 0


def _kill_pid(pid: int) -> None:
    try:
        import psutil  # type: ignore

        p = psutil.Process(pid)
        p.terminate()
        p.wait(timeout=10)
    except Exception:  # noqa: BLE001 - already gone is the desired end state
        pass


if __name__ == "__main__":
    raise SystemExit(main())
