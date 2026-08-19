"""Batch-policy sweep: restart the server per cell, load it, record everything.

Grid: MAX_BATCH_SIZE x MAX_WAIT_MS x offered load (open-loop rps).

Every cell gets a FRESH server process with that cell's env, so no cell inherits
another cell's warm caches, queue state or allocator behaviour. Each cell writes
one line to results/<run-id>/cells.jsonl the moment it finishes -- sessions get
killed, and a half-finished sweep should still have its completed cells on disk.

Charter rule 3: `scripts.box_quiet.require_quiet` runs BEFORE the grid and the
sweep refuses to start if the box is contended. The report is stored next to the
results it gates. There is deliberately no flag to skip it.

Reproduce a run:
    python -m loadgen.sweep --run-id <id> [--server-cmd ...]
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

from loadgen.generator import RunResult, run_open_loop, wait_ready
from loadgen.stats import summarize

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.box_quiet import require_quiet  # noqa: E402

DEFAULT_BATCH_SIZES: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
DEFAULT_WAIT_MS: tuple[float, ...] = (0.0, 5.0, 20.0, 50.0)
# Placeholder loads; --rps must be set from what the dry run calibrated.
DEFAULT_RPS: tuple[float, ...] = (10.0, 40.0, 120.0)
DEFAULT_SERVER_CMD = f"{sys.executable} -m reward_serve"


def split_command(cmd: str) -> list[str]:
    """Split a command string without mangling Windows paths.

    `shlex.split(posix=True)` eats the backslashes in C:\\path\\to\\python.exe;
    `posix=False` keeps them but leaves the surrounding quotes attached to each
    token, which CreateProcess then fails to find. Split non-posix, then strip
    one layer of matching quotes.
    """
    out: list[str] = []
    for tok in shlex.split(cmd, posix=False):
        if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in "\"'":
            tok = tok[1:-1]
        out.append(tok)
    return out


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _pkg_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except Exception:
        return None


def environment_report() -> dict[str, Any]:
    """Measurement conditions. Every published number must be read next to this."""
    try:
        import psutil  # type: ignore

        cpu_logical: int | None = psutil.cpu_count(logical=True)
        cpu_physical: int | None = psutil.cpu_count(logical=False)
    except Exception:
        cpu_logical, cpu_physical = os.cpu_count(), None
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_model": os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor(),
        "cpu_logical": cpu_logical,
        "cpu_physical": cpu_physical,
        "torch_version": _pkg_version("torch"),
        "transformers_version": _pkg_version("transformers"),
        "httpx_version": _pkg_version("httpx"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "torch_num_threads": os.environ.get("TORCH_NUM_THREADS"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "git_commit": _git_commit(),
    }


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
            text=True, timeout=10, check=True,
        ).stdout.strip()
    except Exception:
        return None


class ServerProcess:
    """One server instance for one grid cell."""

    def __init__(self, cmd: Sequence[str], cwd: Path, env: dict[str, str], log_path: Path) -> None:
        self.cmd = list(cmd)
        self.cwd = cwd
        self.env = env
        self.log_path = log_path
        self.proc: subprocess.Popen[bytes] | None = None

    def __enter__(self) -> ServerProcess:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("wb")
        self.proc = subprocess.Popen(
            self.cmd, cwd=str(self.cwd), env=self.env,
            stdout=self._log, stderr=subprocess.STDOUT,
        )
        return self

    def __exit__(self, *exc: object) -> None:
        p = self.proc
        if p is not None and p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=15)
        with contextlib.suppress(Exception):
            self._log.close()

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def log_tail(self, n: int = 40) -> str:
        try:
            return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-n:])
        except Exception:
            return ""


def grid(
    batch_sizes: Sequence[int], wait_ms: Sequence[float], rps: Sequence[float]
) -> Iterator[tuple[int, float, float]]:
    for b in batch_sizes:
        for w in wait_ms:
            for r in rps:
                yield b, w, r


async def run_cell(
    *,
    batch_size: int,
    wait_ms: float,
    rps: float,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any]:
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    cell_id = f"b{batch_size}_w{int(wait_ms)}_r{rps:g}"
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "PORT": str(port),
            "MAX_BATCH_SIZE": str(batch_size),
            "MAX_WAIT_MS": str(wait_ms),
            "QUEUE_DEPTH": str(args.queue_depth),
            "PYTHONPATH": str(args.server_cwd),
        }
    )
    # The contract gives every env var a matching CLI flag, and the flag wins.
    # We set BOTH: the flags are what the server actually reads, the env vars
    # keep simpler targets (the stub) configurable through the same code path.
    cmd = split_command(args.server_cmd)
    if args.server_flags:
        cmd += [
            "--host", "127.0.0.1",
            "--port", str(port),
            "--max-batch-size", str(batch_size),
            "--max-wait-ms", str(wait_ms),
            "--queue-depth", str(args.queue_depth),
        ]
    log_path = out_dir / "server-logs" / f"{cell_id}.log"
    started = time.time()
    record: dict[str, Any] = {
        "cell_id": cell_id,
        "max_batch_size": batch_size,
        "max_wait_ms": wait_ms,
        "target_rps": rps,
        "port": port,
        "command": cmd,
        "server_cwd": str(args.server_cwd),
        "env_overrides": {
            k: env[k] for k in
            ("CUDA_VISIBLE_DEVICES", "PORT", "MAX_BATCH_SIZE", "MAX_WAIT_MS",
             "QUEUE_DEPTH", "PYTHONPATH")
        },
        "warmup_s": args.warmup,
        "duration_s": args.duration,
        "seed": args.seed,
        "repeats": 1,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
    }
    with ServerProcess(cmd, args.server_cwd, env, log_path) as server:
        try:
            record["ready_wait_s"] = round(await wait_ready(base_url, args.ready_timeout), 3)
        except Exception as exc:
            record["status"] = "server_not_ready"
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["server_log_tail"] = server.log_tail()
            return record
        result: RunResult = await run_open_loop(
            base_url, rps, args.duration, args.warmup, args.seed, timeout_s=args.timeout,
        )
        record["stats"] = summarize(result)
        record["status"] = "ok"
        record["server_alive_after"] = server.alive()
    record["wall_s"] = round(time.time() - started, 3)
    return record


async def run_sweep(args: argparse.Namespace) -> int:
    out_dir = Path(args.results_dir) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Charter rule 3 -- gate before anything is measured. No skip flag exists.
    quiet = require_quiet(out_dir / "box_quiet.json")

    cells_path = out_dir / "cells.jsonl"
    cells_path.write_text("", encoding="utf-8")
    cells: list[dict[str, Any]] = []
    combos = list(grid(args.batch_sizes, args.wait_ms, args.rps))
    print(f"[sweep] {len(combos)} cells -> {out_dir}", flush=True)
    for i, (b, w, r) in enumerate(combos, 1):
        print(f"[sweep] {i}/{len(combos)} batch={b} wait={w}ms rps={r}", flush=True)
        rec = await run_cell(batch_size=b, wait_ms=w, rps=r, args=args, out_dir=out_dir)
        cells.append(rec)
        with cells_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    summary = {
        "run_id": args.run_id,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "reproduce_command": (
            f"python -m loadgen.sweep --run-id {args.run_id} "
            f"--server-cmd {shlex.quote(args.server_cmd)} "
            f"--server-cwd {shlex.quote(str(args.server_cwd))} "
            f"--batch-sizes {','.join(str(b) for b in args.batch_sizes)} "
            f"--wait-ms {','.join(str(w) for w in args.wait_ms)} "
            f"--rps {','.join(str(r) for r in args.rps)} "
            f"--duration {args.duration} --warmup {args.warmup} --seed {args.seed} "
            f"--queue-depth {args.queue_depth}"
            f"{'' if args.server_flags else ' --no-server-flags'}"
        ),
        "target_under_test": args.target_label,
        "grid": {
            "max_batch_size": list(args.batch_sizes),
            "max_wait_ms": list(args.wait_ms),
            "target_rps": list(args.rps),
        },
        "load": {
            "mode": "open",
            "duration_s": args.duration,
            "warmup_s": args.warmup,
            "seed": args.seed,
            "repeats_per_cell": 1,
        },
        "box_quiet": quiet,
        "environment": environment_report(),
        "n_cells": len(cells),
        "n_cells_ok": sum(1 for c in cells if c.get("status") == "ok"),
        "cells": cells,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[sweep] wrote {out_dir / 'summary.json'}", flush=True)
    return 0 if summary["n_cells_ok"] == summary["n_cells"] else 1


def _ints(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(",") if x.strip())


def _floats(s: str) -> tuple[float, ...]:
    return tuple(float(x) for x in s.split(",") if x.strip())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="reward-serve batch-policy sweep")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    ap.add_argument("--server-cmd", default=DEFAULT_SERVER_CMD)
    ap.add_argument("--server-cwd", type=Path, default=REPO_ROOT)
    ap.add_argument(
        "--target-label", default="reward_serve",
        help="what was actually loaded, e.g. 'stub_server (no model)'",
    )
    ap.add_argument("--batch-sizes", type=_ints, default=DEFAULT_BATCH_SIZES)
    ap.add_argument("--wait-ms", type=_floats, default=DEFAULT_WAIT_MS)
    ap.add_argument("--rps", type=_floats, default=DEFAULT_RPS)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--warmup", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--queue-depth", type=int, default=256)
    ap.add_argument(
        "--server-flags", action=argparse.BooleanOptionalAction, default=True,
        help="append --port/--max-batch-size/--max-wait-ms/--queue-depth to the "
             "server command (contract section 4). --no-server-flags for targets "
             "that only read the environment.",
    )
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--ready-timeout", type=float, default=180.0)
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return asyncio.run(run_sweep(parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
