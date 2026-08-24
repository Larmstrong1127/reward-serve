"""Batch-policy sweep: restart the server per cell, load it, record everything.

Grid: MAX_BATCH_SIZE x MAX_WAIT_MS x offered load (open-loop rps).

Every REPEAT gets a FRESH server process with that cell's env, so nothing
inherits another run's warm caches, queue state or allocator behaviour. Each cell
writes one line to results/<run-id>/cells.jsonl the moment it finishes -- sessions
get killed, and a half-finished sweep should still have its completed cells on
disk.

Each cell runs `--repeats` times (default 3) with distinct seeds derived from the
base seed, and reports per-repeat statistics plus an aggregate carrying a seeded
bootstrap 95% confidence interval on p99 and the min/max across repeats. One run
per cell can state a number; it cannot say how far that number would move if you
ran it again.

Every measured request is persisted to results/<run-id>/raw/<cell_id>.jsonl, one
line per request, warm-up rows included and flagged. Charter rule 2 wants every
number to have an artifact; pre-computed order statistics are an assertion, and
the raw rows are what makes them checkable. `scripts/recompute_sweep.py`
regenerates the whole summary from those rows and nothing else.

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

from loadgen.generator import RequestRecord, RunResult, run_open_loop, wait_ready
from loadgen.stats import (
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_RESAMPLES,
    GENERATOR_ONLY_KEYS,
    SPAN_DERIVED_KEYS,
    aggregate_repeats,
    repeat_seed,
    summarize,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loadgen.contention import ContentionSampler  # noqa: E402
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


def write_raw_rows(
    path: Path,
    records: Sequence[RequestRecord],
    *,
    cell_id: str,
    repeat: int,
    seed: int,
    batch_size: int,
    wait_ms: float,
    rps: float,
    truncate: bool,
) -> int:
    """Append one line per request to `raw/<cell_id>.jsonl`. Returns rows written.

    Warm-up rows are written too, flagged `is_warmup`, so a reader can verify the
    exclusion instead of trusting it: recompute with and without them and watch
    the published number fall out of the first.

    Timestamps are written unrounded on purpose. Every published statistic is a
    function of these rows, so rounding here would put a floor under how exactly
    anyone can reproduce the numbers.

    The file is always created, even for a repeat that produced nothing -- a
    missing file and an empty file mean different things, and "the server never
    came up" is a result worth being able to see.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    context = {
        "cell_id": cell_id,
        "repeat": repeat,
        "seed": seed,
        "max_batch_size": batch_size,
        "max_wait_ms": wait_ms,
        "target_rps": rps,
    }
    with path.open("w" if truncate else "a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps({**context, **rec.to_json()}) + "\n")
    return len(records)


async def run_repeat(
    *,
    batch_size: int,
    wait_ms: float,
    rps: float,
    seed: int,
    repeat: int,
    args: argparse.Namespace,
    out_dir: Path,
    cell_id: str,
) -> tuple[dict[str, Any], list[RequestRecord]]:
    """One independent server start + warm-up + measured window.

    A repeat is not a re-slice of one run: the server process is new, so the
    allocator, the thread pool and the first forward pass are cold again exactly
    as they were for repeat 0, and the seed differs so the arrival process is a
    different draw from the same Poisson distribution.
    """
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
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
    log_path = out_dir / "server-logs" / f"{cell_id}_rep{repeat}.log"
    started = time.time()
    record: dict[str, Any] = {
        "repeat": repeat,
        "seed": seed,
        "port": port,
        "command": cmd,
        "server_cwd": str(args.server_cwd),
        "env_overrides": {
            k: env[k] for k in
            ("CUDA_VISIBLE_DEVICES", "PORT", "MAX_BATCH_SIZE", "MAX_WAIT_MS",
             "QUEUE_DEPTH", "PYTHONPATH")
        },
        "server_log": f"server-logs/{cell_id}_rep{repeat}.log",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
    }
    records: list[RequestRecord] = []
    with ServerProcess(cmd, args.server_cwd, env, log_path) as server:
        try:
            record["ready_wait_s"] = round(await wait_ready(base_url, args.ready_timeout), 3)
        except Exception as exc:
            record["status"] = "server_not_ready"
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["server_log_tail"] = server.log_tail()
            record["wall_s"] = round(time.time() - started, 3)
            return record, records
        result: RunResult = await run_open_loop(
            base_url, rps, args.duration, args.warmup, seed, timeout_s=args.timeout,
        )
        records = result.records
        record["stats"] = summarize(result)
        record["status"] = "ok"
        record["server_alive_after"] = server.alive()
    record["wall_s"] = round(time.time() - started, 3)
    return record, records


async def run_cell(
    *,
    batch_size: int,
    wait_ms: float,
    rps: float,
    args: argparse.Namespace,
    out_dir: Path,
    repeats: int | None = None,
    cell_id_prefix: str = "",
) -> dict[str, Any]:
    """Run every repeat of one grid cell and aggregate them with an interval.

    The cell-level `stats` are the aggregate (see `loadgen.stats.aggregate_repeats`):
    pooled where pooling is valid, per-repeat means where it is not, and a
    bootstrap 95% CI on p99. `per_repeat` keeps each repeat's own statistics so a
    disagreement between repeats stays visible instead of being averaged into a
    single confident-looking number.
    """
    n_repeats = args.repeats if repeats is None else repeats
    cell_id = f"{cell_id_prefix}b{batch_size}_w{int(wait_ms)}_r{rps:g}"
    raw_path = out_dir / "raw" / f"{cell_id}.jsonl"
    started = time.time()
    record: dict[str, Any] = {
        "cell_id": cell_id,
        "max_batch_size": batch_size,
        "max_wait_ms": wait_ms,
        "target_rps": rps,
        "warmup_s": args.warmup,
        "duration_s": args.duration,
        "base_seed": args.seed,
        "repeats": n_repeats,
        "raw_path": f"raw/{cell_id}.jsonl",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
    }
    if args.smoke:
        record["smoke"] = True
        record["not_a_measurement"] = True

    per_repeat: list[dict[str, Any]] = []
    all_records: list[list[RequestRecord]] = []
    # The quiet gate ran once, before the grid; this sampler records what else
    # the box did during THIS cell, so every cell carries its own evidence.
    with ContentionSampler() as sampler:
        for i in range(n_repeats):
            seed = repeat_seed(args.seed, i)
            rep_rec, rep_records = await run_repeat(
                batch_size=batch_size, wait_ms=wait_ms, rps=rps, seed=seed, repeat=i,
                args=args, out_dir=out_dir, cell_id=cell_id,
            )
            rep_rec["n_raw_rows"] = write_raw_rows(
                raw_path, rep_records, cell_id=cell_id, repeat=i, seed=seed,
                batch_size=batch_size, wait_ms=wait_ms, rps=rps, truncate=(i == 0),
            )
            per_repeat.append(rep_rec)
            all_records.append(rep_records)
    record["contention"] = sampler.report()

    record["per_repeat"] = per_repeat
    ok_idx = [i for i, r in enumerate(per_repeat) if r.get("status") == "ok"]
    record["n_repeats_ok"] = len(ok_idx)
    if not ok_idx:
        first = per_repeat[0] if per_repeat else {}
        record["status"] = str(first.get("status", "no_repeats"))
        record["error"] = first.get("error", "repeats=0")
        record["stats"] = {}
        record["wall_s"] = round(time.time() - started, 3)
        return record
    record["status"] = "ok" if len(ok_idx) == n_repeats else "partial"
    record["stats"] = aggregate_repeats(
        [all_records[i] for i in ok_idx],
        cell_id=cell_id,
        repeat_seeds=[int(per_repeat[i]["seed"]) for i in ok_idx],
        n_resamples=args.bootstrap_resamples,
    )
    record["wall_s"] = round(time.time() - started, 3)
    return record


async def preflight_batching(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    """Prove the server actually batches before spending hours measuring it.

    A batcher that forwards a backlog one item at a time still answers every
    request correctly and still reports `batch_size`, so the only way to catch
    it is to offer more load than one forward pass can absorb and check that the
    achieved batch size rose above 1. That bug was real in this server's history
    (fixed in bec147e). A sweep that ran without this check would produce a
    clean-looking grid measuring the wrong thing.
    """
    rec = await run_cell(
        batch_size=8, wait_ms=20.0, rps=args.preflight_rps, args=args, out_dir=out_dir,
        repeats=1, cell_id_prefix="preflight_",
    )
    stats = rec.get("stats", {})
    mean_batch = stats.get("mean_batch_size")
    check = {
        "cell": rec.get("cell_id"),
        "status": rec.get("status"),
        "offered_rps": args.preflight_rps,
        "mean_batch_size": mean_batch,
        "n_success": stats.get("n_success"),
        "requirement": "mean_batch_size > 1.0 at MAX_BATCH_SIZE=8 under offered load",
    }
    (out_dir / "preflight_batching.json").write_text(json.dumps(check, indent=2), encoding="utf-8")
    if rec.get("status") != "ok":
        raise SystemExit(f"preflight FAIL: server would not run: {json.dumps(check)}")
    if not stats.get("n_success"):
        raise SystemExit(f"preflight FAIL: no successful requests: {json.dumps(check)}")
    if mean_batch is None or mean_batch <= 1.0:
        raise SystemExit(
            "preflight FAIL: the server is not batching -- mean achieved batch size "
            f"{mean_batch} at MAX_BATCH_SIZE=8 under {args.preflight_rps} rps. "
            "Every number in this grid would describe an unbatched server. "
            f"{json.dumps(check)}"
        )
    print(f"[sweep] preflight ok: mean batch {mean_batch} at {args.preflight_rps} rps", flush=True)
    return check


def gate_box_quiet(args: argparse.Namespace, report_path: Path) -> dict[str, Any]:
    """Charter rule 3: contended numbers are never published.

    The gate can be stood down in exactly one situation -- a `--smoke` run, whose
    entire purpose is to exercise the code path in a few seconds on a box that is
    known to be busy, and whose output is stamped `not_a_measurement` in the
    summary and in every cell so it can never be quoted as a result. Asking for
    `--no-box-quiet` without `--smoke` is refused, loudly, here: that combination
    is the one that would produce a publishable-looking file from a contended
    box, which is the exact failure charter rule 3 exists to prevent.
    """
    if args.box_quiet:
        return require_quiet(report_path)
    if not args.smoke:
        raise SystemExit(
            "--no-box-quiet is only accepted together with --smoke. Charter rule 3: "
            "measurements need a quiet box and contended numbers are never published. "
            "A smoke run is not a measurement and is stamped as such; anything else "
            "must pass scripts/box_quiet.py."
        )
    report: dict[str, Any] = {
        "passed": None,
        "gate_skipped": True,
        "smoke": True,
        "not_a_measurement": True,
        "reason": (
            "--smoke --no-box-quiet: the quiet-box gate was NOT run. Nothing in this "
            "run directory is a measurement of anything. It exists to prove the sweep "
            "code path executes end to end and to serve as a test fixture."
        ),
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[sweep] SMOKE RUN -- quiet gate skipped, output is not a measurement", flush=True)
    return report


async def run_sweep(args: argparse.Namespace) -> int:
    out_dir = Path(args.results_dir) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Charter rule 3 -- gate before anything is measured.
    quiet = gate_box_quiet(args, out_dir / "box_quiet.json")

    preflight = await preflight_batching(args, out_dir)

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

    summary: dict[str, Any] = {
        "run_id": args.run_id,
        "smoke": bool(args.smoke),
        "not_a_measurement": bool(args.smoke),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "reproduce_command": (
            f"python -m loadgen.sweep --run-id {args.run_id} "
            f"--server-cmd {shlex.quote(args.server_cmd)} "
            f"--server-cwd {shlex.quote(str(args.server_cwd))} "
            f"--batch-sizes {','.join(str(b) for b in args.batch_sizes)} "
            f"--wait-ms {','.join(str(w) for w in args.wait_ms)} "
            f"--rps {','.join(str(r) for r in args.rps)} "
            f"--duration {args.duration} --warmup {args.warmup} --seed {args.seed} "
            f"--queue-depth {args.queue_depth} --repeats {args.repeats} "
            f"--bootstrap-resamples {args.bootstrap_resamples}"
            f"{'' if args.server_flags else ' --no-server-flags'}"
            f"{' --smoke' if args.smoke else ''}"
            f"{'' if args.box_quiet else ' --no-box-quiet'}"
        ),
        "recompute_command": (
            f"python scripts/recompute_sweep.py {args.results_dir / args.run_id}"
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
            "repeats_per_cell": args.repeats,
            "repeat_seeds": [repeat_seed(args.seed, i) for i in range(args.repeats)],
        },
        "uncertainty": {
            "repeats_per_cell": args.repeats,
            "ci_percentile": 99.0,
            "ci_method": "percentile bootstrap over pooled per-request latencies",
            "ci_resamples": args.bootstrap_resamples,
            "ci_confidence": BOOTSTRAP_CONFIDENCE,
            "ci_seed_derivation": "sha256(cell_id + sorted repeat seeds), see "
                                  "loadgen.stats.bootstrap_seed_for",
            "span_derived_keys_are_repeat_means": list(SPAN_DERIVED_KEYS),
            "keys_not_derivable_from_raw": sorted(GENERATOR_ONLY_KEYS),
        },
        "raw": {
            "dir": "raw/",
            "one_file_per_cell": "raw/<cell_id>.jsonl",
            "one_line_per_request": True,
            "warmup_rows_present": True,
            "warmup_rows_flagged": "is_warmup",
            "note": "every statistic in this file is a pure function of these rows; "
                    "regenerate them with scripts/recompute_sweep.py",
        },
        "box_quiet": quiet,
        "preflight_batching": preflight,
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


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.repeats < 1:
        raise SystemExit(f"--repeats must be >= 1, got {args.repeats}")
    if args.bootstrap_resamples < 1:
        raise SystemExit(
            f"--bootstrap-resamples must be >= 1, got {args.bootstrap_resamples}"
        )
    if not args.box_quiet and not args.smoke:
        raise SystemExit(
            "--no-box-quiet is only accepted together with --smoke (charter rule 3)."
        )
    return args


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
    ap.add_argument(
        "--repeats", type=int, default=3,
        help="independent server start + warm-up + measured window per cell. "
             "One repeat measures a cell; it cannot say how far the number would "
             "move if you ran it again.",
    )
    ap.add_argument(
        "--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES,
        help="resamples for the bootstrap CI on p99 (seeded, so deterministic)",
    )
    ap.add_argument("--queue-depth", type=int, default=256)
    ap.add_argument(
        "--preflight-rps", type=float, default=4.0,
        help="offered load for the pre-grid check that the server really batches",
    )
    ap.add_argument(
        "--server-flags", action=argparse.BooleanOptionalAction, default=True,
        help="append --port/--max-batch-size/--max-wait-ms/--queue-depth to the "
             "server command (contract section 4). --no-server-flags for targets "
             "that only read the environment.",
    )
    ap.add_argument(
        "--smoke", action="store_true",
        help="this run is NOT a measurement. Stamps 'smoke' and 'not_a_measurement' "
             "into the summary and every cell. Use for exercising the code path and "
             "for test fixtures, never for numbers.",
    )
    ap.add_argument(
        "--box-quiet", action=argparse.BooleanOptionalAction, default=True,
        help="run the charter rule 3 quiet-box gate before measuring. "
             "--no-box-quiet is REFUSED unless --smoke is also passed.",
    )
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--ready-timeout", type=float, default=180.0)
    return validate_args(ap.parse_args(argv))


def main(argv: Sequence[str] | None = None) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return asyncio.run(run_sweep(parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
