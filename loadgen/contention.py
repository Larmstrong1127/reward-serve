"""Per-cell contention sampling for the sweep.

The quiet-box gate runs ONCE, before the grid. A full three-repeat grid runs
for hours, and nothing stops a render job, an agent, or a browser from waking
up during cell 40 of 72 -- which would contaminate exactly the cells nobody
re-checks. This sampler closes that gap: every cell carries its own record of
what else the machine was doing while that cell was measured, so a reader can
judge each cell instead of trusting the single gate report at minute zero.

It deliberately does NOT re-run the 30-second gate per cell (that would add
~36 minutes to a 72-cell grid and pause load mid-flight). It samples cheaply
in a background thread while the cell runs:

* total CPU percent (the cell's own load is included -- recorded, not judged),
* GPU memory in use and mean utilisation, via nvidia-smi when present,
* CPU share of FOREIGN python processes: any python.exe that is neither this
  process nor one of its descendants (the servers and workers the sweep itself
  started are ours; a fleet render or another agent is not).

The verdict is conservative and only about what the cell CANNOT be blamed
for: ``contended`` is True when GPU memory exceeded the gate's 2 GiB bar at
any sample, or any foreign python process averaged above 20% of one core.
Total CPU is reported raw and never judged, because the cell is supposed to
be busy.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from typing import Any

GPU_MEM_MAX_MIB = 2048
FOREIGN_PY_MAX_PCT = 20.0
SAMPLE_INTERVAL_S = 2.0


def _gpu_sample() -> tuple[int, int] | None:
    """(memory_used_mib, utilization_pct), or None when nvidia-smi is absent."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip().splitlines()[0]
        mem, util = (int(x.strip()) for x in out.split(","))
        return mem, util
    except Exception:
        return None


class ContentionSampler:
    """Samples the box in a daemon thread while a cell runs.

    Use as a context manager around the cell; read ``report()`` afterwards.
    Sampling failures never propagate into the measurement -- a sampler that
    can crash the sweep would be worse than no sampler.
    """

    def __init__(self, interval_s: float = SAMPLE_INTERVAL_S) -> None:
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cpu_total: list[float] = []
        self._gpu_mem: list[int] = []
        self._gpu_util: list[int] = []
        self._foreign: dict[int, list[float]] = {}
        self._foreign_names: dict[int, str] = {}
        self._error: str | None = None

    # -- internals ---------------------------------------------------------
    def _own_tree(self) -> set[int]:
        import psutil  # type: ignore[import-untyped]

        me = psutil.Process()
        pids = {me.pid}
        try:
            pids.update(c.pid for c in me.children(recursive=True))
        except Exception:
            pass
        return pids

    def _loop(self) -> None:
        try:
            import psutil

            ncpu = psutil.cpu_count() or 1
            # First sample immediately: enumerating every process to find
            # foreign pythons can take longer than a whole short cell, so the
            # cheap total-CPU sample must never wait behind it.
            self._cpu_total.append(psutil.cpu_percent(None))
            ours = self._own_tree()
            foreign: dict[int, Any] = {}
            for p in psutil.process_iter(["name"]):
                name = (p.info["name"] or "").lower()
                if name.startswith("python") and p.pid not in ours:
                    foreign[p.pid] = p
                    self._foreign_names[p.pid] = name
                    try:
                        p.cpu_percent(None)  # prime the counter
                    except Exception:
                        pass
            while not self._stop.wait(self.interval_s):
                self._cpu_total.append(psutil.cpu_percent(None))
                g = _gpu_sample()
                if g is not None:
                    self._gpu_mem.append(g[0])
                    self._gpu_util.append(g[1])
                # Refresh ours: the sweep spawns a new server per repeat.
                ours = self._own_tree()
                for pid, p in list(foreign.items()):
                    if pid in ours:
                        foreign.pop(pid)
                        continue
                    try:
                        self._foreign.setdefault(pid, []).append(p.cpu_percent(None) / ncpu)
                    except Exception:
                        foreign.pop(pid, None)
        except Exception as exc:  # noqa: BLE001 - never fail the cell
            self._error = f"{type(exc).__name__}: {exc}"

    # -- context manager ---------------------------------------------------
    def __enter__(self) -> ContentionSampler:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 2)

    # -- results -----------------------------------------------------------
    def report(self) -> dict[str, Any]:
        hot_foreign = {
            f"{pid}:{self._foreign_names.get(pid, '?')}": round(sum(v) / len(v), 2)
            for pid, v in self._foreign.items()
            if v and sum(v) / len(v) > FOREIGN_PY_MAX_PCT
        }
        gpu_mem_max = max(self._gpu_mem) if self._gpu_mem else None
        contended = bool(hot_foreign) or (
            gpu_mem_max is not None and gpu_mem_max >= GPU_MEM_MAX_MIB
        )
        return {
            "samples": len(self._cpu_total),
            "interval_s": self.interval_s,
            "cpu_total_mean_pct": (
                round(sum(self._cpu_total) / len(self._cpu_total), 2)
                if self._cpu_total
                else None
            ),
            "cpu_total_max_pct": max(self._cpu_total) if self._cpu_total else None,
            "gpu_mem_used_max_mib": gpu_mem_max,
            "gpu_util_mean_pct": (
                round(sum(self._gpu_util) / len(self._gpu_util), 2)
                if self._gpu_util
                else None
            ),
            "foreign_python_hot": hot_foreign,
            "contended": contended,
            "sampler_error": self._error,
            "verdict_basis": (
                "contended = any foreign python averaging > "
                f"{FOREIGN_PY_MAX_PCT}% of one core, or GPU memory >= "
                f"{GPU_MEM_MAX_MIB} MiB at any sample. Total CPU is recorded, "
                "never judged: the cell is supposed to be busy."
            ),
        }
