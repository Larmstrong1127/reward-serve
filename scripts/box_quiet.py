"""Refuse to let a measurement run on a contended box.

PASS requires, over a 30-second window: mean CPU < 15%; GPU memory in use
< 2 GiB AND mean GPU utilisation < 20% (if nvidia-smi exists); and no other
python process averaging > 20% CPU.
Writes a JSON report and exits 0 on PASS, 1 on FAIL. Sweeps must import
`require_quiet()` and store the returned report next to their results.
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

WINDOW_S = 30
CPU_MAX = 15.0
# GPU: the signal that a real job is running is MEMORY held (a render or a
# model holds gigabytes), backed by MEAN utilisation. The original check was a
# raw max-utilisation ceiling of 10%, which was measured to sit inside the idle
# noise band on this box -- desktop compositing spikes 10-21% with nothing
# running (results/box_quiet_calibration.json) -- so it could never pass while
# anyone was logged in. Recalibrated 2026-08-18: idle reads ~440 MiB, 7-8% util.
GPU_MEM_MAX_MIB = 2048
GPU_UTIL_MEAN_MAX = 20.0
OTHER_PY_MAX = 20.0


def _gpu_sample() -> tuple[int, int] | None:
    """(memory_used_mib, utilization_pct), or None if nvidia-smi is absent."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip().splitlines()[0]
        mem, util = (int(x.strip()) for x in out.split(","))
        return mem, util
    except Exception:
        return None


def measure(window_s: int = WINDOW_S) -> dict[str, Any]:
    import psutil  # type: ignore

    me = os.getpid()
    cpu_samples: list[float] = []
    gpu_mem_samples: list[int] = []
    gpu_util_samples: list[int] = []
    other_py: dict[int, list[float]] = {}
    procs = {
        p.pid: p
        for p in psutil.process_iter(["name"])
        if (p.info["name"] or "").lower().startswith("python") and p.pid != me
    }
    for p in procs.values():
        with contextlib.suppress(Exception):
            p.cpu_percent(None)
    end = time.time() + window_s
    while time.time() < end:
        cpu_samples.append(psutil.cpu_percent(interval=1))
        g = _gpu_sample()
        if g is not None:
            gpu_mem_samples.append(g[0])
            gpu_util_samples.append(g[1])
        for pid, p in procs.items():
            with contextlib.suppress(Exception):
                other_py.setdefault(pid, []).append(p.cpu_percent(None) / psutil.cpu_count())
    cpu_mean = sum(cpu_samples) / len(cpu_samples)
    gpu_mem_max = max(gpu_mem_samples) if gpu_mem_samples else None
    gpu_util_mean = (
        round(sum(gpu_util_samples) / len(gpu_util_samples), 2) if gpu_util_samples else None
    )
    gpu_util_max = max(gpu_util_samples) if gpu_util_samples else None
    hot_py = {
        pid: sum(v) / len(v)
        for pid, v in other_py.items()
        if v and sum(v) / len(v) > OTHER_PY_MAX
    }
    gpu_quiet = gpu_mem_max is None or (
        gpu_mem_max < GPU_MEM_MAX_MIB
        and gpu_util_mean is not None
        and gpu_util_mean < GPU_UTIL_MEAN_MAX
    )
    passed = cpu_mean < CPU_MAX and gpu_quiet and not hot_py
    return {
        "passed": passed,
        "window_s": window_s,
        "cpu_mean_pct": round(cpu_mean, 2),
        "gpu_mem_used_max_mib": gpu_mem_max,
        "gpu_util_mean_pct": gpu_util_mean,
        "gpu_util_max_pct": gpu_util_max,
        "hot_python_pids": hot_py,
        "thresholds": {
            "cpu_max": CPU_MAX,
            "gpu_mem_max_mib": GPU_MEM_MAX_MIB,
            "gpu_util_mean_max": GPU_UTIL_MEAN_MAX,
            "other_python_max": OTHER_PY_MAX,
        },
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def require_quiet(report_path: Path | None = None) -> dict[str, Any]:
    report = measure()
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(f"box_quiet FAIL: {json.dumps(report)}")
    return report


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    r = require_quiet(out)
    print(json.dumps(r, indent=2))
