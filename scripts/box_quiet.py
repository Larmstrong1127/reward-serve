"""Refuse to let a measurement run on a contended box.

PASS requires, over a 30-second window: mean CPU < 15%, GPU utilization < 10%
(if nvidia-smi exists), and no other python process averaging > 20% CPU.
Writes a JSON report and exits 0 on PASS, 1 on FAIL. Sweeps must import
`require_quiet()` and store the returned report next to their results.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

WINDOW_S = 30
CPU_MAX = 15.0
GPU_MAX = 10
OTHER_PY_MAX = 20.0


def _gpu_util() -> int | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip().splitlines()[0]
        return int(out)
    except Exception:
        return None


def measure(window_s: int = WINDOW_S) -> dict:
    import psutil  # type: ignore

    me = os.getpid()
    cpu_samples: list[float] = []
    gpu_samples: list[int] = []
    other_py: dict[int, list[float]] = {}
    procs = {p.pid: p for p in psutil.process_iter(["name"]) if (p.info["name"] or "").lower().startswith("python") and p.pid != me}
    for p in procs.values():
        try:
            p.cpu_percent(None)
        except Exception:
            pass
    end = time.time() + window_s
    while time.time() < end:
        cpu_samples.append(psutil.cpu_percent(interval=1))
        g = _gpu_util()
        if g is not None:
            gpu_samples.append(g)
        for pid, p in procs.items():
            try:
                other_py.setdefault(pid, []).append(p.cpu_percent(None) / psutil.cpu_count())
            except Exception:
                pass
    cpu_mean = sum(cpu_samples) / len(cpu_samples)
    gpu_max = max(gpu_samples) if gpu_samples else None
    hot_py = {pid: sum(v) / len(v) for pid, v in other_py.items() if v and sum(v) / len(v) > OTHER_PY_MAX}
    passed = cpu_mean < CPU_MAX and (gpu_max is None or gpu_max < GPU_MAX) and not hot_py
    return {
        "passed": passed,
        "window_s": window_s,
        "cpu_mean_pct": round(cpu_mean, 2),
        "gpu_util_max_pct": gpu_max,
        "hot_python_pids": hot_py,
        "thresholds": {"cpu_max": CPU_MAX, "gpu_max": GPU_MAX, "other_python_max": OTHER_PY_MAX},
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def require_quiet(report_path: Path | None = None) -> dict:
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
