"""Measure what box_quiet actually reads on an idle desktop.

Written after the observability dry run was refused fourteen times in a row.
The obvious explanation -- other work owning the machine -- stopped being true
partway through: the GPU render job ended, GPU memory fell to 458 MiB, and the
gate kept refusing anyway. Before adjusting anything, measure the ruler.

The question this answers is narrow: over repeated 30-second windows on a box
with nothing running but the desktop, what does `box_quiet.measure()` report,
and how does that compare to its own thresholds? A gate whose pass condition
the idle environment cannot meet is not gating a measurement, it is blocking
one, and the difference is only visible with the spread in hand.

This script needs no quiet box, because the box's noise is the thing it
measures. It changes no threshold and decides nothing; it produces the evidence
someone else needs in order to decide.

Run:

    cd D:\ClaudeProjects\reward-serve-wt-observability
    D:\ClaudeProjects\reward-serve\.venv\Scripts\python.exe scripts/box_quiet_calibration.py 6
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.box_quiet import CPU_MAX, GPU_UTIL_MEAN_MAX, measure  # noqa: E402


def main(n_samples: int) -> int:
    samples = []
    for i in range(n_samples):
        report = measure()
        report["sample"] = i + 1
        samples.append(report)
        print(
            f"sample {i + 1}/{n_samples}: "
            f"cpu_mean={report['cpu_mean_pct']}% "
            f"gpu_util_max={report['gpu_util_max_pct']}% "
            f"passed={report['passed']}"
        )

    cpu = [s["cpu_mean_pct"] for s in samples]
    gpu = [s["gpu_util_max_pct"] for s in samples if s["gpu_util_max_pct"] is not None]

    out = {
        "what_this_measures": (
            "box_quiet.measure() repeated on an otherwise idle desktop, to show "
            "what the gate reads when nothing is deliberately running. Compare "
            "the observed range against the thresholds the gate enforces."
        ),
        "n_samples": len(samples),
        "window_s_each": samples[0]["window_s"],
        "thresholds": samples[0]["thresholds"],
        "cpu_mean_pct": {
            "min": min(cpu),
            "median": round(statistics.median(cpu), 2),
            "max": max(cpu),
            "threshold": CPU_MAX,
            "samples_under_threshold": sum(1 for v in cpu if v < CPU_MAX),
        },
        "gpu_util_max_pct": {
            "min": min(gpu) if gpu else None,
            "median": round(statistics.median(gpu), 2) if gpu else None,
            "max": max(gpu) if gpu else None,
            "threshold": GPU_UTIL_MEAN_MAX,
            "samples_under_threshold": sum(1 for v in gpu if v < GPU_UTIL_MEAN_MAX),
            "note": (
                "This is a MAX over the window, not a mean. One frame of desktop "
                "compositing spikes it, so a browser or a chat client left open "
                "is enough to hold it above a 10% ceiling for the whole 30s."
            ),
        },
        "passed": sum(1 for s in samples if s["passed"]),
        "samples": samples,
        "command": f"python scripts/box_quiet_calibration.py {n_samples}",
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    path = results / "box_quiet_calibration.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("cpu_mean_pct", "gpu_util_max_pct", "passed")}, indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 6))
