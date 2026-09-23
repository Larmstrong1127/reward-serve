"""Claude Code PreToolUse hook: refuse a measurement command on a busy box.

The sweep proved the gap this closes. `scripts/box_quiet.py` runs once, at
minute zero, and `loadgen/contention.py` records what happened per cell --
but nothing stopped a SECOND agent (or a ghost interpreter, which happened
twice) from launching another measurement while the first was still running.
Both runs then produce numbers that look exactly like clean ones.

This hook sits at the harness boundary, where the instruction "only measure
when the box is quiet" becomes enforceable instead of aspirational. It is the
Claude Code equivalent of an Omnigent `tool_call` policy: read the tool input
on stdin, answer ALLOW / ASK / DENY on stdout.

Decision table (a measurement command is one matching MEASUREMENT_RE):

* another measurement process is already running          -> DENY
* GPU memory in use >= GPU_MEM_MAX_MIB (a render is up)    -> DENY
* a foreign python averages > FOREIGN_PY_MAX_PCT of a core -> ASK
* otherwise                                                -> ALLOW, and the
  sample is handed back as additionalContext so it can be quoted next to the
  number it gated.

Anything that is not a measurement command exits silently in a few
milliseconds without importing psutil. `--smoke` runs are allowed through:
they exist precisely to be run on a busy box. Every failure inside the hook
itself is fail-OPEN with the error in the reason -- a gate that can wedge the
agent is worse than no gate, and the failure is at least visible.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from typing import Any

MEASUREMENT_RE = re.compile(
    r"loadgen[./\\]sweep|loadgen\.sweep|run_latency|run_accuracy|box_quiet\.py"
    r"|pytest\b[^\n]*bench|record_demo\.py"
    # event-ledger (the sibling Java pipeline measured with this same discipline)
    r"|el-ingest\s+replay|el-bench|el_bench|loadgen[./\\]service",
    re.I,
)
SMOKE_RE = re.compile(r"--smoke\b")
GPU_MEM_MAX_MIB = 2048
FOREIGN_PY_MAX_PCT = 20.0
SAMPLE_S = 3.0
SAMPLE_INTERVAL_S = 0.5

Sampler = Callable[[], dict[str, Any]]


def is_measurement(command: str) -> bool:
    return bool(MEASUREMENT_RE.search(command or ""))


def _gpu_sample() -> tuple[int, int] | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = (
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            .stdout.strip()
            .splitlines()[0]
        )
        mem, util = (int(x.strip()) for x in out.split(","))
        return mem, util
    except Exception:
        return None


def _own_tree() -> set[int]:
    import psutil  # type: ignore[import-untyped]

    me = psutil.Process()
    pids = {me.pid}
    with contextlib.suppress(Exception):
        pids.update(p.pid for p in me.parents())
    with contextlib.suppress(Exception):
        pids.update(c.pid for c in me.children(recursive=True))
    return pids


def sample_box(window_s: float = SAMPLE_S, interval_s: float = SAMPLE_INTERVAL_S) -> dict[str, Any]:
    """A short live sample: in-flight measurements, GPU memory, foreign pythons.

    Deliberately a few seconds, not the gate's 30: a hook that blocks every
    shell command for half a minute would be routed around within a day.
    """
    import psutil

    ours = _own_tree()
    ncpu = psutil.cpu_count() or 1
    inflight: list[str] = []
    foreign: dict[int, Any] = {}
    names: dict[int, str] = {}
    for p in psutil.process_iter(["name", "cmdline"]):
        name = (p.info["name"] or "").lower()
        if not name.startswith("python"):
            continue
        cmd = " ".join(p.info["cmdline"] or [])
        if is_measurement(cmd) and p.pid not in ours:
            inflight.append(f"{p.pid}: {cmd[:160]}")
        if p.pid not in ours:
            foreign[p.pid] = p
            names[p.pid] = name
            with contextlib.suppress(Exception):
                p.cpu_percent(None)
    gpu_mem: list[int] = []
    gpu_util: list[int] = []
    shares: dict[int, list[float]] = {}
    end = time.monotonic() + window_s
    while True:
        g = _gpu_sample()
        if g is not None:
            gpu_mem.append(g[0])
            gpu_util.append(g[1])
        for pid, p in list(foreign.items()):
            try:
                shares.setdefault(pid, []).append(p.cpu_percent(None) / ncpu)
            except Exception:
                foreign.pop(pid, None)
        if time.monotonic() >= end:
            break
        time.sleep(interval_s)
    hot = {
        f"{pid}:{names.get(pid, '?')}": round(sum(v) / len(v), 2)
        for pid, v in shares.items()
        if v and sum(v) / len(v) > FOREIGN_PY_MAX_PCT
    }
    return {
        "inflight_measurements": inflight,
        "gpu_mem_used_max_mib": max(gpu_mem) if gpu_mem else None,
        "gpu_util_mean_pct": round(sum(gpu_util) / len(gpu_util), 2) if gpu_util else None,
        "foreign_python_hot": hot,
        "window_s": window_s,
        "sampled_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def decide(payload: dict[str, Any], sampler: Sampler = sample_box) -> dict[str, Any] | None:
    """Pure decision: hook payload in, Claude Code hook JSON (or None) out.

    None means "nothing to say" -- the command is not a measurement, or is a
    smoke run -- and the hook prints nothing.
    """
    if payload.get("tool_name") != "Bash":
        return None
    command = str((payload.get("tool_input") or {}).get("command") or "")
    if not is_measurement(command) or SMOKE_RE.search(command):
        return None
    try:
        s = sampler()
    except Exception as exc:  # noqa: BLE001 - fail open, visibly
        return _out(
            "allow",
            f"measurement_gate could not sample the box ({type(exc).__name__}: {exc}); "
            "allowing. Treat the result as UNGATED.",
        )
    if s["inflight_measurements"]:
        return _out(
            "deny",
            "measurement_gate: another measurement is already running on this box -- "
            "two concurrent runs contaminate each other and both look clean. "
            "Wait for it or kill it deliberately, then retry.\n  "
            + "\n  ".join(s["inflight_measurements"]),
            s,
        )
    mem = s["gpu_mem_used_max_mib"]
    if mem is not None and mem >= GPU_MEM_MAX_MIB:
        return _out(
            "deny",
            f"measurement_gate: GPU holds {mem} MiB (>= {GPU_MEM_MAX_MIB}); a render or "
            "model is up. This is the exact condition that contaminated the first sweep.",
            s,
        )
    if s["foreign_python_hot"]:
        return _out(
            "ask",
            "measurement_gate: foreign python process(es) busy -- "
            f"{s['foreign_python_hot']}. Confirm the box is really yours.",
            s,
        )
    return _out(
        "allow",
        "measurement_gate: box sampled quiet; quote this sample next to the number.",
        s,
    )


def _out(decision: str, reason: str, sample: dict[str, Any] | None = None) -> dict[str, Any]:
    hso: dict[str, Any] = {
        "hookEventName": "PreToolUse",
        "permissionDecision": decision,
        "permissionDecisionReason": reason,
    }
    if sample is not None:
        hso["additionalContext"] = "measurement_gate sample: " + json.dumps(sample, sort_keys=True)
    return {"hookSpecificOutput": hso}


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except Exception as exc:  # noqa: BLE001
        print(json.dumps(_out("allow", f"measurement_gate: unreadable payload ({exc}); allowing")))
        return 0
    if os.environ.get("MEASUREMENT_GATE_DISABLED") == "1":
        return 0
    result = decide(payload)
    if result is not None:
        print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
