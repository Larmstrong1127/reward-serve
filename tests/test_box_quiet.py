"""The quiet-box gate must survive a machine that has no GPU at all.

`scripts/box_quiet.py` is the charter's rule-3 gate. It is written for the
owner's box, which has an nvidia-smi. A CI runner does not: `shutil.which`
returns None there. The failure this file rules out is the gate crashing (or,
worse, blanket-passing) on a machine where the GPU question cannot be asked --
so these tests pin both halves: absence of a GPU is not an error, and it is
not an excuse either.

Everything here is monkeypatched: no real 30-second sampling window runs.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import psutil
import pytest

import scripts.box_quiet as box_quiet


def _quiet_psutil(monkeypatch: pytest.MonkeyPatch, cpu_pct: float) -> None:
    """A box with `cpu_pct` load, no other python processes, and no sleeping."""
    monkeypatch.setattr(psutil, "cpu_percent", lambda interval=None: cpu_pct)
    monkeypatch.setattr(psutil, "cpu_count", lambda *a, **k: 8)
    monkeypatch.setattr(psutil, "process_iter", lambda *a, **k: iter(()))


# --- nvidia-smi absence -------------------------------------------------------


def test_gpu_sample_is_none_when_nvidia_smi_is_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CI runner has no nvidia-smi on PATH. That must be a None, not a crash."""
    monkeypatch.setattr(box_quiet.shutil, "which", lambda _name: None)
    assert box_quiet._gpu_sample() is None


def test_gpu_sample_is_none_when_nvidia_smi_exists_but_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stub nvidia-smi, a driver mismatch, a timeout: all read as 'no answer'."""
    monkeypatch.setattr(box_quiet.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")

    def boom(*_a: Any, **_k: Any) -> Any:
        raise subprocess.CalledProcessError(9, "nvidia-smi")

    monkeypatch.setattr(box_quiet.subprocess, "run", boom)
    assert box_quiet._gpu_sample() is None


def test_measure_runs_and_passes_on_a_quiet_machine_with_no_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(box_quiet, "_gpu_sample", lambda: None)
    _quiet_psutil(monkeypatch, cpu_pct=3.0)

    report = box_quiet.measure(window_s=1)

    assert report["passed"] is True
    assert report["gpu_mem_used_max_mib"] is None
    assert report["gpu_util_mean_pct"] is None
    assert report["gpu_util_max_pct"] is None
    assert report["cpu_mean_pct"] == 3.0
    assert json.dumps(report)  # the report is serialisable as written


def test_a_missing_gpu_does_not_excuse_a_busy_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Degrading gracefully must not mean degrading into a rubber stamp."""
    monkeypatch.setattr(box_quiet, "_gpu_sample", lambda: None)
    _quiet_psutil(monkeypatch, cpu_pct=91.0)

    assert box_quiet.measure(window_s=1)["passed"] is False


# --- the GPU is present -------------------------------------------------------


def test_a_loaded_gpu_fails_even_when_the_cpu_is_idle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(box_quiet, "_gpu_sample", lambda: (18_000, 97))
    _quiet_psutil(monkeypatch, cpu_pct=2.0)

    report = box_quiet.measure(window_s=1)
    assert report["passed"] is False
    assert report["gpu_mem_used_max_mib"] == 18_000


def test_an_idle_gpu_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Idle on the owner's box reads ~440 MiB and 7-8% util."""
    monkeypatch.setattr(box_quiet, "_gpu_sample", lambda: (440, 8))
    _quiet_psutil(monkeypatch, cpu_pct=2.0)

    assert box_quiet.measure(window_s=1)["passed"] is True


# --- the gate ------------------------------------------------------------------


def test_require_quiet_writes_the_report_and_exits_when_the_box_is_busy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A refusal still has to leave the evidence behind it."""
    monkeypatch.setattr(
        box_quiet, "measure", lambda *a, **k: {"passed": False, "cpu_mean_pct": 80.0}
    )
    report = tmp_path / "nested" / "box_quiet.json"

    with pytest.raises(SystemExit):
        box_quiet.require_quiet(report)

    assert json.loads(report.read_text(encoding="utf-8"))["passed"] is False


def test_require_quiet_returns_the_report_when_the_box_is_quiet(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        box_quiet, "measure", lambda *a, **k: {"passed": True, "cpu_mean_pct": 1.0}
    )
    report = tmp_path / "box_quiet.json"

    assert box_quiet.require_quiet(report)["passed"] is True
    assert report.exists()
