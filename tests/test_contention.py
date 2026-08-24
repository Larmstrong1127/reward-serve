"""The per-cell contention sampler: cheap, honest, and unable to hurt a cell."""

from __future__ import annotations

import time

from loadgen.contention import ContentionSampler


def test_sampler_records_and_reports() -> None:
    with ContentionSampler(interval_s=0.05) as s:
        time.sleep(1.0)
    r = s.report()
    # process enumeration on a busy Windows box can eat most of a short window;
    # the first sample is taken before it, so at least one is guaranteed.
    assert r["samples"] >= 1
    assert r["cpu_total_mean_pct"] is not None
    assert "contended" in r and isinstance(r["contended"], bool)
    assert r["sampler_error"] is None


def test_total_cpu_is_never_the_verdict() -> None:
    """A busy cell must not flag itself: only foreign python or GPU memory judge."""
    s = ContentionSampler(interval_s=0.05)
    s._cpu_total = [95.0, 99.0, 97.0]  # the cell itself, working hard
    r = s.report()
    assert r["contended"] is False
    assert r["cpu_total_max_pct"] == 99.0


def test_foreign_hot_python_flags_contention() -> None:
    s = ContentionSampler(interval_s=0.05)
    s._foreign = {4242: [35.0, 40.0, 30.0]}
    s._foreign_names = {4242: "python.exe"}
    r = s.report()
    assert r["contended"] is True
    assert "4242:python.exe" in r["foreign_python_hot"]


def test_gpu_memory_flags_contention() -> None:
    s = ContentionSampler(interval_s=0.05)
    s._gpu_mem = [400, 23000]  # a render woke up mid-cell
    s._gpu_util = [5, 98]
    r = s.report()
    assert r["contended"] is True
    assert r["gpu_mem_used_max_mib"] == 23000


def test_quiet_foreign_python_does_not_flag() -> None:
    s = ContentionSampler(interval_s=0.05)
    s._foreign = {4242: [1.0, 0.5, 2.0]}
    s._foreign_names = {4242: "python.exe"}
    s._gpu_mem = [400, 450]
    r = s.report()
    assert r["contended"] is False
    assert r["foreign_python_hot"] == {}


def test_sampler_failure_is_reported_not_raised() -> None:
    """A sampler that can crash the sweep is worse than no sampler."""
    s = ContentionSampler(interval_s=0.05)
    s._error = "RuntimeError: synthetic"
    r = s.report()
    assert r["sampler_error"] == "RuntimeError: synthetic"
