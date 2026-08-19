"""The plotter must read summary.json and nothing else, and must not invent cells."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import plot_sweep  # noqa: E402


def _cell(batch: int, wait: float, rps: float, *, p99: float, thr: float,
          status: str = "ok", n_success: int = 100) -> dict[str, Any]:
    return {
        "cell_id": f"b{batch}_w{int(wait)}_r{rps:g}",
        "max_batch_size": batch,
        "max_wait_ms": wait,
        "target_rps": rps,
        "status": status,
        "stats": {
            "latency_p99_ms": p99,
            "throughput_rps": thr,
            "n_success": n_success,
            "mean_batch_size": float(batch),
        },
    }


def _summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    return {"run_id": "synthetic", "cells": cells}


@pytest.fixture
def summary_file(tmp_path: Path) -> Path:
    cells = []
    for rps in (1.0, 2.0):
        for wait in (0.0, 20.0):
            for i, batch in enumerate((1, 2, 4, 8)):
                cells.append(_cell(batch, wait, rps, p99=100.0 * (i + 1), thr=5.0 - i * 0.2))
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(_summary(cells)), encoding="utf-8")
    return path


def test_plot_all_writes_the_three_pngs(summary_file: Path) -> None:
    written = plot_sweep.plot_all(summary_file)
    names = {p.name for p in written}
    assert names == {"p99_vs_batch.png", "throughput_vs_batch.png", "knee.png"}
    for p in written:
        assert p.exists() and p.stat().st_size > 1000, f"{p} looks empty"
        assert p.parent == summary_file.parent, "plots must land beside their summary"


def test_failed_and_empty_cells_are_dropped_not_plotted_as_zero() -> None:
    cells = [
        _cell(1, 0.0, 1.0, p99=100.0, thr=5.0),
        _cell(2, 0.0, 1.0, p99=0.0, thr=0.0, status="server_not_ready"),
        _cell(4, 0.0, 1.0, p99=200.0, thr=4.0, n_success=0),
    ]
    kept, skipped = plot_sweep.load_cells(_summary(cells))
    assert [c["max_batch_size"] for c in kept] == [1]
    assert len(skipped) == 2
    assert any("server_not_ready" in s for s in skipped)
    assert any("no successful requests" in s for s in skipped)


def test_missing_p99_is_dropped() -> None:
    cell = _cell(1, 0.0, 1.0, p99=100.0, thr=5.0)
    cell["stats"]["latency_p99_ms"] = None
    kept, skipped = plot_sweep.load_cells(_summary([cell]))
    assert kept == []
    assert "no p99" in skipped[0]


def test_a_summary_with_nothing_plottable_fails_loudly(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(_summary([_cell(1, 0.0, 1.0, p99=1.0, thr=1.0, n_success=0)])))
    with pytest.raises(SystemExit):
        plot_sweep.plot_all(path)


def test_knee_needs_a_batch_1_baseline_and_does_not_crash_without_one(tmp_path: Path) -> None:
    cells = [_cell(b, 0.0, 1.0, p99=100.0 * b, thr=5.0) for b in (2, 4)]
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(_summary(cells)))
    out = plot_sweep.plot_knee(plot_sweep.load_cells(_summary(cells))[0], tmp_path / "knee.png")
    assert out.exists()
