"""The sweep must refuse a contended box, and must record how to reproduce itself."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from loadgen import sweep as sweep_mod

REPO_ROOT = Path(__file__).resolve().parents[1]
STUB_CMD = f'"{sys.executable}" "{REPO_ROOT / "tests" / "stub_server.py"}"'


def test_grid_is_the_full_cross_product() -> None:
    combos = list(sweep_mod.grid([1, 2], [0.0, 5.0], [10.0]))
    assert len(combos) == 4
    assert combos[0] == (1, 0.0, 10.0)
    assert (2, 5.0, 10.0) in combos


def test_sweep_refuses_to_run_when_box_quiet_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called: dict[str, Any] = {"cells": 0}

    def fake_require_quiet(path: Path | None = None) -> dict[str, Any]:
        report = {"passed": False, "cpu_mean_pct": 91.4}
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report))
        raise SystemExit(f"box_quiet FAIL: {json.dumps(report)}")

    async def fake_run_cell(**kwargs: Any) -> dict[str, Any]:
        called["cells"] += 1
        return {"status": "ok"}

    monkeypatch.setattr(sweep_mod, "require_quiet", fake_require_quiet)
    monkeypatch.setattr(sweep_mod, "run_cell", fake_run_cell)

    with pytest.raises(SystemExit) as exc:
        sweep_mod.main(
            ["--run-id", "gated", "--results-dir", str(tmp_path), "--server-cmd", STUB_CMD]
        )
    assert "box_quiet FAIL" in str(exc.value)
    assert called["cells"] == 0, "the grid ran despite a contended box"
    assert not (tmp_path / "gated" / "cells.jsonl").exists()
    # the failing report is still stored next to where the results would have gone
    assert json.loads((tmp_path / "gated" / "box_quiet.json").read_text())["passed"] is False


def test_sweep_has_no_flag_to_skip_the_box_quiet_gate() -> None:
    src = (REPO_ROOT / "loadgen" / "sweep.py").read_text(encoding="utf-8")
    for forbidden in ("--skip-box-quiet", "--no-box-quiet", "--force"):
        assert forbidden not in src, f"{forbidden} would let a contended run publish"


def test_sweep_runs_a_tiny_grid_and_writes_reproducible_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end against the stub, with the quiet gate stubbed PASS so this
    test does not depend on how busy the machine is."""
    monkeypatch.setattr(
        sweep_mod, "require_quiet",
        lambda path=None: {"passed": True, "note": "monkeypatched in tests"},
    )
    rc = sweep_mod.main([
        "--run-id", "tiny", "--results-dir", str(tmp_path),
        "--server-cmd", STUB_CMD, "--server-cwd", str(REPO_ROOT),
        "--target-label", "stub_server (no model)",
        "--batch-sizes", "1,4", "--wait-ms", "5", "--rps", "30",
        "--duration", "1.5", "--warmup", "0.5",
    ])
    assert rc == 0
    out = tmp_path / "tiny"
    lines = (out / "cells.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    summary = json.loads((out / "summary.json").read_text())
    assert summary["n_cells"] == summary["n_cells_ok"] == 2
    assert summary["box_quiet"]["passed"] is True
    assert "python -m loadgen.sweep --run-id tiny" in summary["reproduce_command"]
    assert summary["environment"]["cuda_visible_devices"] == ""
    assert summary["load"]["repeats_per_cell"] == 1
    for cell in summary["cells"]:
        assert cell["status"] == "ok"
        assert cell["env_overrides"]["MAX_BATCH_SIZE"] in ("1", "4")
        assert cell["env_overrides"]["CUDA_VISIBLE_DEVICES"] == ""
        assert cell["stats"]["n_measured"] > 0
        assert cell["stats"]["n_warmup_excluded"] > 0
        assert cell["command"], "the exact command must be recorded"


def test_free_port_returns_a_usable_port() -> None:
    p = sweep_mod.free_port()
    assert 1024 < p < 65536


def test_preflight_aborts_when_the_server_is_not_batching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A batcher that forwards a backlog one item at a time must stop the grid."""
    monkeypatch.setattr(sweep_mod, "require_quiet", lambda path=None: {"passed": True})
    grid_cells = {"n": 0}

    async def fake_run_cell(**kwargs: Any) -> dict[str, Any]:
        if kwargs.get("batch_size") == 8 and kwargs.get("rps") == 4.0:
            return {"cell_id": "preflight", "status": "ok",
                    "stats": {"mean_batch_size": 1.0, "n_success": 50}}
        grid_cells["n"] += 1
        return {"status": "ok"}

    monkeypatch.setattr(sweep_mod, "run_cell", fake_run_cell)
    with pytest.raises(SystemExit) as exc:
        sweep_mod.main([
            "--run-id", "nobatch", "--results-dir", str(tmp_path),
            "--server-cmd", STUB_CMD, "--batch-sizes", "1", "--wait-ms", "5", "--rps", "10",
        ])
    assert "not batching" in str(exc.value)
    assert grid_cells["n"] == 0, "the grid ran against a non-batching server"
    check = json.loads((tmp_path / "nobatch" / "preflight_batching.json").read_text())
    assert check["mean_batch_size"] == 1.0


def test_preflight_passes_and_lets_the_grid_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sweep_mod, "require_quiet", lambda path=None: {"passed": True})
    grid_cells = {"n": 0}

    async def fake_run_cell(**kwargs: Any) -> dict[str, Any]:
        if kwargs.get("batch_size") == 8 and kwargs.get("rps") == 4.0:
            return {"cell_id": "preflight", "status": "ok",
                    "stats": {"mean_batch_size": 5.5, "n_success": 50}}
        grid_cells["n"] += 1
        return {"status": "ok", "stats": {}}

    monkeypatch.setattr(sweep_mod, "run_cell", fake_run_cell)
    rc = sweep_mod.main([
        "--run-id", "batched", "--results-dir", str(tmp_path),
        "--server-cmd", STUB_CMD, "--batch-sizes", "1,2", "--wait-ms", "5", "--rps", "10",
    ])
    assert rc == 0
    assert grid_cells["n"] == 2
    summary = json.loads((tmp_path / "batched" / "summary.json").read_text())
    assert summary["preflight_batching"]["mean_batch_size"] == 5.5
