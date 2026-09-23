"""The two Claude Code PreToolUse hooks: pure decisions, fail-open, end-to-end.

Nothing here samples a real box. `decide()` takes the sampler as a parameter
so every branch of the decision table is exercised with a fabricated sample,
and one subprocess test pipes a real payload through each script exactly the
way Claude Code will, to pin the stdin/stdout contract.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts.hooks import claim_gate, measurement_gate

ROOT = Path(__file__).resolve().parents[1]


def _bash(cmd: str) -> dict[str, Any]:
    return {"tool_name": "Bash", "tool_input": {"command": cmd}}


def _quiet(**over: Any) -> dict[str, Any]:
    s: dict[str, Any] = {
        "inflight_measurements": [],
        "gpu_mem_used_max_mib": 440,
        "gpu_util_mean_pct": 7.0,
        "foreign_python_hot": {},
        "window_s": 3.0,
        "sampled_at": "t",
    }
    s.update(over)
    return s


def _decision(result: dict[str, Any] | None) -> str | None:
    return None if result is None else result["hookSpecificOutput"]["permissionDecision"]


# --- measurement gate: what counts ------------------------------------------------


@pytest.mark.parametrize(
    "cmd",
    [
        "python -m loadgen.sweep --repeats 3",
        ".venv/Scripts/python.exe loadgen/sweep.py --resume",
        "py -X utf8 quant/run_latency.py",
        "python quant/run_accuracy.py --int8",
        "python scripts/box_quiet.py results/gate.json",
        "pytest tests/test_bench_latency.py",
    ],
)
def test_measurement_commands_are_recognised(cmd: str) -> None:
    assert measurement_gate.is_measurement(cmd)


@pytest.mark.parametrize(
    "cmd",
    [
        "git status",
        "pytest -q",
        "ruff check .",
        "python scripts/recompute_sweep.py results/x",
        "ls",
    ],
)
def test_ordinary_commands_are_ignored_without_sampling(cmd: str) -> None:
    def boom() -> dict[str, Any]:
        raise AssertionError("sampler must not run for a non-measurement")

    assert measurement_gate.decide(_bash(cmd), sampler=boom) is None


def test_smoke_runs_pass_without_sampling() -> None:
    def boom() -> dict[str, Any]:
        raise AssertionError("smoke runs are meant for a busy box")

    assert (
        measurement_gate.decide(
            _bash("python -m loadgen.sweep --smoke --no-box-quiet"), sampler=boom
        )
        is None
    )


def test_other_tools_are_ignored() -> None:
    assert measurement_gate.decide({"tool_name": "Write", "tool_input": {}}, sampler=_quiet) is None


# --- measurement gate: the decision table -----------------------------------------


def test_quiet_box_allows_and_attaches_the_sample() -> None:
    r = measurement_gate.decide(_bash("python -m loadgen.sweep"), sampler=_quiet)
    assert _decision(r) == "allow"
    ctx = r["hookSpecificOutput"]["additionalContext"]
    assert ctx.startswith("measurement_gate sample: ")
    assert json.loads(ctx.split(": ", 1)[1])["gpu_mem_used_max_mib"] == 440


def test_inflight_measurement_denies_and_names_it() -> None:
    s = _quiet(inflight_measurements=["4242: python -m loadgen.sweep --repeats 3"])
    r = measurement_gate.decide(_bash("python -m loadgen.sweep"), sampler=lambda: s)
    assert _decision(r) == "deny"
    assert "4242" in r["hookSpecificOutput"]["permissionDecisionReason"]


def test_gpu_memory_held_denies() -> None:
    s = _quiet(gpu_mem_used_max_mib=23000)
    r = measurement_gate.decide(_bash("python quant/run_latency.py"), sampler=lambda: s)
    assert _decision(r) == "deny"
    assert "23000" in r["hookSpecificOutput"]["permissionDecisionReason"]


def test_no_gpu_is_not_a_reason_to_deny() -> None:
    """A CI runner has no nvidia-smi: None must read as quiet, not as held."""
    r = measurement_gate.decide(
        _bash("python -m loadgen.sweep"), sampler=lambda: _quiet(gpu_mem_used_max_mib=None)
    )
    assert _decision(r) == "allow"


def test_hot_foreign_python_asks_not_denies() -> None:
    s = _quiet(foreign_python_hot={"777:python.exe": 45.0})
    r = measurement_gate.decide(_bash("python -m loadgen.sweep"), sampler=lambda: s)
    assert _decision(r) == "ask"


def test_sampler_failure_fails_open_and_says_so() -> None:
    def broken() -> dict[str, Any]:
        raise RuntimeError("psutil exploded")

    r = measurement_gate.decide(_bash("python -m loadgen.sweep"), sampler=broken)
    assert _decision(r) == "allow"
    assert "UNGATED" in r["hookSpecificOutput"]["permissionDecisionReason"]
    assert "psutil exploded" in r["hookSpecificOutput"]["permissionDecisionReason"]


def test_deny_outranks_ask_when_both_apply() -> None:
    s = _quiet(
        inflight_measurements=["1: python -m loadgen.sweep"], foreign_python_hot={"1:python": 90.0}
    )
    assert (
        _decision(measurement_gate.decide(_bash("python -m loadgen.sweep"), sampler=lambda: s))
        == "deny"
    )


# --- claim gate -------------------------------------------------------------------

ROOTS = ["reward-serve", "prep-a-step/content", "larmstrong1127.github.io"]


def _write(path: str, content: str) -> dict[str, Any]:
    return {"tool_name": "Write", "tool_input": {"file_path": path, "content": content}}


def _edit(path: str, new: str) -> dict[str, Any]:
    return {
        "tool_name": "Edit",
        "tool_input": {"file_path": path, "old_string": "x", "new_string": new},
    }


def test_forbidden_word_on_a_claim_surface_asks() -> None:
    r = claim_gate.decide(
        _write(r"C:\work\reward-serve\README.md", "A distributed serving layer."), ROOTS
    )
    assert _decision(r) == "ask"
    assert "distributed" in r["hookSpecificOutput"]["permissionDecisionReason"]
    # visible to the agent even when the ask is settled by a classifier it never sees
    hso = r["hookSpecificOutput"]
    assert hso["additionalContext"] == hso["permissionDecisionReason"]


def test_disclaimer_on_the_same_line_passes() -> None:
    r = claim_gate.decide(_write("/work/reward-serve/README.md", "This is not distributed."), ROOTS)
    assert r is None


def test_edit_scans_only_the_new_text() -> None:
    r = claim_gate.decide(
        _edit("/work/prep-a-step/content/reward-serve.json", '"pitch": "production-grade"'),
        ROOTS,
    )
    assert _decision(r) == "ask"


def test_code_files_are_never_gated() -> None:
    r = claim_gate.decide(
        _write("/work/reward-serve/reward_serve/router.py", "# a cluster of upstreams"),
        ROOTS,
    )
    assert r is None


def test_charter_is_exempt() -> None:
    r = claim_gate.decide(_write("/work/reward-serve/CHARTER.md", "forbidden: distributed"), ROOTS)
    assert r is None


def test_unrelated_projects_are_not_gated() -> None:
    r = claim_gate.decide(
        _write("/work/some-other-project/README.md", "a cluster of renders"), ROOTS
    )
    assert r is None


def test_claim_gate_uses_the_ci_gate_verbatim() -> None:
    """One word list. If CI and the hook ever disagree, one of them is lying."""
    from scripts import vocab_gate

    assert claim_gate.offending_lines is vocab_gate.offending_lines


# --- the stdin/stdout contract, end to end ----------------------------------------


def _run(
    script: str, payload: dict[str, Any], env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    import os

    e = dict(os.environ)
    e.update(env or {})
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "hooks" / script)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=30,
        env=e,
        cwd=ROOT,
    )


def test_claim_gate_script_speaks_the_hook_protocol() -> None:
    p = _run("claim_gate.py", _write("/work/reward-serve/docs/x.md", "runs at scale"))
    assert p.returncode == 0, p.stderr
    out = json.loads(p.stdout)
    assert out["hookSpecificOutput"]["permissionDecision"] == "ask"


def test_claim_gate_script_is_silent_when_clean() -> None:
    p = _run("claim_gate.py", _write("/work/reward-serve/docs/x.md", "one machine, honestly"))
    assert p.returncode == 0 and p.stdout.strip() == ""


def test_measurement_gate_script_is_silent_for_ordinary_commands() -> None:
    p = _run("measurement_gate.py", _bash("git status"))
    assert p.returncode == 0 and p.stdout.strip() == "", p.stderr


def test_measurement_gate_script_honours_the_kill_switch() -> None:
    p = _run(
        "measurement_gate.py", _bash("python -m loadgen.sweep"), {"MEASUREMENT_GATE_DISABLED": "1"}
    )
    assert p.returncode == 0 and p.stdout.strip() == ""


def test_garbage_stdin_fails_open() -> None:
    p = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "hooks" / "measurement_gate.py")],
        input="not json",
        capture_output=True,
        text=True,
        timeout=30,
        cwd=ROOT,
    )
    assert p.returncode == 0
    assert json.loads(p.stdout)["hookSpecificOutput"]["permissionDecision"] == "allow"


# --- session facts ---------------------------------------------------------------


def test_session_facts_line_marks_a_stale_cache() -> None:
    from scripts.hooks import session_facts

    line = session_facts.facts_line(
        "abc1234", {"sha": "old0000", "collected": 259}, "success @ abc1234", refreshing=True
    )
    assert "STALE" in line and "refreshing" in line and "259" in line


def test_session_facts_line_trusts_a_fresh_cache() -> None:
    from scripts.hooks import session_facts

    line = session_facts.facts_line(
        "abc1234", {"sha": "abc1234", "collected": 261}, None, refreshing=False
    )
    assert "261 tests collected (cached for this commit)" in line
    assert "CI unknown" in line and "STALE" not in line


def test_session_facts_line_survives_no_cache_and_no_git() -> None:
    from scripts.hooks import session_facts

    line = session_facts.facts_line(None, None, None, refreshing=True)
    assert "test count unknown" in line and "collecting" in line


@pytest.mark.parametrize(
    "cmd",
    [
        "java -jar ingest.jar el-ingest replay --date 2026-09-01",
        "python -m loadgen.service --rps 20",
    ],
)
def test_event_ledger_measurement_commands_are_recognised(cmd: str) -> None:
    assert measurement_gate.is_measurement(cmd)
