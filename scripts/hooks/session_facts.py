"""Claude Code SessionStart hook: put reward-serve's live numbers in front of every session.

The 2026-09-09 sweep found every claim surface six tests stale (221 written,
259 true) and nobody had noticed, because the number lived in prose and the
prose is never re-derived. This hook makes the current count and the last CI
conclusion the first thing a session sees, so a stale surface is a visible
disagreement instead of a silent one.

Cost matters: a SessionStart hook that takes 25 seconds is a hook that gets
disabled. `pytest --collect-only` takes ~22 s here, so the count is CACHED in
`results/test_count.json` keyed by commit sha. On a hit the hook prints it and
exits in well under a second. On a miss it prints the stale cached value,
says so, and refreshes the cache in a detached process for next time.
`--refresh` is that detached mode.

Everything fails open: no git, no gh, no cache -> a shorter line, never an
error that blocks the session.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "results" / "test_count.json"
GIT_TIMEOUT = 5
GH_TIMEOUT = 8


def _run(args: list[str], timeout: float) -> str | None:
    try:
        return subprocess.run(
            args, capture_output=True, text=True, timeout=timeout, cwd=ROOT, check=True
        ).stdout.strip()
    except Exception:
        return None


def head_sha() -> str | None:
    return _run(["git", "rev-parse", "--short", "HEAD"], GIT_TIMEOUT)


def ci_status() -> str | None:
    out = _run(
        ["gh", "run", "list", "--limit", "1", "--json", "conclusion,status,headSha"],
        GH_TIMEOUT,
    )
    if not out:
        return None
    try:
        r = json.loads(out)[0]
    except Exception:
        return None
    verdict = r.get("conclusion") or r.get("status") or "?"
    return f"{verdict} @ {str(r.get('headSha', ''))[:7]}"


def read_cache() -> dict[str, Any] | None:
    try:
        data = json.loads(CACHE.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def refresh_cache() -> dict[str, Any]:
    """Collect the suite and write the cache. Slow (~22 s); run detached."""
    sha = head_sha()
    out = _run([sys.executable, "-m", "pytest", "--collect-only", "-q"], 300) or ""
    last = out.splitlines()[-1] if out else ""
    n = None
    for tok in last.split():
        if tok.isdigit():
            n = int(tok)
            break
    rec = {
        "sha": sha,
        "collected": n,
        "collected_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "command": "pytest --collect-only -q",
        "note": "collected = tests + skips; the README states passed and skipped separately",
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    return rec


def spawn_refresh() -> None:
    try:
        flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--refresh"],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
        )
    except Exception:
        pass


def facts_line(
    sha: str | None, cache: dict[str, Any] | None, ci: str | None, *, refreshing: bool
) -> str:
    parts = [f"reward-serve @ {sha or '?'}"]
    if cache and cache.get("collected") is not None:
        c = cache["collected"]
        if cache.get("sha") == sha:
            parts.append(f"{c} tests collected (cached for this commit)")
        else:
            parts.append(
                f"{c} tests collected at {cache.get('sha')} -- STALE for HEAD"
                + (", refreshing in background" if refreshing else "")
            )
    else:
        parts.append("test count unknown" + (", collecting in background" if refreshing else ""))
    parts.append(f"CI {ci}" if ci else "CI unknown (gh unavailable)")
    return "; ".join(parts) + ". Any surface quoting a different test count is stale."


def main(argv: list[str]) -> int:
    if "--refresh" in argv:
        refresh_cache()
        return 0
    if os.environ.get("SESSION_FACTS_DISABLED") == "1":
        return 0
    try:
        sha = head_sha()
        cache = read_cache()
        stale = not cache or cache.get("sha") != sha
        if stale:
            spawn_refresh()
        line = facts_line(sha, cache, ci_status(), refreshing=stale)
    except Exception as exc:  # noqa: BLE001 - fail open
        line = f"session_facts failed ({type(exc).__name__}: {exc})"
    print(
        json.dumps(
            {"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": line}}
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
