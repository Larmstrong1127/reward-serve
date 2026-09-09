"""Claude Code PreToolUse hook: ask before scale vocabulary reaches a claim surface.

`scripts/vocab_gate.py` fails CI when README or docs claim "distributed",
"at scale", "production-grade" and the rest without a disclaimer on the same
line. CI is the last line, though: the words were already written, committed
and pushed by the time it fired, and the same words leaked onto surfaces CI
never sees -- the portfolio site, the interview deck, the profile README, the
job-pipeline standing answers. The claim-surface sweep found the deck still
saying "runs on every PR" weeks after the repo said zero runs.

This hook moves the same check to the moment of writing. On every Write or
Edit whose target is a claim surface, the text about to be written is scanned
with the SAME `offending_lines` the CI gate uses; if a forbidden term appears
without a disclaimer, the tool call becomes an ASK with the offending lines as
the reason. It never denies -- an honest caveat that the regex misreads is
the author's call -- and it never touches code files.

Claim surfaces default to the eight-surface list the sweep rule names. Override
with CLAIM_GATE_ROOTS (path fragments, ';'-separated) when the list changes.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.vocab_gate import offending_lines  # noqa: E402

DEFAULT_ROOTS = [
    "reward-serve",
    "prep-a-step/content",
    "larmstrong1127.github.io",
    "job-pipeline/profile",
    "job-pipeline/applications",
    "job-pipeline/upstream-drafts",
    "resume",
    "profile2.md",
]
CLAIM_SUFFIXES = {".md", ".html", ".json", ".txt", ".yml", ".yaml"}
EXEMPT_NAMES = {"CHARTER.md"}


def roots() -> list[str]:
    env = os.environ.get("CLAIM_GATE_ROOTS")
    return [r for r in env.split(";") if r] if env else DEFAULT_ROOTS


def is_claim_surface(file_path: str, root_fragments: list[str] | None = None) -> bool:
    if not file_path:
        return False
    p = Path(file_path)
    if p.name in EXEMPT_NAMES or p.suffix.lower() not in CLAIM_SUFFIXES:
        return False
    norm = file_path.replace("\\", "/").lower()
    return any(frag.lower() in norm for frag in root_fragments or roots())


def text_to_write(tool_name: str, tool_input: dict[str, Any]) -> str:
    if tool_name == "Write":
        return str(tool_input.get("content") or "")
    if tool_name == "Edit":
        return str(tool_input.get("new_string") or "")
    return ""


def decide(
    payload: dict[str, Any], root_fragments: list[str] | None = None
) -> dict[str, Any] | None:
    tool = payload.get("tool_name")
    if tool not in ("Write", "Edit"):
        return None
    ti = payload.get("tool_input") or {}
    path = str(ti.get("file_path") or "")
    if not is_claim_surface(path, root_fragments):
        return None
    hits = offending_lines(text_to_write(tool, ti))
    if not hits:
        return None
    shown = "\n  ".join(f"{n}: {line[:140]}" for n, line in hits[:8])
    more = f"\n  (+{len(hits) - 8} more)" if len(hits) > 8 else ""
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "ask",
            "permissionDecisionReason": (
                f"claim_gate: {len(hits)} line(s) about to enter a claim surface use scale "
                "vocabulary this one-box project cannot back (distributed / at scale / "
                "production-grade / enterprise / datacenter / cluster). Add a disclaimer on "
                f"the same line or confirm it is intended.\n  {shown}{more}"
            ),
        }
    }


def main() -> int:
    if os.environ.get("CLAIM_GATE_DISABLED") == "1":
        return 0
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        result = decide(payload)
    except Exception as exc:  # noqa: BLE001 - fail open, visibly
        why = f"claim_gate failed ({type(exc).__name__}: {exc}); allowing."
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "allow",
                        "permissionDecisionReason": why,
                    }
                }
            )
        )
        return 0
    if result is not None:
        print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
