"""Fail if prose uses scale vocabulary this one-box project cannot back.

A term is allowed only on a line that also contains an explicit disclaimer
("not", "no ", "never", "single", "one box", "one machine") within the same
sentence-ish window, so caveats like "this is not distributed" still pass.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

FORBIDDEN = [
    r"\bdistributed\b",
    r"\bat scale\b",
    r"\bproduction[- ]grade\b",
    r"\bproduction[- ]ready\b",
    r"\benterprise\b",
    r"\bdatacenter\b",
    r"\bcluster\b",
]
DISCLAIMER = re.compile(r"\b(not|no|never|single|one box|one machine|does not|is not)\b", re.I)
PROSE_GLOBS = ["README.md", "docs/**/*.md"]


def offending_lines(text: str) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for n, line in enumerate(text.splitlines(), 1):
        for pat in FORBIDDEN:
            if re.search(pat, line, re.I) and not DISCLAIMER.search(line):
                out.append((n, line.strip()))
                break
    return out


def main(root: Path) -> int:
    failures = 0
    for glob in PROSE_GLOBS:
        for path in root.glob(glob):
            if "CHARTER" in path.name:
                continue
            for n, line in offending_lines(path.read_text(encoding="utf-8")):
                print(f"{path.relative_to(root)}:{n}: {line}")
                failures += 1
    if failures:
        msg = f"\nvocab_gate: {failures} line(s) claim scale this project cannot back"
        print(msg, file=sys.stderr)
        return 1
    print("vocab_gate: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]))
