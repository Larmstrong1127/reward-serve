"""Dump the measurement conditions a sweep would run under.

    python scripts/env_report.py results/<run-id>/environment.json

Same function the sweep embeds in its summary.json, so a standalone report and
a sweep's own record cannot drift apart.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loadgen.sweep import environment_report  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    report = environment_report()
    text = json.dumps(report, indent=2)
    if args:
        out = Path(args[0])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
