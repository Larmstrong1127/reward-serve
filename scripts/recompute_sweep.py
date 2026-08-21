"""Regenerate a sweep's statistics from its raw per-request rows, and check them.

    python scripts/recompute_sweep.py results/<run-id>              # verify
    python scripts/recompute_sweep.py results/<run-id> --out r.json # and write

This exists to answer one question a cold reader is entitled to ask: *how do I
know the numbers in `summary.json` describe the requests that were actually
sent?* Before this, they could not -- the sweep committed pre-computed order
statistics and nothing to check them against, so `p99 = 1899 ms` was assertable
but not checkable, which charter rule 2 does not accept.

What it reads:

* Statistics are rebuilt from `results/<run-id>/raw/*.jsonl` and **nothing
  else**. Not `summary.json`, not `cells.jsonl`, not the server logs. The cell
  configuration, the repeat index and the seeds all live on every raw row, so
  the raw directory alone is sufficient input.
* `summary.json` is then read ONLY as the thing being checked, and only in the
  verify step, after every number has already been computed.

What it cannot check, and says so instead of pretending:

* `loadgen.stats.GENERATOR_ONLY_KEYS` -- the load generator's own configuration
  and internal counters. `dropped_no_slot` is the honest example: it counts
  requests the generator decided not to send, which by construction have no row
  in `raw/`. These are listed in the report as `not_derivable_from_raw` rather
  than silently skipped.

Exit code 0 if every derivable statistic matches, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loadgen.generator import RequestRecord  # noqa: E402
from loadgen.stats import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    GENERATOR_ONLY_KEYS,
    aggregate_repeats,
    summarize_records,
)

PREFLIGHT_PREFIX = "preflight_"
DEFAULT_TOLERANCE = 1e-9


def read_raw_cell(path: Path) -> tuple[str, dict[int, list[RequestRecord]], dict[int, int]]:
    """One raw file -> (cell_id, {repeat: records}, {repeat: seed}).

    Rows keep their file order, which is send order within a repeat. Nothing here
    sorts or filters: the statistics functions own those decisions, so a reader
    who wants a different definition can apply it to the same rows.
    """
    by_repeat: dict[int, list[RequestRecord]] = defaultdict(list)
    seeds: dict[int, int] = {}
    cell_ids: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{lineno}: not JSON: {exc}") from exc
        cell_ids.add(str(row["cell_id"]))
        rep = int(row["repeat"])
        seeds[rep] = int(row["seed"])
        by_repeat[rep].append(RequestRecord.from_json(row))
    if len(cell_ids) > 1:
        raise SystemExit(f"{path}: rows from more than one cell: {sorted(cell_ids)}")
    cell_id = cell_ids.pop() if cell_ids else path.stem
    return cell_id, dict(by_repeat), seeds


def recompute_cell(path: Path, *, n_resamples: int) -> dict[str, Any]:
    """Every statistic for one cell, from that cell's raw rows alone."""
    cell_id, by_repeat, seeds = read_raw_cell(path)
    order = sorted(by_repeat)
    out: dict[str, Any] = {
        "cell_id": cell_id,
        "raw_file": path.name,
        "n_rows": sum(len(v) for v in by_repeat.values()),
        "n_warmup_rows": sum(1 for v in by_repeat.values() for r in v if r.is_warmup),
        "repeats_found": order,
    }
    if not order:
        out["stats"] = {}
        out["per_repeat"] = []
        out["note"] = "no rows: this cell measured nothing"
        return out
    out["per_repeat"] = [
        {
            "repeat": rep,
            "seed": seeds[rep],
            "stats": summarize_records([r for r in by_repeat[rep] if not r.is_warmup],
                                       include_warmup=True),
        }
        for rep in order
    ]
    out["stats"] = aggregate_repeats(
        [by_repeat[rep] for rep in order],
        cell_id=cell_id,
        repeat_seeds=[seeds[rep] for rep in order],
        n_resamples=n_resamples,
    )
    return out


def recompute_run(run_dir: Path, *, n_resamples: int = BOOTSTRAP_RESAMPLES) -> dict[str, Any]:
    """Every statistic for every cell of a run, from `raw/` alone."""
    raw_dir = run_dir / "raw"
    if not raw_dir.is_dir():
        raise SystemExit(
            f"{raw_dir} does not exist. This run predates per-request persistence, "
            "so its numbers cannot be recomputed -- which is the whole reason raw "
            "rows are now written. Re-run the sweep."
        )
    files = sorted(raw_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no raw/*.jsonl in {raw_dir}")
    cells = [recompute_cell(f, n_resamples=n_resamples) for f in files]
    return {
        "run_dir": str(run_dir),
        "source": "results/<run-id>/raw/*.jsonl only",
        "n_raw_files": len(files),
        "n_resamples": n_resamples,
        "cells": {c["cell_id"]: c for c in cells},
    }


def _close(a: Any, b: Any, tol: float) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y, tol) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_close(a[k], b[k], tol) for k in a)
    return bool(a == b)


def compare_stats(
    stored: dict[str, Any], recomputed: dict[str, Any], *, tol: float
) -> dict[str, Any]:
    """Compare one stats dict, skipping only what raw provably cannot hold."""
    checked: list[str] = []
    mismatches: list[dict[str, Any]] = []
    skipped: list[str] = []
    for key, want in stored.items():
        if key in GENERATOR_ONLY_KEYS:
            skipped.append(key)
            continue
        if key not in recomputed:
            mismatches.append({"key": key, "stored": want, "recomputed": "<missing>"})
            continue
        checked.append(key)
        if not _close(want, recomputed[key], tol):
            mismatches.append({"key": key, "stored": want, "recomputed": recomputed[key]})
    for key in recomputed:
        if key not in stored and key not in GENERATOR_ONLY_KEYS:
            mismatches.append({"key": key, "stored": "<missing>", "recomputed": recomputed[key]})
    return {
        "n_checked": len(checked),
        "not_derivable_from_raw": sorted(skipped),
        "mismatches": mismatches,
    }


def verify(
    run_dir: Path, *, tol: float = DEFAULT_TOLERANCE, n_resamples: int = BOOTSTRAP_RESAMPLES
) -> dict[str, Any]:
    """Recompute from raw, then check `summary.json` against the result."""
    recomputed = recompute_run(run_dir, n_resamples=n_resamples)
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"{summary_path} does not exist; nothing to verify against")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    cell_reports: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cell in summary.get("cells", []):
        cell_id = str(cell.get("cell_id"))
        seen.add(cell_id)
        got = recomputed["cells"].get(cell_id)
        if got is None:
            cell_reports.append(
                {"cell_id": cell_id, "ok": False,
                 "error": "no raw/<cell_id>.jsonl: this cell's numbers cannot be checked"}
            )
            continue
        agg = compare_stats(cell.get("stats") or {}, got["stats"], tol=tol)
        reps: list[dict[str, Any]] = []
        stored_reps = [r for r in (cell.get("per_repeat") or []) if r.get("status") == "ok"]
        got_reps = {r["repeat"]: r for r in got["per_repeat"]}
        for stored_rep in stored_reps:
            rep_i = int(stored_rep["repeat"])
            if rep_i not in got_reps:
                reps.append({"repeat": rep_i, "ok": False, "error": "no raw rows for this repeat"})
                continue
            cmp = compare_stats(
                stored_rep.get("stats") or {}, got_reps[rep_i]["stats"], tol=tol
            )
            reps.append({"repeat": rep_i, "ok": not cmp["mismatches"], **cmp})
        cell_reports.append(
            {
                "cell_id": cell_id,
                "ok": not agg["mismatches"] and all(r["ok"] for r in reps),
                "aggregate": agg,
                "per_repeat": reps,
                "n_raw_rows": got["n_rows"],
                "n_warmup_rows_present": got["n_warmup_rows"],
            }
        )

    extra = sorted(
        c for c in recomputed["cells"]
        if c not in seen and not c.startswith(PREFLIGHT_PREFIX)
    )
    report = {
        "run_dir": str(run_dir),
        "run_id": summary.get("run_id"),
        "smoke": bool(summary.get("smoke")),
        "not_a_measurement": bool(summary.get("not_a_measurement")),
        "tolerance": tol,
        "n_resamples": n_resamples,
        "source_of_recomputed_numbers": "results/<run-id>/raw/*.jsonl only",
        "n_cells": len(cell_reports),
        "n_cells_ok": sum(1 for c in cell_reports if c["ok"]),
        "raw_cells_not_in_summary": extra,
        "cells": cell_reports,
    }
    report["ok"] = bool(
        report["n_cells"] and report["n_cells_ok"] == report["n_cells"] and not extra
    )
    return report


def _print_verdict(report: dict[str, Any]) -> None:
    if report.get("not_a_measurement"):
        print("[recompute] NOTE: this run is stamped not_a_measurement (smoke run).")
    for cell in report["cells"]:
        if cell["ok"]:
            continue
        print(f"[recompute] MISMATCH {cell['cell_id']}", file=sys.stderr)
        for m in cell.get("aggregate", {}).get("mismatches", []):
            print(f"    aggregate {m['key']}: stored={m['stored']} raw={m['recomputed']}",
                  file=sys.stderr)
        for rep in cell.get("per_repeat", []):
            for m in rep.get("mismatches", []):
                print(f"    repeat {rep['repeat']} {m['key']}: "
                      f"stored={m['stored']} raw={m['recomputed']}", file=sys.stderr)
        if "error" in cell:
            print(f"    {cell['error']}", file=sys.stderr)
    n_ok, n = report["n_cells_ok"], report["n_cells"]
    skipped = report["cells"][0]["aggregate"]["not_derivable_from_raw"] if report["cells"] else []
    print(f"[recompute] {n_ok}/{n} cells regenerate exactly from raw/*.jsonl")
    if skipped:
        print(f"[recompute] not derivable from raw (generator-internal): {', '.join(skipped)}")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run_dir", type=Path, help="results/<run-id>")
    ap.add_argument("--out", type=Path, default=None, help="write the report as JSON")
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    ap.add_argument(
        "--resamples", type=int, default=BOOTSTRAP_RESAMPLES,
        help="bootstrap resamples; must match the run's uncertainty.ci_resamples",
    )
    args = ap.parse_args(argv)
    report = verify(args.run_dir, tol=args.tolerance, n_resamples=args.resamples)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[recompute] wrote {args.out}")
    _print_verdict(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
