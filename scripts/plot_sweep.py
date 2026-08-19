"""Plot a sweep from its summary.json. Never from memory, never from prose.

    python scripts/plot_sweep.py results/<run-id>/summary.json

Writes into the same directory as the summary:
  p99_vs_batch.png        p99 latency against MAX_BATCH_SIZE, one line per
                          MAX_WAIT_MS, one panel per offered load
  throughput_vs_batch.png the same layout for successful requests per second
  knee.png                what each batch size bought and what it cost, both
                          relative to that panel's MAX_BATCH_SIZE=1 baseline

Cells that did not complete, or that produced no successful requests, are
dropped and reported on stderr rather than plotted as zero.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

Cell = dict[str, Any]

MARKERS = ("o", "s", "^", "D", "v", "P")


def load_cells(summary: dict[str, Any]) -> tuple[list[Cell], list[str]]:
    """Keep only cells that actually measured something; explain the rest."""
    kept: list[Cell] = []
    skipped: list[str] = []
    for cell in summary.get("cells", []):
        stats = cell.get("stats") or {}
        if cell.get("status") != "ok":
            skipped.append(f"{cell.get('cell_id')}: status={cell.get('status')}")
        elif not stats.get("n_success"):
            skipped.append(f"{cell.get('cell_id')}: no successful requests")
        elif stats.get("latency_p99_ms") is None:
            skipped.append(f"{cell.get('cell_id')}: no p99")
        else:
            kept.append(cell)
    return kept, skipped


def _by_load(cells: list[Cell]) -> dict[float, dict[float, list[Cell]]]:
    """load -> wait_ms -> cells sorted by batch size."""
    out: dict[float, dict[float, list[Cell]]] = defaultdict(lambda: defaultdict(list))
    for c in cells:
        out[float(c["target_rps"])][float(c["max_wait_ms"])].append(c)
    for waits in out.values():
        for lst in waits.values():
            lst.sort(key=lambda c: c["max_batch_size"])
    return out


def _panels(cells: list[Cell]) -> tuple[list[float], dict[float, dict[float, list[Cell]]]]:
    grouped = _by_load(cells)
    return sorted(grouped), grouped


def _series_label(wait_ms: float) -> str:
    return f"MAX_WAIT_MS={wait_ms:g}"


def _finish(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_metric(
    cells: list[Cell], out: Path, *, key: str, ylabel: str, title: str, logy: bool = False
) -> Path:
    loads, grouped = _panels(cells)
    fig, axes = plt.subplots(1, len(loads), figsize=(5.2 * len(loads), 4.4), squeeze=False)
    for ax, load in zip(axes[0], loads, strict=True):
        for i, wait in enumerate(sorted(grouped[load])):
            series = grouped[load][wait]
            xs = [c["max_batch_size"] for c in series]
            ys = [c["stats"][key] for c in series]
            ax.plot(xs, ys, marker=MARKERS[i % len(MARKERS)], label=_series_label(wait))
        ax.set_xscale("log", base=2)
        if logy:
            ax.set_yscale("log")
        ax.set_xticks(sorted({c["max_batch_size"] for c in cells}))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("MAX_BATCH_SIZE")
        ax.set_ylabel(ylabel)
        ax.set_title(f"offered {load:g} rps")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    return _finish(fig, out)


def plot_knee(cells: list[Cell], out: Path) -> Path:
    """Throughput gain against p99 cost, both relative to batch size 1.

    A batch size is worth its latency only while the gain line stays above the
    cost line. Where they cross, larger batches are charging p99 for throughput
    they are not delivering.
    """
    loads, grouped = _panels(cells)
    fig, axes = plt.subplots(1, len(loads), figsize=(5.2 * len(loads), 4.4), squeeze=False)
    for ax, load in zip(axes[0], loads, strict=True):
        for i, wait in enumerate(sorted(grouped[load])):
            series = grouped[load][wait]
            base = next((c for c in series if c["max_batch_size"] == 1), None)
            if base is None:
                continue
            base_thr = base["stats"]["throughput_rps"]
            base_p99 = base["stats"]["latency_p99_ms"]
            if not base_thr or not base_p99:
                continue
            xs = [c["max_batch_size"] for c in series]
            gain = [c["stats"]["throughput_rps"] / base_thr for c in series]
            cost = [c["stats"]["latency_p99_ms"] / base_p99 for c in series]
            m = MARKERS[i % len(MARKERS)]
            ax.plot(xs, gain, marker=m, linestyle="-", label=f"gain, {_series_label(wait)}")
            ax.plot(xs, cost, marker=m, linestyle="--", alpha=0.6,
                    label=f"p99 cost, {_series_label(wait)}")
        ax.axhline(1.0, color="black", linewidth=1, alpha=0.5)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(sorted({c["max_batch_size"] for c in cells}))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("MAX_BATCH_SIZE")
        ax.set_ylabel("x relative to MAX_BATCH_SIZE=1")
        ax.set_title(f"offered {load:g} rps")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Knee: throughput gain (solid) vs p99 cost (dashed), both vs batch size 1")
    return _finish(fig, out)


def plot_all(summary_path: Path) -> list[Path]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cells, skipped = load_cells(summary)
    for s in skipped:
        print(f"[plot] skipped {s}", file=sys.stderr)
    if not cells:
        raise SystemExit(f"no plottable cells in {summary_path}")
    out_dir = summary_path.parent
    written = [
        plot_metric(
            cells, out_dir / "p99_vs_batch.png", key="latency_p99_ms",
            ylabel="p99 latency (ms, successes only)",
            title="p99 latency vs batch size", logy=True,
        ),
        plot_metric(
            cells, out_dir / "throughput_vs_batch.png", key="throughput_rps",
            ylabel="successful requests / s",
            title="throughput vs batch size",
        ),
        plot_knee(cells, out_dir / "knee.png"),
    ]
    for p in written:
        print(f"[plot] wrote {p}")
    return written


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(__doc__)
        return 2
    plot_all(Path(args[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
