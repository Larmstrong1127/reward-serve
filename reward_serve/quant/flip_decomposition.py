"""Where does int8-dynamic's damage land: on near-ties or on confident calls?

618 of 1,987 pairs flip their preferred side under int8-dynamic. That number
has two very different readings: if the flips sit on pairs whose fp32 margin
was near zero, quantization is breaking ties the model was guessing at anyway;
if they sit on confident calls, it is inverting real judgments. This script
answers which, using only the per-pair scores committed under
``results/quantization/raw/`` -- no forward passes.

    python -m reward_serve.quant.flip_decomposition

writes ``results/quantization/flip_decomposition.json``.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

RAW = Path("results/quantization/raw")
OUT = Path("results/quantization/flip_decomposition.json")


def margins(name: str) -> list[float]:
    d = json.loads((RAW / f"scores_{name}_n1987.json").read_text(encoding="utf-8"))
    return [c - r for c, r in zip(d["chosen"], d["rejected"], strict=True)]


def _pct(sorted_xs: list[float], q: float) -> float:
    i = min(len(sorted_xs) - 1, max(0, round(q * (len(sorted_xs) - 1))))
    return sorted_xs[i]


def main() -> None:
    fp32 = margins("fp32")
    int8 = margins("int8-dynamic")
    flipped = [abs(f) for f, q in zip(fp32, int8, strict=True) if (f > 0) != (q > 0)]
    kept = [abs(f) for f, q in zip(fp32, int8, strict=True) if (f > 0) == (q > 0)]

    all_abs = sorted(abs(f) for f in fp32)
    q1, q2, q3 = (_pct(all_abs, x) for x in (0.25, 0.50, 0.75))

    def bucket(m: float) -> str:
        if m <= q1:
            return "q1_weakest"
        if m <= q2:
            return "q2"
        if m <= q3:
            return "q3"
        return "q4_most_confident"

    counts = {"q1_weakest": 0, "q2": 0, "q3": 0, "q4_most_confident": 0}
    for m in flipped:
        counts[bucket(m)] += 1

    report = {
        "what_this_is": (
            "Decomposition of the int8-dynamic preference flips by the fp32 "
            "|margin| of the flipped pair, against quartiles of the full fp32 "
            "|margin| distribution. Arithmetic over the committed raw scores "
            "only; regenerate the raw scores with run_accuracy."
        ),
        "n_pairs": len(fp32),
        "n_flipped": len(flipped),
        "fp32_abs_margin_quartiles": {"q1": q1, "q2_median": q2, "q3": q3},
        "flips_by_fp32_margin_quartile": counts,
        "flipped_pairs_in_top_half_of_confidence": counts["q3"] + counts["q4_most_confident"],
        "flipped_pairs_in_top_quartile_of_confidence": counts["q4_most_confident"],
        "median_abs_fp32_margin_of_flipped_pairs": _pct(sorted(flipped), 0.50),
        "median_abs_fp32_margin_of_kept_pairs": _pct(sorted(kept), 0.50),
        "mean_abs_fp32_margin_flipped": statistics.fmean(flipped),
        "mean_abs_fp32_margin_kept": statistics.fmean(kept),
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
