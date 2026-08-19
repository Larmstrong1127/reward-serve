"""Forward latency per quantized variant, at the checkpoint's sequence budget.

This runner IS timing, so Charter rule 3 applies without exception: it calls
`scripts.box_quiet.require_quiet` before it builds anything, and it refuses on
failure. The refusal report is written to
`results/quantization/box_quiet_latency.json` and no `latency.json` is written -
the absence of that file is the honest artifact of a contended box.

    python -m reward_serve.quant.run_latency --variants fp32,int8-dynamic

Every measurement is a full forward pass over a batch padded to the full
sequence budget read off the checkpoint config, so batch size is the only thing
that varies across a sweep. The filler text is neutral synthetic prose.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse  # noqa: E402
import json  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import torch  # noqa: E402
import transformers  # noqa: E402

from reward_serve.model import DEFAULT_MODEL_ID, RewardModel  # noqa: E402
from reward_serve.quant.quantize import build_variant  # noqa: E402

__all__ = ["FILLER", "budget_filling_pair", "main", "percentile", "run"]

DEFAULT_OUT = Path("results/quantization/latency.json")
QUIET_REPORT = Path("results/quantization/box_quiet_latency.json")
DEFAULT_BATCH_SIZES = (1, 4, 8, 16)
WARMUP_ITERS = 5
MEASURED_ITERS = 30

# Neutral synthetic filler. No real prompts, no persona text (Charter rule 7).
FILLER = (
    "The maintenance schedule for the reference fixture lists each inspection "
    "step in order, with the tolerance for that step written beside it. "
)


def budget_filling_pair(scorer: RewardModel) -> tuple[str, str]:
    """A prompt/response pair long enough that truncation fills the budget.

    Latency at "the 512 budget" means every sequence is actually 512 tokens
    long, otherwise dynamic padding quietly shortens the measured work.
    """
    text = FILLER * 200
    encoded = scorer.tokenizer(
        text, text, truncation="longest_first", max_length=scorer.max_length
    )
    if len(encoded["input_ids"]) != scorer.max_length:
        raise RuntimeError(
            f"filler text encodes to {len(encoded['input_ids'])} tokens, "
            f"not the {scorer.max_length}-token budget"
        )
    return text, text


def percentile(values: list[float], q: float) -> float:
    """Exact order-statistic percentile: no interpolation, no numpy rounding."""
    if not values:
        raise ValueError("percentile of an empty sample")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def _peak_rss_mb() -> float:
    import psutil  # type: ignore[import-untyped]

    return round(float(psutil.Process().memory_info().rss) / (1024 * 1024), 2)


def run(
    variants: list[str],
    out_path: Path = DEFAULT_OUT,
    quiet_report: Path = QUIET_REPORT,
    batch_sizes: tuple[int, ...] = DEFAULT_BATCH_SIZES,
    threads: int = 1,
    iters: int = MEASURED_ITERS,
    warmup: int = WARMUP_ITERS,
    model_id: str = DEFAULT_MODEL_ID,
) -> int:
    """Returns 0 on a completed sweep, 1 when the box gate refused."""
    from scripts.box_quiet import require_quiet

    quiet_report.parent.mkdir(parents=True, exist_ok=True)
    try:
        quiet = require_quiet(quiet_report)
    except SystemExit as exc:
        print(f"box_quiet refused: {exc}", file=sys.stderr)
        print(f"refusal report: {quiet_report}", file=sys.stderr)
        if out_path.exists():
            out_path.unlink()
        return 1

    torch.set_num_threads(threads)
    rows: dict[str, Any] = {}
    for variant in variants:
        build = build_variant(variant, model_id)
        scorer = build.scorer
        pair = budget_filling_pair(scorer)
        per_batch: dict[str, Any] = {}
        for batch in batch_sizes:
            payload = [pair] * batch
            for _ in range(warmup):
                scorer.score_pairs(payload)
            samples: list[float] = []
            for _ in range(iters):
                started = time.perf_counter()
                scorer.score_pairs(payload)
                samples.append((time.perf_counter() - started) * 1000.0)
            per_batch[str(batch)] = {
                "iters": iters,
                "warmup_iters": warmup,
                "p50_ms": round(percentile(samples, 0.50), 3),
                "p95_ms": round(percentile(samples, 0.95), 3),
                "mean_ms": round(statistics.fmean(samples), 3),
                "min_ms": round(min(samples), 3),
                "peak_rss_mb": _peak_rss_mb(),
            }
            print(f"  {variant} batch={batch} p50={per_batch[str(batch)]['p50_ms']} ms", flush=True)
        rows[variant] = {
            "torch_api": build.torch_api,
            "weight_size_mb": build.size_mb,
            "sequence_length_tokens": scorer.max_length,
            "by_batch_size": per_batch,
        }
        del build, scorer

    payload_out: dict[str, Any] = {
        "_comment": (
            "Forward latency for quantized variants, one machine, CPU only. Gated on "
            "scripts/box_quiet.py; the gate report is box_quiet_latency.json beside this file."
        ),
        "model_id": model_id,
        "device": "cpu",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "python_version": sys.version.split()[0],
        "torch_num_threads": torch.get_num_threads(),
        "batch_sizes": list(batch_sizes),
        "command": "python -m reward_serve.quant.run_latency --variants " + ",".join(variants),
        "box_quiet": quiet,
        "variants": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload_out, indent=2), encoding="utf-8")
    print(f"[out] wrote {out_path}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", default="fp32,int8-dynamic")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--iters", type=int, default=MEASURED_ITERS)
    parser.add_argument("--warmup", type=int, default=WARMUP_ITERS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    return run(
        variants=[v.strip() for v in args.variants.split(",") if v.strip()],
        out_path=args.out,
        threads=args.threads,
        iters=args.iters,
        warmup=args.warmup,
    )


if __name__ == "__main__":
    raise SystemExit(main())
