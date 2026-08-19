"""Measure pairwise accuracy and ranking drift for each quantized variant.

Every variant is scored on the SAME pairs as the published fp32 number
(UltraFeedback `test_prefs`, the truncation filter from `eval_split`, the
sequence budget read off the checkpoint config), so the accuracies here are
paired with 0.7026 rather than merely near it.

This runner is NOT timing. It may run on a contended box, so it does not gate
on `scripts/box_quiet.py` - but it records the box_quiet report beside its
output anyway, and stamps the contention verdict into the JSON, so a reader can
see the conditions rather than assume them.

    python -m reward_serve.quant.run_accuracy --variants fp32,int8-dynamic
    python -m reward_serve.quant.run_accuracy --variants fp32 --limit 50
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import torch  # noqa: E402
import transformers  # noqa: E402

from reward_serve.model import DEFAULT_MODEL_ID, RewardModel  # noqa: E402
from reward_serve.quant.eval_split import (  # noqa: E402
    EVAL_SPLIT,
    PUBLISHED_FP32_ACCURACY,
    PUBLISHED_N_PAIRS,
    ULTRAFEEDBACK_DATASET_ID,
    PreferencePair,
    load_eval_pairs,
)
from reward_serve.quant.metrics import (  # noqa: E402
    flipped_fraction,
    kendall_tau_b,
    mean_abs_delta,
    pairwise_accuracy,
)
from reward_serve.quant.quantize import TORCH_API, build_variant  # noqa: E402

__all__ = ["main", "run", "score_split"]

DEFAULT_OUT = Path("results/quantization/accuracy.json")
DEFAULT_SEED = 1234


def score_split(
    scorer: RewardModel,
    pairs: list[PreferencePair],
    batch_size: int,
    checkpoint: Path | None = None,
) -> tuple[list[float], list[float]]:
    """Score every chosen and every rejected response. Returns them in order.

    Chosen and rejected are batched separately, matching the eval harness that
    produced the published number: each side is padded to its own batch maximum
    and the padding is fully masked, so batching cannot move a score.

    A full pass over 1,987 pairs takes tens of minutes on this CPU, and this
    machine kills long sessions. So progress is flushed to `checkpoint` as it
    goes and picked up from there on the next run: a killed pass costs the
    current chunk, not the whole variant. Resume is exact rather than
    approximate - the pairs are scored in a fixed order, so the first K cached
    scores are the first K pairs, and scoring continues at index K.
    """
    chosen: list[float] = []
    rejected: list[float] = []
    if checkpoint is not None and checkpoint.exists():
        cached = json.loads(checkpoint.read_text(encoding="utf-8"))
        chosen, rejected = cached["chosen"], cached["rejected"]
        chosen = chosen[: len(rejected)]
        rejected = rejected[: len(chosen)]
        if chosen:
            print(f"    resuming at pair {len(chosen)}/{len(pairs)}", flush=True)
    started = time.perf_counter()
    resumed_at = len(chosen)
    for start in range(resumed_at, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        chosen.extend(scorer.score_pairs([(p.prompt, p.chosen) for p in chunk]))
        rejected.extend(scorer.score_pairs([(p.prompt, p.rejected) for p in chunk]))
        done = start + len(chunk)
        if done % (batch_size * 25) == 0 or done == len(pairs):
            rate = (done - resumed_at) / max(time.perf_counter() - started, 1e-9)
            print(f"    {done}/{len(pairs)} pairs ({rate:.1f} pairs/s)", flush=True)
            if checkpoint is not None:
                _atomic_write(checkpoint, {"chosen": chosen, "rejected": rejected})
    return chosen, rejected


def _atomic_write(path: Path, payload: dict[str, list[float]]) -> None:
    """Write via a temp file and replace, so a kill mid-write cannot corrupt it."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)


def _margins(chosen: list[float], rejected: list[float]) -> list[float]:
    return [c - r for c, r in zip(chosen, rejected, strict=True)]


def _cache_path(cache_dir: Path, variant: str, n_pairs: int) -> Path:
    return cache_dir / f"scores_{variant}_n{n_pairs}.json"


def run(
    variants: list[str],
    out_path: Path,
    limit: int | None = None,
    batch_size: int = 8,
    threads: int = 1,
    seed: int = DEFAULT_SEED,
    model_id: str = DEFAULT_MODEL_ID,
    use_cache: bool = True,
    measure_box: bool = True,
) -> dict[str, Any]:
    """Score the requested variants and merge them into the results JSON."""
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if "fp32" not in variants:
        variants = ["fp32", *variants]

    print("[build] fp32 (also supplies the tokenizer and the sequence budget)", flush=True)
    fp32_build = build_variant("fp32", model_id)
    tokenizer = fp32_build.scorer.tokenizer
    max_length = fp32_build.scorer.max_length
    print(f"[split] sequence budget {max_length} tokens, from the checkpoint config", flush=True)

    pairs, audit = load_eval_pairs(tokenizer, max_length, limit=limit)
    n_pairs = len(pairs)
    print(
        f"[split] {audit.total} loaded, {audit.dropped_identical} dropped by the truncation "
        f"filter, {n_pairs} evaluated",
        flush=True,
    )

    cache_dir = out_path.parent / "raw"
    cache_dir.mkdir(parents=True, exist_ok=True)

    per_variant: dict[str, dict[str, list[float]]] = {}
    build_facts: dict[str, dict[str, Any]] = {}
    for variant in variants:
        cache = _cache_path(cache_dir, variant, n_pairs)
        build = fp32_build if variant == "fp32" else build_variant(variant, model_id)
        build_facts[variant] = {
            "torch_api": build.torch_api,
            "weight_size_mb": build.size_mb,
            "quantized_linear_modules": build.quantized_linear_modules,
            "plain_linear_modules": build.plain_linear_modules,
        }
        if use_cache and cache.exists():
            print(f"[score] {variant}: reusing cached scores at {cache}", flush=True)
            per_variant[variant] = json.loads(cache.read_text(encoding="utf-8"))
            continue
        print(f"[score] {variant}: {n_pairs} pairs, batch {batch_size}", flush=True)
        started = time.perf_counter()
        partial = cache.with_name(cache.stem + ".partial.json")
        chosen, rejected = score_split(build.scorer, pairs, batch_size, checkpoint=partial)
        elapsed = time.perf_counter() - started
        print(f"[score] {variant}: done in {elapsed / 60:.1f} min", flush=True)
        if len(chosen) != len(pairs) or len(rejected) != len(pairs):
            raise RuntimeError(
                f"{variant}: scored {len(chosen)}/{len(pairs)} pairs - refusing to write a "
                "partial pass as if it were a complete one"
            )
        per_variant[variant] = {"chosen": chosen, "rejected": rejected}
        _atomic_write(cache, per_variant[variant])
        partial.unlink(missing_ok=True)
        if variant != "fp32":
            del build

    base = per_variant["fp32"]
    base_margins = _margins(base["chosen"], base["rejected"])
    base_scores = base["chosen"] + base["rejected"]

    rows: dict[str, Any] = {}
    for variant in variants:
        scores = per_variant[variant]
        margins = _margins(scores["chosen"], scores["rejected"])
        flipped = flipped_fraction(margins, base_margins)
        rows[variant] = {
            **build_facts[variant],
            "pairwise_accuracy": round(pairwise_accuracy(margins), 4),
            "mean_abs_score_delta_vs_fp32": round(
                mean_abs_delta(scores["chosen"] + scores["rejected"], base_scores), 6
            ),
            "mean_abs_margin_delta_vs_fp32": round(mean_abs_delta(margins, base_margins), 6),
            "kendall_tau_margins_vs_fp32": round(kendall_tau_b(margins, base_margins), 6),
            "flipped_fraction_vs_fp32": round(flipped, 6),
            "flipped_pairs_vs_fp32": int(round(flipped * len(margins))),
            "flips_per_1000_pairs": round(flipped * 1000, 2),
        }
        rows[variant]["accuracy_delta_vs_fp32"] = round(
            rows[variant]["pairwise_accuracy"] - rows["fp32"]["pairwise_accuracy"], 4
        )

    box_report: dict[str, Any] | None = None
    if measure_box:
        from scripts.box_quiet import measure

        print("[box] sampling the box for 30s (recorded, not gating)", flush=True)
        box_report = measure()
        box_path = out_path.parent / "box_quiet.json"
        box_path.write_text(json.dumps(box_report, indent=2), encoding="utf-8")

    payload: dict[str, Any] = {
        "_comment": (
            "Pairwise accuracy and ranking drift for quantized variants of "
            "DantheMan124/deberta-preference-reward, measured on the same pairs as the "
            "published fp32 number so the deltas are paired. CPU only."
        ),
        "model_id": model_id,
        "dataset": ULTRAFEEDBACK_DATASET_ID,
        "split": EVAL_SPLIT,
        "n_pairs_loaded": audit.total,
        "n_dropped_by_truncation_filter": audit.dropped_identical,
        "n_pairs_evaluated": n_pairs,
        "max_length_tokens": max_length,
        "score_definition": "raw logit / reward_temperature (ranking-only)",
        "reward_temperature": round(fp32_build.scorer.temperature, 6),
        "published_fp32_accuracy": PUBLISHED_FP32_ACCURACY,
        "published_n_pairs": PUBLISHED_N_PAIRS,
        "device": "cpu",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "python_version": sys.version.split()[0],
        "torch_num_threads": torch.get_num_threads(),
        "seed": seed,
        "batch_size": batch_size,
        "torch_api_reference": TORCH_API,
        "command": (
            "python -m reward_serve.quant.run_accuracy --variants "
            + ",".join(v for v in variants if v != "fp32")
        ),
        "box_quiet_passed": None if box_report is None else box_report["passed"],
        "box_quiet_note": (
            "This runner is not timing, so a contended box does not invalidate it; the "
            "report is stored beside this file as box_quiet.json for the reader to judge."
        ),
        "variants": rows,
    }
    if limit is not None:
        payload["dry_run_limit"] = limit

    if out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        if existing.get("n_pairs_evaluated") == n_pairs:
            merged = {**existing.get("variants", {}), **rows}
            payload["variants"] = merged
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[out] wrote {out_path}", flush=True)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", default="fp32,int8-dynamic")
    parser.add_argument("--limit", type=int, default=None, help="dry run on the first N pairs")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-box-quiet", action="store_true")
    args = parser.parse_args(argv)

    result = run(
        variants=[v.strip() for v in args.variants.split(",") if v.strip()],
        out_path=args.out,
        limit=args.limit,
        batch_size=args.batch_size,
        threads=args.threads,
        seed=args.seed,
        use_cache=not args.no_cache,
        measure_box=not args.no_box_quiet,
    )
    for name, row in result["variants"].items():
        print(f"  {name:<13} acc={row['pairwise_accuracy']:.4f}  size={row['weight_size_mb']} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
