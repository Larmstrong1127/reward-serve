"""Ten-request CPU dry run that the SLO threshold in docs/slo.md is derived from.

This is deliberately the smallest honest measurement: the real checkpoint, on
this machine, on CPU, through the hook-instrumented stub batcher, ten requests.
It is not a sweep and it is not a load test. Its only job is to answer "what
does one forward pass cost here", so that the latency SLO threshold is a number
read off the box instead of a number picked because it sounded good.

Run:

    cd D:\\ClaudeProjects\\reward-serve-wt-observability
    D:\\ClaudeProjects\\reward-serve\\.venv\\Scripts\\python.exe scripts/forward_dry_run.py

Writes results/forward_dry_run.json and results/box_quiet_forward_dry_run.json.
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reward_serve import obs  # noqa: E402
from scripts.box_quiet import require_quiet  # noqa: E402
sys.path.insert(0, str(ROOT / "tests"))
from stub_batcher import StubBatcher  # noqa: E402

MODEL_ID = os.environ.get("MODEL_ID", "DantheMan124/deberta-preference-reward")
N_REQUESTS = 10

PAIRS = [
    ("How do I reverse a list?", "Use the reversed() builtin or a [::-1] slice."),
    ("How do I reverse a list?", "You cannot reverse a list."),
    ("What is a hash map?", "A structure mapping keys to values in constant time."),
    ("What is a hash map?", "It is a kind of sandwich."),
    ("Explain recursion.", "A function defined in terms of a smaller call to itself."),
    ("Explain recursion.", "Recursion is when you repeat yourself a lot."),
    ("How do I read a file?", "Open it with a context manager and call read()."),
    ("How do I read a file?", "Files cannot be read from code."),
    ("What does a compiler do?", "It translates source text into another language."),
    ("What does a compiler do?", "It compiles, obviously."),
]


def main() -> int:
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    quiet = require_quiet(results_dir / "box_quiet_forward_dry_run.json")

    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(min(8, os.cpu_count() or 4))

    load_started = time.perf_counter()
    config = AutoConfig.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()
    load_s = time.perf_counter() - load_started

    # Rule 6 of the charter: the sequence budget comes from the checkpoint, never
    # from a literal in this file.
    max_length = int(
        min(
            getattr(tokenizer, "model_max_length", 10**9),
            getattr(config, "max_position_embeddings", 10**9),
        )
    )
    temperature = float(getattr(config, "reward_temperature", 1.1668))

    forward_calls: list[dict[str, float]] = []

    def forward(payloads: list[tuple[str, str]]) -> list[float]:
        prompts = [p for p, _ in payloads]
        responses = [r for _, r in payloads]
        enc = tokenizer(
            prompts,
            responses,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        started = time.perf_counter()
        with torch.no_grad():
            logits = model(**enc).logits
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        forward_calls.append(
            {
                "batch_size": float(len(payloads)),
                "forward_ms": round(elapsed_ms, 3),
                "padded_tokens": float(enc["input_ids"].shape[1]),
            }
        )
        return [float(v) / temperature for v in logits[:, 0]]

    metrics = obs.setup_metrics()
    tracer = obs.setup_tracing("reward-serve-dry-run", console=False)
    hooks = obs.BatcherHooks(metrics, tracer=tracer)

    latencies_ms: list[float] = []

    async def run() -> list[float]:
        batcher = StubBatcher(
            forward=forward,
            max_batch_size=int(os.environ.get("MAX_BATCH_SIZE", "8")),
            max_wait_ms=float(os.environ.get("MAX_WAIT_MS", "10")),
            on_enqueue=hooks.on_enqueue,
            on_batch_start=hooks.on_batch_start,
            on_batch_end=hooks.on_batch_end,
            on_complete=hooks.on_complete,
        )
        await batcher.start()

        async def one(pair: tuple[str, str]) -> float:
            with hooks.request_span("score.request", **{"rs.pairs": 1}):
                started = time.perf_counter()
                score = await batcher.submit(pair)
                latencies_ms.append((time.perf_counter() - started) * 1000.0)
                return score

        # Requests arrive one at a time, as a real client would open them; the
        # batcher packs whatever has accumulated.
        out = await asyncio.gather(*[one(p) for p in PAIRS[:N_REQUESTS]])
        await batcher.stop()
        return list(out)

    scores = asyncio.run(run())

    fwd_ms = [c["forward_ms"] for c in forward_calls]
    per_item_ms = [c["forward_ms"] / c["batch_size"] for c in forward_calls]
    report = {
        "model_id": MODEL_ID,
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "max_length_from_config": max_length,
        "reward_temperature": temperature,
        "model_load_s": round(load_s, 3),
        "n_requests": N_REQUESTS,
        "forward_calls": forward_calls,
        "forward_ms": {
            "n": len(fwd_ms),
            "min": round(min(fwd_ms), 3),
            "median": round(statistics.median(fwd_ms), 3),
            "max": round(max(fwd_ms), 3),
        },
        "per_item_forward_ms": {
            "min": round(min(per_item_ms), 3),
            "median": round(statistics.median(per_item_ms), 3),
            "max": round(max(per_item_ms), 3),
        },
        "client_latency_ms": {
            "n": len(latencies_ms),
            "min": round(min(latencies_ms), 3),
            "median": round(statistics.median(latencies_ms), 3),
            "max": round(max(latencies_ms), 3),
        },
        "achieved_batch_sizes": sorted({int(c["batch_size"]) for c in forward_calls}),
        "scores_sample": [round(s, 4) for s in scores[:4]],
        "box_quiet": quiet,
        "command": "python scripts/forward_dry_run.py",
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_path = results_dir / "forward_dry_run.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    scrape, _ = obs.metrics_payload(metrics)
    (results_dir / "dry_run_metrics.prom").write_text(scrape.decode("utf-8"), encoding="utf-8")

    print(json.dumps(report["forward_ms"], indent=2))
    print(json.dumps(report["client_latency_ms"], indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
