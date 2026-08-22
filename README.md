# reward-serve

A continuous-batching inference server for my own published reward model,
[`DantheMan124/deberta-preference-reward`](https://huggingface.co/DantheMan124/deberta-preference-reward)
(184M DeBERTa, Bradley-Terry pairwise head), with a load harness, tracing,
metrics, and a written SLO. It runs on one machine, CPU only, and every
number it publishes has a committed artifact and the command that produced it.

I built it to answer a specific question honestly: can I serve a model under
load, see what it is doing, and find out why when it misbehaves? The answer
below is scoped to what has actually been run.

## What has been run

| Claim | Proof |
| --- | --- |
| The batcher batches. 20 concurrent requests → 3 forward passes of 8, 8, 4 | `python scripts/obs_dry_run.py --structure-only` → [`results/obs_dry_run_structure.json`](results/obs_dry_run_structure.json) |
| Scores are identical alone and inside a batch of 8 (≤ 1e-4) | `pytest tests/test_model.py -k identical_alone` |
| A request cancelled in the queue is never forwarded; a request past its deadline is never scored | `pytest tests/test_batcher.py -k "deadline or cancelled"` |
| A full queue answers 429 immediately; a drain loses no in-flight request | `pytest tests/test_server.py -k "429 or drain"` |
| One request's trace names the batch it rode in and every sibling in it | [`results/trace_batch_link_structure.json`](results/trace_batch_link_structure.json) — `batch.forward` carries `batch.size=8` and 8 span links |
| A router and two worker processes come up, split ten requests 5/5, and one trace id spans router → worker → batch | `python scripts/tier_dry_run.py` → [`results/multi-tier-dry-run/dry_run.json`](results/multi-tier-dry-run/dry_run.json) |
| Killing a worker mid-batch loses only the 2 requests already on its wire; the other 187 sent afterwards all succeed, and traffic converges on the survivor in 93 ms | [`results/faults/kill-worker-mid-batch/summary.json`](results/faults/kill-worker-mid-batch/summary.json) |
| A worker made 2 s slow stays `UP` in every health sample — the routing rule, not the health check, shifts load off it, peaking at 100% on A in three sample windows (with 6 router 504s during the fault) | [`results/faults/latency-2s-on-one-worker/summary.json`](results/faults/latency-2s-on-one-worker/summary.json) |
| 1297 rejections (95.2% of offered load) open no circuit breaker and move no worker out of `UP` | [`results/faults/queue-exhaustion/summary.json`](results/faults/queue-exhaustion/summary.json) |
| A wedged worker is never marked down — `/readyz` still answers 200 — but nothing waits past the router's 3 s budget: max observed 3023.9 ms | [`results/faults/hang-worker/summary.json`](results/faults/hang-worker/summary.json) |
| The knee: 72-cell sweep on a gated quiet box — batch size 1 wins everywhere (one run per cell, no intervals; re-run pending) | [`results/2026-08-20-cpu-batch-grid/summary.json`](results/2026-08-20-cpu-batch-grid/summary.json) with `box_quiet.json` beside it |
| int8-dynamic: 0.7026 → 0.5979, 311 flips per 1,000 pairs, ~1.2x speedup — ruled out | [`results/quantization/accuracy.json`](results/quantization/accuracy.json), [`latency.json`](results/quantization/latency.json), [`flip_decomposition.json`](results/quantization/flip_decomposition.json) |
| 213 tests pass and 2 skip with a stated reason, plus `ruff`, `mypy --strict` on both packages, and the vocabulary gate | `pytest -q && ruff check . && mypy --strict reward_serve loadgen && python scripts/vocab_gate.py .` — the same sequence CI runs; what CI does and does not cover is in [`docs/ci.md`](docs/ci.md) |

## What was measured late, and the refusals that preceded it

**The measurements landed on 2026-08-20, on a box the gate finally passed**
(mean CPU 7.93%, GPU 423 MiB — the report sits beside every result it gated).
The headline: **the knee is at batch size 1 — batching never pays on this
CPU.** At 4 rps offered, `MAX_BATCH_SIZE=1` serves 4.65 rps at p99 1.9 s with
zero shed; b32 collapses to 2.12 rps, p99 12.8 s, 43.6% shed, because forward
time grows worse than linearly with batch (207 ms at batch 1 → 7.4 s at mean
batch 18). Full grid, plots and the reasoning: [`docs/knee-analysis.md`](docs/knee-analysis.md).
That run was **one run per cell with no confidence intervals**, and it kept no
per-request records, so its numbers cannot be recomputed and its p99 is the
slowest single request in all 72 cells. The harness now persists every request,
repeats each cell, and puts a bootstrap interval on p99; `scripts/recompute_sweep.py`
regenerates a whole summary from the raw rows. The re-run has not happened —
[`docs/knee-analysis.md`](docs/knee-analysis.md) *Uncertainty* has the method and
the exact command.
The SLO's latency threshold is now measured too — **0.750 s**, derived by the
rule fixed in [`docs/slo.md`](docs/slo.md) before the number existed
(2 × 282 ms max forward + 10 ms wait, rounded up to a bucket edge) — and both
burn-rate alerts are live rather than commented out. The quantization study
gained its latency half: int8-dynamic buys ~1.2x on this CPU, not the 2-4x
the literature reports, because Comet Lake lacks VNNI
([`docs/quantization.md`](docs/quantization.md)). An earlier refused sweep is
kept at [`results/2026-08-18-cpu-batch-grid/`](results/2026-08-18-cpu-batch-grid/) —
failing gate report, no numbers — as the record of what refusing looks like.

The docker stack under `docker/` (server, OTel collector, Prometheus,
Grafana with a committed dashboard) is validated structurally by tests, not
brought up: Docker is not installed on this machine.

The fault runs above ran on a contended box on purpose: they answer behavioural
questions, not latency ones. Every one of their `summary.json` files carries
`"latency_figures_are_indicative_only": true` beside a `box_quiet.json`, set by
the runner rather than by hand, and [`docs/multi-tier.md`](docs/multi-tier.md)
quotes latency from them only as an order of magnitude.

## Five bugs the tests did not catch and the runs did

1. **The batcher forwarded a backlog one item at a time.** Its wait window
   was anchored to the oldest item's enqueue time, so under a backlog the
   window had already expired and the loop broke with a batch of one. The
   unit tests submitted items simultaneously against a fresh timer and never
   saw it; the first live dry run returned `batch_size: 1` on all ten
   requests. Fixed by sweeping the queued backlog before waiting.
2. **Coarse-clock deadline arithmetic collapsed short wait windows.** The
   Windows loop clock advances in ~15.6 ms ticks, so one tick after enqueue
   the computed remaining wait for a sub-resolution `MAX_WAIT_MS` was already
   negative and the batcher flushed a lone item after ~0.18 ms — the sweep
   would have compared wait settings that never differed. (First diagnosed as
   "timers fire early"; a later measurement against `perf_counter` showed
   timers never fire early — 0 of 100 trials — and the early readings were
   the coarse clock unable to see a 5 ms interval. The fix was right, the
   first explanation was not, and both are on the record.) Fixed by
   re-checking the deadline; sub-resolution waits now floor at ~one tick.
3. **Dropped items leaked the observability FIFO.** Timed-out and cancelled
   items were removed before `on_batch_start`, so the queue-depth gauge
   drifted upward forever and later batches linked the wrong request spans.
4. **The router's one retry worked but cost more than the failure.** Connecting
   to the port of a terminated worker took ~2.3 s to be refused on this box, so
   the two retried requests cost 2.54 s and 2.79 s against a p50 of 0.41 s.
   Connect now has its own 250 ms budget separate from the 3 s request budget,
   and retried requests cost 464–516 ms. The pre-fix run is committed at
   `results/faults/kill-worker-mid-batch-before-connect-timeout/`.
5. **A circuit breaker that opened could never close.** Routing filtered on the
   raw breaker state instead of calling the check that moves OPEN to HALF_OPEN,
   so no worker was ever offered its probe and a tripped worker stayed ejected
   for the life of the process. The unit tests passed because they drove the
   breaker directly; the regression test now drives recovery through `pick`.
   Visible in `results/faults/latency-2s-on-one-worker-before-breaker-recovery/`
   as a worker whose share of traffic never returns after its fault is cleared.

## Layout

    reward_serve/   model.py (sequence budget read from the checkpoint config,
                    never hardcoded), batcher.py, server.py, obs.py,
                    router.py + worker.py (the two tiers), faults.py
    faults/         run_faults.py: the four fault-injection scenarios
    loadgen/        open- and closed-loop generator, exact percentiles, gated sweep
    scripts/        box_quiet.py, vocab_gate.py, dry runs, plot_sweep.py,
                    tier_up.py (start/stop router + N workers)
    results/        committed artifacts, each with its box_quiet report
    docs/           contract.md, design-batcher.md, slo.md, observability.md,
                    multi-tier.md (router design + the four fault runs),
                    knee-analysis.md (findings from the 72-cell sweep)
    docker/         compose for server + collector + Prometheus + Grafana

## Run it

    py -3.13 -m venv .venv
    .venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    .venv/Scripts/python -m pip install -e ".[dev]"
    set CUDA_VISIBLE_DEVICES=
    .venv/Scripts/python -m reward_serve --port 8000
    .venv/Scripts/python scripts/dry_run.py --url http://127.0.0.1:8000

Two tiers — one router in front of two worker processes on the same machine:

    .venv/Scripts/python scripts/tier_up.py --workers 2
    .venv/Scripts/python scripts/tier_dry_run.py
    .venv/Scripts/python faults/run_faults.py
    .venv/Scripts/python scripts/tier_up.py --down

## Not done

The quantization study is done — accuracy, flip decomposition, and latency all measured (`docs/quantization.md`). Cancellation after batch
assembly begins still scores the item and discards the result. A batch
abandoned at stop() now fails its callers explicitly rather than hanging them. There is no
request priority and no length bucketing.

On the router tier specifically: the routing rule is not corrected for
fast-failing workers — it concentrated 66.8% of traffic on a worker that was
refusing requests — `/readyz` still cannot see a wedged batching loop, the
worker list is static configuration with no supervisor to restart a killed
worker, and no scenario was run with more than two workers or run twice. The
full list is in [`docs/multi-tier.md`](docs/multi-tier.md). This is not a
service for many users and it is not distributed; it is one box running a
handful of processes.
