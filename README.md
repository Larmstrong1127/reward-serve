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
| Scores are identical alone and inside a batch of 8 (≤ 1e-4) | `pytest tests/test_model.py -k parity` |
| A request cancelled in the queue is never forwarded; a request past its deadline is never scored | `pytest tests/test_batcher.py -k "deadline or cancelled"` |
| A full queue answers 429 immediately; a drain loses no in-flight request | `pytest tests/test_server.py -k "429 or drain"` |
| One request's trace names the batch it rode in and every sibling in it | [`results/trace_batch_link_structure.json`](results/trace_batch_link_structure.json) — `batch.forward` carries `batch.size=8` and 8 span links |
| 98 tests, `ruff`, `mypy --strict`, and the vocabulary gate pass | `pytest -q && ruff check . && mypy --strict reward_serve loadgen && python scripts/vocab_gate.py .` |

## What has not been run — and why it is not in the table

**No latency or throughput numbers are published yet.** The load sweep
(`python -m loadgen.sweep`) refuses to start unless
[`scripts/box_quiet.py`](scripts/box_quiet.py) passes — CPU under 15%, GPU
under 10%, no other hot Python process, for 30 seconds — and it has not
passed on this machine, because a render workload owns the GPU on the
evenings I have been building. The refusal is itself committed:
[`results/2026-08-18-cpu-batch-grid/box_quiet.json`](results/2026-08-18-cpu-batch-grid/box_quiet.json)
holds the failing report and there is deliberately no `summary.json` beside
it. Timings that leaked into dry runs on the contended box are redacted from
the artifacts for the same reason. The SLO in [`docs/slo.md`](docs/slo.md)
therefore ships its availability half complete and its latency threshold as
an explicit placeholder with the derivation rule fixed; a test fails if the
doc and the dashboard ever stop agreeing that it is provisional.

The docker stack under `docker/` (server, OTel collector, Prometheus,
Grafana with a committed dashboard) is validated structurally by tests, not
brought up: Docker is not installed on this machine.

## Three bugs the tests did not catch and the runs did

1. **The batcher forwarded a backlog one item at a time.** Its wait window
   was anchored to the oldest item's enqueue time, so under a backlog the
   window had already expired and the loop broke with a batch of one. The
   unit tests submitted items simultaneously against a fresh timer and never
   saw it; the first live dry run returned `batch_size: 1` on all ten
   requests. Fixed by sweeping the queued backlog before waiting.
2. **Windows asyncio timers fire early.** The loop clock here has 15.625 ms
   resolution and a 5 ms `wait_for` returned in 0.05 ms in 20 of 20 trials.
   Treating that timeout as the deadline collapsed every `MAX_WAIT_MS` under
   ~50 ms to zero — the sweep would have compared wait settings that never
   differed. Fixed by re-checking the deadline against the loop clock; the
   regression test measures a 0.18 ms mean hold on the old code.
3. **Dropped items leaked the observability FIFO.** Timed-out and cancelled
   items were removed before `on_batch_start`, so the queue-depth gauge
   drifted upward forever and later batches linked the wrong request spans.

## Layout

    reward_serve/   model.py (sequence budget read from the checkpoint config,
                    never hardcoded), batcher.py, server.py, obs.py
    loadgen/        open- and closed-loop generator, exact percentiles, gated sweep
    scripts/        box_quiet.py, vocab_gate.py, dry runs, plot_sweep.py
    results/        committed artifacts, each with its box_quiet report
    docs/           contract.md, design-batcher.md, slo.md, observability.md,
                    knee-analysis.md (method written; findings pending the sweep)
    docker/         compose for server + collector + Prometheus + Grafana

## Run it

    py -3.13 -m venv .venv
    .venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    .venv/Scripts/python -m pip install -e ".[dev]"
    set CUDA_VISIBLE_DEVICES=
    .venv/Scripts/python -m reward_serve --port 8000
    .venv/Scripts/python scripts/dry_run.py --url http://127.0.0.1:8000

## Not done

Multi-process serving, fault injection, incident write-ups, and a
quantization study are planned and not started. Cancellation after batch
assembly begins still scores the item and discards the result. There is one
worker process, no request priority, and no length bucketing. This is not a
service for many users and it is not distributed; it is one box.
