# Batch-size knee analysis

**Status: the sweep has NOT been run. This document has no latency or
throughput findings yet.** The harness that produces them is built, tested and
verified end to end against the real server; the measurement was refused by the
quiet-box gate. What that gate saw, and how to produce the missing numbers, is
below. Nothing here is estimated, modelled, or remembered — when the grid runs,
the findings section gets written from the plots, and every number in it will be
readable in `results/2026-08-18-cpu-batch-grid/summary.json`.

## Measurement conditions

Read every future number in this document against these conditions.

From `results/2026-08-18-cpu-batch-grid/environment.json`
(`python scripts/env_report.py results/2026-08-18-cpu-batch-grid/environment.json`):

| | |
|---|---|
| platform | Windows-10-10.0.19045-SP0 |
| CPU | Intel64 Family 6 Model 165 Stepping 5, GenuineIntel |
| cores | 8 physical / 16 logical |
| python | 3.11.15 |
| torch | 2.13.0+cpu |
| transformers | 5.15.0 |
| httpx | 0.28.1 |
| `OMP_NUM_THREADS` | unset (torch picks its own thread count) |
| `TORCH_NUM_THREADS` | unset |
| `CUDA_VISIBLE_DEVICES` | `""` — CPU only, enforced by every entrypoint |
| harness commit | c540166 |

`OMP_NUM_THREADS` being unset is itself a measurement condition: the forward
pass uses whatever thread count torch defaults to on 8 physical cores, and that
default is what the sweep will measure. It is not pinned.

## Why the numbers are missing

Charter rule 3: a latency or throughput sweep runs only on a quiet box, and
contended numbers are never published. The gate is `scripts/box_quiet.py`
(mean CPU < 15% over 30 s, GPU utilization < 10%, no other python process over
20% CPU). It ran, it failed, and the sweep stopped before starting a single
server process.

From `results/2026-08-18-cpu-batch-grid/box_quiet.json`:

| field | value | threshold |
|---|---|---|
| `passed` | `false` | — |
| `cpu_mean_pct` | 19.12 | < 15 |
| `gpu_util_max_pct` | 100 | < 10 |
| `window_s` | 30 | — |
| `measured_at` | 2026-08-18T21:15:34 | — |

The GPU was pinned at 100% by other work on this machine, and the CPU never
settled below the bar across six consecutive checks between 21:08 and 21:15
local time. A latency grid measured against that would be measuring the other
workload as much as the batcher.

The refusal is the artifact. `results/2026-08-18-cpu-batch-grid/` contains the
failing report and the environment record and no `cells.jsonl` and no
`summary.json`, which is exactly what a refused run should leave behind.

## What will be measured

Grid: `MAX_BATCH_SIZE` ∈ {1, 2, 4, 8, 16, 32} × `MAX_WAIT_MS` ∈ {0, 5, 20, 50}
× offered load ∈ {1, 2, 4} requests/s. 72 cells, each with its own freshly
started server process, 15 s of warm-up excluded, 45 s measured, one repeat.

Load is offered **open loop**: arrivals follow a Poisson process at the target
rate and the generator never waits for a response before sending the next
request. A closed-loop generator cannot find a latency knee, because when the
server slows down a closed-loop client slows down with it and the offered load
quietly falls to whatever the server can serve.

The three offered-load levels are low relative to what a hosted service would
see because a single forward pass of this 184M model on this CPU is measured in
hundreds of milliseconds, not microseconds. Calibrating them is part of the
dry-run procedure, not a guess: the run that produced them must itself pass the
quiet gate before its numbers are written down here.

Before the grid, the sweep runs a preflight cell at `MAX_BATCH_SIZE=8` under
offered load and aborts unless the mean achieved batch size exceeds 1. This
server previously had a batcher that forwarded a queued backlog one item at a
time; it answered every request correctly and still reported a `batch_size`
field, so only a live load test could catch it. It is fixed and regression
tested. The preflight exists so that a future regression stops the grid instead
of silently turning 72 cells into 72 measurements of an unbatched server.

## The question this grid answers

At what `MAX_BATCH_SIZE` does p99 latency stop paying for itself in throughput,
at which offered load, and why in terms of the batcher's flush policy?

The flush policy is time-or-size: a batch is forwarded when `MAX_BATCH_SIZE`
items are waiting, or when the oldest waiting item has waited `MAX_WAIT_MS`,
whichever happens first. Which of those two triggers actually fires is a
function of offered load, and it is the mechanism behind any knee:

- When arrivals are sparse relative to the forward time, the size trigger is
  almost never reached, so `MAX_BATCH_SIZE` is inert and `MAX_WAIT_MS` is the
  only setting doing anything — it adds its own value to every request's
  latency and buys whatever partial batching happens to accumulate.
- When arrivals are dense enough that a queue persists across forward passes,
  the size trigger dominates, `MAX_WAIT_MS` becomes inert, and each item's
  latency is queue wait plus a forward pass whose cost depends on the batch it
  rode in.

The knee is where the second regime stops paying. `knee.png` plots that
directly: throughput gain and p99 cost, each relative to the `MAX_BATCH_SIZE=1`
cell in the same panel. A batch size earns its latency only while the gain line
stays above the cost line. The crossing point is the answer, and it is read off
the plot per offered load rather than asserted once for the whole machine.

The per-request records carry `queue_wait_ms` and `forward_ms` measured inside
the server, so the split between "waiting for a batch to fill" and "riding in a
larger forward pass" is attributable rather than inferred.

## Threats to validity

One machine, one CPU, one model, and the GPU deliberately unused — none of this
says anything about how the same batcher would behave on an accelerator, where
a forward pass parallelises across the batch rather than competing for the same
cores. Inputs are synthetic filler text in three length buckets sized in words,
not model tokens, so they exercise the sequence-length dimension only
approximately and carry none of the length distribution of real traffic. The
plan is **one run per cell with no repeats**, so the grid will show no variance
and none will be claimed: any two adjacent cells that differ by less than the
run-to-run noise of this box cannot be told apart by it, and that noise has not
been measured. Warm-up is excluded by wall clock, not by a convergence check.
Each cell restarts the server, which removes cross-cell contamination but means
every cell also pays a cold allocator and a cold first forward pass inside its
warm-up window. Finally, the generator and the server share one box and one
CPU, so at high offered load the load generator competes with the thing it is
measuring; the open-loop arrival timing is guarded by a test, but the
competition is real and is not corrected for.

## Not done

- The 72-cell grid has not run. The quiet gate refused on 2026-08-18.
- No plots exist, because `scripts/plot_sweep.py` reads `summary.json` and
  there is no `summary.json`.
- Offered-load levels have not been calibrated on a quiet box. The values in
  "What will be measured" come from a contended probe and must be re-derived
  from a dry run that passes the gate before any of them are published.
- Cells are not repeated, so run-to-run variance is unmeasured.
- Only the single-pair request form is exercised. The many-pairs form, where
  one request's items can ride in different batches, is untested by this
  harness.

## Reproducing

Run when the box is quiet — the sweep re-checks the gate itself and will refuse
again if it is not:

```
cd D:\ClaudeProjects\reward-serve
.venv\Scripts\python.exe -m loadgen.sweep \
  --run-id 2026-08-18-cpu-batch-grid \
  --server-cmd "D:\ClaudeProjects\reward-serve\.venv\Scripts\python.exe -m reward_serve" \
  --target-label "reward_serve on CPU, DantheMan124/deberta-preference-reward" \
  --batch-sizes 1,2,4,8,16,32 --wait-ms 0,5,20,50 --rps 1,2,4 \
  --duration 45 --warmup 15 --queue-depth 512
.venv\Scripts\python.exe scripts\plot_sweep.py results\2026-08-18-cpu-batch-grid\summary.json
```

The exact command is also written into `summary.json` as `reproduce_command`,
and every cell records its own command, environment overrides and port.
