# Batch-size knee analysis

**Status of the numbers below: measured on 2026-08-20, one run per cell, no
confidence intervals, and not recomputable — that run predates per-request
persistence. The knee finding stands; its precision is unquantified. See
*Uncertainty* for what has been built to fix that and *Reproducing* for the
re-run that will.**

**Historical status line, kept for the record: the sweep has NOT been run. This document has no latency or
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

## Findings (run `2026-08-20-cpu-batch-grid`, quiet box)

The gate passed — mean CPU 7.93%, GPU 423 MiB — and all 72 cells ran
(`results/2026-08-20-cpu-batch-grid/`, one freshly started server per cell,
5 s warm-up excluded, 20 s measured, every number below is in `summary.json`).

> **Read the p99 column with this in mind.** Percentiles here are exact
> nearest-rank, which returns the largest observation whenever `n < 101`. This
> run measured 24–94 successful requests per cell, so **in all 72 cells the
> reported p99 is the single slowest request** (`latency_p99_ms ==
> latency_max_ms`, checkable in `summary.json`). The ordering below is robust —
> batch 1 wins in all 12 rps×wait groups, and the throughput and shed columns
> are unaffected — but no two adjacent p99 values here can be told apart, and
> none has a confidence interval. A re-run with longer windows and repeats is
> specified below; until it lands, treat every p99 as one observation, not an
> estimated tail.

**The knee is at batch size 1. On this CPU, batching never pays.**

At every offered load, `MAX_BATCH_SIZE=1` gives the highest throughput and the
lowest p99 (all cells below at `MAX_WAIT_MS=20`):

| offered | config | throughput | p50 | p99 | shed |
|---|---|---|---|---|---|
| 1 rps | b1 | 1.28 rps | 240 ms | 546 ms | 0% |
| 1 rps | b32 | 1.27 rps | 434 ms | 1,111 ms | 0% |
| 2 rps | b1 | 2.65 rps | 387 ms | 805 ms | 0% |
| 2 rps | b32 | 2.65 rps | 498 ms | 3,020 ms | 0% |
| 4 rps | b1 | **4.65 rps** | 858 ms | 1,899 ms | **0%** |
| 4 rps | b2 | 3.29 rps | 4,917 ms | 5,758 ms | 13.8% |
| 4 rps | b8 | 2.37 rps | 6,184 ms | 7,687 ms | 38.3% |
| 4 rps | b32 | 2.12 rps | 9,489 ms | 12,777 ms | 43.6% |

**Why.** Batching pays when a batch of N forwards in much less than N single
forwards. Here it does not: mean forward time was 207 ms at mean batch 1.0 and
7,431 ms at mean batch 18.2 — worse than linear. An 8-core CPU already
parallelises a single 184M-parameter forward across its cores, so a bigger
batch adds arithmetic without adding hardware, and every request in the batch
then waits for the whole thing. Under pressure that turns into a spiral:
longer forwards grow the queue, the backlog sweep packs bigger batches, and
bigger batches forward slower still — at 4 rps offered, b32 sheds 43.6% of
requests while b1 sheds none.

The `MAX_WAIT_MS` axis is second-order once this dominates: at b1 it is unused
(a batch of one never waits), and at larger caps its effect is swamped by the
forward-time growth.

**What this means for the serving plan.** Run this checkpoint on CPU with
`MAX_BATCH_SIZE=1` — the continuous batcher is still load-bearing there (the
queue, deadlines, cancellation, drain and backpressure all sit in it), but its
packing feature should be off. The batching machinery earns its keep only
where forward time is sublinear in batch size, i.e. on an accelerator with
idle parallelism. That claim is testable the day this runs on one; it is not
claimed here.

Plots: `throughput_vs_batch.png`, `p99_vs_batch.png`, `knee.png` in the run
directory, regenerable with `python scripts/plot_sweep.py <summary.json>`.

## Uncertainty

**Status: the method below is implemented and tested; the numbers it produces
are pending the re-run.** The table above comes from run `2026-08-20`, which
was one run per cell with no per-request records committed, so none of its
figures carries an interval and none of them can be recomputed. That is the gap
this section closes, and it is closed in the harness before it is closed in the
data. Until the re-run lands, read every figure above as a single observation
with unquantified spread.

### What "one run per cell" actually cost

Two specific defects, both checkable in the committed files:

1. **No interval.** `p99 = 1,899 ms` at batch 1, 4 rps is one number from one
   run. Nothing committed says whether a second run would produce 1,850 or
   2,400, so no two adjacent cells in that grid can be told apart.
2. **p99 was the maximum, in every cell.** Percentiles here are exact
   nearest-rank: `p(q) = sorted[ceil(q/100 * n) - 1]`. That returns the largest
   observation whenever `n < 101`. Run `2026-08-20` measured between 24 and 94
   successful requests per cell, so in **all 72 cells** the reported p99 is
   literally the single slowest request of the run — the statistic with the
   worst possible spread. It is checkable in the committed summary:
   `latency_p99_ms == latency_max_ms` in all 72 cells. Any re-run needs enough
   measured requests per cell that p99 stops being the maximum, which is why
   the re-run command below lengthens the measured window.

### Repeats

A cell now runs `--repeats N` times, default 3. A repeat is a fully independent
observation: its own server process started fresh (so the allocator, the thread
pool and the first forward pass are cold again exactly as they were for the
first repeat), its own warm-up window, and its own seed derived from the run's
base seed by `loadgen.stats.repeat_seed`, so the arrival process is a different
draw from the same Poisson distribution rather than a replay of the same one.

Each repeat keeps its own full statistics in `per_repeat`. They are never
collapsed into a single number and thrown away:

- Counts, rates, latency percentiles and the per-request means are computed over
  the **pooled** measured requests of every repeat. These are properties of the
  latency and status distribution, and more samples estimate them better.
- `window_s`, `send_span_s`, `throughput_rps` and `offered_rps_actual` are
  **not** pooled. The repeats are separated in time by a server restart, so a
  pooled wall-clock span would include the gap between them and divide a correct
  success count by the wrong number of seconds. These are computed per repeat
  and averaged, and the per-repeat min and max sit next to the mean under
  `..._repeat_min` / `..._repeat_max`.

### The confidence interval

`summary.json` carries `latency_p99_ci_ms` per cell: a **percentile bootstrap**
95% confidence interval on p99, over the pooled per-request latencies of that
cell's successful requests. The method, in full:

1. Pool the measured successful latencies across the cell's repeats — call it
   *n* observations.
2. Draw *n* of them **with replacement**, 2,000 times.
3. Take the exact nearest-rank p99 of each resample — the same percentile rule
   used everywhere else in this repo, so the interval and the point estimate
   cannot drift apart definitionally.
4. Report the 2.5th and 97.5th percentiles of those 2,000 values as the interval.

It is seeded, so it is reproducible: the seed is `sha256(cell_id + the sorted
repeat seeds)`, and both of those live on every raw row, so a reader holding
only `raw/` computes the identical interval rather than a similar one. A test
asserts that byte-for-byte against the committed run.

**What the interval means.** It is the sampling noise of the percentile: how far
p99 would move if you drew a different set of requests *from the same run
conditions*. It is a statement about how few slow requests a p99 rests on, and
nothing more.

**What it does not mean.** It does not cover run-to-run drift of the machine —
a background process, a thermal state, a different allocator outcome. That is
what the repeats are for, and they are reported separately:
`latency_p99_ms_per_repeat`, `latency_p99_ms_repeat_min`, and
`latency_p99_ms_repeat_max`. When a repeat's p99 falls outside the bootstrap
interval, the cell is flagged `repeats_disagree_beyond_ci: true`. That flag is a
**finding, not a defect to average away**: it means the box moved between
repeats and the difference is larger than sampling noise explains. A cell
carrying it should be read as "this configuration is not reproducible on this
box to better than the per-repeat spread", and any comparison against a
neighbouring cell has to clear that spread, not just the interval.

### Every number is recomputable from raw

`results/<run-id>/raw/<cell_id>.jsonl` holds one line per request — `send_ts`,
`done_ts`, `latency_ms`, `status`, achieved `batch_size`, `queue_wait_ms`,
`forward_ms`, `is_warmup`, plus the cell configuration, repeat index and seed.
Timestamps are written unrounded, because rounding them would put a floor under
how exactly anyone can reproduce the numbers.

**Warm-up rows stay in the file**, flagged `is_warmup`. They are excluded from
every statistic, as they always were, but now that exclusion is verifiable
rather than a claim: recompute with them and the counts change.

```
python scripts/recompute_sweep.py results/<run-id>
```

reads `raw/*.jsonl` and **nothing else** — not `summary.json`, not
`cells.jsonl`, not the server logs — rebuilds every statistic, and only then
opens `summary.json` to check it. Exit code 1 on any mismatch. The generator's
own internal counters (`dropped_no_slot`, `max_inflight_observed`, and the run
configuration echoed into each repeat's stats) cannot be rebuilt from request
rows, because `dropped_no_slot` counts requests that were never sent and so have
no row; those are listed in the report as `not_derivable_from_raw` rather than
silently skipped.

A test runs exactly that against a committed fixture run and asserts equality to
floating-point tolerance, and a second test doctors a p99 in the summary and
asserts the check fails. Nothing else in this document is worth more than that
pair of tests.

## The earlier refusal, kept for the record

Charter rule 3: a latency or throughput sweep runs only on a quiet box, and
contended numbers are never published. The gate is `scripts/box_quiet.py`
(mean CPU < 15% over 30 s; GPU memory < 2 GiB and mean GPU utilization < 20%;
no other python process over 20% CPU). The GPU rule is a 2026-08-18
recalibration: the original max-utilization < 10% ceiling was measured to sit
inside idle desktop-compositing noise (10-21% spikes with nothing running,
`results/box_quiet_calibration.json`) and could never pass while anyone was
logged in. Full disclosure: the published sweep passed the recalibrated gate
(GPU 423 MiB, mean util under the bar) and would have FAILED the original
max-util rule on compositing spikes alone; the CPU rule, which is what
actually distinguishes a loaded box from an idle one here, is unchanged and
was passed at 7.93%. It ran, it failed, and the sweep stopped before starting a single
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

## The grid, as specified before the run

Grid: `MAX_BATCH_SIZE` ∈ {1, 2, 4, 8, 16, 32} × `MAX_WAIT_MS` ∈ {0, 5, 20, 50}
× offered load ∈ {1, 2, 4} requests/s. 72 cells, each repeat with its own
freshly started server process. The published run `2026-08-20` used 5 s of
warm-up and 20 s measured with one repeat — not the 15 s / 45 s this paragraph
previously claimed, which is readable in `summary.json` under `load` and is
corrected here rather than quietly fixed. The re-run uses 3 repeats and a longer
measured window, for the reason given under *Uncertainty*: below 101 measured
successes, nearest-rank p99 is just the slowest request.

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
approximately and carry none of the length distribution of real traffic.

~~The plan is one run per cell with no repeats, so the grid will show no
variance and none will be claimed.~~ **Retracted as a plan, still true of the
published numbers.** The harness now runs `--repeats 3` per cell and reports a
bootstrap 95% interval on p99 plus the min/max across repeats (see
*Uncertainty*), and every request is persisted so any of it can be recomputed.
Run `2026-08-20` — the source of every figure in *Findings* — predates that: it
is one run per cell, it committed no per-request rows, and its run-to-run noise
is unmeasured. Those figures therefore still carry the original caveat in full,
and two adjacent cells of that grid differing by less than an unmeasured noise
band cannot be told apart. The caveat is retired for the numbers, not just for
the plan, only when the re-run in *Reproducing* has landed and this paragraph is
rewritten against its intervals. Separately, that run's p99 is the maximum
observation in all 72 cells (n never reached 101; see *Uncertainty*), so it is
the least stable statistic the grid could have reported.

Warm-up is excluded by wall clock, not by a convergence check.
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
- **The re-run has not happened.** Repeats, intervals and raw persistence are
  built and tested, but the only sweep against the real server on a quiet box is
  still `2026-08-20`, which has none of them. Every figure under *Findings* is
  therefore still a single unrepeated observation with no committed raw rows, and
  cannot be recomputed by `scripts/recompute_sweep.py` — running it against that
  directory refuses, by design, rather than pretending.
- No cell of the published grid has a confidence interval, because intervals are
  computed at run time from the requests that were actually sent and that run
  did not keep them. They cannot be back-filled from order statistics.
- Only the single-pair request form is exercised. The many-pairs form, where
  one request's items can ride in different batches, is untested by this
  harness.

## Reproducing

Run when the box is quiet — the sweep re-checks the gate itself and will refuse
again if it is not:

```
cd <repo root>
.venv\Scripts\python.exe -m loadgen.sweep \
  --run-id 2026-08-21-cpu-batch-grid-repeats \
  --server-cmd ".venv\Scripts\python.exe -m reward_serve" \
  --target-label "reward_serve on CPU, DantheMan124/deberta-preference-reward" \
  --batch-sizes 1,2,4,8,16,32 --wait-ms 0,5,20,50 --rps 1,2,4 \
  --duration 120 --warmup 15 --repeats 3 --queue-depth 512
.venv\Scripts\python.exe scripts\recompute_sweep.py results\2026-08-21-cpu-batch-grid-repeats
.venv\Scripts\python.exe scripts\plot_sweep.py results\2026-08-21-cpu-batch-grid-repeats\summary.json
```

`--duration 120` and `--repeats 3` are not arbitrary. At the lowest offered load,
1 rps, 120 measured seconds gives about 120 successes per repeat, which clears
the `n >= 101` needed for nearest-rank p99 to stop being the slowest request,
and pools to roughly 360 across three repeats so the bootstrap interval rests on
something. The cost is wall time: 72 cells x 3 repeats x (120 s + 15 s warm-up +
about 9 s of server start, measured from run `2026-08-20`) is roughly 8.6 hours,
so this is a detached overnight run, not an interactive one.

The recompute step is not optional decoration. If it does not exit 0, the
summary does not describe the requests that were sent, and nothing from the run
gets published.

The exact command is also written into `summary.json` as `reproduce_command`,
alongside `recompute_command`, and every repeat records its own command,
environment overrides, seed and port.

### Smoke runs are not measurements

`--smoke` exists to exercise the code path in seconds on a box that is known to
be busy. It is the only thing that lets `--no-box-quiet` through, and asking for
that combination without `--smoke` is refused. A smoke run stamps `"smoke":
true, "not_a_measurement": true` into `summary.json`, into every cell, and into
the `box_quiet.json` that records the gate was never run.
`results/2026-08-21-smoke-not-a-measurement/` is one such run, committed as the
fixture the recompute test regenerates. It measures a sleep in
`tests/stub_server.py`. No number in it describes the model, and it must never
be quoted as though it did.
