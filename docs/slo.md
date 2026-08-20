# reward-serve SLO

Scope: one server process on one machine, serving
`DantheMan124/deberta-preference-reward` on CPU. There is no second node and no
failover, so this document describes what one process owes its callers and how
much of that it is allowed to fail to deliver before work stops and someone
fixes it.

Window: **30 days, rolling.**

## 1. The two SLIs

The two SLIs deliberately count different things, and mixing them up would make
both wrong:

- **Availability** is measured from `rs_requests_total`, which counts **scored
  items**. A Form B request carrying eight pairs contributes eight, because the
  item is the batcher's unit of work and the unit that can succeed or fail.
- **Latency** is measured from `rs_request_latency_seconds`, which is observed
  once per **HTTP request**, at the middleware. That is what a caller waits on:
  a Form B request is not served until its last pair is, so charging it eight
  separate latency samples would flatter the number by counting the fast pairs
  eight times over.

Neither ratio therefore has to match the other's denominator, and neither should
be compared against a client-side request count without knowing which is which.

### Availability

The share of items that reached a terminal state other than a server fault or a
timeout. Concretely: `status="ok"` over all terminal statuses. `queue_full`
counts **against** availability, deliberately — a 429 is a promise the server
declined to keep, and hiding load-shedding outside the SLI is the classic way to
make a dashboard green while callers are unhappy.

```promql
sum(rate(rs_requests_total{status="ok"}[30d]))
  /
sum(rate(rs_requests_total[30d]))
```

**Target: 99.5%.**

### Latency

The share of HTTP scoring requests whose end-to-end server-side latency was
under a fixed threshold. Not a percentile of a percentile: the numerator is a
bucket count, so the number is exact rather than interpolated.

```promql
sum(rate(rs_request_latency_seconds_bucket{le="0.750"}[30d]))
  /
sum(rate(rs_request_latency_seconds_count[30d]))
```

**Target: 99% of requests under the threshold.**

## 2. Where the latency threshold comes from

A threshold picked because it is a round number is decoration. This one is
derived from the cost of a forward pass on this machine, by this rule:

    threshold = the smallest LATENCY_BUCKETS edge >= (2 x p_max_forward_ms) + MAX_WAIT_MS

where `p_max_forward_ms` is the slowest forward pass observed in the dry run and
`MAX_WAIT_MS` is the batcher's flush timer (default 10 ms). The factor of two is
not margin for its own sake, it is the worst case the batcher actually produces:
a request that arrives just after a pass has begun waits out that whole pass,
then rides the next one, having also paid the flush timer. A threshold below
that would call the batcher's normal behaviour a failure; a threshold far above
it would stop the SLO noticing anything short of a total stall. Rounding **up to
a bucket edge** matters —
the SLI is a bucket count, so a threshold that is not an edge cannot be measured
without interpolating, which would put an estimate at the centre of the error
budget.

The measurement command is:

    cd D:\ClaudeProjects\reward-serve-wt-observability
    D:\ClaudeProjects\reward-serve\.venv\Scripts\python.exe scripts/obs_dry_run.py

It runs the real server, the real batcher and the real checkpoint with obs
wired in, fires twenty concurrent requests so batches actually form, refuses to
start on a contended box (`scripts/box_quiet.py`, charter rule 3), and writes
`results/obs_dry_run.json` next to the box-quiet report that gated it.

> **Status: measured 2026-08-20 (threshold 0.750 s — see above).** Before that,
> twenty-two attempts were refused by the
> box-quiet gate, for two different reasons, and the second matters more than
> the first.
>
> For the first fourteen the machine was genuinely busy: GPU utilisation pinned
> at 100% with 22 GB resident, from other work that owns the GPU here, and host
> CPU between 20% and 72% — partly that work, partly the other agents building
> this repo on the same box, which the gate correctly reports as hot python
> processes. That is the gate doing its job.
>
> Then the GPU work ended — utilisation fell to 10%, memory to 458 MiB — and the
> gate kept refusing. `results/box_quiet_calibration.json`, produced by
> `python scripts/box_quiet_calibration.py 6`, measures why: across six
> 30-second windows the GPU reading ranged 10–21% with a median of 11.5%, and
> **not one sample fell under the 10% ceiling**. The gate takes the *maximum*
> GPU utilisation over the window, and a single frame of desktop compositing
> from a browser or a chat client clears 10%. As written, that condition cannot
> be met on this machine while anyone is logged in: it does not gate a
> measurement, it blocks one — permanently, and for the load sweeps in
> `loadgen/` exactly as much as for this document. That is not a prediction:
> `results/2026-08-18-cpu-batch-grid/` on main holds a box-quiet report reading
> `"passed": false` and no sweep data beside it, so the load harness hit the
> same wall independently. The same run caught host CPU
> at 54–75%, which is real contention from the sibling agents, and there the
> gate is working correctly.
>
> Both are recorded rather than worked around. I have not touched
> `scripts/box_quiet.py`: charter rule 3 is binding, the file is shared with
> **Measured 2026-08-20.** The box-quiet gate passed (mean CPU 11.72%, GPU
> 436 MiB / 8.63% mean; report inside `results/obs_dry_run.json`) and the dry
> run recorded three real forward passes: min 218 ms, median 235 ms, **max
> 282 ms** at `MAX_WAIT_MS=10`. The fixed rule — smallest `LATENCY_BUCKETS`
> edge at or above `2 x max_forward + MAX_WAIT` = 2x282+10 = **574 ms** — rounds
> up to the **0.750 s** bucket edge. That is the SLI threshold below, the
> `le=` edge in the burn-rate rules, and the dashboard constant; a test keeps
> the three in agreement. The earlier refusals that held this number back are
> preserved in the git history of this paragraph.

## 3. Error budget

At a 99.5% availability target over a 30-day window:

- **In time.** 30 days = 43,200 minutes. Budget = 0.5% = **216 minutes**, or
  3 hours 36 minutes of total unavailability per 30 days. Since this is one
  process on one machine, that budget is also the entire maintenance allowance:
  a restart to pick up a new checkpoint spends from the same 216 minutes as an
  outage does, because to a caller they are the same event.
- **In items.** Budget = 0.5% of items served, so **5,000 failed items per
  1,000,000 served**. At a reference rate of 1 item/second the 30-day window
  holds 2,592,000 items and the budget is **12,960 items**. That reference rate
  is an assumption used to make the number concrete, not a measured throughput;
  no load sweep has been run yet, so the honest form of the budget is the ratio,
  and the live count is:

```promql
0.005 * sum(increase(rs_requests_total[30d]))
```

with what remains of it:

```promql
1 - (
  (1 - (sum(rate(rs_requests_total{status="ok"}[30d])) / sum(rate(rs_requests_total[30d]))))
  / 0.005
)
```

The latency budget is 1% of items over the same window: **10,000 late items per
1,000,000 served**, or 25,920 at the reference rate. It is a separate budget
from the availability one. A service that is up and slow is failing, and
merging the two lets a latency regression hide inside an availability surplus.

## 4. Burn-rate alerts

Two windows per alert, following the multi-window multi-burn pattern: the long
window decides that something is wrong, and the short window decides that it is
still wrong right now. Without the short window an alert keeps firing for an
hour after a five-minute blip has ended; without the long one a single bad
second pages someone.

| alert | windows | multiplier | budget consumed before it fires | action |
|---|---|---|---|---|
| fast | 1h **and** 5m | **14.4x** | 2% of the 30-day budget | page |
| slow | 6h **and** 30m | **6x** | 5% of the 30-day budget | ticket |

Where the multipliers come from: burning at 1x exhausts the budget in exactly
30 days. 14.4x exhausts it in 30/14.4 ≈ 2.1 days, and over a 1-hour window that
rate has already spent 1h/720h × 14.4 = 2% of the month's budget — enough to
justify waking someone. 6x exhausts it in 5 days, and over 6 hours has spent
6/720 × 6 = 5%; that is a real leak but not a 3am one.

### Availability burn

Fast (page):

```promql
(
  (1 - (sum(rate(rs_requests_total{status="ok"}[1h])) / sum(rate(rs_requests_total[1h])))) > (14.4 * 0.005)
)
and
(
  (1 - (sum(rate(rs_requests_total{status="ok"}[5m])) / sum(rate(rs_requests_total[5m])))) > (14.4 * 0.005)
)
```

Slow (ticket):

```promql
(
  (1 - (sum(rate(rs_requests_total{status="ok"}[6h])) / sum(rate(rs_requests_total[6h])))) > (6 * 0.005)
)
and
(
  (1 - (sum(rate(rs_requests_total{status="ok"}[30m])) / sum(rate(rs_requests_total[30m])))) > (6 * 0.005)
)
```

### Latency burn

Fast (page):

```promql
(
  (1 - (sum(rate(rs_request_latency_seconds_bucket{le="0.750"}[1h])) / sum(rate(rs_request_latency_seconds_count[1h])))) > (14.4 * 0.01)
)
and
(
  (1 - (sum(rate(rs_request_latency_seconds_bucket{le="0.750"}[5m])) / sum(rate(rs_request_latency_seconds_count[5m])))) > (14.4 * 0.01)
)
```

Slow (ticket):

```promql
(
  (1 - (sum(rate(rs_request_latency_seconds_bucket{le="0.750"}[6h])) / sum(rate(rs_request_latency_seconds_count[6h])))) > (6 * 0.01)
)
and
(
  (1 - (sum(rate(rs_request_latency_seconds_bucket{le="0.750"}[30m])) / sum(rate(rs_request_latency_seconds_count[30m])))) > (6 * 0.01)
)
```

The alert rules are committed at `docker/rules/slo_burn.yml`; the availability
pair is live, and the latency pair is present but commented out until the
threshold above is measured, because a rule referring to a bucket edge that has
not been chosen would silently match nothing and look healthy.

The dashboard panels "Fast burn", "Slow burn" and "Latency burn" plot these same
expressions divided by their budget ratio, so the y-axis reads directly in
multiples of 1x and the alert thresholds are horizontal lines at 14.4 and 6.

## 5. What this SLO deliberately does not cover

It says nothing about whether the scores are any good. A server can return a
well-formed number in 40 milliseconds for every request in the window and be
ranking responses no better than a coin flip; the pairwise accuracy of the
checkpoint (0.7026 on UltraFeedback test_prefs, N=1987) is a property of the
model card, is measured offline, and is completely invisible to every query in
this document. Nor does it cover throughput: nothing here promises a request
rate, so a server that meets both targets while accepting three items a second
is compliant. It excludes 422s, since a malformed request is the caller's fault
and counting it would let a broken client drain a budget the server cannot
defend. It excludes anything before the socket — client-side time, network,
retries — because this process cannot see it. And it makes no availability
promise about the machine: the process can only fail requests while it is
running, so an unplanned host reboot spends budget without any of these series
recording a single failed item. That last gap is the widest one, and it is the
reason the 216 minutes is treated as covering maintenance too.

## Not done

- No load sweep has been run, so the request-count form of the error budget
  rests on a stated reference rate rather than on measured traffic.
- The latency threshold is not measured (see section 2).
- No alertmanager routes these rules to a person.
- No burn-rate alert exists on queue depth or batch size; they are diagnostic
  panels, not objectives.
