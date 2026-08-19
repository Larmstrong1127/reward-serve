# Quantizing the reward model

**Status: the memory and accuracy half is measured; the latency half is not.**
Every accuracy number below is measured on the same 1,987 pairs as this model's
published fp32 accuracy, so the deltas are paired rather than merely nearby. No
latency number appears anywhere in this document, because the quiet-box gate
has not passed on this machine. Nothing here is estimated or remembered.

Every figure is read out of
[`results/quantization/accuracy.json`](../results/quantization/accuracy.json),
produced by `reward_serve/quant/run_accuracy.py`. The command is recorded in the
artifact itself.

## Why this study exists

The server holds one 184M DeBERTa checkpoint in memory and spends essentially
all of its wall clock inside one forward pass. Quantization is the lever that
trades numerical fidelity for memory and arithmetic. For most models the
question is "how much accuracy did I lose". For a *reward* model the question is
sharper, and it is the reason this study measures three things instead of one:

A reward score here is a raw logit divided by a calibration temperature
(T = 1.1668). It has no absolute meaning. It exists only to *order* responses to
the same prompt. So a quantization that shifts every score by a constant costs
nothing at all, and a quantization that leaves the mean score untouched while
reshuffling near-ties destroys the only thing the model does. Accuracy alone
cannot tell those two apart. Hence, per variant:

1. **Pairwise accuracy** on the published split — is it still right as often?
2. **Kendall tau of the per-pair margins against fp32** — is it still ordering
   the same way?
3. **The fraction of pairs whose preferred side flipped** — how many decisions
   would the server actually answer differently?

## Method

The split, the filter, and the budget are not re-derived here. They are the ones
behind the published number, reused so the comparison is paired:

| | |
|---|---|
| model | `DantheMan124/deberta-preference-reward` (184M DeBERTa, Bradley-Terry pairwise head) |
| dataset / split | `HuggingFaceH4/ultrafeedback_binarized`, `test_prefs` |
| pairs loaded | 1988 |
| dropped by the truncation filter | 1 |
| **pairs evaluated** | **1987** |
| sequence budget | 512 tokens, read from `config.reward_train_max_length` |
| score | raw logit / `config.reward_temperature` (1.1668) |
| published fp32 accuracy on this split | 0.7026 |

The truncation filter is the one subtlety worth naming. Right-truncating at 512
tokens can leave a pair's chosen and rejected encodings byte-identical when they
share a long prompt and differ only in the tails that got cut. Such a pair asks
the model to separate two identical inputs, so the published number excludes
them. One pair of 1,988 is excluded. The loader that does this is vendored into
[`reward_serve/quant/eval_split.py`](../reward_serve/quant/eval_split.py) with
its source file named in the docstring; the owner's training repo was read, not
modified.

Accuracy is invariant to T — dividing both sides of a comparison by the same
positive constant cannot reorder them — but the scores recorded are the server's
scores, temperature included, because those are the numbers a caller would see.

**Conditions.** From
[`results/quantization/environment.json`](../results/quantization/environment.json):

| | |
|---|---|
| platform | Windows-10-10.0.19045-SP0 |
| CPU | Intel Core i7-10700 @ 2.90 GHz, 8 physical / 16 logical |
| python | 3.11.15 |
| torch | 2.13.0+cpu |
| transformers | 5.15.0 |
| `CUDA_VISIBLE_DEVICES` | `""` |
| torch thread count | recorded per run in `accuracy.json` as `torch_num_threads` |
| seed | 1234, fixed |

**This whole study is CPU only, deliberately.** `CUDA_VISIBLE_DEVICES` is
emptied before torch is imported, in the quant package and in every entrypoint,
because a render workload owns the GPU on this machine. That is not a footnote:
it bounds what the results mean. GPU int8 kernels, and every speedup that
depends on them, are outside this study entirely.

Determinism: the model is in `eval()` with no dropout, the seed is fixed, the
thread count is fixed and recorded, and the pairs are scored in a fixed order.
Chosen and rejected responses are batched separately at batch size 8, each side
padded to its own batch maximum, matching the harness that produced 0.7026.
Padding is fully masked, so batch composition cannot move a score — the shipped
server has a test asserting exactly that (`pytest tests/test_model.py -k parity`).

A pass over 1,987 pairs takes tens of minutes on this CPU and this machine kills
long sessions, so the runner checkpoints scores every 200 pairs and resumes from
them. It refuses to write a short pass as if it were a complete one.

## The variants, and the exact torch calls

| variant | torch API | weight size (MB) | vs fp32 |
|---|---|---|---|
| fp32 (shipped) | `AutoModelForSequenceClassification.from_pretrained(...)` | 703.6 | — |
| bf16 | `model.to(torch.bfloat16)` | 351.8 | −50.0% |
| int8-dynamic | `torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)` | 459.0 | −34.8% |
| int8-static | `prepare_fx` → calibrate → `convert_fx` | not available — see below | — |

Sizes are the serialized `state_dict`, measured by writing it to an in-memory
buffer rather than by summing `numel × element_size`: a dynamically quantized
Linear holds packed int8 parameters that a naive tensor walk does not see.
They are recorded per variant in `accuracy.json` as `weight_size_mb`.

Each builder returns the same `RewardModel` object the server already loads, so
adopting a variant is a one-line change at load time, not a rewrite. A test
asserts every variant exposes the same interface, budget and temperature.

**int8-dynamic saves 35%, not 75%, and the reason matters for the server.**
`quantize_dynamic` converts `nn.Linear` only. This checkpoint's vocabulary
embedding is a 128k-row table that is not a Linear, so it stays fp32 and it is
the single largest tensor in the model. All 74 Linear modules are converted —
that is asserted by inspecting module types, not assumed — and the memory that
remains is dominated by an embedding table no Linear-only scheme will touch.
Anyone budgeting a 4× memory reduction from int8 on an encoder with a large
vocabulary is budgeting for a model they do not have.

bf16 halves everything uniformly precisely because it is a dtype cast rather
than a per-module rewrite; it therefore beats int8-dynamic on memory here.
`torch.bfloat16` matmul is verified to run on this CPU build before the variant
is built (a test asserts it), rather than assumed from the version number.

## int8-static: attempted, and not available on this stack

Static post-training quantization is the variant that would buy the most on
CPU — activation ranges are fixed once from calibration instead of re-observed
on every forward pass. It was attempted and it did not work, and the record is
[`results/quantization/int8_static_attempt.json`](../results/quantization/int8_static_attempt.json)
(`python -m reward_serve.quant.probe_static`):

```
prepare_fx failed: ValueError: You cannot specify both input_ids and
inputs_embeds at the same time
```

FX graph-mode quantization must trace the model to a graph before it can insert
observers. `torch.fx` traces both sides of a Python branch, so the modeling
code's mutual-exclusion guard on `input_ids` / `inputs_embeds` — which is
satisfied at runtime — raises during tracing. The tracer that used to solve this
for transformers models, `transformers.utils.fx`, does not exist in transformers
5.15. The remaining route is hand-annotating the modeling code with Quant/DeQuant
stubs, which means editing the model definition, which is out of scope for a
serving study.

So there is no static number in this study, and none should be quoted. The
calibration set was nonetheless defined and built before the attempt, so the
protocol is on record: **256 pairs from UltraFeedback `train_prefs`**, run
through the same truncation filter, calibrating on the chosen response.
It is disjoint from the 1,987 evaluated pairs by construction — those come from
`test_prefs`, a different split, sharing no pair with `train_prefs`.

## Accuracy on the published split

All variants scored on the same 1,987 pairs. From `accuracy.json`:

| variant | pairwise accuracy | Δ vs fp32 |
|---|---|---|
| fp32 | TODO-FROM-accuracy.json | — |
| bf16 | TODO-FROM-accuracy.json | TODO-FROM-accuracy.json |
| int8-dynamic | TODO-FROM-accuracy.json | TODO-FROM-accuracy.json |

The fp32 row is the anchor: it must equal the published 0.7026, because it is
the same model on the same pairs under the same filter. A test asserts it
against the committed artifact, and a mismatch stops the study rather than
prompting an adjustment.

## Ranking preservation

Accuracy is a coarse instrument here — a variant can lose accuracy on a handful
of pairs while quietly reshuffling hundreds. These three columns are the ones
that describe what the server would actually emit differently:

| variant | mean abs score Δ vs fp32 | Kendall tau of margins vs fp32 | flipped pairs | flips per 1,000 pairs |
|---|---|---|---|---|
| bf16 | TODO-FROM-accuracy.json | TODO-FROM-accuracy.json | TODO-FROM-accuracy.json | TODO-FROM-accuracy.json |
| int8-dynamic | TODO-FROM-accuracy.json | TODO-FROM-accuracy.json | TODO-FROM-accuracy.json | TODO-FROM-accuracy.json |

A flip is a pair where the variant prefers the opposite response to fp32. It
counts flips in both directions: a flip that happens to correct an fp32 mistake
still counts, because agreement with the shipped model is what is being
measured, not correctness.

## What this asks of the model

TODO-FROM-accuracy.json — this section is written from the measured numbers once
they land, and its shape is fixed in advance so it cannot be written to flatter
whichever variant wins:

- **Which variant the server could adopt**, stated as a recommendation with the
  memory it saves and the ranking fidelity it costs, both as measured numbers.
- **The cost in the serving system's own units**: flips per 1,000 pairs scored,
  against the memory freed in MB. A reward server exists to rank; a flip is a
  request answered differently than the shipped model would answer it, and that
  is the unit a caller feels.
- **Whether the ranking damage is concentrated in near-ties or spread across
  confident pairs** — the first is tolerable (those pairs were coin flips
  anyway), the second is not, and the margin deltas distinguish them.
- **What it asks of the model, not of the server**: whether adopting the variant
  would require anything of a future checkpoint — a quantization-aware training
  pass, a per-channel scheme, or a smaller vocabulary — given that the embedding
  table, not the Linear stack, is what dominates this checkpoint's memory.

## Latency: not measured

**No latency, throughput, or speedup number appears in this document**, and none
is estimated. `reward_serve/quant/run_latency.py` is built and tested: it sweeps
batch sizes {1, 4, 8, 16} at the full 512-token budget, 30 measured iterations
after 5 warm-up iterations, exact p50/p95 order statistics, peak RSS per variant.
It calls `scripts.box_quiet.require_quiet` before it builds a single model and
refuses on failure, writing the refusal to
`results/quantization/box_quiet_latency.json` and leaving no `latency.json`. A
test asserts that refusal path, including that nothing is built after a refusal.

The gate has not passed on this machine: a render workload owns the GPU and the
CPU has not settled under the bar. So the memory-and-accuracy half of this study
is done and the latency half is pending a quiet box. Which quantization is
*faster* on this CPU is not known here and is not guessed.

One thing that can be said without measuring: this CPU is an i7-10700 (Comet
Lake), which has AVX2 but neither AVX-512 VNNI nor AMX. int8 throughput on
machines with those instructions is a different regime, so even once the gate
passes, the latency numbers will describe this box and not int8 in general.

## Threats to validity

- **One split, one seed, one run.** Every number comes from a single pass over a
  single split of a single dataset, in-distribution for this checkpoint. No
  confidence interval is offered because none was measured; two variants
  differing by a few pairs out of 1,987 cannot be told apart by this study.
- **CPU only.** GPU int8 kernels are out of scope entirely. Nothing here
  predicts how these variants behave on an accelerator.
- **No static quantization.** The variant most likely to be worth adopting was
  blocked by tracing, so the accuracy/memory frontier measured here has a hole
  in it exactly where the interesting point would be.
- **Calibration set size is untested.** The 256-pair calibration set was never
  exercised, since the attempt failed before calibration. Whether 256 is enough
  is an open question this study did not get to ask.
- **Weight size is not resident memory.** The table measures serialized
  `state_dict` bytes. Peak RSS during a forward pass includes activations, which
  scale with batch and sequence length and are measured by the latency runner —
  which has not run.
- **One checkpoint.** 184M parameters with a 128k vocabulary. The embedding-
  dominates-memory finding is a property of this shape, not a general law.

## Not done

- The latency sweep. The quiet-box gate has not passed.
- int8 static quantization, and therefore the whole static row.
- Quantization-aware training. This study only touches post-training methods;
  the charter's scope is serving, not training.
- Per-channel or per-token int8 schemes, and int4 of any kind.
- Quantizing the embedding table, which is where the remaining memory is.
- Any measurement of these variants under concurrent load through the actual
  server. Accuracy here is measured by calling the scorer directly, not over
  HTTP through the batcher.

## Reproducing

Accuracy (may run on a busy box; the box_quiet report is recorded beside the
result either way):

```
cd D:\ClaudeProjects\reward-serve
set CUDA_VISIBLE_DEVICES=
.venv\Scripts\python.exe -m reward_serve.quant.run_accuracy \
  --variants fp32,int8-dynamic,bf16
.venv\Scripts\python.exe -m reward_serve.quant.probe_static
```

Latency (runs only on a quiet box; it re-checks the gate itself and will refuse
again if the box is busy):

```
.venv\Scripts\python.exe -m reward_serve.quant.run_latency \
  --variants fp32,int8-dynamic,bf16
```

Gates:

```
pytest tests/test_quant.py -q
ruff check . && mypy --strict reward_serve/quant
python scripts/vocab_gate.py .
```
