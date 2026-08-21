# What CI checks, and what it does not

The workflow is `.github/workflows/ci.yml`. It runs on pushes to `main` and on
pull requests against `main`, on a GitHub-hosted `ubuntu-latest` runner with
Python 3.13 and CPU-only torch. `CUDA_VISIBLE_DEVICES` is `""` for every step.

## What it verifies

| Step | Command | What a failure means |
|---|---|---|
| Vocabulary gate | `python scripts/vocab_gate.py .` | prose in `README.md` or `docs/**` claims scale this one machine cannot back (charter rule 4) |
| Lint | `ruff check .` | style, imports, unused code, missing annotations |
| Types | `mypy --strict reward_serve` and `mypy --strict loadgen` | a type error in either package; `--strict`, both packages, no `ignore_errors` |
| Tests | `pytest -q` | the suite, CPU only |
| Plain install | `pip install -e .` then import + `--help` | a runtime dependency is declared only in the `dev` extra, so a reader following the README's run instructions would hit an ImportError |

The plain-install job is separate on purpose: the gates job installs
`.[dev]` too, so it cannot see a runtime dependency that only a dev extra
happens to pull in. That is the exact shape of the `datasets` bug this repo
already had once.

The test suite covers the batcher, the server, the router and its breaker, the
fault surface, the load generator and its statistics, the sweep runner, the
plotter, the observability wiring, and the written artefacts — the SLO
document's PromQL, the alert rules, the compose file, the Dockerfile, and the
Grafana dashboard are parsed and checked, not merely present.

Two things reach outside the repo:

- **The published checkpoint.** `tests/test_model.py` and part of
  `tests/test_quant.py` load `DantheMan124/deberta-preference-reward` from the
  Hub. That is deliberate: "the checkpoint still loads under a current
  transformers, and batching does not move a score" is a claim this repo makes,
  and a test that never runs cannot back it. The download is ~712 MB and takes
  about 30 s; `HF_HOME` is cached between runs by `actions/cache`. If the Hub
  is unreachable, those tests **skip with a reason** naming the checkpoint,
  rather than reddening the badge for something that is not a code defect.
- **PyPI and the pytorch CPU wheel index**, at install time.

## What it does NOT verify

This matters more than the list above. A green badge here means the code is
consistent, typed, and correct on a laptop-sized CPU. It does not mean any
number in this repo was reproduced.

- **No load sweep.** `loadgen/sweep.py` is exercised on a tiny synthetic grid
  against a stub server. The committed sweep under `results/` was produced on
  the owner's box and CI does not re-run it.
- **No latency or throughput measurement.** Every number in `README.md`,
  `docs/knee-analysis.md`, `docs/slo.md`, and `docs/quantization.md` comes from
  a committed artefact produced on one quiet machine. A shared CI runner cannot
  pass `scripts/box_quiet.py`, and charter rule 3 says a contended box does not
  produce a published number. CI checks that the *artefacts and the prose
  agree*; it does not regenerate them.
- **No GPU.** There is no GPU on the runner, and the project would refuse it if
  there were.
- **No Docker stack.** The compose file, the collector config, the Prometheus
  config, the alert rules, and the dashboard JSON are parsed and cross-checked
  against the metrics the server actually emits. Nothing is built and no
  container is started. "Prometheus scrapes the server" is verified as
  configuration, not as an observed scrape.
- **No Windows runner.** The suite is developed on Windows and verified on
  Linux by CI; it is not run on both.
- **No coverage threshold, no benchmark regression gate, no security scan.**

## Tests that are skipped, and why

Skips are always explicit and always carry a reason string. A run on a
GitHub-hosted runner reports:

- `promtool is not installed; the structural checker ran instead` —
  `tests/test_slo_and_stack.py` prefers the real Prometheus rule linter when it
  is on PATH and falls back to its own structural checker when it is not. The
  fallback is itself tested against seven broken expressions.
- `latency.json exists; this guard only covers the un-measured state` — a guard
  that only applies before the quantization latency half has been measured.
- The real-checkpoint tests, only when the Hub cannot be reached.

Nothing is silently deselected: `pytest -q` runs every test in `tests/`,
including the `slow` marker, which spawns a router and two synthetic workers as
real processes.

## Runtime

Measured on the owner's box against a cold clone and a cold pip cache
(a GitHub runner will differ, and the first run has no caches at all):

| Step | Cold |
|---|---|
| `pip install --upgrade pip` | 7 s |
| CPU torch from the pytorch index | 91 s |
| `pip install -e ".[dev]"` | 163 s |
| Vocabulary gate | 0.2 s |
| `ruff check .` | 0.8 s |
| `mypy --strict` both packages | 64 s cold cache, 2 s warm |
| `pytest -q` | 136 s (of which ~30 s is the checkpoint download on a cold `HF_HOME`) |

The job carries a 30-minute timeout and a concurrency group that cancels a
superseded run on the same ref.
