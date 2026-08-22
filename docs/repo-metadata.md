# Repository metadata for the day this goes public

Nothing here is applied. It is the exact text to paste when the owner flips the
repository from private to public, kept in the repo so the description and the
topics get the same word-care as the README rather than being typed in a hurry
on the settings page.

Every claim below is one the repository already backs. No word in it says this
runs on more than one machine, because it does not.

## Description (350-char limit; this is 232)

    Continuous-batching inference server for a 184M DeBERTa reward model on one machine, CPU only: bounded queue, deadline-aware batcher, OpenTelemetry traces and Prometheus metrics, fault-injection runs, and a measured batch-size knee.

## Topics

GitHub topics must be lowercase, may contain hyphens and digits, and are capped
at 20 per repository. These are 14.

    ml-inference
    inference-server
    continuous-batching
    batching
    observability
    opentelemetry
    prometheus
    pytorch
    transformers
    reward-model
    fastapi
    python
    load-testing
    circuit-breaker

## Commands, ready to paste

Run from a clone with `gh` authenticated. The first command makes the repository
public and is not reversible without a second deliberate step, so it is listed
separately from the metadata commands.

    gh repo edit Larmstrong1127/reward-serve --description "Continuous-batching inference server for a 184M DeBERTa reward model on one machine, CPU only: bounded queue, deadline-aware batcher, OpenTelemetry traces and Prometheus metrics, fault-injection runs, and a measured batch-size knee."

    gh repo edit Larmstrong1127/reward-serve \
      --add-topic ml-inference \
      --add-topic inference-server \
      --add-topic continuous-batching \
      --add-topic batching \
      --add-topic observability \
      --add-topic opentelemetry \
      --add-topic prometheus \
      --add-topic pytorch \
      --add-topic transformers \
      --add-topic reward-model \
      --add-topic fastapi \
      --add-topic python \
      --add-topic load-testing \
      --add-topic circuit-breaker

    gh repo edit Larmstrong1127/reward-serve --homepage "https://huggingface.co/DantheMan124/deberta-preference-reward"

The flip itself, when the owner decides:

    gh repo edit Larmstrong1127/reward-serve --visibility public --accept-visibility-change-consequences

## Words deliberately not used

- never used, and failed by `scripts/vocab_gate.py`: `distributed`, `at scale`
- never used, and failed by the same gate: `production-ready`, `enterprise`
- never used, and failed by the same gate: `cluster`, `datacenter`
- never used, though the gate does not check them: `high-throughput`,
  `low-latency` — these are claims about numbers, and the numbers here say
  batching did not pay on this CPU, so they would be wrong as well as unbacked.

## Before the flip

- Read `docs/ci.md`: what CI covers and does not cover becomes public too.
- The `results/` tree contains `box_quiet` reports naming this machine's CPU and
  GPU model. That is intended: they are what makes the measurements checkable.
  Nothing else about the machine, and nothing from unrelated work on it, is in
  the repository.
