# reward-serve — engineering charter

Every agent working in this repo reads this file first and treats it as binding.

## What this is
A continuous-batching inference server for the owner's own published model,
`DantheMan124/deberta-preference-reward` (184M DeBERTa, Bradley-Terry pairwise
reward head, calibration temperature 1.1668). It exists to demonstrate, with
reproducible numbers, that the owner can serve a model under load, observe it,
break it deliberately, and drive incidents to root cause. It runs on ONE
Windows box with a 24 GB GPU that other work also uses.

## The audience
A strict hiring manager for an ML inference-serving team. They will clone this
cold, rerun the numbers, and read every claim against the code. Design for that
reader. A claim they cannot reproduce is worse than no claim.

## Non-negotiable rules
1. **CPU-only enforcement.** Set `CUDA_VISIBLE_DEVICES=""` in every entrypoint,
   test, and script. Never allocate on the GPU. A render fleet owns it.
2. **Every number has an artifact.** Any figure that appears in prose must be
   in a committed JSON/CSV under `results/`, produced by a committed script,
   with the exact command recorded next to it. No exceptions, including
   "approximately".
3. **Measurement needs a quiet box.** `scripts/box_quiet.py` must PASS
   (CPU < 15% for 30s, GPU util < 10%, no other python.exe over 20% CPU) before
   any latency/throughput sweep runs. Store the box_quiet report alongside the
   results it gates. Contended numbers are not published — ever.
4. **Forbidden vocabulary in prose:** `distributed`, `at scale`, `production-
   grade`, `production-ready`, `enterprise`, `datacenter`, `cluster` — unless
   inside an explicit caveat sentence saying this is NOT that. `scripts/
   vocab_gate.py` enforces it in CI. This is one box. Say "multi-process",
   "multi-tier", "one machine".
5. **Written vs operated.** Describe only what has been run. Present tense
   only for behaviour a test or a committed run demonstrates. Future work goes
   under a heading called exactly "Not done".
6. **Sequence budget from the checkpoint config**, never hardcoded. The owner
   once served this model at 1024 against a 512-trained checkpoint; that story
   is in the model card. Do not repeat it.
7. **Privacy.** No persona names, no real prompts, no `output/personas` paths,
   no financial figures. Test fixtures are neutral synthetic text.
8. **Commits:** plain, imperative, no AI attribution, no Co-Authored-By.
   Commit incrementally — sessions get killed.
9. **Worktrees.** Each agent works in its own git worktree off `main`. Never
   edit another worktree. Rebase onto main before opening your branch for
   merge; if it conflicts, resolve in your worktree.
10. **Dry run first.** One minimal end-to-end run (10 requests) before any
    sweep. If the dry run cannot pass, the sweep does not start.
11. **Report to the orchestrator** with: what you claim, the command that
    proves it, and what you did NOT do. Never round a partial into a done.

## Layout (fixed — do not restructure)
    reward_serve/        package: server, batcher, model, obs
    loadgen/             load generator + sweep runner
    scripts/             box_quiet.py, vocab_gate.py, plot_*.py
    results/             committed JSON/CSV artifacts + box_quiet reports
    docs/                knee-analysis.md, slo.md, postmortems/, quantization.md
    tests/               pytest; CPU only
    docker/              compose for server + otel-collector + prometheus + grafana
    README.md            claims must pass scripts/vocab_gate.py

## Model facts (verified; do not "improve")
- HF id: DantheMan124/deberta-preference-reward — public
- Pairwise accuracy 0.7026 on UltraFeedback test_prefs, N=1987, 512-token budget
- Score is a raw logit / T; ranking-only semantics
- tokenizer_config was fixed 2026-08-13 for transformers>=4.57 — load with
  a current transformers and assert it loads in a test
