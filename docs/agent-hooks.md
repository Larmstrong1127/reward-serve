# Agent hooks: the gates that run before the agent's tool does

Two rules in `CHARTER.md` were, until this landed, enforced only after the
fact. The quiet-box rule was checked once at the start of a sweep by
`scripts/box_quiet.py`; the vocabulary rule was checked by CI after the words
were already committed. Both leaked:

* Two measurement runs overlapped **twice** during the quantization and sweep
  work. A ghost interpreter from an earlier session launched a duplicate run,
  and nothing in the harness could see that another measurement was already
  in flight. The numbers from both runs looked clean.
* Scale vocabulary reached surfaces CI never scans (the interview deck, the
  portfolio site, the profile README) and sat there for weeks.

The fix is to move both checks to the tool boundary of the coding agent.
Claude Code runs a `PreToolUse` hook before every tool call; the hook reads
the tool's input on stdin and answers `allow`, `ask`, or `deny` on stdout.
That is the same primitive Omnigent calls a contextual policy on `tool_call`
events, and it is native on this Windows box, so the two policies live here as
plain scripts under `scripts/hooks/`.

## `measurement_gate.py` (matcher: `Bash`)

Fires only when the command matches a measurement (`loadgen.sweep`,
`run_latency`, `run_accuracy`, `box_quiet.py`, `record_demo.py`, or a pytest
invocation with `bench` in it). Everything else exits silently in a few
milliseconds without importing psutil. `--smoke` runs pass untouched: they
exist to be run on a busy box.

For a real measurement it samples the box for three seconds and decides:

| observation | decision |
|---|---|
| another measurement process is already running | **deny**, naming the PID and command |
| GPU memory in use >= 2 GiB | **deny** (a render or model is up) |
| a foreign python averages > 20% of one core | **ask** |
| otherwise | **allow**, and the sample rides along as `additionalContext` |

Three seconds is deliberate. The 30-second gate stays where it is, inside the
sweep; a hook that held every shell command for half a minute would be routed
around within a day. The in-flight check is the one the sweep proved missing
and needs no window at all.

The GPU thresholds are the recalibrated ones from `scripts/box_quiet.py` and
`loadgen/contention.py`; a machine with no `nvidia-smi` reads as quiet, not as
held, because CI runners have no GPU and must not be denied for it.

## `claim_gate.py` (matcher: `Write|Edit`)

Fires when the target file is a claim surface (any of the eight surfaces the
sweep rule names, matched by path fragment; override with `CLAIM_GATE_ROOTS`)
and has a prose suffix (`.md .html .json .txt .yml .yaml`). The text about to
be written -- the whole `content` for Write, only `new_string` for Edit -- is
scanned with the **same** `offending_lines` function the CI gate uses. One
word list; a test pins that the hook imports it rather than copying it.

A hit produces **ask** with the offending lines. It never denies: a caveat the
regex misreads is the author's call. It never touches code files, and
`CHARTER.md`, which lists the forbidden words on purpose, is exempt.

## Fail-open, on purpose

Every failure inside either hook -- unreadable stdin, psutil missing, the
sampler raising -- produces `allow` with the error in the reason, so the
agent is never wedged by its own guardrail and the failure is at least
visible in the transcript as "UNGATED". Both hooks have a kill switch
(`MEASUREMENT_GATE_DISABLED=1`, `CLAIM_GATE_DISABLED=1`) for the case where
the gate itself is what is being debugged.

## Known limitations

* The measurement gate matches the command **text**, not what the command
  does. A `git commit` whose message mentions `loadgen.sweep`, or an `echo`
  of the same, pays the three-second sample and gets an `allow`. Harmless, but
  a real false positive; it fired that way on the commit that logged the first
  live fires. Matching on the resolved executable would need the harness to
  tell the hook what it resolved, which it does not.
* The claim gate sees only the text of the one Write or Edit, never the file
  it lands in, so a disclaimer already on the target line does not rescue an
  Edit whose `new_string` carries the term alone. That produces an ask, not a
  deny, by design.
* Neither hook fires for the owner typing in a terminal. They gate the agent.

## Wiring

The hooks are installed in the owner's user-level Claude Code settings so
they fire in every session on this machine, whatever directory it started in.
The entry, for reference (paths are this machine's):

```json
"hooks": {
  "PreToolUse": [
    {"matcher": "Bash",
     "hooks": [{"type": "command", "timeout": 20,
                "command": "D:/ClaudeProjects/reward-serve/.venv/Scripts/python.exe D:/ClaudeProjects/reward-serve/scripts/hooks/measurement_gate.py"}]},
    {"matcher": "Write|Edit",
     "hooks": [{"type": "command", "timeout": 10,
                "command": "D:/ClaudeProjects/reward-serve/.venv/Scripts/python.exe D:/ClaudeProjects/reward-serve/scripts/hooks/claim_gate.py"}]}
  ]
}
```

## What this does and does not prove

Written and tested: 32 tests in `tests/test_hooks.py` cover every row of the
decision table with fabricated samples, and pipe real payloads through each
script as a subprocess to pin the stdin/stdout contract.

Operated: the section below records each time a hook fired for real. A gate
that has never fired is a claim, not a control -- the same standard this
project applies to its SLO alerts and its eval gate.

### Firing log

| date | hook | decision | what it caught |
|---|---|---|---|
| 2026-09-09 05:43 | measurement_gate | allow | installation proof: a harmless `echo` whose text contained `loadgen.sweep`; sample came back as context: GPU 552 MiB, util mean 20.86%, no foreign python, no in-flight measurement |
| 2026-09-09 05:47 | claim_gate | ask | installation proof: a scratch `docs/_hook_proof.md` containing "a distributed system"; the first attempt fired but was INVISIBLE (the ask was settled by the auto-mode classifier and the hook returned no context) -- fixed same hour by returning the verdict as `additionalContext` too; second attempt visible, file deleted |
