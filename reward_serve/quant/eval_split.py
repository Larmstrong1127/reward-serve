"""The exact evaluation split behind the published fp32 number, vendored.

The published pairwise accuracy 0.7026 (UltraFeedback `test_prefs`, N=1987,
512-token budget) was produced by the owner's training repo. Every accuracy in
this study has to be *paired* with that number, which means the same pairs, the
same truncation filter, and the same tokenizer. So the loader is vendored here
rather than reimplemented.

Source (read-only, not modified by this repo):
    evalforge/training/training/data/preference.py
        - PreferencePair
        - load_ultrafeedback_pairs
        - audit_and_filter_pairs
    evalforge/training/eval_reward.py  (the caller)
    evalforge/training/reward_results.json  (the artifact)

The two functions below are copies of that code with types added; the logic is
unchanged. `audit_and_filter_pairs` is what turns 1988 loaded pairs into the
1987 evaluated pairs: one pair's chosen and rejected encodings collide after
truncation at 512 tokens and carry no preference signal.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from collections.abc import Callable  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from typing import Any  # noqa: E402

__all__ = [
    "CALIBRATION_SPLIT",
    "PUBLISHED_FP32_ACCURACY",
    "PUBLISHED_N_PAIRS",
    "ULTRAFEEDBACK_DATASET_ID",
    "AuditStats",
    "PreferencePair",
    "audit_and_filter_pairs",
    "load_calibration_pairs",
    "load_eval_pairs",
    "load_ultrafeedback_pairs",
]

ULTRAFEEDBACK_DATASET_ID = "HuggingFaceH4/ultrafeedback_binarized"
EVAL_SPLIT = "test_prefs"
# Calibration data must never touch the split the accuracies are measured on.
CALIBRATION_SPLIT = "train_prefs"

# The anchor. If the fp32 variant does not reproduce this on this split, the
# study is measuring something other than the shipped model and must stop.
PUBLISHED_FP32_ACCURACY = 0.7026
PUBLISHED_N_PAIRS = 1987


@dataclass(frozen=True)
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str


@dataclass(frozen=True)
class AuditStats:
    total: int
    dropped_identical: int


def flatten_messages(messages: list[dict[str, Any]]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def _default_load_fn(split: str) -> Any:
    import datasets

    return datasets.load_dataset(ULTRAFEEDBACK_DATASET_ID, split=split)


def load_ultrafeedback_pairs(
    split: str = EVAL_SPLIT,
    load_fn: Callable[[str], Any] = _default_load_fn,
) -> list[PreferencePair]:
    """Vendored from evalforge training/training/data/preference.py."""
    pairs: list[PreferencePair] = []
    for row in load_fn(split):
        chosen = row["chosen"][-1]["content"]
        rejected = row["rejected"][-1]["content"]
        if chosen == rejected:
            continue  # no preference signal at all
        prompt_messages = row["chosen"][:-1]
        prompt = flatten_messages(prompt_messages) if len(prompt_messages) > 1 else row["prompt"]
        pairs.append(PreferencePair(prompt=prompt, chosen=chosen, rejected=rejected))
    return pairs


def audit_and_filter_pairs(
    pairs: list[PreferencePair], tokenizer: Any, max_length: int
) -> tuple[list[PreferencePair], AuditStats]:
    """Vendored from evalforge training/training/data/preference.py.

    Drops pairs whose chosen and rejected encodings are identical *because* the
    truncation cap was hit. Those pairs ask the model to separate two identical
    inputs, so they are noise, and the published number excludes them.
    """
    kept: list[PreferencePair] = []
    dropped = 0
    for pair in pairs:
        chosen_ids = tokenizer(pair.prompt, pair.chosen, truncation=True, max_length=max_length)[
            "input_ids"
        ]
        rejected_ids = tokenizer(
            pair.prompt, pair.rejected, truncation=True, max_length=max_length
        )["input_ids"]
        truncated = len(chosen_ids) == max_length or len(rejected_ids) == max_length
        if truncated and chosen_ids == rejected_ids:
            dropped += 1
            continue
        kept.append(pair)
    return kept, AuditStats(total=len(pairs), dropped_identical=dropped)


def load_eval_pairs(
    tokenizer: Any,
    max_length: int,
    limit: int | None = None,
    load_fn: Callable[[str], Any] = _default_load_fn,
) -> tuple[list[PreferencePair], AuditStats]:
    """The evaluated split: load, then apply the truncation filter, in order.

    `limit` truncates *after* the filter, so the first N pairs of a dry run are
    the first N pairs of the full run.
    """
    pairs = load_ultrafeedback_pairs(EVAL_SPLIT, load_fn=load_fn)
    kept, stats = audit_and_filter_pairs(pairs, tokenizer, max_length)
    if limit is not None:
        kept = kept[:limit]
    return kept, stats


def load_calibration_pairs(
    tokenizer: Any,
    max_length: int,
    n_pairs: int = 256,
    load_fn: Callable[[str], Any] = _default_load_fn,
) -> list[PreferencePair]:
    """Calibration pairs, held out of the eval split by construction.

    Drawn from `train_prefs`, not `test_prefs`, so no calibration pair can be
    one of the 1,987 evaluated pairs. The same truncation filter is applied, so
    calibration sees the same shape of input the server will. Filtering stops as
    soon as `n_pairs` survive - the training split has tens of thousands of rows
    and auditing all of them would cost more than the calibration itself.
    """
    kept: list[PreferencePair] = []
    for pair in load_ultrafeedback_pairs(CALIBRATION_SPLIT, load_fn=load_fn):
        one, stats = audit_and_filter_pairs([pair], tokenizer, max_length)
        if stats.dropped_identical:
            continue
        kept.extend(one)
        if len(kept) >= n_pairs:
            break
    return kept[:n_pairs]
