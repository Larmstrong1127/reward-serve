"""The one real-model test: it loads, the budget comes from the checkpoint,
and batching does not change any score.

Everything else in the suite uses a stub scorer. This test downloads the
checkpoint on a cold clone.
"""

from __future__ import annotations

import pytest

from reward_serve.model import DEFAULT_MODEL_ID, RewardModel, resolve_max_length

# Neutral synthetic fixtures of deliberately different lengths, so a batched
# forward pass has real padding to get wrong.
PAIRS: list[tuple[str, str]] = [
    ("How do I reverse a list?", "Use the reversed() builtin."),
    ("How do I reverse a list?", "no"),
    (
        "Explain what a bounded queue is and why a server would want one.",
        "A bounded queue caps how much work can wait, so a slow consumer pushes back "
        "on producers instead of letting memory grow without limit. When it is full "
        "the server can reject fast rather than queueing work it will never serve in "
        "time, which keeps latency bounded for the requests it does accept.",
    ),
    ("What is 2 + 2?", "4"),
    ("Name a colour.", "Blue is a colour."),
    ("Summarise the plot of a story about a lost key.", "A key is lost, then found."),
    ("Define latency.", "Latency is the time between a request and its response."),
    ("Define throughput.", "Throughput counts completed requests per unit of time."),
]


@pytest.fixture(scope="module")
def model() -> RewardModel:
    """Loading it at all is the tokenizer_config regression check."""
    return RewardModel(DEFAULT_MODEL_ID)


def test_model_loads_under_current_transformers(model: RewardModel) -> None:
    assert model.model_id == DEFAULT_MODEL_ID
    assert model.tokenizer is not None
    assert model.model.config.num_labels == 1


def test_sequence_budget_comes_from_the_checkpoint(model: RewardModel) -> None:
    """The budget is read, not assumed. Serving above it is the old incident."""
    assert model.max_length == model.config.reward_train_max_length
    assert 0 < model.max_length <= model.config.max_position_embeddings
    assert model.temperature == pytest.approx(model.config.reward_temperature)


def test_budget_ignores_the_tokenizer_sentinel() -> None:
    """The Hub tokenizer reports a ~1e30 model_max_length; it must not win."""

    class Cfg:
        max_position_embeddings = 512

    class Tok:
        model_max_length = 1000000000000000019884624838656

    # No reward_train_max_length, absurd tokenizer value: fall through to config.
    assert resolve_max_length(Cfg(), Tok()) == 512

    class TrainedCfg(Cfg):
        reward_train_max_length = 256

    assert resolve_max_length(TrainedCfg(), Tok()) == 256


def test_scores_are_identical_alone_and_inside_a_batch_of_eight(model: RewardModel) -> None:
    """Padding must not move any score. This is the claim that batching is safe."""
    assert len(PAIRS) == 8
    batched = model.score_pairs(PAIRS)
    alone = [model.score_pairs([pair])[0] for pair in PAIRS]

    assert len(batched) == 8
    for i, (b, a) in enumerate(zip(batched, alone, strict=True)):
        assert b == pytest.approx(a, abs=1e-4), (
            f"pair {i} scored {b} in a batch of 8 but {a} alone (delta {abs(b - a):.2e})"
        )


def test_scores_are_order_independent(model: RewardModel) -> None:
    """Reordering a batch permutes the scores and nothing else."""
    forward = model.score_pairs(PAIRS)
    backward = model.score_pairs(list(reversed(PAIRS)))
    for i, (f, b) in enumerate(zip(forward, reversed(backward), strict=True)):
        assert f == pytest.approx(b, abs=1e-4), f"pair {i} moved when the batch was reordered"


def test_empty_batch_is_a_no_op(model: RewardModel) -> None:
    assert model.score_pairs([]) == []
