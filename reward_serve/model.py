"""Loads the DeBERTa pairwise reward model and scores prompt/response pairs.

The sequence budget is read from the checkpoint config, never hardcoded. The
owner once served this checkpoint at 1024 tokens against a 512-token training
budget; the fallback chain below exists so that cannot happen again.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
# No HF_HOME default here: the Hub cache root is the caller's to choose. Setting
# one machine's absolute path in library code made a stranger's clone create a
# literal "D:\ClaudeCache\hf" directory in the working directory on Linux.

from collections.abc import Sequence  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from typing import Any  # noqa: E402

import torch  # noqa: E402
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

__all__ = ["DEFAULT_MODEL_ID", "RewardModel", "resolve_max_length"]

DEFAULT_MODEL_ID = "DantheMan124/deberta-preference-reward"

# The Hub tokenizer ships model_max_length as a sentinel near 1e30. Anything at
# or above this is meaningless as a budget and must not be trusted.
_SENTINEL_MAX_LENGTH = 100_000


def resolve_max_length(config: Any, tokenizer: Any) -> int:
    """Resolve the sequence budget from the checkpoint, in priority order.

    1. `config.reward_train_max_length` - what the reward head was trained at.
    2. `tokenizer.model_max_length`, only when it is a real number.
    3. `config.max_position_embeddings`.

    Raises when none of the three yields a usable budget, because guessing here
    is exactly the failure this function exists to prevent.
    """
    trained = getattr(config, "reward_train_max_length", None)
    if isinstance(trained, int) and 0 < trained < _SENTINEL_MAX_LENGTH:
        return trained
    tok_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(tok_max, int) and 0 < tok_max < _SENTINEL_MAX_LENGTH:
        return tok_max
    pos = getattr(config, "max_position_embeddings", None)
    if isinstance(pos, int) and 0 < pos < _SENTINEL_MAX_LENGTH:
        return pos
    raise ValueError("could not resolve a sequence budget from the checkpoint config")


@dataclass(frozen=True)
class ModelInfo:
    """Facts read off the checkpoint, echoed in every score response."""

    model_id: str
    max_length: int
    temperature: float


class RewardModel:
    """Scores prompt/response pairs. Score is a raw logit divided by T.

    The score is ranking-only: it orders responses to the same prompt and
    carries no absolute meaning.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID) -> None:
        self.model_id = model_id
        self.config = AutoConfig.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.eval()
        self.model.to("cpu")

        self.max_length = resolve_max_length(self.config, self.tokenizer)
        temperature = getattr(self.config, "reward_temperature", None)
        if not isinstance(temperature, (int, float)) or temperature <= 0:
            raise ValueError("checkpoint config is missing a usable reward_temperature")
        self.temperature = float(temperature)

        # DeBERTa-v2 is encoder-only with position_biased_input disabled, so the
        # attention mask fully accounts for padding. Right padding keeps the
        # [CLS] pooled token at index 0, which the reward head reads.
        self.tokenizer.padding_side = "right"

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(self.model_id, self.max_length, self.temperature)

    @torch.inference_mode()
    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """One padded forward pass over the pairs. Returns scores in order."""
        if not pairs:
            return []
        prompts = [p for p, _ in pairs]
        responses = [r for _, r in pairs]
        encoded = self.tokenizer(
            prompts,
            responses,
            padding=True,
            truncation="longest_first",
            max_length=self.max_length,
            return_tensors="pt",
        )
        logits = self.model(**encoded).logits
        scores = logits[:, 0].float() / self.temperature
        return [float(s) for s in scores.tolist()]
