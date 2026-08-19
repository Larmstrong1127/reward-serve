"""Synthetic, seeded prompt/response pairs of controlled length.

Neutral filler text only -- no real prompts, no persona content (charter rule 7).
Length is controlled in WORDS, not model tokens: the load generator deliberately
does not import transformers, so it cannot count DeBERTa tokens. The word counts
below are a proxy; the server reports what it actually batched.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Final

# Neutral vocabulary. Deliberately dull; it exists to occupy token slots.
_WORDS: Final[tuple[str, ...]] = (
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "signal", "value", "record", "column", "matrix", "vector", "sample",
    "window", "buffer", "packet", "segment", "channel", "gradient", "weight",
    "measure", "observe", "compare", "select", "combine", "reduce", "expand",
)

# bucket name -> (prompt words, response words)
BUCKETS: Final[dict[str, tuple[int, int]]] = {
    "short": (16, 16),
    "medium": (64, 64),
    "long": (192, 192),
}


@dataclass(frozen=True)
class Sample:
    """One synthetic scoring request."""

    bucket: str
    prompt: str
    response: str


def _filler(rng: random.Random, n_words: int) -> str:
    return " ".join(rng.choice(_WORDS) for _ in range(n_words))


def make_sample(rng: random.Random, bucket: str) -> Sample:
    """Build one sample of the given length bucket from `rng`."""
    if bucket not in BUCKETS:
        raise ValueError(f"unknown bucket {bucket!r}; known: {sorted(BUCKETS)}")
    p_words, r_words = BUCKETS[bucket]
    return Sample(bucket=bucket, prompt=_filler(rng, p_words), response=_filler(rng, r_words))


def build_corpus(seed: int, per_bucket: int = 64, buckets: tuple[str, ...] | None = None) -> list[Sample]:
    """Deterministic corpus: same seed always yields the same list of samples.

    Buckets are INTERLEAVED (short, medium, long, short, ...), not concatenated.
    The generator walks this list with `index % len(corpus)`, so a block-ordered
    corpus would give a short run -- and every warm-up phase -- nothing but the
    first bucket, and the length mix would depend on how many requests happened
    to fit. Interleaving makes the mix the same at any run length.
    """
    names = buckets if buckets is not None else tuple(BUCKETS)
    rng = random.Random(seed)
    return [make_sample(rng, b) for _ in range(per_bucket) for b in names]
