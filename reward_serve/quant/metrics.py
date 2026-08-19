"""Paired comparison metrics between a quantized variant and the fp32 baseline.

A reward model's job is to order responses, so "did the number move" is the
weak question and "did the ordering move" is the strong one. Hence three
measures side by side: absolute score drift, rank correlation of the per-pair
margins, and the fraction of pairs whose preferred side actually flipped.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["flipped_fraction", "kendall_tau_b", "mean_abs_delta", "pairwise_accuracy"]


def pairwise_accuracy(margins: Sequence[float]) -> float:
    """Fraction of pairs scored chosen > rejected. Ties count as wrong."""
    arr = np.asarray(margins, dtype=np.float64)
    return float((arr > 0.0).mean())


def mean_abs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).mean())


def flipped_fraction(margins: Sequence[float], baseline_margins: Sequence[float]) -> float:
    """Fraction of pairs whose preferred side differs from the baseline's."""
    x = np.sign(np.asarray(margins, dtype=np.float64))
    y = np.sign(np.asarray(baseline_margins, dtype=np.float64))
    return float((x != y).mean())


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float:
    """Kendall tau-b, computed directly over all n(n-1)/2 pairs.

    Implemented here rather than pulled from scipy so the study adds no
    dependency it does not otherwise need. n is ~2000, so the O(n^2) form costs
    a few million comparisons - well under a second - and is easier to check by
    eye than a merge-sort inversion count.
    """
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.shape != ya.shape:
        raise ValueError("kendall_tau_b needs two equal-length sequences")
    n = xa.size
    if n < 2:
        raise ValueError("kendall_tau_b needs at least two observations")
    dx = np.sign(xa[:, None] - xa[None, :])
    dy = np.sign(ya[:, None] - ya[None, :])
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    prod = (dx * dy)[upper]
    concordant = float((prod > 0).sum())
    discordant = float((prod < 0).sum())
    ties_x = float(((dx[upper] == 0) & (dy[upper] != 0)).sum())
    ties_y = float(((dy[upper] == 0) & (dx[upper] != 0)).sum())
    denom = np.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    if denom == 0.0:
        raise ValueError("kendall_tau_b is undefined when every observation ties")
    return float((concordant - discordant) / denom)
