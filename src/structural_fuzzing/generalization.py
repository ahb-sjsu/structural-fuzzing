"""Cross-relation generalization gap and probe calibration.

Two validation tools that extend adversarial probing from parameter/signal space into *relation*
space (see the case study in the book, Part IV):

- ``probe_calibration`` — before trusting a probe to rank models, check it against a known-strong
  and a known-weak reference. A strong reference at chance means the probe measures surface, not
  the target (the evaluation-space analogue of the intensity-zero identity).
- ``cross_relation_gap`` — evaluate an adapted model on the relation it was trained for *and* on an
  independent, entity-held-out relation. The gap between the two improvements separates learned
  structure from objective-memorization.

Both operate on *probe scores*: a probe returns, for two models, arrays of similarity/quality
scores on positive pairs and on negative pairs. AUROC = P(positive scores above negative).
Everything is numpy-only and deterministic given ``rng``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based AUROC = P(pos > neg) (Mann-Whitney U), tie-aware. numpy-only."""
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    s = np.r_[pos, neg]
    order = s.argsort(kind="mergesort")
    ranks = np.empty(len(s), float)
    sr = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sr[j + 1] == sr[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    npos, nneg = len(pos), len(neg)
    return (ranks[:npos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def paired_bootstrap_auroc_delta(
    pos_a: np.ndarray,
    neg_a: np.ndarray,
    pos_b: np.ndarray,
    neg_b: np.ndarray,
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Paired bootstrap on ``auroc(model_a) - auroc(model_b)`` for one probe.

    Model A and model B are scored on the *same* positive and negative pairs; each bootstrap round
    resamples the same pair indices for both models, so the comparison is paired. Returns
    ``(mean_delta, lo_2.5%, hi_97.5%)``. A CI excluding 0 is a real difference.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    pos_a, neg_a = np.asarray(pos_a, float), np.asarray(neg_a, float)
    pos_b, neg_b = np.asarray(pos_b, float), np.asarray(neg_b, float)
    npos, nneg = len(pos_a), len(neg_a)
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        pi = rng.integers(0, npos, npos)
        ni = rng.integers(0, nneg, nneg)
        diffs[k] = auroc(pos_a[pi], neg_a[ni]) - auroc(pos_b[pi], neg_b[ni])
    return float(diffs.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


@dataclass
class ProbeCalibration:
    """Result of calibrating a probe against known-strong / known-weak references."""

    strong_score: float
    weak_score: float
    chance: float
    passed: bool
    reason: str


def probe_calibration(
    strong_score: float,
    weak_score: float,
    chance: float = 0.5,
    tol: float = 0.05,
) -> ProbeCalibration:
    """Check whether a probe is trustworthy before using it to rank models.

    A probe is only admissible if a *known-strong* reference scores meaningfully above chance and a
    *known-weak* reference does not score high (which would mean the probe leaks the answer).

    Parameters
    ----------
    strong_score, weak_score : float
        The probe's score (e.g. AUROC) for a reference known to be strong / weak on the target.
    chance : float
        The probe's chance level (0.5 for AUROC).
    tol : float
        Margin above chance that the strong reference must clear.
    """
    if strong_score <= chance + tol:
        return ProbeCalibration(
            strong_score,
            weak_score,
            chance,
            False,
            f"strong reference at chance ({strong_score:.3f} <= {chance + tol:.3f}): "
            "the probe measures surface, not the target",
        )
    if weak_score >= strong_score - tol:
        return ProbeCalibration(
            strong_score,
            weak_score,
            chance,
            False,
            f"weak reference scores as high as strong ({weak_score:.3f} vs {strong_score:.3f}): "
            "the probe leaks the answer",
        )
    return ProbeCalibration(
        strong_score,
        weak_score,
        chance,
        True,
        "strong reference clears chance and weak reference does not: probe is admissible",
    )


@dataclass
class GeneralizationGap:
    """Result of a cross-relation generalization-gap analysis."""

    trained_delta: float
    trained_ci: tuple[float, float]
    independent_delta: float
    independent_ci: tuple[float, float]
    gap: float
    interpretation: str


def cross_relation_gap(
    trained: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    independent: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
) -> GeneralizationGap:
    """Improvement on the trained relation vs. an independent, held-out relation.

    Each argument is ``(pos_adapted, neg_adapted, pos_base, neg_base)``: the adapted and the base
    model's probe scores on positive and negative pairs of that relation. The independent relation
    must be structurally independent of the training signal AND held out at the *entity* level (the
    documents themselves unseen), not merely the pair level — otherwise the "independent" number
    leaks structure.

    ``gap = trained_delta - independent_delta``. A large gap = mostly objective-specialization; a
    gap near zero (both deltas > 0) = transferable structure.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    td, tlo, thi = paired_bootstrap_auroc_delta(*trained, n_boot=n_boot, rng=rng)
    idd, ilo, ihi = paired_bootstrap_auroc_delta(*independent, n_boot=n_boot, rng=rng)
    gap = td - idd
    trained_real = tlo > 0 or thi < 0
    indep_real = ilo > 0 or ihi < 0
    if not trained_real:
        interp = "no improvement even on the trained relation — check probe calibration first"
    elif not indep_real:
        interp = "improves the trained relation only; no measurable transfer to the independent"
    elif abs(gap) < 0.5 * abs(td):
        interp = "improvement transfers substantially — evidence of transferable structure"
    else:
        interp = "improvement is mostly specialization to the trained relation; transfer is small"
    return GeneralizationGap(td, (tlo, thi), idd, (ilo, ihi), gap, interp)
