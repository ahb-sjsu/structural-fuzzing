"""Tests for cross-relation generalization gap and probe calibration."""

from __future__ import annotations

import numpy as np

from structural_fuzzing.generalization import (
    GeneralizationGap,
    ProbeCalibration,
    auroc,
    cross_relation_gap,
    paired_bootstrap_auroc_delta,
    probe_calibration,
)


class TestAuroc:
    def test_perfect_separation(self):
        assert auroc(np.array([0.9, 0.8, 0.95]), np.array([0.1, 0.2, 0.05])) == 1.0

    def test_chance(self):
        x = np.array([0.5, 0.5, 0.5, 0.5])
        assert auroc(x, x) == 0.5

    def test_tie_aware_bounds(self):
        rng = np.random.default_rng(0)
        val = auroc(rng.normal(size=50), rng.normal(size=50))
        assert 0.0 <= val <= 1.0


class TestPairedBootstrap:
    def test_positive_delta_when_a_better(self):
        rng = np.random.default_rng(1)
        pos_a, neg_a = rng.normal(1.0, 0.2, 300), rng.normal(0.0, 0.2, 300)
        pos_b, neg_b = rng.normal(0.3, 0.2, 300), rng.normal(0.0, 0.2, 300)
        mean, lo, hi = paired_bootstrap_auroc_delta(pos_a, neg_a, pos_b, neg_b, n_boot=300)
        assert lo <= mean <= hi
        assert mean > 0 and lo > 0  # A clearly better -> CI excludes 0


class TestProbeCalibration:
    def test_admissible(self):
        r = probe_calibration(strong_score=0.80, weak_score=0.52)
        assert isinstance(r, ProbeCalibration)
        assert r.passed

    def test_strong_at_chance_fails(self):
        r = probe_calibration(strong_score=0.51, weak_score=0.50)
        assert not r.passed
        assert "surface" in r.reason

    def test_probe_leaks_when_weak_scores_high(self):
        r = probe_calibration(strong_score=0.80, weak_score=0.79)
        assert not r.passed
        assert "leaks" in r.reason


class TestCrossRelationGap:
    def _relation(self, rng, sep_adapted, sep_base, n=300):
        return (
            rng.normal(sep_adapted, 0.2, n),
            rng.normal(0.0, 0.2, n),  # adapted pos, neg
            rng.normal(sep_base, 0.2, n),
            rng.normal(0.0, 0.2, n),  # base pos, neg
        )

    def test_large_gap_is_specialization(self):
        rng = np.random.default_rng(3)
        trained = self._relation(rng, sep_adapted=1.2, sep_base=0.4)  # big improvement
        independent = self._relation(rng, sep_adapted=0.42, sep_base=0.4)  # tiny improvement
        g = cross_relation_gap(trained, independent, n_boot=300, rng=np.random.default_rng(7))
        assert isinstance(g, GeneralizationGap)
        assert g.trained_delta > g.independent_delta
        assert g.gap > 0
        assert "specialization" in g.interpretation or "only" in g.interpretation

    def test_transfer_when_gap_small(self):
        rng = np.random.default_rng(4)
        trained = self._relation(rng, sep_adapted=0.9, sep_base=0.4)
        independent = self._relation(rng, sep_adapted=0.85, sep_base=0.4)  # transfers
        g = cross_relation_gap(trained, independent, n_boot=300, rng=np.random.default_rng(7))
        assert g.independent_delta > 0
        assert "transfer" in g.interpretation
