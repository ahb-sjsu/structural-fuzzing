"""Tests for Model Robustness Index."""

from __future__ import annotations

import numpy as np
import pytest

from structural_fuzzing.mri import ModelRobustnessIndex, compute_mri


class TestComputeMRI:
    def test_returns_valid_mri(self, simple_evaluate_fn, simple_params):
        """Test that MRI returns a valid positive value."""
        mri = compute_mri(
            params=simple_params,
            evaluate_fn=simple_evaluate_fn,
            n_perturbations=50,
        )
        assert isinstance(mri, ModelRobustnessIndex)
        assert mri.mri >= 0
        assert mri.n_perturbations == 50

    def test_mri_components(self, simple_evaluate_fn, simple_params):
        """Test MRI components: mean, P75, P95."""
        mri = compute_mri(
            params=simple_params,
            evaluate_fn=simple_evaluate_fn,
            n_perturbations=100,
        )
        # P95 >= P75 >= mean (for non-negative values)
        assert mri.p95_omega >= mri.p75_omega
        assert mri.p75_omega >= mri.mean_omega
        assert mri.mean_omega >= 0

    def test_custom_weights(self, simple_evaluate_fn, simple_params):
        """Test that custom weights change MRI value."""
        mri1 = compute_mri(
            params=simple_params,
            evaluate_fn=simple_evaluate_fn,
            n_perturbations=50,
            weights=(1.0, 0.0, 0.0),
            rng=np.random.default_rng(42),
        )
        mri2 = compute_mri(
            params=simple_params,
            evaluate_fn=simple_evaluate_fn,
            n_perturbations=50,
            weights=(0.0, 0.0, 1.0),
            rng=np.random.default_rng(42),
        )
        # With weight only on mean vs. only on P95, values should differ
        # (unless all perturbation errors are identical, which is unlikely)
        assert mri1.mri != mri2.mri or mri1.mean_omega == mri1.p95_omega

    def test_worst_case(self, simple_evaluate_fn, simple_params):
        """Test worst-case MAE is at least as large as base MAE."""
        base_mae, _ = simple_evaluate_fn(simple_params)
        mri = compute_mri(
            params=simple_params,
            evaluate_fn=simple_evaluate_fn,
            n_perturbations=100,
        )
        assert mri.worst_case_mae >= base_mae

    def test_perturbation_errors_length(self, simple_evaluate_fn, simple_params):
        """Test that we get the right number of perturbation errors."""
        mri = compute_mri(
            params=simple_params,
            evaluate_fn=simple_evaluate_fn,
            n_perturbations=30,
        )
        assert len(mri.perturbation_errors) == 30

    def test_reproducible_with_rng(self, simple_evaluate_fn, simple_params):
        """Test reproducibility with same RNG seed."""
        mri1 = compute_mri(
            params=simple_params,
            evaluate_fn=simple_evaluate_fn,
            n_perturbations=50,
            rng=np.random.default_rng(123),
        )
        mri2 = compute_mri(
            params=simple_params,
            evaluate_fn=simple_evaluate_fn,
            n_perturbations=50,
            rng=np.random.default_rng(123),
        )
        assert mri1.mri == mri2.mri

    def test_constant_model_low_mri(self):
        """Test that a constant model has MRI = 0."""

        def constant_fn(params):
            return 5.0, {"t": 5.0}

        params = np.array([1.0, 1.0])
        mri = compute_mri(
            params=params,
            evaluate_fn=constant_fn,
            n_perturbations=50,
        )
        assert mri.mri == pytest.approx(0.0, abs=1e-10)

    def test_mri_formula(self, simple_evaluate_fn, simple_params):
        """Test MRI = w0*mean + w1*P75 + w2*P95."""
        weights = (0.5, 0.3, 0.2)
        mri = compute_mri(
            params=simple_params,
            evaluate_fn=simple_evaluate_fn,
            n_perturbations=50,
            weights=weights,
            rng=np.random.default_rng(42),
        )
        expected = (
            weights[0] * mri.mean_omega + weights[1] * mri.p75_omega + weights[2] * mri.p95_omega
        )
        assert mri.mri == pytest.approx(expected, rel=1e-10)
