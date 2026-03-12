"""Tests for example models (smoke tests)."""

from __future__ import annotations

import numpy as np
import pytest


class TestGeometricEconomics:
    def test_evaluate_fn_works(self):
        """Test that geometric economics evaluate_fn returns valid output."""
        from examples.geometric_economics.targets import make_evaluate_fn

        evaluate_fn = make_evaluate_fn()
        params = np.ones(9)
        mae, errors = evaluate_fn(params)

        assert isinstance(mae, float)
        assert mae >= 0
        assert isinstance(errors, dict)
        assert len(errors) > 0

    def test_evaluate_fn_with_varied_params(self):
        """Test evaluate_fn with different parameter configurations."""
        from examples.geometric_economics.targets import make_evaluate_fn

        evaluate_fn = make_evaluate_fn()

        # All small variances (all dimensions active and weighted)
        params_small = np.full(9, 0.1)
        mae_small, _ = evaluate_fn(params_small)
        assert mae_small >= 0

        # All large variances (dimensions deactivated)
        params_large = np.full(9, 1e6)
        mae_large, _ = evaluate_fn(params_large)
        assert mae_large >= 0

    def test_model_components(self):
        """Test individual model components."""
        from examples.geometric_economics.model import (
            Prospect,
            mahalanobis_distance,
            prospect_to_state,
            public_goods_state,
            ultimatum_state,
        )

        # Mahalanobis distance with identity
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0])
        d = mahalanobis_distance(a, b, np.eye(3))
        assert d == pytest.approx(1.0, abs=1e-10)

        # Ultimatum state
        state = ultimatum_state(10.0, 50.0)
        assert len(state) == 9
        assert state[0] > 0  # positive payoff

        # Public goods state
        state = public_goods_state(20.0, 50.0)
        assert len(state) == 9

        # Prospect to state
        p = Prospect(outcomes=[100, 0], probabilities=[0.5, 0.5])
        state = prospect_to_state(p)
        assert len(state) == 9

    def test_targets_count(self):
        """Test that we have 16 targets."""
        from examples.geometric_economics.targets import build_targets

        targets = build_targets()
        assert len(targets) == 16

    def test_quick_campaign(self):
        """Smoke test: run a very small campaign."""
        from examples.geometric_economics.model import DIM_NAMES
        from examples.geometric_economics.targets import make_evaluate_fn
        from structural_fuzzing import run_campaign

        # Use only 3 dims to keep test fast
        evaluate_fn = make_evaluate_fn()
        short_dims = DIM_NAMES[:3]

        # Wrap evaluate_fn to only use 3 dims
        def eval3(params):
            import numpy as np

            full = np.full(9, 1e6)
            full[:3] = params[:3]
            return evaluate_fn(full)

        report = run_campaign(
            dim_names=short_dims,
            evaluate_fn=eval3,
            max_subset_dims=1,
            n_mri_perturbations=5,
            run_baselines=False,
            verbose=False,
            n_grid=3,
            n_random=10,
            adversarial_tolerance=100.0,
        )
        assert len(report.subset_results) == 3  # C(3,1) = 3
        assert report.mri_result is not None


class TestDefectPrediction:
    def test_generate_data(self):
        """Test synthetic data generation."""
        from examples.defect_prediction.model import generate_defect_data

        X, y = generate_defect_data(n_samples=100)
        assert X.shape == (100, 16)
        assert y.shape == (100,)
        assert set(np.unique(y)).issubset({0, 1})

    def test_evaluate_fn_works(self):
        """Test that defect prediction evaluate_fn returns valid output."""
        from examples.defect_prediction.model import N_GROUPS, make_evaluate_fn

        evaluate_fn = make_evaluate_fn(n_samples=200)
        params = np.ones(N_GROUPS)
        mae, errors = evaluate_fn(params)

        assert isinstance(mae, float)
        assert mae >= 0
        assert isinstance(errors, dict)
        assert len(errors) == 5  # 5 metrics

    def test_evaluate_fn_with_inactive_groups(self):
        """Test evaluate_fn with some groups deactivated."""
        from examples.defect_prediction.model import N_GROUPS, make_evaluate_fn

        evaluate_fn = make_evaluate_fn(n_samples=200)

        # Only complexity active
        params = np.full(N_GROUPS, 1e6)
        params[1] = 1.0  # Complexity group
        mae, errors = evaluate_fn(params)
        assert mae >= 0

    def test_evaluate_fn_no_groups(self):
        """Test evaluate_fn with all groups deactivated."""
        from examples.defect_prediction.model import N_GROUPS, make_evaluate_fn

        evaluate_fn = make_evaluate_fn(n_samples=200)
        params = np.full(N_GROUPS, 1e6)
        mae, errors = evaluate_fn(params)
        assert mae >= 0

    def test_quick_campaign(self):
        """Smoke test: run a very small campaign on defect prediction."""
        from examples.defect_prediction.model import GROUP_NAMES, make_evaluate_fn
        from structural_fuzzing import run_campaign

        evaluate_fn = make_evaluate_fn(n_samples=100)
        report = run_campaign(
            dim_names=GROUP_NAMES,
            evaluate_fn=evaluate_fn,
            max_subset_dims=1,
            n_mri_perturbations=5,
            run_baselines=False,
            verbose=False,
            n_grid=3,
            n_random=10,
            adversarial_tolerance=100.0,  # high tolerance to skip adversarial
        )
        assert len(report.subset_results) > 0
        assert report.mri_result is not None
