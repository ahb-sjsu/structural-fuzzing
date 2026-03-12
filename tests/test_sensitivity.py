"""Tests for sensitivity profiling."""

from __future__ import annotations

import numpy as np

from structural_fuzzing.sensitivity import SensitivityResult, sensitivity_profile


class TestSensitivityProfile:
    def test_returns_all_dims(self, simple_evaluate_fn, simple_dim_names, simple_params):
        """Test that sensitivity profile returns one result per dimension."""
        results = sensitivity_profile(
            params=simple_params,
            dim_names=simple_dim_names,
            evaluate_fn=simple_evaluate_fn,
        )
        assert len(results) == len(simple_dim_names)

    def test_ranking_correct(self, simple_evaluate_fn, simple_dim_names, simple_params):
        """Test that ranking identifies the important dimension correctly."""
        results = sensitivity_profile(
            params=simple_params,
            dim_names=simple_dim_names,
            evaluate_fn=simple_evaluate_fn,
        )
        # Dim 1 (important_dim) should rank higher than dim 0 (noise_dim)
        dim1_result = next(r for r in results if r.dim == 1)
        dim0_result = next(r for r in results if r.dim == 0)
        assert dim1_result.importance_rank < dim0_result.importance_rank

    def test_ranks_are_sequential(self, simple_evaluate_fn, simple_dim_names, simple_params):
        """Test that ranks are 1, 2, 3, ..."""
        results = sensitivity_profile(
            params=simple_params,
            dim_names=simple_dim_names,
            evaluate_fn=simple_evaluate_fn,
        )
        ranks = sorted(r.importance_rank for r in results)
        assert ranks == list(range(1, len(simple_dim_names) + 1))

    def test_sorted_by_delta_descending(self, simple_evaluate_fn, simple_dim_names, simple_params):
        """Test that results are sorted by delta_mae descending."""
        results = sensitivity_profile(
            params=simple_params,
            dim_names=simple_dim_names,
            evaluate_fn=simple_evaluate_fn,
        )
        for i in range(len(results) - 1):
            assert results[i].delta_mae >= results[i + 1].delta_mae

    def test_zero_sensitivity_detection(self, simple_dim_names):
        """Test that a dimension with no effect has near-zero delta."""

        def constant_fn(params):
            return 5.0, {"t": 5.0}

        params = np.array([1.0, 1.0, 1.0])
        results = sensitivity_profile(
            params=params,
            dim_names=simple_dim_names,
            evaluate_fn=constant_fn,
        )
        for r in results:
            assert abs(r.delta_mae) < 1e-10

    def test_mae_with_is_baseline(self, simple_evaluate_fn, simple_dim_names, simple_params):
        """Test that mae_with equals the baseline MAE for all results."""
        base_mae, _ = simple_evaluate_fn(simple_params)
        results = sensitivity_profile(
            params=simple_params,
            dim_names=simple_dim_names,
            evaluate_fn=simple_evaluate_fn,
        )
        for r in results:
            assert abs(r.mae_with - base_mae) < 1e-10

    def test_result_dataclass(self):
        """Test SensitivityResult creation."""
        sr = SensitivityResult(
            dim=0,
            dim_name="test",
            mae_with=5.0,
            mae_without=8.0,
            delta_mae=3.0,
            importance_rank=1,
        )
        assert sr.dim == 0
        assert sr.delta_mae == 3.0
