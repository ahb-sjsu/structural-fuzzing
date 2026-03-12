"""Tests for core framework: SubsetResult, enumerate_subsets, optimize_subset."""

from __future__ import annotations

import numpy as np

from structural_fuzzing.core import SubsetResult, enumerate_subsets, optimize_subset


class TestSubsetResult:
    def test_creation(self):
        result = SubsetResult(
            dims=(0, 1),
            dim_names=("a", "b"),
            n_dims=2,
            param_values=np.array([1.0, 2.0, 1e6]),
            mae=3.5,
            errors={"t1": 3.0, "t2": 4.0},
        )
        assert result.n_dims == 2
        assert result.mae == 3.5
        assert result.pareto_optimal is False

    def test_repr(self):
        result = SubsetResult(
            dims=(0,),
            dim_names=("alpha",),
            n_dims=1,
            param_values=np.array([1.0]),
            mae=2.0,
            errors={"t1": 2.0},
        )
        assert "alpha" in repr(result)
        assert "2.0000" in repr(result)


class TestOptimizeSubset:
    def test_1d_search(self, simple_evaluate_fn, simple_dim_names):
        """Test 1D grid search finds reasonable optimum."""
        result = optimize_subset(
            active_dims=[1],  # important_dim
            all_dim_names=simple_dim_names,
            evaluate_fn=simple_evaluate_fn,
            n_grid=20,
        )
        assert result.n_dims == 1
        assert result.dims == (1,)
        assert result.dim_names == ("important_dim",)
        # Should find a good value (low param -> high importance)
        assert result.mae < 10.0  # better than baseline

    def test_2d_search(self, simple_evaluate_fn, simple_dim_names):
        """Test 2D grid search."""
        result = optimize_subset(
            active_dims=[1, 2],
            all_dim_names=simple_dim_names,
            evaluate_fn=simple_evaluate_fn,
            n_grid=10,
        )
        assert result.n_dims == 2
        assert result.dims == (1, 2)
        # 2D should be at least as good as 1D
        result_1d = optimize_subset(
            active_dims=[1],
            all_dim_names=simple_dim_names,
            evaluate_fn=simple_evaluate_fn,
            n_grid=20,
        )
        assert result.mae <= result_1d.mae + 0.5  # 2D should not be much worse

    def test_3d_random_search(self, simple_evaluate_fn, simple_dim_names):
        """Test 3D random search."""
        result = optimize_subset(
            active_dims=[0, 1, 2],
            all_dim_names=simple_dim_names,
            evaluate_fn=simple_evaluate_fn,
            n_random=500,
        )
        assert result.n_dims == 3
        assert result.mae >= 0  # MAE is non-negative

    def test_empty_subset(self, simple_evaluate_fn, simple_dim_names):
        """Test empty subset returns a result with no active dims."""
        result = optimize_subset(
            active_dims=[],
            all_dim_names=simple_dim_names,
            evaluate_fn=simple_evaluate_fn,
        )
        assert result.n_dims == 0
        assert result.dims == ()

    def test_inactive_value_propagated(self, simple_dim_names):
        """Test that inactive dimensions are set to inactive_value."""
        inactive_val = 999.0

        def check_fn(params):
            # Verify inactive dims have the right value
            assert params[0] == inactive_val
            assert params[2] == inactive_val
            return 5.0, {"t": 5.0}

        optimize_subset(
            active_dims=[1],
            all_dim_names=simple_dim_names,
            evaluate_fn=check_fn,
            inactive_value=inactive_val,
            n_grid=5,
        )


class TestEnumerateSubsets:
    def test_enumerate_returns_sorted(self, simple_evaluate_fn, simple_dim_names):
        """Test that results are sorted by MAE."""
        results = enumerate_subsets(
            dim_names=simple_dim_names,
            evaluate_fn=simple_evaluate_fn,
            max_dims=2,
            n_grid=10,
        )
        assert len(results) > 0
        for i in range(len(results) - 1):
            assert results[i].mae <= results[i + 1].mae

    def test_enumerate_count(self, simple_evaluate_fn, simple_dim_names):
        """Test correct number of subsets enumerated."""
        # 3 dims, max_dims=2: C(3,1) + C(3,2) = 3 + 3 = 6
        results = enumerate_subsets(
            dim_names=simple_dim_names,
            evaluate_fn=simple_evaluate_fn,
            max_dims=2,
            n_grid=10,
        )
        assert len(results) == 6

    def test_enumerate_max_dims_3(self, simple_evaluate_fn, simple_dim_names):
        """Test enumeration up to 3 dims."""
        # C(3,1) + C(3,2) + C(3,3) = 3 + 3 + 1 = 7
        results = enumerate_subsets(
            dim_names=simple_dim_names,
            evaluate_fn=simple_evaluate_fn,
            max_dims=3,
            n_grid=10,
            n_random=100,
        )
        assert len(results) == 7

    def test_best_includes_important_dim(self, simple_evaluate_fn, simple_dim_names):
        """Test that the best subset includes the important dimension."""
        results = enumerate_subsets(
            dim_names=simple_dim_names,
            evaluate_fn=simple_evaluate_fn,
            max_dims=2,
            n_grid=10,
        )
        # The important_dim (index 1) should appear in one of the top results
        top_3 = results[:3]
        assert any(1 in r.dims for r in top_3)
