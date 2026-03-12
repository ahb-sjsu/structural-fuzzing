"""Tests for baseline comparison methods."""

from __future__ import annotations

from structural_fuzzing.baselines import backward_elimination, forward_selection, lasso_selection


class TestForwardSelection:
    def test_returns_results(self, simple_evaluate_fn, simple_dim_names):
        """Test that forward selection returns results."""
        results = forward_selection(
            dim_names=simple_dim_names,
            evaluate_fn=simple_evaluate_fn,
            n_grid=10,
        )
        assert len(results) > 0

    def test_max_dims(self, simple_evaluate_fn, simple_dim_names):
        """Test max_dims parameter limits results."""
        results = forward_selection(
            dim_names=simple_dim_names,
            evaluate_fn=simple_evaluate_fn,
            max_dims=2,
            n_grid=10,
        )
        assert len(results) == 2

    def test_increasing_dims(self, simple_evaluate_fn, simple_dim_names):
        """Test that each step adds one dimension."""
        results = forward_selection(
            dim_names=simple_dim_names,
            evaluate_fn=simple_evaluate_fn,
            n_grid=10,
        )
        for i, r in enumerate(results):
            assert r.n_dims == i + 1

    def test_structure(self, simple_evaluate_fn, simple_dim_names):
        """Test that results have expected structure."""
        results = forward_selection(
            dim_names=simple_dim_names,
            evaluate_fn=simple_evaluate_fn,
            n_grid=10,
        )
        for r in results:
            assert r.param_values is not None
            assert len(r.errors) > 0
            assert r.mae >= 0


class TestBackwardElimination:
    def test_returns_results(self, simple_evaluate_fn, simple_dim_names):
        """Test that backward elimination returns results."""
        results = backward_elimination(
            dim_names=simple_dim_names,
            evaluate_fn=simple_evaluate_fn,
            n_grid=10,
            n_random=100,
        )
        assert len(results) > 0

    def test_decreasing_dims(self, simple_evaluate_fn, simple_dim_names):
        """Test that dimensions decrease at each step."""
        results = backward_elimination(
            dim_names=simple_dim_names,
            evaluate_fn=simple_evaluate_fn,
            n_grid=10,
            n_random=100,
        )
        n_all = len(simple_dim_names)
        for i, r in enumerate(results):
            assert r.n_dims == n_all - i

    def test_starts_with_all_dims(self, simple_evaluate_fn, simple_dim_names):
        """Test that first result has all dimensions."""
        results = backward_elimination(
            dim_names=simple_dim_names,
            evaluate_fn=simple_evaluate_fn,
            n_grid=10,
            n_random=100,
        )
        assert results[0].n_dims == len(simple_dim_names)

    def test_ends_with_one_dim(self, simple_evaluate_fn, simple_dim_names):
        """Test that last result has one dimension."""
        results = backward_elimination(
            dim_names=simple_dim_names,
            evaluate_fn=simple_evaluate_fn,
            n_grid=10,
            n_random=100,
        )
        assert results[-1].n_dims == 1


class TestLassoSelection:
    def test_returns_results(self, simple_evaluate_fn, simple_dim_names):
        """Test that LASSO selection returns results."""
        results = lasso_selection(
            dim_names=simple_dim_names,
            evaluate_fn=simple_evaluate_fn,
            n_random=200,
        )
        assert len(results) > 0

    def test_sorted_by_dims(self, simple_evaluate_fn, simple_dim_names):
        """Test that results are sorted by n_dims."""
        results = lasso_selection(
            dim_names=simple_dim_names,
            evaluate_fn=simple_evaluate_fn,
            n_random=200,
        )
        for i in range(len(results) - 1):
            assert results[i].n_dims <= results[i + 1].n_dims

    def test_custom_alphas(self, simple_evaluate_fn, simple_dim_names):
        """Test with custom alpha values."""
        results = lasso_selection(
            dim_names=simple_dim_names,
            evaluate_fn=simple_evaluate_fn,
            alphas=[0.01, 0.1, 1.0, 10.0],
            n_random=200,
        )
        assert len(results) > 0
