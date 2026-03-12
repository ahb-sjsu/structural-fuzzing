"""Tests for Pareto frontier extraction."""

from __future__ import annotations

import numpy as np

from structural_fuzzing.core import SubsetResult
from structural_fuzzing.pareto import pareto_frontier


def _make_result(dims, n_dims, mae):
    """Helper to create a SubsetResult for testing."""
    return SubsetResult(
        dims=tuple(dims),
        dim_names=tuple(f"d{d}" for d in dims),
        n_dims=n_dims,
        param_values=np.ones(3),
        mae=mae,
        errors={"t": mae},
    )


class TestParetoFrontier:
    def test_empty_input(self):
        assert pareto_frontier([]) == []

    def test_single_result(self):
        r = _make_result([0], 1, 5.0)
        pareto = pareto_frontier([r])
        assert len(pareto) == 1
        assert pareto[0].pareto_optimal is True

    def test_simple_frontier(self):
        """Test with clear Pareto structure: lower dims have higher MAE."""
        results = [
            _make_result([0], 1, 10.0),
            _make_result([0, 1], 2, 5.0),
            _make_result([0, 1, 2], 3, 2.0),
        ]
        pareto = pareto_frontier(results)
        # All three should be Pareto-optimal (each improves on previous)
        assert len(pareto) == 3
        assert all(r.pareto_optimal for r in pareto)
        # Should be sorted by n_dims
        assert pareto[0].n_dims == 1
        assert pareto[1].n_dims == 2
        assert pareto[2].n_dims == 3

    def test_dominated_result(self):
        """Test that dominated results are excluded."""
        results = [
            _make_result([0], 1, 5.0),  # Good 1D
            _make_result([0, 1], 2, 8.0),  # Worse than 1D: dominated
            _make_result([0, 1, 2], 3, 3.0),  # Better than both: Pareto
        ]
        pareto = pareto_frontier(results)
        # 1D is Pareto (best at k=1)
        # 2D is dominated by 1D (worse MAE, more dims)
        # 3D is Pareto (better MAE than 1D, but needs 3 dims)
        assert len(pareto) == 2
        dims_in_pareto = {r.n_dims for r in pareto}
        assert dims_in_pareto == {1, 3}

    def test_tolerance(self):
        """Test tolerance parameter."""
        results = [
            _make_result([0], 1, 5.0),
            _make_result([0, 1], 2, 4.99),  # Within tolerance of 0.01
        ]
        pareto = pareto_frontier(results, tolerance=0.01)
        # 2D is only 0.01 better, within tolerance -> not a meaningful improvement
        assert len(pareto) == 1
        assert pareto[0].n_dims == 1

    def test_multiple_at_same_dim(self):
        """Test with multiple results at same dimensionality."""
        results = [
            _make_result([0], 1, 10.0),
            _make_result([1], 1, 5.0),  # Better 1D
            _make_result([0, 1], 2, 3.0),
        ]
        pareto = pareto_frontier(results)
        # Best at k=1 is 5.0, best at k=2 is 3.0
        assert len(pareto) == 2
        assert pareto[0].mae == 5.0
        assert pareto[1].mae == 3.0

    def test_marks_pareto_optimal_flag(self):
        """Test that pareto_optimal flag is set correctly."""
        r1 = _make_result([0], 1, 5.0)
        r2 = _make_result([0, 1], 2, 3.0)
        r3 = _make_result([1], 1, 8.0)

        pareto_frontier([r1, r2, r3])
        assert r1.pareto_optimal is True  # best at k=1
        assert r2.pareto_optimal is True  # improves on k=1
        assert r3.pareto_optimal is False  # dominated by r1
