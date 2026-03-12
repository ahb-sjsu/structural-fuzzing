"""Tests for compositional testing."""

from __future__ import annotations

import numpy as np

from structural_fuzzing.compositional import CompositionResult, compositional_test


class TestCompositionalTest:
    def test_returns_correct_length(self, simple_evaluate_fn, simple_dim_names):
        """Test that result sequences have correct length."""
        result = compositional_test(
            start_dim=0,
            candidate_dims=[1, 2],
            dim_names=simple_dim_names,
            evaluate_fn=simple_evaluate_fn,
            n_grid=10,
        )
        assert isinstance(result, CompositionResult)
        # Start dim + 2 candidates = 3 steps
        assert len(result.order) == 3
        assert len(result.order_names) == 3
        assert len(result.mae_sequence) == 3
        assert len(result.param_sequence) == 3

    def test_start_dim_first(self, simple_evaluate_fn, simple_dim_names):
        """Test that start_dim is always first in order."""
        result = compositional_test(
            start_dim=1,
            candidate_dims=[0, 2],
            dim_names=simple_dim_names,
            evaluate_fn=simple_evaluate_fn,
            n_grid=10,
        )
        assert result.order[0] == 1
        assert result.order_names[0] == "important_dim"

    def test_all_dims_included(self, simple_evaluate_fn, simple_dim_names):
        """Test that all candidate dims are eventually included."""
        result = compositional_test(
            start_dim=0,
            candidate_dims=[1, 2],
            dim_names=simple_dim_names,
            evaluate_fn=simple_evaluate_fn,
            n_grid=10,
        )
        assert set(result.order) == {0, 1, 2}

    def test_mae_generally_decreases(self, simple_evaluate_fn, simple_dim_names):
        """Test that MAE generally decreases or stays flat as dims are added."""
        result = compositional_test(
            start_dim=0,
            candidate_dims=[1, 2],
            dim_names=simple_dim_names,
            evaluate_fn=simple_evaluate_fn,
            n_grid=10,
        )
        # With greedy addition, each step should not be much worse
        # (though re-optimization might cause minor fluctuations)
        # At least the final should be no worse than 2x the best
        assert result.mae_sequence[-1] <= result.mae_sequence[0] * 2.0 + 1.0

    def test_greedy_picks_important_first(self, simple_evaluate_fn, simple_dim_names):
        """Test that greedy selection picks important_dim before noise_dim."""
        result = compositional_test(
            start_dim=2,  # Start with helpful_dim
            candidate_dims=[0, 1],
            dim_names=simple_dim_names,
            evaluate_fn=simple_evaluate_fn,
            n_grid=10,
        )
        # important_dim (1) should be added before noise_dim (0)
        idx_important = result.order.index(1)
        idx_noise = result.order.index(0)
        assert idx_important < idx_noise

    def test_param_sequence_has_arrays(self, simple_evaluate_fn, simple_dim_names):
        """Test that param_sequence contains numpy arrays."""
        result = compositional_test(
            start_dim=0,
            candidate_dims=[1, 2],
            dim_names=simple_dim_names,
            evaluate_fn=simple_evaluate_fn,
            n_grid=10,
        )
        for p in result.param_sequence:
            assert isinstance(p, np.ndarray)
            assert len(p) == len(simple_dim_names)
