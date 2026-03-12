"""Tests for adversarial threshold search."""

from __future__ import annotations

import numpy as np
import pytest

from structural_fuzzing.adversarial import AdversarialResult, find_adversarial_threshold


class TestFindAdversarialThreshold:
    def test_finds_threshold_for_sensitive_dim(self):
        """Test that thresholds are found for a sensitive dimension."""

        def sensitive_fn(params):
            # Error depends strongly on params[0]
            error_a = 5.0 + 10.0 * np.log10(max(params[0], 1e-6))
            error_b = 3.0
            errors = {"target_a": error_a, "target_b": error_b}
            mae = sum(abs(v) for v in errors.values()) / len(errors)
            return mae, errors

        params = np.array([1.0, 1.0])
        results = find_adversarial_threshold(
            params=params,
            dim=0,
            dim_names=["sensitive", "stable"],
            evaluate_fn=sensitive_fn,
            tolerance=0.5,
            n_steps=30,
        )
        # Should find at least one threshold (in increase direction)
        assert len(results) > 0
        assert all(isinstance(r, AdversarialResult) for r in results)

    def test_direction_field(self):
        """Test that direction is correctly labeled."""

        def fn(params):
            error = abs(params[0] - 1.0) * 5.0
            return error, {"t": error}

        params = np.array([1.0, 1.0])
        results = find_adversarial_threshold(
            params=params,
            dim=0,
            dim_names=["d0", "d1"],
            evaluate_fn=fn,
            tolerance=0.5,
        )
        directions = {r.direction for r in results}
        assert directions.issubset({"increase", "decrease"})

    def test_no_threshold_for_insensitive_dim(self):
        """Test that no threshold is found for an insensitive dimension."""

        def constant_fn(params):
            return 5.0, {"t": 5.0}

        params = np.array([1.0, 1.0])
        results = find_adversarial_threshold(
            params=params,
            dim=0,
            dim_names=["d0", "d1"],
            evaluate_fn=constant_fn,
            tolerance=0.5,
        )
        assert len(results) == 0

    def test_result_fields(self):
        """Test that AdversarialResult fields are populated correctly."""

        def fn(params):
            error = params[0] * 2.0
            return error, {"t": error}

        params = np.array([1.0, 1.0])
        results = find_adversarial_threshold(
            params=params,
            dim=0,
            dim_names=["d0", "d1"],
            evaluate_fn=fn,
            tolerance=0.5,
        )
        for r in results:
            assert r.dim == 0
            assert r.dim_name == "d0"
            assert r.base_value == pytest.approx(1.0)
            assert r.threshold_ratio > 0
            assert r.target_flipped == "t"

    def test_threshold_ratio(self):
        """Test that threshold_ratio = threshold_value / base_value."""

        def fn(params):
            error = params[0] ** 2
            return error, {"t": error}

        params = np.array([1.0, 1.0])
        results = find_adversarial_threshold(
            params=params,
            dim=0,
            dim_names=["d0", "d1"],
            evaluate_fn=fn,
            tolerance=0.5,
        )
        for r in results:
            assert r.threshold_ratio == pytest.approx(r.threshold_value / r.base_value, rel=1e-6)
