"""Shared test fixtures and utilities."""

from __future__ import annotations

import numpy as np
import pytest


def make_simple_evaluate_fn():
    """3-dim model where dim 1 matters most, dim 2 somewhat, dim 0 not at all.

    Returns a function compatible with structural_fuzzing's evaluate_fn interface.
    """

    def evaluate_fn(params: np.ndarray) -> tuple[float, dict[str, float]]:
        # Dim 1 (index 1) drives accuracy, dim 2 helps, dim 0 is noise
        base_error = 10.0
        if params[1] < 1000:
            base_error -= 5.0 * (1.0 / (1.0 + params[1]))
        if params[2] < 1000:
            base_error -= 3.0 * (1.0 / (1.0 + params[2]))
        # dim 0 doesn't help at all
        errors = {"target_a": base_error, "target_b": base_error * 0.8}
        mae = sum(abs(v) for v in errors.values()) / len(errors)
        return mae, errors

    return evaluate_fn


@pytest.fixture
def simple_evaluate_fn():
    """Fixture providing the simple 3-dim evaluate function."""
    return make_simple_evaluate_fn()


@pytest.fixture
def simple_dim_names():
    """Fixture providing dimension names for the simple model."""
    return ["noise_dim", "important_dim", "helpful_dim"]


@pytest.fixture
def simple_params():
    """Fixture providing reasonable parameter values for the simple model."""
    return np.array([1.0, 0.5, 0.5])
