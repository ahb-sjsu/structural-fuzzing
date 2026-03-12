"""Compositional testing: greedy dimension-building sequences."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from structural_fuzzing.core import optimize_subset


@dataclass
class CompositionResult:
    """Result of compositional (greedy dimension-building) test."""

    order: list[int]
    order_names: list[str]
    mae_sequence: list[float]
    param_sequence: list[np.ndarray] = field(repr=False)


def compositional_test(
    start_dim: int,
    candidate_dims: Sequence[int],
    dim_names: Sequence[str],
    evaluate_fn: Callable[[np.ndarray], tuple[float, dict[str, float]]],
    inactive_value: float = 1e6,
    n_grid: int = 20,
    n_random: int = 5000,
) -> CompositionResult:
    """Build a greedy dimension-addition sequence.

    Starting from start_dim, iteratively add the candidate dimension that
    produces the lowest MAE. At each step, re-optimize all active dimensions.

    Parameters
    ----------
    start_dim : int
        Index of the starting dimension.
    candidate_dims : sequence of int
        Indices of candidate dimensions to add.
    dim_names : sequence of str
        Names for all dimensions.
    evaluate_fn : callable
        Function (params: ndarray) -> (mae: float, errors: dict).
    inactive_value : float
        Value for inactive dimensions.
    n_grid : int
        Grid points for 1D/2D optimization.
    n_random : int
        Random samples for 3D+ optimization.

    Returns
    -------
    CompositionResult
        The greedy construction sequence with MAE at each step.
    """
    active = [start_dim]
    remaining = list(candidate_dims)
    if start_dim in remaining:
        remaining.remove(start_dim)

    order = [start_dim]
    order_names = [dim_names[start_dim]]
    mae_sequence: list[float] = []
    param_sequence: list[np.ndarray] = []

    # Evaluate starting configuration
    result = optimize_subset(
        active_dims=active,
        all_dim_names=dim_names,
        evaluate_fn=evaluate_fn,
        inactive_value=inactive_value,
        n_grid=n_grid,
        n_random=n_random,
    )
    mae_sequence.append(result.mae)
    param_sequence.append(result.param_values.copy())

    while remaining:
        best_mae = float("inf")
        best_dim = remaining[0]
        best_params = None

        for candidate in remaining:
            trial_dims = active + [candidate]
            trial_result = optimize_subset(
                active_dims=trial_dims,
                all_dim_names=dim_names,
                evaluate_fn=evaluate_fn,
                inactive_value=inactive_value,
                n_grid=n_grid,
                n_random=n_random,
            )
            if trial_result.mae < best_mae:
                best_mae = trial_result.mae
                best_dim = candidate
                best_params = trial_result.param_values.copy()

        active.append(best_dim)
        remaining.remove(best_dim)
        order.append(best_dim)
        order_names.append(dim_names[best_dim])
        mae_sequence.append(best_mae)
        param_sequence.append(best_params)  # type: ignore[arg-type]

    return CompositionResult(
        order=order,
        order_names=order_names,
        mae_sequence=mae_sequence,
        param_sequence=param_sequence,
    )
