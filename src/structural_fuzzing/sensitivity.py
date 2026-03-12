"""Sensitivity profiling via ablation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass
class SensitivityResult:
    """Result of ablating a single dimension."""

    dim: int
    dim_name: str
    mae_with: float
    mae_without: float
    delta_mae: float
    importance_rank: int


def sensitivity_profile(
    params: np.ndarray,
    dim_names: Sequence[str],
    evaluate_fn: Callable[[np.ndarray], tuple[float, dict[str, float]]],
    inactive_value: float = 1e6,
) -> list[SensitivityResult]:
    """Compute sensitivity profile by ablating each dimension.

    For each dimension, set its value to inactive_value and measure the
    change in MAE relative to the baseline (all dimensions active).

    Parameters
    ----------
    params : np.ndarray
        Baseline parameter values (1D array, one per dimension).
    dim_names : sequence of str
        Names for all dimensions.
    evaluate_fn : callable
        Function (params: ndarray) -> (mae: float, errors: dict).
    inactive_value : float
        Value to set ablated dimensions to.

    Returns
    -------
    list[SensitivityResult]
        Results sorted by delta_mae descending (most important first).
    """
    # Baseline MAE
    base_mae, _ = evaluate_fn(params)

    results: list[SensitivityResult] = []
    for i, name in enumerate(dim_names):
        ablated = params.copy()
        ablated[i] = inactive_value
        ablated_mae, _ = evaluate_fn(ablated)
        delta = ablated_mae - base_mae
        results.append(
            SensitivityResult(
                dim=i,
                dim_name=name,
                mae_with=base_mae,
                mae_without=ablated_mae,
                delta_mae=delta,
                importance_rank=0,  # assigned below
            )
        )

    # Sort by delta_mae descending and assign ranks
    results.sort(key=lambda r: r.delta_mae, reverse=True)
    for rank, r in enumerate(results, 1):
        r.importance_rank = rank

    return results
