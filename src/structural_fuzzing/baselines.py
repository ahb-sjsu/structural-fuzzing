"""Comparison baselines: forward selection, backward elimination, LASSO."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from structural_fuzzing.core import SubsetResult, optimize_subset


def forward_selection(
    dim_names: Sequence[str],
    evaluate_fn: Callable[[np.ndarray], tuple[float, dict[str, float]]],
    max_dims: int | None = None,
    inactive_value: float = 1e6,
    n_grid: int = 20,
    n_random: int = 5000,
) -> list[SubsetResult]:
    """Standard forward selection: start empty, greedily add best dimension.

    Parameters
    ----------
    dim_names : sequence of str
        Names for all dimensions.
    evaluate_fn : callable
        Function (params: ndarray) -> (mae: float, errors: dict).
    max_dims : int or None
        Maximum dimensions to select. None means all.
    inactive_value : float
        Value for inactive dimensions.
    n_grid : int
        Grid points for optimization.
    n_random : int
        Random samples for 3D+ optimization.

    Returns
    -------
    list[SubsetResult]
        Results at each step of forward selection.
    """
    n_all = len(dim_names)
    if max_dims is None:
        max_dims = n_all

    selected: list[int] = []
    remaining = list(range(n_all))
    results: list[SubsetResult] = []

    for _ in range(min(max_dims, n_all)):
        best_mae = float("inf")
        best_dim = remaining[0]
        best_result = None

        for candidate in remaining:
            trial_dims = selected + [candidate]
            result = optimize_subset(
                active_dims=trial_dims,
                all_dim_names=dim_names,
                evaluate_fn=evaluate_fn,
                inactive_value=inactive_value,
                n_grid=n_grid,
                n_random=n_random,
            )
            if result.mae < best_mae:
                best_mae = result.mae
                best_dim = candidate
                best_result = result

        selected.append(best_dim)
        remaining.remove(best_dim)
        results.append(best_result)  # type: ignore[arg-type]

    return results


def backward_elimination(
    dim_names: Sequence[str],
    evaluate_fn: Callable[[np.ndarray], tuple[float, dict[str, float]]],
    inactive_value: float = 1e6,
    n_grid: int = 20,
    n_random: int = 5000,
) -> list[SubsetResult]:
    """Backward elimination: start with all dims, greedily remove least important.

    Parameters
    ----------
    dim_names : sequence of str
        Names for all dimensions.
    evaluate_fn : callable
        Function (params: ndarray) -> (mae: float, errors: dict).
    inactive_value : float
        Value for inactive dimensions.
    n_grid : int
        Grid points for optimization.
    n_random : int
        Random samples for 3D+ optimization.

    Returns
    -------
    list[SubsetResult]
        Results at each step, from all dims down to 1 dim.
    """
    n_all = len(dim_names)
    active = list(range(n_all))
    results: list[SubsetResult] = []

    # Start with all dimensions
    full_result = optimize_subset(
        active_dims=active,
        all_dim_names=dim_names,
        evaluate_fn=evaluate_fn,
        inactive_value=inactive_value,
        n_grid=n_grid,
        n_random=n_random,
    )
    results.append(full_result)

    while len(active) > 1:
        best_mae = float("inf")
        worst_dim = active[0]
        best_result = None

        for candidate in active:
            trial_dims = [d for d in active if d != candidate]
            result = optimize_subset(
                active_dims=trial_dims,
                all_dim_names=dim_names,
                evaluate_fn=evaluate_fn,
                inactive_value=inactive_value,
                n_grid=n_grid,
                n_random=n_random,
            )
            if result.mae < best_mae:
                best_mae = result.mae
                worst_dim = candidate
                best_result = result

        active.remove(worst_dim)
        results.append(best_result)  # type: ignore[arg-type]

    return results


def lasso_selection(
    dim_names: Sequence[str],
    evaluate_fn: Callable[[np.ndarray], tuple[float, dict[str, float]]],
    alphas: Sequence[float] | None = None,
    inactive_value: float = 1e6,
    n_random: int = 5000,
) -> list[SubsetResult]:
    """L1-penalized parameter optimization.

    For each regularization strength alpha, optimize parameters with an
    L1 penalty on log(params), effectively encouraging sparsity.

    Parameters
    ----------
    dim_names : sequence of str
        Names for all dimensions.
    evaluate_fn : callable
        Function (params: ndarray) -> (mae: float, errors: dict).
    alphas : sequence of float or None
        Regularization strengths to try. None uses defaults.
    inactive_value : float
        Value for effectively inactive dimensions.
    n_random : int
        Number of random samples for search.

    Returns
    -------
    list[SubsetResult]
        Best result at each sparsity level found.
    """
    if alphas is None:
        alphas = np.logspace(-3, 2, 20).tolist()

    n_all = len(dim_names)
    rng = np.random.default_rng(42)
    log_low, log_high = np.log10(0.01), np.log10(100)

    seen_sparsities: dict[int, SubsetResult] = {}

    for alpha in alphas:
        best_penalized_mae = float("inf")
        best_params = None
        best_raw_mae = float("inf")
        best_errors: dict[str, float] = {}

        for _ in range(n_random):
            log_vals = rng.uniform(log_low, log_high, n_all)
            params = 10**log_vals

            raw_mae, errors = evaluate_fn(params)

            # L1 penalty on log(params) -- penalizes deviation from 1.0
            l1_penalty = alpha * np.sum(np.abs(np.log10(params)))
            penalized = raw_mae + l1_penalty

            if penalized < best_penalized_mae:
                best_penalized_mae = penalized
                best_params = params.copy()
                best_raw_mae = raw_mae
                best_errors = errors.copy()

        # Determine active dimensions (those not near 1.0 in log-space)
        # Dimensions with |log10(param)| < 0.5 are considered "inactive" (near 1.0)
        if best_params is not None:
            active_dims = tuple(i for i in range(n_all) if abs(np.log10(best_params[i])) >= 0.5)
            if not active_dims:
                active_dims = tuple(range(n_all))

            n_active = len(active_dims)
            dim_name_tuple = tuple(dim_names[d] for d in active_dims)

            result = SubsetResult(
                dims=active_dims,
                dim_names=dim_name_tuple,
                n_dims=n_active,
                param_values=best_params,
                mae=best_raw_mae,
                errors=best_errors,
            )

            if n_active not in seen_sparsities or result.mae < seen_sparsities[n_active].mae:
                seen_sparsities[n_active] = result

    results = sorted(seen_sparsities.values(), key=lambda r: r.n_dims)
    return results
