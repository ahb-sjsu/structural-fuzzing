"""Core framework: SubsetResult, enumerate_subsets, optimize_subset."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass
class SubsetResult:
    """Result of optimizing a single parameter subset."""

    dims: tuple[int, ...]
    dim_names: tuple[str, ...]
    n_dims: int
    param_values: np.ndarray
    mae: float
    errors: dict[str, float]
    pareto_optimal: bool = False

    def __repr__(self) -> str:
        names = ", ".join(self.dim_names)
        return f"SubsetResult(dims=[{names}], n_dims={self.n_dims}, mae={self.mae:.4f})"


def optimize_subset(
    active_dims: Sequence[int],
    all_dim_names: Sequence[str],
    evaluate_fn: Callable[[np.ndarray], tuple[float, dict[str, float]]],
    inactive_value: float = 1e6,
    n_grid: int = 20,
    n_random: int = 5000,
) -> SubsetResult:
    """Optimize parameter values for a subset of active dimensions.

    For 1D subsets: grid search over n_grid log-spaced values in [0.01, 100].
    For 2D subsets: full grid search of n_grid^2 points.
    For 3D+ subsets: random search with n_random samples in log-space.

    Parameters
    ----------
    active_dims : sequence of int
        Indices of active dimensions.
    all_dim_names : sequence of str
        Names for all dimensions.
    evaluate_fn : callable
        Function (params: ndarray) -> (mae: float, errors: dict).
    inactive_value : float
        Value assigned to inactive dimensions (default 1e6).
    n_grid : int
        Number of grid points per dimension for 1D/2D search.
    n_random : int
        Number of random samples for 3D+ search.

    Returns
    -------
    SubsetResult
        The best parameter configuration found.
    """
    n_all = len(all_dim_names)
    active_dims = tuple(active_dims)
    n_active = len(active_dims)

    if n_active == 0:
        params = np.full(n_all, inactive_value)
        mae, errors = evaluate_fn(params)
        return SubsetResult(
            dims=(),
            dim_names=(),
            n_dims=0,
            param_values=params.copy(),
            mae=mae,
            errors=errors,
        )

    grid_values = np.logspace(np.log10(0.01), np.log10(100), n_grid)

    best_mae = float("inf")
    best_params = None
    best_errors: dict[str, float] = {}

    if n_active == 1:
        # 1D grid search
        for v in grid_values:
            params = np.full(n_all, inactive_value)
            params[active_dims[0]] = v
            mae, errors = evaluate_fn(params)
            if mae < best_mae:
                best_mae = mae
                best_params = params.copy()
                best_errors = errors.copy()

    elif n_active == 2:
        # 2D full grid search
        for v0, v1 in itertools.product(grid_values, grid_values):
            params = np.full(n_all, inactive_value)
            params[active_dims[0]] = v0
            params[active_dims[1]] = v1
            mae, errors = evaluate_fn(params)
            if mae < best_mae:
                best_mae = mae
                best_params = params.copy()
                best_errors = errors.copy()

    else:
        # 3D+ random search in log-space
        rng = np.random.default_rng(42)
        log_low, log_high = np.log10(0.01), np.log10(100)
        for _ in range(n_random):
            params = np.full(n_all, inactive_value)
            log_vals = rng.uniform(log_low, log_high, n_active)
            for i, dim in enumerate(active_dims):
                params[dim] = 10 ** log_vals[i]
            mae, errors = evaluate_fn(params)
            if mae < best_mae:
                best_mae = mae
                best_params = params.copy()
                best_errors = errors.copy()

    dim_names = tuple(all_dim_names[d] for d in active_dims)

    return SubsetResult(
        dims=active_dims,
        dim_names=dim_names,
        n_dims=n_active,
        param_values=best_params,  # type: ignore[arg-type]
        mae=best_mae,
        errors=best_errors,
    )


def enumerate_subsets(
    dim_names: Sequence[str],
    evaluate_fn: Callable[[np.ndarray], tuple[float, dict[str, float]]],
    max_dims: int = 4,
    inactive_value: float = 1e6,
    n_grid: int = 20,
    n_random: int = 5000,
    verbose: bool = False,
) -> list[SubsetResult]:
    """Enumerate all parameter subsets up to max_dims and optimize each.

    Parameters
    ----------
    dim_names : sequence of str
        Names for all dimensions.
    evaluate_fn : callable
        Function (params: ndarray) -> (mae: float, errors: dict).
    max_dims : int
        Maximum subset size to enumerate.
    inactive_value : float
        Value assigned to inactive dimensions.
    n_grid : int
        Grid points per dimension for 1D/2D search.
    n_random : int
        Random samples for 3D+ search.
    verbose : bool
        Print progress information.

    Returns
    -------
    list[SubsetResult]
        All results, sorted by MAE ascending.
    """
    n_all = len(dim_names)
    results: list[SubsetResult] = []

    for k in range(1, min(max_dims, n_all) + 1):
        combos = list(itertools.combinations(range(n_all), k))
        if verbose:
            print(f"  Enumerating {len(combos)} subsets of size {k}...")
        for combo in combos:
            result = optimize_subset(
                active_dims=combo,
                all_dim_names=dim_names,
                evaluate_fn=evaluate_fn,
                inactive_value=inactive_value,
                n_grid=n_grid,
                n_random=n_random,
            )
            results.append(result)

    results.sort(key=lambda r: r.mae)
    return results
