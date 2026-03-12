"""Pareto frontier extraction for subset results."""

from __future__ import annotations

from structural_fuzzing.core import SubsetResult


def pareto_frontier(
    results: list[SubsetResult],
    tolerance: float = 0.01,
) -> list[SubsetResult]:
    """Mark and return Pareto-optimal results.

    A result is Pareto-optimal if no other result has both fewer dimensions
    AND lower MAE (within tolerance). In practice, we find the best MAE at
    each dimensionality level and keep those that form a non-dominated front.

    Parameters
    ----------
    results : list[SubsetResult]
        Subset results to analyze.
    tolerance : float
        Tolerance for MAE comparison. Two MAE values within tolerance are
        considered equivalent.

    Returns
    -------
    list[SubsetResult]
        Pareto-optimal results, sorted by n_dims ascending.
    """
    if not results:
        return []

    # Reset all pareto flags
    for r in results:
        r.pareto_optimal = False

    # Find best MAE at each dimensionality
    best_at_k: dict[int, SubsetResult] = {}
    for r in results:
        k = r.n_dims
        if k not in best_at_k or r.mae < best_at_k[k].mae:
            best_at_k[k] = r

    # Extract candidates sorted by n_dims
    candidates = sorted(best_at_k.values(), key=lambda r: r.n_dims)

    # Filter to Pareto front: a candidate is dominated if another candidate
    # has fewer dims AND better-or-equal MAE
    pareto: list[SubsetResult] = []
    best_mae_so_far = float("inf")

    for candidate in candidates:
        # A candidate is Pareto-optimal if its MAE is better than
        # the best MAE seen at any lower dimensionality (within tolerance)
        if candidate.mae < best_mae_so_far - tolerance:
            candidate.pareto_optimal = True
            pareto.append(candidate)
            best_mae_so_far = candidate.mae
        elif not pareto:
            # Always include the first (lowest-dim) candidate
            candidate.pareto_optimal = True
            pareto.append(candidate)
            best_mae_so_far = candidate.mae

    return pareto
