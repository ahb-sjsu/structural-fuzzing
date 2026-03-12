"""Adversarial threshold search for parameter sensitivity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass
class AdversarialResult:
    """Result of adversarial threshold search for one dimension and direction."""

    dim: int
    dim_name: str
    base_value: float
    threshold_value: float
    threshold_ratio: float
    target_flipped: str
    direction: str  # "increase" or "decrease"


def find_adversarial_threshold(
    params: np.ndarray,
    dim: int,
    dim_names: Sequence[str],
    evaluate_fn: Callable[[np.ndarray], tuple[float, dict[str, float]]],
    tolerance: float = 0.5,
    n_steps: int = 50,
) -> list[AdversarialResult]:
    """Find adversarial thresholds for a single dimension.

    For both increase and decrease directions, performs a log-spaced search
    to find the first perturbation where any target's error changes by more
    than the tolerance.

    Parameters
    ----------
    params : np.ndarray
        Baseline parameter values.
    dim : int
        Index of the dimension to perturb.
    dim_names : sequence of str
        Names for all dimensions.
    evaluate_fn : callable
        Function (params: ndarray) -> (mae: float, errors: dict).
    tolerance : float
        Maximum allowable change in any target error before flagging.
    n_steps : int
        Number of search steps per direction.

    Returns
    -------
    list[AdversarialResult]
        One result per direction where a threshold was found (0-2 results).
    """
    base_value = params[dim]
    _, base_errors = evaluate_fn(params)

    results: list[AdversarialResult] = []

    # Ensure base_value is positive for log-space search
    if base_value <= 0:
        base_value = 0.01

    for direction in ("increase", "decrease"):
        if direction == "increase":
            # Search from base_value to base_value * 1000
            search_values = np.logspace(
                np.log10(base_value),
                np.log10(base_value * 1000),
                n_steps,
            )
        else:
            # Search from base_value to base_value / 1000
            low = max(base_value / 1000, 1e-6)
            search_values = np.logspace(
                np.log10(base_value),
                np.log10(low),
                n_steps,
            )

        for test_value in search_values[1:]:  # skip the first (baseline)
            perturbed = params.copy()
            perturbed[dim] = test_value
            _, pert_errors = evaluate_fn(perturbed)

            # Check if any target changed by more than tolerance
            for target_name in base_errors:
                if target_name in pert_errors:
                    delta = abs(pert_errors[target_name] - base_errors[target_name])
                    if delta > tolerance:
                        ratio = test_value / base_value
                        results.append(
                            AdversarialResult(
                                dim=dim,
                                dim_name=dim_names[dim],
                                base_value=base_value,
                                threshold_value=test_value,
                                threshold_ratio=ratio,
                                target_flipped=target_name,
                                direction=direction,
                            )
                        )
                        break  # found threshold for this direction
            else:
                continue
            break  # move to next direction

    return results
