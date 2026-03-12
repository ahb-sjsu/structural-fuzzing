"""Model Robustness Index (MRI) computation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class ModelRobustnessIndex:
    """Result of MRI computation."""

    mri: float
    mean_omega: float
    p75_omega: float
    p95_omega: float
    n_perturbations: int
    worst_case_mae: float
    perturbation_errors: list[float] = field(repr=False)


def compute_mri(
    params: np.ndarray,
    evaluate_fn: Callable[[np.ndarray], tuple[float, dict[str, float]]],
    n_perturbations: int = 300,
    scale: float = 0.5,
    weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
    rng: np.random.Generator | None = None,
) -> ModelRobustnessIndex:
    """Compute the Model Robustness Index.

    Perturbs parameters in log-space and measures the deviation in MAE
    from the baseline.

    MRI = w0 * mean(omega) + w1 * P75(omega) + w2 * P95(omega)

    where omega_i = |mae_perturbed_i - mae_base|.

    Lower MRI indicates a more robust model.

    Parameters
    ----------
    params : np.ndarray
        Baseline parameter values.
    evaluate_fn : callable
        Function (params: ndarray) -> (mae: float, errors: dict).
    n_perturbations : int
        Number of perturbation samples.
    scale : float
        Standard deviation in log-space for perturbations.
    weights : tuple of three floats
        Weights for (mean, P75, P95) in MRI formula.
    rng : numpy Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    ModelRobustnessIndex
        The computed MRI and supporting statistics.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    base_mae, _ = evaluate_fn(params)

    omegas: list[float] = []
    worst_mae = base_mae

    for _ in range(n_perturbations):
        # Perturb in log-space: params_pert = params * exp(N(0, scale^2))
        noise = rng.normal(0.0, scale, size=params.shape)
        params_pert = params * np.exp(noise)
        # Clamp to [0.001, 1e6]
        params_pert = np.clip(params_pert, 0.001, 1e6)

        pert_mae, _ = evaluate_fn(params_pert)
        omega = abs(pert_mae - base_mae)
        omegas.append(omega)

        if pert_mae > worst_mae:
            worst_mae = pert_mae

    omegas_arr = np.array(omegas)
    mean_omega = float(np.mean(omegas_arr))
    p75_omega = float(np.percentile(omegas_arr, 75))
    p95_omega = float(np.percentile(omegas_arr, 95))

    mri_value = weights[0] * mean_omega + weights[1] * p75_omega + weights[2] * p95_omega

    return ModelRobustnessIndex(
        mri=mri_value,
        mean_omega=mean_omega,
        p75_omega=p75_omega,
        p95_omega=p95_omega,
        n_perturbations=n_perturbations,
        worst_case_mae=worst_mae,
        perturbation_errors=omegas,
    )
