"""Full pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from structural_fuzzing.adversarial import AdversarialResult, find_adversarial_threshold
from structural_fuzzing.baselines import backward_elimination, forward_selection
from structural_fuzzing.compositional import CompositionResult, compositional_test
from structural_fuzzing.core import SubsetResult, enumerate_subsets
from structural_fuzzing.mri import ModelRobustnessIndex, compute_mri
from structural_fuzzing.pareto import pareto_frontier
from structural_fuzzing.sensitivity import SensitivityResult, sensitivity_profile


@dataclass
class StructuralFuzzReport:
    """Complete structural fuzzing campaign report."""

    dim_names: list[str]
    subset_results: list[SubsetResult]
    pareto_results: list[SubsetResult]
    sensitivity_results: list[SensitivityResult]
    mri_result: ModelRobustnessIndex | None
    adversarial_results: list[AdversarialResult]
    composition_result: CompositionResult | None
    forward_results: list[SubsetResult] = field(default_factory=list)
    backward_results: list[SubsetResult] = field(default_factory=list)

    def summary(self) -> str:
        """Generate a text summary of the campaign results."""
        from structural_fuzzing.report import format_report

        return format_report(self)


def run_campaign(
    dim_names: Sequence[str],
    evaluate_fn: Callable[[np.ndarray], tuple[float, dict[str, float]]],
    max_subset_dims: int = 4,
    n_mri_perturbations: int = 300,
    mri_scale: float = 0.5,
    mri_weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
    start_dim: int = 0,
    candidate_dims: Sequence[int] | None = None,
    run_baselines: bool = True,
    adversarial_tolerance: float = 0.5,
    inactive_value: float = 1e6,
    n_grid: int = 20,
    n_random: int = 5000,
    verbose: bool = True,
) -> StructuralFuzzReport:
    """Run a complete structural fuzzing campaign.

    Parameters
    ----------
    dim_names : sequence of str
        Names for all dimensions/parameters.
    evaluate_fn : callable
        Function (params: ndarray) -> (mae: float, errors: dict).
    max_subset_dims : int
        Maximum subset size for enumeration.
    n_mri_perturbations : int
        Number of perturbation samples for MRI.
    mri_scale : float
        Log-space perturbation scale for MRI.
    mri_weights : tuple
        Weights for (mean, P75, P95) in MRI formula.
    start_dim : int
        Starting dimension for compositional test.
    candidate_dims : sequence of int or None
        Candidate dimensions for compositional test. None means all except start.
    run_baselines : bool
        Whether to run forward/backward selection baselines.
    adversarial_tolerance : float
        Tolerance for adversarial threshold search.
    inactive_value : float
        Value for inactive dimensions.
    n_grid : int
        Grid points for optimization.
    n_random : int
        Random samples for 3D+ optimization.
    verbose : bool
        Print progress information.

    Returns
    -------
    StructuralFuzzReport
        Complete campaign results.
    """
    dim_names_list = list(dim_names)
    n_dims = len(dim_names_list)

    # Step 1: Enumerate subsets
    if verbose:
        print("=" * 60)
        print("STRUCTURAL FUZZING CAMPAIGN")
        print("=" * 60)
        print(f"\nDimensions: {n_dims}")
        print(f"Max subset size: {max_subset_dims}")
        print("\n[1/6] Enumerating subsets...")

    subset_results = enumerate_subsets(
        dim_names=dim_names_list,
        evaluate_fn=evaluate_fn,
        max_dims=max_subset_dims,
        inactive_value=inactive_value,
        n_grid=n_grid,
        n_random=n_random,
        verbose=verbose,
    )

    if verbose:
        print(f"  Found {len(subset_results)} configurations")
        if subset_results:
            print(f"  Best MAE: {subset_results[0].mae:.4f} ({subset_results[0]})")

    # Step 2: Pareto frontier
    if verbose:
        print("\n[2/6] Extracting Pareto frontier...")
    pareto_results = pareto_frontier(subset_results)
    if verbose:
        print(f"  {len(pareto_results)} Pareto-optimal configurations")

    # Step 3: Sensitivity profiling (use best result's params)
    if verbose:
        print("\n[3/6] Sensitivity profiling...")
    if subset_results:
        best_params = subset_results[0].param_values
    else:
        best_params = np.ones(n_dims)

    sensitivity_results = sensitivity_profile(
        params=best_params,
        dim_names=dim_names_list,
        evaluate_fn=evaluate_fn,
        inactive_value=inactive_value,
    )
    if verbose:
        print("  Importance ranking:")
        for sr in sensitivity_results[:5]:
            print(f"    {sr.importance_rank}. {sr.dim_name} (delta={sr.delta_mae:.4f})")

    # Step 4: MRI
    if verbose:
        print("\n[4/6] Computing Model Robustness Index...")
    mri_result = compute_mri(
        params=best_params,
        evaluate_fn=evaluate_fn,
        n_perturbations=n_mri_perturbations,
        scale=mri_scale,
        weights=mri_weights,
    )
    if verbose:
        print(f"  MRI = {mri_result.mri:.4f}")
        print(f"  Worst-case MAE = {mri_result.worst_case_mae:.4f}")

    # Step 5: Adversarial threshold search
    if verbose:
        print("\n[5/6] Adversarial threshold search...")
    adversarial_results: list[AdversarialResult] = []
    for i in range(n_dims):
        adv = find_adversarial_threshold(
            params=best_params,
            dim=i,
            dim_names=dim_names_list,
            evaluate_fn=evaluate_fn,
            tolerance=adversarial_tolerance,
        )
        adversarial_results.extend(adv)
    if verbose:
        print(f"  Found {len(adversarial_results)} adversarial thresholds")

    # Step 6: Compositional test
    if verbose:
        print("\n[6/6] Compositional testing...")
    if candidate_dims is None:
        candidate_dims_list = [i for i in range(n_dims) if i != start_dim]
    else:
        candidate_dims_list = list(candidate_dims)

    composition_result = compositional_test(
        start_dim=start_dim,
        candidate_dims=candidate_dims_list,
        dim_names=dim_names_list,
        evaluate_fn=evaluate_fn,
        inactive_value=inactive_value,
        n_grid=n_grid,
        n_random=n_random,
    )
    if verbose:
        print(f"  Build order: {' -> '.join(composition_result.order_names)}")
        print(f"  MAE sequence: {[f'{m:.4f}' for m in composition_result.mae_sequence]}")

    # Optional baselines
    forward_results: list[SubsetResult] = []
    backward_results: list[SubsetResult] = []
    if run_baselines:
        if verbose:
            print("\n[Baselines] Forward selection...")
        forward_results = forward_selection(
            dim_names=dim_names_list,
            evaluate_fn=evaluate_fn,
            max_dims=max_subset_dims,
            inactive_value=inactive_value,
            n_grid=n_grid,
            n_random=n_random,
        )
        if verbose:
            print(f"  {len(forward_results)} steps")

        if verbose:
            print("[Baselines] Backward elimination...")
        backward_results = backward_elimination(
            dim_names=dim_names_list,
            evaluate_fn=evaluate_fn,
            inactive_value=inactive_value,
            n_grid=n_grid,
            n_random=n_random,
        )
        if verbose:
            print(f"  {len(backward_results)} steps")

    if verbose:
        print("\n" + "=" * 60)
        print("CAMPAIGN COMPLETE")
        print("=" * 60)

    return StructuralFuzzReport(
        dim_names=dim_names_list,
        subset_results=subset_results,
        pareto_results=pareto_results,
        sensitivity_results=sensitivity_results,
        mri_result=mri_result,
        adversarial_results=adversarial_results,
        composition_result=composition_result,
        forward_results=forward_results,
        backward_results=backward_results,
    )
