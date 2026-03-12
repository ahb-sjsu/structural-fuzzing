"""Run structural fuzzing campaign on the defect prediction model."""

from structural_fuzzing import run_campaign
from examples.defect_prediction.model import GROUP_NAMES, make_evaluate_fn


def main():
    evaluate_fn = make_evaluate_fn(n_samples=500, seed=42)
    report = run_campaign(
        dim_names=GROUP_NAMES,
        evaluate_fn=evaluate_fn,
        max_subset_dims=5,
        n_mri_perturbations=100,
        start_dim=1,  # Start from Complexity
        verbose=True,
    )
    print(report.summary())


if __name__ == "__main__":
    main()
