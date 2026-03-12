"""Run structural fuzzing campaign on the geometric economics model."""

from structural_fuzzing import run_campaign
from examples.geometric_economics.model import DIM_NAMES
from examples.geometric_economics.targets import make_evaluate_fn


def main():
    evaluate_fn = make_evaluate_fn()
    report = run_campaign(
        dim_names=DIM_NAMES,
        evaluate_fn=evaluate_fn,
        max_subset_dims=5,
        n_mri_perturbations=300,
        start_dim=0,  # Start from Consequences
        verbose=True,
    )
    print(report.summary())


if __name__ == "__main__":
    main()
