"""Structural fuzzing framework for parameterized model validation.

Adapts the adversarial mindset of software fuzzing to model validation:
instead of mutating program inputs to find crashes, we mutate model
parameters to find prediction failures.
"""

from structural_fuzzing.adversarial import AdversarialResult, find_adversarial_threshold
from structural_fuzzing.baselines import backward_elimination, forward_selection, lasso_selection
from structural_fuzzing.compositional import CompositionResult, compositional_test
from structural_fuzzing.core import (
    SubsetResult,
    enumerate_subsets,
    optimize_subset,
)
from structural_fuzzing.mri import ModelRobustnessIndex, compute_mri
from structural_fuzzing.pareto import pareto_frontier
from structural_fuzzing.pipeline import StructuralFuzzReport, run_campaign
from structural_fuzzing.sensitivity import SensitivityResult, sensitivity_profile

__version__ = "0.2.0"

__all__ = [
    "SubsetResult",
    "enumerate_subsets",
    "optimize_subset",
    "pareto_frontier",
    "SensitivityResult",
    "sensitivity_profile",
    "ModelRobustnessIndex",
    "compute_mri",
    "AdversarialResult",
    "find_adversarial_threshold",
    "CompositionResult",
    "compositional_test",
    "StructuralFuzzReport",
    "run_campaign",
    "forward_selection",
    "backward_elimination",
    "lasso_selection",
]
