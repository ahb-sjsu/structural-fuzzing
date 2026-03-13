# structural-fuzzing

[![CI](https://github.com/ahb-sjsu/structural-fuzzing/actions/workflows/ci.yaml/badge.svg)](https://github.com/ahb-sjsu/structural-fuzzing/actions/workflows/ci.yaml)
[![PyPI](https://img.shields.io/pypi/v/structural-fuzzing)](https://pypi.org/project/structural-fuzzing/)
[![Python](https://img.shields.io/pypi/pyversions/structural-fuzzing)](https://pypi.org/project/structural-fuzzing/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/ahb-sjsu/structural-fuzzing/blob/main/LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Structural fuzzing framework for parameterized model validation.

Adapts the adversarial mindset of software fuzzing to model validation:
instead of mutating program inputs to find crashes, we mutate model
parameters to find prediction failures.

**Works with any model that takes parameters and produces predictions** --
scikit-learn classifiers, neural networks, simulation models, economic models,
or any custom function.

## Installation

```bash
pip install structural-fuzzing
```

For development (includes testing and linting tools):

```bash
pip install structural-fuzzing[dev]
```

To run the included examples (requires scikit-learn):

```bash
pip install structural-fuzzing[examples]
```

## Core Concepts

### The evaluate function

Every analysis in structural-fuzzing revolves around a single user-provided function:

```python
def evaluate_fn(params: np.ndarray) -> tuple[float, dict[str, float]]:
    """
    Args:
        params: 1D array with one value per dimension. Values >= 1e6
                are treated as "inactive" (that dimension is turned off).

    Returns:
        mae: Mean absolute error (scalar summary of how well the model performs).
        errors: Dict mapping target names to signed errors (predicted - expected).
    """
```

The framework explores the parameter space by calling this function with different
configurations and analyzing the results.

### Dimensions

Parameters are organized into named **dimensions** -- logical groups like feature
families, risk factors, or model components. Structural fuzzing explores which
combinations of dimensions matter, not just individual parameter sensitivity.

### Inactive dimensions

When a dimension's parameter value is set to `1e6` (the default `inactive_value`),
it signals that the dimension is "turned off." Your `evaluate_fn` should handle
this by excluding that dimension from computation. This allows the framework to
test subsets of your parameter space.

## Quick Start

### Minimal example

```python
import numpy as np
from structural_fuzzing import run_campaign

def evaluate_fn(params):
    # A simple model: prediction quality depends on params[0] and params[1],
    # but params[2] is irrelevant noise.
    errors = {}

    if params[0] < 1e5:  # "feature_a" active
        errors["target_1"] = abs(params[0] - 1.0) * 2
    else:
        errors["target_1"] = 10.0

    if params[1] < 1e5:  # "feature_b" active
        errors["target_2"] = abs(params[1] - 0.5) * 3
    else:
        errors["target_2"] = 8.0

    # params[2] ("noise_dim") has no effect on quality
    errors["target_3"] = 1.0

    mae = sum(abs(v) for v in errors.values()) / len(errors)
    return mae, errors

report = run_campaign(
    dim_names=["feature_a", "feature_b", "noise_dim"],
    evaluate_fn=evaluate_fn,
    verbose=True,
)
print(report.summary())
```

This will run all six analyses and print a complete report showing that
`feature_a` and `feature_b` carry all the signal while `noise_dim` is irrelevant.

## What the Campaign Runs

`run_campaign()` executes six analyses in sequence, plus optional baselines:

| Step | Analysis | What it reveals |
|------|----------|-----------------|
| 1 | **Subset Enumeration** | Tests all dimension combinations up to `max_subset_dims`. Finds which groups of parameters perform best. |
| 2 | **Pareto Frontier** | Identifies configurations that are optimal tradeoffs between accuracy (lower MAE) and simplicity (fewer dimensions). |
| 3 | **Sensitivity Profiling** | Ablates each dimension independently to rank importance. |
| 4 | **Model Robustness Index** | Quantifies stability under random parameter perturbation, including tail behavior (P75, P95). |
| 5 | **Adversarial Threshold Search** | Finds exact parameter values where model behavior flips. |
| 6 | **Compositional Testing** | Greedily builds up dimensions one at a time to find the optimal construction order. |

Optional baselines (forward selection, backward elimination) run when `run_baselines=True`.

## Using Individual Components

You don't have to run the full campaign. Each analysis is available as a
standalone function.

### Subset Enumeration

Test all combinations of dimensions up to a given size:

```python
from structural_fuzzing import enumerate_subsets

results = enumerate_subsets(
    dim_names=["size", "complexity", "halstead", "oo", "process"],
    evaluate_fn=evaluate_fn,
    max_dims=3,       # test all 1D, 2D, and 3D combinations
    n_grid=20,        # grid points for 1D/2D search
    n_random=5000,    # random samples for 3D+ search
    verbose=True,
)

# Results are sorted by MAE (best first)
for r in results[:5]:
    print(f"  dims={r.dim_names}, MAE={r.mae:.4f}")
```

**How optimization works internally:**
- **1D subsets**: Grid search over 20 log-spaced values in [0.01, 100]
- **2D subsets**: Full 2D grid (20 x 20 = 400 evaluations)
- **3D+ subsets**: Random search with 5,000 log-space samples

### Pareto Frontier

Find configurations that are not dominated on both accuracy and complexity:

```python
from structural_fuzzing import enumerate_subsets, pareto_frontier

results = enumerate_subsets(dim_names, evaluate_fn, max_dims=4)
pareto = pareto_frontier(results)

for p in pareto:
    print(f"  k={p.n_dims}: MAE={p.mae:.4f} [{', '.join(p.dim_names)}]")
```

A result is Pareto-optimal if no other result has **both** fewer dimensions
**and** lower MAE. This tells you where adding complexity stops paying off.

### Sensitivity Profiling

Rank dimensions by importance via ablation:

```python
from structural_fuzzing import sensitivity_profile

results = sensitivity_profile(
    params=best_params,        # baseline parameter values (1D array)
    dim_names=["size", "complexity", "halstead", "oo", "process"],
    evaluate_fn=evaluate_fn,
)

for r in results:
    print(f"  {r.importance_rank}. {r.dim_name}: "
          f"delta_MAE={r.delta_mae:+.4f} "
          f"(with={r.mae_with:.4f}, without={r.mae_without:.4f})")
```

Each dimension is set to `inactive_value` one at a time. The resulting MAE
increase (`delta_mae`) measures how much the model depends on that dimension.
Higher `delta_mae` = more important.

### Model Robustness Index (MRI)

Quantify how stable your model is under parameter perturbation:

```python
from structural_fuzzing import compute_mri

mri = compute_mri(
    params=best_params,
    evaluate_fn=evaluate_fn,
    n_perturbations=300,   # number of random perturbations
    scale=0.5,             # log-space perturbation magnitude
    weights=(0.5, 0.3, 0.2),  # weights for (mean, P75, P95)
)

print(f"MRI = {mri.mri:.4f}")           # lower = more robust
print(f"Mean deviation = {mri.mean_omega:.4f}")
print(f"P75 deviation = {mri.p75_omega:.4f}")
print(f"P95 deviation = {mri.p95_omega:.4f}")
print(f"Worst-case MAE = {mri.worst_case_mae:.4f}")
```

**How it works:** Each parameter is perturbed as `params * exp(N(0, scale^2))`,
clamped to [0.001, 1e6]. The MRI is a weighted combination of mean, 75th
percentile, and 95th percentile MAE deviations from baseline.

**Lower MRI = more robust.** The tail weights (P75, P95) capture worst-case
behavior that mean alone would miss.

### Adversarial Threshold Search

Find the exact parameter values where your model breaks:

```python
from structural_fuzzing import find_adversarial_threshold

# Search one dimension at a time
thresholds = find_adversarial_threshold(
    params=best_params,
    dim=0,                                  # which dimension to perturb
    dim_names=["size", "complexity", "halstead", "oo", "process"],
    evaluate_fn=evaluate_fn,
    tolerance=0.5,                          # max acceptable error change
    n_steps=50,                             # search resolution
)

for t in thresholds:
    print(f"  {t.dim_name} ({t.direction}): "
          f"{t.base_value:.4f} -> {t.threshold_value:.4f} "
          f"({t.threshold_ratio:.1f}x), flips '{t.target_flipped}'")
```

For each dimension, the search goes in both directions (increase and decrease)
using log-spaced steps from the baseline to baseline * 1000 (or / 1000). It
reports the first value where any target's error exceeds the tolerance.

Returns 0-2 results per dimension (one per direction where a threshold was found).

### Compositional Testing

Find the optimal order to build up your model, one dimension at a time:

```python
from structural_fuzzing import compositional_test

result = compositional_test(
    start_dim=1,                   # index of the starting dimension
    candidate_dims=[0, 2, 3, 4],   # remaining dimensions to try
    dim_names=["size", "complexity", "halstead", "oo", "process"],
    evaluate_fn=evaluate_fn,
)

print(f"Build order: {' -> '.join(result.order_names)}")
for i, (name, mae) in enumerate(zip(result.order_names, result.mae_sequence)):
    print(f"  Step {i+1}: +{name} => MAE={mae:.4f}")
```

At each step, the framework tries adding each remaining dimension and picks
the one that reduces MAE the most. This reveals:
- Which dimension to start with
- The order of diminishing returns
- When adding more dimensions stops helping

### Baseline Comparisons

Compare against standard feature selection methods:

```python
from structural_fuzzing import forward_selection, backward_elimination

# Forward: start empty, greedily add best dimension
fwd = forward_selection(dim_names, evaluate_fn, max_dims=4)
for r in fwd:
    print(f"  +{r.dim_names[-1]} => k={r.n_dims}, MAE={r.mae:.4f}")

# Backward: start with all, greedily remove worst dimension
bwd = backward_elimination(dim_names, evaluate_fn)
for r in bwd:
    print(f"  k={r.n_dims}: MAE={r.mae:.4f} [{', '.join(r.dim_names)}]")
```

### L1-Penalized (LASSO) Selection

Encourages sparsity through an L1 penalty on log-space parameter values:

```python
from structural_fuzzing import lasso_selection

results = lasso_selection(
    dim_names=dim_names,
    evaluate_fn=evaluate_fn,
    alphas=None,       # uses default log-spaced range [1e-3, 100]
    n_random=5000,
)

for r in results:
    print(f"  k={r.n_dims}: MAE={r.mae:.4f} [{', '.join(r.dim_names)}]")
```

## Configuring `run_campaign()`

All parameters with their defaults:

```python
report = run_campaign(
    # Required
    dim_names=["dim_a", "dim_b", "dim_c"],
    evaluate_fn=evaluate_fn,

    # Subset enumeration
    max_subset_dims=4,       # max combination size (higher = slower but more thorough)
    n_grid=20,               # grid points per dim for 1D/2D optimization
    n_random=5000,           # random samples for 3D+ optimization
    inactive_value=1e6,      # value that marks a dimension as "off"

    # MRI
    n_mri_perturbations=300, # more = smoother MRI estimate
    mri_scale=0.5,           # perturbation magnitude (0.5 = moderate)
    mri_weights=(0.5, 0.3, 0.2),  # emphasis on (mean, P75, P95)

    # Compositional test
    start_dim=0,             # which dimension to start building from
    candidate_dims=None,     # None = all except start_dim

    # Adversarial search
    adversarial_tolerance=0.5,  # error change threshold for "breaking"

    # Baselines
    run_baselines=True,      # run forward/backward selection

    # Output
    verbose=True,            # print progress
)
```

### Tuning for speed vs. thoroughness

**Fast exploration** (good for initial investigation):
```python
report = run_campaign(
    dim_names=dim_names,
    evaluate_fn=evaluate_fn,
    max_subset_dims=2,        # only test 1D and 2D combos
    n_mri_perturbations=50,   # fewer perturbations
    run_baselines=False,      # skip baselines
)
```

**Thorough analysis** (good for final validation):
```python
report = run_campaign(
    dim_names=dim_names,
    evaluate_fn=evaluate_fn,
    max_subset_dims=5,        # test up to 5D combos
    n_grid=30,                # finer grid
    n_random=10000,           # more random samples
    n_mri_perturbations=1000, # smoother MRI
)
```

## Working with Results

### The StructuralFuzzReport object

`run_campaign()` returns a `StructuralFuzzReport` with these fields:

```python
report.dim_names              # list[str] -- dimension names
report.subset_results         # list[SubsetResult] -- all configs, sorted by MAE
report.pareto_results         # list[SubsetResult] -- Pareto-optimal configs
report.sensitivity_results    # list[SensitivityResult] -- importance ranking
report.mri_result             # ModelRobustnessIndex -- robustness stats
report.adversarial_results    # list[AdversarialResult] -- tipping points
report.composition_result     # CompositionResult -- greedy build order
report.forward_results        # list[SubsetResult] -- forward selection baseline
report.backward_results       # list[SubsetResult] -- backward elimination baseline
```

### Text report

```python
print(report.summary())
```

### LaTeX tables (for papers)

```python
from structural_fuzzing.report import format_latex_tables

latex = format_latex_tables(report)
print(latex)  # Pareto, sensitivity, and MRI tables
```

### Programmatic access to results

```python
# Best overall configuration
best = report.subset_results[0]
print(f"Best: {best.dim_names}, MAE={best.mae:.4f}")
print(f"Parameters: {best.param_values}")
print(f"Per-target errors: {best.errors}")

# Most important dimension
most_important = report.sensitivity_results[0]
print(f"Most important: {most_important.dim_name} "
      f"(removing it increases MAE by {most_important.delta_mae:.4f})")

# Fragile dimensions (those with adversarial thresholds)
for adv in report.adversarial_results:
    print(f"{adv.dim_name}: breaks at {adv.threshold_ratio:.1f}x "
          f"baseline ({adv.direction})")
```

## Complete Examples

### Example 1: ML Defect Prediction

Validate a RandomForest defect predictor by fuzzing its feature groups:

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from structural_fuzzing import run_campaign

# Suppose you have X_train, y_train, X_test, y_test with 16 features
# grouped into 5 families:
GROUPS = {
    "Size": [0, 1, 2],            # LOC, SLOC, blank lines
    "Complexity": [3, 4, 5],      # cyclomatic, essential, design
    "Halstead": [6, 7, 8, 9],     # volume, difficulty, effort, time
    "OO": [10, 11, 12],           # coupling, cohesion, inheritance
    "Process": [13, 14, 15],      # revisions, authors, churn
}
GROUP_NAMES = list(GROUPS.keys())
GROUP_INDICES = list(GROUPS.values())

TARGETS = {"Accuracy": 75.0, "Precision": 70.0, "Recall": 65.0, "F1": 67.0}

def evaluate_fn(params):
    # Select features from active groups
    active_features = []
    for i, indices in enumerate(GROUP_INDICES):
        if params[i] < 1000:  # group is active
            active_features.extend(indices)

    if not active_features:
        errors = {name: -val for name, val in TARGETS.items()}
        mae = sum(abs(v) for v in errors.values()) / len(errors)
        return mae, errors

    # Train and evaluate with only active features
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_train[:, active_features], y_train)
    y_pred = rf.predict(X_test[:, active_features])

    errors = {
        "Accuracy": accuracy_score(y_test, y_pred) * 100 - TARGETS["Accuracy"],
        "Precision": precision_score(y_test, y_pred, zero_division=0) * 100 - TARGETS["Precision"],
        "Recall": recall_score(y_test, y_pred, zero_division=0) * 100 - TARGETS["Recall"],
        "F1": f1_score(y_test, y_pred, zero_division=0) * 100 - TARGETS["F1"],
    }
    mae = sum(abs(v) for v in errors.values()) / len(errors)
    return mae, errors

report = run_campaign(
    dim_names=GROUP_NAMES,
    evaluate_fn=evaluate_fn,
    max_subset_dims=5,
    n_mri_perturbations=100,
    start_dim=1,  # start from Complexity
    verbose=True,
)
print(report.summary())
```

### Example 2: Regression Model Validation

Validate a regression model's feature importance structure:

```python
import numpy as np
from structural_fuzzing import run_campaign

# Your trained model and test data
# model = ...
# X_test, y_test = ...

FEATURE_GROUPS = {
    "demographics": [0, 1, 2],
    "financial": [3, 4, 5, 6],
    "behavioral": [7, 8],
    "temporal": [9, 10, 11],
}

def evaluate_fn(params):
    active = []
    for i, indices in enumerate(FEATURE_GROUPS.values()):
        if params[i] < 1e5:
            active.extend(indices)

    if not active:
        return 100.0, {"rmse": 100.0, "r2": -1.0}

    predictions = model.predict(X_test[:, active])
    rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
    ss_res = np.sum((y_test - predictions) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - ss_res / ss_tot

    errors = {
        "rmse": rmse - 5.0,   # target: RMSE < 5.0
        "r2": 0.85 - r2,      # target: R2 > 0.85
    }
    mae = sum(abs(v) for v in errors.values()) / len(errors)
    return mae, errors

report = run_campaign(
    dim_names=list(FEATURE_GROUPS.keys()),
    evaluate_fn=evaluate_fn,
    max_subset_dims=4,
    verbose=True,
)
```

### Example 3: Using Individual Analyses

When you only need specific insights:

```python
import numpy as np
from structural_fuzzing import (
    sensitivity_profile,
    compute_mri,
    find_adversarial_threshold,
)

# After training your model and defining evaluate_fn...
best_params = np.array([1.0, 0.5, 2.0, 0.1])
dim_names = ["alpha", "beta", "gamma", "delta"]

# Q: "Which parameters matter most?"
sensitivity = sensitivity_profile(best_params, dim_names, evaluate_fn)
print("Importance ranking:")
for s in sensitivity:
    print(f"  {s.importance_rank}. {s.dim_name} (delta={s.delta_mae:+.4f})")

# Q: "How stable is this configuration?"
mri = compute_mri(best_params, evaluate_fn, n_perturbations=500)
print(f"\nRobustness: MRI={mri.mri:.4f} (lower=better)")
print(f"Worst case: MAE={mri.worst_case_mae:.4f}")

# Q: "Where does parameter 'beta' break things?"
thresholds = find_adversarial_threshold(
    best_params, dim=1, dim_names=dim_names,
    evaluate_fn=evaluate_fn, tolerance=0.5,
)
for t in thresholds:
    print(f"\n{t.dim_name} breaks at {t.threshold_ratio:.1f}x "
          f"({t.direction}), flipping '{t.target_flipped}'")
```

## Writing an evaluate_fn: Guidelines

### 1. Handle inactive dimensions

Check each dimension's parameter value and exclude it when inactive:

```python
def evaluate_fn(params):
    INACTIVE_THRESHOLD = 1e5  # slightly below 1e6 default

    active_features = []
    for i, feature_indices in enumerate(feature_groups):
        if params[i] < INACTIVE_THRESHOLD:
            active_features.extend(feature_indices)

    # Use only active_features for prediction...
```

### 2. Return meaningful signed errors

Errors should be `predicted - target` (signed), not absolute values.
This lets the framework distinguish overshooting from undershooting:

```python
errors = {
    "accuracy": actual_accuracy - 0.90,   # positive = exceeds target
    "latency": actual_latency - 100.0,    # positive = over budget
}
```

### 3. Handle the empty case

When all dimensions are inactive, return a large but finite MAE:

```python
if not active_features:
    errors = {"metric_a": -target_a, "metric_b": -target_b}
    mae = sum(abs(v) for v in errors.values()) / len(errors)
    return mae, errors
```

### 4. Keep it deterministic

Use fixed random seeds inside `evaluate_fn` if your model involves
randomness (e.g., RF, neural nets). The framework calls `evaluate_fn`
thousands of times and expects consistent outputs for the same inputs.

## Running the Included Examples

The package ships with two complete examples:

```bash
# Defect prediction (requires scikit-learn)
pip install structural-fuzzing[examples]
python -m examples.defect_prediction.run_campaign

# Geometric economics (numpy only)
python -m examples.geometric_economics.run_campaign
```

## Publishing to PyPI

Build and upload:

```bash
pip install build twine
python -m build
twine check dist/*
twine upload dist/*
```

## Testing

```bash
pip install structural-fuzzing[dev]
pytest -v
```

## API Reference

### Functions

| Function | Module | Description |
|----------|--------|-------------|
| `run_campaign()` | `pipeline` | Run all six analyses + baselines |
| `enumerate_subsets()` | `core` | Test all dimension combinations |
| `optimize_subset()` | `core` | Optimize one specific subset |
| `pareto_frontier()` | `pareto` | Extract Pareto-optimal results |
| `sensitivity_profile()` | `sensitivity` | Ablation-based importance ranking |
| `compute_mri()` | `mri` | Model Robustness Index |
| `find_adversarial_threshold()` | `adversarial` | Tipping point search |
| `compositional_test()` | `compositional` | Greedy dimension building |
| `forward_selection()` | `baselines` | Greedy forward selection baseline |
| `backward_elimination()` | `baselines` | Backward elimination baseline |
| `lasso_selection()` | `baselines` | L1-penalized selection |
| `format_report()` | `report` | Text summary |
| `format_latex_tables()` | `report` | LaTeX tables for papers |

### Result Types

| Type | Key Fields |
|------|------------|
| `SubsetResult` | `dims`, `dim_names`, `n_dims`, `param_values`, `mae`, `errors`, `pareto_optimal` |
| `SensitivityResult` | `dim`, `dim_name`, `mae_with`, `mae_without`, `delta_mae`, `importance_rank` |
| `ModelRobustnessIndex` | `mri`, `mean_omega`, `p75_omega`, `p95_omega`, `worst_case_mae`, `n_perturbations` |
| `AdversarialResult` | `dim`, `dim_name`, `base_value`, `threshold_value`, `threshold_ratio`, `target_flipped`, `direction` |
| `CompositionResult` | `order`, `order_names`, `mae_sequence`, `param_sequence` |
| `StructuralFuzzReport` | `dim_names`, `subset_results`, `pareto_results`, `sensitivity_results`, `mri_result`, `adversarial_results`, `composition_result` |

## License

MIT
