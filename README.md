# structural-fuzzing

Structural fuzzing framework for parameterized model validation.

Adapts the adversarial mindset of software fuzzing to model validation:
instead of mutating program inputs to find crashes, we mutate model
parameters to find prediction failures.

## Installation

```bash
pip install -e ".[dev,examples]"
```

## Quick Start

```python
import numpy as np
from structural_fuzzing import run_campaign

# Define your model's evaluate function
def evaluate_fn(params):
    """params: 1D array, one value per dimension.
    Returns: (mae, {target_name: error})
    """
    # Your model logic here
    errors = {"target_1": compute_error_1(params), ...}
    mae = sum(abs(v) for v in errors.values()) / len(errors)
    return mae, errors

# Run the full campaign
report = run_campaign(
    dim_names=["dim_a", "dim_b", "dim_c"],
    evaluate_fn=evaluate_fn,
)
print(report.summary())
```

## Components

1. **Dimension Enumeration** -- Exhaustive search over parameter subsets
2. **Pareto Frontier** -- Identify accuracy-complexity tradeoffs
3. **Sensitivity Profiling** -- Ablation-based importance ranking
4. **Model Robustness Index (MRI)** -- Quantify stability under perturbation
5. **Adversarial Threshold Search** -- Find parameter tipping points
6. **Compositional Testing** -- Greedy dimension-building sequences

## Examples

- `examples/geometric_economics/` -- 9D behavioral economics model with 16 prediction targets
- `examples/defect_prediction/` -- Software defect prediction with sklearn RandomForest

## Testing

```bash
pytest -v
```

## License

MIT
