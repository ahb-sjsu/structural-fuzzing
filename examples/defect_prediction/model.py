"""Sklearn RF defect prediction model with feature groups.

Uses synthetic data that mimics the structure of real defect prediction datasets
(NASA MDP / PROMISE repository style). Fully self-contained and reproducible.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# Feature groups (the "dimensions" for structural fuzzing)
FEATURE_GROUPS = {
    "Size": [0, 1, 2],
    "Complexity": [3, 4, 5],
    "Halstead": [6, 7, 8, 9],
    "OO": [10, 11, 12],
    "Process": [13, 14, 15],
}
GROUP_NAMES = list(FEATURE_GROUPS.keys())
N_FEATURES = 16
N_GROUPS = len(GROUP_NAMES)

FEATURE_NAMES = [
    "LOC", "SLOC", "blank_lines",
    "cyclomatic", "essential_complexity", "design_complexity",
    "halstead_volume", "halstead_difficulty", "halstead_effort", "halstead_time",
    "coupling", "cohesion", "inheritance_depth",
    "revisions", "authors", "churn",
]


def generate_defect_data(
    n_samples: int = 1000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic defect prediction data with known feature group structure.

    Ground truth: defects are primarily driven by complexity and process metrics.
    Size and Halstead are correlated but less predictive. OO metrics are noise.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (X, y) where X is (n_samples, 16) features and y is binary labels.
    """
    rng = np.random.default_rng(seed)

    # Size metrics (weakly predictive)
    loc = rng.lognormal(5, 1, n_samples)
    sloc = loc * rng.uniform(0.7, 0.9, n_samples)
    blank = loc * rng.uniform(0.05, 0.15, n_samples)

    # Complexity (strongly predictive)
    cyclomatic = rng.poisson(5, n_samples).astype(float) + 1
    essential = cyclomatic * rng.uniform(0.3, 0.8, n_samples)
    design = rng.poisson(3, n_samples).astype(float) + 1

    # Halstead (moderately predictive, correlated with size)
    volume = loc * rng.uniform(8, 15, n_samples)
    difficulty = rng.uniform(5, 30, n_samples)
    effort = volume * difficulty
    time_est = effort / 18

    # OO metrics (noise - not predictive)
    coupling = rng.poisson(3, n_samples).astype(float)
    cohesion = rng.uniform(0, 1, n_samples)
    inheritance = (rng.geometric(0.5, n_samples) - 1).astype(float)

    # Process metrics (strongly predictive)
    revisions = rng.poisson(8, n_samples).astype(float) + 1
    authors = rng.poisson(2, n_samples).astype(float) + 1
    churn = revisions * rng.uniform(10, 100, n_samples)

    X = np.column_stack([
        loc, sloc, blank,
        cyclomatic, essential, design,
        volume, difficulty, effort, time_est,
        coupling, cohesion, inheritance,
        revisions, authors, churn,
    ])

    # Defect probability: driven by complexity and process, with some size contribution
    logit = (
        -3
        + 0.1 * np.log1p(cyclomatic)
        + 0.15 * np.log1p(essential)
        + 0.05 * np.log1p(design)
        + 0.12 * np.log1p(revisions)
        + 0.1 * np.log1p(authors)
        + 0.08 * np.log1p(churn / 100)
        + 0.03 * np.log1p(loc / 1000)
        + rng.normal(0, 0.5, n_samples)
    )
    p_defect = 1 / (1 + np.exp(-logit))
    y = (rng.random(n_samples) < p_defect).astype(int)

    return X, y


def make_evaluate_fn(
    n_samples: int = 1000,
    test_fraction: float = 0.3,
    seed: int = 42,
) -> Callable[[np.ndarray], tuple[float, dict[str, float]]]:
    """Create evaluate_fn compatible with structural_fuzzing.

    For each configuration, params[i] represents the weight for feature group i.
    If params[i] >= 1000, the group is considered inactive (excluded).
    Otherwise, features in that group are scaled by 1/params[i].

    Parameters
    ----------
    n_samples : int
        Number of synthetic samples.
    test_fraction : float
        Fraction of data for testing.
    seed : int
        Random seed.

    Returns
    -------
    callable
        Function (params: ndarray) -> (mae: float, errors: dict).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    X, y = generate_defect_data(n_samples=n_samples, seed=seed)

    # Train/test split
    rng = np.random.default_rng(seed + 1)
    n_test = int(n_samples * test_fraction)
    indices = rng.permutation(n_samples)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    group_names = list(FEATURE_GROUPS.keys())
    group_indices = list(FEATURE_GROUPS.values())

    # Prediction quality targets
    target_values = {
        "Accuracy": 75.0,
        "Precision": 70.0,
        "Recall": 65.0,
        "F1": 67.0,
        "AUC": 80.0,
    }

    def evaluate_fn(params: np.ndarray) -> tuple[float, dict[str, float]]:
        # Select features from active groups
        active_features: list[int] = []
        for i, indices in enumerate(group_indices):
            if params[i] < 1000:
                active_features.extend(indices)

        if not active_features:
            # No features: return large errors
            errors = {name: -val for name, val in target_values.items()}
            mae = sum(abs(v) for v in errors.values()) / len(errors)
            return mae, errors

        X_tr = X_train[:, active_features]
        X_te = X_test[:, active_features]

        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
        rf.fit(X_tr, y_train)

        y_pred = rf.predict(X_te)
        y_prob = rf.predict_proba(X_te)

        # Handle case where model only predicts one class
        if y_prob.shape[1] < 2:
            y_prob_pos = np.zeros(len(y_test))
        else:
            y_prob_pos = y_prob[:, 1]

        # Compute metrics
        acc = accuracy_score(y_test, y_pred) * 100
        prec = precision_score(y_test, y_pred, zero_division=0) * 100
        rec = recall_score(y_test, y_pred, zero_division=0) * 100
        f1 = f1_score(y_test, y_pred, zero_division=0) * 100

        try:
            auc = roc_auc_score(y_test, y_prob_pos) * 100
        except ValueError:
            auc = 50.0

        # Errors: predicted - target
        errors = {
            "Accuracy": acc - target_values["Accuracy"],
            "Precision": prec - target_values["Precision"],
            "Recall": rec - target_values["Recall"],
            "F1": f1 - target_values["F1"],
            "AUC": auc - target_values["AUC"],
        }

        mae = sum(abs(v) for v in errors.values()) / len(errors)
        return mae, errors

    return evaluate_fn
