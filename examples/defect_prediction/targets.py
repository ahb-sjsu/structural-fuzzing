"""Prediction targets for defect prediction model.

The targets are defined inline in model.py's make_evaluate_fn for simplicity,
since they depend on the trained model. This module provides constants and
utility functions for reference.
"""

from __future__ import annotations

# Target metric values that the model should achieve
TARGET_ACCURACY = 75.0
TARGET_PRECISION = 70.0
TARGET_RECALL = 65.0
TARGET_F1 = 67.0
TARGET_AUC = 80.0

TARGET_NAMES = ["Accuracy", "Precision", "Recall", "F1", "AUC"]

TARGET_VALUES = {
    "Accuracy": TARGET_ACCURACY,
    "Precision": TARGET_PRECISION,
    "Recall": TARGET_RECALL,
    "F1": TARGET_F1,
    "AUC": TARGET_AUC,
}


def describe_targets() -> str:
    """Return a human-readable description of all targets."""
    lines = ["Defect Prediction Targets:", ""]
    for name, value in TARGET_VALUES.items():
        lines.append(f"  {name}: {value:.1f}%")
    return "\n".join(lines)
