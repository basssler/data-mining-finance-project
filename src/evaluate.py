"""Evaluation helpers for classification results.

This file will later contain readable wrappers around common model
evaluation steps such as accuracy, confusion matrices, and reports.
"""

from sklearn.metrics import accuracy_score


def compute_accuracy(y_true, y_pred) -> float:
    """Return simple classification accuracy."""
    return float(accuracy_score(y_true, y_pred))
