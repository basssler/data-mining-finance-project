"""Shared evaluation helpers for the additive event_v1 experiment lane."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_KEYS = [
    "accuracy",
    "auc_roc",
    "f1",
    "precision",
    "recall",
    "log_loss",
    "positive_prediction_rate",
    "true_class_0_rate",
    "true_class_1_rate",
    "predicted_class_0_rate",
    "predicted_class_1_rate",
]


def _safe_roc_auc(y_true, y_prob) -> float | None:
    """Return ROC AUC or None when the true labels are single-class."""
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return None


def _safe_log_loss(y_true, y_prob) -> float:
    """Return binary log loss with an explicit label set for stability."""
    return float(log_loss(y_true, y_prob, labels=[0, 1]))


def evaluate_classification_run(
    y_true,
    y_prob,
    threshold: float = 0.5,
) -> dict:
    """Return a stable metrics payload for one binary classification run."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    y_pred_arr = (y_prob_arr >= threshold).astype(int)

    true_class_1_rate = float(y_true_arr.mean())
    predicted_class_1_rate = float(y_pred_arr.mean())

    return {
        "threshold": float(threshold),
        "row_count": int(len(y_true_arr)),
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "auc_roc": _safe_roc_auc(y_true_arr, y_prob_arr),
        "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "log_loss": _safe_log_loss(y_true_arr, y_prob_arr),
        "positive_prediction_rate": predicted_class_1_rate,
        "true_class_0_rate": float(1.0 - true_class_1_rate),
        "true_class_1_rate": true_class_1_rate,
        "predicted_class_0_rate": float(1.0 - predicted_class_1_rate),
        "predicted_class_1_rate": predicted_class_1_rate,
        "confusion_matrix": confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).tolist(),
    }


def summarize_fold_metrics(fold_metrics: list[dict]) -> dict:
    """Return mean/std summaries across folds for the shared metric set."""
    summary = {"fold_count": int(len(fold_metrics))}
    for metric_key in METRIC_KEYS:
        metric_values = [
            metric[metric_key]
            for metric in fold_metrics
            if metric.get(metric_key) is not None
        ]
        if metric_values:
            summary[f"{metric_key}_mean"] = float(np.mean(metric_values))
            summary[f"{metric_key}_std"] = float(np.std(metric_values, ddof=0))
        else:
            summary[f"{metric_key}_mean"] = None
            summary[f"{metric_key}_std"] = None
    return summary


def write_json_report(payload: dict, output_path: Path) -> None:
    """Write a machine-readable event_v1 JSON report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_markdown_report(markdown_text: str, output_path: Path) -> None:
    """Write a Markdown report to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown_text, encoding="utf-8")
