"""Generate leak-free walk-forward splits for the stock-date modeling panel."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.project_config import MODEL_PANEL_PATH, SPLIT_REPORT_PATH, VALIDATION_CONFIG, dump_json, ensure_stock_prediction_directories


def generate_walk_forward_splits(dates: pd.Series) -> list[dict[str, str]]:
    """Create expanding-window folds with an embargo gap before validation."""
    unique_dates = pd.Index(sorted(pd.to_datetime(dates).dropna().unique()))
    train_end_pool = unique_dates[unique_dates < pd.Timestamp(VALIDATION_CONFIG["holdout_start"])]
    n_splits = int(VALIDATION_CONFIG["n_splits"])
    gap_days = int(VALIDATION_CONFIG["gap_days"])
    min_train_days = int(VALIDATION_CONFIG["min_train_days"])

    eligible = train_end_pool[min_train_days:]
    if len(eligible) < n_splits:
        raise ValueError("Not enough history to generate the requested number of walk-forward folds.")

    step = max(1, len(eligible) // n_splits)
    fold_boundaries = eligible.to_series().iloc[::step].head(n_splits)
    splits: list[dict[str, str]] = []
    for fold_id, validation_start in enumerate(fold_boundaries, start=1):
        train_end = validation_start - pd.Timedelta(days=gap_days)
        if fold_id < len(fold_boundaries):
            validation_end = fold_boundaries.iloc[fold_id] - pd.Timedelta(days=1)
        else:
            validation_end = pd.Timestamp(VALIDATION_CONFIG["train_validation_end"])

        splits.append(
            {
                "fold": fold_id,
                "train_start": str(pd.Timestamp(VALIDATION_CONFIG["train_validation_start"]).date()),
                "train_end": str(train_end.date()),
                "validation_start": str(pd.Timestamp(validation_start).date()),
                "validation_end": str(pd.Timestamp(validation_end).date()),
                "holdout_start": VALIDATION_CONFIG["holdout_start"],
                "holdout_end": VALIDATION_CONFIG["holdout_end"],
            }
        )
    return splits


def main() -> None:
    """Generate and persist split metadata."""
    ensure_stock_prediction_directories()
    model_panel = pd.read_parquet(MODEL_PANEL_PATH, columns=["date"])
    splits = generate_walk_forward_splits(model_panel["date"])
    dump_json({"splits": splits}, SPLIT_REPORT_PATH)


if __name__ == "__main__":
    main()
