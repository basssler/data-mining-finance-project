"""Purged expanding-window validation helpers for the event_v1 lane."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config_event_v1 import (
    DEFAULT_EMBARGO_DAYS,
    DEFAULT_HOLDOUT_START,
    DEFAULT_HORIZON_DAYS,
    DEFAULT_MIN_TRAIN_DATES,
    DEFAULT_N_SPLITS,
)


@dataclass(frozen=True)
class SplitDateMetadata:
    """Human-readable date metadata for one purged split."""

    train_start_date: str | None
    train_end_date: str | None
    purge_start_date: str | None
    purge_end_date: str | None
    validation_start_date: str
    validation_end_date: str


@dataclass(frozen=True)
class PurgeWindowMetadata:
    """Human-readable purge and embargo ranges for one split."""

    overlap_purge_start_date: str | None
    overlap_purge_end_date: str | None
    embargo_start_date: str | None
    embargo_end_date: str | None


def _normalize_dates(df: pd.DataFrame, date_col: str) -> pd.Series:
    """Return normalized pandas datetimes for the provided date column."""
    if date_col not in df.columns:
        raise ValueError(f"Input DataFrame is missing date column: {date_col}")
    dates = pd.to_datetime(df[date_col], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"Date column contains invalid values: {date_col}")
    return dates


def _format_date(value) -> str | None:
    """Return a YYYY-MM-DD string or None for missing timestamps."""
    if value is None or pd.isna(value):
        return None
    return str(pd.Timestamp(value).date())


def _row_positions_for_dates(dates: pd.Series, selected_dates: np.ndarray) -> list[int]:
    """Return row positions for a selected set of unique dates."""
    if len(selected_dates) == 0:
        return []
    mask = dates.isin(selected_dates)
    return np.flatnonzero(mask.to_numpy()).tolist()


def _build_metadata(
    train_dates: np.ndarray,
    purge_dates: np.ndarray,
    validation_dates: np.ndarray,
) -> SplitDateMetadata:
    """Return readable date ranges for one split."""
    return SplitDateMetadata(
        train_start_date=_format_date(train_dates[0]) if len(train_dates) else None,
        train_end_date=_format_date(train_dates[-1]) if len(train_dates) else None,
        purge_start_date=_format_date(purge_dates[0]) if len(purge_dates) else None,
        purge_end_date=_format_date(purge_dates[-1]) if len(purge_dates) else None,
        validation_start_date=_format_date(validation_dates[0]) or "",
        validation_end_date=_format_date(validation_dates[-1]) or "",
    )


def _split_purge_dates(
    purge_dates: np.ndarray,
    horizon_days: int,
    embargo_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split the removed dates into overlap purge vs embargo segments."""
    overlap_count = max(horizon_days - 1, 0)
    embargo_count = min(max(embargo_days, 0), len(purge_dates))
    overlap_dates = purge_dates[:overlap_count]
    embargo_dates = purge_dates[len(purge_dates) - embargo_count :] if embargo_count else np.array([], dtype=purge_dates.dtype)
    return overlap_dates, embargo_dates


def _build_purge_window_metadata(
    purge_dates: np.ndarray,
    horizon_days: int,
    embargo_days: int,
) -> PurgeWindowMetadata:
    """Return readable purge/embargo date ranges for one split."""
    overlap_dates, embargo_dates = _split_purge_dates(
        purge_dates,
        horizon_days=horizon_days,
        embargo_days=embargo_days,
    )
    return PurgeWindowMetadata(
        overlap_purge_start_date=_format_date(overlap_dates[0]) if len(overlap_dates) else None,
        overlap_purge_end_date=_format_date(overlap_dates[-1]) if len(overlap_dates) else None,
        embargo_start_date=_format_date(embargo_dates[0]) if len(embargo_dates) else None,
        embargo_end_date=_format_date(embargo_dates[-1]) if len(embargo_dates) else None,
    )


def make_event_v1_splits(
    df: pd.DataFrame,
    date_col: str,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    n_splits: int = DEFAULT_N_SPLITS,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
    holdout_start: str | None = DEFAULT_HOLDOUT_START,
    min_train_dates: int = DEFAULT_MIN_TRAIN_DATES,
) -> dict:
    """Create deterministic expanding splits with purge/embargo separation.

    The split policy is date-based rather than row-based because the panel has
    multiple rows per trading date. For each validation fold and the final
    holdout, the last `horizon_days + embargo_days - 1` trading dates before
    the evaluation block are removed from the training sample.
    """
    if horizon_days < 1:
        raise ValueError("horizon_days must be at least 1.")
    if embargo_days < 0:
        raise ValueError("embargo_days must be non-negative.")
    if n_splits < 1:
        raise ValueError("n_splits must be at least 1.")

    dates = _normalize_dates(df, date_col)
    unique_dates = np.array(sorted(dates.drop_duplicates().tolist()))
    if len(unique_dates) == 0:
        raise ValueError("No dates are available to create event_v1 splits.")

    holdout_start_ts = pd.Timestamp(holdout_start) if holdout_start is not None else None
    if holdout_start_ts is None:
        raise ValueError("holdout_start must be provided for event_v1 validation.")

    holdout_mask = unique_dates >= holdout_start_ts
    if not holdout_mask.any():
        raise ValueError("No holdout dates were found at or after the requested holdout_start.")

    holdout_start_idx = int(np.flatnonzero(holdout_mask)[0])
    pre_holdout_dates = unique_dates[:holdout_start_idx]
    holdout_dates = unique_dates[holdout_start_idx:]

    if len(pre_holdout_dates) <= min_train_dates:
        raise ValueError("Not enough pre-holdout dates for the requested event_v1 warmup.")
    if len(holdout_dates) == 0:
        raise ValueError("The final holdout block is empty.")

    remaining_pre_holdout = pre_holdout_dates[min_train_dates:]
    validation_blocks = [block for block in np.array_split(remaining_pre_holdout, n_splits) if len(block)]
    if len(validation_blocks) < n_splits:
        raise ValueError("Not enough pre-holdout dates to form the requested number of folds.")

    purge_span = max(horizon_days + embargo_days - 1, 0)
    folds = []
    for fold_number, validation_dates in enumerate(validation_blocks, start=1):
        validation_start_date = validation_dates[0]
        validation_start_idx = int(np.searchsorted(pre_holdout_dates, validation_start_date))
        train_end_exclusive = max(validation_start_idx - purge_span, 0)
        train_dates = pre_holdout_dates[:train_end_exclusive]
        purge_dates = pre_holdout_dates[train_end_exclusive:validation_start_idx]

        if len(train_dates) == 0:
            raise ValueError(
                "Fold construction produced an empty training block. "
                "Increase min_train_dates or reduce purge settings."
            )

        fold_metadata = _build_metadata(train_dates, purge_dates, validation_dates)
        purge_metadata = _build_purge_window_metadata(
            purge_dates,
            horizon_days=horizon_days,
            embargo_days=embargo_days,
        )
        overlap_dates, embargo_dates = _split_purge_dates(
            purge_dates,
            horizon_days=horizon_days,
            embargo_days=embargo_days,
        )
        folds.append(
            {
                "fold_number": fold_number,
                "train_indices": _row_positions_for_dates(dates, train_dates),
                "validation_indices": _row_positions_for_dates(dates, validation_dates),
                "date_metadata": fold_metadata.__dict__,
                "purge_window_metadata": purge_metadata.__dict__,
                "train_date_count": int(len(train_dates)),
                "validation_date_count": int(len(validation_dates)),
                "purged_date_count": int(len(purge_dates)),
                "overlap_purge_date_count": int(len(overlap_dates)),
                "embargo_date_count": int(len(embargo_dates)),
            }
        )

    holdout_train_end_exclusive = max(len(pre_holdout_dates) - purge_span, 0)
    holdout_train_dates = pre_holdout_dates[:holdout_train_end_exclusive]
    holdout_purge_dates = pre_holdout_dates[holdout_train_end_exclusive:]
    holdout_metadata = _build_metadata(holdout_train_dates, holdout_purge_dates, holdout_dates)
    holdout_purge_metadata = _build_purge_window_metadata(
        holdout_purge_dates,
        horizon_days=horizon_days,
        embargo_days=embargo_days,
    )
    holdout_overlap_dates, holdout_embargo_dates = _split_purge_dates(
        holdout_purge_dates,
        horizon_days=horizon_days,
        embargo_days=embargo_days,
    )

    return {
        "date_column": date_col,
        "horizon_days": int(horizon_days),
        "embargo_days": int(embargo_days),
        "min_train_dates": int(min_train_dates),
        "n_splits": int(n_splits),
        "holdout_start": _format_date(holdout_dates[0]),
        "pre_holdout_start": _format_date(pre_holdout_dates[0]),
        "pre_holdout_end": _format_date(pre_holdout_dates[-1]),
        "holdout_end": _format_date(holdout_dates[-1]),
        "folds": folds,
        "holdout": {
            "train_indices": _row_positions_for_dates(dates, holdout_train_dates),
            "holdout_indices": _row_positions_for_dates(dates, holdout_dates),
            "date_metadata": holdout_metadata.__dict__,
            "purge_window_metadata": holdout_purge_metadata.__dict__,
            "train_date_count": int(len(holdout_train_dates)),
            "holdout_date_count": int(len(holdout_dates)),
            "purged_date_count": int(len(holdout_purge_dates)),
            "overlap_purge_date_count": int(len(holdout_overlap_dates)),
            "embargo_date_count": int(len(holdout_embargo_dates)),
        },
    }
