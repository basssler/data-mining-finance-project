"""Utility helpers shared across the WRDS stock pipeline."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def configure_logging(name: str) -> logging.Logger:
    """Return a console logger with a stable format."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Return a safe numeric division."""
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    valid = denominator.notna() & np.isfinite(denominator) & (denominator != 0)
    result = pd.Series(np.nan, index=numerator.index, dtype="float64")
    result.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    return result


def winsorize_series(series: pd.Series, lower_quantile: float, upper_quantile: float) -> pd.Series:
    """Clip extreme values using quantiles."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return numeric.astype("float64")
    lower = numeric.quantile(lower_quantile)
    upper = numeric.quantile(upper_quantile)
    return numeric.clip(lower=lower, upper=upper)


def log_frame_diagnostics(
    logger: logging.Logger,
    df: pd.DataFrame,
    *,
    name: str,
    date_column: str | None = None,
    key_columns: Iterable[str] | None = None,
) -> None:
    """Log row counts, date ranges, and missing key percentages."""
    logger.info("%s rows: %s", name, f"{len(df):,}")
    if date_column and date_column in df.columns and df[date_column].notna().any():
        date_values = pd.to_datetime(df[date_column], errors="coerce")
        logger.info("%s date range: %s to %s", name, date_values.min(), date_values.max())
    if key_columns:
        for column in key_columns:
            if column not in df.columns:
                logger.warning("%s is missing key column: %s", name, column)
                continue
            missing_pct = float(df[column].isna().mean() * 100)
            logger.info("%s missing %s: %.2f%%", name, column, missing_pct)


def write_missingness_report(df: pd.DataFrame, columns: list[str], output_path: Path) -> pd.DataFrame:
    """Write per-column missingness to CSV and return it."""
    report = (
        pd.DataFrame(
            {
                "column": columns,
                "missing_pct": [float(df[column].isna().mean() * 100) for column in columns],
            }
        )
        .sort_values("missing_pct", ascending=False)
        .reset_index(drop=True)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    return report


def rolling_sign_flip_count(series: pd.Series, window: int) -> pd.Series:
    """Count sign changes inside a rolling window."""
    numeric = pd.to_numeric(series, errors="coerce")
    signs = np.sign(numeric)

    def count_flips(values: np.ndarray) -> float:
        values = values[~np.isnan(values)]
        if len(values) < 2:
            return math.nan
        return float(np.sum(values[1:] * values[:-1] < 0))

    return signs.rolling(window=window, min_periods=window).apply(count_flips, raw=True)


def rolling_max_drawdown(price_series: pd.Series, window: int) -> pd.Series:
    """Compute rolling max drawdown using trailing prices only."""
    prices = pd.to_numeric(price_series, errors="coerce")

    def max_drawdown(values: np.ndarray) -> float:
        values = values[~np.isnan(values)]
        if len(values) < window:
            return math.nan
        running_peak = np.maximum.accumulate(values)
        drawdowns = values / running_peak - 1.0
        return float(np.min(drawdowns))

    return prices.rolling(window=window, min_periods=window).apply(max_drawdown, raw=True)


def robust_zscore(series: pd.Series) -> pd.Series:
    """Return a median/MAD-based z-score."""
    numeric = pd.to_numeric(series, errors="coerce")
    median = numeric.median(skipna=True)
    mad = (numeric - median).abs().median(skipna=True)
    if pd.isna(mad) or mad == 0:
        return pd.Series(np.nan, index=numeric.index, dtype="float64")
    return 0.6745 * (numeric - median) / mad
