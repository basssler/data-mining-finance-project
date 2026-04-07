"""Build additive event_v1 labels from the existing daily price table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.config_event_v1 import (
    DEFAULT_BENCHMARK_MODE,
    DEFAULT_HORIZON_DAYS,
    DEFAULT_NEUTRAL_BAND_BPS,
    LABEL_OUTPUT_PATH,
    PRICE_INPUT_PATH,
    ensure_event_v1_directories,
)

REQUIRED_COLUMNS = [
    "ticker",
    "date",
    "adj_close",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI options for the event_v1 label build."""
    parser = argparse.ArgumentParser(description="Build labels for the event_v1 lane.")
    parser.add_argument("--horizon-days", type=int, default=DEFAULT_HORIZON_DAYS)
    parser.add_argument("--benchmark-mode", default=DEFAULT_BENCHMARK_MODE)
    parser.add_argument("--neutral-band-bps", type=int, default=DEFAULT_NEUTRAL_BAND_BPS)
    return parser.parse_args()


def load_price_data(path: Path) -> pd.DataFrame:
    """Load the existing daily price table used by the locked benchmark."""
    if not path.exists():
        raise FileNotFoundError(f"Price file was not found: {path}")
    df = pd.read_parquet(path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError("Price file is missing required columns: " + ", ".join(missing_columns))
    return df.copy()


def normalize_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dtypes and sort by ticker/date."""
    prepared = df.copy()
    prepared["ticker"] = prepared["ticker"].astype("string")
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    prepared["adj_close"] = pd.to_numeric(prepared["adj_close"], errors="coerce")

    if "forward_return_5d" in prepared.columns:
        prepared["forward_return_5d"] = pd.to_numeric(
            prepared["forward_return_5d"],
            errors="coerce",
        )

    prepared = prepared.dropna(subset=["ticker", "date", "adj_close"]).copy()
    prepared = prepared.sort_values(["ticker", "date"]).reset_index(drop=True)
    return prepared


def _compute_stock_forward_return(df: pd.DataFrame, horizon_days: int) -> pd.Series:
    """Return stock forward return from buffered labels when available.

    The locked price table already contains a 5-day forward return computed
    with a download window buffered past 2024-12-31. Reusing it preserves the
    final in-sample rows. For other horizons, the series is recomputed from
    adjusted close.
    """
    if horizon_days == 5 and "forward_return_5d" in df.columns and df["forward_return_5d"].notna().any():
        return pd.to_numeric(df["forward_return_5d"], errors="coerce")

    future_price = df.groupby("ticker")["adj_close"].shift(-horizon_days)
    return (future_price / df["adj_close"]) - 1.0


def _build_leave_one_out_mean(series: pd.Series, group_keys: pd.Series) -> pd.Series:
    """Return a leave-one-out group mean for a row-level numeric series."""
    numeric_series = pd.to_numeric(series, errors="coerce")
    group_sum = numeric_series.groupby(group_keys).transform("sum")
    group_count = numeric_series.groupby(group_keys).transform("count")
    denominator = group_count - 1
    loo_mean = pd.Series(np.nan, index=numeric_series.index, dtype="float64")

    valid = denominator > 0
    loo_mean.loc[valid] = (group_sum.loc[valid] - numeric_series.loc[valid]) / denominator.loc[valid]
    return loo_mean


def build_event_v1_labels(
    prices_df: pd.DataFrame,
    ticker_col: str = "ticker",
    date_col: str = "date",
    close_col: str = "adj_close",
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    benchmark_mode: str = DEFAULT_BENCHMARK_MODE,
    neutral_band_bps: int | None = DEFAULT_NEUTRAL_BAND_BPS,
) -> pd.DataFrame:
    """Create the event_v1 excess-return label table.

    The benchmark is a leave-one-out equal-weight return across the current
    Consumer Staples universe, which acts as the sector control series.
    """
    if benchmark_mode not in {
        "sector_equal_weight_ex_self",
        "equal_weight_ex_self",
        "universe_equal_weight_ex_self",
    }:
        raise ValueError(f"Unsupported benchmark_mode for event_v1: {benchmark_mode}")

    labels_df = prices_df.copy()
    labels_df[close_col] = pd.to_numeric(labels_df[close_col], errors="coerce")
    labels_df["forward_return_5d"] = _compute_stock_forward_return(labels_df, horizon_days=horizon_days)
    labels_df["benchmark_forward_return_5d"] = _build_leave_one_out_mean(
        labels_df["forward_return_5d"],
        labels_df[date_col],
    )
    labels_df["excess_forward_return_5d"] = (
        labels_df["forward_return_5d"] - labels_df["benchmark_forward_return_5d"]
    )

    labels_df["target_event_v1"] = np.where(
        labels_df["excess_forward_return_5d"].notna(),
        (labels_df["excess_forward_return_5d"] > 0).astype(int),
        np.nan,
    )
    labels_df["target_event_v1"] = pd.Series(
        labels_df["target_event_v1"],
        index=labels_df.index,
    ).astype("Int64")

    if neutral_band_bps is None:
        labels_df["within_neutral_band"] = False
    else:
        band_decimal = neutral_band_bps / 10000.0
        labels_df["within_neutral_band"] = (
            labels_df["excess_forward_return_5d"].abs() < band_decimal
        ).fillna(False)

    output_columns = [
        ticker_col,
        date_col,
        "forward_return_5d",
        "benchmark_forward_return_5d",
        "excess_forward_return_5d",
        "target_event_v1",
        "within_neutral_band",
    ]
    output_df = labels_df[output_columns].copy()
    output_df = output_df.sort_values([ticker_col, date_col]).reset_index(drop=True)
    return output_df


def print_label_summary(df: pd.DataFrame) -> None:
    """Print a compact summary of the event_v1 labels."""
    target_non_null = df["target_event_v1"].notna()
    target_balance = (
        df.loc[target_non_null, "target_event_v1"].astype(int).value_counts(normalize=True).sort_index()
    )

    print("\nEvent V1 Label Summary")
    print("-" * 60)
    print(f"Rows: {len(df):,}")
    print(f"Tickers: {df['ticker'].nunique():,}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Rows with target available: {target_non_null.sum():,}")
    print(f"Rows within neutral band:   {int(df['within_neutral_band'].sum()):,}")
    print(f"Mean stock forward return:  {df['forward_return_5d'].mean():.6f}")
    print(f"Mean benchmark return:      {df['benchmark_forward_return_5d'].mean():.6f}")
    print(f"Mean excess return:         {df['excess_forward_return_5d'].mean():.6f}")
    print(
        "Target balance: "
        f"class_0={float(target_balance.get(0, 0.0) * 100):.2f}%, "
        f"class_1={float(target_balance.get(1, 0.0) * 100):.2f}%"
    )


def main() -> None:
    """Build and save the event_v1 labels."""
    args = parse_args()
    ensure_event_v1_directories()

    print(f"Loading price data from: {PRICE_INPUT_PATH}")
    price_df = load_price_data(PRICE_INPUT_PATH)
    normalized_prices = normalize_price_data(price_df)

    print("Building event_v1 excess-return labels...")
    label_df = build_event_v1_labels(
        normalized_prices,
        horizon_days=args.horizon_days,
        benchmark_mode=args.benchmark_mode,
        neutral_band_bps=args.neutral_band_bps,
    )

    print(f"Saving event_v1 labels to: {LABEL_OUTPUT_PATH}")
    label_df.to_parquet(LABEL_OUTPUT_PATH, index=False)

    print_label_summary(label_df)
    print("\nSaved event_v1 label dataset.")


if __name__ == "__main__":
    main()
