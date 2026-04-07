"""Engineer event_v1 Layer 2 market control features from daily prices."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.config_event_v1 import (
    LAYER2_V2_FEATURE_COLUMNS,
    MARKET_FEATURE_V2_OUTPUT_PATH,
    PRICE_INPUT_PATH,
    ensure_event_v1_directories,
)

REQUIRED_COLUMNS = [
    "ticker",
    "date",
    "open",
    "close",
    "adj_close",
    "volume",
]


def load_price_data(path: Path) -> pd.DataFrame:
    """Load the daily price table and validate the required schema."""
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

    for column in ["open", "close", "adj_close", "volume"]:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    prepared = prepared.dropna(subset=["ticker", "date", "adj_close"]).copy()
    prepared = prepared.sort_values(["ticker", "date"]).reset_index(drop=True)
    return prepared


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide safely and return NaN when the denominator is missing or zero."""
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")

    valid = denominator.notna() & np.isfinite(denominator) & (denominator != 0)
    result = pd.Series(np.nan, index=numerator.index, dtype="float64")
    result.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    return result


def build_leave_one_out_mean(series: pd.Series, group_keys: pd.Series) -> pd.Series:
    """Return a leave-one-out group mean for a row-level numeric series."""
    numeric_series = pd.to_numeric(series, errors="coerce")
    group_sum = numeric_series.groupby(group_keys).transform("sum")
    group_count = numeric_series.groupby(group_keys).transform("count")
    denominator = group_count - 1

    result = pd.Series(np.nan, index=numeric_series.index, dtype="float64")
    valid = denominator > 0
    result.loc[valid] = (group_sum.loc[valid] - numeric_series.loc[valid]) / denominator.loc[valid]
    return result


def build_market_features_v2(
    prices_df: pd.DataFrame,
    benchmark_df: pd.DataFrame | None = None,
    sector_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create the event_v1 Layer 2 market control features."""
    del benchmark_df, sector_df

    feature_df = prices_df.copy()
    grouped_ticker = feature_df.groupby("ticker")
    daily_return = grouped_ticker["adj_close"].pct_change()

    feature_df["stock_return_5d"] = safe_divide(
        feature_df["adj_close"],
        grouped_ticker["adj_close"].shift(5),
    ) - 1.0
    feature_df["stock_return_10d"] = safe_divide(
        feature_df["adj_close"],
        grouped_ticker["adj_close"].shift(10),
    ) - 1.0
    feature_df["stock_return_21d"] = safe_divide(
        feature_df["adj_close"],
        grouped_ticker["adj_close"].shift(21),
    ) - 1.0

    feature_df["sector_return_5d"] = build_leave_one_out_mean(
        feature_df["stock_return_5d"],
        feature_df["date"],
    )
    feature_df["sector_return_10d"] = build_leave_one_out_mean(
        feature_df["stock_return_10d"],
        feature_df["date"],
    )
    feature_df["sector_return_21d"] = build_leave_one_out_mean(
        feature_df["stock_return_21d"],
        feature_df["date"],
    )

    feature_df["rel_return_5d"] = feature_df["stock_return_5d"] - feature_df["sector_return_5d"]
    feature_df["rel_return_10d"] = feature_df["stock_return_10d"] - feature_df["sector_return_10d"]
    feature_df["rel_return_21d"] = feature_df["stock_return_21d"] - feature_df["sector_return_21d"]

    feature_df["daily_return_1d"] = daily_return
    feature_df["sector_daily_return_1d"] = build_leave_one_out_mean(
        feature_df["daily_return_1d"],
        feature_df["date"],
    )

    feature_df["realized_vol_21d"] = (
        feature_df.groupby("ticker")["daily_return_1d"]
        .rolling(window=21, min_periods=21)
        .std()
        .reset_index(level=0, drop=True)
    )
    feature_df["realized_vol_63d"] = (
        feature_df.groupby("ticker")["daily_return_1d"]
        .rolling(window=63, min_periods=63)
        .std()
        .reset_index(level=0, drop=True)
    )
    feature_df["vol_ratio_21d_63d"] = safe_divide(
        feature_df["realized_vol_21d"],
        feature_df["realized_vol_63d"],
    )

    def compute_rolling_beta(group: pd.DataFrame) -> pd.Series:
        covariance = group["daily_return_1d"].rolling(63, min_periods=63).cov(
            group["sector_daily_return_1d"]
        )
        benchmark_variance = group["sector_daily_return_1d"].rolling(63, min_periods=63).var()
        return safe_divide(covariance, benchmark_variance)

    feature_df["beta_63d_to_sector"] = (
        feature_df.groupby("ticker", group_keys=False)
        .apply(compute_rolling_beta)
        .reset_index(level=0, drop=True)
    )

    prior_close = grouped_ticker["close"].shift(1)
    feature_df["overnight_gap_1d"] = safe_divide(feature_df["open"], prior_close) - 1.0
    feature_df["abs_return_shock_1d"] = feature_df["daily_return_1d"].abs()

    rolling_high_21d = (
        feature_df.groupby("ticker")["adj_close"]
        .rolling(window=21, min_periods=21)
        .max()
        .reset_index(level=0, drop=True)
    )
    feature_df["drawdown_21d"] = safe_divide(feature_df["adj_close"], rolling_high_21d) - 1.0

    rolling_mean_21d = (
        feature_df.groupby("ticker")["daily_return_1d"]
        .rolling(window=21, min_periods=21)
        .mean()
        .reset_index(level=0, drop=True)
    )
    rolling_std_21d = (
        feature_df.groupby("ticker")["daily_return_1d"]
        .rolling(window=21, min_periods=21)
        .std()
        .reset_index(level=0, drop=True)
    )
    feature_df["return_zscore_21d"] = safe_divide(
        feature_df["daily_return_1d"] - rolling_mean_21d,
        rolling_std_21d,
    )

    average_volume_20d = (
        feature_df.groupby("ticker")["volume"]
        .rolling(window=20, min_periods=20)
        .mean()
        .reset_index(level=0, drop=True)
    )
    feature_df["volume_ratio_20d"] = safe_divide(feature_df["volume"], average_volume_20d)
    feature_df["log_volume"] = np.log1p(feature_df["volume"])
    feature_df["abnormal_volume_flag"] = (feature_df["volume_ratio_20d"] >= 1.5).astype("int64")

    return feature_df


def save_market_features(df: pd.DataFrame, output_path: Path) -> None:
    """Save the v2 market control features to parquet."""
    keep_columns = ["ticker", "date"] + LAYER2_V2_FEATURE_COLUMNS
    df[keep_columns].to_parquet(output_path, index=False)


def print_summary(df: pd.DataFrame) -> None:
    """Print a compact summary of the v2 market control layer."""
    print("\nEvent V1 Layer 2 Market Feature Summary")
    print("-" * 60)
    print(f"Rows: {len(df):,}")
    print(f"Tickers: {df['ticker'].nunique():,}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    print("\nMissingness")
    print("-" * 60)
    missingness = df[LAYER2_V2_FEATURE_COLUMNS].isna().mean().mul(100).sort_values(ascending=False)
    for column_name, percentage in missingness.items():
        print(f"{column_name:<24} {percentage:>8.2f}%")


def main() -> None:
    """Build and save event_v1 Layer 2 market controls."""
    ensure_event_v1_directories()
    print(f"Loading price data from: {PRICE_INPUT_PATH}")
    price_df = load_price_data(PRICE_INPUT_PATH)

    print("Normalizing price data...")
    normalized_df = normalize_price_data(price_df)

    print("Engineering event_v1 Layer 2 market controls...")
    featured_df = build_market_features_v2(normalized_df)

    print(f"Saving Layer 2 v2 market features to: {MARKET_FEATURE_V2_OUTPUT_PATH}")
    save_market_features(featured_df, MARKET_FEATURE_V2_OUTPUT_PATH)

    print_summary(featured_df)
    print("\nSaved event_v1 Layer 2 market features.")


if __name__ == "__main__":
    main()
