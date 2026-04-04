"""Engineer Layer 2 market features from the daily price table.

This script builds simple price-based features for Layer 2. It does not touch
the target definition. It only reads the daily price/label table and appends
backward-looking market indicators that are safe to use on each trading date.

Input:
    data/interim/prices/prices_with_labels.parquet

Output:
    data/interim/features/layer2_market_features.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.paths import INTERIM_DATA_DIR

REQUIRED_COLUMNS = [
    "ticker",
    "date",
    "adj_close",
    "volume",
]

MARKET_FEATURE_COLUMNS = [
    "return_5d",
    "return_21d",
    "volatility_21d",
    "volume_ratio_20d",
    "rsi_14",
]


def get_input_path() -> Path:
    """Return the daily price input path."""
    return INTERIM_DATA_DIR / "prices" / "prices_with_labels.parquet"


def get_output_path() -> Path:
    """Return the Layer 2 feature output path and create its folder."""
    output_dir = INTERIM_DATA_DIR / "features"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "layer2_market_features.parquet"


def load_price_data(path: Path) -> pd.DataFrame:
    """Load the daily price table and validate required columns."""
    if not path.exists():
        raise FileNotFoundError(f"Price file was not found: {path}")

    df = pd.read_parquet(path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError("Price file is missing required columns: " + ", ".join(missing_columns))

    return df.copy()


def normalize_input_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize types and sort by ticker and date."""
    cleaned = df.copy()
    cleaned["ticker"] = cleaned["ticker"].astype("string")
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    cleaned["adj_close"] = pd.to_numeric(cleaned["adj_close"], errors="coerce")
    cleaned["volume"] = pd.to_numeric(cleaned["volume"], errors="coerce")

    cleaned = cleaned.dropna(subset=["ticker", "date", "adj_close"]).copy()
    cleaned = cleaned.sort_values(["ticker", "date"]).reset_index(drop=True)
    return cleaned


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide safely and return NaN when the denominator is missing or zero."""
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")

    valid_denominator = denominator.notna() & np.isfinite(denominator) & (denominator != 0)
    result = pd.Series(np.nan, index=numerator.index, dtype="float64")
    result.loc[valid_denominator] = numerator.loc[valid_denominator] / denominator.loc[valid_denominator]
    return result


def compute_rsi(price_series: pd.Series, window: int = 14) -> pd.Series:
    """Compute a simple RSI using rolling average gains and losses."""
    delta = price_series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    average_gain = gains.rolling(window=window, min_periods=window).mean()
    average_loss = losses.rolling(window=window, min_periods=window).mean()
    relative_strength = safe_divide(average_gain, average_loss)

    rsi = 100 - (100 / (1 + relative_strength))
    return rsi


def engineer_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create backward-looking market features for each ticker."""
    feature_df = df.copy()

    daily_return = feature_df.groupby("ticker")["adj_close"].pct_change()

    # return_5d = adj_close_t / adj_close_t-5 - 1
    lagged_5d_price = feature_df.groupby("ticker")["adj_close"].shift(5)
    feature_df["return_5d"] = safe_divide(feature_df["adj_close"], lagged_5d_price) - 1.0

    # return_21d = adj_close_t / adj_close_t-21 - 1
    lagged_21d_price = feature_df.groupby("ticker")["adj_close"].shift(21)
    feature_df["return_21d"] = safe_divide(feature_df["adj_close"], lagged_21d_price) - 1.0

    # volatility_21d = rolling std of daily returns over the prior 21 trading days
    feature_df["volatility_21d"] = (
        daily_return.groupby(feature_df["ticker"])
        .rolling(window=21, min_periods=21)
        .std()
        .reset_index(level=0, drop=True)
    )

    # volume_ratio_20d = current volume / 20-day rolling average volume
    average_volume_20d = (
        feature_df.groupby("ticker")["volume"]
        .rolling(window=20, min_periods=20)
        .mean()
        .reset_index(level=0, drop=True)
    )
    feature_df["volume_ratio_20d"] = safe_divide(feature_df["volume"], average_volume_20d)

    # rsi_14 = 14-day relative strength index from adjusted close
    feature_df["rsi_14"] = (
        feature_df.groupby("ticker")["adj_close"]
        .apply(lambda series: compute_rsi(series, window=14))
        .reset_index(level=0, drop=True)
    )

    return feature_df


def calculate_missing_percentages(df: pd.DataFrame, columns: List[str]) -> pd.Series:
    """Return missing-value percentages for selected feature columns."""
    return df[columns].isna().mean().mul(100).sort_index()


def print_market_feature_summary(df: pd.DataFrame) -> None:
    """Print a compact summary of the engineered market features."""
    print("\nMarket Feature Summary")
    print("-" * 60)
    print(f"Number of rows: {len(df):,}")
    print(f"Number of tickers: {df['ticker'].nunique():,}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    missing_percentages = calculate_missing_percentages(df, MARKET_FEATURE_COLUMNS)
    print("\nMissingness for Layer 2 market features")
    print("-" * 60)
    for column_name, percentage in missing_percentages.items():
        print(f"{column_name:<24} {percentage:>8.2f}%")


def save_market_features(df: pd.DataFrame, output_path: Path) -> None:
    """Save the market feature table to parquet."""
    keep_columns = ["ticker", "date"] + MARKET_FEATURE_COLUMNS
    df[keep_columns].to_parquet(output_path, index=False)


def main() -> None:
    """Build the Layer 2 market feature dataset."""
    input_path = get_input_path()
    output_path = get_output_path()

    print(f"Loading price data from: {input_path}")
    price_df = load_price_data(input_path)

    print("Normalizing input data...")
    normalized_df = normalize_input_data(price_df)

    print("Engineering Layer 2 market features...")
    market_feature_df = engineer_market_features(normalized_df)

    print(f"Saving market features to: {output_path}")
    save_market_features(market_feature_df, output_path)

    print_market_feature_summary(market_feature_df)
    print("\nSaved Layer 2 market feature dataset.")


if __name__ == "__main__":
    main()
