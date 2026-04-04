"""Download daily prices and build the future-return label table.

This script supports Layer 1 by creating the target label needed later for
supervised learning. It does not create Layer 2 market features. It only:

- downloads daily OHLCV price data
- computes 5-day forward return from adjusted close
- creates the binary label

Input:
    No local input file required. Prices are downloaded from Yahoo Finance.

Output:
    data/interim/prices/prices_with_labels.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.config import END_DATE, START_DATE
from src.paths import INTERIM_DATA_DIR
from src.universe import get_layer1_tickers

PRICE_COLUMNS = [
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "forward_return_5d",
    "label",
]


def get_output_path() -> Path:
    """Return the output parquet path and create its folder."""
    output_dir = INTERIM_DATA_DIR / "prices"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "prices_with_labels.parquet"


def get_download_end_date() -> str:
    """Extend the download window so the last in-sample rows can get labels.

    The project horizon ends on 2024-12-31, but a 5-day forward return needs a
    few trading days after that date. A small calendar buffer is enough.
    """
    return (pd.Timestamp(END_DATE) + pd.Timedelta(days=14)).strftime("%Y-%m-%d")


def download_prices(tickers: list[str]) -> pd.DataFrame:
    """Download daily prices for the Layer 1 universe."""
    if not tickers:
        raise ValueError("Ticker list is empty.")

    raw = yf.download(
        tickers=tickers,
        start=START_DATE,
        end=get_download_end_date(),
        auto_adjust=False,
        progress=False,
        group_by="column",
    )

    if raw.empty:
        raise ValueError("Price download returned no data.")

    return raw


def reshape_downloaded_prices(raw_prices: pd.DataFrame) -> pd.DataFrame:
    """Convert the yfinance download output into a simple long table."""
    if isinstance(raw_prices.columns, pd.MultiIndex):
        long_df = raw_prices.stack(level=1, future_stack=True).reset_index()
        long_df = long_df.rename(
            columns={
                "Date": "date",
                "Ticker": "ticker",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
    else:
        long_df = raw_prices.reset_index().copy()
        long_df["ticker"] = get_layer1_tickers()[0]
        long_df = long_df.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )

    long_df["date"] = pd.to_datetime(long_df["date"], errors="coerce")
    long_df["ticker"] = long_df["ticker"].astype("string")

    numeric_columns = ["open", "high", "low", "close", "adj_close", "volume"]
    for column in numeric_columns:
        if column in long_df.columns:
            long_df[column] = pd.to_numeric(long_df[column], errors="coerce")

    long_df = long_df.dropna(subset=["ticker", "date", "adj_close"]).copy()
    long_df = long_df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return long_df


def build_labels(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute 5-day forward returns and binary labels.

    Formula:
        forward_return_5d = adj_close_t+5 / adj_close_t - 1
        label = 1 if forward_return_5d > 0 else 0
    """
    labeled = price_df.copy()

    future_adj_close = labeled.groupby("ticker")["adj_close"].shift(-5)
    labeled["forward_return_5d"] = (future_adj_close / labeled["adj_close"]) - 1.0

    labeled["label"] = np.where(
        labeled["forward_return_5d"].notna(),
        (labeled["forward_return_5d"] > 0).astype(int),
        np.nan,
    )
    labeled["label"] = pd.Series(labeled["label"], index=labeled.index).astype("Int64")

    in_sample_mask = (
        labeled["date"] >= pd.Timestamp(START_DATE)
    ) & (
        labeled["date"] <= pd.Timestamp(END_DATE)
    )
    labeled = labeled.loc[in_sample_mask].copy()

    return labeled


def print_price_summary(df: pd.DataFrame) -> None:
    """Print a small summary of the price and label dataset."""
    print("\nPrice Label Summary")
    print("-" * 60)
    print(f"Number of rows: {len(df):,}")
    print(f"Number of tickers: {df['ticker'].nunique():,}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Rows with label available: {df['label'].notna().sum():,}")
    print(f"Rows missing label: {df['label'].isna().sum():,}")


def save_prices(df: pd.DataFrame, output_path: Path) -> None:
    """Save the daily price and label table to parquet."""
    df[PRICE_COLUMNS].to_parquet(output_path, index=False)


def main() -> None:
    """Download prices and create the Layer 1 label table."""
    tickers = get_layer1_tickers()
    output_path = get_output_path()

    print(f"Downloading daily prices for {len(tickers)} tickers...")
    raw_prices = download_prices(tickers)

    print("Reshaping downloaded prices...")
    price_df = reshape_downloaded_prices(raw_prices)

    print("Computing 5-day forward returns and labels...")
    labeled_df = build_labels(price_df)

    print(f"Saving price and label dataset to: {output_path}")
    save_prices(labeled_df, output_path)

    print_price_summary(labeled_df)
    print("\nSaved daily prices with labels for later panel alignment.")


if __name__ == "__main__":
    main()
