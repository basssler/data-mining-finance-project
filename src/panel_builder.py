"""Build the final Layer 1 daily modeling panel.

This script combines:
- daily price/label data
- filing-aligned Layer 1 fundamentals features

The main rule is simple: for each ticker and trading date, only use the most
recent fundamentals row whose filing_date is on or before that trading date.
That prevents future information leakage.

Input:
    data/interim/prices/prices_with_labels.parquet
    data/interim/features/layer1_financial_features.parquet

Output:
    data/processed/modeling/layer1_modeling_panel.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.paths import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

PRICE_REQUIRED_COLUMNS = [
    "ticker",
    "date",
    "forward_return_5d",
    "label",
]

FEATURE_REQUIRED_COLUMNS = [
    "ticker",
    "filing_date",
    "period_end",
]


def get_price_input_path() -> Path:
    """Return the daily price/label input path."""
    return INTERIM_DATA_DIR / "prices" / "prices_with_labels.parquet"


def get_feature_input_path() -> Path:
    """Return the Layer 1 feature input path."""
    return INTERIM_DATA_DIR / "features" / "layer1_financial_features.parquet"


def get_output_path() -> Path:
    """Return the final modeling panel output path and create its folder."""
    output_dir = PROCESSED_DATA_DIR / "modeling"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "layer1_modeling_panel.parquet"


def load_parquet(path: Path, required_columns: list[str], dataset_name: str) -> pd.DataFrame:
    """Load a parquet file and validate a minimal required schema."""
    if not path.exists():
        raise FileNotFoundError(f"{dataset_name} file was not found: {path}")

    df = pd.read_parquet(path)
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"{dataset_name} file is missing required columns: " + ", ".join(missing_columns)
        )

    return df.copy()


def prepare_prices(price_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and sort the daily price/label table."""
    prepared = price_df.copy()
    prepared["ticker"] = prepared["ticker"].astype("string")
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    prepared = prepared.dropna(subset=["ticker", "date"]).copy()
    prepared = prepared.sort_values(["date", "ticker"]).reset_index(drop=True)
    return prepared


def prepare_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and sort the fundamentals feature table."""
    prepared = feature_df.copy()
    prepared["ticker"] = prepared["ticker"].astype("string")
    prepared["filing_date"] = pd.to_datetime(prepared["filing_date"], errors="coerce")
    prepared["period_end"] = pd.to_datetime(prepared["period_end"], errors="coerce")
    prepared = prepared.dropna(subset=["ticker", "filing_date"]).copy()
    prepared = prepared.sort_values(["filing_date", "ticker", "period_end"]).reset_index(drop=True)
    return prepared


def build_panel(price_df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    """Attach the latest available fundamentals row to each daily price row.

    `merge_asof` performs the leakage-safe alignment:
    - match on ticker
    - use the latest fundamentals row with filing_date <= trading date
    """
    panel = pd.merge_asof(
        left=price_df,
        right=feature_df,
        left_on="date",
        right_on="filing_date",
        by="ticker",
        direction="backward",
        allow_exact_matches=True,
    )

    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    return panel


def filter_modeling_rows(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows that are usable for modeling.

    A row is only usable once the ticker has at least one public fundamentals
    filing available. Rows before the first filing are dropped here.
    """
    filtered = panel_df.copy()
    filtered = filtered.dropna(subset=["filing_date", "label"]).copy()
    filtered = filtered.sort_values(["ticker", "date"]).reset_index(drop=True)
    return filtered


def print_panel_summary(raw_panel_df: pd.DataFrame, final_panel_df: pd.DataFrame) -> None:
    """Print a simple alignment summary for sanity checking."""
    print("\nPanel Summary")
    print("-" * 60)
    print(f"Daily rows before fundamentals alignment filter: {len(raw_panel_df):,}")
    print(f"Daily rows in final modeling panel:          {len(final_panel_df):,}")
    print(f"Number of tickers:                           {final_panel_df['ticker'].nunique():,}")
    print(f"Date range:                                  {final_panel_df['date'].min().date()} to {final_panel_df['date'].max().date()}")
    print(f"Rows dropped before first filing:            {len(raw_panel_df) - len(final_panel_df):,}")
    print(f"Rows with label available:                   {final_panel_df['label'].notna().sum():,}")


def save_panel(df: pd.DataFrame, output_path: Path) -> None:
    """Save the final modeling panel to parquet."""
    df.to_parquet(output_path, index=False)


def main() -> None:
    """Build the final Layer 1 daily modeling panel."""
    price_input_path = get_price_input_path()
    feature_input_path = get_feature_input_path()
    output_path = get_output_path()

    print(f"Loading price labels from: {price_input_path}")
    price_df = load_parquet(price_input_path, PRICE_REQUIRED_COLUMNS, "Price label")

    print(f"Loading Layer 1 features from: {feature_input_path}")
    feature_df = load_parquet(feature_input_path, FEATURE_REQUIRED_COLUMNS, "Feature")

    print("Preparing price and fundamentals tables...")
    prepared_prices = prepare_prices(price_df)
    prepared_features = prepare_features(feature_df)

    print("Aligning fundamentals to daily dates without look-ahead leakage...")
    raw_panel_df = build_panel(prepared_prices, prepared_features)

    print("Dropping daily rows that occur before the first available filing...")
    final_panel_df = filter_modeling_rows(raw_panel_df)

    print(f"Saving final modeling panel to: {output_path}")
    save_panel(final_panel_df, output_path)

    print_panel_summary(raw_panel_df, final_panel_df)
    print("\nSaved final Layer 1 modeling panel.")


if __name__ == "__main__":
    main()
