"""Build a daily modeling panel that adds Layer 3 SEC sentiment features.

This script starts from the locked Layer 1 daily modeling panel and merges the
prepared SEC filing sentiment features onto it. The merge rule is the same
leakage-safe rule used for fundamentals:

- match by ticker
- for each trading date, use the most recent filing whose filing_date is on or
  before that date

This keeps the original Layer 1 panel intact while creating a separate panel
for Layer 3 experiments.

Inputs:
    data/processed/modeling/layer1_modeling_panel.parquet
    data/interim/features/layer3_sec_sentiment_features.parquet

Output:
    data/processed/modeling/layer1_layer3_modeling_panel.parquet
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

LAYER1_PANEL_REQUIRED_COLUMNS = [
    "ticker",
    "date",
    "forward_return_5d",
    "label",
]

LAYER3_SENTIMENT_REQUIRED_COLUMNS = [
    "ticker",
    "filing_date",
    "accession_number",
]

LAYER3_FEATURE_COLUMNS = [
    "sec_sentiment_score",
    "sec_positive_prob",
    "sec_negative_prob",
    "sec_neutral_prob",
    "sec_sentiment_abs",
    "sec_sentiment_change_prev",
    "sec_positive_change_prev",
    "sec_negative_change_prev",
    "sec_chunk_count",
    "sec_log_chunk_count",
    "sec_is_10k",
    "sec_is_10q",
]


def get_layer1_panel_path() -> Path:
    """Return the locked Layer 1 daily modeling panel path."""
    return PROCESSED_DATA_DIR / "modeling" / "layer1_modeling_panel.parquet"


def get_layer3_feature_path() -> Path:
    """Return the prepared Layer 3 SEC sentiment feature path."""
    return INTERIM_DATA_DIR / "features" / "layer3_sec_sentiment_features.parquet"


def get_output_path() -> Path:
    """Return the Layer 1 + Layer 3 panel output path and create its folder."""
    output_dir = PROCESSED_DATA_DIR / "modeling"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "layer1_layer3_modeling_panel.parquet"


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


def prepare_layer1_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and sort the locked Layer 1 panel."""
    prepared = df.copy()
    prepared["ticker"] = prepared["ticker"].astype("string")
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    prepared = prepared.dropna(subset=["ticker", "date"]).copy()
    prepared = prepared.sort_values(["date", "ticker"]).reset_index(drop=True)
    return prepared


def prepare_layer3_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and sort the filing-level Layer 3 feature table."""
    prepared = df.copy()
    prepared["ticker"] = prepared["ticker"].astype("string")
    prepared["filing_date"] = pd.to_datetime(prepared["filing_date"], errors="coerce")

    for column in LAYER3_FEATURE_COLUMNS:
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    prepared = prepared.dropna(subset=["ticker", "filing_date"]).copy()
    prepared = prepared.sort_values(["filing_date", "ticker", "accession_number"]).reset_index(drop=True)
    return prepared


def build_panel(layer1_df: pd.DataFrame, layer3_df: pd.DataFrame) -> pd.DataFrame:
    """Attach the latest available SEC sentiment filing to each daily row."""
    panel = pd.merge_asof(
        left=layer1_df,
        right=layer3_df[
            ["ticker", "filing_date", "accession_number", "form_type"] + LAYER3_FEATURE_COLUMNS
        ],
        left_on="date",
        right_on="filing_date",
        by="ticker",
        direction="backward",
        allow_exact_matches=True,
    )

    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    return panel


def print_summary(df: pd.DataFrame) -> None:
    """Print a compact summary of the Layer 1 + Layer 3 daily panel."""
    print("\nLayer 1 + Layer 3 Panel Summary")
    print("-" * 60)
    print(f"Number of rows: {len(df):,}")
    print(f"Number of tickers: {df['ticker'].nunique():,}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    sentiment_available = df["sec_sentiment_score"].notna().sum()
    print(f"Rows with sentiment available: {sentiment_available:,}")
    print(f"Rows missing sentiment:        {len(df) - sentiment_available:,}")

    print("\nMissingness for Layer 3 columns in daily panel")
    print("-" * 60)
    missing_percentages = df[LAYER3_FEATURE_COLUMNS].isna().mean().mul(100).sort_index()
    for column_name, percentage in missing_percentages.items():
        print(f"{column_name:<28} {percentage:>8.2f}%")


def save_panel(df: pd.DataFrame, output_path: Path) -> None:
    """Save the Layer 1 + Layer 3 panel to parquet."""
    df.to_parquet(output_path, index=False)


def main() -> None:
    """Build the daily modeling panel that includes Layer 3 SEC sentiment."""
    layer1_path = get_layer1_panel_path()
    layer3_path = get_layer3_feature_path()
    output_path = get_output_path()

    print(f"Loading Layer 1 panel from: {layer1_path}")
    layer1_df = load_parquet(layer1_path, LAYER1_PANEL_REQUIRED_COLUMNS, "Layer 1 panel")

    print(f"Loading Layer 3 SEC sentiment features from: {layer3_path}")
    layer3_df = load_parquet(
        layer3_path,
        LAYER3_SENTIMENT_REQUIRED_COLUMNS + LAYER3_FEATURE_COLUMNS,
        "Layer 3 sentiment feature",
    )

    print("Preparing Layer 1 and Layer 3 tables...")
    prepared_layer1 = prepare_layer1_panel(layer1_df)
    prepared_layer3 = prepare_layer3_features(layer3_df)

    print("Aligning SEC sentiment to daily dates without look-ahead leakage...")
    panel_df = build_panel(prepared_layer1, prepared_layer3)

    print(f"Saving Layer 1 + Layer 3 panel to: {output_path}")
    save_panel(panel_df, output_path)

    print_summary(panel_df)
    print("\nSaved Layer 1 + Layer 3 modeling panel.")


if __name__ == "__main__":
    main()
