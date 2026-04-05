"""Prepare Layer 3 SEC filing sentiment features for modeling.

This script takes the raw filing-level FinBERT output and turns it into a
cleaner Layer 3 feature table. The output stays at one row per filing, with
each row representing the sentiment information that became public on that
filing date.

This file does not merge sentiment onto the daily panel yet. It only creates
the filing-level Layer 3 features that will later be aligned to daily dates
without look-ahead leakage.

Input:
    data/interim/sentiment/sec_filing_sentiment.parquet

Output:
    data/interim/features/layer3_sec_sentiment_features.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.paths import INTERIM_DATA_DIR

REQUIRED_COLUMNS = [
    "ticker",
    "cik",
    "company_name",
    "form_type",
    "filing_date",
    "period_end",
    "accession_number",
    "positive_prob",
    "negative_prob",
    "neutral_prob",
    "sentiment_score",
    "chunk_count",
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


def get_input_path() -> Path:
    """Return the raw filing-level sentiment parquet path."""
    return INTERIM_DATA_DIR / "sentiment" / "sec_filing_sentiment.parquet"


def get_output_path() -> Path:
    """Return the clean Layer 3 feature output path and create its folder."""
    output_dir = INTERIM_DATA_DIR / "features"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "layer3_sec_sentiment_features.parquet"


def load_sentiment_data(path: Path) -> pd.DataFrame:
    """Load the filing-level sentiment data and validate required columns."""
    if not path.exists():
        raise FileNotFoundError(
            f"SEC filing sentiment file was not found: {path}\n"
            "Run `python -m src.sec_sentiment_features` first."
        )

    df = pd.read_parquet(path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "SEC filing sentiment file is missing required columns: "
            + ", ".join(missing_columns)
        )

    return df.copy()


def normalize_input_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize types and sort filings within each ticker."""
    cleaned = df.copy()
    cleaned["ticker"] = cleaned["ticker"].astype("string")
    cleaned["cik"] = cleaned["cik"].astype("string")
    cleaned["company_name"] = cleaned["company_name"].astype("string")
    cleaned["form_type"] = cleaned["form_type"].astype("string")
    cleaned["accession_number"] = cleaned["accession_number"].astype("string")

    cleaned["filing_date"] = pd.to_datetime(cleaned["filing_date"], errors="coerce")
    cleaned["period_end"] = pd.to_datetime(cleaned["period_end"], errors="coerce")

    numeric_columns = [
        "positive_prob",
        "negative_prob",
        "neutral_prob",
        "sentiment_score",
        "chunk_count",
    ]
    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.dropna(subset=["ticker", "filing_date", "accession_number"]).copy()
    cleaned = cleaned.drop_duplicates(subset=["ticker", "accession_number"]).copy()
    cleaned = cleaned.sort_values(["ticker", "filing_date", "form_type"]).reset_index(drop=True)
    return cleaned


def engineer_layer3_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create simple filing-level Layer 3 sentiment features.

    The main idea is to keep the continuous FinBERT outputs and add a few
    readable features that measure:
    - direction of tone
    - strength of tone
    - change versus the prior filing
    - filing type
    """
    feature_df = df.copy()

    # Continuous filing sentiment score from FinBERT:
    # sec_sentiment_score = positive_prob - negative_prob
    feature_df["sec_sentiment_score"] = feature_df["sentiment_score"]

    # Direct class probabilities from FinBERT.
    feature_df["sec_positive_prob"] = feature_df["positive_prob"]
    feature_df["sec_negative_prob"] = feature_df["negative_prob"]
    feature_df["sec_neutral_prob"] = feature_df["neutral_prob"]

    # Absolute sentiment measures how strong the tone is, regardless of sign.
    feature_df["sec_sentiment_abs"] = feature_df["sec_sentiment_score"].abs()

    # Filing-to-filing change in overall sentiment tone for the same ticker.
    feature_df["sec_sentiment_change_prev"] = (
        feature_df.groupby("ticker")["sec_sentiment_score"].diff()
    )

    # Filing-to-filing change in the positive and negative probabilities.
    feature_df["sec_positive_change_prev"] = (
        feature_df.groupby("ticker")["sec_positive_prob"].diff()
    )
    feature_df["sec_negative_change_prev"] = (
        feature_df.groupby("ticker")["sec_negative_prob"].diff()
    )

    # Chunk count is a rough proxy for filing text size after chunking.
    feature_df["sec_chunk_count"] = feature_df["chunk_count"]

    # Log chunk count compresses very large filing-length differences.
    feature_df["sec_log_chunk_count"] = np.log1p(feature_df["sec_chunk_count"])

    # One-hot-style filing type indicators.
    feature_df["sec_is_10k"] = (feature_df["form_type"] == "10-K").astype("int64")
    feature_df["sec_is_10q"] = (feature_df["form_type"] == "10-Q").astype("int64")

    return feature_df


def build_output_table(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the filing metadata plus engineered Layer 3 features."""
    keep_columns = [
        "ticker",
        "cik",
        "company_name",
        "form_type",
        "filing_date",
        "period_end",
        "accession_number",
    ] + LAYER3_FEATURE_COLUMNS

    output_df = df[keep_columns].copy()
    output_df = output_df.sort_values(["ticker", "filing_date", "form_type"]).reset_index(drop=True)
    return output_df


def calculate_missing_percentages(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Return missing-value percentages for selected feature columns."""
    return df[columns].isna().mean().mul(100).sort_index()


def print_summary(df: pd.DataFrame) -> None:
    """Print a compact summary of the prepared Layer 3 feature table."""
    print("\nLayer 3 SEC Sentiment Feature Summary")
    print("-" * 60)
    print(f"Number of filings: {len(df):,}")
    print(f"Number of tickers: {df['ticker'].nunique():,}")
    print(f"Date range: {df['filing_date'].min().date()} to {df['filing_date'].max().date()}")

    print("\nMissingness for Layer 3 features")
    print("-" * 60)
    missing_percentages = calculate_missing_percentages(df, LAYER3_FEATURE_COLUMNS)
    for column_name, percentage in missing_percentages.items():
        print(f"{column_name:<28} {percentage:>8.2f}%")

    print("\nAverage Layer 3 feature values")
    print("-" * 60)
    print(f"sec_sentiment_score:       {df['sec_sentiment_score'].mean():.4f}")
    print(f"sec_sentiment_abs:         {df['sec_sentiment_abs'].mean():.4f}")
    print(f"sec_positive_prob:         {df['sec_positive_prob'].mean():.4f}")
    print(f"sec_negative_prob:         {df['sec_negative_prob'].mean():.4f}")
    print(f"sec_neutral_prob:          {df['sec_neutral_prob'].mean():.4f}")
    print(f"sec_chunk_count:           {df['sec_chunk_count'].mean():.2f}")

    print("\nFeature Notes")
    print("-" * 60)
    print(
        "These features stay at the filing level for now. "
        "They will later be aligned to daily dates using filing_date so "
        "the model only sees sentiment after it becomes public."
    )
    print(
        "The dominant sentiment label from full SEC filings was not kept as a "
        "model feature because it had no variation in the first FinBERT pass."
    )


def save_output(df: pd.DataFrame, output_path: Path) -> None:
    """Save the prepared Layer 3 feature table to parquet."""
    df.to_parquet(output_path, index=False)


def main() -> None:
    """Build the clean filing-level Layer 3 SEC sentiment feature table."""
    input_path = get_input_path()
    output_path = get_output_path()

    print(f"Loading SEC filing sentiment from: {input_path}")
    sentiment_df = load_sentiment_data(input_path)

    print("Normalizing filing-level sentiment data...")
    normalized_df = normalize_input_data(sentiment_df)

    print("Engineering Layer 3 SEC sentiment features...")
    featured_df = engineer_layer3_features(normalized_df)

    output_df = build_output_table(featured_df)

    print(f"Saving Layer 3 SEC sentiment features to: {output_path}")
    save_output(output_df, output_path)

    print_summary(output_df)
    print("\nSaved Layer 3 SEC sentiment feature dataset.")


if __name__ == "__main__":
    main()
