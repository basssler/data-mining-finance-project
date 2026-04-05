"""Prepare MD&A-only SEC sentiment features for modeling.

This script takes the MD&A-only FinBERT output and turns it into a clean
Layer 3 feature table. The output stays at one row per successfully scored
filing.

Reasonable assumption:
only filings with successfully extracted and scored MD&A are treated as
sentiment updates. Unscored filings are not used to overwrite the last known
MD&A sentiment state in the later daily merge.

Input:
    data/interim/sentiment/sec_filing_sentiment_mda.parquet

Output:
    data/interim/features/layer3_sec_sentiment_mda_features.parquet
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
    "mda_found",
    "positive_prob",
    "negative_prob",
    "neutral_prob",
    "sentiment_score",
    "chunk_count",
    "mda_text_length",
    "extraction_status",
]

LAYER3_MDA_FEATURE_COLUMNS = [
    "mda_sentiment_score",
    "mda_positive_prob",
    "mda_negative_prob",
    "mda_neutral_prob",
    "mda_sentiment_abs",
    "mda_sentiment_change_prev",
    "mda_positive_change_prev",
    "mda_negative_change_prev",
    "mda_chunk_count",
    "mda_log_chunk_count",
    "mda_text_length",
    "mda_log_text_length",
    "mda_is_10k",
    "mda_is_10q",
]


def get_input_path() -> Path:
    """Return the raw MD&A filing-level sentiment parquet path."""
    return INTERIM_DATA_DIR / "sentiment" / "sec_filing_sentiment_mda.parquet"


def get_output_path() -> Path:
    """Return the clean MD&A feature output path and create its folder."""
    output_dir = INTERIM_DATA_DIR / "features"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "layer3_sec_sentiment_mda_features.parquet"


def load_sentiment_data(path: Path) -> pd.DataFrame:
    """Load MD&A filing sentiment data and validate required columns."""
    if not path.exists():
        raise FileNotFoundError(
            f"MD&A SEC filing sentiment file was not found: {path}\n"
            "Run `python -m src.sec_sentiment_features_mda` first."
        )

    df = pd.read_parquet(path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "MD&A SEC filing sentiment file is missing required columns: "
            + ", ".join(missing_columns)
        )

    return df.copy()


def normalize_input_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dtypes and keep only successfully scored MD&A filings."""
    cleaned = df.copy()
    cleaned["ticker"] = cleaned["ticker"].astype("string")
    cleaned["cik"] = cleaned["cik"].astype("string")
    cleaned["company_name"] = cleaned["company_name"].astype("string")
    cleaned["form_type"] = cleaned["form_type"].astype("string")
    cleaned["accession_number"] = cleaned["accession_number"].astype("string")
    cleaned["extraction_status"] = cleaned["extraction_status"].astype("string")

    cleaned["filing_date"] = pd.to_datetime(cleaned["filing_date"], errors="coerce")
    cleaned["period_end"] = pd.to_datetime(cleaned["period_end"], errors="coerce")

    numeric_columns = [
        "mda_found",
        "positive_prob",
        "negative_prob",
        "neutral_prob",
        "sentiment_score",
        "chunk_count",
        "mda_text_length",
    ]
    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.dropna(subset=["ticker", "filing_date", "accession_number"]).copy()
    cleaned = cleaned.drop_duplicates(subset=["ticker", "accession_number"]).copy()

    cleaned = cleaned[cleaned["mda_found"] == 1].copy()
    cleaned = cleaned.sort_values(["ticker", "filing_date", "form_type"]).reset_index(drop=True)
    return cleaned


def engineer_mda_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create filing-level MD&A sentiment features."""
    feature_df = df.copy()

    feature_df["mda_sentiment_score"] = feature_df["sentiment_score"]
    feature_df["mda_positive_prob"] = feature_df["positive_prob"]
    feature_df["mda_negative_prob"] = feature_df["negative_prob"]
    feature_df["mda_neutral_prob"] = feature_df["neutral_prob"]

    feature_df["mda_sentiment_abs"] = feature_df["mda_sentiment_score"].abs()

    feature_df["mda_sentiment_change_prev"] = (
        feature_df.groupby("ticker")["mda_sentiment_score"].diff()
    )
    feature_df["mda_positive_change_prev"] = (
        feature_df.groupby("ticker")["mda_positive_prob"].diff()
    )
    feature_df["mda_negative_change_prev"] = (
        feature_df.groupby("ticker")["mda_negative_prob"].diff()
    )

    feature_df["mda_chunk_count"] = feature_df["chunk_count"]
    feature_df["mda_log_chunk_count"] = np.log1p(feature_df["mda_chunk_count"])
    feature_df["mda_log_text_length"] = np.log1p(feature_df["mda_text_length"])

    feature_df["mda_is_10k"] = (feature_df["form_type"] == "10-K").astype("int64")
    feature_df["mda_is_10q"] = (feature_df["form_type"] == "10-Q").astype("int64")

    return feature_df


def build_output_table(df: pd.DataFrame) -> pd.DataFrame:
    """Keep filing metadata plus engineered MD&A features."""
    keep_columns = [
        "ticker",
        "cik",
        "company_name",
        "form_type",
        "filing_date",
        "period_end",
        "accession_number",
        "extraction_status",
    ] + LAYER3_MDA_FEATURE_COLUMNS

    output_df = df[keep_columns].copy()
    output_df = output_df.sort_values(["ticker", "filing_date", "form_type"]).reset_index(drop=True)
    return output_df


def calculate_missing_percentages(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Return missing-value percentages for selected MD&A feature columns."""
    return df[columns].isna().mean().mul(100).sort_index()


def print_summary(df: pd.DataFrame) -> None:
    """Print a compact summary of the prepared MD&A feature table."""
    print("\nLayer 3 MD&A Sentiment Feature Summary")
    print("-" * 60)
    print(f"Number of scored MD&A filings: {len(df):,}")
    print(f"Number of tickers:             {df['ticker'].nunique():,}")
    print(f"Date range:                   {df['filing_date'].min().date()} to {df['filing_date'].max().date()}")

    print("\nMissingness for MD&A Layer 3 features")
    print("-" * 60)
    missing_percentages = calculate_missing_percentages(df, LAYER3_MDA_FEATURE_COLUMNS)
    for column_name, percentage in missing_percentages.items():
        print(f"{column_name:<28} {percentage:>8.2f}%")

    print("\nAverage MD&A Layer 3 feature values")
    print("-" * 60)
    print(f"mda_sentiment_score:       {df['mda_sentiment_score'].mean():.4f}")
    print(f"mda_sentiment_abs:         {df['mda_sentiment_abs'].mean():.4f}")
    print(f"mda_positive_prob:         {df['mda_positive_prob'].mean():.4f}")
    print(f"mda_negative_prob:         {df['mda_negative_prob'].mean():.4f}")
    print(f"mda_neutral_prob:          {df['mda_neutral_prob'].mean():.4f}")
    print(f"mda_chunk_count:           {df['mda_chunk_count'].mean():.2f}")
    print(f"mda_text_length:           {df['mda_text_length'].mean():.0f}")

    print("\nFeature Notes")
    print("-" * 60)
    print(
        "Only filings with successfully extracted and scored MD&A are kept in this table. "
        "These rows will be carried forward from filing_date onto daily dates later."
    )


def save_output(df: pd.DataFrame, output_path: Path) -> None:
    """Save the prepared MD&A feature table to parquet."""
    df.to_parquet(output_path, index=False)


def main() -> None:
    """Build the clean filing-level MD&A sentiment feature table."""
    input_path = get_input_path()
    output_path = get_output_path()

    print(f"Loading MD&A SEC filing sentiment from: {input_path}")
    sentiment_df = load_sentiment_data(input_path)

    print("Normalizing MD&A filing-level sentiment data...")
    normalized_df = normalize_input_data(sentiment_df)

    print("Engineering Layer 3 MD&A sentiment features...")
    featured_df = engineer_mda_features(normalized_df)

    output_df = build_output_table(featured_df)

    print(f"Saving Layer 3 MD&A sentiment features to: {output_path}")
    save_output(output_df, output_path)

    print_summary(output_df)
    print("\nSaved Layer 3 MD&A sentiment feature dataset.")


if __name__ == "__main__":
    main()
