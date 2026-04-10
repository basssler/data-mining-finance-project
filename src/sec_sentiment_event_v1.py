"""Build daily event-driven SEC sentiment features for the event_v1 lane."""

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
    DEFAULT_SENTIMENT_SOURCE,
    LAYER1_BASE_PANEL_PATH,
    LAYER3_EVENT_FEATURE_COLUMNS,
    SEC_FILING_METADATA_V1_PATH,
    SENTIMENT_EVENT_V1_OUTPUT_PATH,
    ensure_event_v1_directories,
    get_sentiment_input_path,
)

SOURCE_COLUMN_MAP = {
    "full": {
        "score": "sec_sentiment_score",
        "positive": "sec_positive_prob",
        "negative": "sec_negative_prob",
        "neutral": "sec_neutral_prob",
    },
    "mda": {
        "score": "mda_sentiment_score",
        "positive": "mda_positive_prob",
        "negative": "mda_negative_prob",
        "neutral": "mda_neutral_prob",
    },
}


def parse_args() -> argparse.Namespace:
    """Parse CLI options for the event_v1 sentiment builder."""
    parser = argparse.ArgumentParser(description="Build event_v1 SEC sentiment features.")
    parser.add_argument(
        "--sentiment-source",
        choices=sorted(SOURCE_COLUMN_MAP),
        default=DEFAULT_SENTIMENT_SOURCE,
    )
    return parser.parse_args()


def load_panel_dates(path: Path) -> pd.DataFrame:
    """Load the locked Layer 1 panel and keep only ticker/date for alignment."""
    if not path.exists():
        raise FileNotFoundError(f"Layer 1 panel was not found: {path}")

    df = pd.read_parquet(path, columns=["ticker", "date"])
    df["ticker"] = df["ticker"].astype("string")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").astype("datetime64[ns]")
    df = df.dropna(subset=["ticker", "date"]).drop_duplicates().copy()
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    return df


def load_sentiment_data(path: Path, sentiment_source: str) -> pd.DataFrame:
    """Load the chosen filing-level sentiment table."""
    if not path.exists():
        raise FileNotFoundError(f"Sentiment feature file was not found: {path}")

    df = pd.read_parquet(path)
    column_map = SOURCE_COLUMN_MAP[sentiment_source]
    required_columns = [
        "ticker",
        "filing_date",
        column_map["score"],
        column_map["positive"],
        column_map["negative"],
        column_map["neutral"],
    ]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError("Sentiment file is missing required columns: " + ", ".join(missing_columns))
    return df.copy()


def normalize_sentiment_data(df: pd.DataFrame, sentiment_source: str) -> pd.DataFrame:
    """Map source-specific columns to a common filing-level schema."""
    column_map = SOURCE_COLUMN_MAP[sentiment_source]

    prepared = df.copy()
    prepared["ticker"] = prepared["ticker"].astype("string")
    prepared["filing_date"] = pd.to_datetime(prepared["filing_date"], errors="coerce").astype(
        "datetime64[ns]"
    )
    prepared["sentiment_score"] = pd.to_numeric(prepared[column_map["score"]], errors="coerce")
    prepared["positive_prob"] = pd.to_numeric(prepared[column_map["positive"]], errors="coerce")
    prepared["negative_prob"] = pd.to_numeric(prepared[column_map["negative"]], errors="coerce")
    prepared["neutral_prob"] = pd.to_numeric(prepared[column_map["neutral"]], errors="coerce")

    prepared = prepared.dropna(subset=["ticker", "filing_date", "sentiment_score"]).copy()
    prepared = prepared.sort_values(["ticker", "filing_date"]).reset_index(drop=True)
    return prepared[
        [
            "ticker",
            "filing_date",
            "sentiment_score",
            "positive_prob",
            "negative_prob",
            "neutral_prob",
        ]
    ].copy()


def load_sec_timing_metadata(path: Path) -> pd.DataFrame | None:
    """Load SEC timing metadata for filing-level effective-date alignment."""
    if not path.exists():
        return None

    metadata = pd.read_parquet(
        path,
        columns=["ticker", "accession_number", "effective_model_date"],
    )
    metadata["ticker"] = metadata["ticker"].astype("string")
    metadata["accession_number"] = metadata["accession_number"].astype("string")
    metadata["effective_model_date"] = pd.to_datetime(
        metadata["effective_model_date"],
        errors="coerce",
    ).astype("datetime64[ns]")
    metadata = metadata.dropna(
        subset=["ticker", "accession_number", "effective_model_date"]
    ).copy()
    metadata = metadata.sort_values(
        ["ticker", "accession_number", "effective_model_date"]
    ).drop_duplicates(subset=["ticker", "accession_number"])
    return metadata.reset_index(drop=True)


def attach_effective_model_dates(
    sentiment_df: pd.DataFrame,
    panel_dates_df: pd.DataFrame,
    timing_metadata_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Attach a tradable effective date to each filing-level sentiment row.

    Matched filings reuse the SEC metadata `effective_model_date`. Unmatched
    rows fall back to the next tradable date after `filing_date`, which keeps
    ambiguous timestamps conservative.
    """
    featured = sentiment_df.copy()

    if "accession_number" in featured.columns:
        featured["accession_number"] = featured["accession_number"].astype("string")
    else:
        featured["accession_number"] = pd.Series(pd.NA, index=featured.index, dtype="string")

    featured["availability_base_date"] = featured["filing_date"] + pd.Timedelta(days=1)
    if timing_metadata_df is not None and not timing_metadata_df.empty:
        featured = featured.merge(
            timing_metadata_df,
            on=["ticker", "accession_number"],
            how="left",
            validate="many_to_one",
        )
        featured["availability_base_date"] = featured["effective_model_date"].fillna(
            featured["availability_base_date"]
        )
    else:
        featured["effective_model_date"] = pd.NaT

    featured["availability_base_date"] = pd.to_datetime(
        featured["availability_base_date"],
        errors="coerce",
    ).astype("datetime64[ns]")

    panel_dates = panel_dates_df.copy().rename(columns={"date": "panel_date"})
    panel_dates["panel_date"] = pd.to_datetime(
        panel_dates["panel_date"],
        errors="coerce",
    ).astype("datetime64[ns]")
    panel_dates = panel_dates.sort_values(["panel_date", "ticker"]).reset_index(drop=True)

    aligned = pd.merge_asof(
        left=featured.sort_values(["availability_base_date", "ticker"]).reset_index(drop=True),
        right=panel_dates,
        left_on="availability_base_date",
        right_on="panel_date",
        by="ticker",
        direction="forward",
        allow_exact_matches=True,
    )
    aligned["effective_model_date"] = pd.to_datetime(aligned["panel_date"], errors="coerce")
    aligned["effective_model_date"] = aligned["effective_model_date"].astype("datetime64[ns]")
    aligned = aligned.dropna(subset=["effective_model_date"]).copy()
    aligned = aligned.drop(columns=["availability_base_date", "panel_date"])
    aligned = aligned.sort_values(
        ["ticker", "effective_model_date", "filing_date"]
    ).reset_index(drop=True)
    return aligned


def build_sec_sentiment_event_v1(
    sentiment_df: pd.DataFrame,
    ticker_col: str = "ticker",
    filing_date_col: str = "filing_date",
    score_col: str = "sentiment_score",
    panel_dates_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create daily event-aware sentiment features from past filings only."""
    if panel_dates_df is None:
        raise ValueError("panel_dates_df is required to align event_v1 sentiment daily.")

    featured_filings = sentiment_df.copy()
    featured_filings["sec_event_score_latest"] = featured_filings[score_col]
    featured_filings["sec_event_abs_latest"] = featured_filings[score_col].abs()
    featured_filings["sec_event_delta_prev"] = (
        featured_filings.groupby(ticker_col)[score_col].diff()
    )
    featured_filings["sec_event_abs_delta_prev"] = featured_filings["sec_event_delta_prev"].abs()

    previous_score = featured_filings.groupby(ticker_col)[score_col].shift(1)
    featured_filings["sec_event_neg_to_pos_flip"] = (
        (previous_score < 0) & (featured_filings[score_col] > 0)
    ).astype("int64")
    featured_filings["sec_event_pos_to_neg_flip"] = (
        (previous_score > 0) & (featured_filings[score_col] < 0)
    ).astype("int64")

    max_probability = featured_filings[
        ["positive_prob", "negative_prob", "neutral_prob"]
    ].max(axis=1)
    featured_filings["sec_event_uncertainty"] = 1.0 - max_probability
    featured_filings = attach_effective_model_dates(
        sentiment_df=featured_filings,
        panel_dates_df=panel_dates_df,
        timing_metadata_df=load_sec_timing_metadata(SEC_FILING_METADATA_V1_PATH),
    )

    aligned = pd.merge_asof(
        left=panel_dates_df.sort_values(["date", "ticker"]).reset_index(drop=True),
        right=featured_filings[
            [
                ticker_col,
                filing_date_col,
                "effective_model_date",
                "sec_event_score_latest",
                "sec_event_abs_latest",
                "sec_event_delta_prev",
                "sec_event_abs_delta_prev",
                "sec_event_neg_to_pos_flip",
                "sec_event_pos_to_neg_flip",
                "sec_event_uncertainty",
            ]
        ]
        .sort_values(["effective_model_date", ticker_col])
        .reset_index(drop=True),
        left_on="date",
        right_on="effective_model_date",
        by=ticker_col,
        direction="backward",
        allow_exact_matches=True,
    )

    days_since_filing = (
        aligned["date"] - pd.to_datetime(aligned[filing_date_col], errors="coerce")
    ).dt.days
    aligned["sec_event_days_since_filing"] = pd.to_numeric(days_since_filing, errors="coerce")
    aligned["sec_event_decay_30d"] = np.where(
        aligned["sec_event_days_since_filing"].notna(),
        np.exp(-aligned["sec_event_days_since_filing"] / 30.0),
        np.nan,
    )
    aligned["sec_event_score_decayed"] = (
        aligned["sec_event_score_latest"] * aligned["sec_event_decay_30d"]
    )

    keep_columns = ["ticker", "date"] + LAYER3_EVENT_FEATURE_COLUMNS
    output_df = aligned[keep_columns].copy()
    output_df = output_df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return output_df


def print_summary(df: pd.DataFrame) -> None:
    """Print a compact summary of the daily event-driven sentiment table."""
    available_rows = df["sec_event_score_latest"].notna().sum()

    print("\nEvent V1 SEC Sentiment Summary")
    print("-" * 60)
    print(f"Rows: {len(df):,}")
    print(f"Tickers: {df['ticker'].nunique():,}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Rows with filing history available: {available_rows:,}")
    print(f"Rows missing event sentiment:       {len(df) - available_rows:,}")

    print("\nMissingness")
    print("-" * 60)
    missingness = df[LAYER3_EVENT_FEATURE_COLUMNS].isna().mean().mul(100).sort_values(ascending=False)
    for column_name, percentage in missingness.items():
        print(f"{column_name:<28} {percentage:>8.2f}%")


def main() -> None:
    """Build and save daily event-driven sentiment features."""
    args = parse_args()
    ensure_event_v1_directories()

    sentiment_input_path = get_sentiment_input_path(args.sentiment_source)

    print(f"Loading Layer 1 panel dates from: {LAYER1_BASE_PANEL_PATH}")
    panel_dates_df = load_panel_dates(LAYER1_BASE_PANEL_PATH)

    print(f"Loading filing-level sentiment from: {sentiment_input_path}")
    raw_sentiment_df = load_sentiment_data(sentiment_input_path, args.sentiment_source)
    normalized_sentiment_df = normalize_sentiment_data(raw_sentiment_df, args.sentiment_source)

    print("Building daily event-driven sentiment features...")
    featured_df = build_sec_sentiment_event_v1(
        normalized_sentiment_df,
        panel_dates_df=panel_dates_df,
    )

    print(f"Saving event-driven sentiment features to: {SENTIMENT_EVENT_V1_OUTPUT_PATH}")
    featured_df.to_parquet(SENTIMENT_EVENT_V1_OUTPUT_PATH, index=False)

    print_summary(featured_df)
    print("\nSaved event_v1 SEC sentiment features.")


if __name__ == "__main__":
    main()
