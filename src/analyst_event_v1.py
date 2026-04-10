"""Build daily analyst-action features for the additive event_v1 lane.

Source schema assumption for the preferred dataset:
- ticker / symbol field: ``stock``
- timestamp field: ``date``
- action / rating / target-price fields: not structured; parsed from ``title``

Only deterministic title mappings are used. Headlines after 4:00 PM
America/New_York are made available on the next tradable panel date for that
ticker.
"""

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
    ANALYST_EVENT_FEATURE_COLUMNS,
    ANALYST_EVENT_V1_OUTPUT_PATH,
    ANALYST_INPUT_PATH,
    LAYER1_BASE_PANEL_PATH,
    ensure_event_v1_directories,
)

SOURCE_COLUMNS = ["title", "date", "stock"]
AFTER_CLOSE_HOUR = 16
FORWARD_ALIGNMENT_TOLERANCE_DAYS = 7
TARGET_TIMEZONE = "America/New_York"

POSITIVE_RATING_TERMS = (
    "buy",
    "outperform",
    "overweight",
    "positive",
    "accumulate",
)
NEUTRAL_RATING_TERMS = (
    "hold",
    "neutral",
    "equal weight",
    "market perform",
    "sector perform",
    "peer perform",
    "in line",
    "inline",
)
NEGATIVE_RATING_TERMS = (
    "sell",
    "underperform",
    "underweight",
    "negative",
    "reduce",
)


def parse_args() -> argparse.Namespace:
    """Parse CLI options for analyst feature generation."""
    parser = argparse.ArgumentParser(description="Build event_v1 analyst daily features.")
    parser.add_argument("--input-path", default=str(ANALYST_INPUT_PATH))
    parser.add_argument("--panel-path", default=str(LAYER1_BASE_PANEL_PATH))
    parser.add_argument("--output-path", default=str(ANALYST_EVENT_V1_OUTPUT_PATH))
    return parser.parse_args()


def load_panel_dates(path: Path) -> pd.DataFrame:
    """Load the locked Layer 1 panel dates for ticker-date alignment."""
    if not path.exists():
        raise FileNotFoundError(f"Layer 1 panel was not found: {path}")

    df = pd.read_parquet(path, columns=["ticker", "date"])
    df["ticker"] = df["ticker"].astype("string").str.upper()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["date"] = df["date"].astype("datetime64[ns]")
    df = df.dropna(subset=["ticker", "date"]).drop_duplicates().copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


def load_analyst_source(path: Path) -> pd.DataFrame:
    """Load the preferred analyst source with only required raw columns."""
    if not path.exists():
        raise FileNotFoundError(f"Analyst source file was not found: {path}")

    df = pd.read_csv(path, usecols=SOURCE_COLUMNS, dtype="string", low_memory=False)
    missing_columns = [column for column in SOURCE_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError("Analyst source is missing required columns: " + ", ".join(missing_columns))
    return df.rename(columns={"date": "source_date"}).copy()


def _normalize_rating_text(series: pd.Series) -> pd.Series:
    """Normalize extracted rating text into compact lowercase tokens."""
    normalized = series.astype("string").str.lower()
    normalized = normalized.str.replace(r"[^a-z0-9]+", " ", regex=True)
    normalized = normalized.str.replace(r"\s+", " ", regex=True).str.strip()
    return normalized


def _map_rating_sentiment(series: pd.Series) -> pd.Series:
    """Map normalized rating text to a small transparent sentiment scale."""
    sentiment = pd.Series(np.nan, index=series.index, dtype="float64")

    for term in NEGATIVE_RATING_TERMS:
        sentiment = sentiment.mask(series.str.contains(fr"\b{term}\b", na=False, regex=True), -1.0)
    for term in POSITIVE_RATING_TERMS:
        sentiment = sentiment.mask(
            sentiment.isna() & series.str.contains(fr"\b{term}\b", na=False, regex=True),
            1.0,
        )
    for term in NEUTRAL_RATING_TERMS:
        sentiment = sentiment.mask(
            sentiment.isna() & series.str.contains(fr"\b{term}\b", na=False, regex=True),
            0.0,
        )
    return sentiment


def normalize_analyst_source(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Parse direct analyst actions from title text and keep deterministic signals only."""
    prepared = df.copy()
    prepared["ticker"] = prepared["stock"].astype("string").str.upper().str.strip()
    prepared["title"] = prepared["title"].astype("string").str.strip()
    prepared["event_timestamp_utc"] = pd.to_datetime(prepared["source_date"], errors="coerce", utc=True)
    prepared["event_timestamp_local"] = prepared["event_timestamp_utc"].dt.tz_convert(TARGET_TIMEZONE)

    prepared = prepared.dropna(subset=["ticker", "title", "event_timestamp_local"]).copy()
    prepared = prepared.loc[prepared["ticker"] != ""].copy()
    prepared = prepared.sort_values(["ticker", "event_timestamp_local", "title"]).reset_index(drop=True)

    title_lower = prepared["title"].str.lower()

    upgrade_mask = title_lower.str.contains(r"\bupgrades?\b.+?\bto\b", na=False, regex=True)
    downgrade_mask = title_lower.str.contains(r"\bdowngrades?\b.+?\bto\b", na=False, regex=True)
    reiterate_mask = title_lower.str.contains(
        r"\b(?:maintains?|reiterates?|reaffirms?)\b",
        na=False,
        regex=True,
    )
    initiate_mask = title_lower.str.contains(
        r"\b(?:initiates?|assumes?|resumes?)\s+coverage\s+on\b.+?\bwith\b.+?\brating\b",
        na=False,
        regex=True,
    )

    action_type = pd.Series(pd.NA, index=prepared.index, dtype="string")
    action_type = action_type.mask(upgrade_mask, "upgrade")
    action_type = action_type.mask(downgrade_mask, "downgrade")
    action_type = action_type.mask(reiterate_mask, "reiterate")
    action_type = action_type.mask(initiate_mask, "initiate")
    prepared["action_type"] = action_type

    rating_from_upgrade = prepared["title"].str.extract(
        r"\b(?:Upgrades?|Downgrades?)\b.*?\bto\s+([^,.;]+)",
        expand=False,
    )
    rating_from_reiterate = prepared["title"].str.extract(
        r"\b(?:Maintains?|Reiterates?|Reaffirms?)\s+([^,.;]+?)\s+on\b",
        expand=False,
    )
    rating_from_initiate = prepared["title"].str.extract(
        r"\b(?:Initiates?|Assumes?|Resumes?)\s+Coverage\s+On\b.*?\bwith\s+([^,.;]+?)\s+Rating\b",
        expand=False,
    )
    prepared["rating_text"] = (
        rating_from_upgrade.fillna(rating_from_reiterate).fillna(rating_from_initiate).astype("string")
    )
    prepared["rating_text_normalized"] = _normalize_rating_text(prepared["rating_text"])
    prepared["sentiment_score"] = _map_rating_sentiment(prepared["rating_text_normalized"])
    prepared["sentiment_score"] = prepared["sentiment_score"].mask(
        prepared["sentiment_score"].isna() & (prepared["action_type"] == "upgrade"),
        1.0,
    )
    prepared["sentiment_score"] = prepared["sentiment_score"].mask(
        prepared["sentiment_score"].isna() & (prepared["action_type"] == "downgrade"),
        -1.0,
    )

    price_target_from = prepared["title"].str.extract(
        r"\bprice target from \$?([0-9]+(?:\.[0-9]+)?)\s+to \$?([0-9]+(?:\.[0-9]+)?)",
        expand=True,
    )
    prepared["price_target_old"] = pd.to_numeric(price_target_from[0], errors="coerce")
    prepared["price_target_new"] = pd.to_numeric(price_target_from[1], errors="coerce")

    pt_up_mask = title_lower.str.contains(
        r"\b(?:raises?|boosts?|lifts?)\s+price target\b",
        na=False,
        regex=True,
    )
    pt_down_mask = title_lower.str.contains(
        r"\b(?:lowers?|cuts?|reduces?)\s+price target\b",
        na=False,
        regex=True,
    )
    pt_up_mask = pt_up_mask | (
        prepared["price_target_old"].notna()
        & prepared["price_target_new"].notna()
        & (prepared["price_target_new"] > prepared["price_target_old"])
    )
    pt_down_mask = pt_down_mask | (
        prepared["price_target_old"].notna()
        & prepared["price_target_new"].notna()
        & (prepared["price_target_new"] < prepared["price_target_old"])
    )
    prepared["pt_up_flag"] = pt_up_mask.astype("int64")
    prepared["pt_down_flag"] = pt_down_mask.astype("int64")

    prepared["action_revision_score"] = 0.0
    prepared.loc[prepared["action_type"] == "upgrade", "action_revision_score"] = 1.0
    prepared.loc[prepared["action_type"] == "downgrade", "action_revision_score"] = -1.0
    prepared["pt_revision_score"] = prepared["pt_up_flag"] - prepared["pt_down_flag"]
    prepared["revision_score"] = prepared["action_revision_score"] + prepared["pt_revision_score"]

    prepared["event_available_date"] = prepared["event_timestamp_local"].dt.normalize().dt.tz_localize(None)
    after_close_mask = prepared["event_timestamp_local"].dt.hour >= AFTER_CLOSE_HOUR
    prepared.loc[after_close_mask, "event_available_date"] = (
        prepared.loc[after_close_mask, "event_available_date"] + pd.Timedelta(days=1)
    )
    prepared["event_available_date"] = pd.to_datetime(prepared["event_available_date"], errors="coerce")
    prepared["event_available_date"] = prepared["event_available_date"].astype("datetime64[ns]")

    direct_event_mask = (
        prepared["action_type"].notna()
        | prepared["pt_up_flag"].astype(bool)
        | prepared["pt_down_flag"].astype(bool)
    )
    prepared = prepared.loc[direct_event_mask].copy()
    prepared = prepared.drop_duplicates(
        subset=["ticker", "event_timestamp_utc", "title"],
        keep="first",
    ).copy()
    prepared = prepared.sort_values(["ticker", "event_available_date", "event_timestamp_local"]).reset_index(drop=True)

    source_timestamps = pd.to_datetime(df["source_date"], errors="coerce", utc=True)
    schema_summary = {
        "available_fields": {
            "ticker_symbol_field": "stock",
            "timestamp_date_field": "date",
            "action_rating_text_field": "title (derived only; no structured action/rating columns)",
            "sentiment_label_field": "none structured in processed source",
            "target_price_fields": "none structured in processed source",
        },
        "source_row_count": int(len(df)),
        "normalized_row_count": int(len(prepared)),
        "source_date_min": str(source_timestamps.min()),
        "source_date_max": str(source_timestamps.max()),
        "direct_event_date_min": str(prepared["event_available_date"].min()),
        "direct_event_date_max": str(prepared["event_available_date"].max()),
        "direct_event_ticker_count": int(prepared["ticker"].nunique()),
        "rating_text_rows": int(prepared["rating_text"].notna().sum()),
        "sentiment_rows": int(prepared["sentiment_score"].notna().sum()),
        "price_target_old_rows": int(prepared["price_target_old"].notna().sum()),
        "price_target_new_rows": int(prepared["price_target_new"].notna().sum()),
        "upgrade_rows": int((prepared["action_type"] == "upgrade").sum()),
        "downgrade_rows": int((prepared["action_type"] == "downgrade").sum()),
        "reiterate_rows": int((prepared["action_type"] == "reiterate").sum()),
        "initiate_rows": int((prepared["action_type"] == "initiate").sum()),
        "pt_up_rows": int(prepared["pt_up_flag"].sum()),
        "pt_down_rows": int(prepared["pt_down_flag"].sum()),
    }
    return prepared, schema_summary


def align_events_to_panel_dates(
    analyst_df: pd.DataFrame,
    panel_dates_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align events to tradable panel dates and recover last-known event history."""
    panel_dates = panel_dates_df.copy().sort_values(["date", "ticker"]).reset_index(drop=True)
    panel_dates["date"] = pd.to_datetime(panel_dates["date"], errors="coerce").astype("datetime64[ns]")
    panel_dates = panel_dates.rename(columns={"date": "panel_date"})

    analyst_events = analyst_df.copy().sort_values(["event_available_date", "ticker"]).reset_index(drop=True)
    analyst_events["event_available_date"] = pd.to_datetime(
        analyst_events["event_available_date"],
        errors="coerce",
    ).astype("datetime64[ns]")

    forward_aligned = pd.merge_asof(
        left=analyst_events,
        right=panel_dates.sort_values(["panel_date", "ticker"]).reset_index(drop=True),
        left_on="event_available_date",
        right_on="panel_date",
        by="ticker",
        direction="forward",
        allow_exact_matches=True,
        tolerance=pd.Timedelta(days=FORWARD_ALIGNMENT_TOLERANCE_DAYS),
    )
    forward_aligned = forward_aligned.dropna(subset=["panel_date"]).copy()
    forward_aligned = forward_aligned.rename(columns={"panel_date": "date"})

    history_aligned = pd.merge_asof(
        left=panel_dates.sort_values(["panel_date", "ticker"]).reset_index(drop=True),
        right=analyst_events[["ticker", "event_available_date"]]
        .sort_values(["event_available_date", "ticker"])
        .reset_index(drop=True),
        left_on="panel_date",
        right_on="event_available_date",
        by="ticker",
        direction="backward",
        allow_exact_matches=True,
    )
    history_aligned = history_aligned.rename(columns={"panel_date": "date"})
    return forward_aligned, history_aligned


def build_daily_analyst_features(
    aligned_events_df: pd.DataFrame,
    event_history_df: pd.DataFrame,
    panel_dates_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate aligned analyst events into daily event_v1 feature columns."""
    event_rows = aligned_events_df.copy()
    event_rows["daily_event_count"] = 1
    event_rows["daily_upgrade_count"] = (
        (event_rows["action_type"] == "upgrade").fillna(False).astype("int64")
    )
    event_rows["daily_downgrade_count"] = (
        (event_rows["action_type"] == "downgrade").fillna(False).astype("int64")
    )
    event_rows["daily_reiterate_count"] = (
        (event_rows["action_type"] == "reiterate").fillna(False).astype("int64")
    )
    event_rows["daily_sentiment_count"] = event_rows["sentiment_score"].notna().astype("int64")
    event_rows["daily_sentiment_sum"] = event_rows["sentiment_score"].fillna(0.0)
    event_rows["daily_sentiment_sq_sum"] = event_rows["sentiment_score"].fillna(0.0).pow(2)

    daily_aggregates = (
        event_rows.groupby(["ticker", "date"], as_index=False)
        .agg(
            daily_event_count=("daily_event_count", "sum"),
            daily_upgrade_count=("daily_upgrade_count", "sum"),
            daily_downgrade_count=("daily_downgrade_count", "sum"),
            daily_reiterate_count=("daily_reiterate_count", "sum"),
            daily_pt_up_count=("pt_up_flag", "sum"),
            daily_pt_down_count=("pt_down_flag", "sum"),
            daily_revision_score=("revision_score", "sum"),
            daily_sentiment_count=("daily_sentiment_count", "sum"),
            daily_sentiment_sum=("daily_sentiment_sum", "sum"),
            daily_sentiment_sq_sum=("daily_sentiment_sq_sum", "sum"),
        )
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    panel = panel_dates_df.copy().merge(
        daily_aggregates,
        on=["ticker", "date"],
        how="left",
        validate="one_to_one",
    )
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)

    fill_zero_columns = [
        "daily_event_count",
        "daily_upgrade_count",
        "daily_downgrade_count",
        "daily_reiterate_count",
        "daily_pt_up_count",
        "daily_pt_down_count",
        "daily_revision_score",
        "daily_sentiment_count",
        "daily_sentiment_sum",
        "daily_sentiment_sq_sum",
    ]
    for column in fill_zero_columns:
        panel[column] = pd.to_numeric(panel[column], errors="coerce").fillna(0.0)

    group = panel.groupby("ticker", group_keys=False)
    rolling_event_count_5d = (
        group["daily_event_count"].rolling(window=5, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    rolling_upgrade_count_5d = (
        group["daily_upgrade_count"].rolling(window=5, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    rolling_downgrade_count_5d = (
        group["daily_downgrade_count"].rolling(window=5, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    rolling_reiterate_count_5d = (
        group["daily_reiterate_count"].rolling(window=5, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    rolling_pt_up_count_5d = (
        group["daily_pt_up_count"].rolling(window=5, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    rolling_pt_down_count_5d = (
        group["daily_pt_down_count"].rolling(window=5, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    rolling_revision_score_5d = (
        group["daily_revision_score"].rolling(window=5, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    rolling_sentiment_count_5d = (
        group["daily_sentiment_count"].rolling(window=5, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    rolling_sentiment_sum_5d = (
        group["daily_sentiment_sum"].rolling(window=5, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    rolling_sentiment_sq_sum_5d = (
        group["daily_sentiment_sq_sum"].rolling(window=5, min_periods=1).sum().reset_index(level=0, drop=True)
    )

    panel["analyst_event_count_1d"] = panel["daily_event_count"]
    panel["analyst_event_count_5d"] = rolling_event_count_5d
    panel["analyst_upgrade_count_5d"] = rolling_upgrade_count_5d
    panel["analyst_downgrade_count_5d"] = rolling_downgrade_count_5d
    panel["analyst_reiterate_count_5d"] = rolling_reiterate_count_5d
    panel["analyst_pt_up_count_5d"] = rolling_pt_up_count_5d
    panel["analyst_pt_down_count_5d"] = rolling_pt_down_count_5d
    panel["analyst_net_revision_score_5d"] = rolling_revision_score_5d
    panel["analyst_mean_sentiment_1d"] = np.where(
        panel["daily_sentiment_count"] > 0,
        panel["daily_sentiment_sum"] / panel["daily_sentiment_count"],
        np.nan,
    )
    panel["analyst_mean_sentiment_5d"] = np.where(
        rolling_sentiment_count_5d > 0,
        rolling_sentiment_sum_5d / rolling_sentiment_count_5d,
        np.nan,
    )
    rolling_variance = np.where(
        rolling_sentiment_count_5d > 0,
        (rolling_sentiment_sq_sum_5d / rolling_sentiment_count_5d)
        - np.square(panel["analyst_mean_sentiment_5d"]),
        np.nan,
    )
    panel["analyst_sentiment_std_5d"] = np.where(
        rolling_sentiment_count_5d > 0,
        np.sqrt(np.clip(rolling_variance, a_min=0.0, a_max=None)),
        np.nan,
    )

    history = event_history_df[["ticker", "date", "event_available_date"]].copy()
    history = history.rename(columns={"event_available_date": "last_event_available_date"})
    panel = panel.merge(history, on=["ticker", "date"], how="left", validate="one_to_one")
    panel["analyst_days_since_event"] = (
        panel["date"] - pd.to_datetime(panel["last_event_available_date"], errors="coerce")
    ).dt.days.astype("float64")

    keep_columns = ["ticker", "date"] + ANALYST_EVENT_FEATURE_COLUMNS
    output = panel[keep_columns].copy()
    output = output.sort_values(["ticker", "date"]).reset_index(drop=True)
    return output


def print_schema_summary(schema_summary: dict) -> None:
    """Print what the processed analyst source actually exposes."""
    available_fields = schema_summary["available_fields"]

    print("\nAnalyst Source Schema")
    print("-" * 60)
    print(f"Ticker / symbol field: {available_fields['ticker_symbol_field']}")
    print(f"Timestamp / date field: {available_fields['timestamp_date_field']}")
    print(f"Action / rating text:   {available_fields['action_rating_text_field']}")
    print(f"Sentiment / label:      {available_fields['sentiment_label_field']}")
    print(f"Target price fields:    {available_fields['target_price_fields']}")

    print("\nAnalyst Source Coverage")
    print("-" * 60)
    print(f"Raw source rows:             {schema_summary['source_row_count']:,}")
    print(f"Direct event rows kept:      {schema_summary['normalized_row_count']:,}")
    print(f"Source date range:           {schema_summary['source_date_min']} to {schema_summary['source_date_max']}")
    print(
        "Direct event availability:   "
        f"{schema_summary['direct_event_date_min']} to {schema_summary['direct_event_date_max']}"
    )
    print(f"Direct event tickers:        {schema_summary['direct_event_ticker_count']:,}")
    print(f"Rows with parsed rating:     {schema_summary['rating_text_rows']:,}")
    print(f"Rows with sentiment score:   {schema_summary['sentiment_rows']:,}")
    print(f"Rows with old PT parsed:     {schema_summary['price_target_old_rows']:,}")
    print(f"Rows with new PT parsed:     {schema_summary['price_target_new_rows']:,}")
    print(f"Upgrade rows:                {schema_summary['upgrade_rows']:,}")
    print(f"Downgrade rows:              {schema_summary['downgrade_rows']:,}")
    print(f"Reiterate rows:              {schema_summary['reiterate_rows']:,}")
    print(f"Initiate rows:               {schema_summary['initiate_rows']:,}")
    print(f"PT up rows:                  {schema_summary['pt_up_rows']:,}")
    print(f"PT down rows:                {schema_summary['pt_down_rows']:,}")


def print_feature_summary(df: pd.DataFrame) -> None:
    """Print a compact summary of the aligned analyst daily feature table."""
    missingness = df[ANALYST_EVENT_FEATURE_COLUMNS].isna().mean().mul(100).sort_values(ascending=False)

    print("\nEvent V1 Analyst Daily Feature Summary")
    print("-" * 60)
    print(f"Rows: {len(df):,}")
    print(f"Tickers: {df['ticker'].nunique():,}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(
        "Rows with same-day analyst events: "
        f"{int((df['analyst_event_count_1d'] > 0).sum()):,}"
    )

    print("\nTop missingness columns")
    print("-" * 60)
    for column_name, percentage in missingness.head(12).items():
        print(f"{column_name:<30} {percentage:>8.2f}%")


def main() -> None:
    """Build and save daily analyst features for the event_v1 lane."""
    args = parse_args()
    ensure_event_v1_directories()

    panel_path = Path(args.panel_path)
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    print(f"Loading Layer 1 panel dates from: {panel_path}")
    panel_dates_df = load_panel_dates(panel_path)

    print(f"Loading analyst source from: {input_path}")
    raw_analyst_df = load_analyst_source(input_path)

    print("Normalizing analyst titles into direct event rows...")
    analyst_events_df, schema_summary = normalize_analyst_source(raw_analyst_df)
    print_schema_summary(schema_summary)

    print("Aligning analyst events to tradable panel dates...")
    aligned_events_df, event_history_df = align_events_to_panel_dates(
        analyst_df=analyst_events_df,
        panel_dates_df=panel_dates_df,
    )

    print("Building daily analyst feature table...")
    feature_df = build_daily_analyst_features(
        aligned_events_df=aligned_events_df,
        event_history_df=event_history_df,
        panel_dates_df=panel_dates_df,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_parquet(output_path, index=False)

    print(f"Saving analyst daily features to: {output_path}")
    print_feature_summary(feature_df)
    print("\nSaved event_v1 analyst features.")


if __name__ == "__main__":
    main()
