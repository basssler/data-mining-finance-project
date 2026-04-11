"""Build the event_panel_v2 research panel around filing events.

Phase 2 intentionally changes the observation unit only:
- one row = one ticker-event
- event rows come from SEC 10-Q / 10-K filings already present in the repo
- no labels, model changes, or validation changes are introduced here

The panel keeps the old daily/event_v1 artifacts untouched and writes new
artifacts to:
- data/interim/event_panel_v2.parquet
- docs/event_panel_spec_v2.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.config_event_v1 import (
    EVENT_V1_LAYER1_PANEL_PATH,
    FULL_SENTIMENT_INPUT_PATH,
    LAYER1_BASE_PANEL_PATH,
    LAYER1_FEATURE_COLUMNS,
    LAYER2_V2_FEATURE_COLUMNS,
    PRICE_INPUT_PATH,
)
from src.panel_builder import (
    SEC_FILING_METADATA_PATH,
    attach_effective_model_dates as attach_fundamental_effective_dates,
    load_sec_timing_metadata,
    prepare_features,
    prepare_prices,
)
from src.paths import DOCS_DIR, INTERIM_DATA_DIR

EVENT_PANEL_V2_OUTPUT_PATH = INTERIM_DATA_DIR / "event_panel_v2.parquet"
EVENT_PANEL_V2_SPEC_PATH = DOCS_DIR / "event_panel_spec_v2.md"
FUNDAMENTALS_INPUT_PATH = INTERIM_DATA_DIR / "features" / "layer1_financial_features.parquet"
MARKET_INPUT_PATH = INTERIM_DATA_DIR / "features" / "layer2_market_features_v2.parquet"

EVENT_REQUIRED_COLUMNS = [
    "ticker",
    "cik",
    "company_name",
    "form_type",
    "filing_date",
    "filing_timestamp_utc",
    "filing_timestamp_local",
    "timing_bucket",
    "availability_base_date",
    "effective_model_date",
    "accession_number",
]

FUNDAMENTALS_REQUIRED_COLUMNS = [
    "ticker",
    "filing_date",
    "period_end",
    "fiscal_period",
    "fiscal_year",
    "form_type",
] + LAYER1_FEATURE_COLUMNS

MARKET_REQUIRED_COLUMNS = ["ticker", "date"] + LAYER2_V2_FEATURE_COLUMNS

SENTIMENT_CONTEXT_COLUMNS = [
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
]

SENTIMENT_REQUIRED_COLUMNS = [
    "ticker",
    "form_type",
    "filing_date",
    "accession_number",
] + SENTIMENT_CONTEXT_COLUMNS

EVENT_CONTEXT_FEATURE_COLUMNS = [
    "days_since_prior_event",
    "days_since_prior_same_event_type",
]

FUNDAMENTAL_SNAPSHOT_METADATA_COLUMNS = [
    "fund_snapshot_filing_date",
    "fund_snapshot_period_end",
    "fund_snapshot_fiscal_period",
    "fund_snapshot_fiscal_year",
    "fund_snapshot_event_type",
    "fund_snapshot_effective_model_date",
]

EXACT_EVENT_METADATA_COLUMNS = [
    "event_period_end",
    "event_fiscal_period",
    "event_fiscal_year",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the event_panel_v2 build."""
    parser = argparse.ArgumentParser(description="Build the event_panel_v2 research panel.")
    parser.add_argument("--price-path", default=str(PRICE_INPUT_PATH))
    parser.add_argument("--event-source-path", default=str(SEC_FILING_METADATA_PATH))
    parser.add_argument("--fundamentals-path", default=str(FUNDAMENTALS_INPUT_PATH))
    parser.add_argument("--market-path", default=str(MARKET_INPUT_PATH))
    parser.add_argument("--sentiment-path", default=str(FULL_SENTIMENT_INPUT_PATH))
    parser.add_argument("--output-path", default=str(EVENT_PANEL_V2_OUTPUT_PATH))
    parser.add_argument("--spec-path", default=str(EVENT_PANEL_V2_SPEC_PATH))
    return parser.parse_args()


def ensure_parent_dir(path: Path) -> None:
    """Create the parent folder for one output path."""
    path.parent.mkdir(parents=True, exist_ok=True)


def load_parquet(path: Path, required_columns: list[str], dataset_name: str) -> pd.DataFrame:
    """Load one parquet file and validate a minimal schema."""
    if not path.exists():
        raise FileNotFoundError(f"{dataset_name} file was not found: {path}")

    df = pd.read_parquet(path)
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"{dataset_name} file is missing required columns: " + ", ".join(missing_columns)
        )
    return df.copy()


def load_event_source(path: Path) -> pd.DataFrame:
    """Load timing-aligned SEC filing metadata and keep 10-Q / 10-K events only."""
    df = load_parquet(path, EVENT_REQUIRED_COLUMNS, "SEC filing metadata v1")
    prepared = df.copy()
    prepared["ticker"] = prepared["ticker"].astype("string")
    prepared["cik"] = prepared["cik"].astype("string")
    prepared["company_name"] = prepared["company_name"].astype("string")
    prepared["form_type"] = prepared["form_type"].astype("string").str.upper().str.strip()
    prepared["filing_date"] = pd.to_datetime(prepared["filing_date"], errors="coerce")
    prepared["filing_timestamp_utc"] = pd.to_datetime(
        prepared["filing_timestamp_utc"],
        errors="coerce",
        utc=True,
    )
    prepared["filing_timestamp_local"] = pd.to_datetime(
        prepared["filing_timestamp_local"],
        errors="coerce",
    )
    prepared["effective_model_date"] = pd.to_datetime(
        prepared["effective_model_date"],
        errors="coerce",
    ).astype("datetime64[ns]")
    prepared["availability_base_date"] = pd.to_datetime(
        prepared["availability_base_date"],
        errors="coerce",
    ).astype("datetime64[ns]")
    prepared["accession_number"] = prepared["accession_number"].astype("string")
    prepared["timing_bucket"] = prepared["timing_bucket"].astype("string")

    prepared = prepared.loc[prepared["form_type"].isin(["10-Q", "10-K"])].copy()
    prepared = prepared.dropna(
        subset=["ticker", "form_type", "filing_date", "effective_model_date", "accession_number"]
    ).copy()
    prepared = prepared.drop_duplicates(subset=["ticker", "accession_number"]).copy()
    prepared = prepared.sort_values(
        ["effective_model_date", "ticker", "form_type", "accession_number"]
    ).reset_index(drop=True)
    return prepared


def align_event_effective_dates(event_source_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    """Re-align event effective dates to actual trading dates using stored timing bases."""
    panel_dates = prices_df[["ticker", "date"]].drop_duplicates().rename(columns={"date": "panel_date"}).copy()
    panel_dates["panel_date"] = pd.to_datetime(panel_dates["panel_date"], errors="coerce").astype(
        "datetime64[ns]"
    )
    panel_dates = panel_dates.sort_values(["panel_date", "ticker"]).reset_index(drop=True)

    aligned = pd.merge_asof(
        left=event_source_df.sort_values(["availability_base_date", "ticker"]).reset_index(drop=True),
        right=panel_dates,
        left_on="availability_base_date",
        right_on="panel_date",
        by="ticker",
        direction="forward",
        allow_exact_matches=True,
        tolerance=pd.Timedelta(days=7),
    )
    aligned = aligned.dropna(subset=["panel_date"]).copy()
    aligned["effective_model_date"] = aligned["panel_date"].astype("datetime64[ns]")
    aligned = aligned.drop(columns=["panel_date"])
    aligned = aligned.sort_values(
        ["effective_model_date", "ticker", "form_type", "accession_number"]
    ).reset_index(drop=True)
    return aligned


def build_base_event_panel(event_source_df: pd.DataFrame) -> pd.DataFrame:
    """Create the base event table with one row per ticker-event."""
    panel = event_source_df.rename(
        columns={
            "form_type": "event_type",
            "filing_date": "event_date",
            "filing_timestamp_local": "event_timestamp",
            "accession_number": "source_id",
        }
    ).copy()
    panel = panel[
        [
            "ticker",
            "cik",
            "company_name",
            "event_type",
            "event_date",
            "event_timestamp",
            "filing_timestamp_utc",
            "effective_model_date",
            "timing_bucket",
            "source_id",
        ]
    ].copy()

    panel["days_since_prior_event"] = (
        panel.groupby("ticker")["effective_model_date"].diff().dt.days.astype("float64")
    )
    panel["days_since_prior_same_event_type"] = (
        panel.groupby(["ticker", "event_type"])["effective_model_date"]
        .diff()
        .dt.days.astype("float64")
    )

    panel = panel.sort_values(["effective_model_date", "ticker", "event_type"]).reset_index(drop=True)
    return panel


def load_fundamentals(path: Path) -> pd.DataFrame:
    """Load the filing-level fundamentals table."""
    df = load_parquet(path, FUNDAMENTALS_REQUIRED_COLUMNS, "Layer 1 fundamentals")
    prepared = prepare_features(df)
    for column in LAYER1_FEATURE_COLUMNS:
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    prepared["fiscal_period"] = prepared["fiscal_period"].astype("string")
    prepared["fiscal_year"] = pd.to_numeric(prepared["fiscal_year"], errors="coerce").astype("Int64")
    return prepared


def attach_event_fundamentals_metadata(
    panel_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach exact-match current filing metadata for the event itself."""
    exact_lookup = (
        fundamentals_df[
            ["ticker", "form_type", "filing_date", "period_end", "fiscal_period", "fiscal_year"]
        ]
        .sort_values(["ticker", "form_type", "filing_date", "period_end"])
        .drop_duplicates(subset=["ticker", "form_type", "filing_date"], keep="last")
        .rename(
            columns={
                "form_type": "event_type",
                "filing_date": "event_date",
                "period_end": "event_period_end",
                "fiscal_period": "event_fiscal_period",
                "fiscal_year": "event_fiscal_year",
            }
        )
    )

    merged = panel_df.merge(
        exact_lookup,
        on=["ticker", "event_type", "event_date"],
        how="left",
        validate="many_to_one",
    )
    merged["current_filing_fundamentals_available"] = merged["event_period_end"].notna()
    return merged


def attach_fundamental_snapshot(
    panel_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame,
    prices_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the latest valid fundamentals snapshot as of the event's tradable date."""
    aligned_fundamentals = attach_fundamental_effective_dates(
        price_df=prices_df,
        feature_df=fundamentals_df,
        timing_metadata_df=load_sec_timing_metadata(SEC_FILING_METADATA_PATH),
    ).copy()

    aligned_fundamentals = aligned_fundamentals.rename(
        columns={
            "filing_date": "fund_snapshot_filing_date",
            "period_end": "fund_snapshot_period_end",
            "fiscal_period": "fund_snapshot_fiscal_period",
            "fiscal_year": "fund_snapshot_fiscal_year",
            "form_type": "fund_snapshot_event_type",
            "effective_model_date": "fund_snapshot_effective_model_date",
        }
    )

    snapshot_columns = (
        ["ticker", "fund_snapshot_effective_model_date"]
        + FUNDAMENTAL_SNAPSHOT_METADATA_COLUMNS[:-1]
        + LAYER1_FEATURE_COLUMNS
    )
    aligned_fundamentals = aligned_fundamentals[snapshot_columns].copy()
    aligned_fundamentals = aligned_fundamentals.sort_values(
        ["fund_snapshot_effective_model_date", "ticker", "fund_snapshot_period_end"]
    ).reset_index(drop=True)

    panel = pd.merge_asof(
        left=panel_df.sort_values(["effective_model_date", "ticker"]).reset_index(drop=True),
        right=aligned_fundamentals,
        left_on="effective_model_date",
        right_on="fund_snapshot_effective_model_date",
        by="ticker",
        direction="backward",
        allow_exact_matches=True,
    )

    panel["fund_snapshot_is_current_event"] = (
        (panel["fund_snapshot_filing_date"] == panel["event_date"])
        & (panel["fund_snapshot_event_type"] == panel["event_type"])
    ).fillna(False)
    return panel


def load_market_features(path: Path) -> pd.DataFrame:
    """Load the precomputed daily market-control features."""
    df = load_parquet(path, MARKET_REQUIRED_COLUMNS, "Layer 2 market features v2")
    prepared = df.copy()
    prepared["ticker"] = prepared["ticker"].astype("string")
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce").astype("datetime64[ns]")
    for column in LAYER2_V2_FEATURE_COLUMNS:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    prepared = prepared.dropna(subset=["ticker", "date"]).copy()
    prepared = prepared.sort_values(["date", "ticker"]).reset_index(drop=True)
    return prepared


def attach_market_snapshot(panel_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    """Attach market features from the prior trading day only."""
    market_snapshot = market_df.rename(columns={"date": "market_asof_date"}).copy()
    market_snapshot = market_snapshot.sort_values(["market_asof_date", "ticker"]).reset_index(drop=True)

    panel = pd.merge_asof(
        left=panel_df.sort_values(["effective_model_date", "ticker"]).reset_index(drop=True),
        right=market_snapshot,
        left_on="effective_model_date",
        right_on="market_asof_date",
        by="ticker",
        direction="backward",
        allow_exact_matches=False,
    )
    return panel


def load_sentiment_context(path: Path) -> pd.DataFrame:
    """Load filing-level SEC sentiment features for 10-Q / 10-K events."""
    df = load_parquet(path, SENTIMENT_REQUIRED_COLUMNS, "Layer 3 filing sentiment features")
    prepared = df.copy()
    prepared["ticker"] = prepared["ticker"].astype("string")
    prepared["form_type"] = prepared["form_type"].astype("string").str.upper().str.strip()
    prepared["filing_date"] = pd.to_datetime(prepared["filing_date"], errors="coerce")
    prepared["accession_number"] = prepared["accession_number"].astype("string")
    for column in SENTIMENT_CONTEXT_COLUMNS:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    prepared = prepared.loc[prepared["form_type"].isin(["10-Q", "10-K"])].copy()
    prepared = prepared.dropna(subset=["ticker", "accession_number"]).copy()
    prepared = prepared.sort_values(["ticker", "accession_number", "filing_date"]).reset_index(drop=True)
    return prepared


def attach_sentiment_context(panel_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """Attach same-filing sentiment context by accession number."""
    sentiment_lookup = sentiment_df.rename(
        columns={
            "accession_number": "source_id",
            "filing_date": "sentiment_filing_date",
            "form_type": "sentiment_event_type",
        }
    ).copy()

    panel = panel_df.merge(
        sentiment_lookup[
            ["ticker", "source_id", "sentiment_filing_date", "sentiment_event_type"]
            + SENTIMENT_CONTEXT_COLUMNS
        ],
        on=["ticker", "source_id"],
        how="left",
        validate="one_to_one",
    )
    panel["current_filing_sentiment_available"] = panel["sec_sentiment_score"].notna()
    return panel


def finalize_column_order(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Place the most important identity and timing columns first."""
    leading_columns = [
        "ticker",
        "cik",
        "company_name",
        "event_type",
        "event_date",
        "event_timestamp",
        "filing_timestamp_utc",
        "effective_model_date",
        "timing_bucket",
        "source_id",
        "event_period_end",
        "event_fiscal_period",
        "event_fiscal_year",
        "current_filing_fundamentals_available",
        "current_filing_sentiment_available",
        "days_since_prior_event",
        "days_since_prior_same_event_type",
        "fund_snapshot_filing_date",
        "fund_snapshot_period_end",
        "fund_snapshot_fiscal_period",
        "fund_snapshot_fiscal_year",
        "fund_snapshot_event_type",
        "fund_snapshot_effective_model_date",
        "fund_snapshot_is_current_event",
        "market_asof_date",
    ]
    ordered = [column for column in leading_columns if column in panel_df.columns]
    ordered += [
        column
        for column in EXACT_EVENT_METADATA_COLUMNS
        if column in panel_df.columns and column not in ordered
    ]
    ordered += [
        column
        for column in LAYER1_FEATURE_COLUMNS + LAYER2_V2_FEATURE_COLUMNS + SENTIMENT_CONTEXT_COLUMNS
        if column in panel_df.columns and column not in ordered
    ]
    ordered += [column for column in panel_df.columns if column not in ordered]
    return panel_df[ordered].copy()


def validate_panel_structure(
    panel_df: pd.DataFrame,
    trading_dates_df: pd.DataFrame | None = None,
) -> None:
    """Run structural checks that should hold for the event-based design."""
    if panel_df.empty:
        raise ValueError("event_panel_v2 is empty.")

    duplicate_count = int(panel_df.duplicated(subset=["ticker", "source_id"]).sum())
    if duplicate_count > 0:
        raise ValueError(f"event_panel_v2 has duplicate ticker/source_id rows: {duplicate_count}")

    if panel_df["effective_model_date"].isna().any():
        raise ValueError("event_panel_v2 contains rows without effective_model_date.")

    invalid_market_rows = panel_df["market_asof_date"].notna() & (
        panel_df["market_asof_date"] >= panel_df["effective_model_date"]
    )
    if invalid_market_rows.any():
        raise ValueError("Market features are not strictly prior to effective_model_date for all rows.")

    invalid_fundamental_rows = panel_df["fund_snapshot_effective_model_date"].notna() & (
        panel_df["fund_snapshot_effective_model_date"] > panel_df["effective_model_date"]
    )
    if invalid_fundamental_rows.any():
        raise ValueError("Fundamental snapshots extend past effective_model_date for some rows.")

    local_timestamp_date = pd.Series(pd.NaT, index=panel_df.index, dtype="datetime64[ns]")
    timestamp_mask = panel_df["event_timestamp"].notna()
    if timestamp_mask.any():
        local_timestamp_date.loc[timestamp_mask] = (
            panel_df.loc[timestamp_mask, "event_timestamp"].dt.tz_localize(None).dt.normalize()
        )

    after_close_same_day = (
        panel_df["timing_bucket"].eq("after_close")
        & local_timestamp_date.notna()
        & (panel_df["effective_model_date"] <= local_timestamp_date)
    )
    if after_close_same_day.any():
        raise ValueError("After-close events were not shifted beyond the raw event_date.")

    same_day_candidate_mask = (
        panel_df["timing_bucket"].isin(["pre_market", "market_hours"])
        & local_timestamp_date.notna()
    )
    same_day_timing_invalid = same_day_candidate_mask & (
        panel_df["effective_model_date"] != local_timestamp_date
    )
    if same_day_timing_invalid.any() and trading_dates_df is not None:
        trading_dates = trading_dates_df[["ticker", "date"]].drop_duplicates().copy()
        trading_dates["ticker"] = trading_dates["ticker"].astype("string")
        trading_dates["date"] = pd.to_datetime(trading_dates["date"], errors="coerce").astype(
            "datetime64[ns]"
        )
        same_day_lookup = pd.DataFrame(
            {
                "ticker": panel_df.loc[same_day_timing_invalid, "ticker"].astype("string"),
                "timestamp_trading_date": local_timestamp_date.loc[same_day_timing_invalid],
            }
        )
        same_day_lookup = same_day_lookup.merge(
            trading_dates.rename(columns={"date": "timestamp_trading_date"}),
            on=["ticker", "timestamp_trading_date"],
            how="left",
            indicator=True,
        )
        has_same_day_session = same_day_lookup["_merge"].eq("both").to_numpy()
        same_day_invalid_index = panel_df.index[same_day_timing_invalid]
        allowed_non_trading_day_rows = pd.Series(False, index=panel_df.index)
        allowed_non_trading_day_rows.loc[same_day_invalid_index] = ~has_same_day_session
        same_day_timing_invalid = same_day_timing_invalid & ~allowed_non_trading_day_rows
    if same_day_timing_invalid.any():
        raise ValueError("Pre-market or market-hours events are not aligned to same-day exposure.")

    missing_time_invalid = (
        panel_df["timing_bucket"].eq("missing_time_conservative_next_day")
        & (panel_df["effective_model_date"] <= panel_df["event_date"])
    )
    if missing_time_invalid.any():
        raise ValueError("Missing timestamps are not being handled conservatively.")


def _format_scalar(value: object) -> str:
    """Format one value for a markdown table."""
    if pd.isna(value):
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is not None:
            return value.strftime("%Y-%m-%d %H:%M:%S%z")
        if value.hour == 0 and value.minute == 0 and value.second == 0:
            return value.strftime("%Y-%m-%d")
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def to_markdown_table(df: pd.DataFrame) -> str:
    """Render a small DataFrame as a markdown table without extra dependencies."""
    if df.empty:
        return "_No rows._"

    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(_format_scalar(row[column]) for column in columns) + " |")
    return "\n".join([header, separator] + rows)


def build_sample_rows(panel_df: pd.DataFrame, rows_per_type: int = 4) -> pd.DataFrame:
    """Return a small multi-type sample for the spec document."""
    sample_columns = [
        "ticker",
        "event_type",
        "event_date",
        "event_timestamp",
        "effective_model_date",
        "timing_bucket",
        "source_id",
        "event_period_end",
        "market_asof_date",
        "fund_snapshot_filing_date",
        "fund_snapshot_is_current_event",
        "current_filing_sentiment_available",
    ]

    sample_frames = []
    for event_type, group in panel_df.groupby("event_type", sort=True):
        del event_type
        sample_frames.append(group[sample_columns].head(rows_per_type))

    sample_df = pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame()
    return sample_df


def get_comparison_rows() -> tuple[str, int | None]:
    """Load the current daily/event_v1 comparison panel row count when available."""
    for label, path in [
        ("event_v1_layer1_panel", EVENT_V1_LAYER1_PANEL_PATH),
        ("layer1_modeling_panel", LAYER1_BASE_PANEL_PATH),
    ]:
        if path.exists():
            rows = len(pd.read_parquet(path, columns=["ticker", "date"]))
            return label, int(rows)
    return "comparison_panel_unavailable", None


def build_spec_markdown(panel_df: pd.DataFrame, artifact_path: Path = EVENT_PANEL_V2_OUTPUT_PATH) -> str:
    """Create the spec document with diagnostics and explicit alignment rules."""
    event_counts = (
        panel_df["event_type"].value_counts(dropna=False).rename_axis("event_type").reset_index(name="rows")
    )
    timing_counts = (
        panel_df["timing_bucket"]
        .value_counts(dropna=False)
        .rename_axis("timing_bucket")
        .reset_index(name="rows")
    )
    predictive_feature_columns = (
        LAYER1_FEATURE_COLUMNS + LAYER2_V2_FEATURE_COLUMNS + SENTIMENT_CONTEXT_COLUMNS + EVENT_CONTEXT_FEATURE_COLUMNS
    )
    missingness = (
        panel_df[predictive_feature_columns]
        .isna()
        .mean()
        .mul(100.0)
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "column", 0: "missing_pct"})
    )
    top_missingness = missingness.head(12).copy()
    sample_rows = build_sample_rows(panel_df)

    comparison_label, comparison_rows = get_comparison_rows()
    comparison_note = (
        f"The current daily comparison panel `{comparison_label}` has {comparison_rows:,} rows; "
        f"`event_panel_v2` has {len(panel_df):,} rows, so v2 removes daily forward-filled repetition and "
        "stores one observation per information event."
        if comparison_rows is not None
        else "The current daily comparison panel was not available while building this spec."
    )

    included_event_types = ", ".join(sorted(panel_df["event_type"].dropna().astype(str).unique()))

    lines = [
        "# Event Panel V2 Spec",
        "",
        "## Purpose",
        "",
        "Phase 2 rebuilds the main research panel around event rows rather than daily forward-filled rows.",
        "One row in `event_panel_v2` equals one ticker-event.",
        "",
        "## Scope",
        "",
        f"- Included event types: `{included_event_types}`",
        "- Excluded for this phase: earnings announcement rows, because the repo does not contain a clean standalone earnings-date source outside the grouped 8-K path. To keep Phase 2 within scope, v2 ships with 10-Q and 10-K events only.",
        "- Existing `event_v1` and daily modeling panels remain unchanged.",
        "- No labels, model family changes, training changes, or validation changes are included in this artifact.",
        "",
        "## Row Unit",
        "",
        "- `event_date`: raw filing date from the source metadata. In some SEC rows this can differ from the local acceptance-calendar date implied by `event_timestamp`.",
        "- `effective_model_date`: first tradable date when the event is considered available to the model.",
        "- `source_id`: SEC accession number for the filing event.",
        "",
        "## Alignment Rules",
        "",
        "- After-close filings shift to the next tradable date through `effective_model_date`, measured against the local acceptance timestamp when a timestamp exists.",
        "- Before-open filings are available on the same trading day as the local acceptance timestamp.",
        "- Market-hours filings are available on the same trading day as the local acceptance timestamp.",
        "- Missing timestamps are handled conservatively and shift to the next tradable date.",
        "- `effective_model_date` is re-aligned from the stored SEC `availability_base_date` onto actual ticker trading dates so the event panel uses a consistent tradable calendar.",
        "- Market features are attached strictly from the prior trading day: `market_asof_date < effective_model_date`.",
        "- Fundamentals use the latest valid filing snapshot with `fund_snapshot_effective_model_date <= effective_model_date`.",
        "- Sentiment context is same-filing only and is joined by accession number; no daily forward-filled sentiment layer is carried into v2.",
        "- No grouped 8-K EXP-009 features are included in v2.",
        "",
        "## Diagnostics",
        "",
        f"- Row count: `{len(panel_df):,}`",
        f"- Ticker count: `{panel_df['ticker'].nunique():,}`",
        f"- Event date range: `{panel_df['event_date'].min().date()}` to `{panel_df['event_date'].max().date()}`",
        f"- Effective model date range: `{panel_df['effective_model_date'].min().date()}` to `{panel_df['effective_model_date'].max().date()}`",
        f"- Fundamentals feature count: `{len(LAYER1_FEATURE_COLUMNS)}`",
        f"- Market feature count: `{len(LAYER2_V2_FEATURE_COLUMNS)}`",
        f"- Filing sentiment feature count: `{len(SENTIMENT_CONTEXT_COLUMNS)}`",
        f"- Event-context feature count: `{len(EVENT_CONTEXT_FEATURE_COLUMNS)}`",
        "",
        "### Event Counts By Type",
        "",
        to_markdown_table(event_counts),
        "",
        "### Timing Bucket Counts",
        "",
        to_markdown_table(timing_counts),
        "",
        "### Top Missingness Columns",
        "",
        to_markdown_table(top_missingness),
        "",
        "### Sample Rows",
        "",
        to_markdown_table(sample_rows),
        "",
        "## Comparison To Current Daily/Event_V1 Design",
        "",
        comparison_note,
        "In the daily/event_v1 design, one filing can influence many later rows through daily forward-fill. In v2, the filing itself becomes the observation unit and the latest valid snapshot is attached once at the event boundary.",
        "",
        "## Structural Notes",
        "",
        f"- Exact current-filing fundamentals available: `{int(panel_df['current_filing_fundamentals_available'].sum()):,}` of `{len(panel_df):,}` rows",
        f"- Exact current-filing sentiment available: `{int(panel_df['current_filing_sentiment_available'].sum()):,}` of `{len(panel_df):,}` rows",
        f"- Rows where the attached fundamentals snapshot equals the current event: `{int(panel_df['fund_snapshot_is_current_event'].sum()):,}` of `{len(panel_df):,}` rows",
        "- When exact current-filing fundamentals are unavailable, the panel keeps the row and attaches the latest prior valid fundamentals snapshot instead. That fallback is explicit through `current_filing_fundamentals_available` and `fund_snapshot_is_current_event`.",
        "",
        "## Artifact",
        "",
        f"- Main parquet: `{artifact_path}`",
        "",
    ]
    return "\n".join(lines)


def print_summary(panel_df: pd.DataFrame) -> None:
    """Print a compact console summary for the builder run."""
    print("\nEvent Panel V2 Summary")
    print("-" * 60)
    print(f"Rows: {len(panel_df):,}")
    print(f"Tickers: {panel_df['ticker'].nunique():,}")
    print(
        "Event date range: "
        f"{panel_df['event_date'].min().date()} to {panel_df['event_date'].max().date()}"
    )
    print(
        "Effective model date range: "
        f"{panel_df['effective_model_date'].min().date()} to {panel_df['effective_model_date'].max().date()}"
    )
    print("\nEvent counts by type")
    print("-" * 60)
    for event_type, count in panel_df["event_type"].value_counts().items():
        print(f"{event_type:<12} {count:>8,}")
    print("\nTop missingness columns")
    print("-" * 60)
    predictive_feature_columns = (
        LAYER1_FEATURE_COLUMNS + LAYER2_V2_FEATURE_COLUMNS + SENTIMENT_CONTEXT_COLUMNS + EVENT_CONTEXT_FEATURE_COLUMNS
    )
    missingness = panel_df[predictive_feature_columns].isna().mean().mul(100.0).sort_values(ascending=False)
    for column_name, percentage in missingness.head(10).items():
        print(f"{column_name:<30} {percentage:>8.2f}%")


def build_event_panel_v2(
    price_path: Path = PRICE_INPUT_PATH,
    event_source_path: Path = SEC_FILING_METADATA_PATH,
    fundamentals_path: Path = FUNDAMENTALS_INPUT_PATH,
    market_path: Path = MARKET_INPUT_PATH,
    sentiment_path: Path = FULL_SENTIMENT_INPUT_PATH,
) -> pd.DataFrame:
    """Build the event-based Phase 2 research panel."""
    prices_df = prepare_prices(
        load_parquet(price_path, ["ticker", "date", "adj_close"], "Price labels")
    )
    event_source_df = load_event_source(event_source_path)
    event_source_df = align_event_effective_dates(event_source_df, prices_df)
    panel = build_base_event_panel(event_source_df)

    fundamentals_df = load_fundamentals(fundamentals_path)
    panel = attach_event_fundamentals_metadata(panel, fundamentals_df)
    panel = attach_fundamental_snapshot(panel, fundamentals_df, prices_df)

    market_df = load_market_features(market_path)
    panel = attach_market_snapshot(panel, market_df)

    sentiment_df = load_sentiment_context(sentiment_path)
    panel = attach_sentiment_context(panel, sentiment_df)

    panel = finalize_column_order(panel)
    validate_panel_structure(panel, trading_dates_df=prices_df)
    panel = panel.sort_values(["effective_model_date", "ticker", "event_type"]).reset_index(drop=True)
    return panel


def main() -> None:
    """Build the parquet artifact and the spec document."""
    args = parse_args()
    price_path = Path(args.price_path)
    event_source_path = Path(args.event_source_path)
    fundamentals_path = Path(args.fundamentals_path)
    market_path = Path(args.market_path)
    sentiment_path = Path(args.sentiment_path)
    output_path = Path(args.output_path)
    spec_path = Path(args.spec_path)
    ensure_parent_dir(output_path)
    ensure_parent_dir(spec_path)

    print(f"Loading price data from: {price_path}")
    print(f"Loading event sources from: {event_source_path}")
    panel_df = build_event_panel_v2(
        price_path=price_path,
        event_source_path=event_source_path,
        fundamentals_path=fundamentals_path,
        market_path=market_path,
        sentiment_path=sentiment_path,
    )

    print(f"Saving event_panel_v2 to: {output_path}")
    panel_df.to_parquet(output_path, index=False)

    spec_markdown = build_spec_markdown(panel_df, artifact_path=output_path)
    print(f"Writing event panel spec to: {spec_path}")
    spec_path.write_text(spec_markdown, encoding="utf-8")

    print_summary(panel_df)
    print("\nSaved event_panel_v2 artifacts.")


if __name__ == "__main__":
    main()
