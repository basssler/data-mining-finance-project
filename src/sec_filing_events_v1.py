"""Build minimal structured SEC filing-event features for the event_v1 lane.

This module intentionally stays metadata-only:
- no text parsing
- no sentiment
- no MD&A extraction
- no 8-K item parsing

It uses SEC submissions metadata for the existing project universe only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.config import END_DATE, START_DATE
from src.config_event_v1 import (
    LAYER1_BASE_PANEL_PATH,
    SEC_FILING_EVENTS_V1_OUTPUT_PATH,
    ensure_event_v1_directories,
)
from src.paths import INTERIM_DATA_DIR, RAW_DATA_DIR
from src.universe import get_layer1_tickers

LOOKBACK_START = "2014-01-01"
MARKET_OPEN_MINUTES = 9 * 60 + 30
MARKET_CLOSE_MINUTES = 16 * 60
TARGET_TIMEZONE = "America/New_York"
FORM_TYPES = {"8-K", "10-K", "10-Q"}
FORWARD_ALIGNMENT_TOLERANCE_DAYS = 7

SEC_UNIVERSE_MAPPING_V1_PATH = INTERIM_DATA_DIR / "sec" / "sec_universe_mapping_v1.parquet"
SEC_FILING_METADATA_V1_PATH = INTERIM_DATA_DIR / "sec" / "sec_filing_metadata_v1.parquet"

LOCAL_MAPPING_FALLBACK_PATHS = [
    RAW_DATA_DIR / "fundamentals" / "raw_fundamentals.parquet",
    RAW_DATA_DIR / "sec_filings" / "sec_filings_text_index.parquet",
]

MANUAL_SEC_LOOKUP_OVERRIDES = {
    "BF-B": {
        "sec_lookup_ticker": "BF.B",
        "cik": "0000014693",
        "override_note": "Project ticker uses BF-B while SEC commonly lists BF.B.",
    },
    "MMC": {
        "sec_lookup_ticker": "MMC",
        "cik": "0000062709",
        "override_note": "Universe v2 requires a manual SEC CIK override for Marsh & McLennan Companies, Inc.",
    },
}

DAILY_FEATURE_COLUMNS = [
    "sec_is_8k_today",
    "sec_is_10q_today",
    "sec_is_10k_today",
    "sec_filing_count_1d",
    "sec_filing_count_5d",
    "sec_after_close_filing_count_1d",
    "sec_pre_market_filing_count_1d",
    "sec_8k_decay_3d",
    "sec_8k_decay_5d",
    "sec_10q_decay_3d",
    "sec_10k_decay_3d",
    "sec_days_since_any_filing",
    "sec_days_since_8k",
    "sec_days_since_10q",
    "sec_days_since_10k",
]


class SecRequester:
    """Small helper for polite SEC requests with a custom user agent."""

    def __init__(
        self,
        user_agent: str,
        min_delay_seconds: float = 0.25,
        timeout_seconds: int = 30,
    ) -> None:
        self.user_agent = user_agent
        self.min_delay_seconds = min_delay_seconds
        self.timeout_seconds = timeout_seconds
        self._last_request_time = 0.0

    def wait_if_needed(self) -> None:
        """Sleep briefly so requests stay politely spaced out."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.min_delay_seconds:
            time.sleep(self.min_delay_seconds - elapsed)

    def get_json(self, url: str) -> dict:
        """Download and parse JSON from a SEC endpoint."""
        self.wait_if_needed()
        request = Request(
            url,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "application/json",
                "Host": "data.sec.gov" if "data.sec.gov" in url else "www.sec.gov",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                self._last_request_time = time.time()
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(f"HTTP error {exc.code} for URL: {url}") from exc
        except URLError as exc:
            raise RuntimeError(f"Network error for URL: {url}") from exc


def parse_args() -> argparse.Namespace:
    """Parse CLI options for SEC filing-event feature generation."""
    parser = argparse.ArgumentParser(description="Build event_v1 SEC filing-event features.")
    parser.add_argument("--panel-path", default=str(LAYER1_BASE_PANEL_PATH))
    parser.add_argument("--output-path", default=str(SEC_FILING_EVENTS_V1_OUTPUT_PATH))
    parser.add_argument("--mapping-output-path", default=str(SEC_UNIVERSE_MAPPING_V1_PATH))
    parser.add_argument("--metadata-output-path", default=str(SEC_FILING_METADATA_V1_PATH))
    return parser.parse_args()


def get_sec_user_agent() -> str:
    """Build a polite SEC identity string from environment variables if possible."""
    sec_user_agent = os.getenv("SEC_USER_AGENT")
    if sec_user_agent:
        return sec_user_agent

    edgar_name = os.getenv("EDGAR_NAME")
    edgar_email = os.getenv("EDGAR_EMAIL")
    if edgar_name and edgar_email:
        return f"{edgar_name} {edgar_email}"
    if edgar_email:
        return edgar_email

    fallback = "MaxB FinanceCourseProject finance-course@example.com"
    print(
        "Warning: SEC_USER_AGENT or EDGAR_NAME/EDGAR_EMAIL not set. "
        f"Using fallback identity: {fallback}"
    )
    return fallback


def normalize_sec_ticker(ticker: str | None) -> str:
    """Normalize SEC ticker formatting to the project's dash format."""
    if ticker is None:
        return ""
    value = str(ticker).strip().upper()
    return value.replace(".", "-")


def load_panel_dates(path: Path) -> pd.DataFrame:
    """Load the locked Layer 1 panel and keep only ticker/date for alignment."""
    if not path.exists():
        raise FileNotFoundError(f"Layer 1 panel was not found: {path}")

    df = pd.read_parquet(path, columns=["ticker", "date"])
    df["ticker"] = df["ticker"].astype("string[python]").str.upper()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").astype("datetime64[ns]")
    df = df.dropna(subset=["ticker", "date"]).drop_duplicates().copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


def load_local_mapping_fallback() -> pd.DataFrame:
    """Load local ticker/CIK/company-name fallbacks from existing SEC artifacts."""
    frames = []

    fundamentals_path = LOCAL_MAPPING_FALLBACK_PATHS[0]
    if fundamentals_path.exists():
        fundamentals_df = pd.read_parquet(fundamentals_path, columns=["ticker", "cik"])
        fundamentals_df["ticker"] = fundamentals_df["ticker"].astype("string").str.upper()
        fundamentals_df["cik"] = (
            fundamentals_df["cik"].astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(10)
        )
        fundamentals_df["company_name"] = pd.NA
        frames.append(fundamentals_df[["ticker", "cik", "company_name"]])

    sec_index_path = LOCAL_MAPPING_FALLBACK_PATHS[1]
    if sec_index_path.exists():
        sec_index_df = pd.read_parquet(sec_index_path, columns=["ticker", "cik", "company_name"])
        sec_index_df["ticker"] = sec_index_df["ticker"].astype("string").str.upper()
        sec_index_df["cik"] = (
            sec_index_df["cik"].astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(10)
        )
        sec_index_df["company_name"] = sec_index_df["company_name"].astype("string")
        frames.append(sec_index_df[["ticker", "cik", "company_name"]])

    if not frames:
        return pd.DataFrame(columns=["ticker", "cik", "company_name"])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["ticker", "cik"]).copy()
    combined = combined.sort_values(["ticker", "company_name"]).reset_index(drop=True)
    combined = (
        combined.groupby("ticker", as_index=False)
        .agg(
            cik=("cik", "first"),
            company_name=("company_name", lambda s: next((value for value in s if pd.notna(value)), pd.NA)),
        )
        .sort_values("ticker")
        .reset_index(drop=True)
    )
    return combined


def fetch_sec_company_ticker_map(requester: SecRequester) -> pd.DataFrame:
    """Fetch the SEC ticker-to-CIK mapping file and normalize it for project matching."""
    payload = requester.get_json("https://www.sec.gov/files/company_tickers.json")
    rows = []
    for record in payload.values():
        sec_ticker = str(record.get("ticker", "")).upper().strip()
        cik = str(record.get("cik_str", "")).strip()
        company_name = str(record.get("title", "")).strip()
        if not sec_ticker or not cik:
            continue
        rows.append(
            {
                "sec_lookup_ticker": sec_ticker,
                "ticker": normalize_sec_ticker(sec_ticker),
                "cik": cik.zfill(10),
                "company_name": company_name,
                "mapping_source": "sec_company_tickers",
            }
        )
    return pd.DataFrame(rows).drop_duplicates(subset=["sec_lookup_ticker", "cik"]).copy()


def build_universe_sec_mapping(
    requester: SecRequester,
    universe_tickers: list[str],
) -> pd.DataFrame:
    """Create the universe-level SEC mapping table with project-only overrides."""
    sec_map_df = fetch_sec_company_ticker_map(requester)
    local_fallback_df = load_local_mapping_fallback()

    rows = []
    for ticker in universe_tickers:
        override = MANUAL_SEC_LOOKUP_OVERRIDES.get(ticker, {})
        sec_lookup_ticker = str(override.get("sec_lookup_ticker", ticker)).upper()
        sec_matches = sec_map_df.loc[
            (sec_map_df["sec_lookup_ticker"] == sec_lookup_ticker)
            | (sec_map_df["ticker"] == normalize_sec_ticker(sec_lookup_ticker))
            | (sec_map_df["ticker"] == ticker)
        ].copy()

        cik = None
        company_name = None
        mapping_source = None
        if not sec_matches.empty:
            row = sec_matches.iloc[0]
            cik = str(row["cik"]).zfill(10)
            company_name = row["company_name"]
            mapping_source = row["mapping_source"]

        if cik is None or cik == "" or cik == "NAN":
            fallback_match = local_fallback_df.loc[local_fallback_df["ticker"] == ticker]
            if not fallback_match.empty:
                fallback_row = fallback_match.iloc[0]
                cik = str(fallback_row["cik"]).zfill(10)
                company_name = fallback_row["company_name"]
                mapping_source = "local_sec_fallback"

        if override.get("cik"):
            cik = str(override["cik"]).zfill(10)
            mapping_source = "manual_override"

        if cik is None or cik == "" or cik == "NAN":
            raise ValueError(f"Missing SEC CIK mapping for project ticker: {ticker}")

        rows.append(
            {
                "ticker": ticker,
                "cik": cik,
                "company_name": company_name if company_name not in {None, ""} else pd.NA,
                "sec_lookup_ticker": sec_lookup_ticker,
                "mapping_source": mapping_source if mapping_source else "manual_override",
                "manual_override_note": override.get("override_note", pd.NA),
            }
        )

    mapping_df = pd.DataFrame(rows)
    mapping_df["ticker"] = mapping_df["ticker"].astype("string[python]")
    mapping_df["cik"] = mapping_df["cik"].astype("string[python]").str.zfill(10)
    mapping_df["company_name"] = mapping_df["company_name"].astype("string[python]")
    mapping_df["sec_lookup_ticker"] = mapping_df["sec_lookup_ticker"].astype("string[python]")
    mapping_df["mapping_source"] = mapping_df["mapping_source"].astype("string[python]")
    mapping_df["manual_override_note"] = mapping_df["manual_override_note"].astype("string[python]")
    mapping_df = mapping_df.sort_values("ticker").reset_index(drop=True)
    return mapping_df


def submission_arrays_to_frame(arrays: dict) -> pd.DataFrame:
    """Convert one SEC submissions arrays payload into a DataFrame."""
    if not arrays:
        return pd.DataFrame()
    return pd.DataFrame(arrays)


def fetch_company_submission_history(
    requester: SecRequester,
    ticker: str,
    cik: str,
) -> tuple[pd.DataFrame, str]:
    """Fetch one company's submissions history from the SEC submissions API."""
    main_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    payload = requester.get_json(main_url)
    company_name = str(payload.get("name", "")).strip()

    frames = [submission_arrays_to_frame(payload.get("filings", {}).get("recent", {}))]
    for file_record in payload.get("filings", {}).get("files", []):
        file_name = str(file_record.get("name", "")).strip()
        if not file_name:
            continue
        extra_payload = requester.get_json(f"https://data.sec.gov/submissions/{file_name}")
        frames.append(submission_arrays_to_frame(extra_payload))

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if combined.empty:
        return pd.DataFrame(), company_name

    combined["ticker"] = ticker
    combined["cik"] = cik
    combined["company_name"] = company_name if company_name else pd.NA
    return combined, company_name


def normalize_submission_history(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only minimal structured filing metadata needed for this experiment."""
    if df.empty:
        return pd.DataFrame(
            columns=[
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
        )

    prepared = df.copy()
    prepared["ticker"] = prepared["ticker"].astype("string[python]").str.upper()
    prepared["cik"] = prepared["cik"].astype("string[python]").str.zfill(10)
    prepared["company_name"] = prepared["company_name"].astype("string[python]")
    prepared["form_type"] = prepared["form"].astype("string[python]").str.upper().str.strip()
    prepared = prepared.loc[prepared["form_type"].isin(FORM_TYPES)].copy()

    prepared["filing_date"] = pd.to_datetime(prepared["filingDate"], errors="coerce")
    prepared["filing_timestamp_utc"] = pd.to_datetime(
        prepared.get("acceptanceDateTime"),
        errors="coerce",
        utc=True,
    )
    prepared["filing_timestamp_local"] = prepared["filing_timestamp_utc"].dt.tz_convert(TARGET_TIMEZONE)
    prepared["accession_number"] = prepared["accessionNumber"].astype("string[python]")

    prepared = prepared.dropna(subset=["ticker", "cik", "form_type", "filing_date", "accession_number"]).copy()

    prepared = prepared.loc[
        (prepared["filing_date"] >= pd.Timestamp(LOOKBACK_START))
        & (prepared["filing_date"] <= pd.Timestamp(END_DATE))
    ].copy()

    local_minutes = prepared["filing_timestamp_local"].dt.hour * 60 + prepared["filing_timestamp_local"].dt.minute
    missing_time_mask = prepared["filing_timestamp_local"].isna()
    after_close_mask = prepared["filing_timestamp_local"].notna() & (local_minutes >= MARKET_CLOSE_MINUTES)
    pre_market_mask = prepared["filing_timestamp_local"].notna() & (local_minutes < MARKET_OPEN_MINUTES)
    market_hours_mask = prepared["filing_timestamp_local"].notna() & ~after_close_mask & ~pre_market_mask

    prepared["timing_bucket"] = "missing_time_conservative_next_day"
    prepared.loc[after_close_mask, "timing_bucket"] = "after_close"
    prepared.loc[pre_market_mask, "timing_bucket"] = "pre_market"
    prepared.loc[market_hours_mask, "timing_bucket"] = "market_hours"

    prepared["availability_base_date"] = pd.NaT
    prepared["availability_base_date"] = prepared["availability_base_date"].astype("datetime64[ns]")
    prepared.loc[missing_time_mask, "availability_base_date"] = (
        prepared.loc[missing_time_mask, "filing_date"] + pd.Timedelta(days=1)
    )
    prepared.loc[after_close_mask, "availability_base_date"] = (
        prepared.loc[after_close_mask, "filing_timestamp_local"]
        .dt.tz_localize(None)
        .dt.normalize()
        + pd.Timedelta(days=1)
    )
    same_day_mask = pre_market_mask | market_hours_mask
    prepared.loc[same_day_mask, "availability_base_date"] = (
        prepared.loc[same_day_mask, "filing_timestamp_local"].dt.tz_localize(None).dt.normalize()
    )
    prepared["availability_base_date"] = pd.to_datetime(
        prepared["availability_base_date"],
        errors="coerce",
    ).astype("datetime64[ns]")

    prepared = prepared.drop_duplicates(subset=["ticker", "accession_number"]).copy()
    prepared = prepared.sort_values(["ticker", "filing_date", "filing_timestamp_utc", "form_type"]).reset_index(drop=True)

    keep_columns = [
        "ticker",
        "cik",
        "company_name",
        "form_type",
        "filing_date",
        "filing_timestamp_utc",
        "filing_timestamp_local",
        "timing_bucket",
        "availability_base_date",
        "accession_number",
    ]
    output = prepared[keep_columns].copy()
    output["effective_model_date"] = pd.NaT
    output["effective_model_date"] = output["effective_model_date"].astype("datetime64[ns]")
    return output


def align_effective_model_dates(
    filing_df: pd.DataFrame,
    panel_dates_df: pd.DataFrame,
) -> pd.DataFrame:
    """Align filing availability dates to the next actual trading date."""
    panel_dates = panel_dates_df.copy().rename(columns={"date": "panel_date"})
    panel_dates["ticker"] = panel_dates["ticker"].astype("string[python]").str.upper()
    panel_dates["panel_date"] = pd.to_datetime(panel_dates["panel_date"], errors="coerce").astype("datetime64[ns]")
    filings = filing_df.copy()
    filings["ticker"] = filings["ticker"].astype("string[python]").str.upper()
    filings["availability_base_date"] = pd.to_datetime(
        filings["availability_base_date"],
        errors="coerce",
    ).astype("datetime64[ns]")

    aligned = pd.merge_asof(
        left=filings.sort_values(["availability_base_date", "ticker"]).reset_index(drop=True),
        right=panel_dates.sort_values(["panel_date", "ticker"]).reset_index(drop=True),
        left_on="availability_base_date",
        right_on="panel_date",
        by="ticker",
        direction="forward",
        allow_exact_matches=True,
        tolerance=pd.Timedelta(days=FORWARD_ALIGNMENT_TOLERANCE_DAYS),
    )
    aligned = aligned.dropna(subset=["panel_date"]).copy()
    aligned["effective_model_date"] = aligned["panel_date"].astype("datetime64[ns]")
    aligned = aligned.drop(columns=["panel_date"])
    aligned = aligned.sort_values(["ticker", "effective_model_date", "form_type", "accession_number"]).reset_index(drop=True)
    return aligned


def build_daily_feature_table(
    filing_df: pd.DataFrame,
    panel_dates_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate aligned filing metadata into daily ticker-date features."""
    filings = filing_df.copy()
    filings = filings.rename(columns={"effective_model_date": "date"})
    filings["daily_filing_count"] = 1
    filings["daily_8k_count"] = (filings["form_type"] == "8-K").astype("int64")
    filings["daily_10q_count"] = (filings["form_type"] == "10-Q").astype("int64")
    filings["daily_10k_count"] = (filings["form_type"] == "10-K").astype("int64")
    filings["daily_after_close_count"] = (filings["timing_bucket"] == "after_close").astype("int64")
    filings["daily_pre_market_count"] = (filings["timing_bucket"] == "pre_market").astype("int64")

    daily_events = (
        filings.groupby(["ticker", "date"], as_index=False)
        .agg(
            daily_filing_count=("daily_filing_count", "sum"),
            daily_8k_count=("daily_8k_count", "sum"),
            daily_10q_count=("daily_10q_count", "sum"),
            daily_10k_count=("daily_10k_count", "sum"),
            daily_after_close_count=("daily_after_close_count", "sum"),
            daily_pre_market_count=("daily_pre_market_count", "sum"),
        )
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    panel = panel_dates_df.copy().merge(
        daily_events,
        on=["ticker", "date"],
        how="left",
        validate="one_to_one",
    )
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)

    fill_zero_columns = [
        "daily_filing_count",
        "daily_8k_count",
        "daily_10q_count",
        "daily_10k_count",
        "daily_after_close_count",
        "daily_pre_market_count",
    ]
    for column in fill_zero_columns:
        panel[column] = pd.to_numeric(panel[column], errors="coerce").fillna(0.0)

    group = panel.groupby("ticker", group_keys=False)
    panel["sec_is_8k_today"] = (panel["daily_8k_count"] > 0).astype("int64")
    panel["sec_is_10q_today"] = (panel["daily_10q_count"] > 0).astype("int64")
    panel["sec_is_10k_today"] = (panel["daily_10k_count"] > 0).astype("int64")
    panel["sec_filing_count_1d"] = panel["daily_filing_count"]
    panel["sec_filing_count_5d"] = (
        group["daily_filing_count"].rolling(window=5, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    panel["sec_after_close_filing_count_1d"] = panel["daily_after_close_count"]
    panel["sec_pre_market_filing_count_1d"] = panel["daily_pre_market_count"]

    panel["any_filing_event_date"] = panel["date"].where(panel["daily_filing_count"] > 0)
    panel["event_8k_date"] = panel["date"].where(panel["daily_8k_count"] > 0)
    panel["event_10q_date"] = panel["date"].where(panel["daily_10q_count"] > 0)
    panel["event_10k_date"] = panel["date"].where(panel["daily_10k_count"] > 0)

    panel["last_any_filing_date"] = group["any_filing_event_date"].ffill()
    panel["last_8k_date"] = group["event_8k_date"].ffill()
    panel["last_10q_date"] = group["event_10q_date"].ffill()
    panel["last_10k_date"] = group["event_10k_date"].ffill()

    panel["sec_days_since_any_filing"] = (
        panel["date"] - pd.to_datetime(panel["last_any_filing_date"], errors="coerce")
    ).dt.days.astype("float64")
    panel["sec_days_since_8k"] = (
        panel["date"] - pd.to_datetime(panel["last_8k_date"], errors="coerce")
    ).dt.days.astype("float64")
    panel["sec_days_since_10q"] = (
        panel["date"] - pd.to_datetime(panel["last_10q_date"], errors="coerce")
    ).dt.days.astype("float64")
    panel["sec_days_since_10k"] = (
        panel["date"] - pd.to_datetime(panel["last_10k_date"], errors="coerce")
    ).dt.days.astype("float64")

    panel["sec_8k_decay_3d"] = np.where(
        panel["sec_days_since_8k"].notna(),
        np.exp(-panel["sec_days_since_8k"] / 3.0),
        np.nan,
    )
    panel["sec_8k_decay_5d"] = np.where(
        panel["sec_days_since_8k"].notna(),
        np.exp(-panel["sec_days_since_8k"] / 5.0),
        np.nan,
    )
    panel["sec_10q_decay_3d"] = np.where(
        panel["sec_days_since_10q"].notna(),
        np.exp(-panel["sec_days_since_10q"] / 3.0),
        np.nan,
    )
    panel["sec_10k_decay_3d"] = np.where(
        panel["sec_days_since_10k"].notna(),
        np.exp(-panel["sec_days_since_10k"] / 3.0),
        np.nan,
    )

    output = panel[["ticker", "date"] + DAILY_FEATURE_COLUMNS].copy()
    output = output.sort_values(["ticker", "date"]).reset_index(drop=True)
    return output


def print_mapping_summary(mapping_df: pd.DataFrame) -> None:
    """Print a compact summary of the SEC universe mapping table."""
    print("\nSEC Universe Mapping Summary")
    print("-" * 60)
    print(f"Rows: {len(mapping_df):,}")
    print(f"Tickers mapped: {mapping_df['ticker'].nunique():,}")
    override_count = int(mapping_df["manual_override_note"].notna().sum())
    print(f"Manual overrides used: {override_count:,}")


def print_metadata_summary(metadata_df: pd.DataFrame) -> None:
    """Print a compact summary of raw filing metadata coverage."""
    print("\nSEC Filing Metadata Summary")
    print("-" * 60)
    print(f"Rows: {len(metadata_df):,}")
    print(f"Tickers: {metadata_df['ticker'].nunique():,}")
    print(
        f"Date range: {metadata_df['filing_date'].min().date()} to {metadata_df['filing_date'].max().date()}"
    )

    print("\nFilings by form type")
    print("-" * 60)
    for form_type, count in metadata_df["form_type"].value_counts().sort_index().items():
        print(f"{form_type:<10} {count:>8,}")

    print("\nTiming bucket counts")
    print("-" * 60)
    for timing_bucket, count in metadata_df["timing_bucket"].value_counts().sort_index().items():
        print(f"{timing_bucket:<34} {count:>8,}")


def print_feature_summary(df: pd.DataFrame) -> None:
    """Print a compact summary of the aligned daily SEC filing-event table."""
    missingness = df[DAILY_FEATURE_COLUMNS].isna().mean().mul(100).sort_values(ascending=False)

    print("\nEvent V1 SEC Filing-Event Feature Summary")
    print("-" * 60)
    print(f"Rows: {len(df):,}")
    print(f"Tickers: {df['ticker'].nunique():,}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Rows with any filing today: {int((df['sec_filing_count_1d'] > 0).sum()):,}")
    print(f"Rows with 8-K today:        {int((df['sec_is_8k_today'] > 0).sum()):,}")
    print(f"Rows with 10-Q today:       {int((df['sec_is_10q_today'] > 0).sum()):,}")
    print(f"Rows with 10-K today:       {int((df['sec_is_10k_today'] > 0).sum()):,}")

    print("\nTop missingness columns")
    print("-" * 60)
    for column_name, percentage in missingness.head(12).items():
        print(f"{column_name:<30} {percentage:>8.2f}%")


def main() -> None:
    """Build and save minimal structured SEC filing-event features."""
    args = parse_args()
    ensure_event_v1_directories()

    panel_path = Path(args.panel_path)
    output_path = Path(args.output_path)
    mapping_output_path = Path(args.mapping_output_path)
    metadata_output_path = Path(args.metadata_output_path)

    requester = SecRequester(user_agent=get_sec_user_agent(), min_delay_seconds=0.25)
    universe_tickers = get_layer1_tickers()

    print(f"Loading Layer 1 panel dates from: {panel_path}")
    panel_dates_df = load_panel_dates(panel_path)

    print("Building universe-level SEC mapping table...")
    mapping_df = build_universe_sec_mapping(
        requester=requester,
        universe_tickers=universe_tickers,
    )

    metadata_frames = []
    print(f"Fetching SEC submissions metadata for {len(mapping_df)} universe tickers...")
    for row in mapping_df.itertuples(index=False):
        print(f"Fetching {row.ticker} ({row.cik})...")
        company_history_df, company_name = fetch_company_submission_history(
            requester=requester,
            ticker=str(row.ticker),
            cik=str(row.cik),
        )
        normalized_history_df = normalize_submission_history(company_history_df)
        metadata_frames.append(normalized_history_df)
        if company_name:
            mapping_df.loc[mapping_df["ticker"] == row.ticker, "company_name"] = company_name

    raw_metadata_df = pd.concat(metadata_frames, ignore_index=True) if metadata_frames else pd.DataFrame()
    raw_metadata_df = align_effective_model_dates(raw_metadata_df, panel_dates_df)

    mapping_output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mapping_df.to_parquet(mapping_output_path, index=False)
    raw_metadata_df.to_parquet(metadata_output_path, index=False)

    print_mapping_summary(mapping_df)
    print_metadata_summary(raw_metadata_df)

    print("Building daily SEC filing-event feature table...")
    feature_df = build_daily_feature_table(
        filing_df=raw_metadata_df,
        panel_dates_df=panel_dates_df,
    )
    feature_df.to_parquet(output_path, index=False)

    print(f"Saved SEC mapping table to: {mapping_output_path}")
    print(f"Saved raw filing metadata to: {metadata_output_path}")
    print(f"Saved daily filing-event features to: {output_path}")
    print_feature_summary(feature_df)
    print("\nSaved event_v1 SEC filing-event features.")


if __name__ == "__main__":
    main()
