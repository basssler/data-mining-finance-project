"""Build grouped 8-K item-category features for the event_v1 lane.

This module stays strictly metadata-only:
- no full-text parsing
- no sentiment
- no MD&A extraction
- no exhibit parsing

It reuses the EXP-008 universe SEC mapping and timing-aligned filing metadata,
then adds structured 8-K item group features using only the SEC submissions
`items` field.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.config_event_v1 import (
    LAYER1_BASE_PANEL_PATH,
    SEC_8K_GROUPED_EVENT_FEATURE_COLUMNS,
    SEC_8K_GROUPED_EVENTS_V1_OUTPUT_PATH,
    ensure_event_v1_directories,
)
from src.sec_filing_events_v1 import (
    SEC_FILING_METADATA_V1_PATH,
    SEC_UNIVERSE_MAPPING_V1_PATH,
    SecRequester,
    build_universe_sec_mapping,
    fetch_company_submission_history,
    get_sec_user_agent,
    load_panel_dates,
)
from src.universe import get_layer1_tickers

ITEM_TOKEN_PATTERN = re.compile(r"^\d+(?:\.\d+)?$")
CATEGORY_NAMES = [
    "earnings_results",
    "guidance_outlook",
    "leadership_governance",
    "financing_securities",
    "material_agreement_mna",
    "regulatory_legal",
]

# Clean, structured item-to-category mapping only. Broad catch-all items such as
# 8.01 and 9.01 are intentionally excluded, and section-level integers such as
# "5" or "7" are skipped as ambiguous.
ITEM_TO_CATEGORY = {
    "2.02": "earnings_results",  # Results of operations / financial condition.
    "7.01": "guidance_outlook",  # Coarse Reg FD proxy for outlook / guidance updates.
    "5.02": "leadership_governance",  # Director / executive officer changes.
    "5.03": "leadership_governance",  # Charter / bylaw amendments.
    "5.07": "leadership_governance",  # Shareholder votes.
    "2.03": "financing_securities",  # Direct financial obligations.
    "3.02": "financing_securities",  # Unregistered equity sales.
    "3.03": "financing_securities",  # Modifications to security-holder rights.
    "1.01": "material_agreement_mna",  # Entry into material definitive agreements.
    "1.02": "material_agreement_mna",  # Termination of material definitive agreements.
    "2.01": "material_agreement_mna",  # Acquisitions / dispositions.
    "1.03": "regulatory_legal",  # Bankruptcy / receivership.
    "4.01": "regulatory_legal",  # Accountant changes.
    "4.02": "regulatory_legal",  # Non-reliance on prior statements / audit reports.
}

CATEGORY_TODAY_COLUMNS = {
    "earnings_results": "sec_8k_earnings_results_today",
    "guidance_outlook": "sec_8k_guidance_outlook_today",
    "leadership_governance": "sec_8k_leadership_governance_today",
    "financing_securities": "sec_8k_financing_securities_today",
    "material_agreement_mna": "sec_8k_material_agreement_mna_today",
    "regulatory_legal": "sec_8k_regulatory_legal_today",
}

CATEGORY_DECAY_COLUMNS = {
    "earnings_results": "sec_8k_earnings_results_decay_3d",
    "guidance_outlook": "sec_8k_guidance_outlook_decay_3d",
    "leadership_governance": "sec_8k_leadership_governance_decay_3d",
    "financing_securities": "sec_8k_financing_securities_decay_3d",
    "material_agreement_mna": "sec_8k_material_agreement_mna_decay_3d",
    "regulatory_legal": "sec_8k_regulatory_legal_decay_3d",
}

CATEGORY_DAYS_SINCE_COLUMNS = {
    "earnings_results": "sec_days_since_8k_earnings_results",
    "guidance_outlook": "sec_days_since_8k_guidance_outlook",
    "leadership_governance": "sec_days_since_8k_leadership_governance",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI options for grouped 8-K feature generation."""
    parser = argparse.ArgumentParser(description="Build grouped 8-K event_v1 features.")
    parser.add_argument("--panel-path", default=str(LAYER1_BASE_PANEL_PATH))
    parser.add_argument("--mapping-path", default=str(SEC_UNIVERSE_MAPPING_V1_PATH))
    parser.add_argument("--metadata-path", default=str(SEC_FILING_METADATA_V1_PATH))
    parser.add_argument("--output-path", default=str(SEC_8K_GROUPED_EVENTS_V1_OUTPUT_PATH))
    return parser.parse_args()


def load_sec_mapping(path: Path) -> pd.DataFrame:
    """Load the universe-level SEC mapping table from EXP-008."""
    if not path.exists():
        raise FileNotFoundError(
            f"SEC mapping table was not found: {path}. Run EXP-008 or provide --mapping-path."
        )

    mapping_df = pd.read_parquet(path)
    required_columns = ["ticker", "cik", "company_name", "sec_lookup_ticker"]
    missing_columns = [column for column in required_columns if column not in mapping_df.columns]
    if missing_columns:
        raise ValueError("SEC mapping table is missing columns: " + ", ".join(missing_columns))

    mapping_df["ticker"] = mapping_df["ticker"].astype("string[python]").str.upper()
    mapping_df["cik"] = mapping_df["cik"].astype("string[python]").str.zfill(10)
    mapping_df["company_name"] = mapping_df["company_name"].astype("string[python]")
    mapping_df["sec_lookup_ticker"] = mapping_df["sec_lookup_ticker"].astype("string[python]")
    mapping_df = mapping_df.dropna(subset=["ticker", "cik"]).drop_duplicates(subset=["ticker"]).copy()
    mapping_df = mapping_df.sort_values("ticker").reset_index(drop=True)
    return mapping_df


def load_sec_filing_metadata(path: Path) -> pd.DataFrame:
    """Load the aligned EXP-008 SEC metadata and keep exact 8-K rows only."""
    if not path.exists():
        raise FileNotFoundError(
            f"SEC filing metadata was not found: {path}. Run EXP-008 or provide --metadata-path."
        )

    metadata_df = pd.read_parquet(path)
    required_columns = [
        "ticker",
        "cik",
        "form_type",
        "filing_date",
        "filing_timestamp_local",
        "timing_bucket",
        "effective_model_date",
        "accession_number",
    ]
    missing_columns = [column for column in required_columns if column not in metadata_df.columns]
    if missing_columns:
        raise ValueError("SEC filing metadata is missing columns: " + ", ".join(missing_columns))

    metadata_df["ticker"] = metadata_df["ticker"].astype("string[python]").str.upper()
    metadata_df["cik"] = metadata_df["cik"].astype("string[python]").str.zfill(10)
    metadata_df["form_type"] = metadata_df["form_type"].astype("string[python]").str.upper().str.strip()
    metadata_df["filing_date"] = pd.to_datetime(metadata_df["filing_date"], errors="coerce")
    metadata_df["filing_timestamp_local"] = pd.to_datetime(
        metadata_df["filing_timestamp_local"],
        errors="coerce",
    )
    metadata_df["effective_model_date"] = pd.to_datetime(
        metadata_df["effective_model_date"],
        errors="coerce",
    )
    metadata_df["accession_number"] = metadata_df["accession_number"].astype("string[python]")
    metadata_df["timing_bucket"] = metadata_df["timing_bucket"].astype("string[python]")

    metadata_df = metadata_df.loc[metadata_df["form_type"] == "8-K"].copy()
    metadata_df = metadata_df.dropna(
        subset=["ticker", "cik", "filing_date", "effective_model_date", "accession_number"]
    ).copy()
    metadata_df = metadata_df.drop_duplicates(subset=["ticker", "accession_number"]).copy()
    metadata_df = metadata_df.sort_values(["ticker", "effective_model_date", "accession_number"])
    metadata_df = metadata_df.reset_index(drop=True)
    return metadata_df


def fetch_8k_submission_items(
    requester: SecRequester,
    mapping_df: pd.DataFrame,
) -> pd.DataFrame:
    """Fetch exact 8-K submissions for the project universe and keep raw item strings."""
    frames = []
    for row in mapping_df.itertuples(index=False):
        print(f"Fetching 8-K item metadata for {row.ticker} ({row.cik})...")
        company_history_df, _ = fetch_company_submission_history(
            requester=requester,
            ticker=str(row.ticker),
            cik=str(row.cik),
        )
        if company_history_df.empty:
            continue

        history = company_history_df.copy()
        history["form"] = history["form"].astype("string[python]").str.upper().str.strip()
        history = history.loc[history["form"] == "8-K"].copy()
        if history.empty:
            continue

        keep_columns = [
            "ticker",
            "cik",
            "form",
            "filingDate",
            "acceptanceDateTime",
            "accessionNumber",
            "items",
        ]
        history = history[keep_columns].copy()
        frames.append(history)

    if not frames:
        return pd.DataFrame(
            columns=[
                "ticker",
                "cik",
                "form",
                "filingDate",
                "acceptanceDateTime",
                "accessionNumber",
                "items",
            ]
        )

    submissions_df = pd.concat(frames, ignore_index=True)
    submissions_df["ticker"] = submissions_df["ticker"].astype("string[python]").str.upper()
    submissions_df["cik"] = submissions_df["cik"].astype("string[python]").str.zfill(10)
    submissions_df["form"] = submissions_df["form"].astype("string[python]").str.upper().str.strip()
    submissions_df["filingDate"] = pd.to_datetime(submissions_df["filingDate"], errors="coerce")
    submissions_df["acceptanceDateTime"] = pd.to_datetime(
        submissions_df["acceptanceDateTime"],
        errors="coerce",
        utc=True,
    )
    submissions_df["accessionNumber"] = submissions_df["accessionNumber"].astype("string[python]")
    submissions_df["items"] = submissions_df["items"].astype("string[python]")
    submissions_df = submissions_df.dropna(subset=["ticker", "cik", "form", "filingDate"]).copy()
    submissions_df = submissions_df.sort_values(["ticker", "filingDate", "acceptanceDateTime"])
    submissions_df = submissions_df.reset_index(drop=True)
    return submissions_df


def normalize_item_number(raw_item: object) -> str | None:
    """Normalize one SEC item token to a canonical x.yy code when possible."""
    if raw_item is None or (isinstance(raw_item, float) and math.isnan(raw_item)):
        return None

    token = str(raw_item).strip().upper()
    if not token or token == "NAN":
        return None
    token = token.replace("ITEM", "").strip()
    if not ITEM_TOKEN_PATTERN.fullmatch(token):
        return None

    numeric_value = float(token)
    if numeric_value.is_integer():
        return None
    return f"{numeric_value:.2f}"


def build_structured_item_table(
    submission_items_df: pd.DataFrame,
    filing_metadata_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Split raw item strings into a structured 8-K item table and reuse EXP-008 timing."""
    item_rows = []
    filing_rows_with_items = 0

    for record in submission_items_df.itertuples(index=False):
        raw_items = record.items
        if raw_items is None or str(raw_items).strip().upper() in {"", "NAN"}:
            continue

        normalized_items = []
        for token in re.split(r"[;,]", str(raw_items)):
            item_number = normalize_item_number(token)
            if item_number is not None:
                normalized_items.append(item_number)

        if not normalized_items:
            continue

        filing_rows_with_items += 1
        for item_number in normalized_items:
            item_rows.append(
                {
                    "ticker": str(record.ticker).upper(),
                    "cik": str(record.cik).zfill(10),
                    "form_type": "8-K",
                    "filing_date": pd.to_datetime(record.filingDate, errors="coerce"),
                    "filing_timestamp_utc": pd.to_datetime(
                        record.acceptanceDateTime,
                        errors="coerce",
                        utc=True,
                    ),
                    "accession_number": str(record.accessionNumber),
                    "item_number": item_number,
                    "item_category": ITEM_TO_CATEGORY.get(item_number),
                }
            )

    if not item_rows:
        empty = pd.DataFrame(
            columns=[
                "ticker",
                "cik",
                "filing_date",
                "filing_timestamp",
                "effective_model_date",
                "item_number",
                "form_type",
                "item_category",
                "timing_bucket",
                "accession_number",
            ]
        )
        diagnostics = {
            "raw_submission_rows": int(len(submission_items_df)),
            "filing_rows_with_items": 0,
            "structured_item_rows": 0,
            "mapped_item_rows": 0,
            "unmatched_item_rows": 0,
        }
        return empty, diagnostics

    items_df = pd.DataFrame(item_rows)
    items_df = items_df.dropna(subset=["ticker", "cik", "accession_number", "item_number"]).copy()
    items_df = items_df.drop_duplicates(subset=["ticker", "accession_number", "item_number"]).copy()

    metadata_for_merge = filing_metadata_df[
        [
            "ticker",
            "cik",
            "filing_date",
            "filing_timestamp_local",
            "timing_bucket",
            "effective_model_date",
            "accession_number",
            "form_type",
        ]
    ].copy()
    metadata_for_merge = metadata_for_merge.rename(columns={"filing_timestamp_local": "filing_timestamp"})

    merged_df = items_df.merge(
        metadata_for_merge,
        on=["ticker", "accession_number"],
        how="left",
        suffixes=("_raw", ""),
    )
    unmatched_item_rows = int(merged_df["effective_model_date"].isna().sum())
    merged_df = merged_df.dropna(subset=["effective_model_date"]).copy()

    merged_df["cik"] = merged_df["cik"].fillna(merged_df["cik_raw"]).astype("string[python]").str.zfill(10)
    merged_df["filing_date"] = pd.to_datetime(
        merged_df["filing_date"],
        errors="coerce",
    ).fillna(pd.to_datetime(merged_df["filing_date_raw"], errors="coerce"))
    merged_df["filing_timestamp"] = pd.to_datetime(
        merged_df["filing_timestamp"],
        errors="coerce",
    )
    merged_df["effective_model_date"] = pd.to_datetime(
        merged_df["effective_model_date"],
        errors="coerce",
    )
    merged_df["item_category"] = merged_df["item_category"].astype("string[python]")
    merged_df["timing_bucket"] = merged_df["timing_bucket"].astype("string[python]")
    merged_df = merged_df.dropna(subset=["ticker", "cik", "filing_date", "effective_model_date"]).copy()
    merged_df = merged_df.sort_values(
        ["ticker", "effective_model_date", "accession_number", "item_number"]
    ).reset_index(drop=True)

    output_columns = [
        "ticker",
        "cik",
        "filing_date",
        "filing_timestamp",
        "effective_model_date",
        "item_number",
        "form_type",
        "item_category",
        "timing_bucket",
        "accession_number",
    ]
    structured_item_df = merged_df[output_columns].copy()
    diagnostics = {
        "raw_submission_rows": int(len(submission_items_df)),
        "filing_rows_with_items": filing_rows_with_items,
        "structured_item_rows": int(len(structured_item_df)),
        "mapped_item_rows": int(structured_item_df["item_category"].notna().sum()),
        "unmatched_item_rows": unmatched_item_rows,
    }
    return structured_item_df, diagnostics


def build_daily_feature_table(
    structured_item_df: pd.DataFrame,
    panel_dates_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate grouped 8-K item events into daily ticker-date features."""
    panel = panel_dates_df.copy().sort_values(["ticker", "date"]).reset_index(drop=True)

    if structured_item_df.empty:
        for column in SEC_8K_GROUPED_EVENT_FEATURE_COLUMNS:
            panel[column] = np.nan
        zero_columns = [
            "sec_8k_earnings_results_today",
            "sec_8k_guidance_outlook_today",
            "sec_8k_leadership_governance_today",
            "sec_8k_financing_securities_today",
            "sec_8k_material_agreement_mna_today",
            "sec_8k_regulatory_legal_today",
            "sec_8k_item_count_1d",
            "sec_8k_item_count_5d",
        ]
        for column in zero_columns:
            panel[column] = 0.0
        return panel[["ticker", "date"] + SEC_8K_GROUPED_EVENT_FEATURE_COLUMNS].copy()

    items = structured_item_df.copy().rename(columns={"effective_model_date": "date"})
    items["date"] = pd.to_datetime(items["date"], errors="coerce")

    daily_item_counts = (
        items.groupby(["ticker", "date"], as_index=False)
        .agg(sec_8k_item_count_1d=("item_number", "size"))
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    mapped_items = items.loc[items["item_category"].isin(CATEGORY_NAMES)].copy()
    mapped_items["category_count"] = 1
    daily_category_counts = (
        mapped_items.groupby(["ticker", "date", "item_category"], as_index=False)
        .agg(category_count=("category_count", "sum"))
    )
    if daily_category_counts.empty:
        category_pivot = pd.DataFrame(columns=["ticker", "date"])
    else:
        category_pivot = (
            daily_category_counts.pivot_table(
                index=["ticker", "date"],
                columns="item_category",
                values="category_count",
                fill_value=0,
                aggfunc="sum",
            )
            .reset_index()
            .rename_axis(columns=None)
        )

    panel = panel.merge(daily_item_counts, on=["ticker", "date"], how="left", validate="one_to_one")
    panel = panel.merge(category_pivot, on=["ticker", "date"], how="left", validate="one_to_one")
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)

    panel["sec_8k_item_count_1d"] = pd.to_numeric(panel["sec_8k_item_count_1d"], errors="coerce").fillna(0.0)
    for category_name in CATEGORY_NAMES:
        if category_name not in panel.columns:
            panel[category_name] = 0.0
        panel[category_name] = pd.to_numeric(panel[category_name], errors="coerce").fillna(0.0)
        panel[CATEGORY_TODAY_COLUMNS[category_name]] = (panel[category_name] > 0).astype("int64")

    group = panel.groupby("ticker", group_keys=False)
    panel["sec_8k_item_count_5d"] = (
        group["sec_8k_item_count_1d"]
        .rolling(window=5, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    helper_days_since = {}
    for category_name in CATEGORY_NAMES:
        helper_event_date_column = f"{category_name}_event_date"
        helper_last_date_column = f"last_{category_name}_event_date"
        helper_days_since_column = f"{category_name}_days_since"

        panel[helper_event_date_column] = panel["date"].where(panel[CATEGORY_TODAY_COLUMNS[category_name]] > 0)
        panel[helper_last_date_column] = group[helper_event_date_column].ffill()
        panel[helper_days_since_column] = (
            panel["date"] - pd.to_datetime(panel[helper_last_date_column], errors="coerce")
        ).dt.days.astype("float64")
        panel[CATEGORY_DECAY_COLUMNS[category_name]] = np.where(
            panel[helper_days_since_column].notna(),
            np.exp(-panel[helper_days_since_column] / 3.0),
            np.nan,
        )
        helper_days_since[category_name] = helper_days_since_column

    for category_name, output_column in CATEGORY_DAYS_SINCE_COLUMNS.items():
        panel[output_column] = panel[helper_days_since[category_name]]

    output = panel[["ticker", "date"] + SEC_8K_GROUPED_EVENT_FEATURE_COLUMNS].copy()
    output = output.sort_values(["ticker", "date"]).reset_index(drop=True)
    return output


def print_mapping_summary(mapping_df: pd.DataFrame) -> None:
    """Print a compact summary of the reused universe mapping table."""
    print("\nSEC Universe Mapping Reuse Summary")
    print("-" * 60)
    print(f"Rows: {len(mapping_df):,}")
    print(f"Tickers: {mapping_df['ticker'].nunique():,}")


def print_item_summary(structured_item_df: pd.DataFrame, diagnostics: dict[str, int]) -> None:
    """Print audit diagnostics for the structured 8-K item table."""
    print("\nStructured 8-K Item Table Summary")
    print("-" * 60)
    print(f"Raw 8-K submission rows fetched: {diagnostics['raw_submission_rows']:,}")
    print(f"8-K filing rows with any parseable items: {diagnostics['filing_rows_with_items']:,}")
    print(f"Structured 8-K item rows: {diagnostics['structured_item_rows']:,}")
    print(f"Mapped category item rows: {diagnostics['mapped_item_rows']:,}")
    print(f"Unmatched item rows dropped after EXP-008 timing merge: {diagnostics['unmatched_item_rows']:,}")

    if structured_item_df.empty:
        return

    print(f"Tickers: {structured_item_df['ticker'].nunique():,}")
    print(
        f"Date range: {structured_item_df['filing_date'].min().date()} to "
        f"{structured_item_df['filing_date'].max().date()}"
    )

    print("\nMapped item frequencies")
    print("-" * 60)
    mapped_counts = structured_item_df.loc[structured_item_df["item_category"].notna(), "item_number"]
    for item_number, count in mapped_counts.value_counts().sort_index().items():
        print(f"{item_number:<8} {count:>8,}")


def print_feature_summary(feature_df: pd.DataFrame) -> None:
    """Print audit diagnostics for the grouped 8-K feature layer."""
    missingness = (
        feature_df[SEC_8K_GROUPED_EVENT_FEATURE_COLUMNS]
        .isna()
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    print("\nEvent V1 SEC 8-K Grouped Feature Summary")
    print("-" * 60)
    print(f"Rows: {len(feature_df):,}")
    print(f"Tickers: {feature_df['ticker'].nunique():,}")
    print(f"Date range: {feature_df['date'].min().date()} to {feature_df['date'].max().date()}")
    print(f"Feature count: {len(SEC_8K_GROUPED_EVENT_FEATURE_COLUMNS)}")

    print("\nCategory event frequencies")
    print("-" * 60)
    for category_name in CATEGORY_NAMES:
        today_column = CATEGORY_TODAY_COLUMNS[category_name]
        print(f"{today_column:<40} {int((feature_df[today_column] > 0).sum()):>8,}")

    print("\nTop missingness columns")
    print("-" * 60)
    for column_name, percentage in missingness.head(12).items():
        print(f"{column_name:<40} {percentage:>8.2f}%")


def main() -> None:
    """Build and save grouped 8-K event features."""
    args = parse_args()
    ensure_event_v1_directories()

    panel_path = Path(args.panel_path)
    mapping_path = Path(args.mapping_path)
    metadata_path = Path(args.metadata_path)
    output_path = Path(args.output_path)

    print(f"Loading Layer 1 panel dates from: {panel_path}")
    panel_dates_df = load_panel_dates(panel_path)

    if mapping_path.exists():
        print(f"Loading EXP-008 SEC mapping table from: {mapping_path}")
        mapping_df = load_sec_mapping(mapping_path)
    else:
        print("EXP-008 SEC mapping table not found locally; rebuilding the universe mapping.")
        requester = SecRequester(user_agent=get_sec_user_agent(), min_delay_seconds=0.25)
        mapping_df = build_universe_sec_mapping(
            requester=requester,
            universe_tickers=get_layer1_tickers(),
        )

    print(f"Loading EXP-008 SEC filing metadata from: {metadata_path}")
    filing_metadata_df = load_sec_filing_metadata(metadata_path)

    requester = SecRequester(user_agent=get_sec_user_agent(), min_delay_seconds=0.25)
    print(f"Fetching structured 8-K item metadata for {len(mapping_df)} universe tickers...")
    submission_items_df = fetch_8k_submission_items(
        requester=requester,
        mapping_df=mapping_df,
    )

    structured_item_df, item_diagnostics = build_structured_item_table(
        submission_items_df=submission_items_df,
        filing_metadata_df=filing_metadata_df,
    )
    feature_df = build_daily_feature_table(
        structured_item_df=structured_item_df,
        panel_dates_df=panel_dates_df,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_parquet(output_path, index=False)

    print_mapping_summary(mapping_df)
    print_item_summary(structured_item_df, item_diagnostics)
    print(f"\nSaved grouped 8-K features to: {output_path}")
    print_feature_summary(feature_df)
    print("\nSaved event_v1 SEC 8-K grouped features.")


if __name__ == "__main__":
    main()
