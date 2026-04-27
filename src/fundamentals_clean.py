"""Clean raw EDGAR fundamentals into a quarterly-style wide table.

This script reads the long-format raw fundamentals file created by
`src.edgar_pull`, reshapes the concept rows into columns, keeps one
row per `ticker + period_end`, and saves the result as an interim file.

Input:
    data/raw/fundamentals/raw_fundamentals.parquet

Output:
    data/interim/fundamentals/fundamentals_quarterly_clean.parquet

Notes:
    - This cleaner keeps both 10-Q and 10-K filings because that was the
      chosen project assumption.
    - If the same ticker, period, and concept appear more than once,
      the latest filing_date is kept.
    - Missing concepts stay as NaN. No imputation happens here.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import List

import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.accounting_concepts import (
    CONCEPT_SPECS,
    EXPECTED_CONCEPT_COLUMNS,
    export_concept_map,
)
from src.fundamental_fact_selector import select_fundamental_facts
from src.paths import INTERIM_DATA_DIR, QUARTERLY_OUTPUTS_DIAGNOSTICS_DIR, RAW_DATA_DIR

RAW_REQUIRED_COLUMNS = [
    "ticker",
    "cik",
    "filing_date",
    "accepted_datetime",
    "period_end",
    "fiscal_period",
    "fiscal_year",
    "form_type",
    "concept_name",
    "value",
    "unit",
    "raw_tag",
    "source",
]

RAW_OPTIONAL_METADATA_COLUMNS = [
    "raw_tag",
    "accepted_datetime",
    "accession_number",
    "start_date",
    "end_date",
    "frame",
    "fact_duration_days",
    "effective_model_date",
    "selection_reason",
    "derivation_reason",
]

RAW_VALIDATION_METADATA_COLUMNS = [
    "raw_tag",
    "accession_number",
    "start_date",
    "end_date",
    "frame",
    "fact_duration_days",
]

METADATA_COLUMNS = [
    "ticker",
    "cik",
    "filing_date",
    "period_end",
    "fiscal_period",
    "fiscal_year",
    "form_type",
]

TIMING_TABLE_COLUMNS = [
    "ticker",
    "cik",
    "accession_number",
    "filing_date",
    "accepted_datetime",
    "form_type",
    "period_end",
    "fiscal_year",
    "fiscal_period",
    "effective_model_date",
    "timing_source",
    "timing_assumption",
]

PERIOD_KEY_COLUMNS = [
    "ticker",
    "cik",
    "period_end",
]

ALLOWED_FORM_TYPES = {"10-Q", "10-K", "10-Q/A", "10-K/A"}

def get_input_path() -> Path:
    """Return the raw parquet input path."""
    return RAW_DATA_DIR / "fundamentals" / "raw_fundamentals.parquet"


def get_output_path() -> Path:
    """Return the clean parquet output path and create its folder."""
    output_dir = INTERIM_DATA_DIR / "fundamentals"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "fundamentals_quarterly_clean.parquet"


def load_raw_fundamentals(input_path: Path) -> pd.DataFrame:
    """Load the raw fundamentals parquet and validate expected columns."""
    if not input_path.exists():
        raise FileNotFoundError(f"Raw fundamentals file was not found: {input_path}")

    df = pd.read_parquet(input_path)

    required_without_optional = [column for column in RAW_REQUIRED_COLUMNS if column not in RAW_OPTIONAL_METADATA_COLUMNS]
    missing_columns = [column for column in required_without_optional if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "Raw fundamentals file is missing required columns: "
            + ", ".join(missing_columns)
        )

    return df.copy()


def normalize_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize key column types before reshaping."""
    cleaned = df.copy()

    missing_optional = [column for column in RAW_OPTIONAL_METADATA_COLUMNS if column not in cleaned.columns]
    missing_validation_metadata = [column for column in RAW_VALIDATION_METADATA_COLUMNS if column not in cleaned.columns]
    if missing_optional:
        if missing_validation_metadata:
            warnings.warn(
                "Raw fundamentals are missing SEC/XBRL metadata columns required for full "
                "point-in-time validation: "
                + ", ".join(missing_validation_metadata)
                + ". Existing rows will load, but a fresh raw pull is required for complete validation.",
                stacklevel=2,
            )
        for column in missing_optional:
            cleaned[column] = pd.NA

    output_columns = list(dict.fromkeys(RAW_REQUIRED_COLUMNS + RAW_OPTIONAL_METADATA_COLUMNS))
    cleaned = cleaned[output_columns].copy()
    cleaned = cleaned[cleaned["form_type"].isin(ALLOWED_FORM_TYPES)].copy()

    cleaned["ticker"] = cleaned["ticker"].astype("string")
    cleaned["cik"] = cleaned["cik"].astype("string")
    cleaned["fiscal_period"] = cleaned["fiscal_period"].astype("string")
    cleaned["form_type"] = cleaned["form_type"].astype("string")
    cleaned["concept_name"] = cleaned["concept_name"].astype("string")
    cleaned["unit"] = cleaned["unit"].astype("string")
    cleaned["raw_tag"] = cleaned["raw_tag"].astype("string")
    cleaned["source"] = cleaned["source"].astype("string")
    cleaned["accession_number"] = cleaned["accession_number"].astype("string")
    cleaned["frame"] = cleaned["frame"].astype("string")
    cleaned["selection_reason"] = cleaned["selection_reason"].astype("string")
    cleaned["derivation_reason"] = cleaned["derivation_reason"].astype("string")

    cleaned["filing_date"] = pd.to_datetime(cleaned["filing_date"], errors="coerce")
    cleaned["accepted_datetime"] = pd.to_datetime(cleaned["accepted_datetime"], errors="coerce")
    cleaned["period_end"] = pd.to_datetime(cleaned["period_end"], errors="coerce")
    cleaned["start_date"] = pd.to_datetime(cleaned["start_date"], errors="coerce")
    cleaned["end_date"] = pd.to_datetime(cleaned["end_date"], errors="coerce")
    cleaned["effective_model_date"] = pd.to_datetime(cleaned["effective_model_date"], errors="coerce")
    cleaned["fiscal_year"] = pd.to_numeric(cleaned["fiscal_year"], errors="coerce").astype("Int64")
    cleaned["value"] = pd.to_numeric(cleaned["value"], errors="coerce")
    cleaned["fact_duration_days"] = pd.to_numeric(cleaned["fact_duration_days"], errors="coerce")
    missing_duration = cleaned["fact_duration_days"].isna() & cleaned["start_date"].notna() & cleaned["end_date"].notna()
    cleaned.loc[missing_duration, "fact_duration_days"] = (
        cleaned.loc[missing_duration, "end_date"] - cleaned.loc[missing_duration, "start_date"]
    ).dt.days + 1

    # These fields are essential for the later pivot and dedup logic.
    cleaned = cleaned.dropna(
        subset=["ticker", "cik", "filing_date", "period_end", "concept_name"]
    ).copy()

    return cleaned


def build_filing_timing_table(
    timing_metadata_df: pd.DataFrame | None = None,
    raw_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a filing timing table for point-in-time fact selection.

    SEC timing metadata is preferred because it carries the event panel's
    trading-date-aligned `effective_model_date`. If it is unavailable, this
    falls back to a conservative filing-date cutoff and labels the assumption.
    """
    if timing_metadata_df is not None and not timing_metadata_df.empty:
        timing = timing_metadata_df.copy()
        rename_map = {
            "filing_timestamp_utc": "accepted_datetime",
            "filing_timestamp_local": "accepted_datetime",
        }
        for source_col, target_col in rename_map.items():
            if source_col in timing.columns and target_col not in timing.columns:
                timing = timing.rename(columns={source_col: target_col})
                break
        for column in TIMING_TABLE_COLUMNS:
            if column not in timing.columns:
                timing[column] = pd.NA
        timing["timing_source"] = timing["timing_source"].fillna("sec_filing_metadata")
        timing["timing_assumption"] = timing["timing_assumption"].fillna("event_panel_effective_model_date")
    elif raw_df is not None and not raw_df.empty:
        timing = raw_df.copy()
        for column in TIMING_TABLE_COLUMNS:
            if column not in timing.columns:
                timing[column] = pd.NA
        timing["effective_model_date"] = timing["filing_date"]
        timing["timing_source"] = "raw_fundamentals_filing_date"
        timing["timing_assumption"] = "conservative_filing_date_cutoff_no_acceptance_timestamp"
    else:
        return pd.DataFrame(columns=TIMING_TABLE_COLUMNS)

    for column in ["filing_date", "period_end", "effective_model_date", "accepted_datetime"]:
        timing[column] = pd.to_datetime(timing[column], errors="coerce")
    for column in ["ticker", "cik", "accession_number", "form_type", "fiscal_period", "timing_source", "timing_assumption"]:
        timing[column] = timing[column].astype("string")
    timing["fiscal_year"] = pd.to_numeric(timing["fiscal_year"], errors="coerce").astype("Int64")
    timing = timing[TIMING_TABLE_COLUMNS].dropna(
        subset=["ticker", "filing_date", "form_type", "effective_model_date"]
    )
    timing = timing.sort_values(["ticker", "form_type", "filing_date", "accession_number"], na_position="last")
    dedup_subset = ["ticker", "form_type", "filing_date"]
    if timing["accession_number"].notna().any():
        dedup_subset = ["ticker", "accession_number"]
    return timing.drop_duplicates(subset=dedup_subset, keep="first").reset_index(drop=True)


def attach_effective_model_dates_to_raw(
    raw_df: pd.DataFrame,
    timing_table_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Attach filing-level effective dates to raw fact candidates before selection."""
    prepared = raw_df.copy()
    if timing_table_df is None or timing_table_df.empty:
        warnings.warn(
            "No filing timing table was supplied. Cleaner will use conservative filing-date cutoffs "
            "and mark timing validation as incomplete for rows without effective_model_date.",
            stacklevel=2,
        )
        prepared["effective_model_date"] = prepared.get("effective_model_date", pd.NaT)
        if prepared["effective_model_date"].isna().all():
            prepared["effective_model_date"] = prepared["filing_date"]
        return prepared

    timing = build_filing_timing_table(timing_metadata_df=timing_table_df)
    prepared["ticker"] = prepared["ticker"].astype("string")
    prepared["accession_number"] = prepared["accession_number"].astype("string")
    timing["ticker"] = timing["ticker"].astype("string")
    timing["accession_number"] = timing["accession_number"].astype("string")

    if prepared["accession_number"].notna().any() and timing["accession_number"].notna().any():
        merge_columns = ["ticker", "accession_number"]
    else:
        merge_columns = ["ticker", "form_type", "filing_date"]
    timing_columns = merge_columns + ["effective_model_date", "timing_source", "timing_assumption"]
    merged = prepared.drop(columns=["effective_model_date"], errors="ignore").merge(
        timing[timing_columns],
        on=merge_columns,
        how="left",
        validate="many_to_one",
    )
    missing_effective = merged["effective_model_date"].isna()
    if missing_effective.any():
        warnings.warn(
            "Some raw fact rows did not match SEC timing metadata. Using conservative filing-date "
            "cutoffs for unmatched rows and marking timing_assumption accordingly.",
            stacklevel=2,
        )
        merged.loc[missing_effective, "effective_model_date"] = merged.loc[missing_effective, "filing_date"]
        merged.loc[missing_effective, "timing_source"] = "raw_fundamentals_filing_date"
        merged.loc[missing_effective, "timing_assumption"] = "conservative_filing_date_cutoff_no_timing_match"
    return merged


def build_effective_model_date_diagnostics(
    candidate_df: pd.DataFrame,
    selected_df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize point-in-time cutoff coverage and violations."""
    candidate_effective = pd.to_datetime(candidate_df.get("effective_model_date"), errors="coerce")
    candidate_filing = pd.to_datetime(candidate_df.get("filing_date"), errors="coerce")
    selected_effective = pd.to_datetime(selected_df.get("effective_model_date"), errors="coerce")
    selected_filing = pd.to_datetime(selected_df.get("filing_date"), errors="coerce")
    return pd.DataFrame(
        [
            {
                "rows_with_effective_model_date_supplied": int(candidate_effective.notna().sum()),
                "rows_missing_effective_model_date": int(candidate_effective.isna().sum()),
                "candidate_facts_excluded_after_effective_model_date": int(
                    ((candidate_filing > candidate_effective) & candidate_effective.notna()).sum()
                ),
                "selected_facts_after_effective_model_date": int(
                    ((selected_filing > selected_effective) & selected_effective.notna()).sum()
                ),
            }
        ]
    )


def deduplicate_concept_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Select one point-in-time-safe fact for each ticker, period, and concept."""
    selected_df, removed_count, _diagnostics_df = deduplicate_concept_rows_with_diagnostics(df)
    return selected_df, removed_count


def deduplicate_concept_rows_with_diagnostics(df: pd.DataFrame) -> tuple[pd.DataFrame, int, pd.DataFrame]:
    """Select canonical facts and return rejected-candidate diagnostics."""
    result = select_fundamental_facts(df)
    selected_df = result.selected_facts.copy()
    removed_count = len(df) - len(selected_df)
    return selected_df, removed_count, result.diagnostics


def clean_fundamentals_from_raw(
    raw_df: pd.DataFrame,
    timing_table_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run normalized selection and return selected facts plus diagnostics."""
    normalized_df = normalize_raw_data(raw_df)
    if timing_table_df is not None:
        normalized_df = attach_effective_model_dates_to_raw(normalized_df, timing_table_df)
    elif normalized_df["effective_model_date"].isna().all():
        warnings.warn(
            "Raw fundamentals do not include effective_model_date and no timing table was supplied. "
            "Using conservative filing-date cutoffs; full validation requires event timing metadata.",
            stacklevel=2,
        )
        normalized_df["effective_model_date"] = normalized_df["filing_date"]
    selected_df, _removed_count, selection_diagnostics_df = deduplicate_concept_rows_with_diagnostics(normalized_df)
    timing_diagnostics_df = build_effective_model_date_diagnostics(normalized_df, selected_df)
    return selected_df, selection_diagnostics_df, timing_diagnostics_df


def build_period_metadata(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Keep one metadata row per ticker and period using latest filing_date.

    This is separate from the concept pivot on purpose. Different concepts for
    the same period can come from rows that share the same period but repeat the
    metadata. We want one clean metadata row first, then all concept values.
    """
    metadata = df[METADATA_COLUMNS].copy()
    metadata = metadata.sort_values(
        by=["ticker", "period_end", "filing_date"]
    )

    original_row_count = len(metadata)
    metadata = metadata.drop_duplicates(
        subset=["ticker", "period_end"],
        keep="last",
    ).copy()
    removed_count = original_row_count - len(metadata)

    metadata = metadata.sort_values(
        by=["ticker", "period_end", "filing_date"]
    ).reset_index(drop=True)

    return metadata, removed_count


def pivot_concepts_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long concept rows into one wide row per ticker and period_end."""
    wide = df.pivot(
        index=PERIOD_KEY_COLUMNS,
        columns="concept_name",
        values="value",
    ).reset_index()

    # After pivoting, concept names become a column index level.
    # Flattening keeps the final table simple for beginners.
    wide.columns.name = None

    return wide


def combine_metadata_and_concepts(
    metadata_df: pd.DataFrame,
    concept_wide_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge the canonical metadata row with the wide concept table."""
    final_df = metadata_df.merge(
        concept_wide_df,
        on=PERIOD_KEY_COLUMNS,
        how="left",
    )

    final_df = final_df.sort_values(
        by=["ticker", "period_end", "filing_date"]
    ).reset_index(drop=True)

    return final_df


def calculate_missing_percentages(df: pd.DataFrame, columns: List[str]) -> pd.Series:
    """Return missing-value percentages for selected columns."""
    available_columns = [column for column in columns if column in df.columns]
    if not available_columns:
        return pd.Series(dtype="float64")

    missing_percentages = df[available_columns].isna().mean().mul(100)
    return missing_percentages.sort_index()


def print_data_quality_summary(
    df: pd.DataFrame,
    concept_dedup_removed: int,
    period_dedup_removed: int,
) -> None:
    """Print a small console summary of the cleaned dataset."""
    print("\nData Quality Summary")
    print("-" * 60)
    print(f"Number of rows: {len(df):,}")
    print(f"Number of tickers: {df['ticker'].nunique():,}")

    min_period_end = df["period_end"].min()
    max_period_end = df["period_end"].max()
    print(f"Date range: {min_period_end.date()} to {max_period_end.date()}")

    major_fields = METADATA_COLUMNS + [
        column for column in EXPECTED_CONCEPT_COLUMNS if column in df.columns
    ]
    missing_percentages = calculate_missing_percentages(df, major_fields)

    print("\nPercentage missing by major field")
    print("-" * 60)
    for column_name, percentage in missing_percentages.items():
        print(f"{column_name:<30} {percentage:>8.2f}%")

    print("\nDuplicate rows removed")
    print("-" * 60)
    print(f"Concept-level duplicates removed: {concept_dedup_removed:,}")
    print(f"Period-level duplicates removed:  {period_dedup_removed:,}")


def save_clean_fundamentals(df: pd.DataFrame, output_path: Path) -> None:
    """Save the cleaned fundamentals table to parquet."""
    df.to_parquet(output_path, index=False)


def build_coverage_diagnostics(
    concept_df: pd.DataFrame,
    clean_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build overall, ticker-level, and year-level coverage diagnostics."""
    concept_columns = [column for column in EXPECTED_CONCEPT_COLUMNS if column in clean_df.columns]
    total_rows = len(clean_df)
    period_year = clean_df["period_end"].dt.year.astype("Int64")

    overall_rows = []
    for concept_name in concept_columns:
        coverage_count = int(clean_df[concept_name].notna().sum())
        overall_rows.append(
            {
                "concept_name": concept_name,
                "preferred_unit": CONCEPT_SPECS[concept_name].preferred_unit,
                "candidate_tags": "|".join(CONCEPT_SPECS[concept_name].candidate_tags),
                "rows_with_value": coverage_count,
                "total_rows": total_rows,
                "coverage_pct": (coverage_count / total_rows * 100.0) if total_rows else 0.0,
                "missing_pct_clean_table": ((total_rows - coverage_count) / total_rows * 100.0) if total_rows else 0.0,
                "tickers_with_value": int(clean_df.loc[clean_df[concept_name].notna(), "ticker"].nunique()),
                "years_with_value": int(period_year[clean_df[concept_name].notna()].nunique()),
                "retained_raw_rows": int((concept_df["concept_name"] == concept_name).sum()),
            }
        )

    by_ticker_frames = []
    by_year_frames = []
    for concept_name in concept_columns:
        ticker_summary = (
            clean_df.assign(has_value=clean_df[concept_name].notna().astype("int64"))
            .groupby("ticker", dropna=False)
            .agg(rows_with_value=("has_value", "sum"), total_rows=("has_value", "size"))
            .reset_index()
        )
        ticker_summary["coverage_pct"] = ticker_summary["rows_with_value"] / ticker_summary["total_rows"] * 100.0
        ticker_summary["concept_name"] = concept_name
        by_ticker_frames.append(
            ticker_summary[["concept_name", "ticker", "rows_with_value", "total_rows", "coverage_pct"]]
        )

        year_summary = (
            clean_df.assign(period_year=period_year, has_value=clean_df[concept_name].notna().astype("int64"))
            .groupby("period_year", dropna=False)
            .agg(rows_with_value=("has_value", "sum"), total_rows=("has_value", "size"))
            .reset_index()
        )
        year_summary["coverage_pct"] = year_summary["rows_with_value"] / year_summary["total_rows"] * 100.0
        year_summary["concept_name"] = concept_name
        by_year_frames.append(
            year_summary[["concept_name", "period_year", "rows_with_value", "total_rows", "coverage_pct"]]
        )

    overall_df = pd.DataFrame(overall_rows).sort_values("concept_name").reset_index(drop=True)
    by_ticker_df = pd.concat(by_ticker_frames, ignore_index=True) if by_ticker_frames else pd.DataFrame()
    by_year_df = pd.concat(by_year_frames, ignore_index=True) if by_year_frames else pd.DataFrame()
    return overall_df, by_ticker_df, by_year_df


def save_coverage_diagnostics(
    concept_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    output_dir: Path = QUARTERLY_OUTPUTS_DIAGNOSTICS_DIR,
) -> dict[str, Path]:
    """Write accounting concept coverage diagnostics and the versioned concept map."""
    output_dir.mkdir(parents=True, exist_ok=True)
    overall_df, by_ticker_df, by_year_df = build_coverage_diagnostics(concept_df, clean_df)

    overall_path = output_dir / "accounting_concept_coverage.csv"
    by_ticker_path = output_dir / "accounting_concept_coverage_by_ticker.csv"
    by_year_path = output_dir / "accounting_concept_coverage_by_year.csv"
    concept_map_path = output_dir / "accounting_concept_map_v2.json"

    overall_df.to_csv(overall_path, index=False)
    by_ticker_df.to_csv(by_ticker_path, index=False)
    by_year_df.to_csv(by_year_path, index=False)
    export_concept_map(concept_map_path)

    return {
        "overall": overall_path,
        "by_ticker": by_ticker_path,
        "by_year": by_year_path,
        "concept_map": concept_map_path,
    }


def save_selection_diagnostics(
    diagnostics_df: pd.DataFrame,
    output_dir: Path = QUARTERLY_OUTPUTS_DIAGNOSTICS_DIR,
) -> Path:
    """Write fact selection diagnostics for selected and rejected candidates."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "fundamental_fact_selection_diagnostics.csv"
    diagnostics_df.to_csv(output_path, index=False)
    return output_path


def save_effective_model_date_diagnostics(
    diagnostics_df: pd.DataFrame,
    output_dir: Path = QUARTERLY_OUTPUTS_DIAGNOSTICS_DIR,
) -> Path:
    """Write cutoff wiring diagnostics for raw-to-clean fact selection."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "fundamental_effective_model_date_diagnostics.csv"
    diagnostics_df.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    """Run the fundamentals cleaning pipeline from raw long format to wide table."""
    input_path = get_input_path()
    output_path = get_output_path()

    print(f"Loading raw fundamentals from: {input_path}")
    raw_df = load_raw_fundamentals(input_path)

    print("Selecting point-in-time-safe concept rows...")
    concept_deduped_df, selection_diagnostics_df, timing_diagnostics_df = clean_fundamentals_from_raw(raw_df)
    concept_dedup_removed = len(raw_df) - len(concept_deduped_df)

    print("Selecting one metadata row per ticker + period_end...")
    metadata_df, period_dedup_removed = build_period_metadata(concept_deduped_df)

    print("Pivoting long concepts into a wide quarterly table...")
    concept_wide_df = pivot_concepts_to_wide(concept_deduped_df)

    print("Combining metadata with concept columns...")
    final_df = combine_metadata_and_concepts(metadata_df, concept_wide_df)

    print(f"Saving clean fundamentals to: {output_path}")
    save_clean_fundamentals(final_df, output_path)
    diagnostic_paths = save_coverage_diagnostics(concept_deduped_df, final_df)
    selection_diagnostics_path = save_selection_diagnostics(selection_diagnostics_df)
    timing_diagnostics_path = save_effective_model_date_diagnostics(timing_diagnostics_df)

    print_data_quality_summary(
        df=final_df,
        concept_dedup_removed=concept_dedup_removed,
        period_dedup_removed=period_dedup_removed,
    )
    print("\nCoverage diagnostics")
    print("-" * 60)
    print(f"Overall coverage: {diagnostic_paths['overall']}")
    print(f"Coverage by ticker: {diagnostic_paths['by_ticker']}")
    print(f"Coverage by year: {diagnostic_paths['by_year']}")
    print(f"Concept map: {diagnostic_paths['concept_map']}")
    print(f"Fact selection diagnostics: {selection_diagnostics_path}")
    print(f"Effective model date diagnostics: {timing_diagnostics_path}")

    print(f"\nSaved {len(final_df):,} clean rows.")


if __name__ == "__main__":
    main()
