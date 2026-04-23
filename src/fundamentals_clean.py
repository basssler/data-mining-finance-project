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
    concept_priority_lookup,
    export_concept_map,
    source_priority,
)
from src.paths import INTERIM_DATA_DIR, QUARTERLY_OUTPUTS_DIAGNOSTICS_DIR, RAW_DATA_DIR

RAW_REQUIRED_COLUMNS = [
    "ticker",
    "cik",
    "filing_date",
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

METADATA_COLUMNS = [
    "ticker",
    "cik",
    "filing_date",
    "period_end",
    "fiscal_period",
    "fiscal_year",
    "form_type",
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

    missing_columns = [column for column in RAW_REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "Raw fundamentals file is missing required columns: "
            + ", ".join(missing_columns)
        )

    return df.copy()


def normalize_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize key column types before reshaping."""
    cleaned = df.copy()

    if "raw_tag" not in cleaned.columns:
        cleaned["raw_tag"] = pd.NA

    cleaned = cleaned[RAW_REQUIRED_COLUMNS].copy()
    cleaned = cleaned[cleaned["form_type"].isin(ALLOWED_FORM_TYPES)].copy()

    cleaned["ticker"] = cleaned["ticker"].astype("string")
    cleaned["cik"] = cleaned["cik"].astype("string")
    cleaned["fiscal_period"] = cleaned["fiscal_period"].astype("string")
    cleaned["form_type"] = cleaned["form_type"].astype("string")
    cleaned["concept_name"] = cleaned["concept_name"].astype("string")
    cleaned["unit"] = cleaned["unit"].astype("string")
    cleaned["raw_tag"] = cleaned["raw_tag"].astype("string")
    cleaned["source"] = cleaned["source"].astype("string")

    cleaned["filing_date"] = pd.to_datetime(cleaned["filing_date"], errors="coerce")
    cleaned["period_end"] = pd.to_datetime(cleaned["period_end"], errors="coerce")
    cleaned["fiscal_year"] = pd.to_numeric(cleaned["fiscal_year"], errors="coerce").astype("Int64")
    cleaned["value"] = pd.to_numeric(cleaned["value"], errors="coerce")

    # These fields are essential for the later pivot and dedup logic.
    cleaned = cleaned.dropna(
        subset=["ticker", "cik", "filing_date", "period_end", "concept_name"]
    ).copy()

    return cleaned


def deduplicate_concept_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Keep the latest filing_date for each ticker, period_end, and concept.

    This resolves repeated raw fact rows before the wide pivot. The tie-break
    rule follows the project assumption: keep the most recently filed value.
    """
    deduped = df.copy()
    priority_lookup = concept_priority_lookup()
    deduped["_preferred_unit_rank"] = deduped.apply(
        lambda row: 0
        if (
            CONCEPT_SPECS.get(str(row["concept_name"])) is not None
            and str(row["unit"]) == CONCEPT_SPECS[str(row["concept_name"])].preferred_unit
        )
        else 1,
        axis=1,
    )
    deduped["_tag_priority_rank"] = deduped.apply(
        lambda row: priority_lookup.get(str(row["concept_name"]), {}).get(str(row["raw_tag"]), 999),
        axis=1,
    )
    deduped["_source_priority_rank"] = deduped["source"].map(source_priority).astype("Int64")
    deduped = deduped.sort_values(
        by=[
            "ticker",
            "period_end",
            "concept_name",
            "_preferred_unit_rank",
            "_tag_priority_rank",
            "_source_priority_rank",
            "filing_date",
        ],
        ascending=[True, True, True, True, True, True, False],
    ).copy()

    original_row_count = len(deduped)
    deduped = deduped.drop_duplicates(subset=["ticker", "period_end", "concept_name"], keep="first").copy()
    removed_count = original_row_count - len(deduped)
    deduped = deduped.drop(
        columns=["_preferred_unit_rank", "_tag_priority_rank", "_source_priority_rank"],
        errors="ignore",
    )

    return deduped, removed_count


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


def main() -> None:
    """Run the fundamentals cleaning pipeline from raw long format to wide table."""
    input_path = get_input_path()
    output_path = get_output_path()

    print(f"Loading raw fundamentals from: {input_path}")
    raw_df = load_raw_fundamentals(input_path)

    print("Normalizing raw data types...")
    normalized_df = normalize_raw_data(raw_df)

    print("Deduplicating repeated concept rows...")
    concept_deduped_df, concept_dedup_removed = deduplicate_concept_rows(normalized_df)

    print("Selecting one metadata row per ticker + period_end...")
    metadata_df, period_dedup_removed = build_period_metadata(concept_deduped_df)

    print("Pivoting long concepts into a wide quarterly table...")
    concept_wide_df = pivot_concepts_to_wide(concept_deduped_df)

    print("Combining metadata with concept columns...")
    final_df = combine_metadata_and_concepts(metadata_df, concept_wide_df)

    print(f"Saving clean fundamentals to: {output_path}")
    save_clean_fundamentals(final_df, output_path)
    diagnostic_paths = save_coverage_diagnostics(concept_deduped_df, final_df)

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

    print(f"\nSaved {len(final_df):,} clean rows.")


if __name__ == "__main__":
    main()
