"""Capital IQ cleaned universe loader and diagnostics.

The latest market-cap field in this source is point-in-time unsafe for
historical event modeling. Keep it in the universe artifact for current
universe construction and diagnostics only; do not add it to model features
without historical market cap by date.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Mapping

import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.paths import INTERIM_DATA_DIR, QUARTERLY_OUTPUTS_DIAGNOSTICS_DIR, RAW_DATA_DIR

LEGACY_SOURCE_PATH = Path(__file__).resolve().parent / "capitaliq_largecap_us_universe_clean.csv"
RAW_CAPITALIQ_DIR = RAW_DATA_DIR / "capitaliq"
INTERIM_CAPITALIQ_DIR = INTERIM_DATA_DIR / "capitaliq"
DEFAULT_INPUT_PATH = RAW_CAPITALIQ_DIR / "capitaliq_largecap_us_universe_clean.csv"
DEFAULT_OUTPUT_PATH = INTERIM_CAPITALIQ_DIR / "company_universe.parquet"
DEFAULT_UNIVERSE_AUDIT_PATH = QUARTERLY_OUTPUTS_DIAGNOSTICS_DIR / "capitaliq_universe_audit.csv"
DEFAULT_SECTOR_COVERAGE_PATH = QUARTERLY_OUTPUTS_DIAGNOSTICS_DIR / "capitaliq_sector_coverage.csv"
DEFAULT_CIK_RESOLUTION_AUDIT_PATH = QUARTERLY_OUTPUTS_DIAGNOSTICS_DIR / "capitaliq_cik_resolution_audit.csv"

COLUMN_MAP = {
    "Company Name": "company_name",
    "Exchange:Ticker": "exchange_ticker",
    "Ticker": "ticker",
    "Company Type": "company_type",
    "Primary Exchange": "primary_exchange",
    "Secondary Listings": "secondary_listings_raw",
    "Geographic Locations": "geographic_locations_raw",
    "Market Cap USDmm Latest": "market_cap_usdmm_latest",
    "Security Tickers": "security_tickers_raw",
    "Primary Sector": "sector",
    "Industry Classifications": "industry_classification_raw",
    "Headquarters Country/Region": "hq_country",
    "CIK": "cik_raw",
    "Equity Currency": "equity_currency",
}

REQUIRED_COLUMNS = [
    "company_name",
    "ticker",
    "exchange_ticker",
    "company_type",
    "primary_exchange",
    "sector",
    "industry_classification_raw",
    "hq_country",
    "equity_currency",
    "market_cap_usdmm_latest",
    "cik_raw",
    "cik_candidates",
    "cik_resolved",
    "cik_resolution_status",
    "secondary_listings_raw",
    "security_tickers_raw",
    "geographic_locations_raw",
]


def _clean_string(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    cleaned = re.sub(r"\s+", " ", str(value).strip())
    if not cleaned or cleaned in {"-", "--", "nan", "None", "N/A"}:
        return pd.NA
    return cleaned


def normalize_ticker(value: object) -> object:
    cleaned = _clean_string(value)
    if pd.isna(cleaned):
        return pd.NA
    return str(cleaned).upper().replace(" ", "")


def normalize_exchange_ticker(value: object) -> object:
    cleaned = _clean_string(value)
    if pd.isna(cleaned):
        return pd.NA
    parts = [part.strip() for part in str(cleaned).split(":", maxsplit=1)]
    if len(parts) == 2:
        return f"{parts[0]}:{parts[1].upper().replace(' ', '')}"
    return str(cleaned).upper().replace(" ", "")


def normalize_cik(value: object) -> object:
    cleaned = _clean_string(value)
    if pd.isna(cleaned):
        return pd.NA
    text = str(cleaned).replace(",", "")
    if re.fullmatch(r"\d+(\.0+)?", text):
        digits = text.split(".", maxsplit=1)[0]
        if 1 <= len(digits) <= 10:
            return digits.zfill(10)
    return pd.NA


def split_cik_candidates(value: object) -> list[str]:
    cleaned = _clean_string(value)
    if pd.isna(cleaned):
        return []
    candidates: list[str] = []
    for part in re.split(r";", str(cleaned)):
        normalized = normalize_cik(part)
        if not pd.isna(normalized) and str(normalized) not in candidates:
            candidates.append(str(normalized))
    return candidates


def resolve_cik_candidates(
    ticker: object,
    cik_raw: object,
    sec_ticker_to_cik: Mapping[str, str] | None = None,
) -> tuple[list[str], object, str]:
    candidates = split_cik_candidates(cik_raw)
    if not candidates:
        return [], pd.NA, "missing_cik" if pd.isna(_clean_string(cik_raw)) else "invalid_cik"
    if len(candidates) == 1:
        return candidates, candidates[0], "resolved_single"

    mapping = {key.upper(): normalize_cik(value) for key, value in (sec_ticker_to_cik or {}).items()}
    normalized_ticker = normalize_ticker(ticker)
    mapped_cik = mapping.get(str(normalized_ticker)) if not pd.isna(normalized_ticker) else None
    if mapped_cik in candidates:
        return candidates, str(mapped_cik), "resolved_sec_match"

    return candidates, candidates[0], "ambiguous_multiple"


def _standardize_industry_string(value: object) -> object:
    cleaned = _clean_string(value)
    if pd.isna(cleaned):
        return pd.NA
    parts = []
    for part in str(cleaned).split(";"):
        normalized = _clean_string(part)
        if not pd.isna(normalized):
            parts.append(str(normalized).replace(" (Primary)", "").strip())
    return "; ".join(parts) if parts else pd.NA


def derive_industry_from_classification(industry_classification_raw: object, sector: object = pd.NA) -> object:
    cleaned = _standardize_industry_string(industry_classification_raw)
    if pd.isna(cleaned):
        return pd.NA
    sector_text = str(sector).strip().lower() if not pd.isna(sector) else ""
    for part in str(cleaned).split(";"):
        candidate = part.strip()
        if candidate and candidate.lower() != sector_text:
            return candidate
    return pd.NA


def load_capitaliq_universe(
    csv_path: Path | str = DEFAULT_INPUT_PATH,
    sec_ticker_to_cik: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists() and path == DEFAULT_INPUT_PATH and LEGACY_SOURCE_PATH.exists():
        path = LEGACY_SOURCE_PATH
    raw = pd.read_csv(path, dtype="string", keep_default_na=False)
    missing = sorted(set(COLUMN_MAP).difference(raw.columns))
    if missing:
        raise ValueError(f"Capital IQ CSV is missing required columns: {', '.join(missing)}")

    df = raw.rename(columns=COLUMN_MAP)[list(COLUMN_MAP.values())].copy()
    for column in df.columns:
        if column != "market_cap_usdmm_latest":
            df[column] = df[column].map(_clean_string).astype("string")

    df["ticker"] = df["ticker"].map(normalize_ticker).astype("string")
    df["exchange_ticker"] = df["exchange_ticker"].map(normalize_exchange_ticker).astype("string")
    df["market_cap_usdmm_latest"] = pd.to_numeric(
        df["market_cap_usdmm_latest"].astype("string").str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )
    df["sector"] = df["sector"].map(_clean_string).astype("string")
    df["industry_classification_raw"] = df["industry_classification_raw"].map(_standardize_industry_string).astype("string")

    resolutions = [
        resolve_cik_candidates(ticker, cik_raw, sec_ticker_to_cik=sec_ticker_to_cik)
        for ticker, cik_raw in zip(df["ticker"], df["cik_raw"])
    ]
    df["cik_candidates"] = [";".join(candidates) for candidates, _, _ in resolutions]
    df["cik_resolved"] = pd.Series([resolved for _, resolved, _ in resolutions], dtype="string")
    df["cik_resolution_status"] = pd.Series([status for _, _, status in resolutions], dtype="string")

    return df[REQUIRED_COLUMNS].copy()


def build_universe_audit(universe_df: pd.DataFrame) -> pd.DataFrame:
    non_us = universe_df["hq_country"].astype("string").str.lower().ne("united states").fillna(False)
    non_us &= universe_df["hq_country"].notna()
    non_us |= (
        universe_df["geographic_locations_raw"].astype("string").str.contains("United States", case=False, na=False).eq(False)
        & universe_df["geographic_locations_raw"].notna()
    )
    non_us = non_us.fillna(False)
    non_us_count = int(non_us.sum())
    non_us_count = 0 if non_us_count == len(universe_df) and universe_df["hq_country"].isna().all() else non_us_count
    row = {
        "total_rows": int(len(universe_df)),
        "unique_tickers": int(universe_df["ticker"].nunique(dropna=True)),
        "missing_ticker_count": int(universe_df["ticker"].isna().sum()),
        "missing_cik_count": int(universe_df["cik_resolved"].isna().sum()),
        "missing_sector_count": int(universe_df["sector"].isna().sum()),
        "missing_industry_count": int(universe_df["industry_classification_raw"].isna().sum()),
        "duplicate_ticker_count": int(universe_df["ticker"].duplicated(keep=False).sum()),
        "duplicate_exchange_ticker_count": int(universe_df["exchange_ticker"].duplicated(keep=False).sum()),
        "non_us_rows": non_us_count,
        "non_usd_rows": int(universe_df["equity_currency"].astype("string").ne("US Dollar").fillna(False).sum()),
    }
    return pd.DataFrame([row])


def build_sector_coverage(universe_df: pd.DataFrame) -> pd.DataFrame:
    working = universe_df.copy()
    working["sector"] = working["sector"].fillna("missing_sector")
    return (
        working.groupby("sector", dropna=False)
        .agg(
            row_count=("ticker", "size"),
            unique_ticker_count=("ticker", "nunique"),
            mean_market_cap=("market_cap_usdmm_latest", "mean"),
            median_market_cap=("market_cap_usdmm_latest", "median"),
            missing_cik_count=("cik_resolved", lambda series: int(series.isna().sum())),
        )
        .reset_index()
        .sort_values(["row_count", "sector"], ascending=[False, True])
    )


def build_cik_resolution_audit(universe_df: pd.DataFrame) -> pd.DataFrame:
    return universe_df[
        ["ticker", "company_name", "cik_raw", "cik_candidates", "cik_resolved", "cik_resolution_status"]
    ].copy()


def save_capitaliq_universe(
    csv_path: Path | str = DEFAULT_INPUT_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    universe_audit_path: Path | str = DEFAULT_UNIVERSE_AUDIT_PATH,
    sector_coverage_path: Path | str = DEFAULT_SECTOR_COVERAGE_PATH,
    cik_resolution_audit_path: Path | str = DEFAULT_CIK_RESOLUTION_AUDIT_PATH,
    sec_ticker_to_cik: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    universe_df = load_capitaliq_universe(csv_path=csv_path, sec_ticker_to_cik=sec_ticker_to_cik)
    output_path = Path(output_path)
    universe_audit_path = Path(universe_audit_path)
    sector_coverage_path = Path(sector_coverage_path)
    cik_resolution_audit_path = Path(cik_resolution_audit_path)
    for path in [output_path, universe_audit_path, sector_coverage_path, cik_resolution_audit_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    universe_df.to_parquet(output_path, index=False)
    build_universe_audit(universe_df).to_csv(universe_audit_path, index=False)
    build_sector_coverage(universe_df).to_csv(sector_coverage_path, index=False)
    build_cik_resolution_audit(universe_df).to_csv(cik_resolution_audit_path, index=False)
    return universe_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Capital IQ company universe artifact.")
    parser.add_argument("--input-path", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--universe-audit-path", default=str(DEFAULT_UNIVERSE_AUDIT_PATH))
    parser.add_argument("--sector-coverage-path", default=str(DEFAULT_SECTOR_COVERAGE_PATH))
    parser.add_argument("--cik-resolution-audit-path", default=str(DEFAULT_CIK_RESOLUTION_AUDIT_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    universe_df = save_capitaliq_universe(
        csv_path=Path(args.input_path),
        output_path=Path(args.output_path),
        universe_audit_path=Path(args.universe_audit_path),
        sector_coverage_path=Path(args.sector_coverage_path),
        cik_resolution_audit_path=Path(args.cik_resolution_audit_path),
    )
    print(f"Saved Capital IQ company universe rows: {len(universe_df):,}")
    print(f"Output path: {args.output_path}")


if __name__ == "__main__":
    main()
