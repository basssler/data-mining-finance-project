"""Refresh metadata-complete SEC Company Facts raw fundamentals for the golden sample."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src import edgar_pull
from src.universe import get_universe_v2_tickers
from src.paths import RAW_DATA_DIR


DEFAULT_TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "TSLA", "KR"]
DEFAULT_OUTPUT_PATH = RAW_DATA_DIR / "fundamentals" / "golden_raw_fundamentals.parquet"
DEFAULT_STATUS_PATH = RAW_DATA_DIR / "fundamentals" / "golden_raw_fundamentals_status.csv"
REQUIRED_METADATA_COLUMNS = [
    "accession_number",
    "start_date",
    "end_date",
    "frame",
    "fact_duration_days",
    "filing_date",
    "form_type",
    "unit",
    "raw_tag",
]

CIK_OVERRIDES = {
    "MMC": "0000062709",
}


def refresh_golden_raw_fundamentals(
    tickers: list[str],
    output_path: Path,
    status_path: Path,
    *,
    min_delay_seconds: float = 0.25,
    resume: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    requester = edgar_pull.SecRequester(user_agent=edgar_pull.get_sec_user_agent(), min_delay_seconds=min_delay_seconds)
    ticker_to_cik = edgar_pull.load_ticker_to_cik_map(requester)
    ticker_to_cik.update(CIK_OVERRIDES)
    rows: list[dict] = []
    status_rows: list[dict] = []
    existing_df = pd.DataFrame()
    existing_status = pd.DataFrame()
    completed_tickers: set[str] = set()
    if resume and output_path.exists():
        existing_df = pd.read_parquet(output_path)
        rows.extend(existing_df.to_dict("records"))
    if resume and status_path.exists():
        existing_status = pd.read_csv(status_path)
        if not existing_status.empty and "status" in existing_status.columns:
            completed_tickers = set(
                existing_status.loc[existing_status["status"].eq("ok"), "ticker"].astype(str).str.upper()
            )
            status_rows.extend(existing_status.loc[existing_status["status"].eq("ok")].to_dict("records"))

    for ticker in tickers:
        normalized_ticker = ticker.upper().strip()
        if normalized_ticker in completed_tickers:
            continue
        cik = ticker_to_cik.get(normalized_ticker)
        if not cik:
            status_rows.append(
                {
                    "ticker": normalized_ticker,
                    "status": "missing_cik_mapping",
                    "cik": pd.NA,
                    "row_count": 0,
                    "metadata_complete": False,
                    "message": "Ticker was not found in SEC company_tickers mapping.",
                }
            )
            continue
        try:
            ticker_rows = edgar_pull.fetch_company_facts_via_sec(normalized_ticker, cik, requester)
            rows.extend(ticker_rows)
            row_df = edgar_pull.build_dataframe(ticker_rows)
            metadata_complete = all(column in row_df.columns and row_df[column].notna().any() for column in REQUIRED_METADATA_COLUMNS)
            status_rows.append(
                {
                    "ticker": normalized_ticker,
                    "status": "ok" if len(row_df) else "no_rows",
                    "cik": cik,
                    "row_count": len(row_df),
                    "metadata_complete": metadata_complete,
                    "message": "",
                }
            )
        except Exception as exc:
            status_rows.append(
                {
                    "ticker": normalized_ticker,
                    "status": "fetch_failed",
                    "cik": cik,
                    "row_count": 0,
                    "metadata_complete": False,
                    "message": str(exc),
                }
            )

    output_df = edgar_pull.build_dataframe(rows)
    if not output_df.empty:
        output_df = output_df.drop_duplicates().sort_values(["ticker", "filing_date", "period_end", "concept_name"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(output_path, index=False)
    status_df = pd.DataFrame(status_rows)
    status_df.to_csv(status_path, index=False)
    return output_df, status_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh SEC metadata-complete raw fundamentals for golden sample tickers.")
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    parser.add_argument("--tickers-path", default="")
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--status-path", default=str(DEFAULT_STATUS_PATH))
    parser.add_argument("--min-delay-seconds", type=float, default=0.25)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tickers_path:
        tickers = get_universe_v2_tickers(Path(args.tickers_path))
    else:
        tickers = [ticker.strip().upper() for ticker in str(args.tickers).split(",") if ticker.strip()]
    output_df, status_df = refresh_golden_raw_fundamentals(
        tickers,
        Path(args.output_path),
        Path(args.status_path),
        min_delay_seconds=float(args.min_delay_seconds),
        resume=bool(args.resume),
    )
    print("Golden raw fundamentals refresh complete")
    print(f"Rows: {len(output_df):,}")
    print(f"Output: {args.output_path}")
    print(f"Status: {args.status_path}")
    print(status_df.to_string(index=False))


if __name__ == "__main__":
    main()
