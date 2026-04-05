"""Download raw SEC filing text for Layer 3 sentiment work.

This script pulls 10-K and 10-Q filing text for the project universe using
SEC EDGAR via edgartools. It saves:

1. Raw text files for each filing
2. A parquet index with filing metadata and file locations

This step only collects raw text. It does not score sentiment yet.

The script is resumable:
- it saves a parquet index as it goes
- it skips filings that were already downloaded on later runs

Input:
    No local input file is required.

Outputs:
    data/raw/sec_filings/text/<ticker>/<filing>.txt
    data/raw/sec_filings/sec_filings_text_index.parquet
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.config import END_DATE, START_DATE
from src.paths import RAW_DATA_DIR
from src.universe import get_layer1_tickers

FORM_TYPES = ["10-K", "10-Q"]

OUTPUT_COLUMNS = [
    "ticker",
    "cik",
    "company_name",
    "form_type",
    "filing_date",
    "period_end",
    "accession_number",
    "text_file_path",
    "text_length",
    "source",
]


def get_output_dir() -> Path:
    """Return the raw SEC filing output folder and create it if needed."""
    output_dir = RAW_DATA_DIR / "sec_filings"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_text_dir() -> Path:
    """Return the raw text folder and create it if needed."""
    text_dir = get_output_dir() / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    return text_dir


def get_index_output_path() -> Path:
    """Return the parquet index output path."""
    return get_output_dir() / "sec_filings_text_index.parquet"


def load_existing_index(path: Path) -> pd.DataFrame:
    """Load an existing filing-text index if one already exists."""
    if not path.exists():
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = pd.read_parquet(path)
    for column in OUTPUT_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    return df[OUTPUT_COLUMNS].copy()


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


def try_import_edgartools():
    """Import edgartools lazily so the error is clearer if it is missing."""
    try:
        from edgar import Company, set_identity

        return Company, set_identity
    except ImportError as exc:
        raise ImportError(
            "edgartools is required for SEC filing text extraction. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc


def configure_edgartools_identity(set_identity_func, user_agent: str) -> None:
    """Set the SEC identity for edgartools."""
    edgar_name = os.getenv("EDGAR_NAME")
    edgar_email = os.getenv("EDGAR_EMAIL")
    organization = os.getenv("EDGAR_ORGANIZATION")

    if edgar_name and edgar_email:
        kwargs = {"name": edgar_name, "email": edgar_email}
        if organization:
            kwargs["organization"] = organization
        try:
            set_identity_func(**kwargs)
            return
        except TypeError:
            pass

    set_identity_func(user_agent)


def parse_date(date_value) -> Optional[pd.Timestamp]:
    """Safely parse a date-like value to pandas datetime."""
    if date_value is None:
        return None
    parsed = pd.to_datetime(date_value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed


def get_filing_text(filing) -> str:
    """Get the cleanest text representation available for a filing.

    `text()` is preferred because it converts the filing to plain text.
    If that fails or returns nothing, the script falls back to markdown.
    """
    text_content = ""

    try:
        text_content = filing.text()
    except Exception:
        text_content = ""

    if text_content and str(text_content).strip():
        return str(text_content)

    try:
        markdown_content = filing.markdown()
    except Exception:
        markdown_content = ""

    if markdown_content and str(markdown_content).strip():
        return str(markdown_content)

    return ""


def get_accession_number(filing) -> str:
    """Return a filing accession number from common edgartools property names."""
    accession_number = getattr(filing, "accession_number", None)
    if accession_number:
        return str(accession_number)

    accession_no = getattr(filing, "accession_no", None)
    if accession_no:
        return str(accession_no)

    return "unknown_accession"


def make_filing_key(
    ticker: str,
    accession_number: str,
    filing_date: Optional[pd.Timestamp],
    form_type: str,
) -> tuple[str, str, str]:
    """Build a stable key for deduplicating saved filings.

    Accession number is preferred. If it is unavailable, fall back to a key
    based on filing date and form type.
    """
    if accession_number and accession_number != "unknown_accession":
        return (ticker, "accession", accession_number)

    filing_date_text = str(filing_date.date()) if filing_date is not None else "unknown_date"
    return (ticker, form_type, filing_date_text)


def build_existing_keys(existing_df: pd.DataFrame) -> set[tuple[str, str, str]]:
    """Build a lookup set so reruns can skip already-downloaded filings."""
    keys: set[tuple[str, str, str]] = set()
    if existing_df.empty:
        return keys

    for _, row in existing_df.iterrows():
        filing_date = parse_date(row.get("filing_date"))
        key = make_filing_key(
            ticker=str(row.get("ticker", "")),
            accession_number=str(row.get("accession_number", "")),
            filing_date=filing_date,
            form_type=str(row.get("form_type", "")),
        )
        keys.add(key)

    return keys


def build_text_file_path(ticker: str, filing_date: pd.Timestamp, form_type: str, accession_number: str) -> Path:
    """Build a stable local text path for a filing."""
    safe_form_type = form_type.replace("/", "_")
    filename = f"{filing_date.date()}_{safe_form_type}_{accession_number}.txt"
    ticker_dir = get_text_dir() / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)
    return ticker_dir / filename


def fetch_company_filings(company, ticker: str) -> List[object]:
    """Fetch the company's 10-K and 10-Q filings within the project date range.

    Reasonable assumption:
    only original 10-K and 10-Q filings are pulled for the first Layer 3 pass.
    Amendments are skipped to keep the raw text set cleaner and easier to align.
    """
    filings = company.get_filings(form=FORM_TYPES).filter(filing_date=f"{START_DATE}:{END_DATE}")
    return list(filings)


def fetch_and_save_text_rows(
    ticker: str,
    company,
    existing_keys: set[tuple[str, str, str]],
) -> List[dict]:
    """Fetch filing text for one company and save raw text files locally."""
    rows = []
    filings = fetch_company_filings(company, ticker)
    skipped_existing = 0
    recovered_from_disk = 0

    print(f"{ticker}: found {len(filings)} filings")
    for filing in filings:
        try:
            form_type = str(getattr(filing, "form", ""))
            filing_date = parse_date(getattr(filing, "filing_date", None))
            period_end = parse_date(getattr(filing, "period_of_report", None))
            cik = str(getattr(filing, "cik", ""))
            company_name = str(getattr(filing, "company", getattr(filing, "company_name", "")))
            accession_number = get_accession_number(filing)

            if not filing_date:
                continue

            text_file_path = build_text_file_path(
                ticker=ticker,
                filing_date=filing_date,
                form_type=form_type,
                accession_number=accession_number,
            )
            filing_key = make_filing_key(
                ticker=ticker,
                accession_number=accession_number,
                filing_date=filing_date,
                form_type=form_type,
            )
            if filing_key in existing_keys:
                skipped_existing += 1
                continue

            # If a previous interrupted run already wrote the raw text file,
            # reuse it and rebuild the metadata row instead of downloading again.
            if text_file_path.exists():
                rows.append(
                    {
                        "ticker": ticker,
                        "cik": cik,
                        "company_name": company_name,
                        "form_type": form_type,
                        "filing_date": filing_date,
                        "period_end": period_end,
                        "accession_number": accession_number,
                        "text_file_path": str(text_file_path),
                        "text_length": text_file_path.stat().st_size,
                        "source": "sec_edgar_edgartools",
                    }
                )
                recovered_from_disk += 1
                continue

            text_content = get_filing_text(filing)
            if not text_content.strip():
                print(f"{ticker}: empty text skipped for {accession_number}")
                continue

            text_file_path.write_text(text_content, encoding="utf-8")

            rows.append(
                {
                    "ticker": ticker,
                    "cik": cik,
                    "company_name": company_name,
                    "form_type": form_type,
                    "filing_date": filing_date,
                    "period_end": period_end,
                    "accession_number": accession_number,
                    "text_file_path": str(text_file_path),
                    "text_length": len(text_content),
                    "source": "sec_edgar_edgartools",
                }
            )
        except Exception as exc:
            print(f"{ticker}: failed on one filing -> {exc}")

    print(
        f"{ticker}: saved or recovered {len(rows)} filings, "
        f"recovered {recovered_from_disk} from disk, "
        f"skipped {skipped_existing} existing"
    )
    return rows


def build_index_dataframe(rows: List[dict]) -> pd.DataFrame:
    """Create the final filing-text index DataFrame."""
    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = pd.DataFrame(rows)
    df = df[OUTPUT_COLUMNS].copy()
    df["ticker"] = df["ticker"].astype("string")
    df["cik"] = df["cik"].astype("string")
    df["company_name"] = df["company_name"].astype("string")
    df["form_type"] = df["form_type"].astype("string")
    df["accession_number"] = df["accession_number"].astype("string")
    df["text_file_path"] = df["text_file_path"].astype("string")
    df["source"] = df["source"].astype("string")
    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    df["text_length"] = pd.to_numeric(df["text_length"], errors="coerce")

    df = df.drop_duplicates(subset=["ticker", "accession_number"]).copy()
    df = df.sort_values(["ticker", "filing_date", "form_type"]).reset_index(drop=True)
    return df


def print_summary(df: pd.DataFrame) -> None:
    """Print a compact summary of the raw filing-text pull."""
    print("\nSEC Filing Text Summary")
    print("-" * 60)
    print(f"Number of filings: {len(df):,}")
    print(f"Number of tickers: {df['ticker'].nunique():,}")

    if not df.empty:
        print(f"Date range: {df['filing_date'].min().date()} to {df['filing_date'].max().date()}")
        print("\nFilings by form type")
        print("-" * 60)
        form_counts = df["form_type"].value_counts().sort_index()
        for form_type, count in form_counts.items():
            print(f"{form_type:<10} {count:>8,}")

        print("\nAverage text length by form type")
        print("-" * 60)
        avg_lengths = df.groupby("form_type")["text_length"].mean().sort_index()
        for form_type, avg_length in avg_lengths.items():
            print(f"{form_type:<10} {avg_length:>12.0f}")


def main() -> None:
    """Download raw 10-K and 10-Q text for the project universe."""
    Company, set_identity = try_import_edgartools()
    user_agent = get_sec_user_agent()
    configure_edgartools_identity(set_identity, user_agent)

    tickers = get_layer1_tickers()
    index_output_path = get_index_output_path()
    existing_index_df = load_existing_index(index_output_path)
    existing_keys = build_existing_keys(existing_index_df)
    all_rows: List[dict] = existing_index_df.to_dict("records")

    if existing_index_df.empty:
        print("No existing filing-text index found. Starting fresh.")
    else:
        print(f"Loaded existing filing-text index with {len(existing_index_df):,} filings. Resume mode is on.")

    print(f"Processing {len(tickers)} tickers from src.universe...")
    for ticker in tickers:
        try:
            company = Company(ticker)
            rows = fetch_and_save_text_rows(
                ticker=ticker,
                company=company,
                existing_keys=existing_keys,
            )
            if rows:
                all_rows.extend(rows)
                current_index_df = build_index_dataframe(all_rows)
                current_index_df.to_parquet(index_output_path, index=False)
                existing_keys = build_existing_keys(current_index_df)
                all_rows = current_index_df.to_dict("records")
                print(f"{ticker}: checkpoint saved to {index_output_path}")
        except Exception as exc:
            print(f"{ticker}: failed to fetch filings -> {exc}")

    index_df = build_index_dataframe(all_rows)
    index_df.to_parquet(index_output_path, index=False)

    print(f"\nSaved filing-text index to: {index_output_path}")
    print_summary(index_df)


if __name__ == "__main__":
    main()
