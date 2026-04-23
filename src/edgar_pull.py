"""Download raw financial statement facts for a fixed ticker universe.

This script is the Layer 1 fundamentals downloader for the project.
It tries to use `edgartools` first because that is the preferred SEC EDGAR
interface for this project. If `edgartools` is not installed or fails for a
company, it falls back to the SEC Company Facts JSON API on `data.sec.gov`.

Output:
    data/raw/fundamentals/raw_fundamentals.parquet

Saved columns:
    ticker
    cik
    filing_date
    period_end
    fiscal_period
    fiscal_year
    form_type
    concept_name
    value
    unit
    source
"""

from __future__ import annotations

import json
import os
import sys
import time
import gzip
import zlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.config import END_DATE, START_DATE
from src.accounting_concepts import CONCEPT_SPECS
from src.paths import RAW_DATA_DIR
from src.universe import get_layer1_tickers

ALLOWED_FORM_TYPES = {"10-K", "10-Q", "10-K/A", "10-Q/A"}

CONCEPT_CANDIDATES: Dict[str, List[str]] = {
    canonical_name: list(spec.candidate_tags)
    for canonical_name, spec in CONCEPT_SPECS.items()
}

OUTPUT_COLUMNS = [
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


@dataclass
class SecRequester:
    """Small helper for polite SEC requests with a custom user agent."""

    user_agent: str
    min_delay_seconds: float = 0.25
    timeout_seconds: int = 30
    _last_request_time: float = 0.0

    def wait_if_needed(self) -> None:
        """Sleep briefly so requests stay polite and spaced out."""
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
                "Accept-Encoding": "gzip, deflate",
                "Host": "data.sec.gov" if "data.sec.gov" in url else "www.sec.gov",
            },
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                self._last_request_time = time.time()
                raw_bytes = response.read()
                content_encoding = response.headers.get("Content-Encoding", "").lower()

                if content_encoding == "gzip":
                    raw_bytes = gzip.decompress(raw_bytes)
                elif content_encoding == "deflate":
                    raw_bytes = zlib.decompress(raw_bytes)

                return json.loads(raw_bytes.decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(f"HTTP error {exc.code} for URL: {url}") from exc
        except URLError as exc:
            raise RuntimeError(f"Network error for URL: {url}") from exc


def get_output_path() -> Path:
    """Return the final parquet output path and ensure the folder exists."""
    output_dir = RAW_DATA_DIR / "fundamentals"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "raw_fundamentals.parquet"


def get_sec_user_agent() -> str:
    """Build a SEC-friendly user agent from environment variables when possible."""
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


def load_ticker_to_cik_map(requester: SecRequester) -> Dict[str, str]:
    """Load the SEC ticker-to-CIK mapping file."""
    url = "https://www.sec.gov/files/company_tickers.json"
    payload = requester.get_json(url)

    ticker_map: Dict[str, str] = {}
    for record in payload.values():
        ticker = str(record.get("ticker", "")).upper().strip()
        cik_str = str(record.get("cik_str", "")).strip()
        if ticker and cik_str:
            ticker_map[ticker] = cik_str.zfill(10)
    return ticker_map


def parse_date(date_text: Optional[str]) -> Optional[pd.Timestamp]:
    """Safely parse a date string into a pandas timestamp."""
    if not date_text:
        return None
    return pd.to_datetime(date_text, errors="coerce")


def in_date_window(
    filing_date: Optional[pd.Timestamp],
    period_end: Optional[pd.Timestamp],
) -> bool:
    """Keep rows within the official project horizon."""
    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)

    if filing_date is not None and pd.notna(filing_date):
        if filing_date < start or filing_date > end:
            return False

    if period_end is not None and pd.notna(period_end):
        if period_end < start or period_end > end:
            return False

    return True


def normalize_record(record: dict) -> dict:
    """Convert one raw fact into the standard output column format."""
    filing_date = parse_date(record.get("filing_date"))
    period_end = parse_date(record.get("period_end"))

    return {
        "ticker": record.get("ticker"),
        "cik": record.get("cik"),
        "filing_date": filing_date,
        "period_end": period_end,
        "fiscal_period": record.get("fiscal_period"),
        "fiscal_year": record.get("fiscal_year"),
        "form_type": record.get("form_type"),
        "concept_name": record.get("concept_name"),
        "value": record.get("value"),
        "unit": record.get("unit"),
        "raw_tag": record.get("raw_tag"),
        "source": record.get("source"),
    }


def fetch_company_facts_via_sec(
    ticker: str,
    cik: str,
    requester: SecRequester,
) -> List[dict]:
    """Download raw facts directly from the SEC Company Facts endpoint."""
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    payload = requester.get_json(url)
    us_gaap_facts = payload.get("facts", {}).get("us-gaap", {})

    rows: List[dict] = []

    for concept_name, candidate_tags in CONCEPT_CANDIDATES.items():
        concept_found = False

        for tag in candidate_tags:
            tag_payload = us_gaap_facts.get(tag)
            if not tag_payload:
                continue

            concept_found = True
            units = tag_payload.get("units", {})

            for unit_name, facts in units.items():
                for fact in facts:
                    form_type = fact.get("form")
                    if form_type not in ALLOWED_FORM_TYPES:
                        continue

                    filing_date = parse_date(fact.get("filed"))
                    period_end = parse_date(fact.get("end"))
                    if not in_date_window(filing_date, period_end):
                        continue

                    rows.append(
                        normalize_record(
                            {
                                "ticker": ticker,
                                "cik": cik,
                                "filing_date": fact.get("filed"),
                                "period_end": fact.get("end"),
                                "fiscal_period": fact.get("fp"),
                                "fiscal_year": fact.get("fy"),
                                "form_type": form_type,
                                "concept_name": concept_name,
                                "value": fact.get("val"),
                                "unit": unit_name,
                                "raw_tag": tag,
                                "source": "sec_companyfacts",
                            }
                        )
                    )

        if not concept_found:
            print(f"{ticker}: concept missing -> {concept_name}")

    return rows


def try_import_edgartools():
    """Import edgartools lazily so the script still works without it."""
    try:
        from edgar import Company, set_identity

        return Company, set_identity
    except ImportError:
        return None, None


def configure_edgartools_identity(set_identity_func, user_agent: str) -> None:
    """Set identity for edgartools using environment values when available."""
    edgar_name = os.getenv("EDGAR_NAME")
    edgar_email = os.getenv("EDGAR_EMAIL")
    organization = os.getenv("EDGAR_ORGANIZATION")

    if edgar_name and edgar_email:
        kwargs = {"name": edgar_name, "email": edgar_email}
        if organization:
            kwargs["organization"] = organization
        set_identity_func(**kwargs)
        return

    set_identity_func(user_agent)


def first_matching_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    """Return the first available column name from a list of candidates."""
    column_lookup = {str(col).lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in column_lookup:
            return column_lookup[candidate.lower()]
    return None




def fetch_company_facts_via_edgartools(
    ticker: str,
    cik: str,
    user_agent: str,
) -> List[dict]:
    """Try to fetch facts through edgartools and return standardized rows.

    This function is intentionally defensive because `edgartools` may change
    its DataFrame column names across versions.
    """
    Company, set_identity = try_import_edgartools()
    if Company is None or set_identity is None:
        raise RuntimeError("edgartools is not installed.")

    configure_edgartools_identity(set_identity, user_agent)

    company = Company(ticker)
    facts = company.facts
    rows: List[dict] = []

    for concept_name, candidate_tags in CONCEPT_CANDIDATES.items():
        concept_rows_found = False

        for tag in candidate_tags:
            try:
                df = (
                    facts.query()
                    .by_concept(tag)
                    .to_dataframe()
                )
            except Exception:
                continue

            if df is None or df.empty:
                continue

            concept_rows_found = True

            filing_col = first_matching_column(df.columns, ["filing_date", "filed"])
            end_col = first_matching_column(df.columns, ["period_end", "end"])
            fp_col = first_matching_column(df.columns, ["fiscal_period", "fp"])
            fy_col = first_matching_column(df.columns, ["fiscal_year", "fy"])
            form_col = first_matching_column(df.columns, ["form_type", "form"])
            value_col = first_matching_column(df.columns, ["numeric_value", "value", "val"])
            unit_col = first_matching_column(df.columns, ["unit", "units"])

            if not filing_col or not form_col or not value_col:
                continue

            for _, row in df.iterrows():
                form_type = row.get(form_col)
                if form_type not in ALLOWED_FORM_TYPES:
                    continue

                filing_date = parse_date(row.get(filing_col))
                period_end = parse_date(row.get(end_col)) if end_col else None
                if not in_date_window(filing_date, period_end):
                    continue

                rows.append(
                    normalize_record(
                        {
                            "ticker": ticker,
                            "cik": cik,
                            "filing_date": filing_date,
                            "period_end": period_end,
                            "fiscal_period": row.get(fp_col) if fp_col else None,
                            "fiscal_year": row.get(fy_col) if fy_col else None,
                            "form_type": form_type,
                            "concept_name": concept_name,
                            "value": row.get(value_col),
                            "unit": row.get(unit_col) if unit_col else None,
                            "raw_tag": tag,
                            "source": "edgartools_companyfacts",
                        }
                    )
                )

        if not concept_rows_found:
            print(f"{ticker}: concept missing in edgartools -> {concept_name}")

    return rows


def fetch_ticker_facts(
    ticker: str,
    cik: str,
    requester: SecRequester,
) -> List[dict]:
    """Fetch one company's facts, preferring edgartools and falling back to SEC JSON."""
    try:
        rows = fetch_company_facts_via_edgartools(
            ticker=ticker,
            cik=cik,
            user_agent=requester.user_agent,
        )
        if rows:
            print(f"{ticker}: downloaded {len(rows)} rows via edgartools")
            return rows
        print(f"{ticker}: edgartools returned no rows, falling back to SEC JSON")
    except Exception as exc:
        print(f"{ticker}: edgartools failed, falling back to SEC JSON -> {exc}")

    rows = fetch_company_facts_via_sec(
        ticker=ticker,
        cik=cik,
        requester=requester,
    )
    print(f"{ticker}: downloaded {len(rows)} rows via SEC JSON")
    return rows


def build_dataframe(rows: List[dict]) -> pd.DataFrame:
    """Create the final DataFrame and apply basic cleanup."""
    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = pd.DataFrame(rows)
    df = df[OUTPUT_COLUMNS].copy()

    df["ticker"] = df["ticker"].astype("string")
    df["cik"] = df["cik"].astype("string")
    df["fiscal_period"] = df["fiscal_period"].astype("string")
    df["form_type"] = df["form_type"].astype("string")
    df["concept_name"] = df["concept_name"].astype("string")
    df["unit"] = df["unit"].astype("string")
    df["raw_tag"] = df["raw_tag"].astype("string")
    df["source"] = df["source"].astype("string")

    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    df["fiscal_year"] = pd.to_numeric(df["fiscal_year"], errors="coerce").astype("Int64")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna(subset=["ticker", "cik", "filing_date", "concept_name", "value"])
    df = df.drop_duplicates()
    df = df.sort_values(["ticker", "concept_name", "filing_date", "period_end"]).reset_index(drop=True)
    return df


def save_parquet(df: pd.DataFrame, output_path: Path) -> None:
    """Write the final raw fundamentals table to parquet."""
    df.to_parquet(output_path, index=False)


def main() -> None:
    """Run the raw fundamentals pull for the Layer 1 ticker universe."""
    start_time = datetime.now()
    user_agent = get_sec_user_agent()
    requester = SecRequester(user_agent=user_agent, min_delay_seconds=0.25)
    tickers = get_layer1_tickers()

    print("Loading SEC ticker to CIK mapping...")
    ticker_to_cik = load_ticker_to_cik_map(requester)

    all_rows: List[dict] = []
    print(f"Processing {len(tickers)} tickers from src.universe...")
    for ticker in tickers:
        cik = ticker_to_cik.get(ticker)
        if not cik:
            print(f"{ticker}: missing CIK mapping, skipping")
            continue

        print(f"Processing {ticker} ({cik})...")
        rows = fetch_ticker_facts(ticker=ticker, cik=cik, requester=requester)
        all_rows.extend(rows)

    df = build_dataframe(all_rows)
    output_path = get_output_path()
    save_parquet(df, output_path)

    elapsed = datetime.now() - start_time
    print(f"Saved {len(df):,} rows to: {output_path}")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Elapsed: {elapsed}")


if __name__ == "__main__":
    main()
