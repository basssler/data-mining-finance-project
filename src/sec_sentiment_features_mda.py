"""Score MD&A sections from SEC filings with FinBERT for Layer 3 refinement.

This script refines the first Layer 3 SEC sentiment pass by attempting to
extract only the MD&A section from original 10-K and 10-Q filings before
running FinBERT.

The MD&A extraction is heuristic:
- 10-K: Item 7 to Item 7A or Item 8
- 10-Q: Item 2 to Item 3 or Item 4

The script is resumable:
- if an output parquet already exists, previously processed filings are skipped
- rows are checkpointed to disk every few filings

Unlike the full-filing scorer, this script records a row even when MD&A
extraction fails. That makes coverage easy to audit.

Input:
    data/raw/sec_filings/sec_filings_text_index.parquet

Output:
    data/interim/sentiment/sec_filing_sentiment_mda.parquet
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.paths import INTERIM_DATA_DIR
from src.sec_sentiment_features import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHECKPOINT_EVERY,
    DEFAULT_DEVICE,
    MODEL_NAME,
    build_existing_accessions,
    chunk_text,
    filter_original_filings,
    get_input_index_path,
    load_filing_index,
    load_finbert,
    normalize_whitespace,
    read_filing_text,
    score_chunks,
)

DEFAULT_MIN_MDA_CHARS = 500

OUTPUT_COLUMNS = [
    "ticker",
    "cik",
    "company_name",
    "form_type",
    "filing_date",
    "period_end",
    "accession_number",
    "text_file_path",
    "raw_text_length",
    "mda_text_length",
    "chunk_count",
    "mda_found",
    "extraction_status",
    "extraction_method",
    "positive_prob",
    "negative_prob",
    "neutral_prob",
    "sentiment_score",
    "dominant_sentiment",
    "model_name",
    "source",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line options for testing, resuming, or full runs."""
    parser = argparse.ArgumentParser(
        description="Create MD&A-only SEC filing sentiment features with FinBERT."
    )
    parser.add_argument(
        "--max-filings",
        type=int,
        default=None,
        help="Optional cap for a small test run.",
    )
    parser.add_argument(
        "--max-chunks-per-filing",
        type=int,
        default=None,
        help="Optional cap on chunk count for a quick test run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of text chunks scored at once.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY,
        help="How often to checkpoint progress to parquet.",
    )
    parser.add_argument(
        "--min-mda-chars",
        type=int,
        default=DEFAULT_MIN_MDA_CHARS,
        help="Minimum cleaned MD&A section length required before scoring.",
    )
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Ignore any existing MD&A output and process everything again.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default=DEFAULT_DEVICE,
        help="Device preference for FinBERT scoring.",
    )
    return parser.parse_args()


def get_output_dir() -> Path:
    """Return the Layer 3 sentiment output folder and create it if needed."""
    output_dir = INTERIM_DATA_DIR / "sentiment"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_output_path() -> Path:
    """Return the MD&A sentiment parquet output path."""
    return get_output_dir() / "sec_filing_sentiment_mda.parquet"


def load_existing_output(path: Path, rescore: bool) -> pd.DataFrame:
    """Load existing MD&A sentiment output so reruns can resume safely."""
    if rescore or not path.exists():
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = pd.read_parquet(path)
    for column in OUTPUT_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    return df[OUTPUT_COLUMNS].copy()


def normalize_for_section_search(text: str) -> str:
    """Lightly normalize filing text before MD&A section boundary search."""
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00a0": " ",
        "\u2013": "-",
        "\u2014": "-",
    }
    normalized = text
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    return normalized


def find_all_spans(patterns: list[str], text: str, start_pos: int = 0) -> list[tuple[int, int]]:
    """Return all absolute match spans across several patterns."""
    spans: list[tuple[int, int]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text[start_pos:], flags=re.IGNORECASE | re.DOTALL):
            spans.append((start_pos + match.start(), start_pos + match.end()))

    spans = sorted(set(spans), key=lambda item: item[0])
    return spans


def get_mda_patterns(form_type: str) -> tuple[list[str], list[str], str]:
    """Return start and end regex patterns for the filing type's MD&A section."""
    if form_type == "10-K":
        start_patterns = [
            r"ITEM\s*7\.?\s+MANAGEMENT(?:['`]\s*|\s+)S\s+DISCUSSION\s+AND\s+ANALYSIS\s+OF\s+FINANCIAL\s+CONDITION\s+AND\s+RESULTS\s+OF\s+OPERATIONS",
            r"ITEM\s*7\.?\s+MANAGEMENT.{0,40}DISCUSSION\s+AND\s+ANALYSIS",
        ]
        end_patterns = [
            r"ITEM\s*7A\.?\s+QUANTITATIVE\s+AND\s+QUALITATIVE\s+DISCLOSURES\s+ABOUT\s+MARKET\s+RISK",
            r"ITEM\s*8\.?\s+FINANCIAL\s+STATEMENTS\s+AND\s+SUPPLEMENTARY\s+DATA",
        ]
        return start_patterns, end_patterns, "item7_to_item7a_or_item8"

    start_patterns = [
        r"ITEM\s*2\.?\s+MANAGEMENT(?:['`]\s*|\s+)S\s+DISCUSSION\s+AND\s+ANALYSIS\s+OF\s+FINANCIAL\s+CONDITION\s+AND\s+RESULTS\s+OF\s+OPERATIONS",
        r"ITEM\s*2\.?\s+MANAGEMENT.{0,40}DISCUSSION\s+AND\s+ANALYSIS",
    ]
    end_patterns = [
        r"ITEM\s*3\.?\s+QUANTITATIVE\s+AND\s+QUALITATIVE\s+DISCLOSURES\s+ABOUT\s+MARKET\s+RISK",
        r"ITEM\s*4\.?\s+CONTROLS\s+AND\s+PROCEDURES",
    ]
    return start_patterns, end_patterns, "item2_to_item3_or_item4"


def extract_mda_section(
    text: str,
    form_type: str,
    min_mda_chars: int,
) -> tuple[Optional[str], str, str]:
    """Extract the MD&A section from a filing using item-boundary heuristics."""
    searchable_text = normalize_for_section_search(text)
    start_patterns, end_patterns, extraction_method = get_mda_patterns(form_type)

    start_spans = find_all_spans(start_patterns, searchable_text)
    if not start_spans:
        return None, "section_not_found", extraction_method

    end_spans = find_all_spans(end_patterns, searchable_text)
    if not end_spans:
        return None, "section_end_not_found", extraction_method

    best_short_candidate: Optional[str] = None
    best_short_length = -1

    for start_start, start_end in start_spans:
        next_end = next((end_start for end_start, _ in end_spans if end_start > start_end), None)
        if next_end is None:
            continue

        section_text = searchable_text[start_start:next_end]
        cleaned_section = normalize_whitespace(section_text)
        if not cleaned_section:
            continue

        section_length = len(cleaned_section)
        if section_length >= min_mda_chars:
            return cleaned_section, "scored", extraction_method

        if section_length > best_short_length:
            best_short_candidate = cleaned_section
            best_short_length = section_length

    if best_short_candidate is not None:
        return best_short_candidate, "section_too_short", extraction_method

    return None, "section_end_not_found", extraction_method


def score_one_filing_mda(
    filing_row: pd.Series,
    torch,
    tokenizer,
    model,
    device,
    batch_size: int,
    max_chunks_per_filing: Optional[int],
    min_mda_chars: int,
) -> dict:
    """Read, extract, and score the MD&A section for one filing."""
    text_path = Path(str(filing_row["text_file_path"]))
    raw_text = read_filing_text(text_path)
    raw_text_length = len(raw_text)

    mda_text, extraction_status, extraction_method = extract_mda_section(
        text=raw_text,
        form_type=str(filing_row["form_type"]),
        min_mda_chars=min_mda_chars,
    )

    base_row = {
        "ticker": str(filing_row["ticker"]),
        "cik": str(filing_row["cik"]),
        "company_name": str(filing_row["company_name"]),
        "form_type": str(filing_row["form_type"]),
        "filing_date": pd.to_datetime(filing_row["filing_date"], errors="coerce"),
        "period_end": pd.to_datetime(filing_row["period_end"], errors="coerce"),
        "accession_number": str(filing_row["accession_number"]),
        "text_file_path": str(filing_row["text_file_path"]),
        "raw_text_length": raw_text_length,
        "mda_text_length": pd.NA,
        "chunk_count": pd.NA,
        "mda_found": 0,
        "extraction_status": extraction_status,
        "extraction_method": extraction_method,
        "positive_prob": pd.NA,
        "negative_prob": pd.NA,
        "neutral_prob": pd.NA,
        "sentiment_score": pd.NA,
        "dominant_sentiment": pd.NA,
        "model_name": MODEL_NAME,
        "source": "sec_edgar_finbert_mda",
    }

    if mda_text is None:
        return base_row

    base_row["mda_text_length"] = len(mda_text)
    if len(mda_text) < min_mda_chars:
        base_row["extraction_status"] = "section_too_short"
        return base_row

    chunks = chunk_text(
        text=mda_text,
        tokenizer=tokenizer,
        max_tokens=510,
        max_chunks_per_filing=max_chunks_per_filing,
    )
    if not chunks:
        base_row["extraction_status"] = "no_valid_chunks"
        return base_row

    positive_prob, negative_prob, neutral_prob = score_chunks(
        chunks=chunks,
        torch=torch,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=batch_size,
    )
    sentiment_score = positive_prob - negative_prob

    dominant_sentiment = "neutral"
    if positive_prob >= negative_prob and positive_prob >= neutral_prob:
        dominant_sentiment = "positive"
    elif negative_prob >= positive_prob and negative_prob >= neutral_prob:
        dominant_sentiment = "negative"

    base_row.update(
        {
            "chunk_count": len(chunks),
            "mda_found": 1,
            "extraction_status": "scored",
            "positive_prob": positive_prob,
            "negative_prob": negative_prob,
            "neutral_prob": neutral_prob,
            "sentiment_score": sentiment_score,
            "dominant_sentiment": dominant_sentiment,
        }
    )
    return base_row


def build_output_dataframe(rows: list[dict]) -> pd.DataFrame:
    """Create a consistent MD&A sentiment DataFrame."""
    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = pd.DataFrame(rows)
    for column in OUTPUT_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA

    df = df[OUTPUT_COLUMNS].copy()

    string_columns = [
        "ticker",
        "cik",
        "company_name",
        "form_type",
        "accession_number",
        "text_file_path",
        "extraction_status",
        "extraction_method",
        "dominant_sentiment",
        "model_name",
        "source",
    ]
    for column in string_columns:
        df[column] = df[column].astype("string")

    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")

    numeric_columns = [
        "raw_text_length",
        "mda_text_length",
        "chunk_count",
        "mda_found",
        "positive_prob",
        "negative_prob",
        "neutral_prob",
        "sentiment_score",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.drop_duplicates(subset=["ticker", "accession_number"]).copy()
    df = df.sort_values(["ticker", "filing_date", "form_type"]).reset_index(drop=True)
    return df


def checkpoint_results(rows: list[dict], output_path: Path) -> pd.DataFrame:
    """Save current MD&A sentiment results to parquet and return the saved frame."""
    output_df = build_output_dataframe(rows)
    output_df.to_parquet(output_path, index=False)
    return output_df


def print_summary(df: pd.DataFrame) -> None:
    """Print a compact summary of MD&A extraction and scoring coverage."""
    print("\nSEC MD&A Sentiment Summary")
    print("-" * 60)
    print(f"Number of filings processed: {len(df):,}")
    print(f"Number of tickers:          {df['ticker'].nunique():,}")

    if df.empty:
        return

    print(f"Date range:                 {df['filing_date'].min().date()} to {df['filing_date'].max().date()}")
    print(f"Filings with MD&A scored:   {int(df['mda_found'].fillna(0).sum()):,}")
    print(f"Filings without usable MD&A: {len(df) - int(df['mda_found'].fillna(0).sum()):,}")

    print("\nExtraction status counts")
    print("-" * 60)
    status_counts = df["extraction_status"].value_counts(dropna=False).sort_index()
    for status, count in status_counts.items():
        print(f"{str(status):<24} {count:>8,}")

    scored_df = df[df["mda_found"] == 1].copy()
    if scored_df.empty:
        return

    print("\nAverage MD&A sentiment fields")
    print("-" * 60)
    print(f"Positive probability:       {scored_df['positive_prob'].mean():.4f}")
    print(f"Negative probability:       {scored_df['negative_prob'].mean():.4f}")
    print(f"Neutral probability:        {scored_df['neutral_prob'].mean():.4f}")
    print(f"Sentiment score:            {scored_df['sentiment_score'].mean():.4f}")
    print(f"Average MD&A chunk count:   {scored_df['chunk_count'].mean():.2f}")
    print(f"Average MD&A text length:   {scored_df['mda_text_length'].mean():.0f}")


def main() -> None:
    """Create MD&A-only SEC filing sentiment features from raw filing text."""
    args = parse_args()
    input_index_path = get_input_index_path()
    output_path = get_output_path()

    print(f"Loading filing-text index from: {input_index_path}")
    filing_index = load_filing_index(input_index_path)
    filing_index = filter_original_filings(filing_index)

    if args.max_filings is not None:
        filing_index = filing_index.head(args.max_filings).copy()

    existing_output_df = load_existing_output(output_path, rescore=args.rescore)
    existing_accessions = build_existing_accessions(existing_output_df)
    all_rows = existing_output_df.to_dict("records")

    if existing_output_df.empty or args.rescore:
        print("No existing MD&A sentiment output loaded. Starting fresh.")
    else:
        print(f"Loaded existing MD&A sentiment output with {len(existing_output_df):,} processed filings.")

    remaining_filings = filing_index[
        ~filing_index["accession_number"].astype(str).isin(existing_accessions)
    ].copy()

    print(f"Original 10-K / 10-Q filings available: {len(filing_index):,}")
    print(f"Filings remaining to process:           {len(remaining_filings):,}")

    if remaining_filings.empty:
        print("\nEverything is already processed. Nothing to do.")
        print_summary(build_output_dataframe(all_rows))
        return

    torch, tokenizer, model, device = load_finbert(
        model_name=MODEL_NAME,
        requested_device=args.device,
    )

    processed_since_checkpoint = 0
    for row_number, (_, filing_row) in enumerate(remaining_filings.iterrows(), start=1):
        accession_number = str(filing_row["accession_number"])
        ticker = str(filing_row["ticker"])
        print(
            f"Processing {row_number:,}/{len(remaining_filings):,}: "
            f"{ticker} {filing_row['form_type']} {filing_row['filing_date'].date()} "
            f"({accession_number})"
        )

        try:
            scored_row = score_one_filing_mda(
                filing_row=filing_row,
                torch=torch,
                tokenizer=tokenizer,
                model=model,
                device=device,
                batch_size=args.batch_size,
                max_chunks_per_filing=args.max_chunks_per_filing,
                min_mda_chars=args.min_mda_chars,
            )
            all_rows.append(scored_row)
            processed_since_checkpoint += 1
        except Exception as exc:
            print(f"{ticker}: failed to process filing {accession_number} -> {exc}")
            continue

        if processed_since_checkpoint >= args.checkpoint_every:
            checkpoint_df = checkpoint_results(all_rows, output_path)
            processed_since_checkpoint = 0
            print(f"Checkpoint saved to: {output_path} ({len(checkpoint_df):,} filings)")

    final_df = checkpoint_results(all_rows, output_path)
    print(f"\nSaved MD&A filing sentiment to: {output_path}")
    print_summary(final_df)


if __name__ == "__main__":
    main()
