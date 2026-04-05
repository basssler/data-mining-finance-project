"""Score SEC filing text with FinBERT to create Layer 3 sentiment features.

This script reads the raw SEC filing-text index created by
`src/sec_filing_text_pull.py`, filters to original 10-K and 10-Q filings,
loads each filing's raw text from disk, and scores the filing with FinBERT.

Because SEC filings are much longer than FinBERT's token limit, the script:
1. cleans the raw filing text,
2. splits it into tokenizer-sized chunks,
3. scores each chunk,
4. averages chunk probabilities into one filing-level sentiment result.

The script is resumable:
- if an output parquet already exists, previously scored filings are skipped
- progress is checkpointed to disk every few filings

Input:
    data/raw/sec_filings/sec_filings_text_index.parquet

Output:
    data/interim/sentiment/sec_filing_sentiment.parquet
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.config import END_DATE, START_DATE
from src.paths import INTERIM_DATA_DIR, RAW_DATA_DIR

MODEL_NAME = "ProsusAI/finbert"
ORIGINAL_FORM_TYPES = ["10-K", "10-Q"]
CHUNK_TOKEN_LIMIT = 510
DEFAULT_BATCH_SIZE = 8
DEFAULT_CHECKPOINT_EVERY = 10
DEFAULT_DEVICE = "auto"

INDEX_REQUIRED_COLUMNS = [
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
    "clean_text_length",
    "chunk_count",
    "positive_prob",
    "negative_prob",
    "neutral_prob",
    "sentiment_score",
    "dominant_sentiment",
    "model_name",
    "source",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line options for small test runs or resumed runs."""
    parser = argparse.ArgumentParser(
        description="Create filing-level SEC sentiment features with FinBERT."
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
        help=(
            "Optional cap on chunks scored per filing. "
            "Useful for quick tests, but leave unset for a full run."
        ),
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
        help="How often to save progress to parquet while scoring.",
    )
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Ignore any existing output file and score everything again.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default=DEFAULT_DEVICE,
        help=(
            "Device preference for FinBERT scoring. "
            "`auto` uses CUDA when available, otherwise CPU."
        ),
    )
    return parser.parse_args()


def try_import_transformers():
    """Import torch and transformers lazily with a clear installation error."""
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        return torch, AutoTokenizer, AutoModelForSequenceClassification
    except ImportError as exc:
        raise ImportError(
            "FinBERT scoring requires `torch` and `transformers`. "
            "Install them with `pip install -r requirements.txt`."
        ) from exc


def get_input_index_path() -> Path:
    """Return the raw filing-text index path."""
    return RAW_DATA_DIR / "sec_filings" / "sec_filings_text_index.parquet"


def get_output_dir() -> Path:
    """Return the Layer 3 sentiment output folder and create it if needed."""
    output_dir = INTERIM_DATA_DIR / "sentiment"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_output_path() -> Path:
    """Return the filing-level sentiment parquet output path."""
    return get_output_dir() / "sec_filing_sentiment.parquet"


def load_filing_index(path: Path) -> pd.DataFrame:
    """Load the raw filing-text index and validate required columns."""
    if not path.exists():
        raise FileNotFoundError(
            f"Input filing-text index was not found: {path}\n"
            "Run `python -m src.sec_filing_text_pull` first."
        )

    df = pd.read_parquet(path)
    missing_columns = sorted(set(INDEX_REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(
            "Filing-text index is missing required columns: "
            + ", ".join(missing_columns)
        )

    df = df[INDEX_REQUIRED_COLUMNS].copy()
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

    return df


def filter_original_filings(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only original 10-K and 10-Q filings for the first sentiment pass."""
    filtered = df[df["form_type"].isin(ORIGINAL_FORM_TYPES)].copy()
    filtered = filtered[
        (filtered["filing_date"] >= pd.Timestamp(START_DATE))
        & (filtered["filing_date"] <= pd.Timestamp(END_DATE))
    ].copy()
    filtered = filtered.sort_values(["ticker", "filing_date", "form_type"]).reset_index(drop=True)
    return filtered


def load_existing_output(path: Path, rescore: bool) -> pd.DataFrame:
    """Load existing filing sentiment so reruns can resume safely."""
    if rescore or not path.exists():
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = pd.read_parquet(path)
    for column in OUTPUT_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    return df[OUTPUT_COLUMNS].copy()


def build_existing_accessions(df: pd.DataFrame) -> set[str]:
    """Return already-scored accession numbers for resume mode."""
    if df.empty or "accession_number" not in df.columns:
        return set()

    valid = df["accession_number"].dropna().astype(str)
    return {value for value in valid if value}


def normalize_whitespace(text: str) -> str:
    """Collapse noisy SEC whitespace so chunking is more stable."""
    text = text.replace("\x00", " ")
    text = text.replace("\r", " ")
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n\s+", "\n", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def read_filing_text(path: Path) -> str:
    """Read a raw filing text file from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Raw filing text file was not found: {path}")
    return path.read_text(encoding="utf-8", errors="ignore")


def resolve_torch_device(torch, requested_device: str):
    """Choose a torch device based on user preference and CUDA availability."""
    cuda_available = torch.cuda.is_available()

    if requested_device == "cpu":
        return torch.device("cpu")

    if requested_device == "cuda":
        if not cuda_available:
            raise RuntimeError(
                "You requested `--device cuda`, but PyTorch does not see a CUDA GPU. "
                "Check `torch.cuda.is_available()` and your PyTorch install."
            )
        return torch.device("cuda")

    if cuda_available:
        return torch.device("cuda")
    return torch.device("cpu")


def load_finbert(model_name: str = MODEL_NAME, requested_device: str = DEFAULT_DEVICE):
    """Load the FinBERT tokenizer and model on the requested device."""
    torch, AutoTokenizer, AutoModelForSequenceClassification = try_import_transformers()

    print(f"Loading FinBERT model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    device = resolve_torch_device(torch, requested_device=requested_device)
    model.to(device)
    model.eval()

    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
        cuda_version = getattr(torch.version, "cuda", None)
        print(f"FinBERT ready on CUDA device: {device_name}")
        if cuda_version:
            print(f"PyTorch CUDA runtime: {cuda_version}")
    else:
        print("FinBERT ready on CPU.")

    return torch, tokenizer, model, device


def chunk_text(
    text: str,
    tokenizer,
    max_tokens: int = CHUNK_TOKEN_LIMIT,
    max_chunks_per_filing: Optional[int] = None,
) -> list[str]:
    """Split a long filing into FinBERT-sized text chunks.

    We tokenize first so the chunk size matches the model's real token limit.
    Each chunk is then decoded back to text so it can be fed through the model
    in small batches.
    """
    # Tokenize without calling the model-facing encode path on the full filing.
    # That avoids noisy max-length warnings for very long SEC documents because
    # we only send chunk-sized slices through the model later.
    token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
    if not token_ids:
        return []

    chunks: list[str] = []
    for start in range(0, len(token_ids), max_tokens):
        if max_chunks_per_filing is not None and len(chunks) >= max_chunks_per_filing:
            break
        chunk_ids = token_ids[start : start + max_tokens]
        chunk_text = tokenizer.decode(
            chunk_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks


def get_label_index_map(model) -> dict[str, int]:
    """Normalize FinBERT label names into a lowercase lookup map."""
    label_map: dict[str, int] = {}
    for idx, label in model.config.id2label.items():
        label_map[str(label).lower()] = int(idx)
    return label_map


def batched(items: Iterable[str], batch_size: int) -> Iterable[list[str]]:
    """Yield a list of items at a fixed batch size."""
    batch: list[str] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def score_chunks(
    chunks: list[str],
    torch,
    tokenizer,
    model,
    device,
    batch_size: int,
) -> tuple[float, float, float]:
    """Score chunk text with FinBERT and return mean class probabilities."""
    label_index_map = get_label_index_map(model)
    positive_idx = label_index_map["positive"]
    negative_idx = label_index_map["negative"]
    neutral_idx = label_index_map["neutral"]

    all_probabilities: list[np.ndarray] = []
    for batch in batched(chunks, batch_size=batch_size):
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=CHUNK_TOKEN_LIMIT + 2,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.no_grad():
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()

        all_probabilities.append(probabilities)

    stacked = np.vstack(all_probabilities)
    mean_probabilities = stacked.mean(axis=0)

    positive_prob = float(mean_probabilities[positive_idx])
    negative_prob = float(mean_probabilities[negative_idx])
    neutral_prob = float(mean_probabilities[neutral_idx])
    return positive_prob, negative_prob, neutral_prob


def score_one_filing(
    filing_row: pd.Series,
    torch,
    tokenizer,
    model,
    device,
    batch_size: int,
    max_chunks_per_filing: Optional[int],
) -> Optional[dict]:
    """Read, clean, chunk, and score one filing."""
    text_path = Path(str(filing_row["text_file_path"]))
    raw_text = read_filing_text(text_path)
    clean_text = normalize_whitespace(raw_text)
    if not clean_text:
        print(f"{filing_row['ticker']}: empty filing text skipped -> {filing_row['accession_number']}")
        return None

    chunks = chunk_text(
        text=clean_text,
        tokenizer=tokenizer,
        max_tokens=CHUNK_TOKEN_LIMIT,
        max_chunks_per_filing=max_chunks_per_filing,
    )
    if not chunks:
        print(f"{filing_row['ticker']}: no valid chunks -> {filing_row['accession_number']}")
        return None

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

    return {
        "ticker": str(filing_row["ticker"]),
        "cik": str(filing_row["cik"]),
        "company_name": str(filing_row["company_name"]),
        "form_type": str(filing_row["form_type"]),
        "filing_date": pd.to_datetime(filing_row["filing_date"], errors="coerce"),
        "period_end": pd.to_datetime(filing_row["period_end"], errors="coerce"),
        "accession_number": str(filing_row["accession_number"]),
        "text_file_path": str(filing_row["text_file_path"]),
        "text_length": pd.to_numeric(filing_row["text_length"], errors="coerce"),
        "clean_text_length": len(clean_text),
        "chunk_count": len(chunks),
        "positive_prob": positive_prob,
        "negative_prob": negative_prob,
        "neutral_prob": neutral_prob,
        "sentiment_score": sentiment_score,
        "dominant_sentiment": dominant_sentiment,
        "model_name": MODEL_NAME,
        "source": "sec_edgar_finbert",
    }


def build_output_dataframe(rows: list[dict]) -> pd.DataFrame:
    """Create a consistent filing-level sentiment DataFrame."""
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
        "dominant_sentiment",
        "model_name",
        "source",
    ]
    for column in string_columns:
        df[column] = df[column].astype("string")

    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")

    numeric_columns = [
        "text_length",
        "clean_text_length",
        "chunk_count",
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
    """Save current sentiment results to parquet and return the saved frame."""
    output_df = build_output_dataframe(rows)
    output_df.to_parquet(output_path, index=False)
    return output_df


def print_summary(df: pd.DataFrame) -> None:
    """Print a compact summary of the filing-level sentiment output."""
    print("\nSEC Filing Sentiment Summary")
    print("-" * 60)
    print(f"Number of filings scored: {len(df):,}")
    print(f"Number of tickers:        {df['ticker'].nunique():,}")

    if df.empty:
        return

    print(f"Date range:               {df['filing_date'].min().date()} to {df['filing_date'].max().date()}")

    print("\nFilings by form type")
    print("-" * 60)
    form_counts = df["form_type"].value_counts().sort_index()
    for form_type, count in form_counts.items():
        print(f"{form_type:<10} {count:>8,}")

    print("\nAverage sentiment fields")
    print("-" * 60)
    print(f"Positive probability:     {df['positive_prob'].mean():.4f}")
    print(f"Negative probability:     {df['negative_prob'].mean():.4f}")
    print(f"Neutral probability:      {df['neutral_prob'].mean():.4f}")
    print(f"Sentiment score:          {df['sentiment_score'].mean():.4f}")
    print(f"Average chunk count:      {df['chunk_count'].mean():.2f}")

    print("\nDominant sentiment counts")
    print("-" * 60)
    dominant_counts = df["dominant_sentiment"].value_counts().sort_index()
    for sentiment_label, count in dominant_counts.items():
        print(f"{sentiment_label:<10} {count:>8,}")


def main() -> None:
    """Create filing-level sentiment features from raw SEC filing text."""
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
        print("No existing filing sentiment output loaded. Starting fresh.")
    else:
        print(f"Loaded existing filing sentiment output with {len(existing_output_df):,} scored filings.")

    remaining_filings = filing_index[
        ~filing_index["accession_number"].astype(str).isin(existing_accessions)
    ].copy()

    print(f"Original 10-K / 10-Q filings available: {len(filing_index):,}")
    print(f"Filings remaining to score:             {len(remaining_filings):,}")

    if remaining_filings.empty:
        print("\nEverything is already scored. Nothing to do.")
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
            f"Scoring {row_number:,}/{len(remaining_filings):,}: "
            f"{ticker} {filing_row['form_type']} {filing_row['filing_date'].date()} "
            f"({accession_number})"
        )

        try:
            scored_row = score_one_filing(
                filing_row=filing_row,
                torch=torch,
                tokenizer=tokenizer,
                model=model,
                device=device,
                batch_size=args.batch_size,
                max_chunks_per_filing=args.max_chunks_per_filing,
            )
            if scored_row is None:
                continue

            all_rows.append(scored_row)
            processed_since_checkpoint += 1
        except Exception as exc:
            print(f"{ticker}: failed to score filing {accession_number} -> {exc}")
            continue

        if processed_since_checkpoint >= args.checkpoint_every:
            checkpoint_df = checkpoint_results(all_rows, output_path)
            processed_since_checkpoint = 0
            print(f"Checkpoint saved to: {output_path} ({len(checkpoint_df):,} filings)")

    final_df = checkpoint_results(all_rows, output_path)
    print(f"\nSaved filing sentiment to: {output_path}")
    print_summary(final_df)


if __name__ == "__main__":
    main()
