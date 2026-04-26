from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

DEFAULT_INPUT_NEWS_PATH = Path("data") / "raw" / "analyst" / "analyst_ratings_processed.csv"
DEFAULT_OUTPUT_PATH = Path("data") / "processed" / "news_scores_finbert.parquet"
MODEL_NAME = "ProsusAI/finbert"
TICKER_CANDIDATES = ("ticker", "symbol", "stock", "stocks", "tickers")
DATE_CANDIDATES = ("date", "published_date", "article_date", "trading_date", "published_at", "datetime")
TEXT_CANDIDATES = ("headline", "title", "text", "body", "article", "content")
SOURCE_CANDIDATES = ("source", "publisher", "provider", "news_outlet", "source_domain")
CANONICAL_LABELS = ("positive", "neutral", "negative")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score point-in-time financial news with zero-shot ProsusAI/finbert."
    )
    parser.add_argument("--input-news", default=str(DEFAULT_INPUT_NEWS_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--text-col", default="")
    parser.add_argument("--ticker-col", default="")
    parser.add_argument("--date-col", default="")
    parser.add_argument("--source-col", default="")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--min-text-length", type=int, default=5)
    parser.add_argument("--fail-on-empty", action="store_true")
    return parser.parse_args()


def read_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input news file was not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input extension for {path}; expected .csv or .parquet")


def write_frame(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
        return
    raise ValueError(f"Unsupported output extension for {path}; expected .csv or .parquet")


def resolve_column(df: pd.DataFrame, explicit: str, candidates: tuple[str, ...], role: str) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"--{role}-col was provided but is not in input news: {explicit}")
        return explicit
    lower_lookup = {str(column).strip().lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate in lower_lookup:
            return lower_lookup[candidate]
    raise ValueError(f"No {role} column found; provide --{role}-col or include one of: {', '.join(candidates)}")


def resolve_optional_source_col(df: pd.DataFrame, explicit: str) -> str | None:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"--source-col was provided but is not in input news: {explicit}")
        return explicit
    lower_lookup = {str(column).strip().lower(): column for column in df.columns}
    for candidate in SOURCE_CANDIDATES:
        if candidate in lower_lookup:
            return lower_lookup[candidate]
    return None


def normalize_label_mapping(id2label: dict[int | str, str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for raw_index, raw_label in id2label.items():
        label = str(raw_label).strip().lower()
        index = int(raw_index)
        for canonical in CANONICAL_LABELS:
            if canonical in label:
                mapping[canonical] = index
                break
    missing = [label for label in CANONICAL_LABELS if label not in mapping]
    if missing:
        raise ValueError(
            "Could not map FinBERT model labels to positive/neutral/negative. "
            f"id2label={id2label!r}; missing={missing}"
        )
    return mapping


def prepare_news_rows(
    news_df: pd.DataFrame,
    *,
    ticker_col: str,
    date_col: str,
    text_col: str,
    source_col: str | None,
    min_text_length: int,
    max_rows: int = 0,
) -> pd.DataFrame:
    prepared = news_df.copy()
    prepared["_original_row_id"] = np.arange(len(prepared), dtype="int64")
    prepared["ticker"] = prepared[ticker_col].astype("string").str.upper().str.strip()
    prepared["date"] = pd.to_datetime(prepared[date_col], errors="coerce", utc=True).dt.tz_localize(None)
    prepared["_score_text"] = prepared[text_col].astype("string").str.strip()
    usable = (
        prepared["ticker"].notna()
        & prepared["date"].notna()
        & prepared["_score_text"].notna()
        & prepared["_score_text"].str.len().ge(int(min_text_length))
    )
    prepared = prepared.loc[usable].copy()
    if max_rows and max_rows > 0:
        prepared = prepared.head(int(max_rows)).copy()

    output = pd.DataFrame(
        {
            "ticker": prepared["ticker"],
            "date": prepared["date"],
            "headline": prepared["_score_text"],
            "text_id": prepared["_original_row_id"].astype("int64"),
        }
    )
    if source_col is not None:
        output["source"] = prepared[source_col].astype("string")
    return output.reset_index(drop=True)


def empty_scores_frame(include_source: bool) -> pd.DataFrame:
    columns = [
        "ticker",
        "date",
        "finbert_pos",
        "finbert_neu",
        "finbert_neg",
        "confidence",
        "finbert_score",
        "headline",
        "text_id",
    ]
    if include_source:
        columns.append("source")
    return pd.DataFrame(columns=columns)


def load_finbert_components(device_arg: str):
    os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing FinBERT scoring dependencies. Install torch and transformers before running this script."
        ) from exc

    if device_arg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda was requested, but torch.cuda.is_available() is False")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, use_safetensors=False)
    model.to(device)
    model.eval()
    label_mapping = normalize_label_mapping(model.config.id2label)
    return tokenizer, model, device, label_mapping, torch


def score_texts_with_finbert(
    texts: list[str],
    *,
    tokenizer,
    model,
    device: str,
    label_mapping: dict[str, int],
    torch_module,
    batch_size: int,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    total = len(texts)
    for start in range(0, total, int(batch_size)):
        batch_texts = texts[start : start + int(batch_size)]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch_module.no_grad():
            logits = model(**encoded).logits
            probabilities = torch_module.softmax(logits, dim=-1).detach().cpu().numpy()
        for probs in probabilities:
            pos = float(probs[label_mapping["positive"]])
            neu = float(probs[label_mapping["neutral"]])
            neg = float(probs[label_mapping["negative"]])
            rows.append(
                {
                    "finbert_pos": pos,
                    "finbert_neu": neu,
                    "finbert_neg": neg,
                    "confidence": max(pos, neu, neg),
                    "finbert_score": pos - neg,
                }
            )
        print(f"Scored {min(start + int(batch_size), total):,} / {total:,} news rows")
    return pd.DataFrame(rows)


def default_scorer(texts: list[str], *, batch_size: int, device_arg: str) -> pd.DataFrame:
    tokenizer, model, device, label_mapping, torch_module = load_finbert_components(device_arg)
    print(f"Loaded {MODEL_NAME} on {device}")
    return score_texts_with_finbert(
        texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        label_mapping=label_mapping,
        torch_module=torch_module,
        batch_size=batch_size,
    )


def build_news_scores_finbert(
    args: argparse.Namespace,
    *,
    scorer: Callable[[list[str]], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    input_path = Path(args.input_news)
    output_path = Path(args.output)
    news_df = read_frame(input_path)
    ticker_col = resolve_column(news_df, args.ticker_col, TICKER_CANDIDATES, "ticker")
    date_col = resolve_column(news_df, args.date_col, DATE_CANDIDATES, "date")
    text_col = resolve_column(news_df, args.text_col, TEXT_CANDIDATES, "text")
    source_col = resolve_optional_source_col(news_df, args.source_col)

    prepared = prepare_news_rows(
        news_df,
        ticker_col=ticker_col,
        date_col=date_col,
        text_col=text_col,
        source_col=source_col,
        min_text_length=int(args.min_text_length),
        max_rows=int(args.max_rows or 0),
    )
    if prepared.empty:
        message = "No usable news rows after filtering missing ticker/date/text and short text."
        if args.fail_on_empty:
            raise ValueError(message)
        print(f"Warning: {message} Writing empty scores file.")
        output = empty_scores_frame(include_source=source_col is not None)
        write_frame(output, output_path)
        return output

    texts = prepared["headline"].astype(str).tolist()
    if scorer is None:
        scores = default_scorer(texts, batch_size=int(args.batch_size), device_arg=args.device)
    else:
        scores = scorer(texts)
    required_score_columns = ["finbert_pos", "finbert_neu", "finbert_neg", "confidence"]
    missing_score_columns = [column for column in required_score_columns if column not in scores.columns]
    if missing_score_columns:
        raise ValueError("FinBERT scorer did not return required columns: " + ", ".join(missing_score_columns))
    if len(scores) != len(prepared):
        raise ValueError(f"FinBERT scorer returned {len(scores)} rows for {len(prepared)} input texts")

    output = pd.concat(
        [
            prepared[["ticker", "date"]].reset_index(drop=True),
            scores.reset_index(drop=True),
            prepared[[column for column in ["headline", "text_id", "source"] if column in prepared.columns]].reset_index(
                drop=True
            ),
        ],
        axis=1,
    )
    if "finbert_score" not in output.columns:
        output["finbert_score"] = output["finbert_pos"] - output["finbert_neg"]
    write_frame(output, output_path)
    print("FinBERT news scoring complete.")
    print(f"Input news:      {input_path}")
    print(f"Output scores:   {output_path}")
    print(f"Rows scored:     {len(output):,}")
    print(f"Ticker column:   {ticker_col}")
    print(f"Date column:     {date_col}")
    print(f"Text column:     {text_col}")
    if source_col:
        print(f"Source column:   {source_col}")
    return output


def main() -> None:
    build_news_scores_finbert(parse_args())


if __name__ == "__main__":
    main()
