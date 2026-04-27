from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.paths import PROJECT_ROOT


DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "capitaliq_keydev2"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "capitaliq_keydev_news_prepared.parquet"
CANONICAL_FILE_RE = re.compile(r"^(?P<ticker>[A-Z0-9-]+)_(?P<year>20[0-9]{2})\.csv$")
REQUIRED_COLUMNS = {
    "source_page_ticker",
    "year",
    "date",
    "row_company",
    "exchange_ticker",
    "type",
    "headline",
    "situation",
    "source",
    "direct_parent_match",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Capital IQ key developments for FinBERT scoring.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--include-non-parent", action="store_true")
    parser.add_argument("--headline-only", action="store_true")
    return parser.parse_args()


def _is_true(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _clean_text(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
    return cleaned.mask(cleaned.eq("-"), "")


def _read_canonical_files(input_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    ignored_files: list[str] = []
    for path in sorted(input_dir.glob("*.csv")):
        match = CANONICAL_FILE_RE.match(path.name)
        if match is None:
            ignored_files.append(path.name)
            continue
        frame = pd.read_csv(path, dtype="string")
        missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
        frame["_source_file"] = path.name
        frame["_file_ticker"] = match.group("ticker")
        frame["_file_year"] = int(match.group("year"))
        frames.append(frame)
    if not frames:
        raise ValueError(f"No canonical ticker-year CSV files found in {input_dir}")
    return pd.concat(frames, ignore_index=True), ignored_files


def prepare_capitaliq_keydev_news(
    *,
    input_dir: Path,
    output_path: Path,
    include_non_parent: bool = False,
    headline_only: bool = False,
) -> pd.DataFrame:
    raw, ignored_files = _read_canonical_files(input_dir)
    raw["_event_date"] = pd.to_datetime(raw["date"], format="%b-%d-%Y", errors="coerce")
    raw["_direct_parent_match"] = _is_true(raw["direct_parent_match"])
    prepared = raw.copy()
    if not include_non_parent:
        prepared = prepared.loc[prepared["_direct_parent_match"]].copy()

    prepared["ticker"] = prepared["source_page_ticker"].astype("string").str.upper().str.strip()
    prepared["headline"] = _clean_text(prepared["headline"])
    prepared["situation"] = _clean_text(prepared["situation"])
    prepared["type"] = _clean_text(prepared["type"])
    if headline_only:
        prepared["text"] = prepared["headline"]
    else:
        prepared["text"] = prepared["headline"]
        add_situation = prepared["situation"].ne("") & prepared["situation"].ne(prepared["headline"])
        prepared.loc[add_situation, "text"] = (
            prepared.loc[add_situation, "headline"] + " " + prepared.loc[add_situation, "situation"]
        )

    usable = prepared["ticker"].notna() & prepared["_event_date"].notna() & prepared["text"].str.len().ge(5)
    prepared = prepared.loc[usable].copy()
    prepared = prepared.drop_duplicates(subset=["ticker", "_event_date", "headline", "type"], keep="first")
    prepared = prepared.sort_values(["ticker", "_event_date", "headline", "type"]).reset_index(drop=True)

    output = pd.DataFrame(
        {
            "ticker": prepared["ticker"],
            "date": prepared["_event_date"],
            "headline": prepared["text"],
            "capitaliq_headline": prepared["headline"],
            "capitaliq_type": prepared["type"],
            "capitaliq_source": prepared["source"].astype("string"),
            "row_company": prepared["row_company"].astype("string"),
            "exchange_ticker": prepared["exchange_ticker"].astype("string"),
            "direct_parent_match": prepared["_direct_parent_match"].astype(bool),
            "source_file": prepared["_source_file"].astype("string"),
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".parquet":
        output.to_parquet(output_path, index=False)
    elif output_path.suffix.lower() == ".csv":
        output.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported output extension for {output_path}; expected .csv or .parquet")

    print("Capital IQ key developments preparation complete.")
    print(f"Input dir:           {input_dir}")
    print(f"Output:              {output_path}")
    print(f"Canonical raw rows:  {len(raw):,}")
    print(f"Prepared rows:       {len(output):,}")
    print(f"Ignored files:       {', '.join(ignored_files) if ignored_files else 'none'}")
    print(f"Date range:          {output['date'].min().date()} to {output['date'].max().date()}")
    print(f"Tickers:             {output['ticker'].nunique():,}")
    return output


def main() -> None:
    args = parse_args()
    prepare_capitaliq_keydev_news(
        input_dir=Path(args.input_dir),
        output_path=Path(args.output),
        include_non_parent=bool(args.include_non_parent),
        headline_only=bool(args.headline_only),
    )


if __name__ == "__main__":
    main()
