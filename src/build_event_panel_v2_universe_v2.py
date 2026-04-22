"""Build the Phase 5 cross-sector event_panel_v2 universe_v2 artifacts.

This script keeps the locked 34-name pipeline untouched and writes a parallel
set of universe_v2 artifacts only. It reuses the existing event_v2 design:
- one row = one ticker-event
- 10-Q / 10-K only
- fundamentals attached as of the event date
- market features ending at t-1
- same-filing SEC sentiment only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src import edgar_pull, feature_engineering, fundamentals_clean, market_features_v2, prices
from src.event_panel_v2_schema import (
    assert_matches_canonical_base_contract,
    order_columns_with_canonical_base_first,
)
from src.panel_builder_event_v2 import build_event_panel_v2, print_summary as print_event_panel_summary
from src.paths import DATA_DIR, DOCS_DIR, INTERIM_DATA_DIR, RAW_DATA_DIR
from src.sec_filing_events_v1 import (
    SecRequester as FilingRequester,
    align_effective_model_dates,
    build_universe_sec_mapping,
    fetch_company_submission_history,
    get_sec_user_agent as get_sec_metadata_user_agent,
    normalize_submission_history,
)
from src.sec_filing_text_pull import (
    build_existing_keys,
    build_index_dataframe,
    configure_edgartools_identity,
    fetch_company_filings,
    get_accession_number,
    get_filing_text,
    get_sec_user_agent as get_sec_text_user_agent,
    load_existing_index,
    make_filing_key,
    parse_date,
    try_import_edgartools,
)
from src.sec_sentiment_features import (
    build_existing_accessions,
    checkpoint_results,
    filter_original_filings,
    load_existing_output,
    load_filing_index,
    load_finbert,
    print_summary as print_sentiment_summary,
    score_one_filing,
)
from src.sec_sentiment_prepare import (
    build_output_table,
    engineer_layer3_features,
    normalize_input_data as normalize_sentiment_features_input,
    print_summary as print_layer3_summary,
)
from src.universe import UNIVERSE_V2_TICKERS_PATH, get_universe_v2_tickers

REFERENCE_DIR = DATA_DIR / "reference"
UNIVERSE_V2_DOC_PATH = DOCS_DIR / "universe_v2.md"

UNIVERSE_V2_PRICE_PATH = INTERIM_DATA_DIR / "prices" / "prices_with_labels_universe_v2.parquet"
UNIVERSE_V2_RAW_FUNDAMENTALS_PATH = RAW_DATA_DIR / "fundamentals" / "raw_fundamentals_universe_v2.parquet"
UNIVERSE_V2_CLEAN_FUNDAMENTALS_PATH = (
    INTERIM_DATA_DIR / "fundamentals" / "fundamentals_quarterly_clean_universe_v2.parquet"
)
UNIVERSE_V2_LAYER1_FEATURES_PATH = (
    INTERIM_DATA_DIR / "features" / "layer1_financial_features_universe_v2.parquet"
)
UNIVERSE_V2_MARKET_FEATURES_PATH = (
    INTERIM_DATA_DIR / "features" / "layer2_market_features_v2_universe_v2.parquet"
)
UNIVERSE_V2_SEC_TEXT_DIR = RAW_DATA_DIR / "sec_filings" / "text_universe_v2"
UNIVERSE_V2_SEC_TEXT_INDEX_PATH = (
    RAW_DATA_DIR / "sec_filings" / "sec_filings_text_index_universe_v2.parquet"
)
UNIVERSE_V2_SEC_SENTIMENT_RAW_PATH = (
    INTERIM_DATA_DIR / "sentiment" / "sec_filing_sentiment_universe_v2.parquet"
)
UNIVERSE_V2_SEC_SENTIMENT_FEATURES_PATH = (
    INTERIM_DATA_DIR / "features" / "layer3_sec_sentiment_features_universe_v2.parquet"
)
UNIVERSE_V2_SEC_MAPPING_PATH = INTERIM_DATA_DIR / "sec" / "sec_universe_mapping_v2.parquet"
UNIVERSE_V2_SEC_METADATA_PATH = INTERIM_DATA_DIR / "sec" / "sec_filing_metadata_v2.parquet"
UNIVERSE_V2_PANEL_PATH = INTERIM_DATA_DIR / "event_panel_v2_universe_v2.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Phase 5 event_panel_v2 universe_v2 artifacts.")
    parser.add_argument("--tickers-path", default=str(UNIVERSE_V2_TICKERS_PATH))
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--max-chunks-per-filing", type=int, default=None)
    parser.add_argument("--max-filings", type=int, default=None)
    parser.add_argument("--rescore-sentiment", action="store_true")
    return parser.parse_args()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_output_dirs() -> None:
    for path in [
        REFERENCE_DIR,
        UNIVERSE_V2_DOC_PATH.parent,
        UNIVERSE_V2_PRICE_PATH,
        UNIVERSE_V2_RAW_FUNDAMENTALS_PATH,
        UNIVERSE_V2_CLEAN_FUNDAMENTALS_PATH,
        UNIVERSE_V2_LAYER1_FEATURES_PATH,
        UNIVERSE_V2_MARKET_FEATURES_PATH,
        UNIVERSE_V2_SEC_TEXT_INDEX_PATH,
        UNIVERSE_V2_SEC_SENTIMENT_RAW_PATH,
        UNIVERSE_V2_SEC_SENTIMENT_FEATURES_PATH,
        UNIVERSE_V2_SEC_MAPPING_PATH,
        UNIVERSE_V2_SEC_METADATA_PATH,
        UNIVERSE_V2_PANEL_PATH,
    ]:
        ensure_parent_dir(path)
    UNIVERSE_V2_SEC_TEXT_DIR.mkdir(parents=True, exist_ok=True)


def build_price_artifact(tickers: list[str]) -> pd.DataFrame:
    print(f"Downloading daily prices for {len(tickers)} universe_v2 tickers...")
    raw_prices = prices.download_prices(tickers)
    price_df = prices.reshape_downloaded_prices(raw_prices)
    labeled_df = prices.build_labels(price_df)
    prices.save_prices(labeled_df, UNIVERSE_V2_PRICE_PATH)
    prices.print_price_summary(labeled_df)
    return labeled_df


def build_market_artifact(price_df: pd.DataFrame) -> pd.DataFrame:
    print("\nEngineering universe_v2 market features...")
    normalized_prices = market_features_v2.normalize_price_data(price_df)
    feature_df = market_features_v2.build_market_features_v2(normalized_prices)
    market_features_v2.save_market_features(feature_df, UNIVERSE_V2_MARKET_FEATURES_PATH)
    market_features_v2.print_summary(feature_df)
    return feature_df


def build_raw_fundamentals_artifact(tickers: list[str]) -> pd.DataFrame:
    requester = edgar_pull.SecRequester(user_agent=edgar_pull.get_sec_user_agent(), min_delay_seconds=0.25)
    print("\nLoading SEC ticker to CIK mapping for universe_v2 fundamentals...")
    ticker_to_cik = edgar_pull.load_ticker_to_cik_map(requester)
    all_rows: list[dict] = []
    for ticker in tickers:
        cik = ticker_to_cik.get(ticker)
        if not cik:
            print(f"{ticker}: missing CIK mapping, skipping")
            continue
        print(f"Processing fundamentals for {ticker} ({cik})...")
        all_rows.extend(edgar_pull.fetch_ticker_facts(ticker=ticker, cik=cik, requester=requester))
    raw_df = edgar_pull.build_dataframe(all_rows)
    edgar_pull.save_parquet(raw_df, UNIVERSE_V2_RAW_FUNDAMENTALS_PATH)
    print(f"Saved {len(raw_df):,} raw fundamentals rows to: {UNIVERSE_V2_RAW_FUNDAMENTALS_PATH}")
    return raw_df


def build_clean_fundamentals_artifact(raw_df: pd.DataFrame) -> pd.DataFrame:
    print("\nCleaning universe_v2 fundamentals...")
    normalized_df = fundamentals_clean.normalize_raw_data(raw_df)
    concept_deduped_df, concept_dedup_removed = fundamentals_clean.deduplicate_concept_rows(normalized_df)
    metadata_df, period_dedup_removed = fundamentals_clean.build_period_metadata(concept_deduped_df)
    concept_wide_df = fundamentals_clean.pivot_concepts_to_wide(concept_deduped_df)
    final_df = fundamentals_clean.combine_metadata_and_concepts(metadata_df, concept_wide_df)
    fundamentals_clean.save_clean_fundamentals(final_df, UNIVERSE_V2_CLEAN_FUNDAMENTALS_PATH)
    fundamentals_clean.print_data_quality_summary(
        df=final_df,
        concept_dedup_removed=concept_dedup_removed,
        period_dedup_removed=period_dedup_removed,
    )
    return final_df


def build_layer1_feature_artifact(clean_df: pd.DataFrame) -> pd.DataFrame:
    print("\nEngineering universe_v2 Layer 1 features...")
    normalized_df = feature_engineering.normalize_input_data(clean_df)
    feature_df = feature_engineering.engineer_features(normalized_df)
    feature_engineering.save_features(feature_df, UNIVERSE_V2_LAYER1_FEATURES_PATH)
    feature_engineering.print_feature_summary(feature_df)
    return feature_df


def build_text_file_path(text_dir: Path, ticker: str, filing_date: pd.Timestamp, form_type: str, accession_number: str) -> Path:
    safe_form_type = form_type.replace("/", "_")
    ticker_dir = text_dir / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)
    return ticker_dir / f"{filing_date.date()}_{safe_form_type}_{accession_number}.txt"


def fetch_and_save_text_rows_for_universe(
    ticker: str,
    company,
    existing_keys: set[tuple[str, str, str]],
    text_dir: Path,
) -> list[dict]:
    rows = []
    filings = fetch_company_filings(company, ticker)
    skipped_existing = 0
    print(f"{ticker}: found {len(filings)} filings")
    for filing in filings:
        try:
            form_type = str(getattr(filing, "form", ""))
            filing_date = parse_date(getattr(filing, "filing_date", None))
            period_end = parse_date(getattr(filing, "period_of_report", None))
            cik = str(getattr(filing, "cik", ""))
            company_name = str(getattr(filing, "company", getattr(filing, "company_name", "")))
            accession_number = get_accession_number(filing)
            if filing_date is None:
                continue
            text_file_path = build_text_file_path(
                text_dir=text_dir,
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
    print(f"{ticker}: saved or recovered {len(rows)} filings, skipped {skipped_existing} existing")
    return rows


def build_sec_text_artifact(tickers: list[str]) -> pd.DataFrame:
    print("\nDownloading universe_v2 SEC filing text...")
    Company, set_identity = try_import_edgartools()
    configure_edgartools_identity(set_identity, get_sec_text_user_agent())

    existing_index_df = load_existing_index(UNIVERSE_V2_SEC_TEXT_INDEX_PATH)
    existing_keys = build_existing_keys(existing_index_df)
    all_rows: list[dict] = existing_index_df.to_dict("records")
    for ticker in tickers:
        try:
            company = Company(ticker)
            rows = fetch_and_save_text_rows_for_universe(
                ticker=ticker,
                company=company,
                existing_keys=existing_keys,
                text_dir=UNIVERSE_V2_SEC_TEXT_DIR,
            )
            if rows:
                all_rows.extend(rows)
                current_index_df = build_index_dataframe(all_rows)
                current_index_df.to_parquet(UNIVERSE_V2_SEC_TEXT_INDEX_PATH, index=False)
                existing_keys = build_existing_keys(current_index_df)
                all_rows = current_index_df.to_dict("records")
        except Exception as exc:
            print(f"{ticker}: failed to fetch filings -> {exc}")
    index_df = build_index_dataframe(all_rows)
    index_df.to_parquet(UNIVERSE_V2_SEC_TEXT_INDEX_PATH, index=False)
    print(f"Saved universe_v2 SEC filing text index to: {UNIVERSE_V2_SEC_TEXT_INDEX_PATH}")
    return index_df


def build_sec_sentiment_raw_artifact(
    device: str,
    batch_size: int,
    checkpoint_every: int,
    max_chunks_per_filing: int | None,
    max_filings: int | None,
    rescore_sentiment: bool,
) -> pd.DataFrame:
    print("\nScoring universe_v2 SEC filing text with FinBERT...")
    filing_index = load_filing_index(UNIVERSE_V2_SEC_TEXT_INDEX_PATH)
    filing_index = filter_original_filings(filing_index)
    if max_filings is not None:
        filing_index = filing_index.head(max_filings).copy()

    existing_output_df = load_existing_output(UNIVERSE_V2_SEC_SENTIMENT_RAW_PATH, rescore=rescore_sentiment)
    existing_accessions = build_existing_accessions(existing_output_df)
    all_rows = existing_output_df.to_dict("records")
    remaining_filings = filing_index[
        ~filing_index["accession_number"].astype(str).isin(existing_accessions)
    ].copy()

    if remaining_filings.empty:
        print("All universe_v2 filings are already scored.")
        print_sentiment_summary(existing_output_df)
        return existing_output_df

    torch, tokenizer, model, runtime_device = load_finbert(requested_device=device)
    processed_since_checkpoint = 0
    for row_number, (_, filing_row) in enumerate(remaining_filings.iterrows(), start=1):
        accession_number = str(filing_row["accession_number"])
        ticker = str(filing_row["ticker"])
        print(
            f"Scoring {row_number:,}/{len(remaining_filings):,}: "
            f"{ticker} {filing_row['form_type']} {filing_row['filing_date'].date()} ({accession_number})"
        )
        try:
            scored_row = score_one_filing(
                filing_row=filing_row,
                torch=torch,
                tokenizer=tokenizer,
                model=model,
                device=runtime_device,
                batch_size=batch_size,
                max_chunks_per_filing=max_chunks_per_filing,
            )
            if scored_row is None:
                continue
            all_rows.append(scored_row)
            processed_since_checkpoint += 1
        except Exception as exc:
            print(f"{ticker}: failed to score filing {accession_number} -> {exc}")
            continue
        if processed_since_checkpoint >= checkpoint_every:
            checkpoint_df = checkpoint_results(all_rows, UNIVERSE_V2_SEC_SENTIMENT_RAW_PATH)
            processed_since_checkpoint = 0
            print(f"Checkpoint saved to: {UNIVERSE_V2_SEC_SENTIMENT_RAW_PATH} ({len(checkpoint_df):,} filings)")
    final_df = checkpoint_results(all_rows, UNIVERSE_V2_SEC_SENTIMENT_RAW_PATH)
    print_sentiment_summary(final_df)
    return final_df


def build_sec_sentiment_feature_artifact(raw_sentiment_df: pd.DataFrame) -> pd.DataFrame:
    print("\nPreparing universe_v2 filing-level sentiment features...")
    normalized_df = normalize_sentiment_features_input(raw_sentiment_df)
    featured_df = engineer_layer3_features(normalized_df)
    output_df = build_output_table(featured_df)
    output_df.to_parquet(UNIVERSE_V2_SEC_SENTIMENT_FEATURES_PATH, index=False)
    print_layer3_summary(output_df)
    return output_df


def build_sec_metadata_artifact(tickers: list[str], price_df: pd.DataFrame) -> pd.DataFrame:
    print("\nFetching universe_v2 SEC submissions metadata...")
    requester = FilingRequester(user_agent=get_sec_metadata_user_agent(), min_delay_seconds=0.25)
    mapping_df = build_universe_sec_mapping(requester=requester, universe_tickers=tickers)
    metadata_frames = []
    panel_dates_df = price_df[["ticker", "date"]].drop_duplicates().copy()
    panel_dates_df["ticker"] = panel_dates_df["ticker"].astype("string")
    panel_dates_df["date"] = pd.to_datetime(panel_dates_df["date"], errors="coerce").astype("datetime64[ns]")

    for row in mapping_df.itertuples(index=False):
        print(f"Fetching SEC submissions for {row.ticker} ({row.cik})...")
        company_history_df, company_name = fetch_company_submission_history(
            requester=requester,
            ticker=str(row.ticker),
            cik=str(row.cik),
        )
        normalized_history_df = normalize_submission_history(company_history_df)
        metadata_frames.append(normalized_history_df)
        if company_name:
            mapping_df.loc[mapping_df["ticker"] == row.ticker, "company_name"] = company_name

    raw_metadata_df = pd.concat(metadata_frames, ignore_index=True) if metadata_frames else pd.DataFrame()
    raw_metadata_df = align_effective_model_dates(raw_metadata_df, panel_dates_df)
    mapping_df.to_parquet(UNIVERSE_V2_SEC_MAPPING_PATH, index=False)
    raw_metadata_df.to_parquet(UNIVERSE_V2_SEC_METADATA_PATH, index=False)
    print(f"Saved universe_v2 SEC mapping to: {UNIVERSE_V2_SEC_MAPPING_PATH}")
    print(f"Saved universe_v2 SEC metadata to: {UNIVERSE_V2_SEC_METADATA_PATH}")
    return raw_metadata_df


def build_final_panel_artifact() -> pd.DataFrame:
    print("\nBuilding final universe_v2 event panel...")
    panel_df = build_event_panel_v2(
        price_path=UNIVERSE_V2_PRICE_PATH,
        event_source_path=UNIVERSE_V2_SEC_METADATA_PATH,
        fundamentals_path=UNIVERSE_V2_LAYER1_FEATURES_PATH,
        market_path=UNIVERSE_V2_MARKET_FEATURES_PATH,
        sentiment_path=UNIVERSE_V2_SEC_SENTIMENT_FEATURES_PATH,
    )
    canonical_columns = assert_matches_canonical_base_contract(
        panel_df,
        panel_name="event_panel_v2_universe_v2",
    )
    panel_df = order_columns_with_canonical_base_first(panel_df, canonical_columns=canonical_columns)
    panel_df.to_parquet(UNIVERSE_V2_PANEL_PATH, index=False)
    print_event_panel_summary(panel_df)
    return panel_df


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    tickers = get_universe_v2_tickers(Path(args.tickers_path))
    print(f"Loaded universe_v2 with {len(tickers)} tickers from: {args.tickers_path}")

    price_df = build_price_artifact(tickers)
    build_market_artifact(price_df)
    raw_fundamentals_df = build_raw_fundamentals_artifact(tickers)
    clean_fundamentals_df = build_clean_fundamentals_artifact(raw_fundamentals_df)
    build_layer1_feature_artifact(clean_fundamentals_df)
    text_index_df = build_sec_text_artifact(tickers)
    print(f"Universe_v2 SEC text rows: {len(text_index_df):,}")
    raw_sentiment_df = build_sec_sentiment_raw_artifact(
        device=args.device,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        max_chunks_per_filing=args.max_chunks_per_filing,
        max_filings=args.max_filings,
        rescore_sentiment=args.rescore_sentiment,
    )
    build_sec_sentiment_feature_artifact(raw_sentiment_df)
    build_sec_metadata_artifact(tickers, price_df)
    panel_df = build_final_panel_artifact()

    print("\nUniverse V2 Build Complete")
    print("-" * 60)
    print(f"Tickers: {len(tickers):,}")
    print(f"Panel rows: {len(panel_df):,}")
    print(f"Panel path: {UNIVERSE_V2_PANEL_PATH}")


if __name__ == "__main__":
    main()
