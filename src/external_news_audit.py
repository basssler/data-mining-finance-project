from __future__ import annotations

import argparse
import math
import hashlib
import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.pipeline_utils import configure_logging

DEFAULT_DATASET_ID = "Brianferrell787/financial-news-multisource"
DEFAULT_OUTPUT_DIR = Path("outputs") / "quarterly" / "diagnostics"
DEFAULT_REPORT_PATH = Path("reports") / "results" / "external_news_audit_summary.md"
DEFAULT_TARGETED_REPORT_PATH = Path("reports") / "results" / "external_news_targeted_audit_summary.md"
DEFAULT_WINDOWS = (7, 14, 30)
DEFAULT_TRADING_DAY_WINDOWS = {
    "T-1": (1, 1),
    "T-3_to_T-1": (3, 1),
    "T-5_to_T-1": (5, 1),
    "T-10_to_T-1": (10, 1),
}
DEFAULT_TARGETED_SUBSETS = (
    "yahoo_finance_felixdrinkall",
    "sentarl_combined",
    "benzinga_6000stocks",
)
SUSPICIOUS_SUBSETS = {"fnspid_news"}
FINAL_RECOMMENDATION_TEXT = [
    "Do not integrate this dataset into the current quarterly model.",
    "",
    "The targeted subset audit confirms that several Hugging Face subsets contain clean ticker, source, and timestamp metadata, especially `benzinga_6000stocks`, `sentarl_combined`, and `yahoo_finance_felixdrinkall`. However, conservative pre-event coverage remains too sparse for model integration. The best trading-day window, `T-10_to_T-1`, covers only 32 of 1,109 events, or 2.89%.",
    "",
    "Because coverage is concentrated in limited date ranges and only a small share of quarterly events receive usable pre-event news, this dataset would mostly create missing or zero-valued features. Running FinBERT or training models on this layer is not justified.",
    "",
    "Final status: researched and rejected for immediate integration. Revisit later only if the project narrows to a smaller ticker universe, a different date range, or a dedicated news-source prototype.",
]
EVENT_DATE_CANDIDATES = (
    "tradable_date",
    "effective_model_date",
    "event_date",
    "filing_timestamp_utc",
    "event_timestamp",
)

TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,5}$")
SPLIT_RE = re.compile(r"[,;/|\s]+")
NOISE_TOKENS = {
    "A",
    "AN",
    "AND",
    "API",
    "CEO",
    "CFO",
    "CO",
    "CORP",
    "ETF",
    "EPS",
    "FDA",
    "GDP",
    "INC",
    "IPO",
    "LLC",
    "LTD",
    "NASDAQ",
    "NYSE",
    "Q1",
    "Q2",
    "Q3",
    "Q4",
    "SEC",
    "THE",
    "US",
    "USA",
}


@dataclass(frozen=True)
class DatasetFile:
    subset: str
    filename: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safely audit external financial news coverage for the quarterly event panel."
    )
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--capitaliq-universe", required=True)
    parser.add_argument("--event-panel", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--targeted-report-path", default=str(DEFAULT_TARGETED_REPORT_PATH))
    parser.add_argument("--mode", choices=["sample", "full"], default="sample")
    parser.add_argument("--sample-rows-per-subset", type=int, default=50_000)
    parser.add_argument(
        "--max-shards-per-subset",
        type=int,
        default=2,
        help="In sample mode, download/read at most this many parquet shards per subset.",
    )
    parser.add_argument("--coverage-windows", type=int, nargs="+", default=list(DEFAULT_WINDOWS))
    parser.add_argument("--targeted-subsets", nargs="+", default=list(DEFAULT_TARGETED_SUBSETS))
    parser.add_argument(
        "--only-targeted-subsets",
        action="store_true",
        help="Read only the targeted subsets. The main report will still be written for the run scope.",
    )
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel Hugging Face shard download/read workers. Keep modest to avoid cache/network thrash.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_hf_tools():
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "Missing optional dependency 'huggingface_hub'. Install it before running this audit: "
            "pip install huggingface_hub"
        ) from exc
    return HfApi, hf_hub_download


def subset_from_filename(filename: str) -> str:
    parts = filename.split("/")
    if len(parts) == 1:
        return "default"
    if parts[0] in {"data", "train", "validation", "test"} and len(parts) > 2:
        return parts[1]
    return parts[0]


def list_parquet_files(dataset_id: str) -> list[DatasetFile]:
    HfApi, _ = load_hf_tools()
    files = HfApi().list_repo_files(dataset_id, repo_type="dataset")
    parquet_files = [f for f in files if f.lower().endswith(".parquet")]
    if not parquet_files:
        raise ValueError(f"No parquet files found in Hugging Face dataset: {dataset_id}")
    return [DatasetFile(subset=subset_from_filename(f), filename=f) for f in sorted(parquet_files)]


def select_files(
    files: list[DatasetFile],
    mode: str,
    max_shards_per_subset: int,
    subset_filter: set[str] | None = None,
) -> list[DatasetFile]:
    if subset_filter:
        files = [file for file in files if file.subset in subset_filter]
    if mode == "full":
        return files
    grouped: dict[str, list[DatasetFile]] = defaultdict(list)
    for file in files:
        grouped[file.subset].append(file)
    selected: list[DatasetFile] = []
    for subset, subset_files in sorted(grouped.items()):
        selected.extend(subset_files[: max(1, max_shards_per_subset)])
    return selected


def download_dataset_file(dataset_id: str, filename: str) -> Path:
    _, hf_hub_download = load_hf_tools()
    return Path(hf_hub_download(repo_id=dataset_id, filename=filename, repo_type="dataset"))


def compact_schema(schema: Any) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for field in schema:
        rows.append({"column": field.name, "type": str(field.type)})
    return rows


def is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (dict, list, tuple, set)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def parse_extra_fields(value: Any) -> dict[str, Any]:
    if is_missing_scalar(value):
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return {"__raw_extra_fields": stripped}
        return parsed if isinstance(parsed, dict) else {"__extra_fields_value": parsed}
    return {"__extra_fields_value": value}


def flatten_scalars(prefix: str, payload: dict[str, Any], output: dict[str, Any]) -> None:
    for key, value in payload.items():
        flat_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flatten_scalars(flat_key, value, output)
        elif isinstance(value, (list, tuple)):
            output[flat_key] = ";".join(str(v) for v in value[:20])
        else:
            output[flat_key] = value


def normalize_ticker_token(value: Any) -> set[str]:
    if is_missing_scalar(value):
        return set()
    tokens: set[str] = set()
    for raw in SPLIT_RE.split(str(value).upper().replace("$", " ")):
        token = raw.strip().strip("()[]{}'\"")
        if ":" in token:
            token = token.split(":")[-1]
        if token.endswith(".US"):
            token = token[:-3]
        if TICKER_RE.match(token) and token not in NOISE_TOKENS:
            tokens.add(token)
    return tokens


def detect_columns(columns: list[str]) -> dict[str, list[str]]:
    lower = {column: column.lower() for column in columns}
    return {
        "date": [c for c, lc in lower.items() if lc in {"date", "published_date", "publish_date"} or "date" in lc],
        "timestamp": [c for c, lc in lower.items() if "time" in lc or "timestamp" in lc or "published_at" in lc],
        "ticker": [
            c
            for c, lc in lower.items()
            if lc in {"ticker", "tickers", "symbol", "symbols", "stock", "stocks"} or "ticker" in lc
        ],
        "company": [c for c, lc in lower.items() if "company" in lc or lc in {"name", "entity", "entities"}],
        "source": [c for c, lc in lower.items() if "source" in lc or "publisher" in lc or lc in {"site", "provider"}],
        "url": [c for c, lc in lower.items() if "url" in lc or "link" in lc],
        "text": [c for c, lc in lower.items() if lc in {"text", "title", "headline", "summary", "content", "body"}],
        "extra_fields": [c for c, lc in lower.items() if lc == "extra_fields"],
    }


def first_present(row: pd.Series, candidates: list[str]) -> Any:
    for column in candidates:
        if column in row.index and pd.notna(row[column]):
            return row[column]
    return None


def first_flat_value(flat_extra: dict[str, Any], patterns: tuple[str, ...]) -> Any:
    for key, value in flat_extra.items():
        key_lc = key.lower()
        if any(pattern in key_lc for pattern in patterns) and not is_missing_scalar(value):
            return value
    return None


def hash_text(value: Any) -> str | None:
    if is_missing_scalar(value):
        return None
    return hashlib.sha1(str(value).encode("utf-8", errors="ignore")).hexdigest()


def _row_limit_by_file(selected_files: list[DatasetFile], sample_rows_per_subset: int) -> dict[str, int]:
    grouped: dict[str, list[DatasetFile]] = defaultdict(list)
    for dataset_file in selected_files:
        grouped[dataset_file.subset].append(dataset_file)

    limits: dict[str, int] = {}
    for subset_files in grouped.values():
        per_file_limit = int(math.ceil(sample_rows_per_subset / max(1, len(subset_files))))
        for dataset_file in subset_files:
            limits[dataset_file.filename] = per_file_limit
    return limits


def read_dataset_file_sample(
    dataset_id: str,
    dataset_file: DatasetFile,
    row_limit: int,
    batch_size: int,
) -> tuple[list[dict[str, str]], dict[str, Any], pd.DataFrame]:
    try:
        local_path = download_dataset_file(dataset_id, dataset_file.filename)
        parquet_file = pq.ParquetFile(local_path)
    except Exception as exc:  # noqa: BLE001
        return (
            [],
            {
                "subset": dataset_file.subset,
                "filename": dataset_file.filename,
                "sampled_file": False,
                "metadata_rows": None,
                "row_groups": None,
                "error": f"{type(exc).__name__}: {exc}",
            },
            pd.DataFrame(),
        )

    schema_rows = [
        {"subset": dataset_file.subset, "filename": dataset_file.filename, **row}
        for row in compact_schema(parquet_file.schema_arrow)
    ]
    file_row = {
        "subset": dataset_file.subset,
        "filename": dataset_file.filename,
        "sampled_file": True,
        "metadata_rows": parquet_file.metadata.num_rows,
        "row_groups": parquet_file.metadata.num_row_groups,
        "error": None,
    }

    frames: list[pd.DataFrame] = []
    remaining = row_limit
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        if remaining <= 0:
            break
        df = batch.to_pandas()
        if len(df) > remaining:
            df = df.iloc[:remaining].copy()
        frames.append(df)
        remaining -= len(df)

    sample = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if not sample.empty:
        sample.insert(0, "__subset", dataset_file.subset)
    return schema_rows, file_row, sample


def iter_sampled_rows(
    dataset_id: str,
    selected_files: list[DatasetFile],
    sample_rows_per_subset: int,
    batch_size: int,
    logger: logging.Logger,
    workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    schema_rows: list[dict[str, str]] = []
    file_rows: list[dict[str, Any]] = []
    sample_frames: list[pd.DataFrame] = []
    row_limits = _row_limit_by_file(selected_files, sample_rows_per_subset)
    worker_count = max(1, int(workers))

    if worker_count == 1:
        for dataset_file in selected_files:
            row_limit = row_limits[dataset_file.filename]
            logger.info("Reading up to %s rows from %s", f"{row_limit:,}", dataset_file.filename)
            file_schema_rows, file_row, sample = read_dataset_file_sample(
                dataset_id,
                dataset_file,
                row_limit,
                batch_size,
            )
            if file_row.get("error"):
                logger.warning("Could not read %s: %s", dataset_file.filename, file_row["error"])
            schema_rows.extend(file_schema_rows)
            file_rows.append(file_row)
            if not sample.empty:
                sample_frames.append(sample)
    else:
        logger.info("Reading %s selected shards with %s workers", f"{len(selected_files):,}", worker_count)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    read_dataset_file_sample,
                    dataset_id,
                    dataset_file,
                    row_limits[dataset_file.filename],
                    batch_size,
                ): dataset_file
                for dataset_file in selected_files
            }
            for future in as_completed(futures):
                dataset_file = futures[future]
                file_schema_rows, file_row, sample = future.result()
                if file_row.get("error"):
                    logger.warning("Could not read %s: %s", dataset_file.filename, file_row["error"])
                else:
                    logger.info("Read sampled rows from %s", dataset_file.filename)
                schema_rows.extend(file_schema_rows)
                file_rows.append(file_row)
                if not sample.empty:
                    sample_frames.append(sample)

    sample = pd.concat(sample_frames, ignore_index=True, sort=False) if sample_frames else pd.DataFrame()
    return pd.DataFrame(schema_rows), pd.DataFrame(file_rows), sample


def enrich_news_sample(sample: pd.DataFrame, universe_tickers: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if sample.empty:
        return pd.DataFrame(), pd.DataFrame()

    detected = detect_columns(list(sample.columns))
    extra_key_counts: Counter[str] = Counter()
    extra_non_null_counts: Counter[str] = Counter()
    extra_examples: dict[str, str] = {}
    enriched_rows: list[dict[str, Any]] = []

    for _, row in sample.iterrows():
        flat_extra: dict[str, Any] = {}
        for extra_col in detected["extra_fields"]:
            flatten_scalars("", parse_extra_fields(row.get(extra_col)), flat_extra)
        for key, value in flat_extra.items():
            extra_key_counts[key] += 1
            if not is_missing_scalar(value):
                extra_non_null_counts[key] += 1
                extra_examples.setdefault(key, str(value)[:300])

        ticker_candidates: set[str] = set()
        for column in detected["ticker"]:
            ticker_candidates.update(normalize_ticker_token(row.get(column)))
        for key, value in flat_extra.items():
            key_lc = key.lower()
            if "ticker" in key_lc or key_lc in {"symbol", "symbols", "stock", "stocks"}:
                ticker_candidates.update(normalize_ticker_token(value))

        date_value = first_present(row, detected["timestamp"] + detected["date"])
        published_at = pd.to_datetime(date_value, errors="coerce", utc=True)
        published_date = published_at.date() if pd.notna(published_at) else pd.NaT
        source_value = first_present(row, detected["source"])
        if source_value is None:
            source_value = first_flat_value(flat_extra, ("source", "publisher", "provider", "site"))
        text_value = first_present(row, detected["text"])
        matched = sorted(t for t in ticker_candidates if t in universe_tickers)
        enriched_rows.append(
            {
                "__subset": row.get("__subset"),
                "published_at": published_at,
                "published_date": published_date,
                "source": source_value,
                "ticker_candidates": ";".join(sorted(ticker_candidates)),
                "matched_universe_tickers": ";".join(matched),
                "has_universe_ticker_match": bool(matched),
                "text_hash": hash_text(text_value),
            }
        )

    extra_summary = pd.DataFrame(
        [
            {
                "extra_field_key": key,
                "sample_rows_with_key": count,
                "sample_rows_non_null": extra_non_null_counts[key],
                "example": extra_examples.get(key),
            }
            for key, count in extra_key_counts.most_common()
        ]
    )
    return pd.DataFrame(enriched_rows), extra_summary


def load_universe(path: Path) -> pd.DataFrame:
    universe = pd.read_parquet(path)
    if "ticker" not in universe.columns:
        raise ValueError(f"Capital IQ universe is missing required ticker column: {path}")
    universe = universe.copy()
    universe["ticker"] = universe["ticker"].astype(str).str.upper().str.strip()
    return universe


def load_events(path: Path, logger: logging.Logger) -> tuple[pd.DataFrame, str]:
    events = pd.read_parquet(path)
    if "ticker" not in events.columns:
        raise ValueError(f"Event panel is missing required ticker column: {path}")
    event_date_column = next((column for column in EVENT_DATE_CANDIDATES if column in events.columns), None)
    if event_date_column is None:
        raise ValueError(
            f"Event panel is missing a usable event timing column. Tried: {list(EVENT_DATE_CANDIDATES)}"
        )
    if event_date_column != "tradable_date":
        logger.warning(
            "Event panel does not contain tradable_date; using %s as the conservative event anchor.",
            event_date_column,
        )
    events = events.copy()
    events["ticker"] = events["ticker"].astype(str).str.upper().str.strip()
    events["tradable_date"] = pd.to_datetime(events[event_date_column], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    return events.loc[events["ticker"].notna() & events["tradable_date"].notna()].copy(), event_date_column


def explode_news_tickers(enriched: pd.DataFrame) -> pd.DataFrame:
    if enriched.empty or "matched_universe_tickers" not in enriched.columns:
        return pd.DataFrame(columns=["ticker", "published_date", "__subset"])
    rows: list[dict[str, Any]] = []
    for _, row in enriched.iterrows():
        tickers = str(row.get("matched_universe_tickers", "") or "").split(";")
        published_date = pd.to_datetime(row.get("published_date", pd.NaT), errors="coerce")
        if pd.isna(published_date):
            continue
        for ticker in tickers:
            ticker = ticker.strip()
            if ticker:
                rows.append({"ticker": ticker, "published_date": published_date.normalize(), "__subset": row.get("__subset")})
    return pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame(columns=["ticker", "published_date", "__subset"])


def compute_event_coverage(events: pd.DataFrame, news_dates: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if news_dates.empty:
        for window in windows:
            rows.append(
                {
                    "window_days": window,
                    "events": len(events),
                    "covered_events": 0,
                    "coverage_pct": 0.0,
                    "note": "No sampled rows had explicit universe ticker and usable pre-event date metadata.",
                }
            )
        return pd.DataFrame(rows)

    for window in windows:
        merged = events[["ticker", "tradable_date"]].merge(news_dates, on="ticker", how="left")
        in_window = (merged["published_date"] >= merged["tradable_date"] - pd.to_timedelta(window, unit="D")) & (
            merged["published_date"] < merged["tradable_date"]
        )
        covered_keys = merged.loc[in_window, ["ticker", "tradable_date"]].drop_duplicates()
        covered = len(covered_keys)
        rows.append(
            {
                "window_days": window,
                "events": len(events),
                "covered_events": covered,
                "coverage_pct": 100.0 * covered / len(events) if len(events) else 0.0,
                "note": "Sampled lower-bound; excludes event-day and post-event articles.",
            }
        )
    return pd.DataFrame(rows)


def compute_trading_day_coverage(events: pd.DataFrame, news_dates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if news_dates.empty:
        for label, (start_offset, end_offset) in DEFAULT_TRADING_DAY_WINDOWS.items():
            rows.append(
                {
                    "window": label,
                    "start_offset_trading_days": start_offset,
                    "end_offset_trading_days": end_offset,
                    "events": len(events),
                    "covered_events": 0,
                    "coverage_pct": 0.0,
                    "note": "No sampled rows had explicit universe ticker and usable pre-event date metadata.",
                }
            )
        return pd.DataFrame(rows)

    news_lookup = news_dates[["ticker", "published_date"]].drop_duplicates().copy()
    news_lookup["published_date"] = pd.to_datetime(news_lookup["published_date"], errors="coerce").dt.normalize()

    for label, (start_offset, end_offset) in DEFAULT_TRADING_DAY_WINDOWS.items():
        event_windows: list[dict[str, Any]] = []
        for event_id, row in enumerate(events[["ticker", "tradable_date"]].itertuples(index=False)):
            event_anchor = pd.Timestamp(row.tradable_date).normalize()
            for offset in range(end_offset, start_offset + 1):
                event_windows.append(
                    {
                        "event_id": event_id,
                        "ticker": row.ticker,
                        "tradable_date": event_anchor,
                        "published_date": (event_anchor - pd.offsets.BDay(offset)).normalize(),
                    }
                )
        event_window_df = pd.DataFrame(event_windows)
        merged = event_window_df.merge(news_lookup, on=["ticker", "published_date"], how="inner")
        covered = int(merged["event_id"].nunique())
        rows.append(
            {
                "window": label,
                "start_offset_trading_days": start_offset,
                "end_offset_trading_days": end_offset,
                "events": len(events),
                "covered_events": covered,
                "coverage_pct": 100.0 * covered / len(events) if len(events) else 0.0,
                "note": "Weekday trading-day proxy; excludes event-day and post-event articles.",
            }
        )
    return pd.DataFrame(rows)


def summarize_subset(sample: pd.DataFrame, enriched: pd.DataFrame) -> pd.DataFrame:
    if sample.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for subset, group in enriched.groupby("__subset", dropna=False):
        dates = pd.to_datetime(group["published_date"], errors="coerce")
        rows.append(
            {
                "subset": subset,
                "sample_rows": len(group),
                "date_min": dates.min(),
                "date_max": dates.max(),
                "rows_with_date_pct": 100.0 * dates.notna().mean(),
                "rows_with_source_pct": 100.0 * group["source"].notna().mean(),
                "rows_with_ticker_candidate_pct": 100.0 * (group["ticker_candidates"].astype(str).str.len() > 0).mean(),
                "rows_with_universe_ticker_match_pct": 100.0 * group["has_universe_ticker_match"].mean(),
                "unique_universe_tickers_matched": int(
                    len(
                        {
                            token
                            for value in group["matched_universe_tickers"].dropna().astype(str)
                            for token in value.split(";")
                            if token
                        }
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    all_files: list[DatasetFile],
    selected_files: list[DatasetFile],
    schema_df: pd.DataFrame,
    file_sample_df: pd.DataFrame,
    subset_summary: pd.DataFrame,
    extra_summary: pd.DataFrame,
    calendar_coverage: pd.DataFrame,
    trading_day_coverage: pd.DataFrame,
    universe: pd.DataFrame,
    events: pd.DataFrame,
    event_date_column: str,
    targeted_subsets: list[str],
) -> None:
    ensure_parent(path)
    selected_file_count = len(selected_files)
    total_file_count = len(all_files)
    schema_columns = sorted(schema_df["column"].unique().tolist()) if not schema_df.empty else []
    extra_keys = sorted(extra_summary["extra_field_key"].astype(str).unique().tolist()) if not extra_summary.empty else []
    extra_top = extra_summary.head(15) if not extra_summary.empty else pd.DataFrame()
    calendar_coverage_md = (
        calendar_coverage.to_markdown(index=False) if not calendar_coverage.empty else "_No calendar coverage rows produced._"
    )
    trading_day_coverage_md = (
        trading_day_coverage.to_markdown(index=False)
        if not trading_day_coverage.empty
        else "_No trading-day coverage rows produced._"
    )
    subset_md = subset_summary.to_markdown(index=False) if not subset_summary.empty else "_No sample rows produced._"
    extra_md = extra_top.to_markdown(index=False) if not extra_top.empty else "_No `extra_fields` keys found in sample._"
    suspicious_present = sorted(set(subset_summary.get("subset", pd.Series(dtype=str)).astype(str)) & SUSPICIOUS_SUBSETS)

    date_names = [name.lower() for name in schema_columns + extra_keys]
    has_date = any("date" in name or "time" in name or "timestamp" in name for name in date_names)
    has_ticker = any(
        "ticker" in name or name in {"symbol", "symbols", "stock", "stocks"} for name in date_names
    )
    has_source = any(
        "source" in name or "publisher" in name or "provider" in name or name == "site" for name in date_names
    )
    failed_files = int(file_sample_df["error"].notna().sum()) if "error" in file_sample_df.columns else 0
    successful_files = int(file_sample_df["sampled_file"].fillna(False).astype(bool).sum()) if "sampled_file" in file_sample_df.columns else 0
    best_trading_coverage = float(trading_day_coverage["coverage_pct"].max()) if not trading_day_coverage.empty else 0.0
    recommendation = "Do not integrate into the current quarterly model."

    lines = [
        "# Phase 9B External News Coverage Audit",
        "",
        "## Scope",
        "",
        f"- Dataset: `{args.dataset_id}`",
        f"- Mode: `{args.mode}`",
        f"- Listed parquet files: `{total_file_count:,}`",
        f"- Sampled parquet files: `{selected_file_count:,}`",
        f"- Successfully read sampled files: `{successful_files:,}`",
        f"- Sampled files with access/read errors: `{failed_files:,}`",
        f"- Sample rows per subset cap: `{args.sample_rows_per_subset:,}`",
        f"- Capital IQ universe rows: `{len(universe):,}`",
        f"- Event panel rows: `{len(events):,}`",
        f"- Event coverage anchor column: `{event_date_column}`",
        "",
        "## Go/No-Go Readout",
        "",
        "- Safe loading design: yes, this audit lists shards and reads Parquet batches; it does not materialize the full dataset.",
        f"- Date/timestamp metadata detected in schema: `{has_date}`",
        f"- Ticker/symbol metadata detected in schema: `{has_ticker}`",
        f"- Source metadata detected in schema: `{has_source}`",
        f"- Best sampled trading-day coverage: `{best_trading_coverage:.4f}%`",
        f"- Recommendation: **{recommendation}**",
        f"- Preferred targeted subsets: {', '.join(f'`{subset}`' for subset in targeted_subsets)}",
        "",
        "## Final Recommendation",
        "",
        *FINAL_RECOMMENDATION_TEXT,
        "",
        "## Sampled Subset Coverage",
        "",
        subset_md,
        "",
        "## Required Conservative Trading-Day Event Coverage",
        "",
        trading_day_coverage_md,
        "",
        "## Preliminary Calendar-Day Event Coverage Smoke Test",
        "",
        calendar_coverage_md,
        "",
        "## Audit Warnings",
        "",
        f"- This run used `{event_date_column}` as the event anchor. Before final integration, verify that this matches the quarterly label timing and `tradable_date` policy.",
        "- Limited-shard sampling can be date-biased because many subsets are stored in chronological shards and sampled shards may cover narrow date slices.",
        "- The calendar-day coverage table is a smoke test only; integration decisions should use the required trading-day windows.",
        "- Treat `fnspid_news` as suspicious until ticker extraction is inspected; it can show a high ticker-match percentage while matching very few unique universe tickers.",
        (
            f"- Suspicious subsets present in this run: {', '.join(f'`{subset}`' for subset in suspicious_present)}."
            if suspicious_present
            else "- No known suspicious subsets were sampled in this run."
        ),
        "",
        "## Schema Columns",
        "",
        ", ".join(f"`{column}`" for column in schema_columns) if schema_columns else "_No schema captured._",
        "",
        "## Top `extra_fields` Keys",
        "",
        extra_md,
        "",
        "## Leakage Risks",
        "",
        "- Use only articles with publication timestamps strictly before the event anchor; this audit excludes event-day and post-event text from coverage windows.",
        "- Rows with date-only metadata are risky for intraday filings because publication ordering cannot be proven.",
        "- Generic market, source popularity, article counts, or future-resolved company metadata can leak if computed with knowledge after the event date.",
        "- Ticker extraction from free text is not conservative enough for modeling; later integration should require explicit ticker/entity metadata or a point-in-time entity map.",
        "- Duplicate/syndicated articles across sources should be collapsed before any later feature work to avoid overweighting wire duplication.",
        "",
        "## Output Files",
        "",
        f"- `{Path(args.output_dir) / 'external_news_schema.csv'}`",
        f"- `{Path(args.output_dir) / 'external_news_file_sample.csv'}`",
        f"- `{Path(args.output_dir) / 'external_news_subset_summary.csv'}`",
        f"- `{Path(args.output_dir) / 'external_news_extra_fields_summary.csv'}`",
        f"- `{Path(args.output_dir) / 'external_news_calendar_day_event_coverage.csv'}`",
        f"- `{Path(args.output_dir) / 'external_news_trading_day_event_coverage.csv'}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_targeted_report(
    path: Path,
    *,
    args: argparse.Namespace,
    targeted_subsets: list[str],
    subset_summary: pd.DataFrame,
    calendar_coverage: pd.DataFrame,
    trading_day_coverage: pd.DataFrame,
    event_date_column: str,
) -> None:
    ensure_parent(path)
    targeted_subset_md = (
        subset_summary.to_markdown(index=False) if not subset_summary.empty else "_No targeted subset rows sampled._"
    )
    calendar_md = (
        calendar_coverage.to_markdown(index=False) if not calendar_coverage.empty else "_No calendar coverage rows produced._"
    )
    trading_md = (
        trading_day_coverage.to_markdown(index=False)
        if not trading_day_coverage.empty
        else "_No trading-day coverage rows produced._"
    )
    lines = [
        "# Phase 9B External News Targeted Subset Audit",
        "",
        "## Targeted Scope",
        "",
        f"- Dataset: `{args.dataset_id}`",
        f"- Targeted subsets: {', '.join(f'`{subset}`' for subset in targeted_subsets)}",
        f"- Event coverage anchor column: `{event_date_column}`",
        "",
        "## Final Recommendation",
        "",
        *FINAL_RECOMMENDATION_TEXT,
        "",
        "## Targeted Subset Coverage",
        "",
        targeted_subset_md,
        "",
        "## Required Conservative Trading-Day Event Coverage",
        "",
        trading_md,
        "",
        "## Preliminary Calendar-Day Event Coverage Smoke Test",
        "",
        calendar_md,
        "",
        "## Warnings",
        "",
        f"- This run used `{event_date_column}` as the event anchor, not necessarily `tradable_date`; verify quarterly label timing before any integration.",
        "- Limited-shard sampling can be date-biased if sampled shards cover narrow chronological slices.",
        "- `fnspid_news` is intentionally excluded from the preferred target list until ticker extraction is inspected.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    logger = configure_logging("external_news_audit")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading local universe and event panel")
    universe = load_universe(Path(args.capitaliq_universe))
    events, event_date_column = load_events(Path(args.event_panel), logger)
    universe_tickers = set(universe["ticker"].dropna().astype(str).str.upper())
    targeted_subsets = [str(subset) for subset in args.targeted_subsets]

    logger.info("Listing Hugging Face parquet files")
    all_files = list_parquet_files(args.dataset_id)
    selected_files = select_files(
        all_files,
        args.mode,
        args.max_shards_per_subset,
        set(targeted_subsets) if args.only_targeted_subsets else None,
    )

    schema_df, file_sample_df, raw_sample = iter_sampled_rows(
        args.dataset_id,
        selected_files,
        args.sample_rows_per_subset,
        args.batch_size,
        logger,
        args.workers,
    )
    enriched, extra_summary = enrich_news_sample(raw_sample, universe_tickers)
    subset_summary = summarize_subset(raw_sample, enriched)
    news_dates = explode_news_tickers(enriched)
    calendar_coverage = compute_event_coverage(events, news_dates, args.coverage_windows)
    trading_day_coverage = compute_trading_day_coverage(events, news_dates)

    targeted_subset_summary = (
        subset_summary.loc[subset_summary["subset"].isin(targeted_subsets)].copy()
        if "subset" in subset_summary.columns
        else pd.DataFrame()
    )
    targeted_news_dates = (
        news_dates.loc[news_dates["__subset"].isin(targeted_subsets)].copy()
        if "__subset" in news_dates.columns
        else pd.DataFrame(columns=["ticker", "published_date", "__subset"])
    )
    targeted_calendar_coverage = compute_event_coverage(events, targeted_news_dates, args.coverage_windows)
    targeted_trading_day_coverage = compute_trading_day_coverage(events, targeted_news_dates)

    schema_df.drop_duplicates().to_csv(output_dir / "external_news_schema.csv", index=False)
    file_sample_df.to_csv(output_dir / "external_news_file_sample.csv", index=False)
    subset_summary.to_csv(output_dir / "external_news_subset_summary.csv", index=False)
    extra_summary.to_csv(output_dir / "external_news_extra_fields_summary.csv", index=False)
    calendar_coverage.to_csv(output_dir / "external_news_calendar_day_event_coverage.csv", index=False)
    trading_day_coverage.to_csv(output_dir / "external_news_trading_day_event_coverage.csv", index=False)
    targeted_subset_summary.to_csv(output_dir / "external_news_targeted_subset_summary.csv", index=False)
    targeted_calendar_coverage.to_csv(output_dir / "external_news_targeted_calendar_day_event_coverage.csv", index=False)
    targeted_trading_day_coverage.to_csv(output_dir / "external_news_targeted_trading_day_event_coverage.csv", index=False)
    if not enriched.empty:
        enriched.to_parquet(output_dir / "external_news_sampled_metadata.parquet", index=False)

    write_report(
        Path(args.report_path),
        args=args,
        all_files=all_files,
        selected_files=selected_files,
        schema_df=schema_df,
        file_sample_df=file_sample_df,
        subset_summary=subset_summary,
        extra_summary=extra_summary,
        calendar_coverage=calendar_coverage,
        trading_day_coverage=trading_day_coverage,
        universe=universe,
        events=events,
        event_date_column=event_date_column,
        targeted_subsets=targeted_subsets,
    )
    logger.info("Wrote audit report to %s", args.report_path)
    write_targeted_report(
        Path(args.targeted_report_path),
        args=args,
        targeted_subsets=targeted_subsets,
        subset_summary=targeted_subset_summary,
        calendar_coverage=targeted_calendar_coverage,
        trading_day_coverage=targeted_trading_day_coverage,
        event_date_column=event_date_column,
    )
    logger.info("Wrote targeted audit report to %s", args.targeted_report_path)


if __name__ == "__main__":
    main()
