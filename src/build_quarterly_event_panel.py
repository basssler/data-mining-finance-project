"""Build quarterly raw/master/base/feature-ready event panel artifacts.

This integrates the repo's existing event-level panel builder with an explicit
quarterly event master layer:
- raw event master keeps original and amended filing rows
- filtered event master applies documented promotion rules
- base panel keeps only modeling identity/timing metadata
- feature-ready panel preserves the existing event_panel_v2 feature payload
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.config_event_v1 import PRICE_INPUT_PATH
from src.panel_builder import SEC_FILING_METADATA_PATH, prepare_prices
from src.panel_builder_event_v2 import (
    EVENT_REQUIRED_COLUMNS,
    EVENT_PANEL_V2_OUTPUT_PATH,
    FUNDAMENTALS_INPUT_PATH,
    align_event_effective_dates,
    build_event_panel_v2,
    load_fundamentals,
    load_parquet,
)
from src.paths import QUARTERLY_OUTPUTS_DIAGNOSTICS_DIR, QUARTERLY_OUTPUTS_PANELS_DIR
from src.project_config import ensure_stock_prediction_directories
from src.universe import get_project_sector_map

RAW_EVENT_MASTER_PATH = QUARTERLY_OUTPUTS_PANELS_DIR / "quarterly_event_master_raw.parquet"
FILTERED_EVENT_MASTER_PATH = QUARTERLY_OUTPUTS_PANELS_DIR / "quarterly_event_master.parquet"
BASE_PANEL_PATH = QUARTERLY_OUTPUTS_PANELS_DIR / "quarterly_event_panel_base.parquet"
FEATURE_PANEL_PATH = QUARTERLY_OUTPUTS_PANELS_DIR / "quarterly_event_panel_features.parquet"
ROW_COUNT_AUDIT_PATH = QUARTERLY_OUTPUTS_DIAGNOSTICS_DIR / "quarterly_event_row_count_audit.csv"
DUPLICATE_AUDIT_PATH = QUARTERLY_OUTPUTS_DIAGNOSTICS_DIR / "quarterly_event_duplicate_audit.csv"
TIMING_AUDIT_PATH = QUARTERLY_OUTPUTS_DIAGNOSTICS_DIR / "quarterly_event_timing_audit.csv"
TIMING_CONFIDENCE_COUNTS_PATH = QUARTERLY_OUTPUTS_DIAGNOSTICS_DIR / "quarterly_event_timing_confidence_counts.csv"
TIMING_SAMPLE_PATH = QUARTERLY_OUTPUTS_DIAGNOSTICS_DIR / "quarterly_event_timing_samples.csv"

EVENT_TIMEZONE = "America/New_York"
TRADABLE_SESSION_OPEN_HOUR = 9
TRADABLE_SESSION_OPEN_MINUTE = 30
SESSION_OPEN_MINUTE_OF_DAY = TRADABLE_SESSION_OPEN_HOUR * 60 + TRADABLE_SESSION_OPEN_MINUTE
SESSION_CLOSE_MINUTE_OF_DAY = 16 * 60
DEFAULT_LABEL_VERSION = "event_v2_63d_sign"
DEFAULT_FEATURE_VERSION = "quarterly_event_panel_v1"

MODELING_METADATA_COLUMNS = [
    "event_id",
    "ticker",
    "cik",
    "event_type",
    "fiscal_year",
    "fiscal_period",
    "period_end",
    "event_timestamp_raw",
    "event_timezone",
    "event_date_raw",
    "release_session",
    "timing_confidence",
    "tradable_timestamp",
    "tradable_date",
    "feature_snapshot_timestamp",
    "source_file_id",
    "feature_version",
    "label_version",
    "validation_group",
    "sector",
    "industry",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build quarterly event-level panel artifacts.")
    parser.add_argument("--price-path", default=str(PRICE_INPUT_PATH))
    parser.add_argument("--event-source-path", default=str(SEC_FILING_METADATA_PATH))
    parser.add_argument("--fundamentals-path", default=str(FUNDAMENTALS_INPUT_PATH))
    parser.add_argument("--raw-master-path", default=str(RAW_EVENT_MASTER_PATH))
    parser.add_argument("--filtered-master-path", default=str(FILTERED_EVENT_MASTER_PATH))
    parser.add_argument("--base-panel-path", default=str(BASE_PANEL_PATH))
    parser.add_argument("--feature-panel-path", default=str(FEATURE_PANEL_PATH))
    parser.add_argument("--row-count-audit-path", default=str(ROW_COUNT_AUDIT_PATH))
    parser.add_argument("--duplicate-audit-path", default=str(DUPLICATE_AUDIT_PATH))
    parser.add_argument("--timing-audit-path", default=str(TIMING_AUDIT_PATH))
    parser.add_argument("--timing-confidence-counts-path", default=str(TIMING_CONFIDENCE_COUNTS_PATH))
    parser.add_argument("--timing-sample-path", default=str(TIMING_SAMPLE_PATH))
    return parser.parse_args()


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _normalize_release_session(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower()
    mapping = {
        "pre_market": "pre_market",
        "market_hours": "market_hours",
        "after_close": "after_close",
    }
    return normalized.map(mapping).fillna("unknown").astype("string")


def _build_tradable_timestamp(tradable_date: pd.Series) -> pd.Series:
    normalized = pd.to_datetime(tradable_date, errors="coerce").dt.normalize()
    return normalized + pd.Timedelta(hours=TRADABLE_SESSION_OPEN_HOUR, minutes=TRADABLE_SESSION_OPEN_MINUTE)


def _stable_event_id(*parts: object) -> str:
    payload = "|".join("" if value is None or pd.isna(value) else str(value) for value in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _build_validation_group(tradable_date: pd.Series) -> pd.Series:
    normalized = pd.to_datetime(tradable_date, errors="coerce")
    year = normalized.dt.year.astype("Int64").astype("string")
    quarter = normalized.dt.quarter.astype("Int64").astype("string")
    return ("y" + year + "_q" + quarter).astype("string")


def _derive_timing_fields(
    filing_date: pd.Series,
    filing_timestamp_local: pd.Series,
    timing_bucket: pd.Series,
) -> pd.DataFrame:
    filing_date_normalized = pd.to_datetime(filing_date, errors="coerce").dt.normalize()
    local_timestamp = pd.to_datetime(filing_timestamp_local, errors="coerce")
    normalized_bucket = timing_bucket.astype("string").str.strip().str.lower()

    event_date_raw = filing_date_normalized.copy()
    timestamp_present = local_timestamp.notna()
    if timestamp_present.any():
        event_date_raw.loc[timestamp_present] = local_timestamp.loc[timestamp_present].dt.tz_localize(None).dt.normalize()

    release_session = pd.Series("unknown", index=filing_date.index, dtype="string")
    timing_confidence = pd.Series("conservative_fallback", index=filing_date.index, dtype="string")
    availability_base_date = filing_date_normalized + pd.Timedelta(days=1)

    if timestamp_present.any():
        minute_of_day = (
            local_timestamp.loc[timestamp_present].dt.hour * 60
            + local_timestamp.loc[timestamp_present].dt.minute
        )
        pre_market_mask = minute_of_day < SESSION_OPEN_MINUTE_OF_DAY
        after_close_mask = minute_of_day >= SESSION_CLOSE_MINUTE_OF_DAY
        market_hours_mask = ~(pre_market_mask | after_close_mask)

        timestamp_index = local_timestamp.index[timestamp_present]
        release_session.loc[timestamp_index[pre_market_mask.to_numpy()]] = "pre_market"
        release_session.loc[timestamp_index[market_hours_mask.to_numpy()]] = "market_hours"
        release_session.loc[timestamp_index[after_close_mask.to_numpy()]] = "after_close"
        timing_confidence.loc[timestamp_present] = "exact"
        availability_base_date.loc[timestamp_present] = event_date_raw.loc[timestamp_present] + pd.Timedelta(days=1)
        availability_base_date.loc[timestamp_index[pre_market_mask.to_numpy()]] = event_date_raw.loc[
            timestamp_index[pre_market_mask.to_numpy()]
        ]

    inferred_session_mask = ~timestamp_present & normalized_bucket.isin(["pre_market", "market_hours", "after_close"])
    release_session.loc[inferred_session_mask] = normalized_bucket.loc[inferred_session_mask]
    timing_confidence.loc[inferred_session_mask] = "inferred_session"
    availability_base_date.loc[inferred_session_mask] = filing_date_normalized.loc[inferred_session_mask] + pd.Timedelta(days=1)
    pre_market_inferred_mask = inferred_session_mask & normalized_bucket.eq("pre_market")
    availability_base_date.loc[pre_market_inferred_mask] = filing_date_normalized.loc[pre_market_inferred_mask]

    date_only_mask = ~timestamp_present & ~normalized_bucket.isin(["pre_market", "market_hours", "after_close"])
    timing_confidence.loc[date_only_mask] = "inferred_date_only"

    return pd.DataFrame(
        {
            "event_date_raw": event_date_raw.astype("datetime64[ns]"),
            "event_timezone": pd.Series(EVENT_TIMEZONE, index=filing_date.index, dtype="string"),
            "release_session": release_session,
            "timing_confidence": timing_confidence,
            "availability_base_date": availability_base_date.astype("datetime64[ns]"),
        }
    )


def _build_period_lookup(fundamentals_df: pd.DataFrame) -> pd.DataFrame:
    lookup = (
        fundamentals_df[
            ["ticker", "form_type", "filing_date", "period_end", "fiscal_period", "fiscal_year"]
        ]
        .copy()
        .sort_values(["ticker", "form_type", "filing_date", "period_end"])
        .drop_duplicates(subset=["ticker", "form_type", "filing_date"], keep="last")
    )
    lookup["ticker"] = lookup["ticker"].astype("string")
    lookup["form_type"] = lookup["form_type"].astype("string").str.upper().str.strip()
    lookup["filing_date"] = pd.to_datetime(lookup["filing_date"], errors="coerce")
    lookup["period_end"] = pd.to_datetime(lookup["period_end"], errors="coerce")
    lookup["fiscal_period"] = lookup["fiscal_period"].astype("string")
    lookup["fiscal_year"] = pd.to_numeric(lookup["fiscal_year"], errors="coerce").astype("Int64")
    return lookup


def build_raw_event_master(
    event_source_path: Path,
    price_path: Path,
    fundamentals_path: Path,
) -> pd.DataFrame:
    price_df = prepare_prices(load_parquet(price_path, ["ticker", "date", "adj_close"], "Price labels"))
    event_source_df = load_parquet(event_source_path, EVENT_REQUIRED_COLUMNS, "SEC filing metadata v1")
    event_source_df = event_source_df.copy()
    event_source_df["ticker"] = event_source_df["ticker"].astype("string")
    event_source_df["cik"] = event_source_df["cik"].astype("string")
    event_source_df["company_name"] = event_source_df["company_name"].astype("string")
    event_source_df["form_type"] = event_source_df["form_type"].astype("string").str.upper().str.strip()
    event_source_df["filing_date"] = pd.to_datetime(event_source_df["filing_date"], errors="coerce")
    event_source_df["filing_timestamp_local"] = pd.to_datetime(event_source_df["filing_timestamp_local"], errors="coerce")
    event_source_df["filing_timestamp_utc"] = pd.to_datetime(event_source_df["filing_timestamp_utc"], errors="coerce", utc=True)
    event_source_df["availability_base_date"] = pd.to_datetime(event_source_df["availability_base_date"], errors="coerce")
    event_source_df["effective_model_date"] = pd.to_datetime(event_source_df["effective_model_date"], errors="coerce")
    event_source_df["accession_number"] = event_source_df["accession_number"].astype("string")
    event_source_df["timing_bucket"] = event_source_df["timing_bucket"].astype("string")
    event_source_df = event_source_df.loc[event_source_df["form_type"].isin(["10-Q", "10-K", "10-Q/A", "10-K/A"])].copy()
    event_source_df = event_source_df.dropna(subset=["ticker", "form_type", "filing_date", "accession_number"]).copy()
    timing_fields = _derive_timing_fields(
        filing_date=event_source_df["filing_date"],
        filing_timestamp_local=event_source_df["filing_timestamp_local"],
        timing_bucket=event_source_df["timing_bucket"],
    )
    for column in timing_fields.columns:
        event_source_df[column] = timing_fields[column]
    event_source_df["availability_base_date"] = timing_fields["availability_base_date"]
    event_source_df = align_event_effective_dates(event_source_df, price_df)

    fundamentals_df = load_fundamentals(fundamentals_path)
    period_lookup = _build_period_lookup(fundamentals_df)
    master_df = event_source_df.merge(
        period_lookup,
        left_on=["ticker", "form_type", "filing_date"],
        right_on=["ticker", "form_type", "filing_date"],
        how="left",
        validate="many_to_one",
    )

    sector_map = get_project_sector_map()
    master_df["base_event_type"] = master_df["form_type"].str.replace("/A", "", regex=False)
    master_df["is_amendment"] = master_df["form_type"].str.endswith("/A").fillna(False)
    master_df["event_id"] = [
        _stable_event_id(ticker, form_type, accession_number, filing_date.date() if pd.notna(filing_date) else "")
        for ticker, form_type, accession_number, filing_date in zip(
            master_df["ticker"],
            master_df["form_type"],
            master_df["accession_number"],
            master_df["filing_date"],
        )
    ]
    master_df["event_type"] = master_df["form_type"]
    master_df["event_timestamp_raw"] = master_df["filing_timestamp_local"]
    master_df["event_date_raw"] = pd.to_datetime(master_df["event_date_raw"], errors="coerce").dt.normalize()
    master_df["event_timezone"] = master_df["event_timezone"].astype("string")
    master_df["release_session"] = _normalize_release_session(master_df["release_session"])
    master_df["timing_confidence"] = master_df["timing_confidence"].astype("string")
    master_df["tradable_date"] = pd.to_datetime(master_df["effective_model_date"], errors="coerce").dt.normalize()
    master_df["tradable_timestamp"] = _build_tradable_timestamp(master_df["tradable_date"])
    master_df["feature_snapshot_timestamp"] = master_df["tradable_timestamp"]
    master_df["source_file_id"] = master_df["accession_number"]
    master_df["feature_version"] = DEFAULT_FEATURE_VERSION
    master_df["label_version"] = DEFAULT_LABEL_VERSION
    master_df["validation_group"] = _build_validation_group(master_df["tradable_date"])
    master_df["sector"] = master_df["ticker"].map(sector_map).astype("string")
    master_df["industry"] = pd.Series("unknown", index=master_df.index, dtype="string")
    master_df["promotion_status"] = "raw"
    master_df["promotion_reason"] = "unfiltered"

    ordered_columns = MODELING_METADATA_COLUMNS + [
        "base_event_type",
        "is_amendment",
        "company_name",
        "filing_timestamp_utc",
        "effective_model_date",
        "timing_bucket",
        "promotion_status",
        "promotion_reason",
    ]
    ordered_columns = list(dict.fromkeys(column for column in ordered_columns if column in master_df.columns))
    ordered_columns += [column for column in master_df.columns if column not in ordered_columns]
    return master_df[ordered_columns].sort_values(
        ["tradable_date", "ticker", "event_type", "source_file_id"]
    ).reset_index(drop=True)


def promote_raw_event_master(raw_master_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = raw_master_df.copy()
    working["sort_timestamp"] = pd.to_datetime(working["event_timestamp_raw"], errors="coerce")
    working["sort_tradable_timestamp"] = pd.to_datetime(working["tradable_timestamp"], errors="coerce")
    working["sort_source_file_id"] = working["source_file_id"].astype("string")
    working = working.sort_values(
        ["ticker", "sort_source_file_id", "sort_timestamp", "sort_tradable_timestamp", "event_id"]
    ).reset_index(drop=True)

    dropped_rows: list[dict[str, object]] = []
    deduped_by_accession: list[pd.Series] = []
    for _, group in working.groupby(["ticker", "sort_source_file_id"], sort=False, dropna=False):
        if len(group) == 1:
            deduped_by_accession.append(group.iloc[0].copy())
            continue
        kept = group.iloc[-1].copy()
        kept["promotion_status"] = "filtered"
        kept["promotion_reason"] = "kept_latest_duplicate_accession"
        deduped_by_accession.append(kept)
        for row in group.iloc[:-1].itertuples(index=False):
            dropped_rows.append(
                {
                    "event_id": row.event_id,
                    "ticker": row.ticker,
                    "source_file_id": row.source_file_id,
                    "event_type": row.event_type,
                    "drop_stage": "accession_dedup",
                    "drop_reason": "duplicate_accession_older_row",
                    "kept_event_id": kept["event_id"],
                }
            )

    accession_df = pd.DataFrame(deduped_by_accession).reset_index(drop=True)
    accession_df["dedup_group_key"] = (
        accession_df["ticker"].astype("string").fillna("")
        + "|"
        + accession_df["base_event_type"].astype("string").fillna("")
        + "|"
        + accession_df["fiscal_year"].astype("string").fillna("")
        + "|"
        + accession_df["fiscal_period"].astype("string").fillna("")
        + "|"
        + accession_df["period_end"].astype("string").fillna("")
    )

    kept_rows: list[pd.Series] = []
    missing_period_mask = (
        accession_df["fiscal_year"].isna() | accession_df["fiscal_period"].isna() | accession_df["period_end"].isna()
    )
    if missing_period_mask.any():
        missing_period_rows = accession_df.loc[missing_period_mask].copy()
        missing_period_rows["promotion_status"] = "filtered"
        missing_period_rows["promotion_reason"] = "kept_missing_period_metadata"
        kept_rows.extend(missing_period_rows.to_dict("records"))

    eligible_df = accession_df.loc[~missing_period_mask].copy()
    eligible_df = eligible_df.sort_values(
        [
            "ticker",
            "base_event_type",
            "fiscal_year",
            "fiscal_period",
            "period_end",
            "is_amendment",
            "sort_tradable_timestamp",
            "sort_timestamp",
            "sort_source_file_id",
        ]
    ).reset_index(drop=True)
    for _, group in eligible_df.groupby(
        ["ticker", "base_event_type", "fiscal_year", "fiscal_period", "period_end"],
        sort=False,
        dropna=False,
    ):
        originals = group.loc[~group["is_amendment"].fillna(False)].copy()
        amendments = group.loc[group["is_amendment"].fillna(False)].copy()
        if not originals.empty:
            kept = originals.iloc[0].copy()
            kept["promotion_status"] = "filtered"
            kept["promotion_reason"] = "kept_original_filing_for_quarter"
            kept_rows.append(kept.to_dict())
            for row in originals.iloc[1:].itertuples(index=False):
                dropped_rows.append(
                    {
                        "event_id": row.event_id,
                        "ticker": row.ticker,
                        "source_file_id": row.source_file_id,
                        "event_type": row.event_type,
                        "drop_stage": "quarter_dedup",
                        "drop_reason": "extra_original_same_quarter",
                        "kept_event_id": kept["event_id"],
                    }
                )
            for row in amendments.itertuples(index=False):
                dropped_rows.append(
                    {
                        "event_id": row.event_id,
                        "ticker": row.ticker,
                        "source_file_id": row.source_file_id,
                        "event_type": row.event_type,
                        "drop_stage": "quarter_dedup",
                        "drop_reason": "amendment_with_original_present",
                        "kept_event_id": kept["event_id"],
                    }
                )
        else:
            kept = amendments.iloc[-1].copy()
            kept["promotion_status"] = "filtered"
            kept["promotion_reason"] = "kept_latest_amendment_without_original"
            kept_rows.append(kept.to_dict())
            for row in amendments.iloc[:-1].itertuples(index=False):
                dropped_rows.append(
                    {
                        "event_id": row.event_id,
                        "ticker": row.ticker,
                        "source_file_id": row.source_file_id,
                        "event_type": row.event_type,
                        "drop_stage": "quarter_dedup",
                        "drop_reason": "older_amendment_same_quarter",
                        "kept_event_id": kept["event_id"],
                    }
                )

    filtered_df = pd.DataFrame(kept_rows)
    if filtered_df.empty:
        raise ValueError("Filtered quarterly event master is empty.")
    filtered_df = filtered_df.drop(columns=["sort_timestamp", "sort_tradable_timestamp", "sort_source_file_id"], errors="ignore")
    filtered_df = filtered_df.sort_values(["tradable_date", "ticker", "event_type", "source_file_id"]).reset_index(drop=True)
    duplicate_audit_df = pd.DataFrame(dropped_rows).sort_values(
        ["drop_stage", "ticker", "event_type", "source_file_id"],
        na_position="last",
    ).reset_index(drop=True) if dropped_rows else pd.DataFrame(
        columns=["event_id", "ticker", "source_file_id", "event_type", "drop_stage", "drop_reason", "kept_event_id"]
    )
    return filtered_df, duplicate_audit_df


def enrich_feature_panel(
    panel_df: pd.DataFrame,
    filtered_master_df: pd.DataFrame,
) -> pd.DataFrame:
    join_columns = [
        "event_id",
        "ticker",
        "event_type",
        "source_file_id",
        "event_timestamp_raw",
        "event_timezone",
        "event_date_raw",
        "release_session",
        "timing_confidence",
        "tradable_timestamp",
        "tradable_date",
        "feature_snapshot_timestamp",
        "feature_version",
        "label_version",
        "validation_group",
        "sector",
        "industry",
        "promotion_reason",
    ]
    lookup = filtered_master_df[join_columns].rename(columns={"source_file_id": "source_id"}).copy()
    enriched = panel_df.merge(
        lookup,
        on=["ticker", "event_type", "source_id"],
        how="left",
        validate="one_to_one",
    )
    if enriched["event_id"].isna().any():
        missing = enriched.loc[enriched["event_id"].isna(), ["ticker", "event_type", "source_id"]]
        raise ValueError(
            "Filtered quarterly event master did not map onto every feature row; "
            f"missing matches: {missing.head(5).to_dict(orient='records')}"
        )

    enriched["fiscal_year"] = pd.to_numeric(enriched["event_fiscal_year"], errors="coerce").astype("Int64")
    enriched["fiscal_period"] = enriched["event_fiscal_period"].astype("string")
    enriched["period_end"] = pd.to_datetime(enriched["event_period_end"], errors="coerce")
    enriched["source_file_id"] = enriched["source_id"].astype("string")
    ordered_columns = MODELING_METADATA_COLUMNS + [
        "promotion_reason",
        "company_name",
        "effective_model_date",
        "timing_bucket",
        "source_id",
    ]
    ordered_columns = [column for column in ordered_columns if column in enriched.columns]
    ordered_columns += [column for column in enriched.columns if column not in ordered_columns]
    return enriched[ordered_columns].sort_values(
        ["tradable_date", "ticker", "event_type", "source_id"]
    ).reset_index(drop=True)


def build_base_panel(feature_panel_df: pd.DataFrame) -> pd.DataFrame:
    available_columns = [column for column in MODELING_METADATA_COLUMNS + ["promotion_reason"] if column in feature_panel_df.columns]
    return feature_panel_df[available_columns].copy()


def build_row_count_audit(
    raw_master_df: pd.DataFrame,
    filtered_master_df: pd.DataFrame,
    base_panel_df: pd.DataFrame,
    feature_panel_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for artifact_name, df in [
        ("quarterly_event_master_raw", raw_master_df),
        ("quarterly_event_master", filtered_master_df),
        ("quarterly_event_panel_base", base_panel_df),
        ("quarterly_event_panel_features", feature_panel_df),
    ]:
        tradable_date = pd.to_datetime(df.get("tradable_date"), errors="coerce")
        rows.append(
            {
                "artifact_name": artifact_name,
                "row_count": int(len(df)),
                "unique_event_id_count": int(df["event_id"].nunique()) if "event_id" in df.columns else 0,
                "unique_ticker_count": int(df["ticker"].nunique()) if "ticker" in df.columns else 0,
                "tradable_date_min": tradable_date.min(),
                "tradable_date_max": tradable_date.max(),
            }
        )
    return pd.DataFrame(rows)


def build_timing_audit(raw_master_df: pd.DataFrame, filtered_master_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for artifact_name, df in [
        ("quarterly_event_master_raw", raw_master_df),
        ("quarterly_event_master", filtered_master_df),
    ]:
        event_date = pd.to_datetime(df.get("event_date_raw"), errors="coerce").dt.normalize()
        tradable_date = pd.to_datetime(df.get("tradable_date"), errors="coerce").dt.normalize()
        metrics = {
            "exact_timestamp_events": int(df["timing_confidence"].eq("exact").sum()),
            "session_inferred_events": int(df["timing_confidence"].eq("inferred_session").sum()),
            "date_only_conservative_events": int(df["timing_confidence"].isin(["inferred_date_only", "conservative_fallback"]).sum()),
            "after_close_shifted_events": int(
                (df["release_session"].eq("after_close") & tradable_date.gt(event_date)).sum()
            ),
            "market_hours_shifted_events": int(
                (df["release_session"].eq("market_hours") & tradable_date.gt(event_date)).sum()
            ),
            "pre_market_same_day_events": int(
                (df["release_session"].eq("pre_market") & tradable_date.eq(event_date)).sum()
            ),
            "unknown_timestamp_events": int(df["release_session"].eq("unknown").sum()),
        }
        for audit_metric, row_count in metrics.items():
            rows.append(
                {
                    "artifact_name": artifact_name,
                    "audit_metric": audit_metric,
                    "row_count": row_count,
                }
            )
    return pd.DataFrame(rows)


def build_timing_confidence_counts(raw_master_df: pd.DataFrame, filtered_master_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for artifact_name, df in [
        ("quarterly_event_master_raw", raw_master_df),
        ("quarterly_event_master", filtered_master_df),
    ]:
        grouped = (
            df.groupby(["release_session", "timing_confidence"], dropna=False)
            .size()
            .reset_index(name="row_count")
            .sort_values(["release_session", "timing_confidence"])
        )
        for row in grouped.itertuples(index=False):
            rows.append(
                {
                    "artifact_name": artifact_name,
                    "release_session": row.release_session,
                    "timing_confidence": row.timing_confidence,
                    "row_count": int(row.row_count),
                }
            )
    return pd.DataFrame(rows)


def build_timing_samples(filtered_master_df: pd.DataFrame, samples_per_session: int = 3) -> pd.DataFrame:
    sample_columns = [
        "event_id",
        "ticker",
        "event_type",
        "event_date_raw",
        "event_timestamp_raw",
        "event_timezone",
        "release_session",
        "timing_confidence",
        "tradable_date",
        "tradable_timestamp",
        "source_file_id",
        "promotion_reason",
    ]
    samples: list[pd.DataFrame] = []
    working = filtered_master_df.copy().sort_values(["release_session", "tradable_date", "ticker", "source_file_id"])
    for session in ["pre_market", "market_hours", "after_close", "unknown"]:
        group = working.loc[working["release_session"] == session, sample_columns]
        if not group.empty:
            samples.append(group.head(samples_per_session))
    if not samples:
        return pd.DataFrame(columns=sample_columns)
    return pd.concat(samples, ignore_index=True)


def main() -> None:
    args = parse_args()
    ensure_stock_prediction_directories()

    price_path = Path(args.price_path)
    event_source_path = Path(args.event_source_path)
    fundamentals_path = Path(args.fundamentals_path)
    raw_master_path = Path(args.raw_master_path)
    filtered_master_path = Path(args.filtered_master_path)
    base_panel_path = Path(args.base_panel_path)
    feature_panel_path = Path(args.feature_panel_path)
    row_count_audit_path = Path(args.row_count_audit_path)
    duplicate_audit_path = Path(args.duplicate_audit_path)
    timing_audit_path = Path(args.timing_audit_path)
    timing_confidence_counts_path = Path(args.timing_confidence_counts_path)
    timing_sample_path = Path(args.timing_sample_path)

    print(f"Building quarterly raw event master from: {event_source_path}")
    raw_master_df = build_raw_event_master(
        event_source_path=event_source_path,
        price_path=price_path,
        fundamentals_path=fundamentals_path,
    )
    filtered_master_df, duplicate_audit_df = promote_raw_event_master(raw_master_df)

    print("Building feature-ready quarterly panel from existing event_panel_v2 pipeline...")
    panel_df = build_event_panel_v2(
        price_path=price_path,
        event_source_path=event_source_path,
        fundamentals_path=fundamentals_path,
    )
    feature_panel_df = enrich_feature_panel(panel_df, filtered_master_df)
    base_panel_df = build_base_panel(feature_panel_df)
    row_count_audit_df = build_row_count_audit(raw_master_df, filtered_master_df, base_panel_df, feature_panel_df)
    timing_audit_df = build_timing_audit(raw_master_df, filtered_master_df)
    timing_confidence_counts_df = build_timing_confidence_counts(raw_master_df, filtered_master_df)
    timing_sample_df = build_timing_samples(filtered_master_df)

    for path in [
        raw_master_path,
        filtered_master_path,
        base_panel_path,
        feature_panel_path,
        row_count_audit_path,
        duplicate_audit_path,
        timing_audit_path,
        timing_confidence_counts_path,
        timing_sample_path,
    ]:
        _ensure_parent_dir(path)

    raw_master_df.to_parquet(raw_master_path, index=False)
    filtered_master_df.to_parquet(filtered_master_path, index=False)
    base_panel_df.to_parquet(base_panel_path, index=False)
    feature_panel_df.to_parquet(feature_panel_path, index=False)
    row_count_audit_df.to_csv(row_count_audit_path, index=False)
    duplicate_audit_df.to_csv(duplicate_audit_path, index=False)
    timing_audit_df.to_csv(timing_audit_path, index=False)
    timing_confidence_counts_df.to_csv(timing_confidence_counts_path, index=False)
    timing_sample_df.to_csv(timing_sample_path, index=False)

    print("\nQuarterly Event Panel Summary")
    print("-" * 60)
    for row in row_count_audit_df.itertuples(index=False):
        print(f"{row.artifact_name:<30} {row.row_count:>6,} rows")
    print(f"Raw master path: {raw_master_path}")
    print(f"Filtered master path: {filtered_master_path}")
    print(f"Base panel path: {base_panel_path}")
    print(f"Feature panel path: {feature_panel_path}")
    print(f"Row-count audit path: {row_count_audit_path}")
    print(f"Duplicate audit path: {duplicate_audit_path}")
    print(f"Timing audit path: {timing_audit_path}")
    print(f"Timing confidence counts path: {timing_confidence_counts_path}")
    print(f"Timing sample path: {timing_sample_path}")


if __name__ == "__main__":
    main()
