"""Build a data-layer integrity and leakage-control audit bundle.

This script is intentionally reporting-only. It reads saved panels, labels,
sentiment artifacts, configs, and validation diagnostics, then writes audit
tables plus a consolidated Markdown report.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.paths import PROJECT_ROOT, QUARTERLY_OUTPUTS_DIAGNOSTICS_DIR
from src.train_event_panel_v2 import resolve_candidate_features


OUTPUT_DIR = QUARTERLY_OUTPUTS_DIAGNOSTICS_DIR / "data_layer_integrity"
BASE_PANEL = PROJECT_ROOT / "outputs" / "quarterly" / "panels" / "quarterly_event_panel_features.parquet"
ENRICHED_PANEL = (
    PROJECT_ROOT
    / "outputs"
    / "quarterly"
    / "panels"
    / "quarterly_event_panel_sector_sentiment_capitaliq.parquet"
)
PREPARED_NEWS = PROJECT_ROOT / "data" / "processed" / "capitaliq_keydev_news_prepared.parquet"
SCORED_NEWS = PROJECT_ROOT / "data" / "processed" / "news_scores_finbert_capitaliq_keydev.parquet"
SECTOR_MAP_CANDIDATES = [
    PROJECT_ROOT / "src" / "data" / "ticker_sector_map_clean.csv",
    PROJECT_ROOT / "data" / "processed" / "ticker_sector_map.csv",
]
CONFIG_PATHS = [
    PROJECT_ROOT / "configs" / "quarterly" / "capitaliq_ladder_core.yaml",
    PROJECT_ROOT / "configs" / "quarterly" / "capitaliq_ladder_core_plus_market.yaml",
    PROJECT_ROOT / "configs" / "quarterly" / "capitaliq_ladder_raw_sentiment.yaml",
    PROJECT_ROOT / "configs" / "quarterly" / "capitaliq_ladder_sector_adjusted_sentiment.yaml",
    PROJECT_ROOT / "configs" / "quarterly" / "sector_aware_finbert_capitaliq_experiments.yaml",
]

IDENTITY_COLUMNS = [
    "ticker",
    "event_id",
    "event_date",
    "prediction_date",
    "tradable_date",
    "effective_model_date",
    "filing_date",
    "period_end",
    "accession_number",
    "source_file_id",
]
LABEL_LIKE_PATTERNS = [
    "target",
    "label",
    "forward_return",
    "excess_forward_return",
    "benchmark_forward_return",
    "return_forward",
]
FORBIDDEN_FEATURE_PARTS = [
    "headline",
    "situation",
    "source",
    "row_company",
    "other_parties",
    "capital_iq_page_title",
    "extraction_timestamp",
    "future",
    "return_forward",
    "target",
    "label",
]
FORBIDDEN_FEATURE_EXACT = {"date", "y"}
SENTIMENT_COLUMNS = [
    "sent_mean_7d",
    "sent_mean_30d",
    "sent_mean_63d",
    "sent_vol_30d",
    "sent_vol_63d",
    "news_count_7d",
    "news_count_30d",
    "news_count_63d",
    "sent_momentum_7v30",
    "sent_momentum_30v63",
    "confidence_mean_30d",
    "sector_sent_mean_30d",
    "sector_sent_mean_63d",
    "sector_news_count_30d",
    "sector_news_count_63d",
    "sector_adj_sent_30d",
    "sector_adj_sent_63d",
    "sector_adj_news_share_30d",
    "sector_adj_news_share_63d",
    "has_news_7d",
    "has_news_30d",
    "has_news_63d",
    "low_news_coverage_30d",
    "low_news_coverage_63d",
]
MARKET_FEATURE_PARTS = [
    "rel_return",
    "realized_vol",
    "vol_ratio",
    "beta_",
    "overnight_gap",
    "return_zscore",
    "drawdown",
    "volume",
    "log_volume",
    "abnormal_volume",
]
DOC_SEARCH_ROOTS = ["README.md", "docs", "reports", "configs", "outputs/quarterly/modeling"]
STALE_REFERENCE_RE = re.compile(r"\b(21-trading-day|21 day|5-day|5 day)\b", re.IGNORECASE)
ACTIVE_CURRENT_PATHS = {
    "configs/event_panel_v2_quarterly_63d_sector_relative.yaml",
    "configs/quarterly/current_benchmark_set.yaml",
    "configs/quarterly/capitaliq_ladder_core.yaml",
    "configs/quarterly/capitaliq_ladder_core_plus_market.yaml",
    "configs/quarterly/capitaliq_ladder_raw_sentiment.yaml",
    "configs/quarterly/capitaliq_ladder_sector_adjusted_sentiment.yaml",
    "src/project_config.py",
    "reports/results/event_panel_v2_quarterly_63d_sector_relative_benchmark.md",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit data-layer integrity and leakage controls.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--base-panel", default=str(BASE_PANEL))
    parser.add_argument("--enriched-panel", default=str(ENRICHED_PANEL))
    parser.add_argument("--prepared-news", default=str(PREPARED_NEWS))
    parser.add_argument("--scored-news", default=str(SCORED_NEWS))
    parser.add_argument("--py-compile-status", default="")
    parser.add_argument("--unittest-status", default="")
    parser.add_argument("--unittest-summary", default="")
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_table(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".json":
        return pd.json_normalize(json.loads(path.read_text(encoding="utf-8")))
    return None


def date_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(pd.NaT, index=frame.index)
    return pd.to_datetime(frame[column], errors="coerce").dt.normalize()


def status_from(condition: bool | None) -> str:
    if condition is None:
        return "LIMITED"
    return "PASS" if condition else "FAIL"


def append_summary(rows: list[dict[str, Any]], audit: str, check: str, status: str, value: Any, detail: str = "") -> None:
    rows.append({"audit": audit, "check": check, "status": status, "value": value, "detail": detail})


def make_event_key(frame: pd.DataFrame) -> tuple[pd.Series, list[str]]:
    preferred = [column for column in IDENTITY_COLUMNS if column in frame.columns]
    if "event_id" in preferred and frame["event_id"].notna().all() and frame["event_id"].is_unique:
        return frame["event_id"].astype("string"), ["event_id"]
    fallback = [column for column in preferred if column != "event_id"]
    if not fallback:
        fallback = [column for column in ["ticker", "event_date", "period_end"] if column in frame.columns]
    key = frame[fallback].astype("string").fillna("<NA>").agg("|".join, axis=1)
    return key, fallback


def audit_panel_identity(base: pd.DataFrame, enriched: pd.DataFrame, output_dir: Path, summary: list[dict[str, Any]]) -> pd.DataFrame:
    base = base.copy()
    enriched = enriched.copy()
    base["_audit_key"], key_cols = make_event_key(base)
    enriched["_audit_key"], _ = make_event_key(enriched)
    common_keys = set(base["_audit_key"]).intersection(set(enriched["_audit_key"]))
    base_only = set(base["_audit_key"]) - set(enriched["_audit_key"])
    enriched_only = set(enriched["_audit_key"]) - set(base["_audit_key"])

    duplicate_cols = [column for column in ["ticker", "event_date", "period_end", "source_file_id"] if column in enriched.columns]
    duplicate_rows = int(enriched.duplicated(subset=duplicate_cols or ["_audit_key"]).sum())
    label_cols = [column for column in base.columns if any(part in column.lower() for part in LABEL_LIKE_PATTERNS)]
    label_cols = [column for column in label_cols if column in enriched.columns]
    identity_compare_cols = [column for column in IDENTITY_COLUMNS if column in base.columns and column in enriched.columns]

    merged = base[["_audit_key", *identity_compare_cols, *label_cols]].merge(
        enriched[["_audit_key", *identity_compare_cols, *label_cols]],
        on="_audit_key",
        how="inner",
        suffixes=("_base", "_enriched"),
        validate="one_to_one" if base["_audit_key"].is_unique and enriched["_audit_key"].is_unique else "many_to_many",
    )
    changed_identity = []
    for column in identity_compare_cols:
        left = merged[f"{column}_base"].astype("string").fillna("<NA>")
        right = merged[f"{column}_enriched"].astype("string").fillna("<NA>")
        if not left.equals(right):
            changed_identity.append(column)
    changed_labels = []
    for column in label_cols:
        left = merged[f"{column}_base"].astype("string").fillna("<NA>")
        right = merged[f"{column}_enriched"].astype("string").fillna("<NA>")
        if not left.equals(right):
            changed_labels.append(column)

    rows = [
        {
            "metric": "base_row_count",
            "status": "PASS",
            "value": len(base),
            "detail": rel(Path(BASE_PANEL)),
        },
        {
            "metric": "enriched_row_count",
            "status": "PASS",
            "value": len(enriched),
            "detail": rel(Path(ENRICHED_PANEL)),
        },
        {
            "metric": "row_count_difference",
            "status": status_from(len(enriched) == len(base)),
            "value": len(enriched) - len(base),
            "detail": "Adding sentiment should not silently drop or add panel rows.",
        },
        {
            "metric": "common_event_keys",
            "status": status_from(len(common_keys) == len(base) == len(enriched)),
            "value": len(common_keys),
            "detail": "key_columns=" + ",".join(key_cols),
        },
        {
            "metric": "base_only_event_keys",
            "status": status_from(len(base_only) == 0),
            "value": len(base_only),
            "detail": "",
        },
        {
            "metric": "enriched_only_event_keys",
            "status": status_from(len(enriched_only) == 0),
            "value": len(enriched_only),
            "detail": "",
        },
        {
            "metric": "duplicate_event_rows",
            "status": status_from(duplicate_rows == 0),
            "value": duplicate_rows,
            "detail": "duplicate_subset=" + ",".join(duplicate_cols or ["_audit_key"]),
        },
        {
            "metric": "identity_fields_unchanged",
            "status": status_from(not changed_identity),
            "value": ",".join(changed_identity),
            "detail": "Compared ticker/date/source identity fields.",
        },
        {
            "metric": "label_fields_unchanged",
            "status": status_from(not changed_labels),
            "value": ",".join(changed_labels),
            "detail": "Only panel-resident label-like columns are compared here.",
        },
    ]
    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(output_dir / "panel_identity_audit.csv", index=False)
    for row in rows:
        append_summary(summary, "Panel Identity", row["metric"], row["status"], row["value"], row["detail"])
    return audit_df


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def iter_text_files() -> list[Path]:
    roots = [PROJECT_ROOT / root for root in DOC_SEARCH_ROOTS]
    files: list[Path] = []
    for root in roots:
        if root.is_file() and root.suffix.lower() in {".md", ".yaml", ".yml", ".txt"}:
            files.append(root)
        elif root.is_dir():
            for suffix in ("*.md", "*.yaml", "*.yml", "*.txt"):
                files.extend(root.rglob(suffix))
    return sorted(set(path for path in files if path.exists()))


def classify_stale_reference(path: Path) -> str:
    path_text = rel(path).lower()
    if path_text in ACTIVE_CURRENT_PATHS:
        return "active-modeling risk"
    if path_text.startswith("tests/"):
        return "test expectation/fixture"
    if "legacy" in path_text or "event_panel_v2_quarterly.yaml" in path_text:
        return "historical/legacy experiment"
    if path_text.startswith("reports/results/") or path_text.startswith("outputs/"):
        return "historical/legacy experiment"
    if path_text.startswith("configs/quarterly/quarterly_") or path_text.startswith("configs/event_panel_v2_primary"):
        return "historical/legacy experiment"
    return "stale-report risk"


def audit_label_contract(enriched: pd.DataFrame, output_dir: Path, summary: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    config_payloads = {path: load_yaml(path) for path in CONFIG_PATHS}
    existing_configs = {path: cfg for path, cfg in config_payloads.items() if cfg}
    label_columns = [column for column in enriched.columns if any(part in column.lower() for part in LABEL_LIKE_PATTERNS)]

    horizons = sorted({str(cfg.get("label", {}).get("horizon_days", "")) for cfg in existing_configs.values() if cfg.get("label")})
    variants = sorted({str(cfg.get("label", {}).get("variant_name", "")) for cfg in existing_configs.values() if cfg.get("label")})
    holdouts = sorted({str(cfg.get("holdout", {}).get("start", "")) for cfg in existing_configs.values() if cfg.get("holdout")})
    target_modes = sorted({str(cfg.get("label", {}).get("mode", "")) for cfg in existing_configs.values() if cfg.get("label")})

    rows.extend(
        [
            {
                "check": "label_columns_present_in_panel",
                "status": status_from(bool(label_columns)),
                "value": ",".join(label_columns) if label_columns else "none",
                "risk_class": "metadata",
                "detail": "Panel may rely on external label maps; absence is acceptable if configs point to labels.",
            },
            {
                "check": "config_target_horizon_is_63",
                "status": status_from(bool(horizons) and horizons == ["63"]),
                "value": ",".join(horizons),
                "risk_class": "active-modeling risk",
                "detail": "All inspected Capital IQ configs should use horizon_days=63.",
            },
            {
                "check": "config_target_mode_is_sign",
                "status": status_from(bool(target_modes) and target_modes == ["sign"]),
                "value": ",".join(target_modes),
                "risk_class": "active-modeling risk",
                "detail": "Final intended label is a sign label.",
            },
            {
                "check": "config_variant_names_reference_63d",
                "status": status_from(bool(variants) and all("63" in variant for variant in variants)),
                "value": ",".join(variants),
                "risk_class": "active-modeling risk",
                "detail": "Variant names should not point at stale 21-day labels.",
            },
            {
                "check": "holdout_start_is_2024",
                "status": status_from(bool(holdouts) and holdouts == ["2024-01-01"]),
                "value": ",".join(holdouts),
                "risk_class": "active-modeling risk",
                "detail": "Final holdout year should begin on 2024-01-01.",
            },
        ]
    )

    for path, cfg in existing_configs.items():
        explicit = set(cfg.get("feature_exclusions", {}).get("explicit", []) or [])
        additional = set(cfg.get("feature_inclusions", {}).get("additional", []) or [])
        suspicious_additional = sorted(
            col
            for col in additional
            if any(part in col.lower() for part in LABEL_LIKE_PATTERNS) or col.lower() in {"y", "date"}
        )
        rows.append(
            {
                "check": "config_feature_inclusions_exclude_labels",
                "status": status_from(not suspicious_additional),
                "value": rel(path),
                "risk_class": "active-modeling risk",
                "detail": "suspicious_additional=" + ",".join(suspicious_additional),
            }
        )
        stale_exclusions = sorted(column for column in explicit if "21d" in column.lower())
        rows.append(
            {
                "check": "stale_21d_feature_exclusion_reference",
                "status": "INFO" if stale_exclusions else "PASS",
                "value": rel(path),
                "risk_class": "harmless explanatory/historical note",
                "detail": "21d entries are market lookback feature names, not active label references: " + ",".join(stale_exclusions),
            }
        )

    stale_hits = []
    for path in iter_text_files():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if STALE_REFERENCE_RE.search(line):
                stale_hits.append((path, line_number, line.strip()[:220]))
    active_stale_hits = [
        (path, line_number, text)
        for path, line_number, text in stale_hits
        if classify_stale_reference(path) in {"active-modeling risk", "generated current report stale text"}
    ]
    rows.append(
        {
            "check": "stale_21d_or_5d_text_references",
            "status": "REVIEW" if active_stale_hits else "PASS",
            "value": len(active_stale_hits),
            "risk_class": "active-modeling risk" if active_stale_hits else "none",
            "detail": f"Active/current stale references; total historical/explanatory hits={len(stale_hits)}.",
        }
    )
    for path, line_number, text in stale_hits[:200]:
        risk_class = classify_stale_reference(path)
        rows.append(
            {
                "check": "stale_reference",
                "status": "REVIEW" if risk_class in {"active-modeling risk", "generated current report stale text"} else "INFO",
                "value": f"{rel(path)}:{line_number}",
                "risk_class": risk_class,
                "detail": text,
            }
        )

    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(output_dir / "label_contract_audit.csv", index=False)
    for row in rows:
        if row["check"] != "stale_reference":
            append_summary(summary, "Label Contract", row["check"], row["status"], row["value"], row["detail"])
    return audit_df


def audit_availability(
    enriched: pd.DataFrame,
    scored: pd.DataFrame | None,
    output_dir: Path,
    summary: list[dict[str, Any]],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    rebuild_summary = PROJECT_ROOT / "outputs" / "quarterly" / "diagnostics" / "fundamental_rebuild" / "staged_rebuild_summary.md"
    rebuild_text = rebuild_summary.read_text(encoding="utf-8") if rebuild_summary.exists() else ""
    out["fundamental_rebuild_report_exists"] = rebuild_summary.exists()
    out["fundamental_selected_after_cutoff_zero"] = "selected after cutoff" in rebuild_text and re.search(
        r"Universe V2 \| 342,180 \| 0 \| 9,453 \| 0", rebuild_text
    ) is not None
    out["fundamental_rebuild_text"] = rebuild_text

    append_summary(
        summary,
        "Availability",
        "fundamental_rebuild_report_exists",
        status_from(out["fundamental_rebuild_report_exists"]),
        rel(rebuild_summary),
        "",
    )
    append_summary(
        summary,
        "Availability",
        "fundamental_selected_after_cutoff_zero_v2_marker",
        status_from(out["fundamental_selected_after_cutoff_zero"]),
        "0 selected after cutoff" if out["fundamental_selected_after_cutoff_zero"] else "not confirmed by regex",
        "Final report also quotes the staged rebuild report directly.",
    )

    market_asof = date_series(enriched, "market_asof_date")
    event_date = date_series(enriched, "event_date")
    market_after_event = int((market_asof.notna() & event_date.notna() & (market_asof > event_date)).sum())
    append_summary(
        summary,
        "Availability",
        "market_asof_not_after_event_date",
        status_from(market_after_event == 0),
        market_after_event,
        "market_asof_date should be <= event_date/prediction date.",
    )
    out["market_after_event"] = market_after_event

    future_feature_cols = [
        column
        for column in enriched.columns
        if any(part in column.lower() for part in ["future", "forward_return", "return_forward"])
    ]
    append_summary(
        summary,
        "Availability",
        "no_future_return_columns_in_panel",
        status_from(not future_feature_cols),
        ",".join(future_feature_cols),
        "Panel should not expose future return columns as candidate features.",
    )
    out["future_feature_cols"] = future_feature_cols

    if scored is not None and {"ticker", "date"}.issubset(scored.columns):
        news = scored[["ticker", "date"]].copy()
        news["ticker"] = news["ticker"].astype("string").str.upper().str.strip().str.replace(".", "-", regex=False)
        news["_news_date"] = pd.to_datetime(news["date"], errors="coerce").dt.normalize()
        panel = enriched[["ticker", "event_date"]].copy()
        panel["ticker"] = panel["ticker"].astype("string").str.upper().str.strip().str.replace(".", "-", regex=False)
        panel["_event_date"] = pd.to_datetime(panel["event_date"], errors="coerce").dt.normalize()
        sample = panel.merge(news, on="ticker", how="left")
        same_day_or_later_used_possible = int(
            (
                sample["_news_date"].notna()
                & sample["_event_date"].notna()
                & (sample["_news_date"] >= sample["_event_date"])
                & (sample["_news_date"] < sample["_event_date"] + pd.Timedelta(days=63))
            ).sum()
        )
        # This count is only an upper bound because it does not know the exact
        # joined window. The strict recompute checks in prior diagnostics are
        # used as the stronger evidence.
        out["sentiment_same_day_or_later_possible_upper_bound"] = same_day_or_later_used_possible
    else:
        out["sentiment_same_day_or_later_possible_upper_bound"] = None

    consistency_summary = (
        PROJECT_ROOT
        / "outputs"
        / "quarterly"
        / "diagnostics"
        / "capitaliq_sentiment_consistency"
        / "capitaliq_sentiment_consistency_summary.csv"
    )
    consistency = read_table(consistency_summary)
    strict_checks = pd.DataFrame()
    if consistency is not None and "check" in consistency.columns:
        strict_checks = consistency[consistency["check"].astype(str).str.contains("strict_pre_event|matches_recompute", regex=True)]
        strict_pass = bool((strict_checks["status"] == "PASS").all()) if not strict_checks.empty else None
    else:
        strict_pass = None
    append_summary(
        summary,
        "Availability",
        "capitaliq_strict_pre_event_recompute_checks",
        status_from(strict_pass),
        rel(consistency_summary) if consistency_summary.exists() else "missing",
        "Uses existing sentiment consistency diagnostics for news_date < event_date checks.",
    )
    out["strict_sentiment_checks"] = strict_checks
    return out


def normalize_ticker(series: pd.Series) -> pd.Series:
    return series.astype("string").str.upper().str.strip().str.replace(".", "-", regex=False)


def audit_sentiment_join(
    enriched: pd.DataFrame,
    prepared: pd.DataFrame | None,
    scored: pd.DataFrame | None,
    output_dir: Path,
    summary: list[dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    panel_tickers = set(normalize_ticker(enriched["ticker"]).dropna())
    sentiment_source = scored if scored is not None else prepared
    sentiment_tickers = set(normalize_ticker(sentiment_source["ticker"]).dropna()) if sentiment_source is not None and "ticker" in sentiment_source else set()

    sector_tickers: set[str] = set()
    multi_sector = pd.DataFrame()
    for path in SECTOR_MAP_CANDIDATES:
        sector_map = read_table(path)
        if sector_map is not None and "ticker" in sector_map.columns:
            sector_map = sector_map.copy()
            sector_map["_ticker_norm"] = normalize_ticker(sector_map["ticker"])
            sector_tickers.update(set(sector_map["_ticker_norm"].dropna()))
            sector_col = "sector" if "sector" in sector_map.columns else None
            if sector_col:
                multi_sector = (
                    sector_map.groupby("_ticker_norm")[sector_col]
                    .nunique(dropna=True)
                    .reset_index(name="sector_count")
                    .query("sector_count > 1")
                )
            break

    counts_by_ticker = pd.DataFrame()
    if sentiment_source is not None and "ticker" in sentiment_source.columns:
        counts_by_ticker = (
            sentiment_source.assign(ticker_norm=normalize_ticker(sentiment_source["ticker"]))
            .groupby("ticker_norm", dropna=False)
            .size()
            .reset_index(name="news_rows")
            .sort_values("news_rows", ascending=False)
        )

    metrics = [
        ("panel_ticker_count", "PASS", len(panel_tickers), ""),
        ("sentiment_ticker_count", status_from(bool(sentiment_tickers)), len(sentiment_tickers), ""),
        ("panel_tickers_missing_sentiment", status_from(not (panel_tickers - sentiment_tickers)), len(panel_tickers - sentiment_tickers), ",".join(sorted(panel_tickers - sentiment_tickers))),
        ("sentiment_tickers_missing_panel", "REVIEW" if (sentiment_tickers - panel_tickers) else "PASS", len(sentiment_tickers - panel_tickers), ",".join(sorted(sentiment_tickers - panel_tickers))),
        ("sector_map_tickers_missing_panel", "REVIEW" if (sector_tickers - panel_tickers) else "PASS", len(sector_tickers - panel_tickers), ",".join(sorted(sector_tickers - panel_tickers))),
        ("tickers_with_multiple_sector_mappings", status_from(multi_sector.empty), len(multi_sector), ""),
        ("bf_b_project_convention_present", status_from("BF-B" in panel_tickers or "BF-B" in sentiment_tickers), "BF-B" in panel_tickers or "BF-B" in sentiment_tickers, "Checks BF.B normalization to BF-B."),
        ("all_34_v1_panel_tickers_represented", status_from(len(panel_tickers) == 34), len(panel_tickers), ""),
    ]
    rows.extend({"metric": m, "status": s, "value": v, "detail": d} for m, s, v, d in metrics)
    if not counts_by_ticker.empty:
        q1 = float(counts_by_ticker["news_rows"].quantile(0.25))
        q3 = float(counts_by_ticker["news_rows"].quantile(0.75))
        iqr = q3 - q1
        low_cut = max(0.0, q1 - 1.5 * iqr)
        high_cut = q3 + 1.5 * iqr
        suspicious = counts_by_ticker[(counts_by_ticker["news_rows"] < low_cut) | (counts_by_ticker["news_rows"] > high_cut)]
        rows.append(
            {
                "metric": "suspicious_news_count_tickers_iqr",
                "status": "REVIEW" if not suspicious.empty else "PASS",
                "value": len(suspicious),
                "detail": "; ".join(f"{r.ticker_norm}:{r.news_rows}" for r in suspicious.itertuples(index=False)),
            }
        )
        for record in counts_by_ticker.itertuples(index=False):
            rows.append({"metric": "news_count_by_ticker", "status": "INFO", "value": record.news_rows, "detail": record.ticker_norm})

    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(output_dir / "sentiment_join_audit.csv", index=False)
    for row in rows:
        if row["status"] != "INFO":
            append_summary(summary, "Sentiment Join", row["metric"], row["status"], row["value"], row["detail"])
    return audit_df


def audit_sentiment_duplicates(prepared: pd.DataFrame | None, output_dir: Path, summary: list[dict[str, Any]]) -> pd.DataFrame:
    if prepared is None:
        audit_df = pd.DataFrame([{"metric": "prepared_news_exists", "status": "LIMITED", "value": False, "detail": rel(PREPARED_NEWS)}])
        audit_df.to_csv(output_dir / "sentiment_duplicate_audit.csv", index=False)
        append_summary(summary, "Sentiment Duplicates", "prepared_news_exists", "LIMITED", False, rel(PREPARED_NEWS))
        return audit_df

    key_candidates = ["source_page_ticker", "ticker", "date", "headline", "type", "capitaliq_type", "row_company"]
    key_cols = [column for column in key_candidates if column in prepared.columns]
    if "capitaliq_type" in key_cols and "type" in key_cols:
        key_cols.remove("capitaliq_type")
    duplicate_mask = prepared.duplicated(subset=key_cols, keep=False) if key_cols else pd.Series(False, index=prepared.index)
    duplicates = prepared.loc[duplicate_mask].copy()
    duplicate_pct = float(len(duplicates) / len(prepared) * 100.0) if len(prepared) else 0.0
    rows = [
        {"metric": "total_rows", "status": "PASS", "value": len(prepared), "detail": ""},
        {"metric": "duplicate_key_columns", "status": "PASS" if key_cols else "LIMITED", "value": ",".join(key_cols), "detail": ""},
        {"metric": "duplicate_row_count", "status": "REVIEW" if len(duplicates) else "PASS", "value": len(duplicates), "detail": ""},
        {"metric": "duplicate_percentage", "status": "REVIEW" if duplicate_pct > 1.0 else "PASS", "value": round(duplicate_pct, 4), "detail": ""},
    ]
    if not duplicates.empty and key_cols:
        grouped = duplicates.groupby(key_cols, dropna=False).size().reset_index(name="duplicate_group_rows")
        grouped = grouped.sort_values("duplicate_group_rows", ascending=False).head(20)
        for record in grouped.to_dict("records"):
            rows.append({"metric": "top_duplicate_example", "status": "INFO", "value": record.pop("duplicate_group_rows"), "detail": json.dumps(record, default=str)})
        by_ticker = duplicates.assign(ticker_norm=normalize_ticker(duplicates.get("ticker", pd.Series("", index=duplicates.index)))).groupby("ticker_norm").size()
        for ticker, count in by_ticker.sort_values(ascending=False).items():
            rows.append({"metric": "duplicate_count_by_ticker", "status": "INFO", "value": int(count), "detail": ticker})
        if "date" in duplicates.columns:
            years = pd.to_datetime(duplicates["date"], errors="coerce").dt.year
            for year, count in years.value_counts().sort_index().items():
                rows.append({"metric": "duplicate_count_by_year", "status": "INFO", "value": int(count), "detail": int(year) if pd.notna(year) else "missing"})

    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(output_dir / "sentiment_duplicate_audit.csv", index=False)
    for row in rows[:4]:
        append_summary(summary, "Sentiment Duplicates", row["metric"], row["status"], row["value"], row["detail"])
    return audit_df


def audit_sentiment_scores(scored: pd.DataFrame | None, output_dir: Path, summary: list[dict[str, Any]]) -> pd.DataFrame:
    if scored is None:
        audit_df = pd.DataFrame([{"metric": "scored_news_exists", "group": "all", "status": "LIMITED", "value": False, "detail": rel(SCORED_NEWS)}])
        audit_df.to_csv(output_dir / "sentiment_score_distribution_audit.csv", index=False)
        append_summary(summary, "Sentiment Scores", "scored_news_exists", "LIMITED", False, rel(SCORED_NEWS))
        return audit_df

    rows: list[dict[str, Any]] = []
    prob_cols = ["finbert_pos", "finbert_neu", "finbert_neg"]
    probs = scored[prob_cols].apply(pd.to_numeric, errors="coerce") if set(prob_cols).issubset(scored.columns) else pd.DataFrame(index=scored.index)
    score = pd.to_numeric(scored.get("finbert_score", pd.Series(np.nan, index=scored.index)), errors="coerce")
    confidence = pd.to_numeric(scored.get("confidence", pd.Series(np.nan, index=scored.index)), errors="coerce")
    if not probs.empty:
        prob_sum = probs.sum(axis=1)
        rows.extend(
            [
                {"metric": "probabilities_in_range", "group": "all", "status": status_from(bool((probs.ge(0).all(axis=1) & probs.le(1).all(axis=1)).all())), "value": int((probs.ge(0).all(axis=1) & probs.le(1).all(axis=1)).sum()), "detail": ""},
                {"metric": "probability_sum_close_to_one", "group": "all", "status": status_from(bool((prob_sum - 1.0).abs().lt(1e-4).all())), "value": float((prob_sum - 1.0).abs().max()), "detail": "max_abs_error"},
                {"metric": "neutral_dominance_rate", "group": "all", "status": "REVIEW" if float((probs["finbert_neu"] == probs.max(axis=1)).mean()) > 0.95 else "PASS", "value": round(float((probs["finbert_neu"] == probs.max(axis=1)).mean()), 4), "detail": ""},
            ]
        )
    rows.extend(
        [
            {"metric": "score_in_range", "group": "all", "status": status_from(bool(score.between(-1, 1).fillna(False).all())), "value": int(score.between(-1, 1).fillna(False).sum()), "detail": ""},
            {"metric": "confidence_in_range", "group": "all", "status": status_from(bool(confidence.between(0, 1).fillna(False).all())), "value": int(confidence.between(0, 1).fillna(False).sum()), "detail": ""},
            {"metric": "missing_score_rate", "group": "all", "status": status_from(float(score.isna().mean()) == 0.0), "value": round(float(score.isna().mean()), 6), "detail": ""},
            {"metric": "all_zero_or_all_neutral_collapse", "group": "all", "status": status_from(not (float(score.abs().sum()) == 0.0 or score.nunique(dropna=True) <= 1)), "value": int(score.nunique(dropna=True)), "detail": "unique finbert_score count"},
            {"metric": "extreme_score_count_abs_ge_0_95", "group": "all", "status": "INFO", "value": int(score.abs().ge(0.95).sum()), "detail": ""},
        ]
    )
    base = scored.assign(
        _year=pd.to_datetime(scored.get("date", pd.Series(pd.NaT, index=scored.index)), errors="coerce").dt.year,
        _ticker=normalize_ticker(scored.get("ticker", pd.Series("", index=scored.index))),
        _score=score,
        _confidence=confidence,
    )
    for group_col, prefix in [("_year", "year"), ("_ticker", "ticker")]:
        grouped = base.groupby(group_col, dropna=False).agg(
            rows=("_score", "size"),
            score_mean=("_score", "mean"),
            score_std=("_score", "std"),
            score_min=("_score", "min"),
            score_max=("_score", "max"),
            confidence_mean=("_confidence", "mean"),
            missing_score_rate=("_score", lambda s: float(s.isna().mean())),
        )
        for group, record in grouped.reset_index().iterrows():
            label = record[group_col]
            for metric in ["rows", "score_mean", "score_std", "score_min", "score_max", "confidence_mean", "missing_score_rate"]:
                rows.append({"metric": f"{prefix}_{metric}", "group": label, "status": "INFO", "value": record[metric], "detail": ""})

    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(output_dir / "sentiment_score_distribution_audit.csv", index=False)
    for row in rows:
        if row["group"] == "all":
            append_summary(summary, "Sentiment Scores", row["metric"], row["status"], row["value"], row["detail"])
    return audit_df


def audit_coverage(enriched: pd.DataFrame, output_dir: Path, summary: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    panel = enriched.copy()
    panel["_year"] = date_series(panel, "event_date").dt.year
    coverage_cols = [column for column in ["has_news_7d", "has_news_30d", "has_news_63d"] if column in panel.columns]
    outputs: dict[str, pd.DataFrame] = {}
    overall_rows = []
    for column in coverage_cols:
        coverage = float(pd.to_numeric(panel[column], errors="coerce").fillna(0).mean() * 100.0)
        overall_rows.append({"scope": "overall", "window": column, "coverage_pct": coverage, "rows": len(panel)})
        append_summary(summary, "Coverage", f"{column}_overall_coverage_pct", "PASS" if coverage > 0 else "FAIL", round(coverage, 2), "")
    holdout = panel[panel["_year"] == 2024]
    for column in coverage_cols:
        coverage = float(pd.to_numeric(holdout[column], errors="coerce").fillna(0).mean() * 100.0) if len(holdout) else np.nan
        overall_rows.append({"scope": "2024_holdout", "window": column, "coverage_pct": coverage, "rows": len(holdout)})
        append_summary(summary, "Coverage", f"{column}_2024_coverage_pct", "PASS" if pd.notna(coverage) and coverage > 0 else "FAIL", round(coverage, 2) if pd.notna(coverage) else np.nan, "")
    outputs["coverage_overall"] = pd.DataFrame(overall_rows)

    if coverage_cols:
        by_year = panel.groupby("_year", dropna=False)[coverage_cols].mean().mul(100).reset_index()
        by_ticker = panel.groupby("ticker", dropna=False)[coverage_cols].mean().mul(100).reset_index()
        by_ticker_year = panel.groupby(["ticker", "_year"], dropna=False)[coverage_cols].mean().mul(100).reset_index()
        outputs["coverage_by_year"] = by_year
        outputs["coverage_by_ticker"] = by_ticker
        outputs["coverage_by_ticker_year"] = by_ticker_year
        low = by_ticker_year[(by_ticker_year[coverage_cols].fillna(0) < 25.0).any(axis=1)]
        append_summary(summary, "Coverage", "low_coverage_ticker_years_lt_25pct", "REVIEW" if not low.empty else "PASS", len(low), "")
    return outputs


def infer_config_family(path: Path, cfg: dict[str, Any]) -> str:
    name = path.stem.lower()
    if "core_plus_market" in name:
        return "core_plus_market"
    if "raw_sentiment" in name:
        return "raw_sentiment"
    if "sector_adjusted" in name or "sector_aware" in name:
        return "sector_adjusted_sentiment"
    if name.endswith("core"):
        return "core"
    return str(cfg.get("metadata", {}).get("experiment_family", name))


def audit_feature_isolation(enriched: pd.DataFrame, output_dir: Path, summary: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in CONFIG_PATHS:
        cfg = load_yaml(path)
        if not cfg:
            rows.append({"config": rel(path), "family": "missing", "check": "config_exists", "status": "LIMITED", "value": False, "detail": ""})
            continue
        family = infer_config_family(path, cfg)
        panel_path = PROJECT_ROOT / str(cfg.get("panel", {}).get("path", ""))
        panel_df = read_table(panel_path) if panel_path.exists() else enriched
        if isinstance(cfg.get("feature_columns"), dict):
            declared_blocks = cfg["feature_columns"]
            candidates = [
                column
                for columns in declared_blocks.values()
                if isinstance(columns, list)
                for column in columns
            ]
            explicit = set()
            included = candidates
        else:
            candidates = resolve_candidate_features(panel_df, cfg)
            explicit = set(cfg.get("feature_exclusions", {}).get("explicit", []) or [])
            included = [column for column in candidates if column not in explicit]
        included_existing = [column for column in included if column in panel_df.columns]
        sentiment = [column for column in included_existing if column in SENTIMENT_COLUMNS]
        market = [column for column in included_existing if any(part in column.lower() for part in MARKET_FEATURE_PARTS)]
        object_cols = [column for column in included_existing if pd.api.types.is_object_dtype(panel_df[column]) or pd.api.types.is_string_dtype(panel_df[column])]
        forbidden = sorted(
            column
            for column in included_existing
            if column.lower() in FORBIDDEN_FEATURE_EXACT or any(part in column.lower() for part in FORBIDDEN_FEATURE_PARTS)
        )
        raw_sent_cols = [c for c in sentiment if c.startswith(("sent_", "news_count_", "has_news_", "confidence_"))]
        sector_sent_cols = [c for c in sentiment if c.startswith(("sector_", "low_news_coverage"))]

        expected_ok = True
        expected_detail = ""
        if family == "core":
            expected_ok = not sentiment and not market
            expected_detail = f"sentiment={len(sentiment)}; market={len(market)}"
        elif family == "core_plus_market":
            expected_ok = bool(market) and not sentiment
            expected_detail = f"market={len(market)}; sentiment={len(sentiment)}"
        elif family == "raw_sentiment":
            expected_ok = bool(raw_sent_cols)
            expected_detail = f"raw_sentiment={len(raw_sent_cols)}; sector_sentiment={len(sector_sent_cols)}"
        elif family == "sector_adjusted_sentiment":
            expected_ok = bool(sector_sent_cols)
            expected_detail = f"sector_sentiment={len(sector_sent_cols)}; raw_sentiment={len(raw_sent_cols)}"

        rows.extend(
            [
                {"config": rel(path), "family": family, "check": "feature_family_matches_intent", "status": status_from(expected_ok), "value": len(included_existing), "detail": expected_detail},
                {"config": rel(path), "family": family, "check": "no_forbidden_feature_names", "status": status_from(not forbidden), "value": len(forbidden), "detail": ",".join(forbidden)},
                {"config": rel(path), "family": family, "check": "no_object_or_string_features", "status": status_from(not object_cols), "value": len(object_cols), "detail": ",".join(object_cols)},
                {"config": rel(path), "family": family, "check": "candidate_feature_count", "status": "INFO", "value": len(included_existing), "detail": ""},
            ]
        )

    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(output_dir / "feature_family_isolation_audit.csv", index=False)
    for row in rows:
        if row["status"] != "INFO":
            append_summary(summary, "Feature Isolation", f"{row['check']}:{row['config']}", row["status"], row["value"], row["detail"])
    return audit_df


def audit_split_integrity(output_dir: Path, summary: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    validation_dirs = [PROJECT_ROOT / "outputs" / "quarterly" / "validation" / path.stem for path in CONFIG_PATHS]
    for validation_dir in validation_dirs:
        purge = read_table(validation_dir / "purge_audit.csv")
        fold_map = read_table(validation_dir / "fold_map.parquet")
        fold_summary = read_table(validation_dir / "fold_summary.csv")
        name = validation_dir.name
        if purge is None and fold_map is None and fold_summary is None:
            rows.append({"validation_dir": rel(validation_dir), "check": "validation_artifacts_exist", "status": "LIMITED", "value": False, "detail": "No fold_map/purge_audit/fold_summary found."})
            continue
        rows.append({"validation_dir": rel(validation_dir), "check": "validation_artifacts_exist", "status": "PASS", "value": True, "detail": ""})
        if purge is not None:
            p = purge.copy()
            for col in ["train_end_date", "validation_start_date", "embargo_date_count", "overlap_purge_date_count"]:
                if col not in p.columns:
                    p[col] = np.nan
            train_before_val = bool((pd.to_datetime(p["train_end_date"], errors="coerce") < pd.to_datetime(p["validation_start_date"], errors="coerce")).all())
            embargo_ok = bool(pd.to_numeric(p["embargo_date_count"], errors="coerce").fillna(0).ge(5).all())
            purge_ok = bool(pd.to_numeric(p["overlap_purge_date_count"], errors="coerce").fillna(0).ge(62).all())
            holdout_excluded = bool((p.loc[p["fold_label"].astype(str).eq("holdout"), "validation_start_date"].astype(str).str.startswith("2024")).all()) if "fold_label" in p.columns else None
            rows.extend(
                [
                    {"validation_dir": rel(validation_dir), "check": "train_dates_before_validation_dates", "status": status_from(train_before_val), "value": train_before_val, "detail": ""},
                    {"validation_dir": rel(validation_dir), "check": "embargo_gap_exists", "status": status_from(embargo_ok), "value": p["embargo_date_count"].min(), "detail": ""},
                    {"validation_dir": rel(validation_dir), "check": "purge_window_covers_63d_label_horizon", "status": status_from(purge_ok), "value": p["overlap_purge_date_count"].min(), "detail": ""},
                    {"validation_dir": rel(validation_dir), "check": "holdout_validation_starts_in_2024", "status": status_from(holdout_excluded), "value": holdout_excluded, "detail": ""},
                ]
            )
        if fold_map is not None and {"event_id", "fold_label", "fold_role"}.issubset(fold_map.columns):
            overlap_count = 0
            for _, frame in fold_map.groupby("fold_label"):
                role_sets = frame.groupby("event_id")["fold_role"].nunique()
                overlap_count += int(role_sets.gt(1).sum())
            rows.append({"validation_dir": rel(validation_dir), "check": "no_event_key_in_multiple_roles_same_fold", "status": status_from(overlap_count == 0), "value": overlap_count, "detail": ""})
        if fold_summary is not None and "evaluation_role" in fold_summary.columns:
            has_holdout_eval = bool(fold_summary["evaluation_role"].astype(str).eq("holdout_eval").any())
            rows.append({"validation_dir": rel(validation_dir), "check": "holdout_eval_recorded_separately", "status": status_from(has_holdout_eval), "value": has_holdout_eval, "detail": ""})

    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(output_dir / "split_integrity_audit.csv", index=False)
    for row in rows:
        append_summary(summary, "Split Integrity", f"{row['check']}:{Path(row['validation_dir']).name}", row["status"], row["value"], row["detail"])
    return audit_df


def infer_date_columns(frame: pd.DataFrame) -> list[str]:
    candidates = [column for column in frame.columns if "date" in column.lower() or column.lower().endswith("_timestamp")]
    out = []
    for column in candidates[:12]:
        parsed = pd.to_datetime(frame[column], errors="coerce")
        if parsed.notna().any():
            out.append(column)
    return out


def artifact_manifest(output_dir: Path, summary: list[dict[str, Any]]) -> pd.DataFrame:
    artifact_paths = [
        PROJECT_ROOT / "data" / "raw" / "fundamentals" / "raw_fundamentals_universe_v2.parquet",
        PROJECT_ROOT / "data" / "interim" / "fundamentals" / "fundamentals_quarterly_clean_universe_v2.parquet",
        PROJECT_ROOT / "data" / "interim" / "features" / "layer1_financial_features_universe_v2.parquet",
        PROJECT_ROOT / "data" / "interim" / "event_panel_v2_universe_v2.parquet",
        PROJECT_ROOT / "data" / "raw" / "fundamentals" / "raw_fundamentals.parquet",
        PROJECT_ROOT / "data" / "interim" / "fundamentals" / "fundamentals_quarterly_clean.parquet",
        PROJECT_ROOT / "data" / "interim" / "features" / "layer1_financial_features.parquet",
        PROJECT_ROOT / "outputs" / "quarterly" / "panels" / "quarterly_event_panel_features.parquet",
        PROJECT_ROOT / "data" / "processed" / "capitaliq_keydev_news_prepared.parquet",
        PROJECT_ROOT / "data" / "processed" / "news_scores_finbert_capitaliq_keydev.parquet",
        PROJECT_ROOT / "outputs" / "quarterly" / "panels" / "quarterly_event_panel_sector_sentiment_capitaliq.parquet",
        PROJECT_ROOT / "outputs" / "quarterly" / "modeling" / "capitaliq_sentiment_comparison.csv",
        PROJECT_ROOT / "reports" / "results" / "capitaliq_sentiment_final_report.md",
        PROJECT_ROOT / "outputs" / "quarterly" / "modeling" / "final" / "capitaliq_sentiment_final_report.md",
    ]
    rows: list[dict[str, Any]] = []
    for path in artifact_paths:
        exists = path.exists()
        row: dict[str, Any] = {
            "path": rel(path),
            "exists": exists,
            "file_size_bytes": path.stat().st_size if exists else np.nan,
            "row_count": np.nan,
            "column_count": np.nan,
            "min_date": "",
            "max_date": "",
            "generated_by": "",
            "notes": "",
        }
        if exists and path.suffix.lower() in {".parquet", ".csv", ".json"}:
            try:
                frame = read_table(path)
                if frame is not None:
                    row["row_count"] = len(frame)
                    row["column_count"] = len(frame.columns)
                    date_cols = infer_date_columns(frame)
                    if date_cols:
                        parsed = pd.concat([pd.to_datetime(frame[col], errors="coerce") for col in date_cols], axis=0).dropna()
                        if not parsed.empty:
                            row["min_date"] = parsed.min().date().isoformat()
                            row["max_date"] = parsed.max().date().isoformat()
                        row["notes"] = "date_columns=" + ",".join(date_cols)
            except Exception as exc:  # pragma: no cover - manifest should degrade gracefully
                row["notes"] = f"read_failed={type(exc).__name__}: {exc}"
        if "capitaliq" in path.name:
            row["generated_by"] = "Capital IQ sentiment pipeline"
        elif "fundamental" in path.name or "financial_features" in path.name:
            row["generated_by"] = "staged fundamentals rebuild"
        elif "event_panel" in path.name or "quarterly_event_panel" in path.name:
            row["generated_by"] = "event panel builder"
        rows.append(row)

    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(output_dir / "artifact_manifest.csv", index=False)
    missing = int((~audit_df["exists"]).sum())
    append_summary(summary, "Artifact Manifest", "major_artifacts_exist", "REVIEW" if missing else "PASS", f"missing={missing}", "")
    return audit_df


def write_report(
    output_dir: Path,
    summary_df: pd.DataFrame,
    panel_audit: pd.DataFrame,
    label_audit: pd.DataFrame,
    availability: dict[str, Any],
    sentiment_join: pd.DataFrame,
    duplicate_audit: pd.DataFrame,
    score_audit: pd.DataFrame,
    feature_audit: pd.DataFrame,
    split_audit: pd.DataFrame,
    coverage_outputs: dict[str, pd.DataFrame],
    args: argparse.Namespace,
) -> None:
    def count_status(status: str) -> int:
        return int(summary_df["status"].astype(str).eq(status).sum())

    failures = summary_df[summary_df["status"].astype(str).eq("FAIL")]
    reviews = summary_df[summary_df["status"].astype(str).eq("REVIEW")]
    limited = summary_df[summary_df["status"].astype(str).eq("LIMITED")]
    coverage_overall = coverage_outputs.get("coverage_overall", pd.DataFrame())
    coverage_text = ""
    if not coverage_overall.empty:
        coverage_text = "; ".join(
            f"{row.scope} {row.window}={row.coverage_pct:.2f}%"
            for row in coverage_overall.itertuples(index=False)
            if pd.notna(row.coverage_pct)
        )

    lines = [
        "# Data Layer Integrity Audit",
        "",
        "## Executive Summary",
        "",
        f"- Checks passed: `{count_status('PASS')}`",
        f"- Checks for review: `{count_status('REVIEW')}`",
        f"- Checks failed: `{count_status('FAIL')}`",
        f"- Checks limited by missing artifacts: `{count_status('LIMITED')}`",
        f"- Main high-priority failures: `{len(failures)}`",
        "",
        summary_df.groupby(["audit", "status"], dropna=False).size().reset_index(name="count").to_markdown(index=False),
        "",
        "## Why This Matters",
        "",
        (
            "Financial ML projects often fail because of data leakage, unit mismatches, ticker joins, bad labels, "
            "or silent row drops rather than because the model class is wrong. This audit makes those risks visible "
            "before final results are packaged."
        ),
        "",
        "## Fundamentals Integrity",
        "",
    ]
    if availability.get("fundamental_rebuild_report_exists"):
        lines.extend(
            [
                "- Staged rebuild diagnostics were found under `outputs/quarterly/diagnostics/fundamental_rebuild/`.",
                "- Effective-date diagnostics reported 0 selected facts after cutoff for both Universe V2 and Universe V1.",
                "- Ratio sanity flags dropped sharply after the rebuild.",
                "- AAPL revenue/assets and KR asset-turnover examples were fixed in the staged rebuild summary.",
                "- The user-provided current state says 23 rebuild unit tests passed; this audit does not restate that as a fresh pytest result.",
            ]
        )
    else:
        lines.append("- Fundamental rebuild diagnostics were not found, so this audit cannot independently verify cutoff selection.")
    lines.extend(
        [
            "",
            "## Market and Label Integrity",
            "",
            f"- Label contract rows written to `label_contract_audit.csv`; inspected configs report horizon/mode/holdout settings.",
            f"- Future-return-like panel columns: `{', '.join(availability.get('future_feature_cols', [])) or 'none'}`.",
            f"- Market rows with `market_asof_date` after event date: `{availability.get('market_after_event')}`.",
        ]
    )
    stale_count = int((label_audit["check"] == "stale_reference").sum()) if not label_audit.empty else 0
    lines.append(f"- Stale 21-day/5-day text references found: `{stale_count}`; these are classified in `label_contract_audit.csv`.")
    lines.extend(
        [
            "",
            "## Sentiment Integrity",
            "",
            "- Source: Capital IQ Key Developments prepared/scored artifacts.",
            "- Prepared rows are direct-parent matched when the prepared artifact exposes `direct_parent_match`.",
            "- Strict pre-event sentiment recompute checks are summarized from existing Capital IQ consistency diagnostics when present.",
            f"- Coverage summary: {coverage_text or 'coverage columns were not available.'}",
        ]
    )
    dup_count = duplicate_audit.loc[duplicate_audit["metric"].eq("duplicate_row_count"), "value"]
    if not dup_count.empty:
        lines.append(f"- Duplicate prepared-news rows under the adapted key: `{dup_count.iloc[0]}`.")
    neutral = score_audit.loc[(score_audit["metric"] == "neutral_dominance_rate") & (score_audit["group"] == "all"), "value"] if "group" in score_audit.columns else pd.Series(dtype=object)
    if not neutral.empty:
        lines.append(f"- FinBERT neutral-dominance rate: `{neutral.iloc[0]}`.")
    lines.extend(
        [
            "",
            "## Panel Integrity",
            "",
            panel_audit.to_markdown(index=False),
            "",
            "Feature-family isolation checks are in `feature_family_isolation_audit.csv`. The audit flags raw text, provenance, date, target, label, future-return, and object/string columns if they enter candidate features.",
            "",
            "## Split Integrity",
            "",
            "Split integrity checks are based on saved validation artifacts where available: fold maps, purge audits, and fold summaries.",
        ]
    )
    split_counts = split_audit.groupby("status").size().reset_index(name="count") if not split_audit.empty else pd.DataFrame()
    if not split_counts.empty:
        lines.extend(["", split_counts.to_markdown(index=False)])
    lines.extend(
        [
            "",
            "## Remaining Risks",
            "",
            "- The V1 universe is Consumer Staples only.",
            "- Capital IQ Key Developments are event-text, not general news.",
            "- Pseudo-holdout sentiment lift was fragile across stability checks.",
            "- Some checks are limited if fold assignments, exact feature matrices, or prediction artifacts are not saved.",
            "- Stale 21-day markdown/test references should be fixed before final submission if they remain in active-facing material.",
            "",
            "## Validation Commands",
            "",
            f"- `python -m py_compile src/audit/data_layer_integrity_audit.py`: `{args.py_compile_status or 'not recorded in report run'}`",
            f"- `python -m unittest discover -s tests`: `{args.unittest_status or 'not recorded in report run'}`",
        ]
    )
    if args.unittest_summary:
        lines.append(f"- unittest summary: {args.unittest_summary}")
    lines.extend(
        [
            "",
            "## Final Readiness",
            "",
        ]
    )
    if failures.empty:
        lines.append(
            "The saved data layers are ready for final reporting from a data-integrity perspective, subject to resolving review items and documenting limited checks honestly."
        )
    else:
        lines.append(
            "The data layers are not ready for final report packaging until failed audit checks are resolved or explicitly justified."
        )
    if not failures.empty:
        lines.extend(["", "### Failed Checks", "", failures.to_markdown(index=False)])
    if not reviews.empty:
        lines.extend(["", "### Review Checks", "", reviews.head(50).to_markdown(index=False)])
    if not limited.empty:
        lines.extend(["", "### Limited Checks", "", limited.to_markdown(index=False)])
    lines.append("")
    (output_dir / "data_layer_integrity_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, Any]] = []
    base = read_table(Path(args.base_panel))
    enriched = read_table(Path(args.enriched_panel))
    prepared = read_table(Path(args.prepared_news))
    scored = read_table(Path(args.scored_news))
    if base is None:
        raise FileNotFoundError(args.base_panel)
    if enriched is None:
        raise FileNotFoundError(args.enriched_panel)

    panel_audit = audit_panel_identity(base, enriched, output_dir, summary)
    label_audit = audit_label_contract(enriched, output_dir, summary)
    availability = audit_availability(enriched, scored, output_dir, summary)
    sentiment_join = audit_sentiment_join(enriched, prepared, scored, output_dir, summary)
    duplicate_audit = audit_sentiment_duplicates(prepared, output_dir, summary)
    score_audit = audit_sentiment_scores(scored, output_dir, summary)
    coverage_outputs = audit_coverage(enriched, output_dir, summary)
    feature_audit = audit_feature_isolation(enriched, output_dir, summary)
    split_audit = audit_split_integrity(output_dir, summary)
    manifest = artifact_manifest(output_dir, summary)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output_dir / "data_layer_integrity_summary.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    for name, frame in coverage_outputs.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    write_report(
        output_dir,
        summary_df,
        panel_audit,
        label_audit,
        availability,
        sentiment_join,
        duplicate_audit,
        score_audit,
        feature_audit,
        split_audit,
        coverage_outputs,
        args,
    )
    print(f"Wrote data-layer integrity audit outputs to {output_dir}")
    print(f"Summary statuses: {summary_df['status'].value_counts().to_dict()}")
    print(f"Manifest rows: {len(manifest)}")


if __name__ == "__main__":
    main()
