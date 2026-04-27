"""Validate Capital IQ KeyDev FinBERT sentiment panel consistency."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.paths import PROJECT_ROOT, QUARTERLY_OUTPUTS_DIAGNOSTICS_DIR, QUARTERLY_OUTPUTS_PANELS_DIR


CANONICAL_FILE_RE = re.compile(r"^[A-Z0-9-]+_20[0-9]{2}\.csv$")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run consistency checks for the Capital IQ KeyDev FinBERT panel.")
    parser.add_argument("--raw-dir", default=str(PROJECT_ROOT / "data" / "raw" / "capitaliq_keydev2"))
    parser.add_argument("--prepared", default=str(PROJECT_ROOT / "data" / "processed" / "capitaliq_keydev_news_prepared.parquet"))
    parser.add_argument("--scored", default=str(PROJECT_ROOT / "data" / "processed" / "news_scores_finbert_capitaliq_keydev.parquet"))
    parser.add_argument("--base-panel", default=str(QUARTERLY_OUTPUTS_PANELS_DIR / "quarterly_event_panel_features.parquet"))
    parser.add_argument(
        "--enriched-panel",
        default=str(QUARTERLY_OUTPUTS_PANELS_DIR / "quarterly_event_panel_sector_sentiment_capitaliq.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(QUARTERLY_OUTPUTS_DIAGNOSTICS_DIR / "capitaliq_sentiment_consistency"),
    )
    return parser.parse_args()


def _read_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def _date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def _metric(rows: list[dict[str, object]], check: str, status: str, value: object, detail: str = "") -> None:
    rows.append({"check": check, "status": status, "value": value, "detail": detail})


def _status(condition: bool) -> str:
    return "PASS" if condition else "FAIL"


def _coverage_by_group(panel: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    prepared = panel.assign(event_year=_date(panel["event_date"]).dt.year)
    existing = [column for column in group_cols if column in prepared.columns]
    if not existing:
        return pd.DataFrame()
    grouped = (
        prepared
        .groupby(existing, dropna=False)
        .agg(
            rows=("event_id", "size"),
            coverage_7d=("has_news_7d", "mean"),
            coverage_30d=("has_news_30d", "mean"),
            coverage_63d=("has_news_63d", "mean"),
            avg_news_count_30d=("news_count_30d", "mean"),
            avg_news_count_63d=("news_count_63d", "mean"),
        )
        .reset_index()
    )
    return grouped.sort_values(existing).reset_index(drop=True)


def _recompute_ticker_counts(panel: pd.DataFrame, scored: pd.DataFrame, window: int) -> pd.Series:
    news = scored[["ticker", "date"]].copy()
    news["ticker"] = news["ticker"].astype("string").str.upper().str.strip()
    news["_news_date"] = _date(news["date"])
    news = news.dropna(subset=["ticker", "_news_date"])
    groups = {ticker: frame["_news_date"].sort_values().to_numpy(dtype="datetime64[ns]") for ticker, frame in news.groupby("ticker")}

    counts: list[int] = []
    for row in panel[["ticker", "event_date"]].itertuples(index=False):
        ticker = str(row.ticker).upper().strip()
        event_date = pd.to_datetime(row.event_date, errors="coerce")
        if pd.isna(event_date):
            counts.append(0)
            continue
        dates = groups.get(ticker)
        if dates is None:
            counts.append(0)
            continue
        end = np.datetime64(event_date.normalize())
        start = np.datetime64((event_date - pd.Timedelta(days=window)).normalize())
        counts.append(int(((dates >= start) & (dates < end)).sum()))
    return pd.Series(counts, index=panel.index)


def run_checks(args: argparse.Namespace) -> dict[str, Path]:
    raw_dir = Path(args.raw_dir)
    prepared_path = Path(args.prepared)
    scored_path = Path(args.scored)
    base_panel_path = Path(args.base_panel)
    enriched_panel_path = Path(args.enriched_panel)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared = _read_frame(prepared_path)
    scored = _read_frame(scored_path)
    base_panel = _read_frame(base_panel_path)
    enriched = _read_frame(enriched_panel_path)
    rows: list[dict[str, object]] = []

    raw_files = sorted(path.name for path in raw_dir.glob("*.csv"))
    canonical_files = [name for name in raw_files if CANONICAL_FILE_RE.match(name)]
    ignored_files = [name for name in raw_files if not CANONICAL_FILE_RE.match(name)]
    _metric(rows, "raw_canonical_file_count", "PASS", len(canonical_files))
    _metric(
        rows,
        "raw_ignored_files",
        _status(set(ignored_files) == {"PG_20241.csv", "PG_2024_TEXTPARSE_TEST.csv"}),
        ",".join(ignored_files),
    )

    _metric(rows, "prepared_row_count", _status(len(prepared) == 24347), len(prepared))
    _metric(rows, "scored_row_count_matches_prepared", _status(len(scored) == len(prepared)), len(scored))
    _metric(rows, "prepared_direct_parent_match_all_true", _status(bool(prepared["direct_parent_match"].all())), int(prepared["direct_parent_match"].sum()))
    noncanonical_prepared = prepared.loc[~prepared["source_file"].astype("string").isin(canonical_files)]
    _metric(rows, "prepared_uses_only_canonical_files", _status(noncanonical_prepared.empty), len(noncanonical_prepared))

    merge_cols = ["ticker", "date", "headline"]
    prepared_key = prepared[merge_cols].copy()
    scored_key = scored[merge_cols].copy()
    matched = scored_key.merge(prepared_key.drop_duplicates(), on=merge_cols, how="left", indicator=True)
    _metric(rows, "scored_rows_match_prepared_keys", _status((matched["_merge"] == "both").all()), int((matched["_merge"] == "both").sum()))
    _metric(rows, "scored_text_id_unique", _status(scored["text_id"].is_unique if "text_id" in scored.columns else False), scored.get("text_id", pd.Series(dtype=object)).nunique())

    prob_cols = ["finbert_pos", "finbert_neu", "finbert_neg"]
    probs = scored[prob_cols].apply(pd.to_numeric, errors="coerce")
    in_range = probs.ge(0).all(axis=1) & probs.le(1).all(axis=1)
    prob_sum = probs.sum(axis=1)
    confidence = pd.to_numeric(scored["confidence"], errors="coerce")
    finbert_score = pd.to_numeric(scored["finbert_score"], errors="coerce")
    _metric(rows, "finbert_probabilities_in_range", _status(bool(in_range.all())), int(in_range.sum()))
    _metric(rows, "finbert_probability_sum_close_to_one", _status(bool((prob_sum.sub(1).abs() < 1e-4).all())), float(prob_sum.sub(1).abs().max()))
    _metric(rows, "finbert_score_equals_pos_minus_neg", _status(bool((finbert_score.sub(probs["finbert_pos"] - probs["finbert_neg"]).abs() < 1e-6).all())), float(finbert_score.sub(probs["finbert_pos"] - probs["finbert_neg"]).abs().max()))
    _metric(rows, "confidence_equals_max_probability", _status(bool((confidence.sub(probs.max(axis=1)).abs() < 1e-6).all())), float(confidence.sub(probs.max(axis=1)).abs().max()))

    _metric(rows, "enriched_panel_row_count_matches_base", _status(len(enriched) == len(base_panel)), f"{len(base_panel)}->{len(enriched)}")
    base_events = set(base_panel["event_id"].astype(str))
    enriched_events = set(enriched["event_id"].astype(str))
    _metric(rows, "enriched_panel_event_ids_match_base", _status(base_events == enriched_events), f"missing={len(base_events - enriched_events)}, extra={len(enriched_events - base_events)}")
    _metric(rows, "enriched_event_id_unique", _status(enriched["event_id"].is_unique), int(enriched["event_id"].nunique()))
    duplicate_source_keys = enriched.duplicated(subset=["ticker", "event_type", "source_file_id"]).sum()
    _metric(rows, "no_duplicate_ticker_event_source_rows", _status(duplicate_source_keys == 0), int(duplicate_source_keys))

    added_sentiment_cols = [column for column in SENTIMENT_COLUMNS if column in enriched.columns and column not in base_panel.columns]
    _metric(rows, "expected_sentiment_columns_added", _status(len(added_sentiment_cols) == len(SENTIMENT_COLUMNS)), len(added_sentiment_cols))

    for window in (7, 30, 63):
        expected = _recompute_ticker_counts(enriched, scored, window)
        actual = pd.to_numeric(enriched[f"news_count_{window}d"], errors="coerce").fillna(0).astype(int)
        mismatches = int((expected != actual).sum())
        _metric(rows, f"news_count_{window}d_matches_strict_pre_event_recompute", _status(mismatches == 0), mismatches)
        has_actual = pd.to_numeric(enriched[f"has_news_{window}d"], errors="coerce").fillna(0).astype(int)
        has_mismatches = int(((expected > 0).astype(int) != has_actual).sum())
        _metric(rows, f"has_news_{window}d_matches_recompute", _status(has_mismatches == 0), has_mismatches)

    for column in ["sent_mean_7d", "sent_mean_30d", "sent_mean_63d", "sector_adj_sent_30d", "sector_adj_sent_63d"]:
        values = pd.to_numeric(enriched[column], errors="coerce")
        bad = int((values.abs() > 2).sum())
        _metric(rows, f"{column}_within_expected_score_range", _status(bad == 0), bad)

    event_year = _date(enriched["event_date"]).dt.year
    holdout = enriched.loc[event_year == 2024].copy()
    _metric(rows, "holdout_2024_rows", _status(len(holdout) > 0), len(holdout))
    for window in (30, 63):
        coverage = float(pd.to_numeric(holdout[f"has_news_{window}d"], errors="coerce").fillna(0).mean()) if len(holdout) else np.nan
        _metric(rows, f"holdout_2024_{window}d_coverage", _status(pd.notna(coverage) and coverage >= 0.80), coverage)

    summary = pd.DataFrame(rows)
    score_distribution = scored.assign(
        probability_sum=prob_sum,
        date=_date(scored["date"]),
        year=_date(scored["date"]).dt.year,
    ).agg(
        rows=("ticker", "size"),
        tickers=("ticker", "nunique"),
        min_date=("date", "min"),
        max_date=("date", "max"),
        pos_mean=("finbert_pos", "mean"),
        neu_mean=("finbert_neu", "mean"),
        neg_mean=("finbert_neg", "mean"),
        score_mean=("finbert_score", "mean"),
        score_std=("finbert_score", "std"),
        confidence_mean=("confidence", "mean"),
    )
    score_distribution = score_distribution.reset_index().rename(columns={"index": "metric", 0: "value"})

    by_year = _coverage_by_group(enriched, ["event_year"])
    by_ticker = _coverage_by_group(enriched, ["ticker"])
    by_sector = _coverage_by_group(enriched, ["sector"])

    paths = {
        "summary": output_dir / "capitaliq_sentiment_consistency_summary.csv",
        "score_distribution": output_dir / "capitaliq_finbert_score_distribution.csv",
        "coverage_by_year": output_dir / "capitaliq_sentiment_coverage_by_year.csv",
        "coverage_by_ticker": output_dir / "capitaliq_sentiment_coverage_by_ticker.csv",
        "coverage_by_sector": output_dir / "capitaliq_sentiment_coverage_by_sector.csv",
        "markdown": output_dir / "capitaliq_sentiment_consistency_check.md",
    }
    summary.to_csv(paths["summary"], index=False)
    score_distribution.to_csv(paths["score_distribution"], index=False)
    by_year.to_csv(paths["coverage_by_year"], index=False)
    by_ticker.to_csv(paths["coverage_by_ticker"], index=False)
    by_sector.to_csv(paths["coverage_by_sector"], index=False)

    failures = summary.loc[summary["status"].eq("FAIL")]
    lines = [
        "# Capital IQ Sentiment Consistency Check",
        "",
        f"- Prepared rows: `{len(prepared):,}`",
        f"- Scored rows: `{len(scored):,}`",
        f"- Base panel rows: `{len(base_panel):,}`",
        f"- Enriched panel rows: `{len(enriched):,}`",
        f"- Failed checks: `{len(failures):,}`",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Coverage By Year",
        "",
        by_year.to_markdown(index=False),
    ]
    if not failures.empty:
        lines.extend(["", "## Failures", "", failures.to_markdown(index=False)])
    paths["markdown"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return paths


def main() -> None:
    paths = run_checks(parse_args())
    print("Capital IQ sentiment consistency check complete")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
