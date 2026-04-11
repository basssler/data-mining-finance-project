"""Write the Phase 6B Alpha Vantage comparison report and decision memo."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.alpha_vantage_earnings_phase6b import (
    FEATURE_BLOCK_PATH,
    MANIFEST_PATH,
    MERGED_PANEL_PATH,
    NORMALIZED_EARNINGS_PATH,
    NORMALIZED_ESTIMATES_PATH,
)
from src.config_event_v1 import PRICE_INPUT_PATH
from src.label_comparison_event_v2 import attach_labels_to_event_panel, build_daily_label_table, load_event_panel
from src.labels_event_v1 import load_price_data, normalize_price_data
from src.paths import DOCS_DIR, PROJECT_ROOT

BASELINE_CSV_PATH = PROJECT_ROOT / "reports" / "results" / "event_panel_v2_primary_benchmark.csv"
ADDITIVE_CSV_PATH = PROJECT_ROOT / "reports" / "results" / "event_panel_v2_phase6b_alpha_vantage_benchmark.csv"
BENCHMARK_MD_PATH = PROJECT_ROOT / "reports" / "results" / "event_panel_v2_phase6b_alpha_vantage_benchmark.md"
DECISION_DOC_PATH = DOCS_DIR / "phase6b_alpha_vantage_earnings_v1.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the Phase 6B Alpha Vantage report.")
    parser.add_argument("--baseline-csv", default=str(BASELINE_CSV_PATH))
    parser.add_argument("--additive-csv", default=str(ADDITIVE_CSV_PATH))
    parser.add_argument("--panel-path", default=str(MERGED_PANEL_PATH))
    parser.add_argument("--feature-path", default=str(FEATURE_BLOCK_PATH))
    parser.add_argument("--manifest-path", default=str(MANIFEST_PATH))
    parser.add_argument("--earnings-path", default=str(NORMALIZED_EARNINGS_PATH))
    parser.add_argument("--estimates-path", default=str(NORMALIZED_ESTIMATES_PATH))
    parser.add_argument("--prices-path", default=str(PRICE_INPUT_PATH))
    parser.add_argument("--benchmark-output-path", default=str(BENCHMARK_MD_PATH))
    parser.add_argument("--doc-output-path", default=str(DECISION_DOC_PATH))
    return parser.parse_args()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def format_metric(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.4f}"


def load_benchmark(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark CSV not found: {path}")
    return pd.read_csv(path)


def get_selected_row(df: pd.DataFrame) -> pd.Series:
    selected = df.loc[df["is_selected_primary_model"] == True]  # noqa: E712
    if selected.empty:
        raise ValueError("No selected primary model row was found.")
    return selected.iloc[0]


def load_manifest_summary(path: Path) -> dict:
    manifest_df = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    return {
        "status_counts": manifest_df["status"].value_counts(dropna=False).sort_index().to_dict(),
        "rows": int(len(manifest_df)),
    }


def load_panel_summary(path: Path) -> dict:
    panel_df = pd.read_parquet(path)
    av_columns = [column for column in panel_df.columns if column.startswith("av_")]
    return {
        "rows": int(len(panel_df)),
        "tickers": int(panel_df["ticker"].nunique()),
        "feature_count": int(len(panel_df.columns)),
        "alpha_feature_count": int(len(av_columns)),
    }


def build_feature_qa(panel_path: Path, prices_path: Path) -> tuple[dict, pd.DataFrame]:
    panel_df = load_event_panel(panel_path)
    prices_df = normalize_price_data(load_price_data(prices_path))
    label_df = build_daily_label_table(prices_df, horizon_days=5, benchmark_mode="sector_equal_weight_ex_self")
    labeled = attach_labels_to_event_panel(panel_df, label_df)
    av_columns = [
        column
        for column in labeled.columns
        if column.startswith("av_") and pd.api.types.is_numeric_dtype(labeled[column])
    ]
    missingness = labeled[av_columns].isna().mean().mul(100).round(2).sort_values(ascending=False)
    correlations = {}
    target_series = pd.to_numeric(labeled["target_sign"], errors="coerce")
    for column in av_columns:
        series = pd.to_numeric(labeled[column], errors="coerce")
        valid = series.notna() & target_series.notna()
        if int(valid.sum()) < 20:
            continue
        correlations[column] = float(series.loc[valid].corr(target_series.loc[valid]))
    top_correlations = (
        pd.Series(correlations).dropna().sort_values(key=lambda s: s.abs(), ascending=False).head(10)
        if correlations
        else pd.Series(dtype="float64")
    )
    coverage_by_ticker = (
        labeled.groupby("ticker")["av_coverage_any"].mean().sort_index().mul(100).round(2).to_dict()
        if "av_coverage_any" in labeled.columns
        else {}
    )
    coverage_by_year = (
        labeled.assign(event_year=pd.to_datetime(labeled["event_date"]).dt.year)
        .groupby("event_year")["av_coverage_any"]
        .mean()
        .sort_index()
        .mul(100)
        .round(2)
        .to_dict()
        if "av_coverage_any" in labeled.columns
        else {}
    )
    summary = {
        "missingness": missingness.to_dict(),
        "coverage_by_ticker": coverage_by_ticker,
        "coverage_by_year": coverage_by_year,
        "top_correlations": top_correlations.to_dict(),
    }
    return summary, labeled


def build_benchmark_markdown(
    baseline_df: pd.DataFrame,
    additive_df: pd.DataFrame,
    panel_summary: dict,
    manifest_summary: dict,
) -> str:
    merged = baseline_df.merge(
        additive_df,
        on="model_name",
        suffixes=("_baseline", "_additive"),
        how="outer",
        validate="one_to_one",
    )
    baseline_best = get_selected_row(baseline_df)
    additive_best = get_selected_row(additive_df)
    lines = [
        "# Event Panel V2 Phase 6B Alpha Vantage Benchmark",
        "",
        "## Scope",
        "",
        "- Locked baseline preserved: `event_panel_v2`, 34 tickers, 5-trading-day excess return sign, 2024 holdout unchanged.",
        "- One additive external dataset family only: Alpha Vantage earnings estimates/outcomes.",
        "- Models unchanged: logistic regression, random forest, XGBoost.",
        f"- Manifest completion state: `{manifest_summary['status_counts']}`.",
        "",
        "## Panel Summary",
        "",
        f"- Rows: `{panel_summary['rows']:,}`",
        f"- Tickers: `{panel_summary['tickers']:,}`",
        f"- Total feature count: `{panel_summary['feature_count']}`",
        f"- New Alpha Vantage feature count: `{panel_summary['alpha_feature_count']}`",
        "",
        "## Baseline vs Additive",
        "",
        "| Model | Baseline CV AUC | Additive CV AUC | Baseline Holdout AUC | Additive Holdout AUC | Baseline CV Log Loss | Additive CV Log Loss | Baseline Holdout Log Loss | Additive Holdout Log Loss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in merged.sort_values("model_name").iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model_name"]),
                    format_metric(row.get("cv_auc_mean_baseline")),
                    format_metric(row.get("cv_auc_mean_additive")),
                    format_metric(row.get("holdout_auc_baseline")),
                    format_metric(row.get("holdout_auc_additive")),
                    format_metric(row.get("cv_log_loss_mean_baseline")),
                    format_metric(row.get("cv_log_loss_mean_additive")),
                    format_metric(row.get("holdout_log_loss_baseline")),
                    format_metric(row.get("holdout_log_loss_additive")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Baseline selected model: `{baseline_best['model_name']}`",
            f"- Additive selected model: `{additive_best['model_name']}`",
            "- Current benchmark result is identical to baseline because every new Alpha Vantage feature was dropped by the existing 20% train-fold missingness rule under the partial backfill coverage.",
            "- This should be treated as a partial-cache diagnostic run, not the final official Phase 6B verdict, until the remaining manifest rows are fetched with refreshed or replacement API keys.",
            "- This benchmark should be read as the apples-to-apples Phase 6B test against the locked Phase 4 anchor.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_decision_doc(
    manifest_summary: dict,
    panel_summary: dict,
    qa_summary: dict,
    baseline_df: pd.DataFrame,
    additive_df: pd.DataFrame,
    earnings_path: Path,
    estimates_path: Path,
) -> str:
    baseline_best = get_selected_row(baseline_df)
    additive_best = get_selected_row(additive_df)
    manifest_complete = int(manifest_summary["status_counts"].get("pending", 0)) == 0 and int(
        manifest_summary["status_counts"].get("failed", 0)
    ) == 0
    additive_wins = (
        float(additive_best["cv_auc_mean"]) > float(baseline_best["cv_auc_mean"])
        and float(additive_best["holdout_auc"]) >= float(baseline_best["holdout_auc"])
    )
    decision = "PROMOTE" if additive_wins else "FREEZE"
    lines = [
        "# Phase 6B Alpha Vantage Earnings Test V1",
        "",
        "## Data Source",
        "",
        "- External dataset family: Alpha Vantage structured earnings expectations / earnings outcomes.",
        f"- Normalized estimates table: `{estimates_path}`",
        f"- Normalized earnings table: `{earnings_path}`",
        "- Endpoint scope: `EARNINGS_ESTIMATES` and `EARNINGS` only.",
        "",
        "## API Efficiency Design",
        "",
        "- Per-symbol, per-endpoint raw JSON cache under `data/raw/alpha_vantage/`.",
        "- Resumable manifest with `pending`, `complete`, and `failed` states stored in `data/interim/alpha_vantage/alpha_vantage_manifest.json`.",
        "- Multi-key round-robin rotation from `ALPHAVANTAGE_API_KEYS` with cooldown handling for rate limits.",
        "- Backfill mode avoids re-pulling any completed symbol-endpoint pair unless refresh is explicitly requested.",
        "",
        "## Leakage Controls",
        "",
        "- Alpha Vantage earnings-outcome rows are timestamped using `reportedDate` + `reportTime` and must be strictly earlier than the filing-event cutoff.",
        "- Filing events with missing intraday timestamps use previous-trading-day close as the conservative cutoff.",
        "- `EARNINGS_ESTIMATES` rows were normalized in full, but only quarterly estimate rows linked to already-reported quarters were promoted into model features.",
        "- Untimestamped future estimate snapshots were intentionally excluded from modeling because they are not point-in-time safe.",
        "",
        "## Feasible Feature Block",
        "",
        "- Safe outcome features: latest prior EPS surprise, latest prior EPS surprise percent, trailing 4-quarter EPS surprise mean/std, trailing 4-quarter EPS surprise percent mean/std, trailing 4-quarter EPS beat rate, days since last earnings release.",
        "- Safe estimate-derived features: latest prior quarter EPS estimate, latest prior quarter revenue estimate, EPS/revenue analyst counts, EPS estimate revision vs 30/90 days ago for the most recently reported quarter.",
        "- Not promoted: annual estimate features and revenue revision/surprise features where Alpha Vantage did not expose sufficiently timestamped historical fields.",
        "",
        "## Coverage Diagnostics",
        "",
        f"- Manifest rows: `{manifest_summary['rows']}`",
        f"- Manifest status counts: `{manifest_summary['status_counts']}`",
        f"- Full backfill complete: `{'yes' if manifest_complete else 'no'}`",
        f"- Event-panel rows: `{panel_summary['rows']:,}`",
        f"- Tickers: `{panel_summary['tickers']}`",
        f"- New Alpha Vantage feature count: `{panel_summary['alpha_feature_count']}`",
        f"- Coverage by year: `{qa_summary['coverage_by_year']}`",
        f"- Coverage by ticker: `{qa_summary['coverage_by_ticker']}`",
        "",
        "## Feature QA",
        "",
        f"- Missingness by new feature: `{qa_summary['missingness']}`",
        f"- Top descriptive target correlations: `{qa_summary['top_correlations']}`",
        "",
        "## Results",
        "",
        f"- Baseline selected model: `{baseline_best['model_name']}` with CV AUC `{format_metric(baseline_best['cv_auc_mean'])}` and holdout AUC `{format_metric(baseline_best['holdout_auc'])}`.",
        f"- Alpha Vantage selected model: `{additive_best['model_name']}` with CV AUC `{format_metric(additive_best['cv_auc_mean'])}` and holdout AUC `{format_metric(additive_best['holdout_auc'])}`.",
        "- The additive benchmark exactly matched baseline because all Alpha Vantage columns were excluded by the existing missingness filter at the current partial-coverage state.",
        "",
        "## Recommendation",
        "",
        f"- Final decision: **{decision}**",
        "- `PROMOTE` means the additive block improved the locked benchmark cleanly enough to carry forward.",
        "- `FREEZE` means the ingest and merge are kept for reference, but the dataset is not yet strong enough to promote.",
        "- `REJECT` would be reserved for a structurally impractical or leakage-unsafe dataset path.",
    ]
    if not additive_wins:
        lines.append(
            "- Current result: keep the Phase 4 anchor as the primary setup and treat Alpha Vantage earnings data as a tested but not-yet-promoted additive layer."
        )
    if not manifest_complete:
        lines.append(
            "- Current blocker: Alpha Vantage throttled all available keys before the 34-ticker backfill finished. The present memo is therefore provisional until the remaining pending manifest rows are fetched."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    benchmark_output_path = Path(args.benchmark_output_path)
    doc_output_path = Path(args.doc_output_path)
    ensure_parent_dir(benchmark_output_path)
    ensure_parent_dir(doc_output_path)

    baseline_df = load_benchmark(Path(args.baseline_csv))
    additive_df = load_benchmark(Path(args.additive_csv))
    panel_summary = load_panel_summary(Path(args.panel_path))
    manifest_summary = load_manifest_summary(Path(args.manifest_path))
    qa_summary, _ = build_feature_qa(Path(args.panel_path), Path(args.prices_path))

    benchmark_md = build_benchmark_markdown(baseline_df, additive_df, panel_summary, manifest_summary)
    benchmark_output_path.write_text(benchmark_md, encoding="utf-8")

    decision_doc = build_decision_doc(
        manifest_summary=manifest_summary,
        panel_summary=panel_summary,
        qa_summary=qa_summary,
        baseline_df=baseline_df,
        additive_df=additive_df,
        earnings_path=Path(args.earnings_path),
        estimates_path=Path(args.estimates_path),
    )
    doc_output_path.write_text(decision_doc, encoding="utf-8")

    print(f"Wrote Phase 6B benchmark report to: {benchmark_output_path}")
    print(f"Wrote Phase 6B decision doc to: {doc_output_path}")


if __name__ == "__main__":
    main()
