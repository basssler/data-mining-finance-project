"""Write the Phase 6 SEC filing sentiment comparison report."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.paths import DOCS_DIR, INTERIM_DATA_DIR, PROJECT_ROOT

PHASE4_BENCHMARK_CSV = PROJECT_ROOT / "reports" / "results" / "event_panel_v2_primary_benchmark.csv"
PHASE6_BENCHMARK_CSV = PROJECT_ROOT / "reports" / "results" / "event_panel_v2_sec_sentiment_v1_benchmark.csv"
PHASE4_PANEL_PATH = INTERIM_DATA_DIR / "event_panel_v2.parquet"
PHASE6_PANEL_PATH = INTERIM_DATA_DIR / "event_panel_v2_sec_sentiment_v1.parquet"
PHASE6_REPORT_MD = PROJECT_ROOT / "reports" / "results" / "event_panel_v2_sec_sentiment_v1_benchmark.md"
PHASE6_DECISION_DOC = DOCS_DIR / "phase6_sec_sentiment_test_v1.md"
USED_SENTIMENT_ARTIFACT = INTERIM_DATA_DIR / "features" / "layer3_sec_sentiment_features.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the Phase 6 SEC sentiment comparison report.")
    parser.add_argument("--phase4-csv", default=str(PHASE4_BENCHMARK_CSV))
    parser.add_argument("--phase6-csv", default=str(PHASE6_BENCHMARK_CSV))
    parser.add_argument("--phase4-panel", default=str(PHASE4_PANEL_PATH))
    parser.add_argument("--phase6-panel", default=str(PHASE6_PANEL_PATH))
    parser.add_argument("--report-md", default=str(PHASE6_REPORT_MD))
    parser.add_argument("--decision-doc", default=str(PHASE6_DECISION_DOC))
    return parser.parse_args()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def format_metric(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.4f}"


def load_benchmark(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark CSV was not found: {path}")
    return pd.read_csv(path)


def load_panel_stats(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Panel parquet was not found: {path}")
    panel_df = pd.read_parquet(path)
    sentiment_columns = [column for column in panel_df.columns if "sentiment" in column or column.startswith("sec_")]
    return {
        "rows": int(len(panel_df)),
        "tickers": int(panel_df["ticker"].nunique()),
        "feature_count": int(len(panel_df.columns)),
        "sentiment_columns": sentiment_columns,
    }


def get_selected_row(df: pd.DataFrame) -> pd.Series:
    selected = df.loc[df["is_selected_primary_model"] == True]  # noqa: E712
    if selected.empty:
        raise ValueError("No selected primary model row was found.")
    return selected.iloc[0]


def build_model_table(phase4_df: pd.DataFrame, phase6_df: pd.DataFrame) -> list[str]:
    merged = phase4_df.merge(
        phase6_df,
        on="model_name",
        suffixes=("_phase4", "_phase6"),
        how="outer",
        validate="one_to_one",
    )
    lines = [
        "| Model | Phase 4 CV AUC | Phase 6 CV AUC | Phase 4 CV Log Loss | Phase 6 CV Log Loss | Phase 4 Holdout AUC | Phase 6 Holdout AUC | Phase 4 Holdout Log Loss | Phase 6 Holdout Log Loss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in merged.sort_values("model_name").iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model_name"]),
                    format_metric(row.get("cv_auc_mean_phase4")),
                    format_metric(row.get("cv_auc_mean_phase6")),
                    format_metric(row.get("cv_log_loss_mean_phase4")),
                    format_metric(row.get("cv_log_loss_mean_phase6")),
                    format_metric(row.get("holdout_auc_phase4")),
                    format_metric(row.get("holdout_auc_phase6")),
                    format_metric(row.get("holdout_log_loss_phase4")),
                    format_metric(row.get("holdout_log_loss_phase6")),
                ]
            )
            + " |"
        )
    return lines


def build_report_md(phase4_df: pd.DataFrame, phase6_df: pd.DataFrame, phase4_stats: dict, phase6_stats: dict) -> str:
    phase4_best = get_selected_row(phase4_df)
    phase6_best = get_selected_row(phase6_df)
    lines = [
        "# Event Panel V2 SEC Sentiment V1 Benchmark",
        "",
        "## Scope",
        "",
        "- Locked 34-ticker setup only.",
        "- Same 5-trading-day excess return sign label.",
        "- Same 2024 holdout and same three anchor models.",
        "- No universe expansion and no new external datasets.",
        "",
        "## Important Note",
        "",
        "- The selected SEC filing sentiment artifact was already embedded in the locked `event_panel_v2` Phase 4 anchor.",
        "- This Phase 6 run is therefore a reproducibility and explicit documentation pass for that existing dataset path, not a truly new incremental additive signal test.",
        "",
        "## Panel Comparison",
        "",
        "| Panel | Rows | Tickers | Feature Count | Selected Model |",
        "|---|---:|---:|---:|---|",
        f"| Phase 4 anchor | {phase4_stats['rows']:,} | {phase4_stats['tickers']:,} | {phase4_stats['feature_count']:,} | {phase4_best['model_name']} |",
        f"| Phase 6 sec sentiment v1 | {phase6_stats['rows']:,} | {phase6_stats['tickers']:,} | {phase6_stats['feature_count']:,} | {phase6_best['model_name']} |",
        "",
        "## Per-Model Comparison",
        "",
        *build_model_table(phase4_df, phase6_df),
        "",
    ]
    return "\n".join(lines) + "\n"


def build_decision_doc(phase4_df: pd.DataFrame, phase6_df: pd.DataFrame, phase4_stats: dict, phase6_stats: dict) -> str:
    phase4_best = get_selected_row(phase4_df)
    phase6_best = get_selected_row(phase6_df)
    decision = "FREEZE"
    lines = [
        "# Phase 6 SEC Sentiment Test V1",
        "",
        "## Dataset Choice",
        "",
        f"- Used artifact: `{USED_SENTIMENT_ARTIFACT}`",
        "- Why it was chosen: it is the existing filing-level SEC sentiment artifact already used by the event-panel builder, is available locally, and is joined at the filing-event level by accession number.",
        "- Timing safety: the event panel attaches same-filing sentiment only after the filing becomes available via `effective_model_date`; no daily forward-filled sentiment layer is used in v2.",
        "",
        "## Structural Finding",
        "",
        "- The locked `event_panel_v2` Phase 4 anchor already contains the SEC filing sentiment feature columns at full coverage.",
        "- As a result, this Phase 6 panel is not a new additive merge in practice; it is an explicit rerun and freeze-point for the already-embedded SEC sentiment path.",
        "",
        "## Comparison Against Phase 4 Anchor",
        "",
        f"- Row count: Phase 4 `{phase4_stats['rows']:,}` vs Phase 6 `{phase6_stats['rows']:,}`",
        f"- Feature count: Phase 4 `{phase4_stats['feature_count']:,}` vs Phase 6 `{phase6_stats['feature_count']:,}`",
        f"- Selected primary model: Phase 4 `{phase4_best['model_name']}` vs Phase 6 `{phase6_best['model_name']}`",
        f"- Best CV AUC: Phase 4 `{format_metric(phase4_best['cv_auc_mean'])}` vs Phase 6 `{format_metric(phase6_best['cv_auc_mean'])}`",
        f"- Best holdout AUC: Phase 4 `{format_metric(phase4_best['holdout_auc'])}` vs Phase 6 `{format_metric(phase6_best['holdout_auc'])}`",
        "",
        "## Decision",
        "",
        f"- Final decision: **{decision}**",
        "- Rationale: SEC filing sentiment is not newly improving the locked benchmark here because it was already present in the baseline panel. Keep the code and artifacts for reference, but do not treat this phase as proof of a new additive lift.",
        "- Promotion status: do not promote as a separate new dataset layer from this Phase 6 run.",
        "- Rejection status: do not reject the sentiment path outright either, because the path is already part of the event-panel baseline and remains timing-safe.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    report_md = Path(args.report_md)
    decision_doc = Path(args.decision_doc)
    ensure_parent_dir(report_md)
    ensure_parent_dir(decision_doc)

    phase4_df = load_benchmark(Path(args.phase4_csv))
    phase6_df = load_benchmark(Path(args.phase6_csv))
    phase4_stats = load_panel_stats(Path(args.phase4_panel))
    phase6_stats = load_panel_stats(Path(args.phase6_panel))

    report_md.write_text(build_report_md(phase4_df, phase6_df, phase4_stats, phase6_stats), encoding="utf-8")
    decision_doc.write_text(build_decision_doc(phase4_df, phase6_df, phase4_stats, phase6_stats), encoding="utf-8")
    print(f"Wrote Phase 6 report to: {report_md}")
    print(f"Wrote Phase 6 decision doc to: {decision_doc}")


if __name__ == "__main__":
    main()
