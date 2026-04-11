"""Write the Phase 5 universe_v2 benchmark comparison report."""

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

from src.paths import INTERIM_DATA_DIR, PROJECT_ROOT

PRIMARY_BENCHMARK_CSV_PATH = PROJECT_ROOT / "reports" / "results" / "event_panel_v2_primary_benchmark.csv"
UNIVERSE_V2_BENCHMARK_CSV_PATH = (
    PROJECT_ROOT / "reports" / "results" / "event_panel_v2_universe_v2_benchmark.csv"
)
PRIMARY_PANEL_PATH = INTERIM_DATA_DIR / "event_panel_v2.parquet"
UNIVERSE_V2_PANEL_PATH = INTERIM_DATA_DIR / "event_panel_v2_universe_v2.parquet"
OUTPUT_MARKDOWN_PATH = PROJECT_ROOT / "reports" / "results" / "event_panel_v2_universe_v2_benchmark.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the Phase 5 universe_v2 comparison report.")
    parser.add_argument("--current-benchmark-csv", default=str(PRIMARY_BENCHMARK_CSV_PATH))
    parser.add_argument("--expanded-benchmark-csv", default=str(UNIVERSE_V2_BENCHMARK_CSV_PATH))
    parser.add_argument("--current-panel-path", default=str(PRIMARY_PANEL_PATH))
    parser.add_argument("--expanded-panel-path", default=str(UNIVERSE_V2_PANEL_PATH))
    parser.add_argument("--output-path", default=str(OUTPUT_MARKDOWN_PATH))
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
    panel_df = pd.read_parquet(path, columns=["ticker", "event_type", "event_date"])
    event_counts = (
        panel_df["event_type"].value_counts(dropna=False).sort_index().to_dict()
        if "event_type" in panel_df.columns
        else {}
    )
    return {
        "rows": int(len(panel_df)),
        "tickers": int(panel_df["ticker"].nunique()),
        "event_counts": event_counts,
        "date_min": str(pd.to_datetime(panel_df["event_date"]).min().date()),
        "date_max": str(pd.to_datetime(panel_df["event_date"]).max().date()),
    }


def get_selected_row(df: pd.DataFrame) -> pd.Series:
    selected = df.loc[df["is_selected_primary_model"] == True]  # noqa: E712
    if selected.empty:
        raise ValueError("Benchmark CSV does not contain a selected primary model row.")
    return selected.iloc[0]


def build_per_model_table(current_df: pd.DataFrame, expanded_df: pd.DataFrame) -> list[str]:
    merged = current_df.merge(
        expanded_df,
        on="model_name",
        suffixes=("_current", "_expanded"),
        how="outer",
        validate="one_to_one",
    )
    lines = [
        "| Model | 34-Name CV AUC | Universe_v2 CV AUC | 34-Name CV Log Loss | Universe_v2 CV Log Loss | 34-Name Holdout AUC | Universe_v2 Holdout AUC | 34-Name Holdout Log Loss | Universe_v2 Holdout Log Loss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in merged.sort_values("model_name").iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model_name"]),
                    format_metric(row.get("cv_auc_mean_current")),
                    format_metric(row.get("cv_auc_mean_expanded")),
                    format_metric(row.get("cv_log_loss_mean_current")),
                    format_metric(row.get("cv_log_loss_mean_expanded")),
                    format_metric(row.get("holdout_auc_current")),
                    format_metric(row.get("holdout_auc_expanded")),
                    format_metric(row.get("holdout_log_loss_current")),
                    format_metric(row.get("holdout_log_loss_expanded")),
                ]
            )
            + " |"
        )
    return lines


def build_event_count_text(stats: dict) -> str:
    parts = [f"{event_type}={count:,}" for event_type, count in sorted(stats["event_counts"].items())]
    return ", ".join(parts) if parts else "none"


def build_decision_section(current_best: pd.Series, expanded_best: pd.Series) -> list[str]:
    holdout_improved = float(expanded_best["holdout_auc"]) > float(current_best["holdout_auc"])
    cv_improved = float(expanded_best["cv_auc_mean"]) > float(current_best["cv_auc_mean"])
    same_model = str(expanded_best["model_name"]) == str(current_best["model_name"])
    should_switch = holdout_improved and cv_improved

    lines = [
        "## Decision",
        "",
        f"- Did more names improve stability or signal? `{'yes' if should_switch else 'mixed_or_no'}`. Expanded-universe best CV AUC moved from `{format_metric(current_best['cv_auc_mean'])}` to `{format_metric(expanded_best['cv_auc_mean'])}`, and best holdout AUC moved from `{format_metric(current_best['holdout_auc'])}` to `{format_metric(expanded_best['holdout_auc'])}`.",
        f"- Did the same primary model remain best? `{'yes' if same_model else 'no'}`. Current locked winner: `{current_best['model_name']}`. Expanded-universe winner: `{expanded_best['model_name']}`.",
        f"- Should the expanded universe become the new default research universe? `{'yes' if should_switch else 'not_yet'}`.",
    ]
    if not should_switch:
        lines.append(
            "- Recommendation: keep the wider universe as a ready scaling path, but do not promote it to the default unless the rerun shows a clean improvement on both CV and 2024 holdout."
        )
    return lines


def build_markdown(
    current_df: pd.DataFrame,
    expanded_df: pd.DataFrame,
    current_stats: dict,
    expanded_stats: dict,
) -> str:
    current_best = get_selected_row(current_df)
    expanded_best = get_selected_row(expanded_df)
    lines = [
        "# Event Panel V2 Universe V2 Benchmark",
        "",
        "## Scope",
        "",
        "- Locked setup unchanged: `event_panel_v2`, `5-trading-day excess return sign`, logistic regression, random forest, XGBoost.",
        "- This report compares the locked 34-name benchmark against the Phase 5 expanded large-cap cross-sector universe.",
        "",
        "## Panel Comparison",
        "",
        "| Panel | Rows | Tickers | Event Date Range | Event Counts | Selected Model |",
        "|---|---:|---:|---|---|---|",
        f"| 34-name locked panel | {current_stats['rows']:,} | {current_stats['tickers']:,} | {current_stats['date_min']} to {current_stats['date_max']} | {build_event_count_text(current_stats)} | {current_best['model_name']} |",
        f"| universe_v2 expanded panel | {expanded_stats['rows']:,} | {expanded_stats['tickers']:,} | {expanded_stats['date_min']} to {expanded_stats['date_max']} | {build_event_count_text(expanded_stats)} | {expanded_best['model_name']} |",
        "",
        "## Per-Model Comparison",
        "",
        *build_per_model_table(current_df, expanded_df),
        "",
        "## Feature Exclusions",
        "",
        f"- Explicit exclusions carried forward: `{expanded_best['explicit_feature_exclusions']}`",
        f"- Auto all-missing exclusions on expanded universe: `{expanded_best['auto_all_missing_exclusions']}`",
        f"- Auto constant exclusions on expanded universe: `{expanded_best['auto_constant_exclusions']}`",
        "",
        *build_decision_section(current_best, expanded_best),
        "",
        "## Interpretation",
        "",
        "- This Phase 5 report is the scale test only. It does not change the observation unit, label horizon, feature families, or model set.",
        "- If the expanded universe wins cleanly, it becomes the better pre-external-data anchor because it adds cross-sectional breadth without changing the locked method stack.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path)
    ensure_parent_dir(output_path)

    current_df = load_benchmark(Path(args.current_benchmark_csv))
    expanded_df = load_benchmark(Path(args.expanded_benchmark_csv))
    current_stats = load_panel_stats(Path(args.current_panel_path))
    expanded_stats = load_panel_stats(Path(args.expanded_panel_path))

    markdown = build_markdown(current_df, expanded_df, current_stats, expanded_stats)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"Wrote Phase 5 benchmark report to: {output_path}")


if __name__ == "__main__":
    main()
