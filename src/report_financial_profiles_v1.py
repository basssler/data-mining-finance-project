"""Analyze financial profiles and test coarse segment benchmarks on event_panel_v2."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.config_event_v1 import PRICE_INPUT_PATH
from src.label_comparison_event_v2 import (
    VariantSpec,
    apply_variant_label_mode,
    attach_labels_to_event_panel,
    build_daily_label_table,
    candidate_feature_columns,
    clip_outliers,
    compute_global_feature_exclusions,
    evaluate_extended,
    fit_model,
    format_metric,
    load_event_panel,
    select_usable_features,
    summarize_metric_dicts,
)
from src.labels_event_v1 import load_price_data, normalize_price_data
from src.train_event_panel_v2 import load_config
from src.validation_event_v1 import make_event_v1_splits

DEFAULT_CONFIG_PATH = Path("configs") / "event_panel_v2_primary.yaml"
DEFAULT_PROFILE_ANALYSIS_MD = (
    Path("reports") / "results" / "financial_profile_return_analysis_v1.md"
)
DEFAULT_SEGMENT_BENCHMARK_CSV = (
    Path("reports") / "results" / "event_panel_v2_segment_benchmark_v1.csv"
)
DEFAULT_SEGMENT_BENCHMARK_MD = (
    Path("reports") / "results" / "event_panel_v2_segment_benchmark_v1.md"
)
DEFAULT_MIN_SEGMENT_ROWS = 120
DEFAULT_MIN_SEGMENT_TICKERS = 8
DEFAULT_MIN_SEGMENT_HOLDOUT_ROWS = 20
DEFAULT_MIN_SEGMENT_DATES = 80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze financial profiles and run coarse segment benchmarks."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--prices-path", default=str(PRICE_INPUT_PATH))
    parser.add_argument("--profile-output-path", default=str(DEFAULT_PROFILE_ANALYSIS_MD))
    parser.add_argument("--segment-csv-output-path", default=str(DEFAULT_SEGMENT_BENCHMARK_CSV))
    parser.add_argument("--segment-md-output-path", default=str(DEFAULT_SEGMENT_BENCHMARK_MD))
    parser.add_argument("--min-segment-rows", type=int, default=DEFAULT_MIN_SEGMENT_ROWS)
    parser.add_argument("--min-segment-tickers", type=int, default=DEFAULT_MIN_SEGMENT_TICKERS)
    parser.add_argument(
        "--min-segment-holdout-rows",
        type=int,
        default=DEFAULT_MIN_SEGMENT_HOLDOUT_ROWS,
    )
    parser.add_argument("--min-segment-dates", type=int, default=DEFAULT_MIN_SEGMENT_DATES)
    return parser.parse_args()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def map_coarse_segment(panel_df: pd.DataFrame) -> pd.Series:
    """Collapse fine-grained financial profiles into coarse modeling groups."""
    fine_profile = panel_df["financial_profile"].astype("string")
    overall = pd.to_numeric(panel_df["overall_financial_health_score"], errors="coerce")

    coarse = pd.Series("mixed_other", index=panel_df.index, dtype="string")
    coarse.loc[fine_profile.eq("high_growth_fragile")] = "growth_fragile"
    coarse.loc[fine_profile.eq("distressed_weak_quality")] = "financially_weak"
    coarse.loc[fine_profile.isin(["stable_compounder", "mature_defensive"])] = "financially_strong"

    coarse.loc[(coarse == "mixed_other") & (overall >= 0.67)] = "financially_strong"
    coarse.loc[(coarse == "mixed_other") & (overall <= 0.33)] = "financially_weak"
    return coarse


def load_labeled_panel(config: dict, prices_path: Path) -> tuple[pd.DataFrame, VariantSpec]:
    panel_path = Path(config["panel"]["path"])
    panel_df = load_event_panel(panel_path)
    prices_df = normalize_price_data(load_price_data(prices_path))
    variant = VariantSpec(
        variant_name=str(config["label"]["variant_name"]),
        horizon_days=int(config["label"]["horizon_days"]),
        label_mode=str(config["label"]["mode"]),
    )
    label_df = build_daily_label_table(
        prices_df,
        horizon_days=variant.horizon_days,
        benchmark_mode=str(config["label"]["benchmark_mode"]),
    )
    labeled_panel = attach_labels_to_event_panel(panel_df, label_df)
    labeled_panel["financial_profile"] = labeled_panel["financial_profile"].astype("string")
    labeled_panel["coarse_segment"] = map_coarse_segment(labeled_panel)
    return labeled_panel, variant


def summarize_group_returns(
    df: pd.DataFrame,
    group_column: str,
    holdout_start: str,
) -> pd.DataFrame:
    working = df.copy()
    working["target_sign"] = pd.to_numeric(working["target_sign"], errors="coerce")
    working["excess_forward_return"] = pd.to_numeric(
        working["excess_forward_return"],
        errors="coerce",
    )
    working["is_holdout"] = working["date"] >= pd.Timestamp(holdout_start)
    working = working.dropna(subset=[group_column, "excess_forward_return", "target_sign"]).copy()

    rows = []
    for group_value, group_df in working.groupby(group_column, dropna=False):
        holdout_df = group_df.loc[group_df["is_holdout"]].copy()
        rows.append(
            {
                group_column: str(group_value),
                "rows": int(len(group_df)),
                "tickers": int(group_df["ticker"].nunique()),
                "mean_excess_return": float(group_df["excess_forward_return"].mean()),
                "median_excess_return": float(group_df["excess_forward_return"].median()),
                "hit_rate": float(group_df["target_sign"].mean()),
                "holdout_rows": int(len(holdout_df)),
                "holdout_mean_excess_return": (
                    float(holdout_df["excess_forward_return"].mean()) if not holdout_df.empty else np.nan
                ),
                "holdout_hit_rate": (
                    float(holdout_df["target_sign"].mean()) if not holdout_df.empty else np.nan
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("rows", ascending=False).reset_index(drop=True)


def markdown_table_from_df(df: pd.DataFrame, float_columns: list[str]) -> list[str]:
    header = "| " + " | ".join(df.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    lines = [header, separator]
    for _, row in df.iterrows():
        rendered = []
        for column in df.columns:
            value = row[column]
            if column in float_columns:
                if pd.isna(value):
                    rendered.append("n/a")
                else:
                    rendered.append(f"{float(value):.4f}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return lines


def build_profile_analysis_markdown(
    fine_df: pd.DataFrame,
    coarse_df: pd.DataFrame,
    holdout_start: str,
) -> str:
    lines = [
        "# Financial Profile Return Analysis V1",
        "",
        "## Scope",
        "",
        "- Panel: `event_panel_v2` with the enriched financial-profile features.",
        "- Label framing: `5-trading-day excess return sign`.",
        f"- Holdout boundary: `{holdout_start}`.",
        "- Goal: test whether financial profiles differ in average post-event behavior before splitting the model by segment.",
        "",
        "## Fine Profile Results",
        "",
        *markdown_table_from_df(
            fine_df,
            [
                "mean_excess_return",
                "median_excess_return",
                "hit_rate",
                "holdout_mean_excess_return",
                "holdout_hit_rate",
            ],
        ),
        "",
        "## Coarse Segment Results",
        "",
        *markdown_table_from_df(
            coarse_df,
            [
                "mean_excess_return",
                "median_excess_return",
                "hit_rate",
                "holdout_mean_excess_return",
                "holdout_hit_rate",
            ],
        ),
        "",
        "## Interpretation",
        "",
        "- This report is descriptive only. It does not prove tradable segment alpha, but it does test whether the groups are economically different enough to justify segmented models.",
        "- Coarse segments are the safer modeling unit because several fine profiles are too sparse for stable fold-by-fold training.",
    ]
    return "\n".join(lines) + "\n"


def segment_eligibility_reasons(
    segment_df: pd.DataFrame,
    holdout_start: str,
    min_rows: int,
    min_tickers: int,
    min_holdout_rows: int,
    min_dates: int,
) -> list[str]:
    reasons = []
    if len(segment_df) < min_rows:
        reasons.append(f"rows<{min_rows}")
    if segment_df["ticker"].nunique() < min_tickers:
        reasons.append(f"tickers<{min_tickers}")
    if segment_df["date"].nunique() < min_dates:
        reasons.append(f"dates<{min_dates}")

    holdout_df = segment_df.loc[segment_df["date"] >= pd.Timestamp(holdout_start)].copy()
    if len(holdout_df) < min_holdout_rows:
        reasons.append(f"holdout_rows<{min_holdout_rows}")
    if holdout_df["target_sign"].dropna().nunique() < 2:
        reasons.append("holdout_one_class")
    return reasons


def run_segment_model_matrix(
    panel_df: pd.DataFrame,
    variant: VariantSpec,
    model_names: list[str],
    candidate_features: list[str],
    explicit_exclusions: list[str],
    holdout_start: str,
    n_splits: int,
    embargo_days: int,
    min_train_dates: int,
) -> pd.DataFrame:
    split_payload = make_event_v1_splits(
        df=panel_df,
        date_col="date",
        horizon_days=variant.horizon_days,
        n_splits=n_splits,
        embargo_days=embargo_days,
        holdout_start=holdout_start,
        min_train_dates=min_train_dates,
    )

    global_candidates = [column for column in candidate_features if column not in explicit_exclusions]
    kept_global, auto_all_missing, auto_constant = compute_global_feature_exclusions(
        panel_df,
        global_candidates,
        holdout_start=holdout_start,
    )

    rows = []
    for model_name in model_names:
        fold_metrics = []
        last_usable = []
        last_dropped_missing = []
        last_dropped_constant = []

        for fold in split_payload["folds"]:
            train_full = panel_df.iloc[fold["train_indices"]].copy()
            validation_full = panel_df.iloc[fold["validation_indices"]].copy()
            train_active, _ = apply_variant_label_mode(train_full, variant)
            validation_active, _ = apply_variant_label_mode(validation_full, variant)
            if train_active["target"].nunique(dropna=True) < 2:
                continue
            if validation_active["target"].nunique(dropna=True) < 2:
                continue

            usable_features, _, dropped_missing, dropped_constant = select_usable_features(
                train_active,
                kept_global,
            )
            clipped_train, clipped_validation = clip_outliers(
                train_active,
                validation_active,
                usable_features,
            )
            fitted_model, _ = fit_model(
                model_name,
                clipped_train[usable_features],
                clipped_train["target"].astype(int),
            )
            y_prob = fitted_model.predict_proba(clipped_validation[usable_features])[:, 1]
            fold_metrics.append(evaluate_extended(clipped_validation, y_prob, threshold=0.5))
            last_usable = usable_features
            last_dropped_missing = dropped_missing
            last_dropped_constant = dropped_constant

        if not fold_metrics:
            continue

        holdout_train_full = panel_df.iloc[split_payload["holdout"]["train_indices"]].copy()
        holdout_full = panel_df.iloc[split_payload["holdout"]["holdout_indices"]].copy()
        holdout_train_active, _ = apply_variant_label_mode(holdout_train_full, variant)
        holdout_active, _ = apply_variant_label_mode(holdout_full, variant)
        if holdout_train_active["target"].nunique(dropna=True) < 2:
            continue
        if holdout_active["target"].nunique(dropna=True) < 2:
            continue

        holdout_usable, _, holdout_dropped_missing, holdout_dropped_constant = select_usable_features(
            holdout_train_active,
            kept_global,
        )
        clipped_train, clipped_holdout = clip_outliers(
            holdout_train_active,
            holdout_active,
            holdout_usable,
        )
        fitted_model, backend = fit_model(
            model_name,
            clipped_train[holdout_usable],
            clipped_train["target"].astype(int),
        )
        holdout_prob = fitted_model.predict_proba(clipped_holdout[holdout_usable])[:, 1]
        holdout_metrics = evaluate_extended(clipped_holdout, holdout_prob, threshold=0.5)
        cv_summary = summarize_metric_dicts(fold_metrics)

        rows.append(
            {
                "model_name": model_name,
                "cv_auc_mean": cv_summary.get("auc_roc_mean"),
                "cv_log_loss_mean": cv_summary.get("log_loss_mean"),
                "holdout_auc": holdout_metrics.get("auc_roc"),
                "holdout_log_loss": holdout_metrics.get("log_loss"),
                "holdout_precision": holdout_metrics.get("precision"),
                "holdout_recall": holdout_metrics.get("recall"),
                "cv_fold_count": cv_summary["fold_count"],
                "usable_feature_count_last_fold": len(last_usable),
                "usable_feature_columns_last_fold": json.dumps(last_usable),
                "fold_missingness_exclusions_last_fold": json.dumps(sorted(set(last_dropped_missing))),
                "fold_constant_exclusions_last_fold": json.dumps(sorted(set(last_dropped_constant))),
                "holdout_missingness_exclusions": json.dumps(sorted(set(holdout_dropped_missing))),
                "holdout_constant_exclusions": json.dumps(sorted(set(holdout_dropped_constant))),
                "auto_all_missing_exclusions": json.dumps(auto_all_missing),
                "auto_constant_exclusions": json.dumps(auto_constant),
                "xgboost_backend": backend if model_name == "xgboost" else "cpu",
            }
        )

    result_df = pd.DataFrame(rows).sort_values("model_name").reset_index(drop=True)
    if result_df.empty:
        return result_df
    best_model_name = result_df.sort_values(
        by=["cv_auc_mean", "holdout_auc"],
        ascending=[False, False],
    ).iloc[0]["model_name"]
    result_df["is_selected_segment_model"] = result_df["model_name"] == best_model_name
    return result_df


def build_segment_benchmark(
    labeled_panel_df: pd.DataFrame,
    config: dict,
    variant: VariantSpec,
    min_rows: int,
    min_tickers: int,
    min_holdout_rows: int,
    min_dates: int,
) -> tuple[pd.DataFrame, list[dict]]:
    candidate_features = candidate_feature_columns(labeled_panel_df)
    rows = []
    skips = []

    for segment_name, segment_df in labeled_panel_df.groupby("coarse_segment", dropna=False):
        segment_df = segment_df.copy().sort_values(["date", "ticker"]).reset_index(drop=True)
        reasons = segment_eligibility_reasons(
            segment_df=segment_df,
            holdout_start=str(config["holdout"]["start"]),
            min_rows=min_rows,
            min_tickers=min_tickers,
            min_holdout_rows=min_holdout_rows,
            min_dates=min_dates,
        )
        if reasons:
            skips.append(
                {
                    "segment": str(segment_name),
                    "rows": int(len(segment_df)),
                    "tickers": int(segment_df["ticker"].nunique()),
                    "dates": int(segment_df["date"].nunique()),
                    "skip_reasons": reasons,
                }
            )
            continue

        segment_result_df = run_segment_model_matrix(
            panel_df=segment_df,
            variant=variant,
            model_names=list(config["models"]),
            candidate_features=candidate_features,
            explicit_exclusions=list(config["feature_exclusions"]["explicit"]),
            holdout_start=str(config["holdout"]["start"]),
            n_splits=int(config["cv"]["n_splits"]),
            embargo_days=int(config["cv"]["embargo_days"]),
            min_train_dates=min(
                int(config["cv"]["min_train_dates"]),
                max(40, int(segment_df["date"].nunique() * 0.5)),
            ),
        )
        if segment_result_df.empty:
            skips.append(
                {
                    "segment": str(segment_name),
                    "rows": int(len(segment_df)),
                    "tickers": int(segment_df["ticker"].nunique()),
                    "dates": int(segment_df["date"].nunique()),
                    "skip_reasons": ["no_valid_fold_matrix"],
                }
            )
            continue

        segment_result_df.insert(0, "segment", str(segment_name))
        segment_result_df.insert(1, "segment_rows", int(len(segment_df)))
        segment_result_df.insert(2, "segment_tickers", int(segment_df["ticker"].nunique()))
        segment_result_df.insert(3, "segment_dates", int(segment_df["date"].nunique()))
        segment_result_df.insert(
            4,
            "holdout_rows",
            int((segment_df["date"] >= pd.Timestamp(config["holdout"]["start"])).sum()),
        )
        rows.append(segment_result_df)

    if not rows:
        return pd.DataFrame(), skips
    return pd.concat(rows, ignore_index=True), skips


def build_segment_markdown(
    segment_df: pd.DataFrame,
    skip_rows: list[dict],
    holdout_start: str,
) -> str:
    lines = [
        "# Event Panel V2 Segment Benchmark V1",
        "",
        "## Scope",
        "",
        "- Modeling unit: coarse financial segments built from the enriched profile scores.",
        "- Label framing: `5-trading-day excess return sign`.",
        f"- Holdout boundary: `{holdout_start}`.",
        "- Segment-specific models were only run when the segment cleared minimum rows, ticker breadth, date breadth, and holdout-row safeguards.",
        "",
    ]

    if not segment_df.empty:
        lines.extend(
            [
                "## Per-Segment Model Results",
                "",
                "| Segment | Rows | Tickers | Dates | Holdout Rows | Model | Mean CV AUC | Holdout AUC | Mean CV Log Loss | Holdout Log Loss | Selected |",
                "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|",
            ]
        )
        for _, row in segment_df.sort_values(["segment", "model_name"]).iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["segment"]),
                        str(int(row["segment_rows"])),
                        str(int(row["segment_tickers"])),
                        str(int(row["segment_dates"])),
                        str(int(row["holdout_rows"])),
                        str(row["model_name"]),
                        format_metric(row["cv_auc_mean"]),
                        format_metric(row["holdout_auc"]),
                        format_metric(row["cv_log_loss_mean"]),
                        format_metric(row["holdout_log_loss"]),
                        "yes" if bool(row["is_selected_segment_model"]) else "",
                    ]
                )
                + " |"
            )
    else:
        lines.extend(["## Per-Segment Model Results", "", "_No segment passed the safeguards._"])

    lines.extend(["", "## Skipped Segments", ""])
    if skip_rows:
        lines.extend(
            [
                "| Segment | Rows | Tickers | Dates | Reasons |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for item in skip_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(item["segment"]),
                        str(item["rows"]),
                        str(item["tickers"]),
                        str(item["dates"]),
                        ", ".join(item["skip_reasons"]),
                    ]
                )
                + " |"
            )
    else:
        lines.append("_None._")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The goal here is not to force every profile into its own model. The goal is to test whether coarse, economically distinct groups are stable enough to justify separate fits.",
            "- Any segment skipped here should be treated as analysis-only until more rows or broader universe coverage are available.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    profile_output_path = Path(args.profile_output_path)
    segment_csv_output_path = Path(args.segment_csv_output_path)
    segment_md_output_path = Path(args.segment_md_output_path)
    ensure_parent_dir(profile_output_path)
    ensure_parent_dir(segment_csv_output_path)
    ensure_parent_dir(segment_md_output_path)

    config = load_config(Path(args.config))
    labeled_panel_df, variant = load_labeled_panel(config, Path(args.prices_path))

    fine_summary = summarize_group_returns(
        labeled_panel_df,
        group_column="financial_profile",
        holdout_start=str(config["holdout"]["start"]),
    )
    coarse_summary = summarize_group_returns(
        labeled_panel_df,
        group_column="coarse_segment",
        holdout_start=str(config["holdout"]["start"]),
    )
    profile_markdown = build_profile_analysis_markdown(
        fine_df=fine_summary,
        coarse_df=coarse_summary,
        holdout_start=str(config["holdout"]["start"]),
    )
    profile_output_path.write_text(profile_markdown, encoding="utf-8")

    segment_result_df, skip_rows = build_segment_benchmark(
        labeled_panel_df=labeled_panel_df,
        config=config,
        variant=variant,
        min_rows=int(args.min_segment_rows),
        min_tickers=int(args.min_segment_tickers),
        min_holdout_rows=int(args.min_segment_holdout_rows),
        min_dates=int(args.min_segment_dates),
    )
    if not segment_result_df.empty:
        segment_result_df.to_csv(segment_csv_output_path, index=False)
    segment_markdown = build_segment_markdown(
        segment_df=segment_result_df,
        skip_rows=skip_rows,
        holdout_start=str(config["holdout"]["start"]),
    )
    segment_md_output_path.write_text(segment_markdown, encoding="utf-8")

    print(f"Wrote financial profile analysis to: {profile_output_path}")
    if not segment_result_df.empty:
        print(f"Wrote segment benchmark CSV to: {segment_csv_output_path}")
    print(f"Wrote segment benchmark Markdown to: {segment_md_output_path}")


if __name__ == "__main__":
    main()
