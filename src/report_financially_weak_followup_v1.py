"""Directly compare full-panel vs financially_weak segment models on the same subset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.config_event_v1 import PRICE_INPUT_PATH
from src.evaluate_event_v1 import evaluate_classification_run
from src.label_comparison_event_v2 import (
    VariantSpec,
    apply_variant_label_mode,
    attach_labels_to_event_panel,
    build_daily_label_table,
    clip_outliers,
    fit_model,
    load_event_panel,
    safe_rank_ic,
)
from src.labels_event_v1 import load_price_data, normalize_price_data
from src.report_financial_profiles_v1 import map_coarse_segment
from src.train_event_panel_v2 import load_config

DEFAULT_CONFIG_PATH = Path("configs") / "event_panel_v2_primary.yaml"
PRIMARY_BENCHMARK_CSV_PATH = Path("reports") / "results" / "event_panel_v2_primary_benchmark.csv"
SEGMENT_BENCHMARK_CSV_PATH = Path("reports") / "results" / "event_panel_v2_segment_benchmark_v1.csv"
FOLLOWUP_MD_PATH = Path("reports") / "results" / "financially_weak_followup_v1.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare full-panel and financially_weak models on the same subset."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--prices-path", default=str(PRICE_INPUT_PATH))
    parser.add_argument("--primary-benchmark-csv", default=str(PRIMARY_BENCHMARK_CSV_PATH))
    parser.add_argument("--segment-benchmark-csv", default=str(SEGMENT_BENCHMARK_CSV_PATH))
    parser.add_argument("--output-path", default=str(FOLLOWUP_MD_PATH))
    return parser.parse_args()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def get_selected_row(path: Path, selected_column: str, extra_filter: dict | None = None) -> pd.Series:
    df = pd.read_csv(path)
    if extra_filter:
        for key, value in extra_filter.items():
            df = df.loc[df[key] == value].copy()
    selected = df.loc[df[selected_column] == True]  # noqa: E712
    if selected.empty:
        raise ValueError(f"No selected row found in {path}.")
    return selected.iloc[0]


def load_labeled_panel(config: dict, prices_path: Path) -> tuple[pd.DataFrame, VariantSpec]:
    panel_df = load_event_panel(Path(config["panel"]["path"]))
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


def parse_feature_list(serialized: str) -> list[str]:
    parsed = json.loads(serialized)
    return [str(item) for item in parsed]


def compute_metrics(scoring_df: pd.DataFrame, y_prob) -> dict:
    metrics = evaluate_classification_run(
        y_true=scoring_df["target"].astype(int),
        y_prob=y_prob,
        threshold=0.5,
    )
    metrics["rank_ic_spearman"] = safe_rank_ic(scoring_df["excess_forward_return"], y_prob)
    return metrics


def fit_and_score(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[dict, str]:
    clipped_train, clipped_test = clip_outliers(train_df, test_df, feature_columns)
    fitted_model, backend = fit_model(
        model_name,
        clipped_train[feature_columns],
        clipped_train["target"].astype(int),
    )
    y_prob = fitted_model.predict_proba(clipped_test[feature_columns])[:, 1]
    return compute_metrics(clipped_test, y_prob), backend


def build_followup_markdown(
    subset_rows: int,
    subset_tickers: int,
    full_model_name: str,
    full_metrics: dict,
    segment_model_name: str,
    segment_metrics: dict,
    full_backend: str,
    segment_backend: str,
) -> str:
    lines = [
        "# Financially Weak Follow-Up V1",
        "",
        "## Scope",
        "",
        "- This is the direct apples-to-apples follow-up requested after the profile analysis.",
        "- Both models are evaluated on the same `financially_weak` 2024 holdout subset.",
        "- Comparison target: does a dedicated segment model outperform the enriched full-panel model on that subset?",
        "",
        "## Holdout Subset",
        "",
        f"- Rows: `{subset_rows}`",
        f"- Tickers: `{subset_tickers}`",
        "",
        "## Model Comparison",
        "",
        "| Training Scope | Model | Backend | Holdout AUC | Holdout Log Loss | Holdout Precision | Holdout Recall | Holdout Rank IC |",
        "|---|---|---|---:|---:|---:|---:|---:|",
        "| "
        + " | ".join(
            [
                "full_panel",
                full_model_name,
                full_backend,
                f"{full_metrics['auc_roc']:.4f}" if full_metrics["auc_roc"] is not None else "n/a",
                f"{full_metrics['log_loss']:.4f}",
                f"{full_metrics['precision']:.4f}",
                f"{full_metrics['recall']:.4f}",
                (
                    f"{full_metrics['rank_ic_spearman']:.4f}"
                    if full_metrics["rank_ic_spearman"] is not None
                    else "n/a"
                ),
            ]
        )
        + " |",
        "| "
        + " | ".join(
            [
                "financially_weak_only",
                segment_model_name,
                segment_backend,
                f"{segment_metrics['auc_roc']:.4f}" if segment_metrics["auc_roc"] is not None else "n/a",
                f"{segment_metrics['log_loss']:.4f}",
                f"{segment_metrics['precision']:.4f}",
                f"{segment_metrics['recall']:.4f}",
                (
                    f"{segment_metrics['rank_ic_spearman']:.4f}"
                    if segment_metrics["rank_ic_spearman"] is not None
                    else "n/a"
                ),
            ]
        )
        + " |",
        "",
        "## Interpretation",
        "",
    ]

    if segment_metrics["auc_roc"] is not None and full_metrics["auc_roc"] is not None:
        if float(segment_metrics["auc_roc"]) > float(full_metrics["auc_roc"]):
            lines.append(
                "- On the financially weak subset, the dedicated segment model beat the enriched full-panel model on holdout AUC."
            )
        else:
            lines.append(
                "- On the financially weak subset, the enriched full-panel model remained as good or better on holdout AUC than the dedicated segment model."
            )
    lines.append(
        "- This follow-up is narrower than the benchmark report: it tests segment specialization only on the subset where segmentation looked most promising."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path)
    ensure_parent_dir(output_path)

    config = load_config(Path(args.config))
    labeled_panel_df, variant = load_labeled_panel(config, Path(args.prices_path))

    full_selected = get_selected_row(
        Path(args.primary_benchmark_csv),
        selected_column="is_selected_primary_model",
    )
    weak_selected = get_selected_row(
        Path(args.segment_benchmark_csv),
        selected_column="is_selected_segment_model",
        extra_filter={"segment": "financially_weak"},
    )

    full_feature_columns = parse_feature_list(full_selected["usable_feature_columns_last_fold"])
    weak_feature_columns = parse_feature_list(weak_selected["usable_feature_columns_last_fold"])

    prepared_panel, _ = apply_variant_label_mode(labeled_panel_df, variant)
    train_df = prepared_panel.loc[prepared_panel["date"] < pd.Timestamp(config["holdout"]["start"])].copy()
    holdout_df = prepared_panel.loc[prepared_panel["date"] >= pd.Timestamp(config["holdout"]["start"])].copy()

    weak_train_df = train_df.loc[train_df["coarse_segment"] == "financially_weak"].copy()
    weak_holdout_df = holdout_df.loc[holdout_df["coarse_segment"] == "financially_weak"].copy()

    full_metrics, full_backend = fit_and_score(
        model_name=str(full_selected["model_name"]),
        train_df=train_df,
        test_df=weak_holdout_df,
        feature_columns=full_feature_columns,
    )
    weak_metrics, weak_backend = fit_and_score(
        model_name=str(weak_selected["model_name"]),
        train_df=weak_train_df,
        test_df=weak_holdout_df,
        feature_columns=weak_feature_columns,
    )

    markdown = build_followup_markdown(
        subset_rows=int(len(weak_holdout_df)),
        subset_tickers=int(weak_holdout_df["ticker"].nunique()),
        full_model_name=str(full_selected["model_name"]),
        full_metrics=full_metrics,
        segment_model_name=str(weak_selected["model_name"]),
        segment_metrics=weak_metrics,
        full_backend=full_backend,
        segment_backend=weak_backend,
    )
    output_path.write_text(markdown, encoding="utf-8")
    print(f"Wrote financially weak follow-up to: {output_path}")


if __name__ == "__main__":
    main()
