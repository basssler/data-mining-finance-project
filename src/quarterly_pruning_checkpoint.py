from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.benchmark_event_panel_v2_pruning import evaluate_regime, load_selected_model_row
from src.label_comparison_event_v2 import VariantSpec, attach_labels_to_event_panel, build_daily_label_table, load_event_panel
from src.labels_event_v1 import load_price_data, normalize_price_data
from src.train_event_panel_v2 import format_metric, load_config

DEFAULT_CONFIG_PATH = Path("configs") / "event_panel_v2_quarterly_feature_design_core.yaml"
DEFAULT_BENCHMARK_CSV_PATH = Path("reports") / "results" / "levels_plus_deltas_plus_cross_sectional_benchmark.csv"
DEFAULT_OUTPUT_CSV_PATH = Path("reports") / "results" / "quarterly_pruning_checkpoint.csv"
DEFAULT_OUTPUT_MD_PATH = Path("reports") / "results" / "quarterly_pruning_checkpoint.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a light quarterly pruning checkpoint.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--benchmark-csv", default=str(DEFAULT_BENCHMARK_CSV_PATH))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV_PATH))
    parser.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD_PATH))
    return parser.parse_args()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def correlation_prune_features(panel_df: pd.DataFrame, features: list[str], holdout_start: str, threshold: float = 0.95) -> list[str]:
    pre_holdout = panel_df.loc[panel_df["date"] < pd.Timestamp(holdout_start), features].copy()
    pre_holdout = pre_holdout.apply(pd.to_numeric, errors="coerce")
    missingness = pre_holdout.isna().mean().to_dict()
    corr = pre_holdout.corr(method="spearman").abs()
    drop: set[str] = set()
    for i, left in enumerate(features):
        if left in drop or left not in corr.columns:
            continue
        for right in features[i + 1 :]:
            if right in drop or right not in corr.columns:
                continue
            value = corr.loc[left, right]
            if pd.isna(value) or value < threshold:
                continue
            left_missing = float(missingness.get(left, 1.0))
            right_missing = float(missingness.get(right, 1.0))
            if left_missing > right_missing:
                drop.add(left)
                break
            drop.add(right)
    return [feature for feature in features if feature not in drop]


def build_markdown(result_df: pd.DataFrame) -> str:
    lines = [
        "# Quarterly Pruning Checkpoint",
        "",
        "| Regime | Model | Feature Count | CV AUC Mean | CV AUC Std | Worst Fold AUC | Holdout AUC | Holdout Delta vs Baseline |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in result_df.iterrows():
        feature_count = row.get("usable_feature_count_last_fold", row.get("feature_count"))
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["regime_name"]),
                    str(row["model_name"]),
                    str(int(feature_count)),
                    format_metric(row["cv_auc_mean"]),
                    format_metric(row["cv_auc_std"]),
                    format_metric(row["worst_fold_auc"]),
                    format_metric(row["holdout_auc"]),
                    format_metric(row["holdout_auc_delta_vs_baseline"]),
                ]
            )
            + " |"
        )
    best = result_df.sort_values(["holdout_auc", "cv_auc_mean"], ascending=[False, False]).iloc[0]
    lines.extend(["", "## Readout", "", f"- Best pruning regime: `{best['regime_name']}` with holdout AUC `{format_metric(best['holdout_auc'])}`."])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    benchmark_row = load_selected_model_row(Path(args.benchmark_csv))
    output_csv_path = Path(args.output_csv)
    output_md_path = Path(args.output_md)
    ensure_parent_dir(output_csv_path)
    ensure_parent_dir(output_md_path)

    panel_df = load_event_panel(Path(config["panel"]["path"]))
    prices_df = normalize_price_data(load_price_data(Path(config["prices"]["path"])))
    variant = VariantSpec(
        variant_name=str(config["label"]["variant_name"]),
        horizon_days=int(config["label"]["horizon_days"]),
        label_mode=str(config["label"]["mode"]),
        quantile=config["label"].get("quantile"),
        threshold=config["label"].get("threshold"),
    )
    label_df = build_daily_label_table(
        prices_df,
        horizon_days=variant.horizon_days,
        benchmark_mode=str(config["label"]["benchmark_mode"]),
    )
    labeled_panel_df = attach_labels_to_event_panel(panel_df, label_df)

    baseline_features = json.loads(benchmark_row["usable_feature_columns_last_fold"])
    corr_pruned_features = correlation_prune_features(
        labeled_panel_df,
        baseline_features,
        holdout_start=str(config["holdout"]["start"]),
        threshold=0.95,
    )
    regimes = [
        ("unpruned_logistic", baseline_features, "logistic_regression"),
        ("correlation_pruned_logistic", corr_pruned_features, "logistic_regression"),
        ("unpruned_elastic_net", baseline_features, "elastic_net_logistic"),
        ("correlation_pruned_elastic_net", corr_pruned_features, "elastic_net_logistic"),
    ]

    rows: list[dict[str, object]] = []
    for regime_name, regime_features, model_name in regimes:
        result = evaluate_regime(
            regime_name=regime_name,
            regime_features=regime_features,
            panel_df=labeled_panel_df,
            variant=variant,
            explicit_exclusions=list(config["feature_exclusions"]["explicit"]),
            holdout_start=str(config["holdout"]["start"]),
            n_splits=int(config["cv"]["n_splits"]),
            embargo_days=int(config["cv"]["embargo_days"]),
            min_train_dates=int(config["cv"]["min_train_dates"]),
            model_name=model_name,
            max_missingness_pct=float(config["feature_exclusions"]["max_missingness_pct"]),
        )
        rows.append(result)

    result_df = pd.DataFrame(rows)
    baseline = result_df.loc[result_df["regime_name"] == "unpruned_logistic"].iloc[0]
    result_df["holdout_auc_delta_vs_baseline"] = result_df["holdout_auc"] - float(baseline["holdout_auc"])
    result_df = result_df.sort_values(["holdout_auc", "cv_auc_mean"], ascending=[False, False]).reset_index(drop=True)
    result_df.to_csv(output_csv_path, index=False)
    output_md_path.write_text(build_markdown(result_df), encoding="utf-8")

    print(f"Wrote quarterly pruning checkpoint CSV to: {output_csv_path}")
    print(f"Wrote quarterly pruning checkpoint Markdown to: {output_md_path}")


if __name__ == "__main__":
    main()
