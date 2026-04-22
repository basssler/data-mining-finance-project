from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.label_comparison_event_v2 import (
    VariantSpec,
    attach_labels_to_event_panel,
    build_daily_label_table,
    load_event_panel,
)
from src.labels_event_v1 import load_price_data, normalize_price_data
from src.quarterly_stability import build_family_view_summary, pick_best_holdout_row, pick_best_stable_row
from src.quarterly_stability import compute_model_stability_artifacts
from src.train_event_panel_v2 import format_metric, resolve_candidate_features, resolve_max_missingness_pct

DEFAULT_CONFIG_PATHS = [
    Path("configs") / "event_panel_v2_quarterly_stability_core_raw_components.yaml",
    Path("configs") / "event_panel_v2_quarterly_stability_core_additive.yaml",
    Path("configs") / "event_panel_v2_quarterly_stability_core_capped.yaml",
    Path("configs") / "event_panel_v2_quarterly_stability_core_bucketed.yaml",
    Path("configs") / "event_panel_v2_quarterly_stability_sentiment_raw_components.yaml",
    Path("configs") / "event_panel_v2_quarterly_stability_sentiment_additive.yaml",
    Path("configs") / "event_panel_v2_quarterly_stability_sentiment_capped.yaml",
    Path("configs") / "event_panel_v2_quarterly_stability_sentiment_bucketed.yaml",
]
DEFAULT_SUMMARY_CSV_PATH = Path("reports") / "results" / "quarterly_stability_comparison.csv"
DEFAULT_SUMMARY_MD_PATH = Path("reports") / "results" / "quarterly_stability_comparison.md"
DEFAULT_MATRIX_CSV_PATH = Path("reports") / "results" / "quarterly_stability_matrix.csv"
DEFAULT_FOLD_METRICS_CSV_PATH = Path("reports") / "results" / "quarterly_stability_fold_metrics.csv"
DEFAULT_FOLD_CONCENTRATION_CSV_PATH = (
    Path("reports") / "results" / "quarterly_stability_fold_concentration.csv"
)
CURRENT_BASELINE_HOLDOUT_AUC = 0.4411
CURRENT_ALPHA_HOLDOUT_AUC = 0.4683
CURRENT_BEST_DESIGNED_HOLDOUT_AUC = 0.4956
VIEW_LABELS = {
    "trainer_selected": "Trainer Selected",
    "best_holdout": "Best Holdout",
    "best_stable": "Best Stable",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and summarize quarterly stability experiments.")
    parser.add_argument("--configs", nargs="+", default=[str(path) for path in DEFAULT_CONFIG_PATHS])
    parser.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY_CSV_PATH))
    parser.add_argument("--summary-md", default=str(DEFAULT_SUMMARY_MD_PATH))
    parser.add_argument("--matrix-csv", default=str(DEFAULT_MATRIX_CSV_PATH))
    parser.add_argument("--fold-metrics-csv", default=str(DEFAULT_FOLD_METRICS_CSV_PATH))
    parser.add_argument("--fold-concentration-csv", default=str(DEFAULT_FOLD_CONCENTRATION_CSV_PATH))
    return parser.parse_args()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def run_training(config_path: Path) -> None:
    subprocess.run([sys.executable, "src/train_event_panel_v2.py", "--config", str(config_path)], check=True)


def load_labeled_panel_bundle(config: dict, cache: dict) -> tuple[pd.DataFrame, VariantSpec, list[str]]:
    panel_path = Path(config["panel"]["path"])
    prices_path = Path(config["prices"]["path"])
    label_config = config["label"]
    cache_key = (
        str(panel_path),
        str(prices_path),
        str(label_config["variant_name"]),
        int(label_config["horizon_days"]),
        str(label_config["mode"]),
        str(label_config["benchmark_mode"]),
    )
    if cache_key not in cache:
        panel_df = load_event_panel(panel_path)
        prices_df = normalize_price_data(load_price_data(prices_path))
        variant = VariantSpec(
            variant_name=str(label_config["variant_name"]),
            horizon_days=int(label_config["horizon_days"]),
            label_mode=str(label_config["mode"]),
        )
        label_df = build_daily_label_table(
            prices_df,
            horizon_days=variant.horizon_days,
            benchmark_mode=str(label_config["benchmark_mode"]),
        )
        labeled_panel_df = attach_labels_to_event_panel(panel_df, label_df)
        candidate_features = resolve_candidate_features(labeled_panel_df, config)
        cache[cache_key] = (labeled_panel_df, variant, candidate_features)
    return cache[cache_key]


def compute_config_stability_artifacts(
    config_path: Path,
    config: dict,
    panel_cache: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labeled_panel_df, variant, candidate_features = load_labeled_panel_bundle(config, panel_cache)
    benchmark_csv_path = Path(config["outputs"]["csv"])
    benchmark_df = pd.read_csv(benchmark_csv_path)
    metadata = config.get("metadata", {})
    experiment_family = str(metadata.get("experiment_family", config["panel"]["name"]))
    interaction_style = str(metadata.get("interaction_style", "default"))
    design_note = str(metadata.get("design_note", ""))

    stability_df, fold_metrics_df, concentration_df = compute_model_stability_artifacts(
        panel_df=labeled_panel_df,
        variant=variant,
        model_names=list(config["models"]),
        candidate_features=candidate_features,
        explicit_exclusions=list(config["feature_exclusions"]["explicit"]),
        holdout_start=str(config["holdout"]["start"]),
        n_splits=int(config["cv"]["n_splits"]),
        embargo_days=int(config["cv"]["embargo_days"]),
        min_train_dates=int(config["cv"]["min_train_dates"]),
        threshold=0.5,
        max_missingness_pct=resolve_max_missingness_pct(config.get("feature_exclusions")),
    )
    matrix_df = benchmark_df.merge(stability_df, on="model_name", how="left", validate="one_to_one")

    for df in [matrix_df, fold_metrics_df, concentration_df]:
        df["config_path"] = str(config_path)
        df["panel_name"] = str(config["panel"]["name"])
        df["experiment_family"] = experiment_family
        df["interaction_style"] = interaction_style
        df["design_note"] = design_note

    matrix_df["beats_plain_quarterly"] = matrix_df["holdout_auc"] > CURRENT_BASELINE_HOLDOUT_AUC
    matrix_df["beats_quarterly_alpha_vantage"] = matrix_df["holdout_auc"] >= CURRENT_ALPHA_HOLDOUT_AUC
    matrix_df["beats_current_best_quarterly"] = matrix_df["holdout_auc"] >= CURRENT_BEST_DESIGNED_HOLDOUT_AUC
    return matrix_df, fold_metrics_df, concentration_df


def build_markdown(summary_df: pd.DataFrame) -> str:
    stable_views = summary_df.loc[summary_df["view_name"] == "best_stable"].copy()
    holdout_views = summary_df.loc[summary_df["view_name"] == "best_holdout"].copy()
    best_stable = pick_best_stable_row(stable_views)
    best_holdout = pick_best_holdout_row(holdout_views)
    lines = [
        "# Quarterly Stability Comparison",
        "",
        "## Summary",
        "",
        f"- Plain quarterly checkpoint: holdout AUC `{CURRENT_BASELINE_HOLDOUT_AUC:.4f}`.",
        f"- Quarterly Alpha Vantage checkpoint: holdout AUC `{CURRENT_ALPHA_HOLDOUT_AUC:.4f}`.",
        f"- Current best redesigned quarterly checkpoint: holdout AUC `{CURRENT_BEST_DESIGNED_HOLDOUT_AUC:.4f}`.",
        (
            f"- Best overall stable recommendation: family `{best_stable['experiment_family']}`, "
            f"style `{best_stable['interaction_style']}`, model `{best_stable['model_name']}`, "
            f"holdout AUC `{format_metric(best_stable['holdout_auc'])}`, "
            f"worst-fold AUC `{format_metric(best_stable['worst_fold_auc'])}`, "
            f"CV AUC std `{format_metric(best_stable['cv_auc_std'])}`, and dominant top-3 feature "
            f"`{best_stable['dominant_feature_name']}` appearing in `{int(best_stable['dominant_feature_top3_folds'])}` folds."
        ),
        (
            f"- Best raw holdout configuration remains family `{best_holdout['experiment_family']}`, "
            f"style `{best_holdout['interaction_style']}`, model `{best_holdout['model_name']}`, "
            f"with holdout AUC `{format_metric(best_holdout['holdout_auc'])}`."
        ),
        "",
        "## Family Views",
        "",
        "| Family | View | Style | Model | Holdout AUC | CV Mean AUC | CV AUC Std | Worst Fold AUC | Dominant Top-3 Feature | Dominant Top-3 Folds | Holdout Rows | Beat AV | Beat Current Best |",
        "|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---|---|",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["experiment_family"]),
                    VIEW_LABELS.get(str(row["view_name"]), str(row["view_name"])),
                    str(row["interaction_style"]),
                    str(row["model_name"]),
                    format_metric(row["holdout_auc"]),
                    format_metric(row["cv_auc_mean"]),
                    format_metric(row["cv_auc_std"]),
                    format_metric(row["worst_fold_auc"]),
                    str(row["dominant_feature_name"]),
                    str(int(row["dominant_feature_top3_folds"])),
                    str(int(row["holdout_row_count"])),
                    "yes" if bool(row["beats_quarterly_alpha_vantage"]) else "",
                    "yes" if bool(row["beats_current_best_quarterly"]) else "",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Readout", ""])
    if bool(best_stable["beats_current_best_quarterly"]):
        lines.append("- A stabilized quarterly candidate now matches or exceeds the current best redesigned quarterly holdout checkpoint.")
    elif bool(best_stable["beats_quarterly_alpha_vantage"]):
        lines.append(
            "- No stabilized candidate cleared the current best redesigned quarterly checkpoint, but the promoted stable lane still beats the quarterly Alpha Vantage checkpoint. Treat this as a stability/performance tradeoff rather than a full replacement."
        )
    else:
        lines.append(
            "- The stabilized candidates did not beat the quarterly Alpha Vantage checkpoint. Keep the previous redesigned lane as the active quarterly benchmark and use this report to demote unstable feature patterns."
        )
    lines.append(
        "- `trainer_selected` is kept for audit only. `best_stable` is the quarterly recommendation because it balances holdout performance with fold survival and driver concentration."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    summary_csv_path = Path(args.summary_csv)
    summary_md_path = Path(args.summary_md)
    matrix_csv_path = Path(args.matrix_csv)
    fold_metrics_csv_path = Path(args.fold_metrics_csv)
    fold_concentration_csv_path = Path(args.fold_concentration_csv)
    for output_path in [
        summary_csv_path,
        summary_md_path,
        matrix_csv_path,
        fold_metrics_csv_path,
        fold_concentration_csv_path,
    ]:
        ensure_parent_dir(output_path)

    panel_cache: dict = {}
    matrix_frames: list[pd.DataFrame] = []
    fold_metric_frames: list[pd.DataFrame] = []
    concentration_frames: list[pd.DataFrame] = []
    for config_text in args.configs:
        config_path = Path(config_text)
        config = load_yaml(config_path)
        run_training(config_path)
        matrix_df, fold_metrics_df, concentration_df = compute_config_stability_artifacts(
            config_path=config_path,
            config=config,
            panel_cache=panel_cache,
        )
        matrix_frames.append(matrix_df)
        fold_metric_frames.append(fold_metrics_df)
        concentration_frames.append(concentration_df)

    matrix_df = (
        pd.concat(matrix_frames, ignore_index=True)
        .sort_values(["experiment_family", "interaction_style", "holdout_auc", "model_name"], ascending=[True, True, False, True])
        .reset_index(drop=True)
    )
    fold_metrics_df = (
        pd.concat(fold_metric_frames, ignore_index=True)
        .sort_values(["experiment_family", "interaction_style", "model_name", "fold_number"])
        .reset_index(drop=True)
    )
    concentration_df = (
        pd.concat(concentration_frames, ignore_index=True)
        .sort_values(["experiment_family", "interaction_style", "model_name", "fold_number", "rank"])
        .reset_index(drop=True)
    )
    summary_df = build_family_view_summary(matrix_df)
    summary_df["beats_plain_quarterly"] = summary_df["holdout_auc"] > CURRENT_BASELINE_HOLDOUT_AUC
    summary_df["beats_quarterly_alpha_vantage"] = summary_df["holdout_auc"] >= CURRENT_ALPHA_HOLDOUT_AUC
    summary_df["beats_current_best_quarterly"] = summary_df["holdout_auc"] >= CURRENT_BEST_DESIGNED_HOLDOUT_AUC

    matrix_df.to_csv(matrix_csv_path, index=False)
    fold_metrics_df.to_csv(fold_metrics_csv_path, index=False)
    concentration_df.to_csv(fold_concentration_csv_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)
    summary_md_path.write_text(build_markdown(summary_df), encoding="utf-8")

    print(f"Wrote quarterly stability matrix CSV to: {matrix_csv_path}")
    print(f"Wrote quarterly stability fold metrics CSV to: {fold_metrics_csv_path}")
    print(f"Wrote quarterly stability fold concentration CSV to: {fold_concentration_csv_path}")
    print(f"Wrote quarterly stability summary CSV to: {summary_csv_path}")
    print(f"Wrote quarterly stability summary Markdown to: {summary_md_path}")


if __name__ == "__main__":
    main()
