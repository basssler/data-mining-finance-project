"""Train the locked Phase 4 event_panel_v2 benchmark matrix."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.label_comparison_event_v2 import (
    VariantSpec,
    apply_variant_label_mode,
    attach_labels_to_event_panel,
    build_daily_label_table,
    candidate_feature_columns,
    choose_best_model,
    clip_outliers,
    compute_global_feature_exclusions,
    evaluate_extended,
    fit_model,
    format_metric,
    load_event_panel,
    safe_rank_ic,
    select_usable_features,
    summarize_metric_dicts,
)
from src.labels_event_v1 import load_price_data, normalize_price_data
from src.validation_event_v1 import make_event_v1_splits
from src.config_event_v1 import PRICE_INPUT_PATH

DEFAULT_CONFIG_PATH = Path("configs") / "event_panel_v2_primary.yaml"
OLD_BASELINE_METRICS_PATH = Path("reports") / "results" / "event_v1_layer1_metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the locked event_panel_v2 benchmark.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    return parser.parse_args()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file was not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def set_random_seeds(seed_block: dict) -> None:
    random.seed(int(seed_block.get("python", 42)))
    np.random.seed(int(seed_block.get("numpy", 42)))


def load_old_baseline() -> dict | None:
    if not OLD_BASELINE_METRICS_PATH.exists():
        return None
    payload = json.loads(OLD_BASELINE_METRICS_PATH.read_text(encoding="utf-8"))
    best = payload["best_model"]
    return {
        "panel_name": payload["panel_name"],
        "model_name": best["model_name"],
        "cv_auc": best["cv_summary"]["auc_roc_mean"],
        "cv_log_loss": best["cv_summary"]["log_loss_mean"],
        "holdout_auc": best["holdout"]["holdout_metrics"]["auc_roc"],
        "holdout_log_loss": best["holdout"]["holdout_metrics"]["log_loss"],
    }


def run_model_matrix(
    panel_df: pd.DataFrame,
    variant: VariantSpec,
    model_names: list[str],
    candidate_features: list[str],
    explicit_exclusions: list[str],
    holdout_start: str,
    n_splits: int,
    embargo_days: int,
    min_train_dates: int,
    threshold: float,
    panel_name: str,
) -> tuple[pd.DataFrame, dict]:
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
        last_missingness = {}

        for fold in split_payload["folds"]:
            train_full = panel_df.iloc[fold["train_indices"]].copy()
            validation_full = panel_df.iloc[fold["validation_indices"]].copy()
            train_active, _ = apply_variant_label_mode(train_full, variant)
            validation_active, _ = apply_variant_label_mode(validation_full, variant)
            if train_active["target"].nunique(dropna=True) < 2 or validation_active["target"].nunique(dropna=True) < 2:
                continue

            usable_features, missingness_by_feature, dropped_missing, dropped_constant = select_usable_features(
                train_active,
                kept_global,
            )
            clipped_train, clipped_validation = clip_outliers(train_active, validation_active, usable_features)
            fitted_model, _ = fit_model(
                model_name,
                clipped_train[usable_features],
                clipped_train["target"].astype(int),
            )
            y_prob = fitted_model.predict_proba(clipped_validation[usable_features])[:, 1]
            fold_metrics.append(evaluate_extended(clipped_validation, y_prob, threshold=threshold))

            last_usable = usable_features
            last_dropped_missing = dropped_missing
            last_dropped_constant = dropped_constant
            last_missingness = missingness_by_feature

        holdout_train_full = panel_df.iloc[split_payload["holdout"]["train_indices"]].copy()
        holdout_full = panel_df.iloc[split_payload["holdout"]["holdout_indices"]].copy()
        holdout_train_active, _ = apply_variant_label_mode(holdout_train_full, variant)
        holdout_active, _ = apply_variant_label_mode(holdout_full, variant)
        holdout_usable, _, holdout_dropped_missing, holdout_dropped_constant = select_usable_features(
            holdout_train_active,
            kept_global,
        )
        clipped_train, clipped_holdout = clip_outliers(holdout_train_active, holdout_active, holdout_usable)
        fitted_model, backend = fit_model(
            model_name,
            clipped_train[holdout_usable],
            clipped_train["target"].astype(int),
        )
        holdout_prob = fitted_model.predict_proba(clipped_holdout[holdout_usable])[:, 1]
        holdout_metrics = evaluate_extended(clipped_holdout, holdout_prob, threshold=threshold)
        cv_summary = summarize_metric_dicts(fold_metrics)
        rows.append(
            {
                "panel_name": panel_name,
                "label_variant": variant.variant_name,
                "model_name": model_name,
                "cv_auc_mean": cv_summary.get("auc_roc_mean"),
                "cv_log_loss_mean": cv_summary.get("log_loss_mean"),
                "cv_precision_mean": cv_summary.get("precision_mean"),
                "cv_recall_mean": cv_summary.get("recall_mean"),
                "cv_rank_ic_mean": cv_summary.get("rank_ic_spearman_mean"),
                "holdout_auc": holdout_metrics.get("auc_roc"),
                "holdout_log_loss": holdout_metrics.get("log_loss"),
                "holdout_precision": holdout_metrics.get("precision"),
                "holdout_recall": holdout_metrics.get("recall"),
                "holdout_rank_ic": holdout_metrics.get("rank_ic_spearman"),
                "cv_fold_count": cv_summary["fold_count"],
                "holdout_row_count": holdout_metrics["row_count"],
                "xgboost_backend": backend if model_name == "xgboost" else "cpu",
                "explicit_feature_exclusions": json.dumps(explicit_exclusions),
                "auto_all_missing_exclusions": json.dumps(auto_all_missing),
                "auto_constant_exclusions": json.dumps(auto_constant),
                "fold_missingness_exclusions_last_fold": json.dumps(sorted(set(last_dropped_missing))),
                "fold_constant_exclusions_last_fold": json.dumps(sorted(set(last_dropped_constant))),
                "holdout_missingness_exclusions": json.dumps(sorted(set(holdout_dropped_missing))),
                "holdout_constant_exclusions": json.dumps(sorted(set(holdout_dropped_constant))),
                "usable_feature_count_last_fold": len(last_usable),
                "usable_feature_columns_last_fold": json.dumps(last_usable),
                "train_missingness_by_feature_pct_last_fold": json.dumps(last_missingness),
                "holdout_start": holdout_start,
                "n_splits": n_splits,
                "embargo_days": embargo_days,
                "min_train_dates": min_train_dates,
            }
        )

    result_df = pd.DataFrame(rows).sort_values("model_name").reset_index(drop=True)
    best_model_name = choose_best_model(result_df.to_dict(orient="records"))
    result_df["is_selected_primary_model"] = result_df["model_name"] == best_model_name
    summary = {
        "best_model_name": best_model_name,
        "explicit_exclusions": explicit_exclusions,
        "auto_all_missing": auto_all_missing,
        "auto_constant": auto_constant,
        "split_payload": split_payload,
    }
    return result_df, summary


def build_markdown_report(result_df: pd.DataFrame, summary: dict, old_baseline: dict | None) -> str:
    best_row = result_df.loc[result_df["is_selected_primary_model"]].iloc[0]
    lines = [
        "# Event Panel V2 Primary Benchmark",
        "",
        "## Locked Setup",
        "",
        "- Primary panel: `event_panel_v2`",
        "- Primary label: `5-trading-day excess return sign`",
        "- Models: `logistic_regression`, `random_forest`, `xgboost`",
        "- 2024 holdout policy: unchanged",
        "- This report is the new post-fix anchor to use before universe expansion.",
        "",
        "## Per-Model Results",
        "",
        "| Model | Mean CV AUC | Mean CV Log Loss | Holdout AUC | Holdout Log Loss | Holdout Precision | Holdout Recall | Holdout Rank IC | XGBoost Backend | Selected Primary |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for _, row in result_df.sort_values("model_name").iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model_name"]),
                    format_metric(row["cv_auc_mean"]),
                    format_metric(row["cv_log_loss_mean"]),
                    format_metric(row["holdout_auc"]),
                    format_metric(row["holdout_log_loss"]),
                    format_metric(row["holdout_precision"]),
                    format_metric(row["holdout_recall"]),
                    format_metric(row["holdout_rank_ic"]),
                    str(row["xgboost_backend"]),
                    "yes" if row["is_selected_primary_model"] else "",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Feature Exclusions",
            "",
            f"- Explicit exclusions: `{', '.join(summary['explicit_exclusions']) if summary['explicit_exclusions'] else 'none'}`",
            f"- Auto all-missing exclusions: `{', '.join(summary['auto_all_missing']) if summary['auto_all_missing'] else 'none'}`",
            f"- Auto constant exclusions: `{', '.join(summary['auto_constant']) if summary['auto_constant'] else 'none'}`",
            "",
            "## Selected Primary Model",
            "",
            f"- Selected model: `{best_row['model_name']}`",
            f"- Mean CV AUC: `{format_metric(best_row['cv_auc_mean'])}`",
            f"- Mean CV log loss: `{format_metric(best_row['cv_log_loss_mean'])}`",
            f"- 2024 holdout AUC: `{format_metric(best_row['holdout_auc'])}`",
            f"- 2024 holdout log loss: `{format_metric(best_row['holdout_log_loss'])}`",
            "",
            "## Interpretation",
            "",
        ]
    )
    if old_baseline is not None:
        lines.append(
            f"- Against the old daily/event_v1 direction (`{old_baseline['panel_name']}` best model `{old_baseline['model_name']}`), "
            f"the redesigned event setup improves best CV AUC from `{format_metric(old_baseline['cv_auc'])}` to `{format_metric(best_row['cv_auc_mean'])}` "
            f"and best holdout AUC from `{format_metric(old_baseline['holdout_auc'])}` to `{format_metric(best_row['holdout_auc'])}`."
        )
    lines.append(
        "- The redesigned setup is directionally better than the old daily research path, but the edge is still modest. This should be treated as a cleaner anchor, not as proof that the problem is solved."
    )
    if str(result_df.loc[result_df["model_name"] == "xgboost", "xgboost_backend"].iloc[0]) == "cpu":
        lines.append(
            "- XGBoost ran on CPU in this benchmark because the local stack does not support clean CUDA prediction without the device-mismatch warning."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    set_random_seeds(config.get("random_seed", {}))

    panel_path = Path(config["panel"]["path"])
    csv_path = Path(config["outputs"]["csv"])
    markdown_path = Path(config["outputs"]["markdown"])
    ensure_parent_dir(csv_path)
    ensure_parent_dir(markdown_path)

    print(f"Loading event panel from: {panel_path}")
    panel_df = load_event_panel(panel_path)
    panel_name = str(config.get("panel", {}).get("name", "event_panel_v2"))
    prices_path = Path(config.get("prices", {}).get("path", str(PRICE_INPUT_PATH)))
    print(f"Loading prices from: {prices_path}")
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
    labeled_panel_df = attach_labels_to_event_panel(panel_df, label_df)

    result_df, summary = run_model_matrix(
        panel_df=labeled_panel_df,
        variant=variant,
        model_names=list(config["models"]),
        candidate_features=candidate_feature_columns(labeled_panel_df),
        explicit_exclusions=list(config["feature_exclusions"]["explicit"]),
        holdout_start=str(config["holdout"]["start"]),
        n_splits=int(config["cv"]["n_splits"]),
        embargo_days=int(config["cv"]["embargo_days"]),
        min_train_dates=int(config["cv"]["min_train_dates"]),
        threshold=0.5,
        panel_name=panel_name,
    )

    print(f"Saving benchmark CSV to: {csv_path}")
    result_df.to_csv(csv_path, index=False)

    old_baseline = load_old_baseline()
    markdown = build_markdown_report(result_df, summary, old_baseline)
    print(f"Saving benchmark Markdown to: {markdown_path}")
    markdown_path.write_text(markdown, encoding="utf-8")

    best_row = result_df.loc[result_df["is_selected_primary_model"]].iloc[0]
    print("\nPhase 4 Benchmark Summary")
    print("-" * 60)
    for _, row in result_df.sort_values("model_name").iterrows():
        print(
            f"{row['model_name']:<20} cv_auc={format_metric(row['cv_auc_mean'])} "
            f"holdout_auc={format_metric(row['holdout_auc'])} backend={row['xgboost_backend']}"
        )
    print(f"\nSelected primary model: {best_row['model_name']}")


if __name__ == "__main__":
    main()
