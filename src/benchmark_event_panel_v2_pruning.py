"""Benchmark conservative pruning regimes for the locked event_panel_v2 setup."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.analyze_event_panel_v2_primary_shap import build_variant, load_labeled_panel
from src.label_comparison_event_v2 import (
    apply_variant_label_mode,
    clip_outliers,
    compute_global_feature_exclusions,
    evaluate_extended,
    fit_model,
    format_metric,
    resolve_max_missingness_pct,
    select_usable_features,
    summarize_metric_dicts,
)
from src.train_event_panel_v2 import load_config, resolve_candidate_features
from src.validation_event_v1 import make_event_v1_splits

DEFAULT_CONFIG_PATH = Path("configs") / "event_panel_v2_primary.yaml"
DEFAULT_BENCHMARK_CSV_PATH = Path("reports") / "results" / "event_panel_v2_primary_benchmark.csv"
DEFAULT_SHAP_CSV_PATH = Path("reports") / "results" / "event_panel_v2_primary_shap_importance.csv"
DEFAULT_OUTPUT_CSV_PATH = Path("reports") / "results" / "event_panel_v2_pruning_benchmark.csv"
DEFAULT_OUTPUT_MD_PATH = Path("reports") / "results" / "event_panel_v2_pruning_benchmark.md"

PRICE_VOLUME_TOKENS = (
    "return",
    "vol",
    "gap",
    "volume",
    "beta",
    "drawdown",
    "shock",
)
AVAILABILITY_FEATURES = {
    "current_filing_fundamentals_available",
    "current_filing_sentiment_available",
    "fund_snapshot_is_current_event",
}
EVENT_CONTEXT_FEATURES = {
    "days_since_prior_event",
    "days_since_prior_same_event_type",
}
REDUCED_VOL_CLUSTER_REMOVALS = {
    "realized_vol_21d",
    "realized_vol_63d",
    "vol_ratio_21d_63d",
    "volume_ratio_20d",
    "log_volume",
    "abnormal_volume_flag",
    "return_zscore_21d",
    "abs_return_shock_1d",
    "overnight_gap_1d",
}
HIGH_RISK_SINGLE_REMOVALS = [
    "beta_63d_to_sector",
    "sec_positive_prob",
    "cash_ratio",
    "log_volume",
]
COMPACT_REGIMES = {"top_10_shap_only", "top_15_shap_only", "top_20_shap_only"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark pruning regimes for event_panel_v2.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--benchmark-csv", default=str(DEFAULT_BENCHMARK_CSV_PATH))
    parser.add_argument("--shap-csv", default=str(DEFAULT_SHAP_CSV_PATH))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV_PATH))
    parser.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD_PATH))
    return parser.parse_args()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def is_price_volume_feature(feature: str) -> bool:
    return any(token in feature for token in PRICE_VOLUME_TOKENS)


def is_sentiment_feature(feature: str) -> bool:
    return feature.startswith("sec_")


def is_availability_feature(feature: str) -> bool:
    return feature in AVAILABILITY_FEATURES or feature.endswith("_available") or feature.endswith("_is_current_event")


def is_event_context_feature(feature: str) -> bool:
    return feature in EVENT_CONTEXT_FEATURES


def load_selected_model_row(path: Path) -> pd.Series:
    benchmark_df = pd.read_csv(path)
    selected = benchmark_df.loc[benchmark_df["is_selected_primary_model"] == True]  # noqa: E712
    if selected.empty:
        raise ValueError(f"No selected primary model row found in benchmark CSV: {path}")
    return selected.iloc[0]


def build_regime_map(usable_features: list[str], shap_path: Path) -> dict[str, list[str]]:
    shap_df = pd.read_csv(shap_path)
    ordered_top = [feature for feature in shap_df["feature"].tolist() if feature in set(usable_features)]
    price_volume_features = [feature for feature in usable_features if is_price_volume_feature(feature)]
    sentiment_features = [feature for feature in usable_features if is_sentiment_feature(feature)]
    availability_features = [feature for feature in usable_features if is_availability_feature(feature)]
    event_context_features = [feature for feature in usable_features if is_event_context_feature(feature)]
    fundamentals = [
        feature
        for feature in usable_features
        if feature not in set(price_volume_features + sentiment_features + availability_features + event_context_features)
    ]

    regimes: dict[str, list[str]] = {
        "full_49_baseline": list(usable_features),
        "no_availability_flags": [feature for feature in usable_features if feature not in set(availability_features)],
        "top_10_shap_only": ordered_top[:10],
        "top_15_shap_only": ordered_top[:15],
        "top_20_shap_only": ordered_top[:20],
        "price_volume_only": price_volume_features,
        "price_volume_plus_sentiment": sorted(set(price_volume_features + sentiment_features), key=usable_features.index),
        "price_volume_plus_fundamentals": sorted(set(price_volume_features + fundamentals), key=usable_features.index),
        "reduced_vol_cluster": [
            feature for feature in usable_features if feature not in REDUCED_VOL_CLUSTER_REMOVALS
        ],
    }
    for feature in HIGH_RISK_SINGLE_REMOVALS:
        if feature in usable_features:
            regimes[f"drop_{feature}"] = [column for column in usable_features if column != feature]

    return {label: features for label, features in regimes.items() if features}


def driver_summary(fold_top_feature_lists: list[list[str]]) -> str:
    if not fold_top_feature_lists:
        return ""
    counts = Counter()
    for feature_list in fold_top_feature_lists:
        counts.update(feature_list)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return json.dumps(dict(ordered[:5]))


def evaluate_regime(
    regime_name: str,
    regime_features: list[str],
    panel_df: pd.DataFrame,
    variant,
    explicit_exclusions: list[str],
    holdout_start: str,
    n_splits: int,
    embargo_days: int,
    min_train_dates: int,
    model_name: str,
    max_missingness_pct: float = 20.0,
) -> dict:
    split_payload = make_event_v1_splits(
        df=panel_df,
        date_col="date",
        horizon_days=variant.horizon_days,
        n_splits=n_splits,
        embargo_days=embargo_days,
        holdout_start=holdout_start,
        min_train_dates=min_train_dates,
    )
    global_candidates = [column for column in regime_features if column not in explicit_exclusions]
    kept_global, auto_all_missing, auto_constant = compute_global_feature_exclusions(
        panel_df,
        global_candidates,
        holdout_start=holdout_start,
    )

    fold_metrics = []
    fold_top_driver_lists: list[list[str]] = []
    last_usable = []
    last_missingness: dict[str, float] = {}
    last_dropped_missing: list[str] = []
    last_dropped_constant: list[str] = []

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
            max_missingness_pct=max_missingness_pct,
        )
        clipped_train, clipped_validation = clip_outliers(train_active, validation_active, usable_features)
        fitted_model, _ = fit_model(
            model_name,
            clipped_train[usable_features],
            clipped_train["target"].astype(int),
        )
        y_prob = fitted_model.predict_proba(clipped_validation[usable_features])[:, 1]
        fold_metrics.append(evaluate_extended(clipped_validation, y_prob, threshold=0.5))

        importances = getattr(fitted_model.named_steps["model"], "feature_importances_", None)
        if importances is not None:
            top_fold_features = [
                feature
                for _, feature in sorted(
                    zip(np.asarray(importances, dtype=float), usable_features),
                    key=lambda pair: (-pair[0], pair[1]),
                )[:3]
            ]
            fold_top_driver_lists.append(top_fold_features)

        last_usable = usable_features
        last_missingness = missingness_by_feature
        last_dropped_missing = dropped_missing
        last_dropped_constant = dropped_constant

    if not fold_metrics:
        raise ValueError(f"No valid CV folds produced for regime: {regime_name}")

    holdout_train_full = panel_df.iloc[split_payload["holdout"]["train_indices"]].copy()
    holdout_full = panel_df.iloc[split_payload["holdout"]["holdout_indices"]].copy()
    holdout_train_active, _ = apply_variant_label_mode(holdout_train_full, variant)
    holdout_active, _ = apply_variant_label_mode(holdout_full, variant)
    holdout_usable, _, holdout_dropped_missing, holdout_dropped_constant = select_usable_features(
        holdout_train_active,
        kept_global,
        max_missingness_pct=max_missingness_pct,
    )
    clipped_train, clipped_holdout = clip_outliers(holdout_train_active, holdout_active, holdout_usable)
    fitted_model, backend = fit_model(
        model_name,
        clipped_train[holdout_usable],
        clipped_train["target"].astype(int),
    )
    holdout_prob = fitted_model.predict_proba(clipped_holdout[holdout_usable])[:, 1]
    holdout_metrics = evaluate_extended(clipped_holdout, holdout_prob, threshold=0.5)
    cv_summary = summarize_metric_dicts(fold_metrics)
    worst_fold_auc = min(
        (
            float(metrics["auc_roc"])
            for metrics in fold_metrics
            if metrics.get("auc_roc") is not None
        ),
        default=None,
    )

    holdout_importances = getattr(fitted_model.named_steps["model"], "feature_importances_", None)
    holdout_top_drivers = []
    if holdout_importances is not None:
        holdout_top_drivers = [
            feature
            for _, feature in sorted(
                zip(np.asarray(holdout_importances, dtype=float), holdout_usable),
                key=lambda pair: (-pair[0], pair[1]),
            )[:5]
        ]

    return {
        "regime_name": regime_name,
        "regime_group": (
            "compact"
            if regime_name in COMPACT_REGIMES
            else "hygiene"
            if regime_name in {"no_availability_flags", "reduced_vol_cluster"}
            else "stress_test"
            if regime_name.startswith("drop_")
            else "block_mix"
        ),
        "model_name": model_name,
        "cv_auc_mean": cv_summary.get("auc_roc_mean"),
        "cv_auc_std": cv_summary.get("auc_roc_std"),
        "worst_fold_auc": worst_fold_auc,
        "cv_log_loss_mean": cv_summary.get("log_loss_mean"),
        "cv_log_loss_std": cv_summary.get("log_loss_std"),
        "cv_precision_mean": cv_summary.get("precision_mean"),
        "cv_recall_mean": cv_summary.get("recall_mean"),
        "cv_rank_ic_mean": cv_summary.get("rank_ic_spearman_mean"),
        "holdout_auc": holdout_metrics.get("auc_roc"),
        "holdout_log_loss": holdout_metrics.get("log_loss"),
        "holdout_precision": holdout_metrics.get("precision"),
        "holdout_recall": holdout_metrics.get("recall"),
        "holdout_rank_ic": holdout_metrics.get("rank_ic_spearman"),
        "feature_count": int(len(holdout_usable)),
        "configured_feature_count": int(len(regime_features)),
        "configured_features": json.dumps(regime_features),
        "holdout_usable_features": json.dumps(holdout_usable),
        "auto_all_missing_exclusions": json.dumps(auto_all_missing),
        "auto_constant_exclusions": json.dumps(auto_constant),
        "fold_missingness_exclusions_last_fold": json.dumps(sorted(set(last_dropped_missing))),
        "fold_constant_exclusions_last_fold": json.dumps(sorted(set(last_dropped_constant))),
        "holdout_missingness_exclusions": json.dumps(sorted(set(holdout_dropped_missing))),
        "holdout_constant_exclusions": json.dumps(sorted(set(holdout_dropped_constant))),
        "train_missingness_by_feature_pct_last_fold": json.dumps(last_missingness),
        "cv_fold_count": cv_summary["fold_count"],
        "holdout_row_count": holdout_metrics["row_count"],
        "xgboost_backend": backend,
        "cv_fold_top_driver_counts": driver_summary(fold_top_driver_lists),
        "holdout_top_drivers": json.dumps(holdout_top_drivers),
    }


def classify_regime(row: pd.Series, baseline: pd.Series) -> str:
    if str(row["regime_name"]) == "full_49_baseline":
        return "baseline_reference"
    nonworse_cv_auc = float(row["cv_auc_mean"]) >= float(baseline["cv_auc_mean"])
    nonworse_cv_log_loss = float(row["cv_log_loss_mean"]) <= float(baseline["cv_log_loss_mean"])
    better_holdout = (
        float(row["holdout_auc"]) >= float(baseline["holdout_auc"])
        or float(row["holdout_log_loss"]) <= float(baseline["holdout_log_loss"])
    )
    if nonworse_cv_auc and nonworse_cv_log_loss and better_holdout:
        return "candidate_winner"
    holdout_only = better_holdout and not (nonworse_cv_auc and nonworse_cv_log_loss)
    if holdout_only:
        return "interesting_not_promotable"
    return "not_promotable"


def build_markdown_report(result_df: pd.DataFrame, baseline_row: pd.Series, locked_row: pd.Series) -> str:
    compact_df = result_df.loc[result_df["regime_group"] == "compact"].sort_values(
        ["holdout_auc", "cv_auc_mean", "cv_log_loss_mean"],
        ascending=[False, False, True],
    )
    best_compact = compact_df.iloc[0] if not compact_df.empty else None
    candidate_rows = result_df.loc[result_df["promotion_status"] == "candidate_winner"].copy()

    lines = [
        "# Event Panel V2 Pruning Benchmark",
        "",
        "## Summary",
        "",
        f"- Locked baseline reference from primary benchmark: CV AUC `{format_metric(locked_row['cv_auc_mean'])}`, holdout AUC `{format_metric(locked_row['holdout_auc'])}`.",
        f"- Reproduced full_49 baseline in pruning lane: CV AUC `{format_metric(baseline_row['cv_auc_mean'])}`, holdout AUC `{format_metric(baseline_row['holdout_auc'])}`.",
        "- Conservative rule: a regime is promotable only if it is non-worse on CV AUC, non-worse on CV log loss, and at least as good on holdout AUC or holdout log loss.",
        "",
        "## Regime Comparison",
        "",
        "| Regime | Group | Features | CV AUC | CV AUC Std | CV Log Loss | CV Log Loss Std | Holdout AUC | Holdout Log Loss | Holdout Precision | Holdout Recall | Holdout Rank IC | Promotion |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in result_df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["regime_name"]),
                    str(row["regime_group"]),
                    str(int(row["feature_count"])),
                    format_metric(row["cv_auc_mean"]),
                    format_metric(row["cv_auc_std"]),
                    format_metric(row["cv_log_loss_mean"]),
                    format_metric(row["cv_log_loss_std"]),
                    format_metric(row["holdout_auc"]),
                    format_metric(row["holdout_log_loss"]),
                    format_metric(row["holdout_precision"]),
                    format_metric(row["holdout_recall"]),
                    format_metric(row["holdout_rank_ic"]),
                    str(row["promotion_status"]),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Readout", ""])
    if candidate_rows.empty:
        lines.append("- No pruning regime met the conservative promotion rule in this run.")
    else:
        winners = ", ".join(candidate_rows["regime_name"].tolist())
        lines.append(f"- Candidate winner regimes under the conservative rule: `{winners}`.")
    if best_compact is not None:
        lines.append(
            f"- Best compact regime by holdout/CV ordering was `{best_compact['regime_name']}` with `{int(best_compact['feature_count'])}` features, CV AUC `{format_metric(best_compact['cv_auc_mean'])}`, and holdout AUC `{format_metric(best_compact['holdout_auc'])}`."
        )
        lines.append(
            f"- Compact regime fold-driver stability snapshot: `{best_compact['cv_fold_top_driver_counts'] or '{}'};` holdout top drivers `{best_compact['holdout_top_drivers']}`."
        )

    no_availability = result_df.loc[result_df["regime_name"] == "no_availability_flags"]
    if not no_availability.empty:
        row = no_availability.iloc[0]
        lines.append(
            f"- Availability-flag removal moved holdout AUC to `{format_metric(row['holdout_auc'])}` and holdout log loss to `{format_metric(row['holdout_log_loss'])}`."
        )
    reduced_vol = result_df.loc[result_df["regime_name"] == "reduced_vol_cluster"]
    if not reduced_vol.empty:
        row = reduced_vol.iloc[0]
        lines.append(
            f"- Reduced volatility/volume cluster regime reached CV AUC `{format_metric(row['cv_auc_mean'])}` and holdout AUC `{format_metric(row['holdout_auc'])}`."
        )
    beta_drop = result_df.loc[result_df["regime_name"] == "drop_beta_63d_to_sector"]
    if not beta_drop.empty:
        row = beta_drop.iloc[0]
        lines.append(
            f"- Dropping `beta_63d_to_sector` changed holdout AUC from `{format_metric(baseline_row['holdout_auc'])}` to `{format_metric(row['holdout_auc'])}`, confirming whether the feature remains fragile-important after retraining."
        )

    lines.extend(
        [
            "",
            "## WRDS Checkpoint",
            "",
            "- Treat the current full_49 baseline as locked until the same matrix is rerun with WRDS-added feature sets.",
            "- If a compact regime is promising now, keep it as a lean benchmark candidate rather than replacing the default model immediately.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    output_csv_path = Path(args.output_csv)
    output_md_path = Path(args.output_md)
    ensure_parent_dir(output_csv_path)
    ensure_parent_dir(output_md_path)

    locked_row = load_selected_model_row(Path(args.benchmark_csv))
    model_name = str(locked_row["model_name"])
    variant = build_variant(config)
    labeled_panel_df = load_labeled_panel(config)
    candidate_features = resolve_candidate_features(labeled_panel_df, config)
    explicit_exclusions = list(config["feature_exclusions"]["explicit"])

    holdout_start = str(config["holdout"]["start"])
    global_candidates = [column for column in candidate_features if column not in explicit_exclusions]
    kept_global, _, _ = compute_global_feature_exclusions(
        labeled_panel_df,
        global_candidates,
        holdout_start=holdout_start,
    )
    split_payload = make_event_v1_splits(
        df=labeled_panel_df,
        date_col="date",
        horizon_days=variant.horizon_days,
        n_splits=int(config["cv"]["n_splits"]),
        embargo_days=int(config["cv"]["embargo_days"]),
        holdout_start=holdout_start,
        min_train_dates=int(config["cv"]["min_train_dates"]),
    )
    holdout_train_full = labeled_panel_df.iloc[split_payload["holdout"]["train_indices"]].copy()
    holdout_train_active, _ = apply_variant_label_mode(holdout_train_full, variant)
    configured_max_missingness_pct = resolve_max_missingness_pct(config.get("feature_exclusions"))
    full_usable_features, _, _, _ = select_usable_features(
        holdout_train_active,
        kept_global,
        max_missingness_pct=configured_max_missingness_pct,
    )

    regime_map = build_regime_map(full_usable_features, Path(args.shap_csv))
    rows = []
    for regime_name, features in regime_map.items():
        rows.append(
            evaluate_regime(
                regime_name=regime_name,
                regime_features=features,
                panel_df=labeled_panel_df,
                variant=variant,
                explicit_exclusions=explicit_exclusions,
                holdout_start=holdout_start,
                n_splits=int(config["cv"]["n_splits"]),
                embargo_days=int(config["cv"]["embargo_days"]),
                min_train_dates=int(config["cv"]["min_train_dates"]),
                model_name=model_name,
                max_missingness_pct=configured_max_missingness_pct,
            )
        )

    result_df = pd.DataFrame(rows).sort_values("regime_name").reset_index(drop=True)
    baseline_row = result_df.loc[result_df["regime_name"] == "full_49_baseline"].iloc[0]
    result_df["cv_auc_delta_vs_baseline"] = result_df["cv_auc_mean"] - float(baseline_row["cv_auc_mean"])
    result_df["cv_log_loss_delta_vs_baseline"] = result_df["cv_log_loss_mean"] - float(baseline_row["cv_log_loss_mean"])
    result_df["holdout_auc_delta_vs_baseline"] = result_df["holdout_auc"] - float(baseline_row["holdout_auc"])
    result_df["holdout_log_loss_delta_vs_baseline"] = (
        result_df["holdout_log_loss"] - float(baseline_row["holdout_log_loss"])
    )
    result_df["promotion_status"] = result_df.apply(classify_regime, axis=1, baseline=baseline_row)
    result_df["matches_locked_primary_benchmark"] = result_df["regime_name"] == "full_49_baseline"

    result_df.to_csv(output_csv_path, index=False)
    markdown = build_markdown_report(result_df, baseline_row, locked_row)
    output_md_path.write_text(markdown, encoding="utf-8")

    print(f"Selected model: {model_name}")
    print(f"Wrote pruning CSV to: {output_csv_path}")
    print(f"Wrote pruning Markdown to: {output_md_path}")


if __name__ == "__main__":
    main()
