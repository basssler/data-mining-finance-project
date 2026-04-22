"""Run SHAP-driven ablation and decile diagnostics for the primary event panel model."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd
import yaml

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
    candidate_feature_columns,
    clip_outliers,
    compute_global_feature_exclusions,
    fit_model,
    format_metric,
    load_event_panel,
    resolve_max_missingness_pct,
    safe_rank_ic,
    select_usable_features,
)
from src.labels_event_v1 import load_price_data, normalize_price_data
from src.train_event_panel_v2 import load_config, resolve_candidate_features
from src.validation_event_v1 import make_event_v1_splits

DEFAULT_CONFIG_PATH = Path("configs") / "event_panel_v2_primary.yaml"
DEFAULT_BENCHMARK_CSV_PATH = Path("reports") / "results" / "event_panel_v2_primary_benchmark.csv"
DEFAULT_SHAP_CSV_PATH = Path("reports") / "results" / "event_panel_v2_primary_shap_importance.csv"
DEFAULT_ABLATION_CSV_PATH = Path("reports") / "results" / "event_panel_v2_primary_feature_ablation.csv"
DEFAULT_DECILE_CSV_PATH = Path("reports") / "results" / "event_panel_v2_primary_feature_deciles.csv"
DEFAULT_REPORT_MD_PATH = Path("reports") / "results" / "event_panel_v2_primary_feature_diagnostics.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SHAP-guided diagnostics for event_panel_v2.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--benchmark-csv", default=str(DEFAULT_BENCHMARK_CSV_PATH))
    parser.add_argument("--shap-csv", default=str(DEFAULT_SHAP_CSV_PATH))
    parser.add_argument("--ablation-csv", default=str(DEFAULT_ABLATION_CSV_PATH))
    parser.add_argument("--decile-csv", default=str(DEFAULT_DECILE_CSV_PATH))
    parser.add_argument("--report-md", default=str(DEFAULT_REPORT_MD_PATH))
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_selected_model_name(path: Path) -> str:
    benchmark_df = pd.read_csv(path)
    selected = benchmark_df.loc[benchmark_df["is_selected_primary_model"] == True]  # noqa: E712
    if selected.empty:
        raise ValueError(f"No selected primary model row found in benchmark CSV: {path}")
    return str(selected.iloc[0]["model_name"])


def build_variant(config: dict) -> VariantSpec:
    return VariantSpec(
        variant_name=str(config["label"]["variant_name"]),
        horizon_days=int(config["label"]["horizon_days"]),
        label_mode=str(config["label"]["mode"]),
    )


def load_labeled_panel(config: dict) -> pd.DataFrame:
    panel_path = Path(config["panel"]["path"])
    panel_df = load_event_panel(panel_path)
    prices_path = Path(config.get("prices", {}).get("path", str(PRICE_INPUT_PATH)))
    prices_df = normalize_price_data(load_price_data(prices_path))
    variant = build_variant(config)
    label_df = build_daily_label_table(
        prices_df,
        horizon_days=variant.horizon_days,
        benchmark_mode=str(config["label"]["benchmark_mode"]),
    )
    return attach_labels_to_event_panel(panel_df, label_df)


def get_holdout_training_frames(
    panel_df: pd.DataFrame,
    config: dict,
    variant: VariantSpec,
    candidate_features: list[str],
    explicit_exclusions: list[str],
) -> dict:
    holdout_start = str(config["holdout"]["start"])
    split_payload = make_event_v1_splits(
        df=panel_df,
        date_col="date",
        horizon_days=variant.horizon_days,
        n_splits=int(config["cv"]["n_splits"]),
        embargo_days=int(config["cv"]["embargo_days"]),
        holdout_start=holdout_start,
        min_train_dates=int(config["cv"]["min_train_dates"]),
    )
    global_candidates = [column for column in candidate_features if column not in explicit_exclusions]
    kept_global, auto_all_missing, auto_constant = compute_global_feature_exclusions(
        panel_df,
        global_candidates,
        holdout_start=holdout_start,
    )

    holdout_train_full = panel_df.iloc[split_payload["holdout"]["train_indices"]].copy()
    holdout_full = panel_df.iloc[split_payload["holdout"]["holdout_indices"]].copy()
    holdout_train_active, _ = apply_variant_label_mode(holdout_train_full, variant)
    holdout_active, _ = apply_variant_label_mode(holdout_full, variant)
    holdout_usable, missingness_by_feature, dropped_missing, dropped_constant = select_usable_features(
        holdout_train_active,
        kept_global,
        max_missingness_pct=resolve_max_missingness_pct(config.get("feature_exclusions")),
    )
    clipped_train, clipped_holdout = clip_outliers(holdout_train_active, holdout_active, holdout_usable)
    return {
        "train_df": clipped_train,
        "holdout_df": clipped_holdout,
        "usable_features": holdout_usable,
        "kept_global": kept_global,
        "auto_all_missing": auto_all_missing,
        "auto_constant": auto_constant,
        "missingness_by_feature": missingness_by_feature,
        "dropped_missing": dropped_missing,
        "dropped_constant": dropped_constant,
    }


def evaluate_feature_set(
    model_name: str,
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    features: list[str],
    label: str,
) -> dict:
    if not features:
        raise ValueError(f"No features remained for ablation: {label}")
    fitted_model, backend = fit_model(
        model_name,
        train_df[features],
        train_df["target"].astype(int),
    )
    holdout_prob = fitted_model.predict_proba(holdout_df[features])[:, 1]
    metrics = evaluate_classification_run(
        y_true=holdout_df["target"].astype(int),
        y_prob=holdout_prob,
        threshold=0.5,
    )
    metrics["rank_ic_spearman"] = safe_rank_ic(holdout_df["excess_forward_return"], holdout_prob)
    metrics["scenario"] = label
    metrics["model_name"] = model_name
    metrics["feature_count"] = int(len(features))
    metrics["features"] = ", ".join(features)
    metrics["xgboost_backend"] = backend
    return metrics


def build_ablation_plan(top_features: list[str], usable_features: list[str]) -> list[tuple[str, list[str]]]:
    usable_set = set(usable_features)
    ordered_top = [feature for feature in top_features if feature in usable_set]
    price_vol_tokens = (
        "return",
        "vol",
        "gap",
        "volume",
        "beta",
        "drawdown",
        "shock",
    )
    sentiment_features = [feature for feature in usable_features if feature.startswith("sec_")]
    price_vol_features = [
        feature for feature in usable_features if any(token in feature for token in price_vol_tokens)
    ]
    availability_features = [
        feature
        for feature in usable_features
        if feature.endswith("_available") or feature.endswith("_is_current_event")
    ]
    plan: list[tuple[str, list[str]]] = [("baseline", list(usable_features))]
    for feature in ordered_top[: min(5, len(ordered_top))]:
        plan.append((f"drop_{feature}", [col for col in usable_features if col != feature]))
    if ordered_top[:3]:
        excluded = set(ordered_top[:3])
        plan.append(("drop_top_3_shap", [col for col in usable_features if col not in excluded]))
    if ordered_top[:5]:
        excluded = set(ordered_top[:5])
        plan.append(("drop_top_5_shap", [col for col in usable_features if col not in excluded]))
    if ordered_top:
        plan.append(("top_10_shap_only", [col for col in ordered_top[:10]]))
    if price_vol_features:
        plan.append(("price_volume_only", price_vol_features))
    if sentiment_features:
        plan.append(("sentiment_only", sentiment_features))
    if availability_features:
        plan.append(("no_availability_flags", [col for col in usable_features if col not in availability_features]))

    deduped: list[tuple[str, list[str]]] = []
    seen: set[tuple[str, ...]] = set()
    for label, features in plan:
        feature_tuple = tuple(features)
        if feature_tuple and feature_tuple not in seen:
            deduped.append((label, features))
            seen.add(feature_tuple)
    return deduped


def compute_decile_rows(
    holdout_df: pd.DataFrame,
    top_features: list[str],
) -> list[dict]:
    rows: list[dict] = []
    for feature in top_features:
        if feature not in holdout_df.columns:
            continue
        feature_series = pd.to_numeric(holdout_df[feature], errors="coerce")
        valid_mask = feature_series.notna() & holdout_df["target"].notna() & holdout_df["pred_prob"].notna()
        if int(valid_mask.sum()) < 20:
            continue
        valid_df = holdout_df.loc[valid_mask, ["target", "pred_prob", "excess_forward_return"]].copy()
        valid_df["feature_value"] = feature_series.loc[valid_mask]
        try:
            valid_df["feature_decile"] = pd.qcut(valid_df["feature_value"], q=10, labels=False, duplicates="drop")
        except ValueError:
            continue
        grouped = valid_df.groupby("feature_decile", dropna=True)
        for decile, group in grouped:
            rows.append(
                {
                    "feature": feature,
                    "feature_decile": int(decile) + 1,
                    "row_count": int(len(group)),
                    "feature_value_min": float(group["feature_value"].min()),
                    "feature_value_max": float(group["feature_value"].max()),
                    "mean_feature_value": float(group["feature_value"].mean()),
                    "actual_positive_rate": float(group["target"].mean()),
                    "mean_pred_prob": float(group["pred_prob"].mean()),
                    "mean_excess_forward_return": float(group["excess_forward_return"].mean()),
                }
            )
    return rows


def build_markdown_report(
    model_name: str,
    top_features: list[str],
    ablation_df: pd.DataFrame,
    decile_df: pd.DataFrame,
) -> str:
    baseline_row = ablation_df.loc[ablation_df["scenario"] == "baseline"].iloc[0]
    direct_drop_df = ablation_df.loc[ablation_df["scenario"].str.startswith("drop_")].copy()
    direct_drop_df["auc_drop"] = baseline_row["auc_roc"] - direct_drop_df["auc_roc"]
    most_damaging_drop = direct_drop_df.sort_values(["auc_drop", "log_loss"], ascending=[False, True]).iloc[0]
    best_compact_df = ablation_df.loc[ablation_df["scenario"].isin(["top_10_shap_only", "price_volume_only"])].copy()
    best_compact = (
        best_compact_df.sort_values(["auc_roc", "log_loss"], ascending=[False, True]).iloc[0]
        if not best_compact_df.empty
        else None
    )
    lines = [
        "# Event Panel V2 Primary Feature Diagnostics",
        "",
        "## Scope",
        "",
        f"- Selected model under test: `{model_name}`",
        f"- Baseline holdout AUC: `{format_metric(baseline_row['auc_roc'])}`",
        f"- Baseline holdout log loss: `{format_metric(baseline_row['log_loss'])}`",
        f"- Baseline holdout rank IC: `{format_metric(baseline_row['rank_ic_spearman'])}`",
        f"- SHAP anchor features used for diagnostics: `{', '.join(top_features[:10])}`",
        "",
        "## Feature Ablation",
        "",
        "| Scenario | Feature Count | Holdout AUC | Holdout Log Loss | Precision | Recall | Rank IC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in ablation_df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["scenario"]),
                    str(int(row["feature_count"])),
                    format_metric(row["auc_roc"]),
                    format_metric(row["log_loss"]),
                    format_metric(row["precision"]),
                    format_metric(row["recall"]),
                    format_metric(row["rank_ic_spearman"]),
                ]
            )
            + " |"
        )

    strongest_drop = (
        ablation_df.loc[ablation_df["scenario"] != "baseline"]
        .assign(auc_drop=lambda df: baseline_row["auc_roc"] - df["auc_roc"])
        .sort_values(["auc_drop", "log_loss"], ascending=[False, True])
        .iloc[0]
    )
    lines.extend(
        [
            "",
            "## Key Read",
            "",
            f"- Strongest direct removal signal came from `{most_damaging_drop['scenario']}`, moving holdout AUC from `{format_metric(baseline_row['auc_roc'])}` to `{format_metric(most_damaging_drop['auc_roc'])}`.",
        ]
    )
    if best_compact is not None:
        lines.append(
            f"- Most efficient reduced feature set in this run was `{best_compact['scenario']}`, reaching holdout AUC `{format_metric(best_compact['auc_roc'])}` with `{int(best_compact['feature_count'])}` features."
        )

    lines.extend(
        [
            "",
            "## Top-Feature Deciles",
            "",
            "| Feature | Lowest Decile Hit Rate | Highest Decile Hit Rate | Lowest Decile Mean Pred Prob | Highest Decile Mean Pred Prob |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for feature in top_features[:10]:
        feature_rows = decile_df.loc[decile_df["feature"] == feature].sort_values("feature_decile")
        if feature_rows.empty:
            continue
        low = feature_rows.iloc[0]
        high = feature_rows.iloc[-1]
        lines.append(
            "| "
            + " | ".join(
                [
                    feature,
                    format_metric(low["actual_positive_rate"]),
                    format_metric(high["actual_positive_rate"]),
                    format_metric(low["mean_pred_prob"]),
                    format_metric(high["mean_pred_prob"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    benchmark_csv_path = Path(args.benchmark_csv)
    shap_csv_path = Path(args.shap_csv)
    ablation_csv_path = Path(args.ablation_csv)
    decile_csv_path = Path(args.decile_csv)
    report_md_path = Path(args.report_md)
    for path in [ablation_csv_path, decile_csv_path, report_md_path]:
        ensure_parent_dir(path)

    model_name = load_selected_model_name(benchmark_csv_path)
    shap_df = pd.read_csv(shap_csv_path)
    top_features = shap_df["feature"].head(int(args.top_k)).tolist()
    variant = build_variant(config)
    labeled_panel_df = load_labeled_panel(config)
    candidate_features = resolve_candidate_features(labeled_panel_df, config)
    explicit_exclusions = list(config["feature_exclusions"]["explicit"])
    holdout_payload = get_holdout_training_frames(
        labeled_panel_df,
        config,
        variant,
        candidate_features,
        explicit_exclusions,
    )

    ablation_rows = []
    plan = build_ablation_plan(top_features, holdout_payload["usable_features"])
    for label, features in plan:
        ablation_rows.append(
            evaluate_feature_set(
                model_name,
                holdout_payload["train_df"],
                holdout_payload["holdout_df"],
                features,
                label,
            )
        )
    ablation_df = pd.DataFrame(ablation_rows).sort_values("scenario").reset_index(drop=True)
    baseline_row = ablation_df.loc[ablation_df["scenario"] == "baseline"].iloc[0]
    ablation_df["auc_delta_vs_baseline"] = ablation_df["auc_roc"] - baseline_row["auc_roc"]
    ablation_df["log_loss_delta_vs_baseline"] = ablation_df["log_loss"] - baseline_row["log_loss"]
    ablation_df.to_csv(ablation_csv_path, index=False)

    fitted_model, _ = fit_model(
        model_name,
        holdout_payload["train_df"][holdout_payload["usable_features"]],
        holdout_payload["train_df"]["target"].astype(int),
    )
    holdout_scored = holdout_payload["holdout_df"].copy()
    holdout_scored["pred_prob"] = fitted_model.predict_proba(
        holdout_scored[holdout_payload["usable_features"]]
    )[:, 1]
    decile_rows = compute_decile_rows(holdout_scored, top_features)
    decile_df = pd.DataFrame(decile_rows).sort_values(["feature", "feature_decile"]).reset_index(drop=True)
    decile_df.to_csv(decile_csv_path, index=False)

    markdown = build_markdown_report(model_name, top_features, ablation_df, decile_df)
    report_md_path.write_text(markdown, encoding="utf-8")

    print(f"Selected model: {model_name}")
    print(f"Wrote ablation CSV to: {ablation_csv_path}")
    print(f"Wrote decile CSV to: {decile_csv_path}")
    print(f"Wrote Markdown diagnostics to: {report_md_path}")


if __name__ == "__main__":
    main()
