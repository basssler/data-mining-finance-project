from __future__ import annotations

import math
from pathlib import Path
import sys

import pandas as pd
import yaml
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train_event_panel_v2 import load_config, run_model_matrix, save_validation_artifacts
from src.label_comparison_event_v2 import VariantSpec
from src.train_event_panel_v2 import (
    attach_labels_to_event_panel,
    build_daily_label_table,
    build_markdown_report,
    build_validation_artifacts,
    load_event_panel,
    load_price_data,
    normalize_price_data,
    resolve_report_metadata,
    resolve_concentration_output_path,
    resolve_candidate_features,
    resolve_promotion_strategy,
    resolve_threshold,
    set_random_seeds,
)


OUT_DIR = PROJECT_ROOT / "outputs" / "quarterly" / "modeling" / "capitaliq_layer_eval_all_models"
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"

LAYERS = [
    ("L1", "Layer 1", PROJECT_ROOT / "configs" / "quarterly" / "capitaliq_ladder_core.yaml"),
    ("L1_L2", "Layer 1+2", PROJECT_ROOT / "configs" / "quarterly" / "capitaliq_ladder_core_plus_market.yaml"),
    (
        "L1_L2_L3",
        "Layer 1+2+3",
        PROJECT_ROOT / "configs" / "quarterly" / "capitaliq_ladder_sector_adjusted_sentiment.yaml",
    ),
]

MODELS = [
    "logistic_regression",
    "random_forest",
    "hist_gradient_boosting",
    "xgboost",
    "catboost",
]

METRICS = [
    ("auc", "AUROC", "higher"),
    ("f1", "F1-Score", "higher"),
    ("log_loss", "Log Loss", "lower"),
]


def _prepare_config(config_path: Path, layer_slug: str) -> dict:
    config = load_config(config_path)
    config["models"] = MODELS
    config["outputs"]["csv"] = str(OUT_DIR / f"{layer_slug}.csv")
    config["outputs"]["markdown"] = str(OUT_DIR / f"{layer_slug}.md")
    config["outputs"]["shap_plot"] = str(OUT_DIR / f"{layer_slug}_shap_summary.png")
    config["outputs"]["shap_csv"] = str(OUT_DIR / f"{layer_slug}_shap_importance.csv")
    config["outputs"]["validation_dir"] = str(OUT_DIR / layer_slug / "validation")
    config["outputs"]["concentration_csv"] = str(OUT_DIR / f"{layer_slug}_concentration.csv")
    return config


def _run_layer(layer_slug: str, config_path: Path) -> pd.DataFrame:
    config = _prepare_config(config_path, layer_slug)
    set_random_seeds(config.get("random_seed", {}))
    panel_df = load_event_panel(Path(config["panel"]["path"]))
    prices_df = normalize_price_data(load_price_data(Path(config["prices"]["path"])))
    label_df = build_daily_label_table(
        prices_df,
        horizon_days=int(config["label"]["horizon_days"]),
        benchmark_mode=str(config["label"].get("benchmark_mode", "sector_equal_weight_ex_self")),
    )
    labeled_panel_df = attach_labels_to_event_panel(panel_df, label_df)
    variant = VariantSpec(
        str(config["label"]["variant_name"]),
        int(config["label"]["horizon_days"]),
        str(config["label"]["mode"]),
        threshold=float(config["label"].get("threshold", 0.015)),
        quantile=float(config["label"].get("quantile", 0.80)),
    )
    result_df, summary = run_model_matrix(
        panel_df=labeled_panel_df,
        variant=variant,
        model_names=list(config["models"]),
        candidate_features=resolve_candidate_features(labeled_panel_df, config),
        explicit_exclusions=list(config["feature_exclusions"]["explicit"]),
        holdout_start=str(config["holdout"]["start"]),
        n_splits=int(config["cv"]["n_splits"]),
        embargo_days=int(config["cv"]["embargo_days"]),
        min_train_dates=int(config["cv"].get("min_train_dates", 252)),
        threshold=resolve_threshold(config),
        panel_name=str(config["panel"]["name"]),
        max_missingness_pct=float(config["feature_exclusions"].get("max_missingness_pct", 20.0)),
        promotion_strategy=resolve_promotion_strategy(config),
        config=config,
    )
    csv_path = Path(config["outputs"]["csv"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(csv_path, index=False)
    Path(config["outputs"]["markdown"]).write_text(
        build_markdown_report(
            result_df=result_df,
            summary=summary,
            report_metadata=resolve_report_metadata(config),
            old_baseline=summary.get("old_baseline"),
        ),
        encoding="utf-8",
    )
    concentration_path = resolve_concentration_output_path(config)
    summary["concentration_df"].to_csv(concentration_path, index=False)
    validation_dir = Path(config["outputs"]["validation_dir"])
    fold_map_df, purge_audit_df = build_validation_artifacts(labeled_panel_df, summary["split_payload"])
    save_validation_artifacts(validation_dir, fold_map_df, summary["fold_summary_df"], purge_audit_df)
    fold_df = summary["fold_summary_df"].copy()
    fold_df.insert(0, "layer", layer_slug)
    return fold_df


def _paired_tests(fold_df: pd.DataFrame) -> pd.DataFrame:
    comparisons = [("L1", "L1_L2"), ("L1_L2", "L1_L2_L3"), ("L1", "L1_L2_L3")]
    rows = []
    validation = fold_df[fold_df["evaluation_role"] == "validation"].copy()
    for model in MODELS:
        model_df = validation[validation["model_name"] == model]
        for left, right in comparisons:
            merged = model_df[model_df["layer"] == left].merge(
                model_df[model_df["layer"] == right],
                on=["model_name", "fold_label"],
                suffixes=("_left", "_right"),
            )
            for metric, label, direction in METRICS:
                left_values = pd.to_numeric(merged[f"{metric}_left"], errors="coerce")
                right_values = pd.to_numeric(merged[f"{metric}_right"], errors="coerce")
                valid = pd.DataFrame({"left": left_values, "right": right_values}).dropna()
                diff = valid["right"] - valid["left"]
                improved = diff if direction == "higher" else -diff
                if len(valid) > 1 and not math.isclose(float(improved.std(ddof=1)), 0.0, abs_tol=1e-15):
                    t_stat, p_value = stats.ttest_rel(valid["right"], valid["left"], nan_policy="omit")
                else:
                    t_stat, p_value = (math.nan, math.nan)
                rows.append(
                    {
                        "model_name": model,
                        "comparison": f"{left}_vs_{right}",
                        "metric": label,
                        "n_pairs": len(valid),
                        "left_mean": valid["left"].mean(),
                        "right_mean": valid["right"].mean(),
                        "mean_diff_right_minus_left": diff.mean(),
                        "mean_improvement": improved.mean(),
                        "t_stat": t_stat,
                        "p_value_two_sided": p_value,
                        "significant_improvement_at_0_05": bool(
                            pd.notna(p_value) and p_value < 0.05 and improved.mean() > 0
                        ),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fold_frames = []
    for layer_slug, _, config_path in LAYERS:
        fold_frames.append(_run_layer(layer_slug, config_path))
    fold_df = pd.concat(fold_frames, ignore_index=True)
    fold_df.to_csv(OUT_DIR / "fold_summary_all_layers.csv", index=False)

    validation = fold_df[fold_df["evaluation_role"] == "validation"].copy()
    rows = []
    for layer_slug, layer_title, _ in LAYERS:
        for model in MODELS:
            subset = validation[(validation["layer"] == layer_slug) & (validation["model_name"] == model)]
            rows.append(
                {
                    "layer": layer_slug,
                    "layer_title": layer_title,
                    "model_name": model,
                    "AUROC": subset["auc"].mean(),
                    "F1-Score": subset["f1"].mean(),
                    "Log Loss": subset["log_loss"].mean(),
                    "n_folds": int(subset["fold_label"].nunique()),
                }
            )
    metrics_df = pd.DataFrame(rows)
    tests_df = _paired_tests(fold_df)
    metrics_df.to_csv(RESULTS_DIR / "capitaliq_layer_all_models_metrics.csv", index=False)
    tests_df.to_csv(RESULTS_DIR / "capitaliq_layer_all_models_paired_t_tests.csv", index=False)
    print(metrics_df.to_string(index=False))
    print()
    print(tests_df.to_string(index=False))


if __name__ == "__main__":
    main()
