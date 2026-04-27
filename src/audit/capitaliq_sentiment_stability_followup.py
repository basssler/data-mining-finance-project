from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.label_comparison_event_v2 import (
    VariantSpec,
    apply_variant_label_mode,
    attach_labels_to_event_panel,
    build_daily_label_table,
    clip_outliers,
    compute_global_feature_exclusions,
    evaluate_extended,
    fit_model,
    select_usable_features,
)
from src.labels_event_v1 import load_price_data, normalize_price_data
from src.train_event_panel_v2 import (
    load_config,
    load_event_panel,
    load_prebuilt_label_map,
    resolve_candidate_features,
    resolve_max_missingness_pct,
    resolve_model_params,
)


MODELING_DIR = Path("outputs") / "quarterly" / "modeling"
VALIDATION_DIR = Path("outputs") / "quarterly" / "validation"
CONFIG_DIR = Path("configs") / "quarterly"
BASE_CONFIG_PATH = CONFIG_DIR / "capitaliq_ladder_core_plus_market.yaml"
SECTOR_CONFIG_PATH = CONFIG_DIR / "capitaliq_ladder_sector_adjusted_sentiment.yaml"
RUNGS = [
    ("A", "core", "capitaliq_ladder_core"),
    ("B", "core_plus_market", "capitaliq_ladder_core_plus_market"),
    ("C", "raw_sentiment", "capitaliq_ladder_raw_sentiment"),
    ("D", "within_sector_adjusted_sentiment", "capitaliq_ladder_sector_adjusted_sentiment"),
]
MODELS = ["logistic_regression", "random_forest", "xgboost"]
SENTIMENT_GROUPS = {
    "sentiment_means_only": ["sent_mean_7d", "sent_mean_30d", "sent_mean_63d"],
    "news_counts_only": ["news_count_7d", "news_count_30d", "news_count_63d", "has_news_7d", "has_news_30d", "has_news_63d"],
    "sentiment_momentum_only": ["sent_momentum_7v30", "sent_momentum_30v63"],
    "sector_adjusted_only": [
        "sector_adj_sent_30d",
        "sector_adj_sent_63d",
        "sector_adj_news_share_30d",
        "sector_adj_news_share_63d",
    ],
    "all_capitaliq_sentiment_features": [
        "sent_mean_7d",
        "sent_mean_30d",
        "sent_mean_63d",
        "sent_vol_30d",
        "sent_vol_63d",
        "news_count_7d",
        "news_count_30d",
        "news_count_63d",
        "sent_momentum_7v30",
        "sent_momentum_30v63",
        "confidence_mean_30d",
        "has_news_7d",
        "has_news_30d",
        "has_news_63d",
        "low_news_coverage_30d",
        "low_news_coverage_63d",
        "sector_sent_mean_30d",
        "sector_sent_mean_63d",
        "sector_news_count_30d",
        "sector_news_count_63d",
        "sector_adj_sent_30d",
        "sector_adj_sent_63d",
        "sector_adj_news_share_30d",
        "sector_adj_news_share_63d",
    ],
}
FOCUS_FEATURES = [
    "sent_mean_30d",
    "sent_mean_63d",
    "sector_adj_sent_30d",
    "sector_adj_sent_63d",
    "news_count_30d",
    "news_count_63d",
    "sector_news_count_63d",
    "sent_momentum_7v30",
    "sent_momentum_30v63",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Capital IQ sentiment stability follow-up diagnostics.")
    parser.add_argument("--skip-ablation-runs", action="store_true")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    return parser.parse_args()


def fmt(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def load_label_df(config: dict) -> pd.DataFrame:
    label_cfg = config["label"]
    if label_cfg.get("path"):
        return load_prebuilt_label_map(Path(label_cfg["path"]))
    prices = normalize_price_data(load_price_data(Path(config["prices"]["path"])))
    return build_daily_label_table(
        prices,
        horizon_days=int(label_cfg["horizon_days"]),
        benchmark_mode=str(label_cfg.get("benchmark_mode", "sector_equal_weight_ex_self")),
    )


def load_labeled_panel(config: dict) -> pd.DataFrame:
    panel = load_event_panel(Path(config["panel"]["path"]))
    labeled = attach_labels_to_event_panel(panel, load_label_df(config))
    if str(config["label"]["mode"]) == "sign" and "target_sign" in labeled.columns:
        labeled["target"] = labeled["target_sign"]
    return labeled


def variant_from_config(config: dict) -> VariantSpec:
    label = config["label"]
    return VariantSpec(
        variant_name=str(label["variant_name"]),
        horizon_days=int(label["horizon_days"]),
        label_mode=str(label["mode"]),
        threshold=float(label.get("threshold", 0.015)),
    )


def apples_to_apples() -> pd.DataFrame:
    rows = []
    for rung, family, stem in RUNGS:
        result = pd.read_csv(MODELING_DIR / f"{stem}.csv")
        fold = pd.read_csv(VALIDATION_DIR / stem / "fold_summary.csv")
        for _, row in result.iterrows():
            fold_subset = fold[(fold["model_name"] == row["model_name"]) & (fold["evaluation_role"] == "holdout_eval")]
            holdout_f1 = float(fold_subset["f1"].iloc[0]) if not fold_subset.empty else math.nan
            rows.append(
                {
                    "rung": rung,
                    "family": family,
                    "model_name": row["model_name"],
                    "cv_auc_mean": row["cv_auc_mean"],
                    "cv_auc_std": row["cv_auc_std"],
                    "worst_fold_auc": row["worst_fold_auc"],
                    "holdout_auc": row["holdout_auc"],
                    "holdout_log_loss": row["holdout_log_loss"],
                    "holdout_f1": holdout_f1,
                    "feature_count": int(row["usable_feature_count_last_fold"]),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(MODELING_DIR / "capitaliq_sentiment_apples_to_apples.csv", index=False)
    lines = [
        "# Capital IQ Sentiment Apples-to-Apples Model Comparison",
        "",
        "Capital IQ Key Developments sentiment was tested as an incremental event-text feature layer in the quarterly Consumer Staples panel.",
        "",
        markdown_table(
            out,
            [
                "rung",
                "family",
                "model_name",
                "cv_auc_mean",
                "cv_auc_std",
                "worst_fold_auc",
                "holdout_auc",
                "holdout_log_loss",
                "holdout_f1",
                "feature_count",
            ],
        ),
        "",
    ]
    (MODELING_DIR / "capitaliq_sentiment_apples_to_apples.md").write_text("\n".join(lines), encoding="utf-8")
    return out


def create_ablation_configs() -> list[tuple[str, Path]]:
    base = load_config(BASE_CONFIG_PATH)
    base_additional = list(base.get("feature_inclusions", {}).get("additional", []) or [])
    jobs = [("core_plus_market_only", BASE_CONFIG_PATH)]
    for ablation, features in SENTIMENT_GROUPS.items():
        cfg = json.loads(json.dumps(base))
        cfg["panel"]["name"] = f"capitaliq_ablation_{ablation}"
        cfg["feature_inclusions"]["additional"] = base_additional + features
        cfg["metadata"]["report_title"] = f"Capital IQ Sentiment Ablation: {ablation}"
        cfg["metadata"]["panel_display_name"] = f"capitaliq_ablation_{ablation}"
        cfg["metadata"]["experiment_family"] = f"capitaliq_ablation_{ablation}"
        cfg["metadata"]["design_note"] = f"capitaliq_sentiment_ablation_{ablation}"
        cfg["outputs"] = {
            "csv": f"outputs/quarterly/modeling/capitaliq_ablation_{ablation}.csv",
            "markdown": f"outputs/quarterly/modeling/capitaliq_ablation_{ablation}.md",
            "shap_plot": f"outputs/quarterly/modeling/capitaliq_ablation_{ablation}_shap_summary.png",
            "shap_csv": f"outputs/quarterly/modeling/capitaliq_ablation_{ablation}_shap_importance.csv",
            "validation_dir": f"outputs/quarterly/validation/capitaliq_ablation_{ablation}",
        }
        path = CONFIG_DIR / f"capitaliq_ablation_{ablation}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        jobs.append((ablation, path))
    return jobs


def run_ablation_jobs(skip: bool) -> pd.DataFrame:
    jobs = create_ablation_configs()
    rows = []
    commands = []
    for ablation, config_path in jobs:
        if ablation == "core_plus_market_only":
            stem = "capitaliq_ladder_core_plus_market"
        else:
            stem = f"capitaliq_ablation_{ablation}"
            if not skip:
                cmd = [sys.executable, "src/train_event_panel_v2.py", "--config", str(config_path)]
                commands.append(" ".join(cmd))
                subprocess.run(cmd, cwd=Path.cwd(), check=False)
        result_path = MODELING_DIR / f"{stem}.csv"
        if not result_path.exists():
            continue
        result = pd.read_csv(result_path)
        selected = result.loc[result["is_selected_primary_model"].astype(bool)]
        selected = selected.iloc[0] if not selected.empty else result.sort_values("holdout_auc", ascending=False).iloc[0]
        rows.append(
            {
                "ablation": ablation,
                "selected_model": selected["model_name"],
                "cv_auc_mean": selected["cv_auc_mean"],
                "cv_auc_std": selected["cv_auc_std"],
                "worst_fold_auc": selected["worst_fold_auc"],
                "holdout_auc": selected["holdout_auc"],
                "holdout_log_loss": selected["holdout_log_loss"],
                "feature_count": int(selected["usable_feature_count_last_fold"]),
                "dominant_feature": selected.get("dominant_feature_name"),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(MODELING_DIR / "capitaliq_sentiment_ablation.csv", index=False)
    best = out.sort_values("holdout_auc", ascending=False).iloc[0] if not out.empty else None
    lines = [
        "# Capital IQ Sentiment Ablation",
        "",
        "These are untuned ablations on top of the core + market control. The goal is to separate event-text sentiment from news/event volume proxies.",
        "",
        markdown_table(
            out,
            ["ablation", "selected_model", "cv_auc_mean", "cv_auc_std", "worst_fold_auc", "holdout_auc", "holdout_log_loss", "feature_count", "dominant_feature"],
        ),
        "",
    ]
    if best is not None:
        lines.append(
            f"Best selected-model holdout AUC came from `{best['ablation']}` at `{fmt(best['holdout_auc'])}`."
        )
        if "news_count" in str(best.get("dominant_feature", "")) or "count" in str(best["ablation"]):
            lines.append("This points toward event-attention/news-volume as an important part of the signal.")
        elif "sent" in str(best.get("dominant_feature", "")) or "sentiment" in str(best["ablation"]):
            lines.append("This points toward event-text sentiment carrying incremental information.")
    (MODELING_DIR / "capitaliq_sentiment_ablation.md").write_text("\n".join(lines), encoding="utf-8")
    return out


def fit_predict_window(config_path: Path, *, model_name: str, test_start: pd.Timestamp, test_end: pd.Timestamp, conservative_gap_days: int = 68) -> dict:
    config = load_config(config_path)
    panel = load_labeled_panel(config)
    variant = variant_from_config(config)
    panel, _ = apply_variant_label_mode(panel, variant)
    candidate = resolve_candidate_features(panel, config)
    explicit = list(config.get("feature_exclusions", {}).get("explicit", []))
    train_cutoff = test_start - pd.Timedelta(days=conservative_gap_days)
    train = panel[panel["date"] < train_cutoff].copy()
    test = panel[(panel["date"] >= test_start) & (panel["date"] <= test_end)].copy()
    global_candidates = [column for column in candidate if column not in explicit]
    kept, _, _ = compute_global_feature_exclusions(panel, global_candidates, holdout_start=str(test_start.date()))
    usable, _, _, _ = select_usable_features(train, kept, max_missingness_pct=resolve_max_missingness_pct(config))
    clipped_train, clipped_test = clip_outliers(train, test, usable)
    model, backend = fit_model(
        model_name,
        clipped_train[usable],
        clipped_train["target"].astype(int),
        model_params=resolve_model_params(config, model_name),
    )
    prob = model.predict_proba(clipped_test[usable])[:, 1]
    metrics = evaluate_extended(clipped_test, prob, threshold=0.5)
    return {
        "model_name": model_name,
        "backend": backend,
        "auc": metrics.get("auc_roc"),
        "f1": metrics.get("f1"),
        "log_loss": metrics.get("log_loss"),
        "row_count": metrics.get("row_count"),
        "feature_count": len(usable),
        "y_true": clipped_test["target"].astype(int).to_numpy(),
        "y_prob": prob,
        "event_id": clipped_test["event_id"].astype(str).to_numpy() if "event_id" in clipped_test.columns else np.arange(len(clipped_test)).astype(str),
    }


def year_holdouts() -> pd.DataFrame:
    rows = []
    for year in [2021, 2022, 2023, 2024]:
        start = pd.Timestamp(year=year, month=1, day=1)
        end = pd.Timestamp(year=year, month=12, day=31)
        for config_name, config_path in [("core_plus_market", BASE_CONFIG_PATH), ("within_sector_adjusted_sentiment", SECTOR_CONFIG_PATH)]:
            for model_name in MODELS:
                result = fit_predict_window(config_path, model_name=model_name, test_start=start, test_end=end)
                rows.append(
                    {
                        "year": year,
                        "config": config_name,
                        "model_name": model_name,
                        "auc": result["auc"],
                        "f1": result["f1"],
                        "log_loss": result["log_loss"],
                        "row_count": result["row_count"],
                        "feature_count": result["feature_count"],
                    }
                )
    out = pd.DataFrame(rows)
    lines = [
        "# Capital IQ Sentiment Year-by-Year Pseudo-Holdouts",
        "",
        "Pseudo-holdouts train only on prior data and test one calendar year. A conservative 68-calendar-day gap is used before each test year to reduce label-window overlap; this is diagnostic, not a replacement for the locked 2024 holdout.",
        "",
        markdown_table(out, ["year", "config", "model_name", "auc", "f1", "log_loss", "row_count", "feature_count"]),
        "",
    ]
    pivot = out.pivot_table(index=["year", "model_name"], columns="config", values="auc", aggfunc="first").reset_index()
    if {"core_plus_market", "within_sector_adjusted_sentiment"}.issubset(pivot.columns):
        pivot["auc_delta_sentiment_minus_control"] = pivot["within_sector_adjusted_sentiment"] - pivot["core_plus_market"]
        lines.extend(["## AUC Delta", "", markdown_table(pivot, ["year", "model_name", "core_plus_market", "within_sector_adjusted_sentiment", "auc_delta_sentiment_minus_control"]), ""])
    (MODELING_DIR / "capitaliq_sentiment_year_holdouts.md").write_text("\n".join(lines), encoding="utf-8")
    return out


def bootstrap_holdout(iterations: int) -> pd.DataFrame:
    control_model = pd.read_csv(MODELING_DIR / "capitaliq_ladder_core_plus_market.csv")
    sentiment_model = pd.read_csv(MODELING_DIR / "capitaliq_ladder_sector_adjusted_sentiment.csv")
    control_name = control_model.loc[control_model["is_selected_primary_model"].astype(bool), "model_name"].iloc[0]
    sentiment_name = sentiment_model.loc[sentiment_model["is_selected_primary_model"].astype(bool), "model_name"].iloc[0]
    control = fit_predict_window(BASE_CONFIG_PATH, model_name=control_name, test_start=pd.Timestamp("2024-01-01"), test_end=pd.Timestamp("2024-12-31"), conservative_gap_days=68)
    sentiment = fit_predict_window(SECTOR_CONFIG_PATH, model_name=sentiment_name, test_start=pd.Timestamp("2024-01-01"), test_end=pd.Timestamp("2024-12-31"), conservative_gap_days=68)
    prediction_df = pd.DataFrame(
        {
            "event_id": sentiment["event_id"],
            "target": sentiment["y_true"],
            "core_plus_market_prob": control["y_prob"],
            "within_sector_sentiment_prob": sentiment["y_prob"],
        }
    )
    prediction_df.to_csv(MODELING_DIR / "capitaliq_sentiment_2024_holdout_predictions.csv", index=False)
    rng = np.random.default_rng(42)
    y = prediction_df["target"].to_numpy()
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    deltas = []
    for _ in range(int(iterations)):
        sample_idx = np.concatenate(
            [
                rng.choice(pos_idx, size=len(pos_idx), replace=True),
                rng.choice(neg_idx, size=len(neg_idx), replace=True),
            ]
        )
        if len(np.unique(y[sample_idx])) < 2:
            continue
        sent_auc = roc_auc_score(y[sample_idx], prediction_df["within_sector_sentiment_prob"].to_numpy()[sample_idx])
        ctrl_auc = roc_auc_score(y[sample_idx], prediction_df["core_plus_market_prob"].to_numpy()[sample_idx])
        deltas.append(sent_auc - ctrl_auc)
    summary = pd.DataFrame(
        [
            {
                "iterations": len(deltas),
                "control_model": control_name,
                "sentiment_model": sentiment_name,
                "control_auc": roc_auc_score(y, prediction_df["core_plus_market_prob"]),
                "sentiment_auc": roc_auc_score(y, prediction_df["within_sector_sentiment_prob"]),
                "mean_delta": float(np.mean(deltas)),
                "p05_delta": float(np.percentile(deltas, 5)),
                "p95_delta": float(np.percentile(deltas, 95)),
            }
        ]
    )
    lines = [
        "# Capital IQ Sentiment 2024 Holdout Bootstrap",
        "",
        "This diagnostic refits only the selected core+market and within-sector sentiment models to recreate 2024 holdout probabilities, then bootstraps the AUC delta with class-balanced resampling.",
        "",
        markdown_table(summary, list(summary.columns)),
        "",
        f"Predictions saved to `outputs/quarterly/modeling/capitaliq_sentiment_2024_holdout_predictions.csv`.",
        "",
    ]
    (MODELING_DIR / "capitaliq_sentiment_holdout_bootstrap.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def feature_stability() -> pd.DataFrame:
    rows = []
    for stem in ["capitaliq_ladder_raw_sentiment", "capitaliq_ladder_sector_adjusted_sentiment"]:
        conc = pd.read_csv(MODELING_DIR / f"{stem}_concentration.csv")
        for model_name in MODELS:
            sub = conc[(conc["model_name"] == model_name) & (conc["rank"] <= 3)]
            for feature in FOCUS_FEATURES:
                feature_rows = sub[sub["feature"] == feature]
                rows.append(
                    {
                        "rung": stem,
                        "model_name": model_name,
                        "feature": feature,
                        "top3_count_all_evals": len(feature_rows),
                        "top3_count_validation": int((feature_rows["evaluation_role"] == "validation").sum()),
                        "top3_count_holdout": int((feature_rows["evaluation_role"] == "holdout_eval").sum()),
                        "mean_importance_when_top3": feature_rows["importance_mean"].mean() if not feature_rows.empty else np.nan,
                    }
                )
    out = pd.DataFrame(rows)
    lines = [
        "# Capital IQ Sentiment Feature Importance Stability",
        "",
        "Counts show how often focus sentiment features appeared in top-3 feature diagnostics across validation folds and holdout.",
        "",
        markdown_table(out, ["rung", "model_name", "feature", "top3_count_validation", "top3_count_holdout", "mean_importance_when_top3"]),
        "",
    ]
    (MODELING_DIR / "capitaliq_sentiment_feature_stability.md").write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    args = parse_args()
    MODELING_DIR.mkdir(parents=True, exist_ok=True)
    apples_to_apples()
    run_ablation_jobs(skip=bool(args.skip_ablation_runs))
    year_holdouts()
    bootstrap_holdout(iterations=int(args.bootstrap_iterations))
    feature_stability()
    print("Capital IQ sentiment stability follow-up complete.")


if __name__ == "__main__":
    main()
