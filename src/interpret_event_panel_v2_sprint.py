"""Run an interpretation-first analysis sprint for the locked event_panel_v2 baseline."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

matplotlib.use("Agg")

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import shap

from src.config_event_v1 import PRICE_INPUT_PATH
from src.label_comparison_event_v2 import (
    VariantSpec,
    apply_variant_label_mode,
    attach_labels_to_event_panel,
    build_daily_label_table,
    clip_outliers,
    compute_global_feature_exclusions,
    fit_model,
    load_event_panel,
    safe_rank_ic,
    select_usable_features,
)
from src.labels_event_v1 import load_price_data, normalize_price_data
from src.train_event_panel_v2 import (
    build_markdown_report,
    ensure_parent_dir,
    load_config,
    resolve_candidate_features,
    run_model_matrix,
)
from src.validation_event_v1 import make_event_v1_splits

DEFAULT_CONFIG_PATH = Path("configs") / "event_panel_v2_primary.yaml"
DEFAULT_ARTIFACT_ROOT = Path("artifacts")
DEFAULT_TOP_FEATURES = 10
DEFAULT_CLUSTER_THRESHOLD = 0.8
DEFAULT_INTERACTION_SAMPLE = 256


FEATURE_FAMILY_RULES: list[tuple[str, set[str]]] = [
    (
        "event_nlp",
        {
            "sec_sentiment_score",
            "sec_positive_prob",
            "sec_negative_prob",
            "sec_neutral_prob",
            "sec_sentiment_abs",
            "sec_sentiment_change_prev",
            "sec_positive_change_prev",
            "sec_negative_change_prev",
            "sec_chunk_count",
            "sec_log_chunk_count",
        },
    ),
    (
        "momentum_returns",
        {
            "rel_return_5d",
            "rel_return_10d",
            "rel_return_21d",
            "overnight_gap_1d",
            "abs_return_shock_1d",
            "drawdown_21d",
            "return_zscore_21d",
        },
    ),
    (
        "volatility_risk",
        {
            "realized_vol_21d",
            "realized_vol_63d",
            "vol_ratio_21d_63d",
            "beta_63d_to_sector",
        },
    ),
    (
        "liquidity_trading",
        {
            "volume_ratio_20d",
            "log_volume",
            "abnormal_volume_flag",
        },
    ),
    (
        "fundamentals",
        {
            "current_ratio",
            "quick_ratio",
            "cash_ratio",
            "working_capital_to_total_assets",
            "debt_to_equity",
            "debt_to_assets",
            "long_term_debt_ratio",
            "gross_margin",
            "operating_margin",
            "net_margin",
            "roa",
            "roe",
            "asset_turnover",
            "inventory_turnover",
            "receivables_turnover",
            "revenue_growth_qoq",
            "revenue_growth_yoy",
            "earnings_growth_qoq",
            "earnings_growth_yoy",
            "cfo_to_net_income",
            "accruals_ratio",
            "liquidity_profile_score",
            "solvency_profile_score",
            "profitability_profile_score",
            "growth_quality_profile_score",
            "overall_financial_health_score",
        },
    ),
    (
        "timing_context",
        {
            "days_since_prior_event",
            "days_since_prior_same_event_type",
            "current_filing_fundamentals_available",
            "current_filing_sentiment_available",
            "fund_snapshot_is_current_event",
        },
    ),
]

FAMILY_DESCRIPTIONS = {
    "event_nlp": "SEC filing tone, sentiment probability mix, and document intensity.",
    "momentum_returns": "Short-horizon relative returns and recent price shock behavior.",
    "volatility_risk": "Trailing risk regime, sector beta, and volatility scaling.",
    "liquidity_trading": "Volume level, abnormal turnover, and trading intensity.",
    "fundamentals": "Accounting quality, balance-sheet strength, profitability, and growth.",
    "timing_context": "Recency and event/current-snapshot context flags.",
    "other": "Unmapped feature requiring manual interpretation.",
}

INTERACTION_PAIRS = [
    ("sec_positive_prob", "rel_return_10d"),
    ("sec_positive_prob", "log_volume"),
    ("beta_63d_to_sector", "realized_vol_21d"),
    ("cash_ratio", "realized_vol_21d"),
    ("fund_snapshot_is_current_event", "sec_positive_prob"),
    ("volume_ratio_20d", "abs_return_shock_1d"),
]


@dataclass
class HoldoutPayload:
    train_df: pd.DataFrame
    holdout_df: pd.DataFrame
    usable_features: list[str]
    split_payload: dict
    fitted_model: object
    backend: str
    transformed_holdout: pd.DataFrame
    shap_values: np.ndarray
    explainer: object


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in df.to_dict(orient="records"):
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the event_panel_v2 interpretation sprint.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--top-features", type=int, default=DEFAULT_TOP_FEATURES)
    parser.add_argument("--cluster-threshold", type=float, default=DEFAULT_CLUSTER_THRESHOLD)
    parser.add_argument("--interaction-sample", type=int, default=DEFAULT_INTERACTION_SAMPLE)
    return parser.parse_args()


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


def get_feature_family(feature_name: str) -> str:
    for family, members in FEATURE_FAMILY_RULES:
        if feature_name in members:
            return family
    return "other"


def build_feature_family_map(feature_columns: list[str]) -> pd.DataFrame:
    rows = []
    for feature in feature_columns:
        family = get_feature_family(feature)
        rows.append(
            {
                "feature_name": feature,
                "feature_family": family,
                "economic_interpretation": FAMILY_DESCRIPTIONS[family],
                "notes": "",
            }
        )
    return pd.DataFrame(rows).sort_values(["feature_family", "feature_name"]).reset_index(drop=True)


def resolve_baseline_outputs(config: dict) -> tuple[Path, Path, Path, Path]:
    outputs = config["outputs"]
    csv_path = Path(outputs["csv"])
    markdown_path = Path(outputs["markdown"])
    shap_plot_path = Path(outputs.get("shap_plot", csv_path.with_name(f"{csv_path.stem}_shap_summary.png")))
    shap_csv_path = Path(outputs.get("shap_csv", csv_path.with_name(f"{csv_path.stem}_shap_importance.csv")))
    return csv_path, markdown_path, shap_plot_path, shap_csv_path


def freeze_baseline(
    labeled_panel_df: pd.DataFrame,
    config: dict,
    baseline_dir: Path,
) -> tuple[pd.DataFrame, dict, Path, Path]:
    variant = build_variant(config)
    candidate_features = resolve_candidate_features(labeled_panel_df, config)
    explicit_exclusions = list(config["feature_exclusions"]["explicit"])
    result_df, summary = run_model_matrix(
        panel_df=labeled_panel_df,
        variant=variant,
        model_names=list(config["models"]),
        candidate_features=candidate_features,
        explicit_exclusions=explicit_exclusions,
        holdout_start=str(config["holdout"]["start"]),
        n_splits=int(config["cv"]["n_splits"]),
        embargo_days=int(config["cv"]["embargo_days"]),
        min_train_dates=int(config["cv"]["min_train_dates"]),
        threshold=0.5,
        panel_name=str(config.get("panel", {}).get("name", "event_panel_v2")),
    )
    benchmark_csv_path, benchmark_md_path, shap_plot_path, shap_csv_path = resolve_baseline_outputs(config)
    if not benchmark_csv_path.exists():
        ensure_parent_dir(benchmark_csv_path)
        result_df.to_csv(benchmark_csv_path, index=False)
    if not benchmark_md_path.exists():
        ensure_parent_dir(benchmark_md_path)
        benchmark_md_path.write_text(build_markdown_report(result_df, summary, old_baseline=None), encoding="utf-8")

    baseline_dir.mkdir(parents=True, exist_ok=True)
    selected_row = result_df.loc[result_df["is_selected_primary_model"]].iloc[0]
    usable_features = json.loads(selected_row["usable_feature_columns_last_fold"])
    feature_manifest_df = pd.DataFrame(
        {
            "feature_name": usable_features,
            "feature_family": [get_feature_family(name) for name in usable_features],
        }
    )
    feature_manifest_df.to_csv(baseline_dir / "feature_manifest.csv", index=False)

    model_config = {
        "config": config,
        "selected_model_name": str(selected_row["model_name"]),
        "selected_model_cv_auc_mean": selected_row["cv_auc_mean"],
        "selected_model_holdout_auc": selected_row["holdout_auc"],
        "selected_model_holdout_log_loss": selected_row["holdout_log_loss"],
        "summary": {
            "best_model_name": summary["best_model_name"],
            "explicit_exclusions": summary["explicit_exclusions"],
            "auto_all_missing": summary["auto_all_missing"],
            "auto_constant": summary["auto_constant"],
        },
    }
    (baseline_dir / "model_config.json").write_text(json.dumps(model_config, indent=2), encoding="utf-8")
    result_df.to_json(baseline_dir / "metrics.json", orient="records", indent=2)

    notes = [
        "# Baseline Notes",
        "",
        f"- Panel: `{config['panel']['name']}`",
        f"- Panel path: `{config['panel']['path']}`",
        f"- Target variant: `{config['label']['variant_name']}`",
        f"- Holdout start: `{config['holdout']['start']}`",
        f"- Selected model: `{selected_row['model_name']}`",
        f"- Candidate feature count: `{len(candidate_features)}`",
        f"- Usable feature count last fold: `{selected_row['usable_feature_count_last_fold']}`",
        f"- SHAP plot source: `{shap_plot_path}`",
        f"- SHAP CSV source: `{shap_csv_path}`",
        "- Preprocessing assumptions: median imputation, train-only clipping, model-specific scaling where applicable.",
        "- Feature timestamp logic: event rows keyed on `effective_model_date`; market features must be prior-day aligned.",
    ]
    (baseline_dir / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")

    if shap_plot_path.exists():
        shutil.copy2(shap_plot_path, baseline_dir / "shap_summary.png")

    return result_df, summary, shap_plot_path, shap_csv_path


def fit_selected_model_for_holdout(
    labeled_panel_df: pd.DataFrame,
    config: dict,
    selected_model_name: str,
) -> HoldoutPayload:
    variant = build_variant(config)
    holdout_start = str(config["holdout"]["start"])
    candidate_features = resolve_candidate_features(labeled_panel_df, config)
    explicit_exclusions = list(config["feature_exclusions"]["explicit"])
    split_payload = make_event_v1_splits(
        df=labeled_panel_df,
        date_col="date",
        horizon_days=variant.horizon_days,
        n_splits=int(config["cv"]["n_splits"]),
        embargo_days=int(config["cv"]["embargo_days"]),
        holdout_start=holdout_start,
        min_train_dates=int(config["cv"]["min_train_dates"]),
    )
    global_candidates = [column for column in candidate_features if column not in explicit_exclusions]
    kept_global, _, _ = compute_global_feature_exclusions(
        labeled_panel_df,
        global_candidates,
        holdout_start=holdout_start,
    )
    holdout_train_full = labeled_panel_df.iloc[split_payload["holdout"]["train_indices"]].copy()
    holdout_full = labeled_panel_df.iloc[split_payload["holdout"]["holdout_indices"]].copy()
    holdout_train_active, _ = apply_variant_label_mode(holdout_train_full, variant)
    holdout_active, _ = apply_variant_label_mode(holdout_full, variant)
    holdout_usable, _, _, _ = select_usable_features(holdout_train_active, kept_global)
    clipped_train, clipped_holdout = clip_outliers(holdout_train_active, holdout_active, holdout_usable)
    fitted_model, backend = fit_model(
        selected_model_name,
        clipped_train[holdout_usable],
        clipped_train["target"].astype(int),
    )
    imputer = fitted_model.named_steps["imputer"]
    model = fitted_model.named_steps["model"]
    transformed_holdout = pd.DataFrame(
        imputer.transform(clipped_holdout[holdout_usable]),
        columns=holdout_usable,
        index=clipped_holdout.index,
    )
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(transformed_holdout)
    shap_matrix = np.asarray(shap_values[-1] if isinstance(shap_values, list) else shap_values)
    if shap_matrix.ndim == 3:
        shap_matrix = shap_matrix[:, :, -1]
    return HoldoutPayload(
        train_df=clipped_train,
        holdout_df=clipped_holdout,
        usable_features=holdout_usable,
        split_payload=split_payload,
        fitted_model=fitted_model,
        backend=backend,
        transformed_holdout=transformed_holdout,
        shap_values=shap_matrix,
        explainer=explainer,
    )


def compute_train_metric(y_true: pd.Series, y_prob: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if y_true.nunique(dropna=True) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def run_family_ablation(
    labeled_panel_df: pd.DataFrame,
    config: dict,
    selected_model_name: str,
    feature_family_df: pd.DataFrame,
    output_csv: Path,
    output_md: Path,
) -> pd.DataFrame:
    family_to_features = feature_family_df.groupby("feature_family")["feature_name"].apply(list).to_dict()
    variant = build_variant(config)
    holdout_start = str(config["holdout"]["start"])
    split_payload = make_event_v1_splits(
        df=labeled_panel_df,
        date_col="date",
        horizon_days=variant.horizon_days,
        n_splits=int(config["cv"]["n_splits"]),
        embargo_days=int(config["cv"]["embargo_days"]),
        holdout_start=holdout_start,
        min_train_dates=int(config["cv"]["min_train_dates"]),
    )
    combos = [
        ("event_nlp_only", ["event_nlp"]),
        ("market_only", ["momentum_returns", "volatility_risk", "liquidity_trading"]),
        ("fundamentals_only", ["fundamentals"]),
        ("event_plus_market", ["event_nlp", "momentum_returns", "volatility_risk", "liquidity_trading"]),
        ("event_plus_fundamentals", ["event_nlp", "fundamentals"]),
        ("market_plus_fundamentals", ["momentum_returns", "volatility_risk", "liquidity_trading", "fundamentals"]),
        (
            "full_model",
            ["event_nlp", "momentum_returns", "volatility_risk", "liquidity_trading", "fundamentals", "timing_context"],
        ),
    ]
    rows = []
    for model_name, families in combos:
        candidate_features = [
            feature
            for family in families
            for feature in family_to_features.get(family, [])
            if feature in labeled_panel_df.columns
        ]
        candidate_features = list(dict.fromkeys(candidate_features))
        if not candidate_features:
            continue
        fold_scores = []
        last_top_features: list[str] = []
        for fold in split_payload["folds"]:
            train_full = labeled_panel_df.iloc[fold["train_indices"]].copy()
            validation_full = labeled_panel_df.iloc[fold["validation_indices"]].copy()
            train_active, _ = apply_variant_label_mode(train_full, variant)
            validation_active, _ = apply_variant_label_mode(validation_full, variant)
            usable_features, _, _, _ = select_usable_features(train_active, candidate_features)
            clipped_train, clipped_validation = clip_outliers(train_active, validation_active, usable_features)
            fitted_model, _ = fit_model(
                selected_model_name,
                clipped_train[usable_features],
                clipped_train["target"].astype(int),
            )
            train_prob = fitted_model.predict_proba(clipped_train[usable_features])[:, 1]
            validation_prob = fitted_model.predict_proba(clipped_validation[usable_features])[:, 1]
            fold_scores.append(
                {
                    "train_auc": compute_train_metric(clipped_train["target"].astype(int), train_prob),
                    "val_auc": compute_train_metric(clipped_validation["target"].astype(int), validation_prob),
                }
            )
            if selected_model_name == "xgboost":
                model = fitted_model.named_steps["model"]
                top_series = pd.Series(model.feature_importances_, index=usable_features).sort_values(ascending=False)
                last_top_features = top_series.head(5).index.tolist()
        mean_train_auc = float(np.mean([score["train_auc"] for score in fold_scores]))
        mean_val_auc = float(np.mean([score["val_auc"] for score in fold_scores]))
        rows.append(
            {
                "model_name": model_name,
                "feature_families_used": ", ".join(families),
                "train_metric": mean_train_auc,
                "val_metric": mean_val_auc,
                "overfit_gap": mean_train_auc - mean_val_auc,
                "feature_count": len(candidate_features),
                "top_shap_features": ", ".join(last_top_features),
                "directional_reasonableness": "manual_review",
                "notes": "",
            }
        )
    ablation_df = pd.DataFrame(rows).sort_values("model_name").reset_index(drop=True)
    ensure_parent_dir(output_csv)
    ablation_df.to_csv(output_csv, index=False)
    lines = [
        "# Ablation Summary",
        "",
        f"- Selected model held constant: `{selected_model_name}`",
        f"- Strongest validation setup: `{ablation_df.sort_values('val_metric', ascending=False).iloc[0]['model_name']}`",
        f"- Weakest validation setup: `{ablation_df.sort_values('val_metric', ascending=True).iloc[0]['model_name']}`",
        "",
        "| Model | Families | Train AUC | Validation AUC | Overfit Gap | Feature Count |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in ablation_df.iterrows():
        lines.append(
            f"| {row['model_name']} | {row['feature_families_used']} | {row['train_metric']:.4f} | "
            f"{row['val_metric']:.4f} | {row['overfit_gap']:.4f} | {int(row['feature_count'])} |"
        )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ablation_df


def correlation_cluster_members(matrix: pd.DataFrame, threshold: float) -> list[list[str]]:
    features = list(matrix.columns)
    seen: set[str] = set()
    components: list[list[str]] = []
    for feature in features:
        if feature in seen:
            continue
        stack = [feature]
        component: list[str] = []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.append(current)
            neighbors = matrix.index[(matrix.loc[current].abs() >= threshold) & (matrix.index != current)].tolist()
            stack.extend(neighbors)
        components.append(sorted(component))
    return sorted(components, key=lambda members: (-len(members), members[0]))


def run_correlation_analysis(
    train_df: pd.DataFrame,
    usable_features: list[str],
    threshold: float,
    output_dir: Path,
) -> pd.DataFrame:
    numeric = train_df[usable_features].apply(pd.to_numeric, errors="coerce")
    pearson = numeric.corr(method="pearson")
    spearman = numeric.corr(method="spearman")
    ensure_parent_dir(output_dir / "pearson_corr.csv")
    pearson.to_csv(output_dir / "pearson_corr.csv")
    spearman.to_csv(output_dir / "spearman_corr.csv")
    components = correlation_cluster_members(spearman.fillna(0.0), threshold=threshold)
    rows = []
    for idx, component in enumerate(components, start=1):
        for feature in component:
            neighbors = [
                neighbor
                for neighbor in component
                if neighbor != feature and abs(float(spearman.loc[feature, neighbor])) >= threshold
            ]
            rows.append(
                {
                    "cluster_id": idx,
                    "feature_name": feature,
                    "correlated_neighbors": ", ".join(neighbors),
                    "possible_redundancy_flag": len(component) > 1,
                    "recommended_action": "review_cluster" if len(component) > 1 else "keep_for_now",
                }
            )
    cluster_df = pd.DataFrame(rows).sort_values(["cluster_id", "feature_name"]).reset_index(drop=True)
    cluster_df.to_csv(output_dir / "feature_clusters.csv", index=False)
    notes = [
        "# Correlation Notes",
        "",
        f"- Correlation threshold for cluster linkage: `{threshold:.2f}` on absolute Spearman correlation.",
        f"- Largest cluster size: `{max(len(c) for c in components) if components else 0}`",
        f"- Multi-feature clusters: `{sum(1 for c in components if len(c) > 1)}`",
        "",
        "High-priority clusters:",
    ]
    for component in components[:10]:
        if len(component) > 1:
            notes.append(f"- `{', '.join(component)}`")
    (output_dir / "correlation_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    return cluster_df


def detect_pattern_type(feature_values: pd.Series, shap_values: pd.Series) -> tuple[str, str, str]:
    valid = feature_values.notna() & shap_values.notna()
    if int(valid.sum()) < 20:
        return "insufficient_data", "unknown", "low"
    corr = spearmanr(feature_values.loc[valid], shap_values.loc[valid]).statistic
    corr = float(corr) if corr is not None and not math.isnan(corr) else 0.0
    q10 = float(feature_values.loc[valid].quantile(0.10))
    q90 = float(feature_values.loc[valid].quantile(0.90))
    tail_gap = float(
        shap_values.loc[valid & (feature_values <= q10)].mean()
        - shap_values.loc[valid & (feature_values >= q90)].mean()
    )
    if abs(corr) >= 0.5:
        pattern = "roughly_monotonic"
    elif abs(tail_gap) >= shap_values.loc[valid].std():
        pattern = "tail_driven"
    else:
        pattern = "nonlinear_or_noisy"
    threshold = "possible_threshold" if abs(tail_gap) >= 0.05 else "weak_threshold_signal"
    stability = "high" if abs(corr) >= 0.6 else "medium" if abs(corr) >= 0.3 else "low"
    return pattern, threshold, stability


def candidate_transform(pattern_type: str) -> str:
    if pattern_type == "roughly_monotonic":
        return "keep_raw_or_winsorize"
    if pattern_type == "tail_driven":
        return "consider_regime_bin_or_tail_clip"
    if pattern_type == "nonlinear_or_noisy":
        return "consider_nonlinear_transform"
    return "manual_review"


def run_shap_dependence(
    holdout_payload: HoldoutPayload,
    priority_features: list[str],
    output_dir: Path,
    summary_path: Path,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    feature_index = {feature: idx for idx, feature in enumerate(holdout_payload.transformed_holdout.columns)}
    for feature in priority_features:
        if feature not in feature_index:
            continue
        idx = feature_index[feature]
        x = holdout_payload.transformed_holdout[feature]
        y = pd.Series(holdout_payload.shap_values[:, idx], index=x.index, dtype="float64")
        pattern, threshold_behavior, stability = detect_pattern_type(x, y)
        rows.append(
            {
                "feature_name": feature,
                "pattern_type": pattern,
                "threshold_behavior": threshold_behavior,
                "stability": stability,
                "candidate_transform": candidate_transform(pattern),
                "notes": "",
            }
        )
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(x, y, s=14, alpha=0.6)
        ax.set_xlabel(feature)
        ax.set_ylabel("SHAP value")
        ax.set_title(f"SHAP dependence: {feature}")
        fig.tight_layout()
        fig.savefig(output_dir / f"{feature}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    summary_df = pd.DataFrame(rows).sort_values("feature_name").reset_index(drop=True)
    summary_path.write_text(
        "# SHAP Dependence Summary\n\n" + dataframe_to_markdown(summary_df) + "\n",
        encoding="utf-8",
    )
    return summary_df


def run_shap_interactions(
    holdout_payload: HoldoutPayload,
    interaction_pairs: list[tuple[str, str]],
    output_dir: Path,
    summary_path: Path,
    sample_size: int,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    features = holdout_payload.transformed_holdout.columns.tolist()
    feature_index = {feature: idx for idx, feature in enumerate(features)}
    sampled = holdout_payload.transformed_holdout.head(sample_size)
    interaction_values = holdout_payload.explainer.shap_interaction_values(sampled)
    interaction_array = np.asarray(interaction_values)
    if interaction_array.ndim == 4:
        interaction_array = interaction_array[:, :, :, -1]
    rows = []
    for feature_a, feature_b in interaction_pairs:
        if feature_a not in feature_index or feature_b not in feature_index:
            continue
        idx_a = feature_index[feature_a]
        idx_b = feature_index[feature_b]
        strength = float(np.abs(interaction_array[:, idx_a, idx_b]).mean())
        rows.append(
            {
                "feature_a": feature_a,
                "feature_b": feature_b,
                "interaction_strength": strength,
                "economic_story": "manual_review",
                "candidate_feature_idea": "interaction_or_regime_feature",
                "notes": "",
            }
        )
        fig, ax = plt.subplots(figsize=(7, 5))
        sc = ax.scatter(sampled[feature_a], sampled[feature_b], c=interaction_array[:, idx_a, idx_b], cmap="coolwarm", s=18)
        ax.set_xlabel(feature_a)
        ax.set_ylabel(feature_b)
        ax.set_title(f"SHAP interaction: {feature_a} x {feature_b}")
        fig.colorbar(sc, ax=ax, label="Interaction SHAP")
        fig.tight_layout()
        fig.savefig(output_dir / f"{feature_a}__{feature_b}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    interaction_df = pd.DataFrame(rows).sort_values("interaction_strength", ascending=False).reset_index(drop=True)
    summary_path.write_text(
        "# SHAP Interaction Summary\n\n" + dataframe_to_markdown(interaction_df) + "\n",
        encoding="utf-8",
    )
    return interaction_df


def run_leakage_audit(
    feature_columns: list[str],
    output_csv: Path,
    output_md: Path,
) -> pd.DataFrame:
    rows = []
    for feature in feature_columns:
        risk = "low"
        evidence = "Uses event-safe panel alignment by default."
        overlap = "no"
        available = "yes"
        fix = ""
        if feature.startswith("sec_"):
            risk = "medium"
            evidence = "SEC sentiment features depend on filing availability and same-event alignment."
            fix = "Verify filing timestamp and same-event extraction path."
        if feature in {"current_filing_sentiment_available", "fund_snapshot_is_current_event"}:
            risk = "medium"
            evidence = "Availability/context flags can encode timing or panel-construction structure."
            fix = "Confirm these flags do not act as a shortcut for target timing."
        if any(token in feature for token in ["rel_return_", "vol", "gap", "shock", "volume", "drawdown", "zscore", "beta"]):
            risk = "medium"
            evidence = "Market feature must be built from windows ending strictly before effective_model_date."
            overlap = "possible_if_window_misaligned"
            fix = "Confirm rolling window ends at t-1 and does not bleed into forward return window."
        if feature in {"days_since_prior_event", "days_since_prior_same_event_type"}:
            risk = "low"
            evidence = "Recency features are likely safe if computed from prior events only."
        rows.append(
            {
                "feature_name": feature,
                "data_available_at_prediction_time": available,
                "possible_overlap_with_target": overlap,
                "leakage_risk_level": risk,
                "evidence": evidence,
                "recommended_fix": fix,
            }
        )
    leakage_df = pd.DataFrame(rows).sort_values(["leakage_risk_level", "feature_name"]).reset_index(drop=True)
    leakage_df.to_csv(output_csv, index=False)
    lines = [
        "# Leakage Notes",
        "",
        f"- Medium-risk features: `{int((leakage_df['leakage_risk_level'] == 'medium').sum())}`",
        f"- Low-risk features: `{int((leakage_df['leakage_risk_level'] == 'low').sum())}`",
        "- This audit is repo-logic based and should be paired with row-level spot checks if timing code changes again.",
    ]
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return leakage_df


def run_fold_stability(
    labeled_panel_df: pd.DataFrame,
    config: dict,
    selected_model_name: str,
    feature_family_df: pd.DataFrame,
    output_csv: Path,
    output_md: Path,
) -> pd.DataFrame:
    variant = build_variant(config)
    candidate_features = resolve_candidate_features(labeled_panel_df, config)
    explicit_exclusions = list(config["feature_exclusions"]["explicit"])
    holdout_start = str(config["holdout"]["start"])
    split_payload = make_event_v1_splits(
        df=labeled_panel_df,
        date_col="date",
        horizon_days=variant.horizon_days,
        n_splits=int(config["cv"]["n_splits"]),
        embargo_days=int(config["cv"]["embargo_days"]),
        holdout_start=holdout_start,
        min_train_dates=int(config["cv"]["min_train_dates"]),
    )
    global_candidates = [column for column in candidate_features if column not in explicit_exclusions]
    kept_global, _, _ = compute_global_feature_exclusions(labeled_panel_df, global_candidates, holdout_start)
    rank_rows = []
    sign_rows = []
    for fold_idx, fold in enumerate(split_payload["folds"], start=1):
        train_full = labeled_panel_df.iloc[fold["train_indices"]].copy()
        validation_full = labeled_panel_df.iloc[fold["validation_indices"]].copy()
        train_active, _ = apply_variant_label_mode(train_full, variant)
        validation_active, _ = apply_variant_label_mode(validation_full, variant)
        usable_features, _, _, _ = select_usable_features(train_active, kept_global)
        clipped_train, clipped_validation = clip_outliers(train_active, validation_active, usable_features)
        fitted_model, _ = fit_model(
            selected_model_name,
            clipped_train[usable_features],
            clipped_train["target"].astype(int),
        )
        imputer = fitted_model.named_steps["imputer"]
        model = fitted_model.named_steps["model"]
        transformed_validation = pd.DataFrame(
            imputer.transform(clipped_validation[usable_features]),
            columns=usable_features,
            index=clipped_validation.index,
        )
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(transformed_validation)
        shap_array = np.asarray(shap_values[-1] if isinstance(shap_values, list) else shap_values)
        if shap_array.ndim == 3:
            shap_array = shap_array[:, :, -1]
        importance = pd.Series(np.abs(shap_array).mean(axis=0), index=usable_features).sort_values(ascending=False)
        for rank, (feature, value) in enumerate(importance.items(), start=1):
            rank_rows.append(
                {
                    "fold_id": fold_idx,
                    "feature_name": feature,
                    "shap_rank": rank,
                    "mean_abs_shap": float(value),
                }
            )
            feature_series = transformed_validation[feature]
            shap_series = pd.Series(
                shap_array[:, transformed_validation.columns.get_loc(feature)],
                index=transformed_validation.index,
            )
            if feature_series.nunique(dropna=True) <= 1 or shap_series.nunique(dropna=True) <= 1:
                continue
            corr = spearmanr(feature_series, shap_series).statistic
            if corr is not None and not math.isnan(corr):
                sign_rows.append(
                    {
                        "fold_id": fold_idx,
                        "feature_name": feature,
                        "sign_positive": float(corr) >= 0,
                    }
                )
    rank_df = pd.DataFrame(rank_rows)
    sign_df = pd.DataFrame(sign_rows)
    summary = (
        rank_df.groupby("feature_name")
        .agg(
            mean_shap_rank=("shap_rank", "mean"),
            rank_std=("shap_rank", "std"),
            mean_abs_shap=("mean_abs_shap", "mean"),
        )
        .reset_index()
    )
    sign_summary = (
        sign_df.groupby("feature_name")["sign_positive"]
        .mean()
        .rename("sign_consistency")
        .reset_index()
    )
    stability_df = summary.merge(sign_summary, on="feature_name", how="left").fillna({"sign_consistency": 0.5})
    stability_df["stability_label"] = np.where(
        (stability_df["rank_std"].fillna(999.0) <= 5)
        & (
            stability_df["sign_consistency"].between(0.8, 1.0)
            | stability_df["sign_consistency"].between(0.0, 0.2)
        ),
        "stable",
        np.where(stability_df["rank_std"].fillna(999.0) <= 10, "mixed", "unstable"),
    )
    stability_df = stability_df.sort_values(["mean_shap_rank", "rank_std"]).reset_index(drop=True)
    stability_df.to_csv(output_csv, index=False)
    family_map = feature_family_df[["feature_name", "feature_family"]]
    family_stability = stability_df.merge(family_map, on="feature_name", how="left")
    family_summary = family_stability.groupby("feature_family")["mean_abs_shap"].sum().sort_values(ascending=False)
    lines = [
        "# Fold Stability Summary",
        "",
        f"- Stable features: `{int((stability_df['stability_label'] == 'stable').sum())}`",
        f"- Mixed features: `{int((stability_df['stability_label'] == 'mixed').sum())}`",
        f"- Unstable features: `{int((stability_df['stability_label'] == 'unstable').sum())}`",
        "",
        "Family-level mean absolute SHAP mass:",
    ]
    for family, total in family_summary.items():
        lines.append(f"- `{family}`: `{total:.4f}`")
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return stability_df


def build_feature_decision_sheet(
    shap_csv_path: Path,
    dependence_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    interaction_df: pd.DataFrame,
    leakage_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    output_csv: Path,
) -> pd.DataFrame:
    shap_df = pd.read_csv(shap_csv_path).reset_index(drop=True)
    shap_df["current_shap_rank"] = shap_df.index + 1
    cluster_neighbors = cluster_df.set_index("feature_name")["correlated_neighbors"].to_dict()
    leakage_map = leakage_df.set_index("feature_name")["leakage_risk_level"].to_dict()
    dependence_map = dependence_df.set_index("feature_name").to_dict(orient="index") if not dependence_df.empty else {}
    stability_map = stability_df.set_index("feature_name").to_dict(orient="index") if not stability_df.empty else {}
    interaction_map: dict[str, list[str]] = {}
    for row in interaction_df.to_dict(orient="records"):
        interaction_map.setdefault(row["feature_a"], []).append(row["feature_b"])
        interaction_map.setdefault(row["feature_b"], []).append(row["feature_a"])
    rows = []
    for row in shap_df.to_dict(orient="records"):
        feature = row["feature"]
        family = get_feature_family(feature)
        leakage = leakage_map.get(feature, "unknown")
        correlated_neighbors = cluster_neighbors.get(feature, "")
        dependence = dependence_map.get(feature, {})
        stability = stability_map.get(feature, {})
        if leakage == "medium":
            bucket = "leakage_suspect"
            next_action = "timing_spot_check"
        elif correlated_neighbors:
            bucket = "potential_duplicate"
            next_action = "cluster_representative_test"
        elif dependence.get("pattern_type") == "nonlinear_or_noisy":
            bucket = "potential_transform"
            next_action = "test_transform"
        elif stability.get("stability_label") == "unstable":
            bucket = "needs_investigation"
            next_action = "review_fold_dependence"
        elif int(row["current_shap_rank"]) <= 10:
            bucket = "core_keeper"
            next_action = "keep_and_monitor"
        else:
            bucket = "test_prune"
            next_action = "ablation_or_cluster_test"
        rows.append(
            {
                "feature_name": feature,
                "feature_family": family,
                "current_shap_rank": int(row["current_shap_rank"]),
                "economic_interpretation": FAMILY_DESCRIPTIONS[family],
                "likely_signal_mechanism": "event_panel_v2_baseline",
                "pattern_type": dependence.get("pattern_type", ""),
                "correlated_neighbors": correlated_neighbors,
                "interaction_candidates": ", ".join(sorted(set(interaction_map.get(feature, [])))),
                "leakage_risk": leakage,
                "decision_bucket": bucket,
                "next_action": next_action,
                "notes": "",
            }
        )
    decision_df = pd.DataFrame(rows).sort_values("current_shap_rank").reset_index(drop=True)
    decision_df.to_csv(output_csv, index=False)
    return decision_df


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    artifact_root = Path(args.artifact_root)
    baseline_dir = artifact_root / "baseline_v1"
    analysis_dir = artifact_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    labeled_panel_df = load_labeled_panel(config)
    baseline_results_df, _, _, shap_csv_path = freeze_baseline(labeled_panel_df, config, baseline_dir)
    selected_model_name = str(
        baseline_results_df.loc[baseline_results_df["is_selected_primary_model"], "model_name"].iloc[0]
    )

    holdout_payload = fit_selected_model_for_holdout(labeled_panel_df, config, selected_model_name)
    feature_family_df = build_feature_family_map(holdout_payload.usable_features)
    feature_family_df.to_csv(analysis_dir / "feature_family_map.csv", index=False)

    run_family_ablation(
        labeled_panel_df=labeled_panel_df,
        config=config,
        selected_model_name=selected_model_name,
        feature_family_df=feature_family_df,
        output_csv=analysis_dir / "ablation_results.csv",
        output_md=analysis_dir / "ablation_summary.md",
    )
    cluster_df = run_correlation_analysis(
        train_df=holdout_payload.train_df,
        usable_features=holdout_payload.usable_features,
        threshold=float(args.cluster_threshold),
        output_dir=analysis_dir,
    )

    shap_importance_df = pd.read_csv(shap_csv_path)
    priority_features = [
        feature
        for feature in [
            "beta_63d_to_sector",
            "sec_positive_prob",
            "rel_return_10d",
            "cash_ratio",
            "log_volume",
            "vol_ratio_21d_63d",
            "realized_vol_21d",
            "rel_return_5d",
            "volume_ratio_20d",
            "abs_return_shock_1d",
        ]
        if feature in holdout_payload.usable_features
    ]
    if len(priority_features) < int(args.top_features):
        for feature in shap_importance_df["feature"].tolist():
            if feature not in priority_features and feature in holdout_payload.usable_features:
                priority_features.append(feature)
            if len(priority_features) >= int(args.top_features):
                break

    dependence_df = run_shap_dependence(
        holdout_payload=holdout_payload,
        priority_features=priority_features,
        output_dir=analysis_dir / "shap_dependence",
        summary_path=analysis_dir / "shap_dependence_summary.md",
    )
    interaction_df = run_shap_interactions(
        holdout_payload=holdout_payload,
        interaction_pairs=INTERACTION_PAIRS,
        output_dir=analysis_dir / "shap_interactions",
        summary_path=analysis_dir / "shap_interaction_summary.md",
        sample_size=int(args.interaction_sample),
    )
    leakage_df = run_leakage_audit(
        feature_columns=holdout_payload.usable_features,
        output_csv=analysis_dir / "leakage_audit.csv",
        output_md=analysis_dir / "leakage_notes.md",
    )
    stability_df = run_fold_stability(
        labeled_panel_df=labeled_panel_df,
        config=config,
        selected_model_name=selected_model_name,
        feature_family_df=feature_family_df,
        output_csv=analysis_dir / "fold_stability.csv",
        output_md=analysis_dir / "fold_stability_summary.md",
    )
    build_feature_decision_sheet(
        shap_csv_path=shap_csv_path,
        dependence_df=dependence_df,
        cluster_df=cluster_df,
        interaction_df=interaction_df,
        leakage_df=leakage_df,
        stability_df=stability_df,
        output_csv=analysis_dir / "feature_decision_sheet.csv",
    )

    print(f"Baseline artifacts written to: {baseline_dir}")
    print(f"Analysis artifacts written to: {analysis_dir}")


if __name__ == "__main__":
    main()
