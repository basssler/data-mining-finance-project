"""Compare 5d vs 21d labels on event_panel_v2 using three anchor models."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from xgboost.core import XGBoostError

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.config_event_v1 import (
    DEFAULT_BENCHMARK_MODE,
    DEFAULT_CLIP_LOWER_QUANTILE,
    DEFAULT_CLIP_UPPER_QUANTILE,
    DEFAULT_EMBARGO_DAYS,
    DEFAULT_HOLDOUT_START,
    DEFAULT_N_SPLITS,
    DEFAULT_THRESHOLD,
    LAYER1_FEATURE_COLUMNS,
    LAYER2_V2_FEATURE_COLUMNS,
    PRICE_INPUT_PATH,
    REPORTS_RESULTS_DIR,
)
from src.evaluate_event_v1 import evaluate_classification_run
from src.labels_event_v1 import load_price_data, normalize_price_data
from src.panel_builder_event_v2 import (
    EVENT_CONTEXT_FEATURE_COLUMNS,
    EVENT_PANEL_V2_OUTPUT_PATH,
    SENTIMENT_CONTEXT_COLUMNS,
)
from src.paths import DOCS_DIR
from src.universe import get_project_sector_map
from src.validation_event_v1 import make_event_v1_splits

LABEL_COMPARISON_DOC_PATH = DOCS_DIR / "label_comparison_v1.md"
LABEL_COMPARISON_CSV_PATH = REPORTS_RESULTS_DIR / "label_comparison_v1.csv"
DEFAULT_EVENT_V2_MIN_TRAIN_DATES = 252
DEFAULT_MAX_MISSINGNESS_PCT = 20.0
DEFAULT_QUANTILE = 0.2


@dataclass(frozen=True)
class VariantSpec:
    variant_name: str
    horizon_days: int
    label_mode: str
    quantile: float | None = None


def has_clean_xgboost_gpu_support() -> bool:
    """Return True only when the local stack can use XGBoost GPU cleanly."""
    return importlib.util.find_spec("cupy") is not None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare label horizons on event_panel_v2.")
    parser.add_argument("--panel-path", default=str(EVENT_PANEL_V2_OUTPUT_PATH))
    parser.add_argument("--doc-path", default=str(LABEL_COMPARISON_DOC_PATH))
    parser.add_argument("--csv-path", default=str(LABEL_COMPARISON_CSV_PATH))
    parser.add_argument("--holdout-start", default=DEFAULT_HOLDOUT_START)
    parser.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS)
    parser.add_argument("--embargo-days", type=int, default=DEFAULT_EMBARGO_DAYS)
    parser.add_argument("--min-train-dates", type=int, default=DEFAULT_EVENT_V2_MIN_TRAIN_DATES)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--include-quantile", action="store_true")
    parser.add_argument("--quantile", type=float, default=DEFAULT_QUANTILE)
    return parser.parse_args()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _build_leave_one_out_mean(series: pd.Series, group_keys: pd.Series) -> pd.Series:
    numeric_series = pd.to_numeric(series, errors="coerce")
    group_sum = numeric_series.groupby(group_keys).transform("sum")
    group_count = numeric_series.groupby(group_keys).transform("count")
    denominator = group_count - 1
    loo_mean = pd.Series(np.nan, index=numeric_series.index, dtype="float64")
    valid = denominator > 0
    loo_mean.loc[valid] = (group_sum.loc[valid] - numeric_series.loc[valid]) / denominator.loc[valid]
    return loo_mean


def _compute_stock_forward_return(df: pd.DataFrame, horizon_days: int) -> pd.Series:
    if horizon_days == 5 and "forward_return_5d" in df.columns and df["forward_return_5d"].notna().any():
        return pd.to_numeric(df["forward_return_5d"], errors="coerce")
    future_price = df.groupby("ticker")["adj_close"].shift(-horizon_days)
    return (future_price / df["adj_close"]) - 1.0


def _resolve_sector_series(tickers: pd.Series) -> pd.Series:
    sector_map = get_project_sector_map()
    normalized = tickers.astype("string").str.strip().str.upper()
    sectors = normalized.map(sector_map)
    missing_tickers = sorted(normalized.loc[sectors.isna()].dropna().unique().tolist())
    if missing_tickers:
        raise ValueError(
            "sector_equal_weight_ex_self requires a local sector mapping for every ticker. "
            f"Missing sectors for: {', '.join(missing_tickers)}"
        )
    return sectors.astype("string")


def build_daily_label_table(
    prices_df: pd.DataFrame,
    horizon_days: int,
    benchmark_mode: str = DEFAULT_BENCHMARK_MODE,
) -> pd.DataFrame:
    if benchmark_mode not in {
        "sector_equal_weight_ex_self",
        "equal_weight_ex_self",
        "universe_equal_weight_ex_self",
    }:
        raise ValueError(f"Unsupported benchmark_mode: {benchmark_mode}")
    labels_df = prices_df.copy()
    labels_df["forward_return"] = _compute_stock_forward_return(labels_df, horizon_days=horizon_days)
    if benchmark_mode == "sector_equal_weight_ex_self":
        labels_df["sector"] = _resolve_sector_series(labels_df["ticker"])
        benchmark_group_keys = pd.MultiIndex.from_frame(labels_df[["date", "sector"]])
    else:
        benchmark_group_keys = labels_df["date"]
    labels_df["benchmark_forward_return"] = _build_leave_one_out_mean(
        labels_df["forward_return"],
        benchmark_group_keys,
    )
    labels_df["excess_forward_return"] = (
        labels_df["forward_return"] - labels_df["benchmark_forward_return"]
    )
    labels_df["target_sign"] = np.where(
        labels_df["excess_forward_return"].notna(),
        (labels_df["excess_forward_return"] > 0).astype(int),
        np.nan,
    )
    labels_df["target_sign"] = pd.Series(labels_df["target_sign"], index=labels_df.index).astype("Int64")
    return labels_df[
        ["ticker", "date", "forward_return", "benchmark_forward_return", "excess_forward_return", "target_sign"]
    ].copy()


def load_event_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"event_panel_v2 file was not found: {path}")
    df = pd.read_parquet(path)
    df["ticker"] = df["ticker"].astype("string")
    df["date"] = pd.to_datetime(df["effective_model_date"], errors="coerce").astype("datetime64[ns]")
    df["effective_model_date"] = pd.to_datetime(df["effective_model_date"], errors="coerce")
    for column in [
        "current_filing_fundamentals_available",
        "current_filing_sentiment_available",
        "fund_snapshot_is_current_event",
    ]:
        if column in df.columns:
            df[column] = df[column].astype("int64")
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


def candidate_feature_columns(panel_df: pd.DataFrame) -> list[str]:
    candidates = (
        list(LAYER1_FEATURE_COLUMNS)
        + list(LAYER2_V2_FEATURE_COLUMNS)
        + list(SENTIMENT_CONTEXT_COLUMNS)
        + list(EVENT_CONTEXT_FEATURE_COLUMNS)
        + [
            "current_filing_fundamentals_available",
            "current_filing_sentiment_available",
            "fund_snapshot_is_current_event",
        ]
    )
    return [column for column in candidates if column in panel_df.columns]


def attach_labels_to_event_panel(panel_df: pd.DataFrame, label_df: pd.DataFrame) -> pd.DataFrame:
    joined = panel_df.merge(
        label_df,
        on=["ticker", "date"],
        how="left",
        validate="many_to_one",
    )
    return joined.sort_values(["date", "ticker"]).reset_index(drop=True)


def compute_global_feature_exclusions(
    panel_df: pd.DataFrame,
    feature_columns: list[str],
    holdout_start: str,
) -> tuple[list[str], list[str], list[str]]:
    pre_holdout_df = panel_df.loc[panel_df["date"] < pd.Timestamp(holdout_start)].copy()
    if pre_holdout_df.empty:
        raise ValueError("No pre-holdout rows are available for feature checks.")
    all_missing = []
    constant = []
    kept = []
    for column in feature_columns:
        series = pre_holdout_df[column]
        if series.isna().all():
            all_missing.append(column)
            continue
        if series.dropna().nunique() <= 1:
            constant.append(column)
            continue
        kept.append(column)
    return kept, all_missing, constant


def clip_outliers(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clipped_train = train_df.copy()
    clipped_test = test_df.copy()
    for column in feature_columns:
        lower_bound = clipped_train[column].quantile(DEFAULT_CLIP_LOWER_QUANTILE)
        upper_bound = clipped_train[column].quantile(DEFAULT_CLIP_UPPER_QUANTILE)
        if pd.isna(lower_bound) or pd.isna(upper_bound):
            continue
        clipped_train[column] = clipped_train[column].clip(lower=lower_bound, upper=upper_bound)
        clipped_test[column] = clipped_test[column].clip(lower=lower_bound, upper=upper_bound)
    return clipped_train, clipped_test


def select_usable_features(
    train_df: pd.DataFrame,
    candidate_columns: list[str],
    max_missingness_pct: float = DEFAULT_MAX_MISSINGNESS_PCT,
) -> tuple[list[str], dict[str, float], list[str], list[str]]:
    usable_features = []
    missingness_by_feature = {}
    dropped_for_missingness = []
    dropped_for_constant = []
    for column in candidate_columns:
        missing_pct = float(train_df[column].isna().mean() * 100.0)
        missingness_by_feature[column] = missing_pct
        if train_df[column].notna().sum() == 0 or missing_pct > float(max_missingness_pct):
            dropped_for_missingness.append(column)
            continue
        if train_df[column].dropna().nunique() <= 1:
            dropped_for_constant.append(column)
            continue
        usable_features.append(column)
    if not usable_features:
        raise ValueError("No usable feature columns remained after filtering.")
    return usable_features, missingness_by_feature, dropped_for_missingness, dropped_for_constant


def resolve_max_missingness_pct(feature_exclusions: dict | None = None) -> float:
    """Return the configured missingness threshold, defaulting for legacy callers."""
    if feature_exclusions is None:
        return DEFAULT_MAX_MISSINGNESS_PCT
    value = feature_exclusions.get("max_missingness_pct", DEFAULT_MAX_MISSINGNESS_PCT)
    return float(value)


def build_logistic_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def build_random_forest_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=20,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )


def build_xgboost_pipeline(device: str) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=5,
                    reg_lambda=1.0,
                    random_state=42,
                    n_jobs=1,
                    tree_method="hist",
                    device=device,
                    eval_metric="logloss",
                ),
            ),
        ]
    )


def fit_model(
    model_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[Pipeline, str]:
    if model_name == "logistic_regression":
        model = build_logistic_pipeline()
        return model.fit(x_train, y_train), "cpu"
    if model_name == "random_forest":
        model = build_random_forest_pipeline()
        return model.fit(x_train, y_train), "cpu"
    if model_name == "xgboost":
        if has_clean_xgboost_gpu_support():
            try:
                model = build_xgboost_pipeline(device="cuda")
                return model.fit(x_train, y_train), "cuda"
            except (XGBoostError, ValueError):
                pass
        model = build_xgboost_pipeline(device="cpu")
        return model.fit(x_train, y_train), "cpu"
    raise ValueError(f"Unsupported model: {model_name}")


def safe_rank_ic(excess_returns: pd.Series, scores) -> float | None:
    score_series = pd.Series(scores, index=excess_returns.index, dtype="float64")
    valid = excess_returns.notna() & score_series.notna()
    if int(valid.sum()) < 3:
        return None
    statistic = spearmanr(excess_returns.loc[valid], score_series.loc[valid]).statistic
    if statistic is None or (isinstance(statistic, float) and math.isnan(statistic)):
        return None
    return float(statistic)


def evaluate_extended(scoring_df: pd.DataFrame, y_prob, threshold: float) -> dict:
    metrics = evaluate_classification_run(
        y_true=scoring_df["target"].astype(int),
        y_prob=y_prob,
        threshold=threshold,
    )
    metrics["rank_ic_spearman"] = safe_rank_ic(scoring_df["excess_forward_return"], y_prob)
    return metrics


def summarize_metric_dicts(metric_dicts: list[dict]) -> dict:
    keys = sorted(
        {
            key
            for metrics in metric_dicts
            for key in metrics.keys()
            if key not in {"confusion_matrix", "threshold", "row_count"}
        }
    )
    summary = {"fold_count": int(len(metric_dicts))}
    for key in keys:
        values = [metrics[key] for metrics in metric_dicts if metrics.get(key) is not None]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values, ddof=0))
        else:
            summary[f"{key}_mean"] = None
            summary[f"{key}_std"] = None
    return summary


def apply_variant_label_mode(
    df: pd.DataFrame,
    variant: VariantSpec,
    reference_excess: pd.Series | None = None,
) -> tuple[pd.DataFrame, dict]:
    prepared = df.copy()
    thresholds = {"lower": None, "upper": None}
    if variant.label_mode == "sign":
        prepared["target"] = prepared["target_sign"].astype("Int64")
        prepared = prepared.dropna(subset=["target", "excess_forward_return"]).copy()
        return prepared, thresholds
    if variant.label_mode == "quantile":
        if variant.quantile is None or reference_excess is None:
            raise ValueError("Quantile mode requires train-derived quantile thresholds.")
        lower = float(reference_excess.quantile(variant.quantile))
        upper = float(reference_excess.quantile(1.0 - variant.quantile))
        thresholds = {"lower": lower, "upper": upper}
        prepared = prepared.dropna(subset=["excess_forward_return"]).copy()
        keep_mask = (prepared["excess_forward_return"] <= lower) | (prepared["excess_forward_return"] >= upper)
        prepared = prepared.loc[keep_mask].copy()
        prepared["target"] = (prepared["excess_forward_return"] >= upper).astype("Int64")
        return prepared, thresholds
    raise ValueError(f"Unsupported label mode: {variant.label_mode}")


def build_variant_specs(include_quantile: bool, quantile: float) -> list[VariantSpec]:
    variants = [
        VariantSpec("event_v2_5d_sign", 5, "sign"),
        VariantSpec("event_v2_21d_sign", 21, "sign"),
    ]
    if include_quantile:
        suffix = str(int(round(quantile * 100)))
        variants.extend(
            [
                VariantSpec(f"event_v2_5d_quantile_{suffix}", 5, "quantile", quantile=quantile),
                VariantSpec(f"event_v2_21d_quantile_{suffix}", 21, "quantile", quantile=quantile),
            ]
        )
    return variants


def choose_best_model(model_records: list[dict]) -> str:
    ranked = []
    for record in model_records:
        auc_value = record["cv_auc_mean"]
        log_loss_value = record["cv_log_loss_mean"]
        ranked.append(
            (
                -(auc_value if auc_value is not None else float("-inf")),
                log_loss_value if log_loss_value is not None else float("inf"),
                record["model_name"],
            )
        )
    ranked.sort()
    return ranked[0][2]


def run_variant(
    variant: VariantSpec,
    panel_df: pd.DataFrame,
    candidate_features: list[str],
    global_dead_features: list[str],
    global_constant_features: list[str],
    threshold: float,
    holdout_start: str,
    n_splits: int,
    embargo_days: int,
    min_train_dates: int,
) -> list[dict]:
    split_payload = make_event_v1_splits(
        df=panel_df,
        date_col="date",
        horizon_days=variant.horizon_days,
        n_splits=n_splits,
        embargo_days=embargo_days,
        holdout_start=holdout_start,
        min_train_dates=min_train_dates,
    )
    model_rows = []
    for model_name in ["logistic_regression", "random_forest", "xgboost"]:
        fold_metrics = []
        last_usable_features: list[str] = []
        last_dropped_missing: list[str] = []
        last_dropped_constant: list[str] = []
        last_missingness: dict[str, float] = {}
        for fold in split_payload["folds"]:
            train_full = panel_df.iloc[fold["train_indices"]].copy()
            validation_full = panel_df.iloc[fold["validation_indices"]].copy()
            if variant.label_mode == "quantile":
                reference_excess = train_full["excess_forward_return"].dropna()
                train_active, _ = apply_variant_label_mode(train_full, variant, reference_excess=reference_excess)
                validation_active, _ = apply_variant_label_mode(validation_full, variant, reference_excess=reference_excess)
            else:
                train_active, _ = apply_variant_label_mode(train_full, variant)
                validation_active, _ = apply_variant_label_mode(validation_full, variant)
            if train_active["target"].nunique(dropna=True) < 2 or validation_active["target"].nunique(dropna=True) < 2:
                continue
            usable_features, missingness_by_feature, dropped_missing, dropped_constant = select_usable_features(
                train_active,
                candidate_features,
            )
            clipped_train, clipped_validation = clip_outliers(train_active, validation_active, usable_features)
            fitted_model, _ = fit_model(
                model_name,
                clipped_train[usable_features],
                clipped_train["target"].astype(int),
            )
            y_prob = fitted_model.predict_proba(clipped_validation[usable_features])[:, 1]
            fold_metrics.append(evaluate_extended(clipped_validation, y_prob, threshold=threshold))
            last_usable_features = usable_features
            last_dropped_missing = dropped_missing
            last_dropped_constant = dropped_constant
            last_missingness = missingness_by_feature
        if not fold_metrics:
            raise ValueError(f"No valid CV folds were produced for {variant.variant_name} / {model_name}.")

        holdout_train_full = panel_df.iloc[split_payload["holdout"]["train_indices"]].copy()
        holdout_full = panel_df.iloc[split_payload["holdout"]["holdout_indices"]].copy()
        if variant.label_mode == "quantile":
            reference_excess = holdout_train_full["excess_forward_return"].dropna()
            holdout_train_active, thresholds = apply_variant_label_mode(
                holdout_train_full,
                variant,
                reference_excess=reference_excess,
            )
            holdout_active, _ = apply_variant_label_mode(
                holdout_full,
                variant,
                reference_excess=reference_excess,
            )
        else:
            holdout_train_active, thresholds = apply_variant_label_mode(holdout_train_full, variant)
            holdout_active, _ = apply_variant_label_mode(holdout_full, variant)
        holdout_usable_features, _, holdout_dropped_missing, holdout_dropped_constant = select_usable_features(
            holdout_train_active,
            candidate_features,
        )
        clipped_train, clipped_holdout = clip_outliers(holdout_train_active, holdout_active, holdout_usable_features)
        fitted_model, backend = fit_model(
            model_name,
            clipped_train[holdout_usable_features],
            clipped_train["target"].astype(int),
        )
        holdout_prob = fitted_model.predict_proba(clipped_holdout[holdout_usable_features])[:, 1]
        holdout_metrics = evaluate_extended(clipped_holdout, holdout_prob, threshold=threshold)
        cv_summary = summarize_metric_dicts(fold_metrics)
        model_rows.append(
            {
                "variant_name": variant.variant_name,
                "horizon_days": variant.horizon_days,
                "label_mode": variant.label_mode,
                "quantile": variant.quantile,
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
                "usable_feature_count_last_fold": len(last_usable_features),
                "usable_feature_columns_last_fold": json.dumps(last_usable_features),
                "global_dead_feature_exclusions": json.dumps(global_dead_features),
                "global_constant_feature_exclusions": json.dumps(global_constant_features),
                "fold_missingness_exclusions_last_fold": json.dumps(sorted(set(last_dropped_missing))),
                "fold_constant_exclusions_last_fold": json.dumps(sorted(set(last_dropped_constant))),
                "holdout_missingness_exclusions": json.dumps(sorted(set(holdout_dropped_missing))),
                "holdout_constant_exclusions": json.dumps(sorted(set(holdout_dropped_constant))),
                "xgboost_backend": backend if model_name == "xgboost" else "cpu",
                "holdout_quantile_thresholds": json.dumps(thresholds),
                "split_min_train_dates": min_train_dates,
                "split_n_splits": n_splits,
                "split_embargo_days": embargo_days,
                "holdout_start": holdout_start,
                "train_missingness_by_feature_pct_last_fold": json.dumps(last_missingness),
            }
        )
    best_model_name = choose_best_model(model_rows)
    for row in model_rows:
        row["is_best_model_for_variant"] = row["model_name"] == best_model_name
    return model_rows


def determine_recommendation(sign_best_rows: pd.DataFrame) -> str:
    row_5d = sign_best_rows.loc[sign_best_rows["horizon_days"] == 5].iloc[0]
    row_21d = sign_best_rows.loc[sign_best_rows["horizon_days"] == 21].iloc[0]
    if (
        (row_21d["cv_auc_mean"] > row_5d["cv_auc_mean"])
        and (row_21d["cv_log_loss_mean"] <= row_5d["cv_log_loss_mean"])
        and (row_21d["holdout_auc"] >= row_5d["holdout_auc"])
        and (row_21d["holdout_log_loss"] <= row_5d["holdout_log_loss"])
    ):
        return "switch to 21-day as primary"
    if (
        (row_5d["cv_auc_mean"] > row_21d["cv_auc_mean"])
        and (row_5d["cv_log_loss_mean"] <= row_21d["cv_log_loss_mean"])
        and (row_5d["holdout_auc"] >= row_21d["holdout_auc"])
        and (row_5d["holdout_log_loss"] <= row_21d["holdout_log_loss"])
    ):
        return "keep 5-day as primary"
    return "neither horizon is yet strong enough"


def format_metric(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.4f}"


def build_markdown_report(
    result_df: pd.DataFrame,
    feature_candidates: list[str],
    global_dead_features: list[str],
    global_constant_features: list[str],
    recommendation: str,
) -> str:
    sign_best_rows = (
        result_df.loc[(result_df["label_mode"] == "sign") & (result_df["is_best_model_for_variant"])]
        .sort_values("horizon_days")
        .reset_index(drop=True)
    )
    sign_rows = []
    for _, row in sign_best_rows.iterrows():
        sign_rows.append(
            "| "
            + " | ".join(
                [
                    f"{int(row['horizon_days'])}-day",
                    str(row["model_name"]),
                    format_metric(row["cv_auc_mean"]),
                    format_metric(row["cv_log_loss_mean"]),
                    format_metric(row["holdout_auc"]),
                    format_metric(row["holdout_log_loss"]),
                    format_metric(row["holdout_precision"]),
                    format_metric(row["holdout_recall"]),
                    format_metric(row["holdout_rank_ic"]),
                ]
            )
            + " |"
        )
    model_rows = []
    for _, row in result_df.sort_values(["variant_name", "model_name"]).iterrows():
        model_rows.append(
            "| "
            + " | ".join(
                [
                    str(row["variant_name"]),
                    str(row["model_name"]),
                    format_metric(row["cv_auc_mean"]),
                    format_metric(row["cv_log_loss_mean"]),
                    format_metric(row["holdout_auc"]),
                    format_metric(row["holdout_log_loss"]),
                    format_metric(row["holdout_precision"]),
                    format_metric(row["holdout_recall"]),
                    format_metric(row["holdout_rank_ic"]),
                    "yes" if row["is_best_model_for_variant"] else "",
                ]
            )
            + " |"
        )
    lines = [
        "# Label Comparison V1",
        "",
        "## Scope",
        "",
        "- Primary panel: `event_panel_v2` only",
        "- Holdout policy: unchanged 2024 holdout",
        "- CV policy: expanding purged date splits reused from event_v1, with `min_train_dates=252` because the event panel has fewer unique dates than the daily panel",
        "- Model families: logistic regression, random forest, XGBoost",
        "",
        "## Feature Exclusions",
        "",
        f"- Candidate feature count before exclusions: `{len(feature_candidates)}`",
        f"- Global all-missing exclusions: `{', '.join(global_dead_features) if global_dead_features else 'none'}`",
        f"- Global constant exclusions: `{', '.join(global_constant_features) if global_constant_features else 'none'}`",
        "- Additional fold-level exclusions were applied when train-fold missingness exceeded 20% or a feature became constant inside a training fold.",
        "",
        "## Best Sign-Horizon Comparison",
        "",
        "| Horizon | Best Model | Mean CV AUC | Mean CV Log Loss | Holdout AUC | Holdout Log Loss | Holdout Precision | Holdout Recall | Holdout Rank IC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        *sign_rows,
        "",
        "## Full Model Table",
        "",
        "| Variant | Model | Mean CV AUC | Mean CV Log Loss | Holdout AUC | Holdout Log Loss | Holdout Precision | Holdout Recall | Holdout Rank IC | Best For Variant |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        *model_rows,
        "",
        "## Recommendation",
        "",
        f"- Direct recommendation: **{recommendation}**",
        "- Recommendation rule: prefer the horizon whose best model improves CV AUC and CV log loss without reversing the direction on the 2024 holdout.",
        "- If neither horizon wins cleanly across both CV and holdout, treat the horizon choice as unresolved rather than forcing a switch.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    ensure_parent_dir(Path(args.doc_path))
    ensure_parent_dir(Path(args.csv_path))
    print(f"Loading event_panel_v2 from: {args.panel_path}")
    panel_df = load_event_panel(Path(args.panel_path))
    print(f"Loading prices from: {PRICE_INPUT_PATH}")
    prices_df = normalize_price_data(load_price_data(PRICE_INPUT_PATH))
    feature_candidates = candidate_feature_columns(panel_df)
    all_rows = []
    global_dead_features_final: list[str] = []
    global_constant_features_final: list[str] = []
    for variant in build_variant_specs(args.include_quantile, args.quantile):
        print(f"\\nBuilding labels for variant: {variant.variant_name}")
        label_df = build_daily_label_table(prices_df, horizon_days=variant.horizon_days)
        labeled_panel_df = attach_labels_to_event_panel(panel_df, label_df)
        kept_features, global_dead_features, global_constant_features = compute_global_feature_exclusions(
            labeled_panel_df,
            feature_candidates,
            holdout_start=args.holdout_start,
        )
        global_dead_features_final = global_dead_features
        global_constant_features_final = global_constant_features
        print(f"Running model comparison for: {variant.variant_name}")
        all_rows.extend(
            run_variant(
                variant=variant,
                panel_df=labeled_panel_df,
                candidate_features=kept_features,
                global_dead_features=global_dead_features,
                global_constant_features=global_constant_features,
                threshold=args.threshold,
                holdout_start=args.holdout_start,
                n_splits=args.n_splits,
                embargo_days=args.embargo_days,
                min_train_dates=args.min_train_dates,
            )
        )
    result_df = pd.DataFrame(all_rows).sort_values(["variant_name", "model_name"]).reset_index(drop=True)
    sign_best_rows = result_df.loc[
        (result_df["label_mode"] == "sign") & (result_df["is_best_model_for_variant"])
    ].copy()
    recommendation = determine_recommendation(sign_best_rows)
    print(f"Saving CSV comparison to: {args.csv_path}")
    result_df.to_csv(args.csv_path, index=False)
    markdown_text = build_markdown_report(
        result_df=result_df,
        feature_candidates=feature_candidates,
        global_dead_features=global_dead_features_final,
        global_constant_features=global_constant_features_final,
        recommendation=recommendation,
    )
    print(f"Writing Markdown comparison to: {args.doc_path}")
    Path(args.doc_path).write_text(markdown_text, encoding="utf-8")
    print("\\nLabel Comparison Summary")
    print("-" * 60)
    for _, row in result_df.loc[result_df["is_best_model_for_variant"]].sort_values(["label_mode", "horizon_days"]).iterrows():
        print(
            f"{row['variant_name']:<28} best={row['model_name']:<20} "
            f"cv_auc={format_metric(row['cv_auc_mean'])} holdout_auc={format_metric(row['holdout_auc'])}"
        )
    print(f"\\nRecommendation: {recommendation}")


if __name__ == "__main__":
    main()
