from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.label_comparison_event_v2 import (
    apply_variant_label_mode,
    clip_outliers,
    compute_global_feature_exclusions,
    evaluate_extended,
    fit_model,
    select_usable_features,
)
from src.validation_event_v1 import make_event_v1_splits

BEST_HOLDOUT_SORT_COLUMNS = [
    "holdout_auc_sort",
    "cv_auc_mean_sort",
    "cv_log_loss_mean_sort",
    "interaction_style",
    "model_name",
]
BEST_HOLDOUT_ASCENDING = [False, False, True, True, True]

BEST_STABLE_SORT_COLUMNS = [
    "holdout_auc_sort",
    "worst_fold_auc_sort",
    "cv_auc_std_sort",
    "dominant_feature_top3_folds_sort",
    "cv_auc_mean_sort",
    "cv_log_loss_mean_sort",
    "interaction_style",
    "model_name",
]
BEST_STABLE_ASCENDING = [False, False, True, True, False, True, True, True]

VIEW_ORDER = {
    "trainer_selected": 0,
    "best_holdout": 1,
    "best_stable": 2,
}


def _prepare_sort_columns(df: pd.DataFrame) -> pd.DataFrame:
    sortable = df.copy()
    sortable["holdout_auc_sort"] = sortable["holdout_auc"].fillna(float("-inf"))
    sortable["cv_auc_mean_sort"] = sortable["cv_auc_mean"].fillna(float("-inf"))
    sortable["cv_log_loss_mean_sort"] = sortable["cv_log_loss_mean"].fillna(float("inf"))
    sortable["cv_auc_std_sort"] = sortable["cv_auc_std"].fillna(float("inf"))
    sortable["worst_fold_auc_sort"] = sortable["worst_fold_auc"].fillna(float("-inf"))
    sortable["dominant_feature_top3_folds_sort"] = sortable["dominant_feature_top3_folds"].fillna(float("inf"))
    sortable["interaction_style"] = sortable["interaction_style"].fillna("")
    return sortable


def pick_best_holdout_row(df: pd.DataFrame) -> pd.Series:
    sortable = _prepare_sort_columns(df)
    best_index = sortable.sort_values(BEST_HOLDOUT_SORT_COLUMNS, ascending=BEST_HOLDOUT_ASCENDING).index[0]
    return df.loc[best_index]


def pick_best_stable_row(df: pd.DataFrame) -> pd.Series:
    sortable = _prepare_sort_columns(df)
    best_index = sortable.sort_values(BEST_STABLE_SORT_COLUMNS, ascending=BEST_STABLE_ASCENDING).index[0]
    return df.loc[best_index]


def compute_model_stability_artifacts(
    panel_df: pd.DataFrame,
    variant,
    model_names: list[str],
    candidate_features: list[str],
    explicit_exclusions: list[str],
    holdout_start: str,
    n_splits: int,
    embargo_days: int,
    min_train_dates: int,
    threshold: float,
    max_missingness_pct: float,
    permutation_repeats: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    kept_global, _, _ = compute_global_feature_exclusions(
        panel_df,
        global_candidates,
        holdout_start=holdout_start,
    )

    summary_rows: list[dict] = []
    fold_metric_rows: list[dict] = []
    concentration_rows: list[dict] = []
    for model_name in model_names:
        fold_aucs: list[float] = []
        top_feature_counter: Counter[str] = Counter()
        model_fold_count = 0
        for fold_number, fold in enumerate(split_payload["folds"], start=1):
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
            validation_target = clipped_validation["target"].astype(int)
            validation_prob = fitted_model.predict_proba(clipped_validation[usable_features])[:, 1]
            fold_metrics = evaluate_extended(clipped_validation, validation_prob, threshold=threshold)
            fold_auc = fold_metrics.get("auc_roc")
            if fold_auc is not None:
                fold_aucs.append(float(fold_auc))
            model_fold_count += 1
            fold_metric_rows.append(
                {
                    "model_name": model_name,
                    "fold_number": fold_number,
                    "fold_auc": fold_metrics.get("auc_roc"),
                    "fold_log_loss": fold_metrics.get("log_loss"),
                    "fold_precision": fold_metrics.get("precision"),
                    "fold_recall": fold_metrics.get("recall"),
                    "fold_rank_ic": fold_metrics.get("rank_ic_spearman"),
                    "fold_row_count": fold_metrics.get("row_count"),
                    "usable_feature_count": len(usable_features),
                    "usable_features": json_dumps_sorted(usable_features),
                    "dropped_missing": json_dumps_sorted(dropped_missing),
                    "dropped_constant": json_dumps_sorted(dropped_constant),
                    "missingness_by_feature_pct": json_dumps_obj(missingness_by_feature),
                }
            )

            importance = permutation_importance(
                fitted_model,
                clipped_validation[usable_features],
                validation_target,
                scoring="roc_auc",
                n_repeats=permutation_repeats,
                random_state=42,
                n_jobs=1,
            )
            fold_importance_df = (
                pd.DataFrame(
                    {
                        "feature": usable_features,
                        "importance_mean": importance.importances_mean,
                        "importance_std": importance.importances_std,
                    }
                )
                .sort_values(["importance_mean", "feature"], ascending=[False, True])
                .reset_index(drop=True)
            )
            for rank, (_, row) in enumerate(fold_importance_df.head(3).iterrows(), start=1):
                feature_name = str(row["feature"])
                top_feature_counter[feature_name] += 1
                concentration_rows.append(
                    {
                        "model_name": model_name,
                        "fold_number": fold_number,
                        "rank": rank,
                        "feature": feature_name,
                        "importance_mean": float(row["importance_mean"]),
                        "importance_std": float(row["importance_std"]),
                    }
                )

        if model_fold_count == 0:
            raise ValueError(f"No valid fold metrics were produced for model: {model_name}")
        dominant_feature_name = None
        dominant_feature_top3_folds = 0
        if top_feature_counter:
            dominant_feature_name, dominant_feature_top3_folds = sorted(
                top_feature_counter.items(),
                key=lambda item: (-item[1], item[0]),
            )[0]
        summary_rows.append(
            {
                "model_name": model_name,
                "cv_auc_std": float(np.std(fold_aucs, ddof=0)) if fold_aucs else np.nan,
                "worst_fold_auc": float(min(fold_aucs)) if fold_aucs else np.nan,
                "diagnostic_fold_count": int(model_fold_count),
                "dominant_feature_name": dominant_feature_name,
                "dominant_feature_top3_folds": int(dominant_feature_top3_folds),
            }
        )

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(fold_metric_rows),
        pd.DataFrame(concentration_rows),
    )


def build_family_view_summary(matrix_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for family_name, family_df in matrix_df.groupby("experiment_family", sort=True):
        selected_candidates = family_df.loc[family_df["is_selected_primary_model"] == True]  # noqa: E712
        if not selected_candidates.empty:
            trainer_selected = pick_best_holdout_row(selected_candidates)
            rows.append(_tag_view_row(trainer_selected, family_name, "trainer_selected"))
        rows.append(_tag_view_row(pick_best_holdout_row(family_df), family_name, "best_holdout"))
        rows.append(_tag_view_row(pick_best_stable_row(family_df), family_name, "best_stable"))
    if not rows:
        return pd.DataFrame()
    summary_df = pd.DataFrame(rows)
    summary_df["view_order"] = summary_df["view_name"].map(VIEW_ORDER).astype("int64")
    return summary_df.sort_values(["experiment_family", "view_order"]).reset_index(drop=True)


def _tag_view_row(row: pd.Series, family_name: str, view_name: str) -> dict:
    tagged = row.to_dict()
    tagged["experiment_family"] = family_name
    tagged["view_name"] = view_name
    return tagged


def json_dumps_sorted(values: list[str]) -> str:
    return json_dumps_obj(sorted(set(values)))


def json_dumps_obj(payload) -> str:
    import json

    return json.dumps(payload, sort_keys=True)
