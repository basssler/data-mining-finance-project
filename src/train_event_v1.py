"""Train the additive event_v1 panels with shared validation and reporting."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.config_event_v1 import (
    DEFAULT_CLIP_LOWER_QUANTILE,
    DEFAULT_CLIP_UPPER_QUANTILE,
    DEFAULT_EMBARGO_DAYS,
    DEFAULT_HOLDOUT_START,
    DEFAULT_HORIZON_DAYS,
    DEFAULT_MAX_MISSINGNESS_PCT,
    DEFAULT_MIN_TRAIN_DATES,
    DEFAULT_N_SPLITS,
    DEFAULT_THRESHOLD,
    EVENT_V1_FULL_METRICS_PATH,
    EVENT_V1_LAYER1_LAYER2_METRICS_PATH,
    EVENT_V1_LAYER1_METRICS_PATH,
    EVENT_V1_SUMMARY_PATH,
    PANEL_CHOICES,
    TARGET_COLUMN,
    ensure_event_v1_directories,
    get_candidate_feature_columns,
    get_metrics_output_path,
    get_panel_path,
    get_predictions_output_path,
)
from src.evaluate_event_v1 import (
    evaluate_classification_run,
    summarize_fold_metrics,
    write_json_report,
    write_markdown_report,
)
from src.validation_event_v1 import make_event_v1_splits


def parse_args() -> argparse.Namespace:
    """Parse CLI options for event_v1 model training."""
    parser = argparse.ArgumentParser(description="Train the event_v1 experiment lane.")
    parser.add_argument("--panel", choices=PANEL_CHOICES, required=True)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    return parser.parse_args()


def load_panel(path: Path, candidate_features: list[str]) -> pd.DataFrame:
    """Load a modeling panel and validate the required schema."""
    if not path.exists():
        raise FileNotFoundError(f"Modeling panel file was not found: {path}")

    df = pd.read_parquet(path)
    required_columns = ["ticker", "date", TARGET_COLUMN] + candidate_features
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError("Panel is missing required columns: " + ", ".join(missing_columns))
    return df.copy()


def prepare_modeling_data(df: pd.DataFrame, candidate_features: list[str]) -> pd.DataFrame:
    """Normalize dtypes and keep rows with a valid event_v1 target."""
    prepared = df.copy()
    prepared["ticker"] = prepared["ticker"].astype("string")
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    prepared[TARGET_COLUMN] = pd.to_numeric(prepared[TARGET_COLUMN], errors="coerce").astype("Int64")

    for column in candidate_features:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    prepared = prepared.dropna(subset=["ticker", "date", TARGET_COLUMN]).copy()
    prepared = prepared.sort_values(["date", "ticker"]).reset_index(drop=True)
    return prepared


def select_usable_features(
    train_df: pd.DataFrame,
    candidate_columns: list[str],
) -> tuple[list[str], dict[str, float]]:
    """Keep features with observed values and acceptable missingness."""
    usable_features = []
    missingness_by_feature = {}
    for column in candidate_columns:
        missing_pct = float(train_df[column].isna().mean() * 100)
        missingness_by_feature[column] = missing_pct
        if train_df[column].notna().any() and missing_pct <= DEFAULT_MAX_MISSINGNESS_PCT:
            usable_features.append(column)

    if not usable_features:
        raise ValueError("No usable event_v1 feature columns were found in the training data.")

    return usable_features, missingness_by_feature


def clip_outliers(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clip feature extremes using train-set quantiles only."""
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


def build_models() -> dict[str, Pipeline]:
    """Return the shared model family for event_v1."""
    return {
        "logistic_regression": Pipeline(
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
        ),
        "random_forest": Pipeline(
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
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_depth=6,
                        learning_rate=0.05,
                        max_iter=300,
                        min_samples_leaf=50,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def build_prediction_frame(
    scoring_df: pd.DataFrame,
    y_prob,
    y_pred,
    panel_name: str,
    model_name: str,
    prediction_set: str,
    fold_number: int | None,
) -> pd.DataFrame:
    """Return a tidy prediction frame for validation or holdout rows."""
    frame = scoring_df[["ticker", "date", TARGET_COLUMN]].copy()
    if "excess_forward_return_5d" in scoring_df.columns:
        frame["excess_forward_return_5d"] = scoring_df["excess_forward_return_5d"]
    frame["panel_name"] = panel_name
    frame["model_name"] = model_name
    frame["prediction_set"] = prediction_set
    frame["fold_number"] = fold_number
    frame["predicted_probability"] = y_prob
    frame["predicted_label"] = y_pred
    return frame


def run_cross_validation(
    df: pd.DataFrame,
    panel_name: str,
    candidate_features: list[str],
    threshold: float,
) -> tuple[dict, dict]:
    """Run expanding purged CV for each model and collect fold predictions."""
    split_payload = make_event_v1_splits(
        df=df,
        date_col="date",
        horizon_days=DEFAULT_HORIZON_DAYS,
        n_splits=DEFAULT_N_SPLITS,
        embargo_days=DEFAULT_EMBARGO_DAYS,
        holdout_start=DEFAULT_HOLDOUT_START,
        min_train_dates=DEFAULT_MIN_TRAIN_DATES,
    )

    results = {}
    model_predictions = {}
    for model_name, model in build_models().items():
        fold_results = []
        fold_prediction_frames = []

        for fold in split_payload["folds"]:
            train_df = df.iloc[fold["train_indices"]].copy()
            validation_df = df.iloc[fold["validation_indices"]].copy()
            usable_feature_columns, missingness_by_feature = select_usable_features(
                train_df,
                candidate_features,
            )
            clipped_train_df, clipped_validation_df = clip_outliers(
                train_df,
                validation_df,
                usable_feature_columns,
            )

            x_train = clipped_train_df[usable_feature_columns]
            y_train = clipped_train_df[TARGET_COLUMN].astype(int)
            x_validation = clipped_validation_df[usable_feature_columns]
            y_validation = clipped_validation_df[TARGET_COLUMN].astype(int)

            fitted_model = model.fit(x_train, y_train)
            y_prob = fitted_model.predict_proba(x_validation)[:, 1]
            y_pred = fitted_model.predict(x_validation)

            metrics = evaluate_classification_run(
                y_true=y_validation,
                y_prob=y_prob,
                threshold=threshold,
            )
            fold_results.append(
                {
                    "fold_number": fold["fold_number"],
                    "date_metadata": fold["date_metadata"],
                    "metrics": metrics,
                    "usable_feature_columns": usable_feature_columns,
                    "train_missingness_by_feature_pct": missingness_by_feature,
                }
            )
            fold_prediction_frames.append(
                build_prediction_frame(
                    scoring_df=clipped_validation_df,
                    y_prob=y_prob,
                    y_pred=y_pred,
                    panel_name=panel_name,
                    model_name=model_name,
                    prediction_set="validation",
                    fold_number=fold["fold_number"],
                )
            )

        cv_summary = summarize_fold_metrics([fold["metrics"] for fold in fold_results])
        results[model_name] = {
            "fold_results": fold_results,
            "cv_summary": cv_summary,
        }
        model_predictions[model_name] = pd.concat(fold_prediction_frames, ignore_index=True)

    return split_payload, {"results": results, "predictions": model_predictions}


def choose_best_model(model_results: dict) -> str:
    """Choose the best model by mean CV AUC, then mean CV log loss."""
    ranked_models = []
    for model_name, payload in model_results.items():
        summary = payload["cv_summary"]
        auc_value = summary.get("auc_roc_mean")
        log_loss_value = summary.get("log_loss_mean")
        ranked_models.append(
            (
                -(auc_value if auc_value is not None else float("-inf")),
                log_loss_value if log_loss_value is not None else float("inf"),
                model_name,
            )
        )
    ranked_models = sorted(ranked_models)
    return ranked_models[0][2]


def fit_best_model_on_holdout(
    df: pd.DataFrame,
    panel_name: str,
    best_model_name: str,
    candidate_features: list[str],
    split_payload: dict,
    threshold: float,
) -> tuple[dict, pd.DataFrame]:
    """Refit the selected best model on pre-holdout data and score 2024 holdout."""
    train_df = df.iloc[split_payload["holdout"]["train_indices"]].copy()
    holdout_df = df.iloc[split_payload["holdout"]["holdout_indices"]].copy()

    usable_feature_columns, missingness_by_feature = select_usable_features(train_df, candidate_features)
    clipped_train_df, clipped_holdout_df = clip_outliers(train_df, holdout_df, usable_feature_columns)

    model = build_models()[best_model_name]
    x_train = clipped_train_df[usable_feature_columns]
    y_train = clipped_train_df[TARGET_COLUMN].astype(int)
    x_holdout = clipped_holdout_df[usable_feature_columns]
    y_holdout = clipped_holdout_df[TARGET_COLUMN].astype(int)

    fitted_model = model.fit(x_train, y_train)
    y_prob = fitted_model.predict_proba(x_holdout)[:, 1]
    y_pred = fitted_model.predict(x_holdout)

    holdout_metrics = evaluate_classification_run(
        y_true=y_holdout,
        y_prob=y_prob,
        threshold=threshold,
    )
    holdout_payload = {
        "holdout_metrics": holdout_metrics,
        "usable_feature_columns": usable_feature_columns,
        "train_missingness_by_feature_pct": missingness_by_feature,
        "date_metadata": split_payload["holdout"]["date_metadata"],
    }
    prediction_frame = build_prediction_frame(
        scoring_df=clipped_holdout_df,
        y_prob=y_prob,
        y_pred=y_pred,
        panel_name=panel_name,
        model_name=best_model_name,
        prediction_set="holdout",
        fold_number=None,
    )
    return holdout_payload, prediction_frame


def save_best_model_predictions(
    validation_predictions: pd.DataFrame,
    holdout_predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save tidy validation and holdout predictions for the selected model."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([validation_predictions, holdout_predictions], ignore_index=True)
    combined.to_parquet(output_path, index=False)


def build_metrics_payload(
    panel_name: str,
    panel_path: Path,
    candidate_features: list[str],
    split_payload: dict,
    model_results: dict,
    best_model_name: str,
    holdout_payload: dict,
    threshold: float,
) -> dict:
    """Assemble the event_v1 metrics JSON payload."""
    fold_metadata = [
        {
            "fold_number": fold["fold_number"],
            "date_metadata": fold["date_metadata"],
            "train_date_count": fold["train_date_count"],
            "validation_date_count": fold["validation_date_count"],
            "purged_date_count": fold["purged_date_count"],
        }
        for fold in split_payload["folds"]
    ]
    holdout_metadata = {
        "date_metadata": split_payload["holdout"]["date_metadata"],
        "train_date_count": split_payload["holdout"]["train_date_count"],
        "holdout_date_count": split_payload["holdout"]["holdout_date_count"],
        "purged_date_count": split_payload["holdout"]["purged_date_count"],
    }

    return {
        "panel_name": panel_name,
        "panel_path": str(panel_path),
        "target_column": TARGET_COLUMN,
        "selection_rule": (
            "highest mean CV AUC-ROC, tie-break lowest mean CV log loss, "
            "final tie-break model name"
        ),
        "candidate_feature_columns": candidate_features,
        "validation_policy": {
            "date_column": split_payload["date_column"],
            "horizon_days": split_payload["horizon_days"],
            "embargo_days": split_payload["embargo_days"],
            "min_train_dates": split_payload["min_train_dates"],
            "n_splits": split_payload["n_splits"],
            "holdout_start": split_payload["holdout_start"],
            "folds": fold_metadata,
            "holdout": holdout_metadata,
        },
        "preprocessing": {
            "max_missingness_pct": DEFAULT_MAX_MISSINGNESS_PCT,
            "clip_lower_quantile": DEFAULT_CLIP_LOWER_QUANTILE,
            "clip_upper_quantile": DEFAULT_CLIP_UPPER_QUANTILE,
            "threshold": threshold,
        },
        "model_results": model_results,
        "best_model": {
            "model_name": best_model_name,
            "cv_summary": model_results[best_model_name]["cv_summary"],
            "holdout": holdout_payload,
        },
    }


def maybe_write_family_summary() -> None:
    """Write the family-level summary once all three event_v1 runs exist."""
    metric_paths = {
        "event_v1_layer1": EVENT_V1_LAYER1_METRICS_PATH,
        "event_v1_layer1_layer2": EVENT_V1_LAYER1_LAYER2_METRICS_PATH,
        "event_v1_full": EVENT_V1_FULL_METRICS_PATH,
    }
    if not all(path.exists() for path in metric_paths.values()):
        return

    records = {}
    for panel_name, path in metric_paths.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        best_model = payload["best_model"]
        records[panel_name] = {
            "best_model": best_model["model_name"],
            "cv_auc": best_model["cv_summary"]["auc_roc_mean"],
            "cv_log_loss": best_model["cv_summary"]["log_loss_mean"],
            "holdout_auc": best_model["holdout"]["holdout_metrics"]["auc_roc"],
            "holdout_log_loss": best_model["holdout"]["holdout_metrics"]["log_loss"],
        }

    layer1 = records["event_v1_layer1"]
    layer2 = records["event_v1_layer1_layer2"]
    full = records["event_v1_full"]

    layer2_real_win = (
        layer2["cv_auc"] > layer1["cv_auc"]
        and layer2["cv_log_loss"] < layer1["cv_log_loss"]
        and layer2["holdout_auc"] >= layer1["holdout_auc"]
        and layer2["holdout_log_loss"] <= layer1["holdout_log_loss"]
    )
    full_real_win = (
        full["cv_auc"] > layer2["cv_auc"]
        and full["cv_log_loss"] < layer2["cv_log_loss"]
        and full["holdout_auc"] >= layer2["holdout_auc"]
        and full["holdout_log_loss"] <= layer2["holdout_log_loss"]
    )

    markdown_lines = [
        "# Event V1 Summary",
        "",
        "| Run | Best Model | CV AUC | CV Log Loss | Holdout AUC | Holdout Log Loss |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for panel_name, record in records.items():
        markdown_lines.append(
            f"| {panel_name} | {record['best_model']} | "
            f"{record['cv_auc']:.4f} | {record['cv_log_loss']:.4f} | "
            f"{record['holdout_auc']:.4f} | {record['holdout_log_loss']:.4f} |"
        )

    markdown_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Layer 2 v2 real win versus event_v1_layer1: {'yes' if layer2_real_win else 'no'}.",
            f"- Full panel real win versus event_v1_layer1_layer2: {'yes' if full_real_win else 'no'}.",
            "- A run is only treated as promising when CV AUC improves, CV log loss improves, "
            "and the 2024 holdout does not reverse the direction of the improvement.",
            "- Threshold metrics such as F1 and recall were not treated as sufficient on their own.",
        ]
    )
    write_markdown_report("\n".join(markdown_lines) + "\n", EVENT_V1_SUMMARY_PATH)


def main() -> None:
    """Train the requested event_v1 panel and save metrics/predictions."""
    args = parse_args()
    ensure_event_v1_directories()

    panel_path = get_panel_path(args.panel)
    metrics_output_path = get_metrics_output_path(args.panel)
    predictions_output_path = get_predictions_output_path(args.panel)
    candidate_features = get_candidate_feature_columns(args.panel)

    print(f"Loading event_v1 panel from: {panel_path}")
    panel_df = load_panel(panel_path, candidate_features)

    print("Preparing modeling data...")
    prepared_df = prepare_modeling_data(panel_df, candidate_features)

    print("Running shared event_v1 cross-validation...")
    split_payload, cv_payload = run_cross_validation(
        df=prepared_df,
        panel_name=args.panel,
        candidate_features=candidate_features,
        threshold=args.threshold,
    )

    model_results = cv_payload["results"]
    best_model_name = choose_best_model(model_results)
    print(f"Best model by mean CV AUC: {best_model_name}")

    print("Refitting the selected model on pre-holdout data and scoring 2024 holdout...")
    holdout_payload, holdout_predictions = fit_best_model_on_holdout(
        df=prepared_df,
        panel_name=args.panel,
        best_model_name=best_model_name,
        candidate_features=candidate_features,
        split_payload=split_payload,
        threshold=args.threshold,
    )

    print(f"Saving predictions to: {predictions_output_path}")
    save_best_model_predictions(
        validation_predictions=cv_payload["predictions"][best_model_name],
        holdout_predictions=holdout_predictions,
        output_path=predictions_output_path,
    )

    metrics_payload = build_metrics_payload(
        panel_name=args.panel,
        panel_path=panel_path,
        candidate_features=candidate_features,
        split_payload=split_payload,
        model_results=model_results,
        best_model_name=best_model_name,
        holdout_payload=holdout_payload,
        threshold=args.threshold,
    )
    print(f"Saving metrics to: {metrics_output_path}")
    write_json_report(metrics_payload, metrics_output_path)

    maybe_write_family_summary()
    print("\nSaved event_v1 training outputs.")


if __name__ == "__main__":
    main()
