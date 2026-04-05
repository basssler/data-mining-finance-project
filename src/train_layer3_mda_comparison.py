"""Compare Layer 1, full-filing Layer 3, and MD&A Layer 3 on the same split.

This script evaluates three setups on the same time-based holdout:
- Layer 1 only
- Layer 1 + full-filing SEC sentiment
- Layer 1 + MD&A-only SEC sentiment

Inputs:
    data/processed/modeling/layer1_modeling_panel.parquet
    data/processed/modeling/layer1_layer3_modeling_panel.parquet
    data/processed/modeling/layer1_layer3_mda_modeling_panel.parquet

Outputs:
    outputs/comparison/layer3_mda_comparison_metrics.json
    outputs/comparison/layer3_mda_comparison_predictions.parquet
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.evaluate import compute_classification_metrics
from src.paths import OUTPUTS_DIR, PROCESSED_DATA_DIR

TARGET_COLUMN = "label"
TEST_SIZE = 0.20
MAX_MISSINGNESS_PCT = 20.0
CLIP_LOWER_QUANTILE = 0.01
CLIP_UPPER_QUANTILE = 0.99

LAYER1_FEATURE_COLUMNS = [
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
]

LAYER3_FULL_FEATURE_COLUMNS = [
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
    "sec_is_10k",
    "sec_is_10q",
]

LAYER3_MDA_FEATURE_COLUMNS = [
    "mda_sentiment_score",
    "mda_positive_prob",
    "mda_negative_prob",
    "mda_neutral_prob",
    "mda_sentiment_abs",
    "mda_sentiment_change_prev",
    "mda_positive_change_prev",
    "mda_negative_change_prev",
    "mda_chunk_count",
    "mda_log_chunk_count",
    "mda_text_length",
    "mda_log_text_length",
    "mda_is_10k",
    "mda_is_10q",
]


def get_layer1_panel_path() -> Path:
    """Return the locked Layer 1 daily modeling panel path."""
    return PROCESSED_DATA_DIR / "modeling" / "layer1_modeling_panel.parquet"


def get_layer1_layer3_panel_path() -> Path:
    """Return the daily panel with full-filing Layer 3 sentiment."""
    return PROCESSED_DATA_DIR / "modeling" / "layer1_layer3_modeling_panel.parquet"


def get_layer1_layer3_mda_panel_path() -> Path:
    """Return the daily panel with MD&A-only Layer 3 sentiment."""
    return PROCESSED_DATA_DIR / "modeling" / "layer1_layer3_mda_modeling_panel.parquet"


def get_metrics_output_path() -> Path:
    """Return the Layer 3 MD&A comparison metrics JSON path."""
    output_dir = OUTPUTS_DIR / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "layer3_mda_comparison_metrics.json"


def get_predictions_output_path() -> Path:
    """Return the Layer 3 MD&A comparison prediction parquet path."""
    output_dir = OUTPUTS_DIR / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "layer3_mda_comparison_predictions.parquet"


def load_parquet(path: Path, required_columns: list[str], dataset_name: str) -> pd.DataFrame:
    """Load a parquet file and validate required columns."""
    if not path.exists():
        raise FileNotFoundError(f"{dataset_name} file was not found: {path}")

    df = pd.read_parquet(path)
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"{dataset_name} file is missing required columns: " + ", ".join(missing_columns)
        )

    return df.copy()


def prepare_modeling_data(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Normalize dtypes and keep rows with a valid target."""
    prepared = df.copy()
    prepared["ticker"] = prepared["ticker"].astype("string")
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    prepared[TARGET_COLUMN] = pd.to_numeric(prepared[TARGET_COLUMN], errors="coerce").astype("Int64")

    for column in feature_columns:
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    prepared = prepared.dropna(subset=["ticker", "date", TARGET_COLUMN]).copy()
    prepared = prepared.sort_values(["date", "ticker"]).reset_index(drop=True)
    return prepared


def split_by_time(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Create the same time-based train/test split used in prior layers."""
    unique_dates = sorted(df["date"].dropna().unique())
    split_index = int(len(unique_dates) * (1 - TEST_SIZE))
    split_index = max(1, min(split_index, len(unique_dates) - 1))
    split_date = pd.Timestamp(unique_dates[split_index])

    train_df = df[df["date"] < split_date].copy()
    test_df = df[df["date"] >= split_date].copy()
    return train_df, test_df, split_date


def select_usable_features(
    train_df: pd.DataFrame,
    candidate_columns: list[str],
) -> tuple[list[str], dict[str, float]]:
    """Keep columns with observed values and acceptable missingness."""
    usable_features = []
    missingness_by_feature = {}
    for column in candidate_columns:
        missing_pct = float(train_df[column].isna().mean() * 100)
        missingness_by_feature[column] = missing_pct
        if train_df[column].notna().any() and missing_pct <= MAX_MISSINGNESS_PCT:
            usable_features.append(column)

    if not usable_features:
        raise ValueError("No usable feature columns were found in the training data.")

    return usable_features, missingness_by_feature


def clip_outliers(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clip feature outliers using train-set quantiles only."""
    clipped_train = train_df.copy()
    clipped_test = test_df.copy()

    for column in feature_columns:
        lower_bound = clipped_train[column].quantile(CLIP_LOWER_QUANTILE)
        upper_bound = clipped_train[column].quantile(CLIP_UPPER_QUANTILE)
        if pd.isna(lower_bound) or pd.isna(upper_bound):
            continue

        clipped_train[column] = clipped_train[column].clip(lower=lower_bound, upper=upper_bound)
        clipped_test[column] = clipped_test[column].clip(lower=lower_bound, upper=upper_bound)

    return clipped_train, clipped_test


def build_models() -> dict[str, Pipeline]:
    """Create the same baseline model set for fair comparison."""
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
                        n_jobs=-1,
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


def run_model_set(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_columns: list[str]) -> dict:
    """Train the model set and collect test metrics."""
    x_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN].astype(int)
    x_test = test_df[feature_columns]
    y_test = test_df[TARGET_COLUMN].astype(int)

    results = {}
    for model_name, model in build_models().items():
        model.fit(x_train, y_train)
        y_prob = model.predict_proba(x_test)[:, 1]
        y_pred = model.predict(x_test)
        results[model_name] = {
            "metrics": compute_classification_metrics(y_test, y_pred, y_prob),
            "y_pred": y_pred,
            "y_prob": y_prob,
        }
    return results


def choose_best_result(results: dict) -> tuple[str, dict]:
    """Choose the best result using AUC-ROC."""
    best_name = max(results.keys(), key=lambda name: results[name]["metrics"]["auc_roc"])
    return best_name, results[best_name]


def save_metrics(payload: dict, output_path: Path) -> None:
    """Save comparison metrics as JSON."""
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_predictions(test_df: pd.DataFrame, comparison_results: dict, output_path: Path) -> None:
    """Save best predictions for each feature setup."""
    frames = []
    for setup_name, setup_payload in comparison_results.items():
        best_model_name = setup_payload["best_model"]
        best_result = setup_payload["results"][best_model_name]

        frame = test_df[["ticker", "date", "forward_return_5d", TARGET_COLUMN]].copy()
        frame["feature_setup"] = setup_name
        frame["model_name"] = best_model_name
        frame["predicted_label"] = best_result["y_pred"]
        frame["predicted_probability"] = best_result["y_prob"]
        frames.append(frame)

    pd.concat(frames, ignore_index=True).to_parquet(output_path, index=False)


def print_summary(
    split_date: pd.Timestamp,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    comparison_results: dict,
) -> None:
    """Print a concise Layer 1 versus full-filing versus MD&A summary."""
    print("\nLayer 3 MD&A Comparison Summary")
    print("-" * 60)
    print(f"Train rows: {len(train_df):,}")
    print(f"Test rows:  {len(test_df):,}")
    print(f"Split date: {split_date.date()}")

    for setup_name, setup_payload in comparison_results.items():
        print(f"\n{setup_name}")
        print("-" * 60)
        print(f"Usable features: {len(setup_payload['usable_feature_columns'])}")
        print(f"Best model: {setup_payload['best_model']}")
        for model_name, result in setup_payload["results"].items():
            metrics = result["metrics"]
            print(
                f"{model_name:<24} "
                f"AUC={metrics['auc_roc']:.4f} "
                f"F1={metrics['f1']:.4f} "
                f"Precision={metrics['precision']:.4f} "
                f"Recall={metrics['recall']:.4f} "
                f"LogLoss={metrics['log_loss']:.4f}"
            )


def main() -> None:
    """Compare Layer 1, full-filing Layer 3, and MD&A Layer 3 on the same holdout."""
    layer1_path = get_layer1_panel_path()
    layer1_layer3_path = get_layer1_layer3_panel_path()
    layer1_layer3_mda_path = get_layer1_layer3_mda_panel_path()
    metrics_output_path = get_metrics_output_path()
    predictions_output_path = get_predictions_output_path()

    print(f"Loading Layer 1 panel from: {layer1_path}")
    layer1_df = load_parquet(
        layer1_path,
        ["ticker", "date", "forward_return_5d", TARGET_COLUMN] + LAYER1_FEATURE_COLUMNS,
        "Layer 1 panel",
    )

    print(f"Loading Layer 1 + full-filing Layer 3 panel from: {layer1_layer3_path}")
    layer1_layer3_df = load_parquet(
        layer1_layer3_path,
        ["ticker", "date", "forward_return_5d", TARGET_COLUMN]
        + LAYER1_FEATURE_COLUMNS
        + LAYER3_FULL_FEATURE_COLUMNS,
        "Layer 1 + full-filing Layer 3 panel",
    )

    print(f"Loading Layer 1 + MD&A Layer 3 panel from: {layer1_layer3_mda_path}")
    layer1_layer3_mda_df = load_parquet(
        layer1_layer3_mda_path,
        ["ticker", "date", "forward_return_5d", TARGET_COLUMN]
        + LAYER1_FEATURE_COLUMNS
        + LAYER3_MDA_FEATURE_COLUMNS,
        "Layer 1 + MD&A Layer 3 panel",
    )

    feature_set_candidates = {
        "layer1_only": LAYER1_FEATURE_COLUMNS,
        "layer1_plus_layer3_full": LAYER1_FEATURE_COLUMNS + LAYER3_FULL_FEATURE_COLUMNS,
        "layer1_plus_layer3_mda": LAYER1_FEATURE_COLUMNS + LAYER3_MDA_FEATURE_COLUMNS,
    }

    prepared_layer1 = prepare_modeling_data(layer1_df, LAYER1_FEATURE_COLUMNS)
    prepared_layer1_layer3 = prepare_modeling_data(
        layer1_layer3_df,
        LAYER1_FEATURE_COLUMNS + LAYER3_FULL_FEATURE_COLUMNS,
    )
    prepared_layer1_layer3_mda = prepare_modeling_data(
        layer1_layer3_mda_df,
        LAYER1_FEATURE_COLUMNS + LAYER3_MDA_FEATURE_COLUMNS,
    )

    print("Creating shared time-based train/test split...")
    train_df, test_df, split_date = split_by_time(prepared_layer1)

    train_df_layer3 = prepared_layer1_layer3[prepared_layer1_layer3["date"] < split_date].copy()
    test_df_layer3 = prepared_layer1_layer3[prepared_layer1_layer3["date"] >= split_date].copy()

    train_df_layer3_mda = prepared_layer1_layer3_mda[
        prepared_layer1_layer3_mda["date"] < split_date
    ].copy()
    test_df_layer3_mda = prepared_layer1_layer3_mda[
        prepared_layer1_layer3_mda["date"] >= split_date
    ].copy()

    setup_frames = {
        "layer1_only": (train_df, test_df),
        "layer1_plus_layer3_full": (train_df_layer3, test_df_layer3),
        "layer1_plus_layer3_mda": (train_df_layer3_mda, test_df_layer3_mda),
    }

    comparison_results = {}
    for setup_name, candidate_columns in feature_set_candidates.items():
        print(f"Training model set for {setup_name}...")
        current_train_df, current_test_df = setup_frames[setup_name]
        usable_feature_columns, missingness_by_feature = select_usable_features(
            current_train_df,
            candidate_columns,
        )
        clipped_train_df, clipped_test_df = clip_outliers(
            current_train_df,
            current_test_df,
            usable_feature_columns,
        )
        results = run_model_set(clipped_train_df, clipped_test_df, usable_feature_columns)
        best_model_name, _ = choose_best_result(results)

        comparison_results[setup_name] = {
            "usable_feature_columns": usable_feature_columns,
            "train_missingness_by_feature_pct": missingness_by_feature,
            "best_model": best_model_name,
            "results": results,
        }

    print(f"Saving comparison metrics to: {metrics_output_path}")
    metrics_payload = {
        "split_date": str(split_date.date()),
        "max_missingness_pct": MAX_MISSINGNESS_PCT,
        "clip_lower_quantile": CLIP_LOWER_QUANTILE,
        "clip_upper_quantile": CLIP_UPPER_QUANTILE,
        "comparison_results": {
            setup_name: {
                "usable_feature_columns": payload["usable_feature_columns"],
                "train_missingness_by_feature_pct": payload["train_missingness_by_feature_pct"],
                "best_model": payload["best_model"],
                "results": {
                    model_name: result["metrics"]
                    for model_name, result in payload["results"].items()
                },
            }
            for setup_name, payload in comparison_results.items()
        },
    }
    save_metrics(metrics_payload, metrics_output_path)

    print(f"Saving comparison predictions to: {predictions_output_path}")
    save_predictions(test_df, comparison_results, predictions_output_path)

    print_summary(split_date, train_df, test_df, comparison_results)
    print("\nSaved Layer 3 MD&A comparison outputs.")


if __name__ == "__main__":
    main()
