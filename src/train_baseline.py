"""Train the first Layer 1 baseline model on the daily modeling panel.

This script trains a simple logistic regression classifier using only the
engineered Layer 1 financial-statement features. It uses a time-aware split:
the last chunk of dates is held out for testing so there is no random leakage.

Input:
    data/processed/modeling/layer1_modeling_panel.parquet

Outputs:
    outputs/baseline/layer1_logistic_metrics.json
    outputs/baseline/layer1_logistic_predictions.parquet
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


def get_input_path() -> Path:
    """Return the final modeling panel path."""
    return PROCESSED_DATA_DIR / "modeling" / "layer1_modeling_panel.parquet"


def get_metrics_output_path() -> Path:
    """Return the metrics JSON output path and create its folder."""
    output_dir = OUTPUTS_DIR / "baseline"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "layer1_baseline_metrics.json"


def get_predictions_output_path() -> Path:
    """Return the prediction parquet output path and create its folder."""
    output_dir = OUTPUTS_DIR / "baseline"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "layer1_baseline_predictions.parquet"


def load_modeling_panel(path: Path) -> pd.DataFrame:
    """Load the modeling panel and validate required columns."""
    if not path.exists():
        raise FileNotFoundError(f"Modeling panel file was not found: {path}")

    df = pd.read_parquet(path)

    required_columns = ["ticker", "date", TARGET_COLUMN] + LAYER1_FEATURE_COLUMNS
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "Modeling panel is missing required columns: " + ", ".join(missing_columns)
        )

    return df.copy()


def prepare_modeling_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dtypes and keep rows with a valid binary target."""
    prepared = df.copy()
    prepared["ticker"] = prepared["ticker"].astype("string")
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    prepared[TARGET_COLUMN] = pd.to_numeric(prepared[TARGET_COLUMN], errors="coerce").astype("Int64")

    for column in LAYER1_FEATURE_COLUMNS:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    prepared = prepared.dropna(subset=["ticker", "date", TARGET_COLUMN]).copy()
    prepared = prepared.sort_values(["date", "ticker"]).reset_index(drop=True)
    return prepared


def split_by_time(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split the panel into train and test sets using unique trading dates."""
    unique_dates = sorted(df["date"].dropna().unique())
    if len(unique_dates) < 10:
        raise ValueError("Not enough unique dates to create a time-based train/test split.")

    split_index = int(len(unique_dates) * (1 - TEST_SIZE))
    split_index = max(1, min(split_index, len(unique_dates) - 1))
    split_date = pd.Timestamp(unique_dates[split_index])

    train_df = df[df["date"] < split_date].copy()
    test_df = df[df["date"] >= split_date].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Time split produced an empty train or test set.")

    return train_df, test_df, split_date


def select_usable_features(train_df: pd.DataFrame) -> tuple[list[str], dict[str, float]]:
    """Keep features with observed values and acceptable missingness."""
    usable_features = []
    missingness_by_feature = {}
    for column in LAYER1_FEATURE_COLUMNS:
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
    """Clip feature extremes using train-set quantiles only.

    This is a simple winsorization-style step that reduces the influence of
    very large ratio outliers without leaking test information.
    """
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
    """Create a small set of baseline models for comparison."""
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


def fit_and_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_columns: list[str]):
    """Train baseline models and return per-model test predictions and metrics."""
    x_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN].astype(int)

    x_test = test_df[feature_columns]
    y_test = test_df[TARGET_COLUMN].astype(int)

    results = {}
    for model_name, model in build_models().items():
        model.fit(x_train, y_train)

        y_prob = model.predict_proba(x_test)[:, 1]
        y_pred = model.predict(x_test)

        metrics = compute_classification_metrics(y_test, y_pred, y_prob)
        results[model_name] = {
            "model": model,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "metrics": metrics,
        }

    return y_test, results


def save_metrics(metrics: dict, output_path: Path) -> None:
    """Write metrics to a JSON file."""
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def save_predictions(
    test_df: pd.DataFrame,
    y_pred,
    y_prob,
    model_name: str,
    output_path: Path,
) -> None:
    """Save best-model test predictions for later inspection."""
    predictions_df = test_df[["ticker", "date", "forward_return_5d", TARGET_COLUMN]].copy()
    predictions_df["model_name"] = model_name
    predictions_df["predicted_label"] = y_pred
    predictions_df["predicted_probability"] = y_prob
    predictions_df.to_parquet(output_path, index=False)


def get_class_balance(y: pd.Series) -> dict[str, float]:
    """Return class proportions for a binary target series."""
    proportions = y.astype(int).value_counts(normalize=True).sort_index()
    return {
        "class_0_pct": float(proportions.get(0, 0.0) * 100),
        "class_1_pct": float(proportions.get(1, 0.0) * 100),
    }


def print_training_summary(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_date: pd.Timestamp,
    feature_columns: list[str],
    missingness_by_feature: dict[str, float],
    all_results: dict,
) -> None:
    """Print a concise baseline comparison summary."""
    train_balance = get_class_balance(train_df[TARGET_COLUMN])
    test_balance = get_class_balance(test_df[TARGET_COLUMN])

    print("\nBaseline Training Summary")
    print("-" * 60)
    print("Models: Logistic Regression, Random Forest, HistGradientBoosting")
    print(f"Usable feature count: {len(feature_columns)}")
    print(f"Dropped unusable features: {len(LAYER1_FEATURE_COLUMNS) - len(feature_columns)}")
    dropped_sparse = [
        name for name, pct in missingness_by_feature.items()
        if name not in feature_columns and pct <= 100.0
    ]
    print(f"Missingness filter: drop train features above {MAX_MISSINGNESS_PCT:.0f}% missing")
    print(f"Outlier clipping: train-set quantiles {CLIP_LOWER_QUANTILE:.0%} to {CLIP_UPPER_QUANTILE:.0%}")
    print(f"Train rows: {len(train_df):,}")
    print(f"Test rows:  {len(test_df):,}")
    print(f"Split date: {split_date.date()}")
    print(
        f"Train class balance: 0={train_balance['class_0_pct']:.2f}%, "
        f"1={train_balance['class_1_pct']:.2f}%"
    )
    print(
        f"Test class balance:  0={test_balance['class_0_pct']:.2f}%, "
        f"1={test_balance['class_1_pct']:.2f}%"
    )
    print(f"Dropped sparse/all-NaN features: {dropped_sparse}")

    print("\nModel Comparison")
    print("-" * 60)
    for model_name, result in all_results.items():
        metrics = result["metrics"]
        print(
            f"{model_name:<24} "
            f"AUC={metrics['auc_roc']:.4f} "
            f"F1={metrics['f1']:.4f} "
            f"Precision={metrics['precision']:.4f} "
            f"Recall={metrics['recall']:.4f} "
            f"LogLoss={metrics['log_loss']:.4f}"
        )


def choose_best_model(results: dict) -> tuple[str, dict]:
    """Pick the best model using AUC-ROC as the primary metric."""
    best_name = max(results.keys(), key=lambda name: results[name]["metrics"]["auc_roc"])
    return best_name, results[best_name]


def main() -> None:
    """Train and evaluate the first Layer 1 baseline model."""
    input_path = get_input_path()
    metrics_output_path = get_metrics_output_path()
    predictions_output_path = get_predictions_output_path()

    print(f"Loading modeling panel from: {input_path}")
    panel_df = load_modeling_panel(input_path)

    print("Preparing modeling data...")
    prepared_df = prepare_modeling_data(panel_df)

    print("Creating time-based train/test split...")
    train_df, test_df, split_date = split_by_time(prepared_df)
    feature_columns, missingness_by_feature = select_usable_features(train_df)

    print("Clipping extreme ratio values using train-set quantiles...")
    train_df, test_df = clip_outliers(train_df, test_df, feature_columns)

    print("Training baseline model set...")
    _, results = fit_and_predict(train_df, test_df, feature_columns)
    best_model_name, best_result = choose_best_model(results)

    print(f"Saving metrics to: {metrics_output_path}")
    metrics_payload = {
        "split_date": str(split_date.date()),
        "usable_feature_columns": feature_columns,
        "train_missingness_by_feature_pct": missingness_by_feature,
        "max_missingness_pct": MAX_MISSINGNESS_PCT,
        "clip_lower_quantile": CLIP_LOWER_QUANTILE,
        "clip_upper_quantile": CLIP_UPPER_QUANTILE,
        "results": {name: result["metrics"] for name, result in results.items()},
        "best_model": best_model_name,
    }
    save_metrics(metrics_payload, metrics_output_path)

    print(f"Saving predictions to: {predictions_output_path}")
    save_predictions(
        test_df,
        best_result["y_pred"],
        best_result["y_prob"],
        best_model_name,
        predictions_output_path,
    )

    print_training_summary(
        train_df,
        test_df,
        split_date,
        feature_columns,
        missingness_by_feature,
        results,
    )
    print(f"\nBest model by AUC-ROC: {best_model_name}")
    print("\nSaved Layer 1 baseline outputs.")


if __name__ == "__main__":
    main()
