"""Train the locked Phase 4 event_panel_v2 benchmark matrix."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
import random
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml
from sklearn.inspection import permutation_importance

matplotlib.use("Agg")

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.label_comparison_event_v2 import (
    VariantSpec,
    apply_variant_label_mode,
    attach_labels_to_event_panel,
    build_daily_label_table,
    candidate_feature_columns,
    choose_best_model,
    clip_outliers,
    compute_global_feature_exclusions,
    evaluate_extended,
    fit_model,
    format_metric,
    load_event_panel,
    resolve_max_missingness_pct,
    safe_rank_ic,
    select_usable_features,
    summarize_metric_dicts,
)
from src.labels_event_v1 import load_price_data, normalize_price_data
from src.validation_event_v1 import make_event_v1_splits
from src.config_event_v1 import PRICE_INPUT_PATH

DEFAULT_CONFIG_PATH = Path("configs") / "event_panel_v2_primary.yaml"
OLD_BASELINE_METRICS_PATH = Path("reports") / "results" / "event_v1_layer1_metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the locked event_panel_v2 benchmark.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    return parser.parse_args()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file was not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve_validation_output_dir(config: dict) -> Path | None:
    validation_dir = config.get("outputs", {}).get("validation_dir")
    if not validation_dir:
        return None
    return Path(str(validation_dir))


def resolve_promotion_strategy(config: dict) -> str:
    strategy = str(config.get("promotion", {}).get("strategy", "cv_mean_auc")).strip().lower()
    if strategy not in {"cv_mean_auc", "stability_aware"}:
        raise ValueError(f"Unsupported promotion.strategy: {strategy}")
    return strategy


def resolve_candidate_features(panel_df: pd.DataFrame, config: dict) -> list[str]:
    base_candidates = candidate_feature_columns(panel_df)
    additional = list(config.get("feature_inclusions", {}).get("additional", []))
    prefixed = [
        column
        for column in panel_df.columns
        if column.startswith(("av_", "qfd_"))
        and (
            pd.api.types.is_numeric_dtype(panel_df[column])
            or pd.api.types.is_bool_dtype(panel_df[column])
        )
    ]
    combined = list(
        dict.fromkeys(base_candidates + prefixed + [column for column in additional if column in panel_df.columns])
    )
    return combined


def load_prebuilt_label_map(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Label map file was not found: {path}")
    label_df = pd.read_parquet(path).copy()
    required_columns = ["ticker", "date", "target"]
    missing_columns = [column for column in required_columns if column not in label_df.columns]
    if missing_columns:
        raise ValueError("Prebuilt label map is missing required columns: " + ", ".join(missing_columns))
    label_df["ticker"] = label_df["ticker"].astype("string")
    label_df["date"] = pd.to_datetime(label_df["date"], errors="coerce")
    label_df["target"] = pd.to_numeric(label_df["target"], errors="coerce").astype("Int64")
    if "excess_forward_return" in label_df.columns:
        label_df["excess_forward_return"] = pd.to_numeric(label_df["excess_forward_return"], errors="coerce")
    return label_df.sort_values(["date", "ticker"]).reset_index(drop=True)


def set_random_seeds(seed_block: dict) -> None:
    random.seed(int(seed_block.get("python", 42)))
    np.random.seed(int(seed_block.get("numpy", 42)))


def load_old_baseline() -> dict | None:
    if not OLD_BASELINE_METRICS_PATH.exists():
        return None
    payload = json.loads(OLD_BASELINE_METRICS_PATH.read_text(encoding="utf-8"))
    best = payload["best_model"]
    return {
        "panel_name": payload["panel_name"],
        "model_name": best["model_name"],
        "cv_auc": best["cv_summary"]["auc_roc_mean"],
        "cv_log_loss": best["cv_summary"]["log_loss_mean"],
        "holdout_auc": best["holdout"]["holdout_metrics"]["auc_roc"],
        "holdout_log_loss": best["holdout"]["holdout_metrics"]["log_loss"],
    }


def resolve_shap_output_paths(config: dict) -> tuple[Path, Path]:
    outputs = config["outputs"]
    csv_path = Path(outputs["csv"])
    shap_plot = Path(outputs.get("shap_plot", csv_path.with_name(f"{csv_path.stem}_shap_summary.png")))
    shap_csv = Path(outputs.get("shap_csv", csv_path.with_name(f"{csv_path.stem}_shap_importance.csv")))
    return shap_plot, shap_csv


def resolve_concentration_output_path(config: dict) -> Path:
    outputs = config["outputs"]
    csv_path = Path(outputs["csv"])
    return Path(outputs.get("concentration_csv", csv_path.with_name(f"{csv_path.stem}_concentration.csv")))


def build_default_label_description(label_config: dict) -> str:
    horizon_days = int(label_config["horizon_days"])
    label_mode = str(label_config["mode"])
    if label_mode == "sign":
        return f"{horizon_days}-trading-day excess return sign"
    if label_mode == "thresholded":
        threshold = float(label_config.get("threshold", 0.015))
        return f"{horizon_days}-trading-day thresholded excess return (+/-{threshold * 100.0:.1f}%)"
    if label_mode == "quantile":
        return f"{horizon_days}-trading-day excess return quantile"
    return f"{horizon_days}-trading-day {label_mode} label"


def resolve_report_metadata(config: dict) -> dict[str, str]:
    metadata = config.get("metadata", {})
    panel_name = str(config.get("panel", {}).get("name", "event_panel_v2"))
    panel_display_name = str(metadata.get("panel_display_name", panel_name))
    label_description = str(
        metadata.get("label_description", build_default_label_description(config["label"]))
    )
    report_title = str(metadata.get("report_title", f"{panel_display_name} Benchmark"))
    setup_note = str(
        metadata.get(
            "setup_note",
            "This report is the new post-fix anchor to use before universe expansion.",
        )
    )
    interpretation_note = str(
        metadata.get(
            "interpretation_note",
            "The redesigned setup is directionally better than the old daily research path, "
            "but the edge is still modest. This should be treated as a cleaner anchor, not as proof that the problem is solved.",
        )
    )
    return {
        "panel_name": panel_name,
        "panel_display_name": panel_display_name,
        "label_description": label_description,
        "report_title": report_title,
        "setup_note": setup_note,
        "interpretation_note": interpretation_note,
    }


def choose_best_model_with_stability(model_records: list[dict]) -> str:
    ranked = []
    for record in model_records:
        ranked.append(
            (
                -(record.get("worst_fold_auc") if record.get("worst_fold_auc") is not None else float("-inf")),
                record.get("cv_auc_std") if record.get("cv_auc_std") is not None else float("inf"),
                -(record.get("cv_auc_mean") if record.get("cv_auc_mean") is not None else float("-inf")),
                record.get("cv_log_loss_mean") if record.get("cv_log_loss_mean") is not None else float("inf"),
                -(record.get("holdout_auc") if record.get("holdout_auc") is not None else float("-inf")),
                record["model_name"],
            )
        )
    ranked.sort()
    return ranked[0][5]


def resolve_threshold(config: dict) -> float:
    return float(config.get("scoring", {}).get("threshold", 0.5))


def resolve_tuning_output_dir(config: dict) -> Path:
    outputs = config.get("outputs", {})
    csv_path = Path(outputs["csv"])
    return Path(outputs.get("tuning_dir", csv_path.with_name(f"{csv_path.stem}_tuning")))


def resolve_model_params(
    config: dict,
    model_name: str,
    override_params: dict | None = None,
    seed_override: int | None = None,
) -> dict:
    params = dict(config.get(model_name, {}))
    for key in ["prefer_gpu_if_clean", "fallback_to_cpu", "note", "enabled"]:
        params.pop(key, None)
    if override_params:
        params.update(override_params)
    if seed_override is not None:
        if model_name == "catboost":
            params["random_seed"] = int(seed_override)
        else:
            params["random_state"] = int(seed_override)
    return params


def resolve_tuning_spec(config: dict) -> dict:
    tuning = config.get("tuning", {}) or {}
    objective = tuning.get("objective", {}) or {}
    reproducibility = tuning.get("reproducibility", {}) or {}
    return {
        "enabled": bool(tuning.get("enabled", False)),
        "n_trials": int(tuning.get("n_trials", 0)),
        "objective_metric": str(objective.get("metric", "mean_cv_auc")),
        "stability_penalty": float(objective.get("stability_penalty", 0.0)),
        "concentration_penalty": float(objective.get("concentration_penalty", 0.0)),
        "models": tuning.get("models", {}) or {},
        "reproducibility_seeds": [int(seed) for seed in reproducibility.get("seeds", [42, 52, 62])],
        "max_holdout_auc_std": float(reproducibility.get("max_holdout_auc_std", 0.02)),
        "max_cv_auc_std": float(reproducibility.get("max_cv_auc_std", 0.02)),
    }


def resolve_reference_benchmark_row(config: dict) -> dict | None:
    reference_path = config.get("promotion", {}).get("reference_results_csv")
    if not reference_path:
        return None
    benchmark_path = Path(str(reference_path))
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Promotion reference benchmark was not found: {benchmark_path}")
    benchmark_df = pd.read_csv(benchmark_path)
    selected = benchmark_df.loc[benchmark_df["is_selected_primary_model"] == True]  # noqa: E712
    if selected.empty:
        raise ValueError(f"Promotion reference benchmark has no selected model row: {benchmark_path}")
    return selected.iloc[0].to_dict()


def _extract_model_importance_series(
    fitted_model,
    feature_names: list[str],
    x_frame: pd.DataFrame,
    y_target: pd.Series,
) -> pd.Series | None:
    model = fitted_model.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype="float64")
        return pd.Series(values, index=feature_names, dtype="float64")
    if hasattr(model, "coef_"):
        values = np.asarray(model.coef_, dtype="float64")
        if values.ndim > 1:
            values = values[-1]
        return pd.Series(np.abs(values), index=feature_names, dtype="float64")
    try:
        importance = permutation_importance(
            fitted_model,
            x_frame[feature_names],
            y_target.astype(int),
            scoring="roc_auc",
            n_repeats=5,
            random_state=42,
            n_jobs=1,
        )
    except ValueError:
        return None
    return pd.Series(importance.importances_mean, index=feature_names, dtype="float64")


def _collect_top_feature_rows(
    *,
    fitted_model,
    feature_names: list[str],
    x_frame: pd.DataFrame,
    y_target: pd.Series,
    model_name: str,
    fold_label: str,
    evaluation_role: str,
) -> tuple[list[dict], list[str]]:
    importance_series = _extract_model_importance_series(fitted_model, feature_names, x_frame, y_target)
    if importance_series is None or importance_series.empty:
        return [], []
    ranked = (
        importance_series.sort_values(ascending=False)
        .head(3)
        .reset_index()
        .rename(columns={"index": "feature", 0: "importance_mean"})
    )
    rows: list[dict] = []
    top_features: list[str] = []
    for rank, record in enumerate(ranked.to_dict(orient="records"), start=1):
        feature_name = str(record["feature"])
        top_features.append(feature_name)
        rows.append(
            {
                "model_name": model_name,
                "fold_label": fold_label,
                "evaluation_role": evaluation_role,
                "rank": rank,
                "feature": feature_name,
                "importance_mean": float(record["importance_mean"]),
            }
        )
    return rows, top_features


def _default_search_space(model_name: str) -> dict[str, dict]:
    if model_name == "xgboost":
        return {
            "max_depth": {"type": "int", "low": 3, "high": 8},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.20, "log": True},
            "n_estimators": {"type": "int", "low": 150, "high": 500},
            "subsample": {"type": "float", "low": 0.6, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
            "min_child_weight": {"type": "float", "low": 1.0, "high": 10.0},
            "reg_alpha": {"type": "float", "low": 1e-4, "high": 10.0, "log": True},
            "reg_lambda": {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
            "gamma": {"type": "float", "low": 1e-4, "high": 5.0, "log": True},
        }
    if model_name == "lightgbm":
        return {
            "num_leaves": {"type": "int", "low": 16, "high": 96},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.20, "log": True},
            "n_estimators": {"type": "int", "low": 150, "high": 500},
            "feature_fraction": {"type": "float", "low": 0.6, "high": 1.0},
            "bagging_fraction": {"type": "float", "low": 0.6, "high": 1.0},
            "lambda_l1": {"type": "float", "low": 1e-4, "high": 10.0, "log": True},
            "lambda_l2": {"type": "float", "low": 1e-4, "high": 10.0, "log": True},
            "min_child_samples": {"type": "int", "low": 10, "high": 60},
        }
    if model_name == "catboost":
        return {
            "depth": {"type": "int", "low": 4, "high": 10},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.20, "log": True},
            "iterations": {"type": "int", "low": 150, "high": 500},
            "l2_leaf_reg": {"type": "float", "low": 1.0, "high": 10.0},
            "random_strength": {"type": "float", "low": 1e-4, "high": 5.0, "log": True},
            "bagging_temperature": {"type": "float", "low": 0.0, "high": 5.0},
        }
    raise ValueError(f"Unsupported tuned model: {model_name}")


def _suggest_trial_params(trial, model_name: str, search_space: dict | None = None) -> dict:
    resolved_space = search_space or _default_search_space(model_name)
    params: dict[str, object] = {}
    for param_name, spec in resolved_space.items():
        spec_type = str(spec.get("type", "float")).strip().lower()
        if spec_type == "int":
            params[param_name] = trial.suggest_int(
                param_name,
                int(spec["low"]),
                int(spec["high"]),
                step=int(spec.get("step", 1)),
                log=bool(spec.get("log", False)),
            )
            continue
        if spec_type == "float":
            params[param_name] = trial.suggest_float(
                param_name,
                float(spec["low"]),
                float(spec["high"]),
                step=spec.get("step"),
                log=bool(spec.get("log", False)),
            )
            continue
        if spec_type == "categorical":
            params[param_name] = trial.suggest_categorical(param_name, list(spec["choices"]))
            continue
        raise ValueError(f"Unsupported search space type for {model_name}.{param_name}: {spec_type}")
    return params


def _safe_class_balance(target: pd.Series) -> float | None:
    if target.empty:
        return None
    return float(target.astype(int).mean())


def build_validation_artifacts(
    panel_df: pd.DataFrame,
    split_payload: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_map_rows: list[dict] = []
    purge_audit_rows: list[dict] = []
    working_df = panel_df.reset_index(drop=True)
    base_columns = ["ticker", "date"]
    optional_columns = [column for column in ["event_id", "effective_model_date"] if column in working_df.columns]
    identity_columns = base_columns + optional_columns

    def append_rows(row_indices: list[int], fold_label: str, role: str) -> None:
        if not row_indices:
            return
        selected = working_df.iloc[row_indices].copy()
        selected["row_index"] = selected.index.astype("int64")
        selected["fold_label"] = fold_label
        selected["fold_role"] = role
        fold_map_rows.extend(selected[["row_index", *identity_columns, "fold_label", "fold_role"]].to_dict(orient="records"))

    for fold in split_payload["folds"]:
        fold_number = int(fold["fold_number"])
        fold_label = f"fold_{fold_number}"
        metadata = fold["date_metadata"]
        purge_metadata = fold.get("purge_window_metadata", {})
        validation_mask = (
            (working_df["date"] >= pd.Timestamp(metadata["validation_start_date"]))
            & (working_df["date"] <= pd.Timestamp(metadata["validation_end_date"]))
        )
        train_mask = (
            working_df["date"] >= pd.Timestamp(metadata["train_start_date"])
            if metadata["train_start_date"] is not None
            else pd.Series(False, index=working_df.index)
        ) & (
            working_df["date"] <= pd.Timestamp(metadata["train_end_date"])
            if metadata["train_end_date"] is not None
            else pd.Series(False, index=working_df.index)
        )
        overlap_mask = (
            (working_df["date"] >= pd.Timestamp(purge_metadata["overlap_purge_start_date"]))
            & (working_df["date"] <= pd.Timestamp(purge_metadata["overlap_purge_end_date"]))
            if purge_metadata.get("overlap_purge_start_date") and purge_metadata.get("overlap_purge_end_date")
            else pd.Series(False, index=working_df.index)
        )
        embargo_mask = (
            (working_df["date"] >= pd.Timestamp(purge_metadata["embargo_start_date"]))
            & (working_df["date"] <= pd.Timestamp(purge_metadata["embargo_end_date"]))
            if purge_metadata.get("embargo_start_date") and purge_metadata.get("embargo_end_date")
            else pd.Series(False, index=working_df.index)
        )
        append_rows(working_df.index[train_mask].tolist(), fold_label, "train")
        append_rows(working_df.index[overlap_mask].tolist(), fold_label, "purged_overlap")
        append_rows(working_df.index[embargo_mask].tolist(), fold_label, "embargo")
        append_rows(working_df.index[validation_mask].tolist(), fold_label, "validation")
        purge_audit_rows.append(
            {
                "fold_label": fold_label,
                "evaluation_role": "validation",
                "train_start_date": metadata["train_start_date"],
                "train_end_date": metadata["train_end_date"],
                "overlap_purge_start_date": purge_metadata.get("overlap_purge_start_date"),
                "overlap_purge_end_date": purge_metadata.get("overlap_purge_end_date"),
                "embargo_start_date": purge_metadata.get("embargo_start_date"),
                "embargo_end_date": purge_metadata.get("embargo_end_date"),
                "validation_start_date": metadata["validation_start_date"],
                "validation_end_date": metadata["validation_end_date"],
                "train_date_count": int(fold["train_date_count"]),
                "validation_date_count": int(fold["validation_date_count"]),
                "purged_date_count": int(fold["purged_date_count"]),
                "overlap_purge_date_count": int(fold.get("overlap_purge_date_count", 0)),
                "embargo_date_count": int(fold.get("embargo_date_count", 0)),
                "train_row_count": int(train_mask.sum()),
                "validation_row_count": int(validation_mask.sum()),
                "purged_overlap_row_count": int(overlap_mask.sum()),
                "embargo_row_count": int(embargo_mask.sum()),
            }
        )

    holdout = split_payload["holdout"]
    holdout_metadata = holdout["date_metadata"]
    holdout_purge_metadata = holdout.get("purge_window_metadata", {})
    holdout_eval_mask = (
        (working_df["date"] >= pd.Timestamp(holdout_metadata["validation_start_date"]))
        & (working_df["date"] <= pd.Timestamp(holdout_metadata["validation_end_date"]))
    )
    holdout_train_mask = (
        working_df["date"] >= pd.Timestamp(holdout_metadata["train_start_date"])
        if holdout_metadata["train_start_date"] is not None
        else pd.Series(False, index=working_df.index)
    ) & (
        working_df["date"] <= pd.Timestamp(holdout_metadata["train_end_date"])
        if holdout_metadata["train_end_date"] is not None
        else pd.Series(False, index=working_df.index)
    )
    holdout_overlap_mask = (
        (working_df["date"] >= pd.Timestamp(holdout_purge_metadata["overlap_purge_start_date"]))
        & (working_df["date"] <= pd.Timestamp(holdout_purge_metadata["overlap_purge_end_date"]))
        if holdout_purge_metadata.get("overlap_purge_start_date") and holdout_purge_metadata.get("overlap_purge_end_date")
        else pd.Series(False, index=working_df.index)
    )
    holdout_embargo_mask = (
        (working_df["date"] >= pd.Timestamp(holdout_purge_metadata["embargo_start_date"]))
        & (working_df["date"] <= pd.Timestamp(holdout_purge_metadata["embargo_end_date"]))
        if holdout_purge_metadata.get("embargo_start_date") and holdout_purge_metadata.get("embargo_end_date")
        else pd.Series(False, index=working_df.index)
    )
    append_rows(working_df.index[holdout_train_mask].tolist(), "holdout", "holdout_train")
    append_rows(working_df.index[holdout_overlap_mask].tolist(), "holdout", "purged_overlap")
    append_rows(working_df.index[holdout_embargo_mask].tolist(), "holdout", "embargo")
    append_rows(working_df.index[holdout_eval_mask].tolist(), "holdout", "holdout_eval")
    purge_audit_rows.append(
        {
            "fold_label": "holdout",
            "evaluation_role": "holdout_eval",
            "train_start_date": holdout_metadata["train_start_date"],
            "train_end_date": holdout_metadata["train_end_date"],
            "overlap_purge_start_date": holdout_purge_metadata.get("overlap_purge_start_date"),
            "overlap_purge_end_date": holdout_purge_metadata.get("overlap_purge_end_date"),
            "embargo_start_date": holdout_purge_metadata.get("embargo_start_date"),
            "embargo_end_date": holdout_purge_metadata.get("embargo_end_date"),
            "validation_start_date": holdout_metadata["validation_start_date"],
            "validation_end_date": holdout_metadata["validation_end_date"],
            "train_date_count": int(holdout["train_date_count"]),
            "validation_date_count": int(holdout["holdout_date_count"]),
            "purged_date_count": int(holdout["purged_date_count"]),
            "overlap_purge_date_count": int(holdout.get("overlap_purge_date_count", 0)),
            "embargo_date_count": int(holdout.get("embargo_date_count", 0)),
            "train_row_count": int(holdout_train_mask.sum()),
            "validation_row_count": int(holdout_eval_mask.sum()),
            "purged_overlap_row_count": int(holdout_overlap_mask.sum()),
            "embargo_row_count": int(holdout_embargo_mask.sum()),
        }
    )
    fold_map_df = pd.DataFrame(fold_map_rows).sort_values(["row_index", "fold_label", "fold_role"]).reset_index(drop=True)
    purge_audit_df = pd.DataFrame(purge_audit_rows).sort_values(["fold_label"]).reset_index(drop=True)
    return fold_map_df, purge_audit_df


def save_validation_artifacts(
    validation_dir: Path,
    fold_map_df: pd.DataFrame,
    fold_summary_df: pd.DataFrame,
    purge_audit_df: pd.DataFrame,
) -> None:
    validation_dir.mkdir(parents=True, exist_ok=True)
    fold_map_df.to_parquet(validation_dir / "fold_map.parquet", index=False)
    fold_summary_df.to_csv(validation_dir / "fold_summary.csv", index=False)
    purge_audit_df.to_csv(validation_dir / "purge_audit.csv", index=False)


def save_tuning_artifacts(tuning_dir: Path, result_df: pd.DataFrame, tuning_artifacts: dict[str, pd.DataFrame]) -> None:
    tuning_dir.mkdir(parents=True, exist_ok=True)
    result_df[["model_name", "used_tuning", "tuned_model_params"]].to_csv(
        tuning_dir / "best_model_configs.csv",
        index=False,
    )
    for artifact_name, artifact_df in tuning_artifacts.items():
        if artifact_df is None or artifact_df.empty:
            continue
        safe_name = str(artifact_name).replace("__", "_").replace("/", "_")
        artifact_df.to_csv(tuning_dir / f"{safe_name}.csv", index=False)


def evaluate_model_candidate(
    *,
    panel_df: pd.DataFrame,
    variant: VariantSpec,
    split_payload: dict,
    kept_global: list[str],
    model_name: str,
    model_params: dict,
    threshold: float,
    panel_name: str,
    max_missingness_pct: float,
) -> tuple[dict, list[dict], list[dict]]:
    fold_metrics: list[dict] = []
    fold_summary_rows: list[dict] = []
    concentration_rows: list[dict] = []
    top_feature_counter: Counter[str] = Counter()
    last_usable: list[str] = []
    last_dropped_missing: list[str] = []
    last_dropped_constant: list[str] = []
    last_missingness: dict[str, float] = {}

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
        fitted_model, backend = fit_model(
            model_name,
            clipped_train[usable_features],
            clipped_train["target"].astype(int),
            model_params=model_params,
        )
        y_prob = fitted_model.predict_proba(clipped_validation[usable_features])[:, 1]
        fold_result = evaluate_extended(clipped_validation, y_prob, threshold=threshold)
        fold_metrics.append(fold_result)
        fold_label = f"fold_{int(fold['fold_number'])}"
        fold_summary_rows.append(
            {
                "panel_name": panel_name,
                "label_variant": variant.variant_name,
                "model_name": model_name,
                "fold_label": fold_label,
                "evaluation_role": "validation",
                "train_start_date": fold["date_metadata"]["train_start_date"],
                "train_end_date": fold["date_metadata"]["train_end_date"],
                "validation_start_date": fold["date_metadata"]["validation_start_date"],
                "validation_end_date": fold["date_metadata"]["validation_end_date"],
                "train_row_count": int(len(train_active)),
                "validation_row_count": int(len(validation_active)),
                "purged_date_count": int(fold["purged_date_count"]),
                "overlap_purge_date_count": int(fold.get("overlap_purge_date_count", 0)),
                "embargo_date_count": int(fold.get("embargo_date_count", 0)),
                "train_positive_rate": _safe_class_balance(train_active["target"]),
                "validation_positive_rate": _safe_class_balance(validation_active["target"]),
                "auc": fold_result.get("auc_roc"),
                "f1": fold_result.get("f1"),
                "log_loss": fold_result.get("log_loss"),
                "calibration_score": fold_result.get("brier_score"),
                "backend": backend,
            }
        )
        fold_concentration_rows, fold_top_features = _collect_top_feature_rows(
            fitted_model=fitted_model,
            feature_names=usable_features,
            x_frame=clipped_validation,
            y_target=clipped_validation["target"],
            model_name=model_name,
            fold_label=fold_label,
            evaluation_role="validation",
        )
        concentration_rows.extend(fold_concentration_rows)
        top_feature_counter.update(fold_top_features)
        last_usable = usable_features
        last_dropped_missing = dropped_missing
        last_dropped_constant = dropped_constant
        last_missingness = missingness_by_feature

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
        model_params=model_params,
    )
    holdout_prob = fitted_model.predict_proba(clipped_holdout[holdout_usable])[:, 1]
    holdout_metrics = evaluate_extended(clipped_holdout, holdout_prob, threshold=threshold)
    holdout_concentration_rows, holdout_top_features = _collect_top_feature_rows(
        fitted_model=fitted_model,
        feature_names=holdout_usable,
        x_frame=clipped_holdout,
        y_target=clipped_holdout["target"],
        model_name=model_name,
        fold_label="holdout",
        evaluation_role="holdout_eval",
    )
    concentration_rows.extend(holdout_concentration_rows)
    top_feature_counter.update(holdout_top_features)
    cv_summary = summarize_metric_dicts(fold_metrics)
    fold_aucs = [metrics.get("auc_roc") for metrics in fold_metrics if metrics.get("auc_roc") is not None]
    dominant_feature_name = None
    dominant_feature_top3_folds = 0
    if top_feature_counter:
        dominant_feature_name, dominant_feature_top3_folds = sorted(
            top_feature_counter.items(),
            key=lambda item: (-item[1], item[0]),
        )[0]
    fold_summary_rows.append(
        {
            "panel_name": panel_name,
            "label_variant": variant.variant_name,
            "model_name": model_name,
            "fold_label": "holdout",
            "evaluation_role": "holdout_eval",
            "train_start_date": split_payload["holdout"]["date_metadata"]["train_start_date"],
            "train_end_date": split_payload["holdout"]["date_metadata"]["train_end_date"],
            "validation_start_date": split_payload["holdout"]["date_metadata"]["validation_start_date"],
            "validation_end_date": split_payload["holdout"]["date_metadata"]["validation_end_date"],
            "train_row_count": int(len(holdout_train_active)),
            "validation_row_count": int(len(holdout_active)),
            "purged_date_count": int(split_payload["holdout"]["purged_date_count"]),
            "overlap_purge_date_count": int(split_payload["holdout"].get("overlap_purge_date_count", 0)),
            "embargo_date_count": int(split_payload["holdout"].get("embargo_date_count", 0)),
            "train_positive_rate": _safe_class_balance(holdout_train_active["target"]),
            "validation_positive_rate": _safe_class_balance(holdout_active["target"]),
            "auc": holdout_metrics.get("auc_roc"),
            "f1": holdout_metrics.get("f1"),
            "log_loss": holdout_metrics.get("log_loss"),
            "calibration_score": holdout_metrics.get("brier_score"),
            "backend": backend,
        }
    )
    row = {
        "panel_name": panel_name,
        "label_variant": variant.variant_name,
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
        "best_fold_auc": float(max(fold_aucs)) if fold_aucs else None,
        "worst_fold_auc": float(min(fold_aucs)) if fold_aucs else None,
        "cv_auc_std": cv_summary.get("auc_roc_std"),
        "holdout_row_count": holdout_metrics["row_count"],
        "model_backend": backend,
        "xgboost_backend": backend if model_name == "xgboost" else "cpu",
        "fold_missingness_exclusions_last_fold": json.dumps(sorted(set(last_dropped_missing))),
        "fold_constant_exclusions_last_fold": json.dumps(sorted(set(last_dropped_constant))),
        "holdout_missingness_exclusions": json.dumps(sorted(set(holdout_dropped_missing))),
        "holdout_constant_exclusions": json.dumps(sorted(set(holdout_dropped_constant))),
        "usable_feature_count_last_fold": len(last_usable),
        "usable_feature_columns_last_fold": json.dumps(last_usable),
        "train_missingness_by_feature_pct_last_fold": json.dumps(last_missingness),
        "holdout_start": None,
        "n_splits": None,
        "embargo_days": None,
        "min_train_dates": None,
        "dominant_feature_name": dominant_feature_name,
        "dominant_feature_top3_folds": int(dominant_feature_top3_folds),
        "feature_concentration_ratio": (
            float(dominant_feature_top3_folds) / float(max(len(fold_metrics) + 1, 1))
            if dominant_feature_name is not None
            else None
        ),
    }
    return row, fold_summary_rows, concentration_rows


def tune_model_with_optuna(
    *,
    panel_df: pd.DataFrame,
    variant: VariantSpec,
    split_payload: dict,
    kept_global: list[str],
    model_name: str,
    config: dict,
    threshold: float,
    panel_name: str,
    max_missingness_pct: float,
    tuning_spec: dict,
) -> tuple[dict, pd.DataFrame]:
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - dependency/runtime guard
        raise ImportError("Optuna tuning was requested but `optuna` is not installed.") from exc

    model_tuning = tuning_spec["models"].get(model_name, {}) or {}
    n_trials = int(model_tuning.get("n_trials", tuning_spec["n_trials"]))
    if n_trials <= 0:
        return {}, pd.DataFrame()

    sampler = optuna.samplers.TPESampler(seed=int(config.get("random_seed", {}).get("numpy", 42)))
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial) -> float:
        search_space = model_tuning.get("search_space")
        candidate_params = resolve_model_params(
            config,
            model_name,
            override_params=_suggest_trial_params(trial, model_name, search_space=search_space),
        )
        row, _, _ = evaluate_model_candidate(
            panel_df=panel_df,
            variant=variant,
            split_payload=split_payload,
            kept_global=kept_global,
            model_name=model_name,
            model_params=candidate_params,
            threshold=threshold,
            panel_name=panel_name,
            max_missingness_pct=max_missingness_pct,
        )
        mean_cv_auc = float(row["cv_auc_mean"]) if row["cv_auc_mean"] is not None else float("-inf")
        cv_auc_std = float(row["cv_auc_std"]) if row["cv_auc_std"] is not None else 1.0
        concentration_ratio = float(row["feature_concentration_ratio"] or 0.0)
        score = mean_cv_auc
        score -= tuning_spec["stability_penalty"] * cv_auc_std
        score -= tuning_spec["concentration_penalty"] * concentration_ratio
        trial.set_user_attr("cv_auc_mean", row["cv_auc_mean"])
        trial.set_user_attr("cv_auc_std", row["cv_auc_std"])
        trial.set_user_attr("holdout_auc", row["holdout_auc"])
        trial.set_user_attr("feature_concentration_ratio", row["feature_concentration_ratio"])
        return score

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    trials_df = study.trials_dataframe().copy()
    if trials_df.empty:
        return {}, trials_df
    return resolve_model_params(config, model_name, override_params=study.best_trial.params), trials_df


def run_model_matrix(
    panel_df: pd.DataFrame,
    variant: VariantSpec,
    model_names: list[str],
    candidate_features: list[str],
    explicit_exclusions: list[str],
    holdout_start: str,
    n_splits: int,
    embargo_days: int,
    min_train_dates: int,
    threshold: float,
    panel_name: str,
    max_missingness_pct: float = 20.0,
    promotion_strategy: str = "cv_mean_auc",
    config: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    config = config or {}
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
    kept_global, auto_all_missing, auto_constant = compute_global_feature_exclusions(
        panel_df,
        global_candidates,
        holdout_start=holdout_start,
    )

    rows = []
    fold_summary_rows = []
    concentration_rows = []
    tuning_artifacts: dict[str, pd.DataFrame] = {}
    tuning_spec = resolve_tuning_spec(config)
    reference_row = resolve_reference_benchmark_row(config) if config else None
    for model_name in model_names:
        tuned_params: dict = {}
        if tuning_spec["enabled"] and model_name in {"xgboost", "lightgbm", "catboost"}:
            tuned_params, tuning_artifacts[model_name] = tune_model_with_optuna(
                panel_df=panel_df,
                variant=variant,
                split_payload=split_payload,
                kept_global=kept_global,
                model_name=model_name,
                config=config,
                threshold=threshold,
                panel_name=panel_name,
                max_missingness_pct=max_missingness_pct,
                tuning_spec=tuning_spec,
            )
        model_params = resolve_model_params(config, model_name, override_params=tuned_params)
        row, model_fold_rows, model_concentration_rows = evaluate_model_candidate(
            panel_df=panel_df,
            variant=variant,
            split_payload=split_payload,
            kept_global=kept_global,
            model_name=model_name,
            model_params=model_params,
            threshold=threshold,
            panel_name=panel_name,
            max_missingness_pct=max_missingness_pct,
        )
        row.update(
            {
                "explicit_feature_exclusions": json.dumps(explicit_exclusions),
                "auto_all_missing_exclusions": json.dumps(auto_all_missing),
                "auto_constant_exclusions": json.dumps(auto_constant),
                "holdout_start": holdout_start,
                "n_splits": n_splits,
                "embargo_days": embargo_days,
                "min_train_dates": min_train_dates,
                "used_tuning": bool(tuned_params),
                "tuned_model_params": json.dumps(model_params, sort_keys=True),
            }
        )
        if tuning_artifacts.get(model_name) is not None and not tuning_artifacts[model_name].empty:
            row["optuna_best_objective"] = float(tuning_artifacts[model_name]["value"].max())
        else:
            row["optuna_best_objective"] = None
        reproducibility_rows = []
        for seed in tuning_spec["reproducibility_seeds"]:
            seeded_params = resolve_model_params(config, model_name, override_params=tuned_params, seed_override=seed)
            seeded_row, _, _ = evaluate_model_candidate(
                panel_df=panel_df,
                variant=variant,
                split_payload=split_payload,
                kept_global=kept_global,
                model_name=model_name,
                model_params=seeded_params,
                threshold=threshold,
                panel_name=panel_name,
                max_missingness_pct=max_missingness_pct,
            )
            reproducibility_rows.append(
                {
                    "model_name": model_name,
                    "seed": int(seed),
                    "cv_auc_mean": seeded_row["cv_auc_mean"],
                    "holdout_auc": seeded_row["holdout_auc"],
                }
            )
        reproducibility_df = pd.DataFrame(reproducibility_rows)
        row["reproducibility_cv_auc_std"] = float(reproducibility_df["cv_auc_mean"].std(ddof=0))
        row["reproducibility_holdout_auc_std"] = float(reproducibility_df["holdout_auc"].std(ddof=0))
        row["reproducible_under_threshold"] = bool(
            row["reproducibility_cv_auc_std"] <= tuning_spec["max_cv_auc_std"]
            and row["reproducibility_holdout_auc_std"] <= tuning_spec["max_holdout_auc_std"]
        )
        tuning_artifacts[f"{model_name}__reproducibility"] = reproducibility_df
        rows.append(row)
        fold_summary_rows.extend(model_fold_rows)
        concentration_rows.extend(model_concentration_rows)

    result_df = pd.DataFrame(rows).sort_values("model_name").reset_index(drop=True)
    if promotion_strategy == "stability_aware":
        best_model_name = choose_best_model_with_stability(result_df.to_dict(orient="records"))
    else:
        best_model_name = choose_best_model(result_df.to_dict(orient="records"))
    result_df["is_selected_primary_model"] = result_df["model_name"] == best_model_name
    selected_row = result_df.loc[result_df["is_selected_primary_model"]].iloc[0].to_dict()
    promotion_cfg = config.get("promotion", {}) or {}
    holdout_delta_required = float(promotion_cfg.get("min_holdout_auc_delta", 0.0))
    selected_holdout = float(selected_row["holdout_auc"]) if selected_row["holdout_auc"] is not None else float("-inf")
    selected_worst_fold = float(selected_row["worst_fold_auc"]) if selected_row["worst_fold_auc"] is not None else float("-inf")
    selected_stable = bool(selected_row.get("reproducible_under_threshold", True))
    promoted = selected_stable
    promotion_reasons = []
    if reference_row is not None:
        reference_holdout = float(reference_row.get("holdout_auc", float("-inf")))
        reference_worst_fold = float(reference_row.get("worst_fold_auc", float("-inf")))
        promoted = promoted and (selected_holdout > (reference_holdout + holdout_delta_required))
        promoted = promoted and (selected_worst_fold >= reference_worst_fold)
        if selected_holdout <= (reference_holdout + holdout_delta_required):
            promotion_reasons.append("holdout_not_better_than_reference")
        if selected_worst_fold < reference_worst_fold:
            promotion_reasons.append("worst_fold_below_reference")
    if not selected_stable:
        promotion_reasons.append("reproducibility_threshold_failed")
    result_df["promotion_status"] = "reference_only"
    result_df["promotion_reason"] = "not_selected"
    result_df.loc[result_df["is_selected_primary_model"], "promotion_status"] = (
        "promoted" if promoted else "candidate_only"
    )
    result_df.loc[result_df["is_selected_primary_model"], "promotion_reason"] = (
        "beats_reference_with_stability"
        if promoted
        else ",".join(promotion_reasons) if promotion_reasons else "selected_without_reference_gate"
    )
    fold_summary_df = pd.DataFrame(fold_summary_rows)
    if not fold_summary_df.empty:
        worst_fold_by_model = (
            fold_summary_df.loc[fold_summary_df["evaluation_role"] == "validation"]
            .groupby("model_name", sort=False)["auc"]
            .min()
            .rename("worst_fold_auc")
        )
        fold_summary_df = fold_summary_df.merge(worst_fold_by_model, on="model_name", how="left")
    summary = {
        "best_model_name": best_model_name,
        "promotion_strategy": promotion_strategy,
        "explicit_exclusions": explicit_exclusions,
        "auto_all_missing": auto_all_missing,
        "auto_constant": auto_constant,
        "split_payload": split_payload,
        "fold_summary_df": fold_summary_df,
        "concentration_df": pd.DataFrame(concentration_rows),
        "tuning_artifacts": tuning_artifacts,
        "reference_row": reference_row,
        "promotion_recommended": promoted,
    }
    return result_df, summary


def export_selected_model_shap(
    panel_df: pd.DataFrame,
    variant: VariantSpec,
    candidate_features: list[str],
    explicit_exclusions: list[str],
    holdout_start: str,
    n_splits: int,
    embargo_days: int,
    min_train_dates: int,
    selected_model_name: str,
    selected_model_params: dict | None,
    shap_plot_path: Path,
    shap_csv_path: Path,
    max_missingness_pct: float = 20.0,
) -> dict | None:
    if selected_model_name != "xgboost":
        return None

    try:
        import matplotlib.pyplot as plt
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP export requires the optional 'shap' dependency. Install it with `pip install shap`."
        ) from exc

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

    holdout_train_full = panel_df.iloc[split_payload["holdout"]["train_indices"]].copy()
    holdout_full = panel_df.iloc[split_payload["holdout"]["holdout_indices"]].copy()
    holdout_train_active, _ = apply_variant_label_mode(holdout_train_full, variant)
    holdout_active, _ = apply_variant_label_mode(holdout_full, variant)
    holdout_usable, _, _, _ = select_usable_features(
        holdout_train_active,
        kept_global,
        max_missingness_pct=max_missingness_pct,
    )
    clipped_train, clipped_holdout = clip_outliers(holdout_train_active, holdout_active, holdout_usable)
    fitted_model, backend = fit_model(
        selected_model_name,
        clipped_train[holdout_usable],
        clipped_train["target"].astype(int),
        model_params=selected_model_params,
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
    if isinstance(shap_values, list):
        shap_matrix = np.asarray(shap_values[-1])
    else:
        shap_matrix = np.asarray(shap_values)
    if shap_matrix.ndim == 3:
        shap_matrix = shap_matrix[:, :, -1]

    mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
    shap_importance_df = (
        pd.DataFrame(
            {
                "feature": holdout_usable,
                "mean_abs_shap": mean_abs_shap,
                "mean_feature_value": transformed_holdout.mean(axis=0).to_numpy(),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    ensure_parent_dir(shap_plot_path)
    ensure_parent_dir(shap_csv_path)
    shap_importance_df.to_csv(shap_csv_path, index=False)

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_matrix, transformed_holdout, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(shap_plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    return {
        "plot_path": str(shap_plot_path),
        "csv_path": str(shap_csv_path),
        "row_count": int(len(transformed_holdout)),
        "feature_count": int(len(holdout_usable)),
        "backend": backend,
        "top_feature": str(shap_importance_df.iloc[0]["feature"]) if not shap_importance_df.empty else None,
    }


def build_markdown_report(
    result_df: pd.DataFrame,
    summary: dict,
    old_baseline: dict | None,
    report_metadata: dict[str, str],
) -> str:
    best_row = result_df.loc[result_df["is_selected_primary_model"]].iloc[0]
    model_list = ", ".join(f"`{name}`" for name in result_df["model_name"].tolist())
    lines = [
        f"# {report_metadata['report_title']}",
        "",
        "## Locked Setup",
        "",
        f"- Primary panel: `{report_metadata['panel_display_name']}`",
        f"- Primary label: `{report_metadata['label_description']}`",
        f"- Models: {model_list}",
        "- 2024 holdout policy: unchanged",
        f"- {report_metadata['setup_note']}",
        "",
        "## Per-Model Results",
        "",
        "| Model | Mean CV AUC | CV AUC Std | Worst Fold AUC | Holdout AUC | Holdout Log Loss | Holdout Rank IC | Backend | Dominant Feature | Concentration | Repro Holdout Std | Promotion |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|",
    ]
    for _, row in result_df.sort_values("model_name").iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model_name"]),
                    format_metric(row["cv_auc_mean"]),
                    format_metric(row["cv_auc_std"]),
                    format_metric(row["worst_fold_auc"]),
                    format_metric(row["holdout_auc"]),
                    format_metric(row["holdout_log_loss"]),
                    format_metric(row["holdout_rank_ic"]),
                    str(row.get("model_backend", "cpu")),
                    str(row.get("dominant_feature_name") or "n/a"),
                    format_metric(row.get("feature_concentration_ratio")),
                    format_metric(row.get("reproducibility_holdout_auc_std")),
                    str(row.get("promotion_status", "")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Feature Exclusions",
            "",
            f"- Explicit exclusions: `{', '.join(summary['explicit_exclusions']) if summary['explicit_exclusions'] else 'none'}`",
            f"- Auto all-missing exclusions: `{', '.join(summary['auto_all_missing']) if summary['auto_all_missing'] else 'none'}`",
            f"- Auto constant exclusions: `{', '.join(summary['auto_constant']) if summary['auto_constant'] else 'none'}`",
            "",
            "## Selected Primary Model",
            "",
            f"- Selected model: `{best_row['model_name']}`",
            f"- Promotion strategy: `{summary.get('promotion_strategy', 'cv_mean_auc')}`",
            f"- Mean CV AUC: `{format_metric(best_row['cv_auc_mean'])}`",
            f"- CV AUC std: `{format_metric(best_row['cv_auc_std'])}`",
            f"- Worst fold AUC: `{format_metric(best_row['worst_fold_auc'])}`",
            f"- 2024 holdout AUC: `{format_metric(best_row['holdout_auc'])}`",
            f"- 2024 holdout log loss: `{format_metric(best_row['holdout_log_loss'])}`",
            f"- Dominant feature concentration: `{format_metric(best_row.get('feature_concentration_ratio'))}`",
            f"- Reproducibility holdout AUC std: `{format_metric(best_row.get('reproducibility_holdout_auc_std'))}`",
            f"- Promotion status: `{best_row.get('promotion_status', 'n/a')}`",
            f"- Promotion reason: `{best_row.get('promotion_reason', 'n/a')}`",
            "",
            "## Interpretation",
            "",
        ]
    )
    if old_baseline is not None:
        lines.append(
            f"- Against the old daily/event_v1 direction (`{old_baseline['panel_name']}` best model `{old_baseline['model_name']}`), "
            f"the redesigned event setup improves best CV AUC from `{format_metric(old_baseline['cv_auc'])}` to `{format_metric(best_row['cv_auc_mean'])}` "
            f"and best holdout AUC from `{format_metric(old_baseline['holdout_auc'])}` to `{format_metric(best_row['holdout_auc'])}`."
    )
    lines.append(f"- {report_metadata['interpretation_note']}")
    if "xgboost" in set(result_df["model_name"]) and str(result_df.loc[result_df["model_name"] == "xgboost", "xgboost_backend"].iloc[0]).startswith("cpu"):
        lines.append(
            f"- XGBoost ran on CPU in this benchmark because GPU was not accepted by the runtime: `{result_df.loc[result_df['model_name'] == 'xgboost', 'xgboost_backend'].iloc[0]}`."
        )
    if summary.get("reference_row") is not None:
        lines.append(
            f"- Reference benchmark for promotion was `{summary['reference_row'].get('model_name', 'unknown')}` with holdout AUC `{format_metric(summary['reference_row'].get('holdout_auc'))}`."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    set_random_seeds(config.get("random_seed", {}))
    report_metadata = resolve_report_metadata(config)

    panel_path = Path(config["panel"]["path"])
    csv_path = Path(config["outputs"]["csv"])
    markdown_path = Path(config["outputs"]["markdown"])
    shap_plot_path, shap_csv_path = resolve_shap_output_paths(config)
    concentration_csv_path = resolve_concentration_output_path(config)
    tuning_output_dir = resolve_tuning_output_dir(config)
    validation_output_dir = resolve_validation_output_dir(config)
    promotion_strategy = resolve_promotion_strategy(config)
    ensure_parent_dir(csv_path)
    ensure_parent_dir(markdown_path)
    ensure_parent_dir(concentration_csv_path)

    print(f"Loading event panel from: {panel_path}")
    panel_df = load_event_panel(panel_path)
    panel_name = str(config.get("panel", {}).get("name", "event_panel_v2"))
    label_path_value = config.get("label", {}).get("path")
    variant = VariantSpec(
        variant_name=str(config["label"]["variant_name"]),
        horizon_days=int(config["label"]["horizon_days"]),
        label_mode="precomputed" if label_path_value else str(config["label"]["mode"]),
        quantile=config["label"].get("quantile"),
        threshold=config["label"].get("threshold"),
    )
    if label_path_value:
        label_path = Path(str(label_path_value))
        print(f"Loading prebuilt label map from: {label_path}")
        label_df = load_prebuilt_label_map(label_path)
    else:
        prices_path = Path(config.get("prices", {}).get("path", str(PRICE_INPUT_PATH)))
        print(f"Loading prices from: {prices_path}")
        prices_df = normalize_price_data(load_price_data(prices_path))
        label_df = build_daily_label_table(
            prices_df,
            horizon_days=variant.horizon_days,
            benchmark_mode=str(config["label"]["benchmark_mode"]),
        )
    labeled_panel_df = attach_labels_to_event_panel(panel_df, label_df)

    result_df, summary = run_model_matrix(
        panel_df=labeled_panel_df,
        variant=variant,
        model_names=list(config["models"]),
        candidate_features=resolve_candidate_features(labeled_panel_df, config),
        explicit_exclusions=list(config["feature_exclusions"]["explicit"]),
        holdout_start=str(config["holdout"]["start"]),
        n_splits=int(config["cv"]["n_splits"]),
        embargo_days=int(config["cv"]["embargo_days"]),
        min_train_dates=int(config["cv"]["min_train_dates"]),
        threshold=resolve_threshold(config),
        panel_name=panel_name,
        max_missingness_pct=resolve_max_missingness_pct(config.get("feature_exclusions")),
        promotion_strategy=promotion_strategy,
        config=config,
    )

    print(f"Saving benchmark CSV to: {csv_path}")
    result_df.to_csv(csv_path, index=False)
    if not summary["concentration_df"].empty:
        print(f"Saving concentration diagnostics to: {concentration_csv_path}")
        summary["concentration_df"].to_csv(concentration_csv_path, index=False)
    save_tuning_artifacts(tuning_output_dir, result_df, summary["tuning_artifacts"])

    if validation_output_dir is not None:
        fold_map_df, purge_audit_df = build_validation_artifacts(labeled_panel_df, summary["split_payload"])
        save_validation_artifacts(
            validation_dir=validation_output_dir,
            fold_map_df=fold_map_df,
            fold_summary_df=summary["fold_summary_df"],
            purge_audit_df=purge_audit_df,
        )
        print(f"Saved validation artifacts to: {validation_output_dir}")

    old_baseline = load_old_baseline()
    markdown = build_markdown_report(result_df, summary, old_baseline, report_metadata)
    print(f"Saving benchmark Markdown to: {markdown_path}")
    markdown_path.write_text(markdown, encoding="utf-8")

    shap_summary = export_selected_model_shap(
        panel_df=labeled_panel_df,
        variant=variant,
        candidate_features=resolve_candidate_features(labeled_panel_df, config),
        explicit_exclusions=list(config["feature_exclusions"]["explicit"]),
        holdout_start=str(config["holdout"]["start"]),
        n_splits=int(config["cv"]["n_splits"]),
        embargo_days=int(config["cv"]["embargo_days"]),
        min_train_dates=int(config["cv"]["min_train_dates"]),
        selected_model_name=str(summary["best_model_name"]),
        selected_model_params=json.loads(
            str(result_df.loc[result_df["is_selected_primary_model"], "tuned_model_params"].iloc[0])
        ),
        shap_plot_path=shap_plot_path,
        shap_csv_path=shap_csv_path,
        max_missingness_pct=resolve_max_missingness_pct(config.get("feature_exclusions")),
    )

    best_row = result_df.loc[result_df["is_selected_primary_model"]].iloc[0]
    print("\nPhase 4 Benchmark Summary")
    print("-" * 60)
    for _, row in result_df.sort_values("model_name").iterrows():
        print(
            f"{row['model_name']:<20} cv_auc={format_metric(row['cv_auc_mean'])} "
            f"holdout_auc={format_metric(row['holdout_auc'])} backend={row.get('model_backend', row['xgboost_backend'])}"
        )
    print(f"\nSelected primary model: {best_row['model_name']}")
    if shap_summary is not None:
        print(
            f"SHAP exported for selected model on {shap_summary['row_count']} holdout rows "
            f"across {shap_summary['feature_count']} features."
        )
        print(f"SHAP top feature: {shap_summary['top_feature']}")
        print(f"SHAP plot: {shap_summary['plot_path']}")
        print(f"SHAP CSV: {shap_summary['csv_path']}")


if __name__ == "__main__":
    main()
