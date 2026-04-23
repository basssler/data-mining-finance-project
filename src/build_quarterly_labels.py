"""Build quarterly event labels from tradable filing timestamps.

This script materializes versioned label maps for the quarterly event lane and
logs distribution diagnostics overall and by validation fold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.config_event_v1 import PRICE_INPUT_PATH
from src.label_comparison_event_v2 import build_daily_label_table
from src.labels_event_v1 import load_price_data, normalize_price_data
from src.paths import QUARTERLY_OUTPUTS_LABELS_DIR
from src.project_config import ensure_stock_prediction_directories
from src.validation_event_v1 import make_event_v1_splits

DEFAULT_PANEL_PATH = Path("outputs") / "quarterly" / "panels" / "quarterly_event_panel_features.parquet"
DEFAULT_HORIZONS = [5, 10, 21, 63]
DEFAULT_THRESHOLD = 0.015
DEFAULT_QUANTILE_BUCKETS = 3
DEFAULT_BENCHMARK_MODE = "sector_equal_weight_ex_self"
DEFAULT_RECOMMENDED_LABEL = "event_v2_21d_excess_sign"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build quarterly event label maps.")
    parser.add_argument("--panel-path", default=str(DEFAULT_PANEL_PATH))
    parser.add_argument("--price-path", default=str(PRICE_INPUT_PATH))
    parser.add_argument("--benchmark-mode", default=DEFAULT_BENCHMARK_MODE)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--holdout-start", default="2024-01-01")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--embargo-days", type=int, default=5)
    parser.add_argument("--min-train-dates", type=int, default=252)
    return parser.parse_args()


def load_quarterly_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Quarterly panel file was not found: {path}")
    panel_df = pd.read_parquet(path).copy()
    required = ["ticker", "event_id", "tradable_date"]
    missing = [column for column in required if column not in panel_df.columns]
    if missing:
        raise ValueError("Quarterly panel is missing required columns: " + ", ".join(missing))
    panel_df["ticker"] = panel_df["ticker"].astype("string")
    panel_df["event_id"] = panel_df["event_id"].astype("string")
    panel_df["date"] = pd.to_datetime(panel_df["tradable_date"], errors="coerce").dt.normalize()
    panel_df = panel_df.dropna(subset=["ticker", "event_id", "date"]).copy()
    if "validation_group" in panel_df.columns:
        panel_df["validation_group"] = panel_df["validation_group"].astype("string")
    return panel_df.sort_values(["date", "ticker", "event_id"]).reset_index(drop=True)


def attach_base_returns(
    panel_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    horizon_days: int,
    benchmark_mode: str,
) -> pd.DataFrame:
    daily_label_df = build_daily_label_table(
        prices_df=prices_df,
        horizon_days=horizon_days,
        benchmark_mode=benchmark_mode,
    )
    base = panel_df.merge(
        daily_label_df,
        on=["ticker", "date"],
        how="left",
        validate="many_to_one",
    )
    return base.sort_values(["date", "ticker", "event_id"]).reset_index(drop=True)


def _event_variant_name(horizon_days: int, family: str) -> str:
    mapping = {
        "excess": f"event_v2_{horizon_days}d_excess_sign",
        "thresholded": f"event_v2_{horizon_days}d_excess_threshold",
        "quantile": f"event_v2_{horizon_days}d_excess_quantile",
    }
    return mapping[family]


def _label_map_path(horizon_days: int, family: str) -> Path:
    return QUARTERLY_OUTPUTS_LABELS_DIR / f"label_map_{family}_{horizon_days}d.parquet"


def _diagnostic_path(horizon_days: int, family: str) -> Path:
    return QUARTERLY_OUTPUTS_LABELS_DIR / f"label_diagnostics_{family}_{horizon_days}d.json"


def build_excess_label_map(base_df: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    label_map = base_df.copy()
    label_map["target"] = pd.Series(pd.NA, index=label_map.index, dtype="Int64")
    valid = label_map["excess_forward_return"].notna()
    label_map.loc[valid, "target"] = (label_map.loc[valid, "excess_forward_return"] > 0).astype("int64")
    label_map["label_family"] = "excess"
    label_map["label_variant"] = _event_variant_name(horizon_days, "excess")
    label_map["dropped_ambiguous"] = False
    label_map["label_available"] = valid
    return label_map


def build_thresholded_label_map(base_df: pd.DataFrame, horizon_days: int, threshold: float) -> pd.DataFrame:
    label_map = base_df.copy()
    label_map["target"] = pd.Series(pd.NA, index=label_map.index, dtype="Int64")
    valid = label_map["excess_forward_return"].notna()
    positive = valid & (label_map["excess_forward_return"] > threshold)
    negative = valid & (label_map["excess_forward_return"] < -threshold)
    ambiguous = valid & ~(positive | negative)
    label_map.loc[positive, "target"] = 1
    label_map.loc[negative, "target"] = 0
    label_map["label_family"] = "thresholded"
    label_map["label_variant"] = _event_variant_name(horizon_days, "thresholded")
    label_map["dropped_ambiguous"] = ambiguous
    label_map["label_available"] = positive | negative
    label_map["threshold"] = float(threshold)
    return label_map


def build_quantile_label_map(base_df: pd.DataFrame, horizon_days: int, buckets: int = DEFAULT_QUANTILE_BUCKETS) -> pd.DataFrame:
    label_map = base_df.copy()
    label_map["target"] = pd.Series(pd.NA, index=label_map.index, dtype="Int64")
    label_map["label_family"] = "quantile"
    label_map["label_variant"] = _event_variant_name(horizon_days, "quantile")
    label_map["dropped_ambiguous"] = False
    label_map["label_available"] = False

    for _, group in label_map.groupby("date", sort=False):
        valid = group["excess_forward_return"].notna()
        valid_count = int(valid.sum())
        if valid_count < buckets:
            if valid_count > 0:
                label_map.loc[group.index[valid], "dropped_ambiguous"] = True
            continue

        ranked = group.loc[valid, "excess_forward_return"].rank(method="first", pct=True)
        lower_mask = ranked <= (1.0 / buckets)
        upper_mask = ranked > ((buckets - 1) / buckets)
        middle_mask = ~(lower_mask | upper_mask)

        lower_index = ranked.index[lower_mask]
        upper_index = ranked.index[upper_mask]
        middle_index = ranked.index[middle_mask]

        label_map.loc[lower_index, "target"] = 0
        label_map.loc[upper_index, "target"] = 1
        label_map.loc[lower_index.union(upper_index), "label_available"] = True
        label_map.loc[middle_index, "dropped_ambiguous"] = True

    label_map["quantile_buckets"] = int(buckets)
    return label_map


def summarize_label_distribution(label_df: pd.DataFrame) -> dict[str, int | float | None]:
    kept = label_df.loc[label_df["label_available"]].copy()
    class_counts = kept["target"].astype("Int64").value_counts(dropna=False).to_dict()
    return {
        "row_count": int(len(label_df)),
        "label_available_count": int(label_df["label_available"].sum()),
        "dropped_ambiguous_count": int(label_df["dropped_ambiguous"].sum()),
        "missing_return_count": int(label_df["excess_forward_return"].isna().sum()),
        "class_0_count": int(class_counts.get(0, 0)),
        "class_1_count": int(class_counts.get(1, 0)),
        "class_1_rate": (
            float(kept["target"].astype(int).mean()) if not kept.empty else None
        ),
        "mean_excess_return": (
            float(pd.to_numeric(label_df["excess_forward_return"], errors="coerce").mean())
            if label_df["excess_forward_return"].notna().any()
            else None
        ),
    }


def build_fold_distribution_summary(
    label_df: pd.DataFrame,
    horizon_days: int,
    holdout_start: str,
    n_splits: int,
    embargo_days: int,
    min_train_dates: int,
) -> dict[str, dict[str, int | float | None]]:
    split_payload = make_event_v1_splits(
        df=label_df,
        date_col="date",
        horizon_days=horizon_days,
        n_splits=n_splits,
        embargo_days=embargo_days,
        holdout_start=holdout_start,
        min_train_dates=min_train_dates,
    )
    by_fold: dict[str, dict[str, int | float | None]] = {}
    for fold in split_payload["folds"]:
        by_fold[f"fold_{fold['fold_number']}_train"] = summarize_label_distribution(
            label_df.iloc[fold["train_indices"]]
        )
        by_fold[f"fold_{fold['fold_number']}_validation"] = summarize_label_distribution(
            label_df.iloc[fold["validation_indices"]]
        )
    by_fold["holdout_train"] = summarize_label_distribution(label_df.iloc[split_payload["holdout"]["train_indices"]])
    by_fold["holdout_eval"] = summarize_label_distribution(label_df.iloc[split_payload["holdout"]["holdout_indices"]])
    return by_fold


def build_label_diagnostic_payload(
    label_df: pd.DataFrame,
    horizon_days: int,
    family: str,
    holdout_start: str,
    n_splits: int,
    embargo_days: int,
    min_train_dates: int,
) -> dict[str, object]:
    formulas = {
        "excess": "excess_forward_return = forward_return - benchmark_forward_return; target = 1 if excess_forward_return > 0 else 0",
        "thresholded": "excess_forward_return = forward_return - benchmark_forward_return; target = 1 if excess_forward_return > +0.015, 0 if excess_forward_return < -0.015, else dropped",
        "quantile": "Within each tradable_date, rank excess_forward_return cross-sectionally; top tercile = 1, bottom tercile = 0, middle tercile = dropped",
    }
    return {
        "label_variant": str(label_df["label_variant"].iloc[0]),
        "label_family": family,
        "horizon_days": int(horizon_days),
        "formula": formulas[family],
        "overall": summarize_label_distribution(label_df),
        "by_fold": build_fold_distribution_summary(
            label_df=label_df,
            horizon_days=horizon_days,
            holdout_start=holdout_start,
            n_splits=n_splits,
            embargo_days=embargo_days,
            min_train_dates=min_train_dates,
        ),
    }


def save_label_artifacts(
    label_df: pd.DataFrame,
    diagnostic_payload: dict[str, object],
    horizon_days: int,
    family: str,
) -> None:
    label_path = _label_map_path(horizon_days, family)
    diagnostic_path = _diagnostic_path(horizon_days, family)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_columns = [
        "event_id",
        "ticker",
        "date",
        "validation_group",
        "forward_return",
        "benchmark_forward_return",
        "excess_forward_return",
        "target",
        "label_available",
        "dropped_ambiguous",
        "label_family",
        "label_variant",
    ]
    extra_columns = [column for column in ["threshold", "quantile_buckets"] if column in label_df.columns]
    label_df[label_columns + extra_columns].to_parquet(label_path, index=False)
    diagnostic_path.write_text(json.dumps(diagnostic_payload, indent=2), encoding="utf-8")


def write_recommendation_summary(all_payloads: list[dict[str, object]]) -> None:
    summary_path = QUARTERLY_OUTPUTS_LABELS_DIR / "quarterly_label_recommendation.md"
    lines = [
        "# Quarterly Label Recommendation",
        "",
        f"Default quarterly benchmark label: `{DEFAULT_RECOMMENDED_LABEL}`",
        "",
        "Reason:",
        "- It matches the redesign brief's primary label: 21-trading-day excess return versus the sector benchmark.",
        "- It keeps all non-ambiguous events, which makes it the clean benchmark to compare against the thresholded and quantile variants.",
        "- The thresholded 21-day label is kept as the cleaner secondary benchmark for robustness checks.",
        "",
        "Implemented formulas:",
        "- `excess`: stock forward return minus sector leave-one-out benchmark, then sign.",
        "- `thresholded`: same excess return, but keep only moves outside +/-1.5%.",
        "- `quantile`: top tercile vs bottom tercile within each tradable date, middle dropped.",
        "",
        "Generated variants:",
    ]
    for payload in all_payloads:
        overall = payload["overall"]
        lines.append(
            f"- `{payload['label_variant']}`: kept `{overall['label_available_count']}` rows, "
            f"dropped ambiguous `{overall['dropped_ambiguous_count']}`, class_1_rate `{overall['class_1_rate']}`."
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_stock_prediction_directories()
    panel_df = load_quarterly_panel(Path(args.panel_path))
    prices_df = normalize_price_data(load_price_data(Path(args.price_path)))

    payloads: list[dict[str, object]] = []
    for horizon_days in DEFAULT_HORIZONS:
        base_df = attach_base_returns(
            panel_df=panel_df,
            prices_df=prices_df,
            horizon_days=horizon_days,
            benchmark_mode=str(args.benchmark_mode),
        )
        families = {
            "excess": build_excess_label_map(base_df, horizon_days),
            "thresholded": build_thresholded_label_map(base_df, horizon_days, threshold=float(args.threshold)),
            "quantile": build_quantile_label_map(base_df, horizon_days),
        }
        for family, label_df in families.items():
            payload = build_label_diagnostic_payload(
                label_df=label_df,
                horizon_days=horizon_days,
                family=family,
                holdout_start=str(args.holdout_start),
                n_splits=int(args.n_splits),
                embargo_days=int(args.embargo_days),
                min_train_dates=int(args.min_train_dates),
            )
            payloads.append(payload)
            save_label_artifacts(label_df, payload, horizon_days, family)

    write_recommendation_summary(payloads)
    print(f"Saved quarterly label maps and diagnostics to: {QUARTERLY_OUTPUTS_LABELS_DIR}")


if __name__ == "__main__":
    main()
