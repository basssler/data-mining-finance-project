from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.build_quarterly_labels import (
    attach_base_returns,
    build_excess_label_map,
    build_label_diagnostic_payload,
    build_quantile_label_map,
    build_thresholded_label_map,
    load_quarterly_panel,
)
from src.labels_event_v1 import load_price_data, normalize_price_data
from src.train_event_panel_v2 import format_metric, load_config

DEFAULT_BASE_CONFIG_PATH = Path("configs") / "event_panel_v2_quarterly_feature_design_core.yaml"
DEFAULT_OUTPUT_CSV_PATH = Path("reports") / "results" / "quarterly_label_family_comparison.csv"
DEFAULT_OUTPUT_MD_PATH = Path("reports") / "results" / "quarterly_label_family_comparison.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run quarterly label-family comparison on the current core panel.")
    parser.add_argument("--base-config", default=str(DEFAULT_BASE_CONFIG_PATH))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV_PATH))
    parser.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD_PATH))
    return parser.parse_args()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def selected_row(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path)
    selected = df.loc[df["is_selected_primary_model"] == True]  # noqa: E712
    if selected.empty:
        raise ValueError(f"No selected model row found in: {csv_path}")
    return selected.iloc[0]


def build_label_specs() -> list[dict[str, object]]:
    return [
        {"family_name": "21d_excess_sign", "family": "excess", "horizon_days": 21},
        {"family_name": "21d_excess_thresholded", "family": "thresholded", "horizon_days": 21, "threshold": 0.015},
        {"family_name": "21d_excess_quantile", "family": "quantile", "horizon_days": 21},
        {"family_name": "10d_excess_thresholded", "family": "thresholded", "horizon_days": 10, "threshold": 0.015},
        {"family_name": "10d_excess_sign", "family": "excess", "horizon_days": 10},
    ]


def build_label_df(base_df: pd.DataFrame, spec: dict[str, object]) -> pd.DataFrame:
    family = str(spec["family"])
    horizon_days = int(spec["horizon_days"])
    if family == "excess":
        return build_excess_label_map(base_df, horizon_days=horizon_days)
    if family == "thresholded":
        return build_thresholded_label_map(base_df, horizon_days=horizon_days, threshold=float(spec["threshold"]))
    if family == "quantile":
        return build_quantile_label_map(base_df, horizon_days=horizon_days)
    raise ValueError(f"Unsupported label family: {family}")


def build_training_join_label_df(label_df: pd.DataFrame) -> pd.DataFrame:
    join_columns = ["ticker", "date"]
    duplicate_mask = label_df.duplicated(subset=join_columns, keep=False)
    if duplicate_mask.any():
        conflict_counts = (
            label_df.loc[duplicate_mask]
            .groupby(join_columns, dropna=False)["target"]
            .nunique(dropna=False)
        )
        conflicts = conflict_counts.loc[conflict_counts > 1]
        if not conflicts.empty:
            sample_key = conflicts.index[0]
            raise ValueError(
                "Conflicting duplicate labels detected for training join key "
                f"{sample_key}; quarterly label comparison cannot proceed safely."
            )
    keep_columns = [
        column
        for column in [
            "event_id",
            "ticker",
            "date",
            "target",
            "forward_return",
            "benchmark_forward_return",
            "excess_forward_return",
        ]
        if column in label_df.columns
    ]
    return (
        label_df[keep_columns]
        .drop_duplicates(subset=join_columns, keep="first")
        .sort_values(join_columns)
        .reset_index(drop=True)
    )


def build_markdown(result_df: pd.DataFrame) -> str:
    lines = [
        "# Quarterly Label Family Comparison",
        "",
        "## Selected Models",
        "",
        "| Label Family | Selected Model | CV AUC Mean | CV AUC Std | Worst Fold AUC | Holdout AUC | Holdout Rows | Class 1 Rate | Dropped Ambiguous |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in result_df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["label_family_name"]),
                    str(row["selected_model_name"]),
                    format_metric(row["cv_auc_mean"]),
                    format_metric(row["cv_auc_std"]),
                    format_metric(row["worst_fold_auc"]),
                    format_metric(row["holdout_auc"]),
                    str(int(row["holdout_row_count"])),
                    format_metric(row["class_1_rate"]),
                    str(int(row["dropped_ambiguous_count"])),
                ]
            )
            + " |"
        )
    best = result_df.sort_values(["holdout_auc", "cv_auc_mean"], ascending=[False, False]).iloc[0]
    lines.extend(
        [
            "",
            "## Readout",
            "",
            (
                f"- Best label family by holdout/CV ordering: `{best['label_family_name']}` with selected model "
                f"`{best['selected_model_name']}` and holdout AUC `{format_metric(best['holdout_auc'])}`."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    base_config_path = Path(args.base_config)
    output_csv_path = Path(args.output_csv)
    output_md_path = Path(args.output_md)
    ensure_parent_dir(output_csv_path)
    ensure_parent_dir(output_md_path)

    base_config = load_config(base_config_path)
    panel_df = load_quarterly_panel(Path("outputs") / "quarterly" / "panels" / "quarterly_event_panel_features.parquet")
    prices_df = normalize_price_data(load_price_data(Path(base_config["prices"]["path"])))
    tmp_config_path = output_csv_path.with_suffix(".tmp.yaml")
    rows: list[dict[str, object]] = []

    for spec in build_label_specs():
        family_name = str(spec["family_name"])
        horizon_days = int(spec["horizon_days"])
        base_df = attach_base_returns(
            panel_df=panel_df,
            prices_df=prices_df,
            horizon_days=horizon_days,
            benchmark_mode=str(base_config["label"]["benchmark_mode"]),
        )
        label_df = build_label_df(base_df, spec)
        label_path = Path("outputs") / "quarterly" / "labels" / f"checkpoint_{family_name}.parquet"
        diagnostic_path = Path("outputs") / "quarterly" / "labels" / f"checkpoint_{family_name}.diagnostics.json"
        ensure_parent_dir(label_path)
        build_training_join_label_df(label_df).to_parquet(label_path, index=False)
        diagnostic_payload = build_label_diagnostic_payload(
            label_df=label_df,
            horizon_days=horizon_days,
            family=str(spec["family"]),
            holdout_start=str(base_config["holdout"]["start"]),
            n_splits=int(base_config["cv"]["n_splits"]),
            embargo_days=int(base_config["cv"]["embargo_days"]),
            min_train_dates=int(base_config["cv"]["min_train_dates"]),
        )
        diagnostic_path.write_text(json.dumps(diagnostic_payload, indent=2), encoding="utf-8")

        config = load_config(base_config_path)
        config["label"]["path"] = str(label_path)
        config["label"]["variant_name"] = str(label_df["label_variant"].iloc[0])
        config["label"]["horizon_days"] = horizon_days
        config["label"]["mode"] = "sign"
        config["panel"]["name"] = f"{config['panel']['name']}_{family_name}"
        config["metadata"]["report_title"] = f"{config['metadata']['report_title']} {family_name}"
        config["outputs"]["csv"] = str(output_csv_path.parent / f"{family_name}_benchmark.csv")
        config["outputs"]["markdown"] = str(output_csv_path.parent / f"{family_name}_benchmark.md")
        config["outputs"]["shap_plot"] = str(output_csv_path.parent / f"{family_name}_shap_summary.png")
        config["outputs"]["shap_csv"] = str(output_csv_path.parent / f"{family_name}_shap_importance.csv")
        write_yaml(tmp_config_path, config)
        subprocess.run([sys.executable, "src/train_event_panel_v2.py", "--config", str(tmp_config_path)], check=True)

        selected = selected_row(Path(config["outputs"]["csv"]))
        overall = diagnostic_payload["overall"]
        rows.append(
            {
                "label_family_name": family_name,
                "label_variant": str(label_df["label_variant"].iloc[0]),
                "selected_model_name": selected["model_name"],
                "cv_auc_mean": float(selected["cv_auc_mean"]),
                "cv_auc_std": float(selected["cv_auc_std"]),
                "worst_fold_auc": float(selected["worst_fold_auc"]),
                "holdout_auc": float(selected["holdout_auc"]),
                "holdout_row_count": int(selected["holdout_row_count"]),
                "class_1_rate": overall["class_1_rate"],
                "dropped_ambiguous_count": int(overall["dropped_ambiguous_count"]),
                "label_available_count": int(overall["label_available_count"]),
            }
        )

    if tmp_config_path.exists():
        tmp_config_path.unlink()

    result_df = pd.DataFrame(rows).sort_values(["holdout_auc", "cv_auc_mean"], ascending=[False, False]).reset_index(drop=True)
    result_df.to_csv(output_csv_path, index=False)
    output_md_path.write_text(build_markdown(result_df), encoding="utf-8")

    print(f"Wrote quarterly label-family comparison CSV to: {output_csv_path}")
    print(f"Wrote quarterly label-family comparison Markdown to: {output_md_path}")


if __name__ == "__main__":
    main()
