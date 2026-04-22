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

from src.train_event_panel_v2 import format_metric

DEFAULT_BASE_CONFIG_PATH = Path("configs") / "event_panel_v2_quarterly_feature_design_sentiment.yaml"
DEFAULT_OUTPUT_CSV_PATH = Path("reports") / "results" / "quarterly_feature_ablation.csv"
DEFAULT_OUTPUT_MD_PATH = Path("reports") / "results" / "quarterly_feature_ablation.md"
SHORT_HORIZON_FEATURES = [
    "rel_return_5d",
    "rel_return_10d",
    "overnight_gap_1d",
    "abs_return_shock_1d",
    "volume_ratio_20d",
    "log_volume",
    "abnormal_volume_flag",
]
SUSPICIOUS_MEDIUM_FEATURES = [
    "rel_return_21d",
    "realized_vol_63d",
    "vol_ratio_21d_63d",
    "drawdown_21d",
    "return_zscore_21d",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run quarterly feature ablations against the best redesign family.")
    parser.add_argument("--base-config", default=str(DEFAULT_BASE_CONFIG_PATH))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV_PATH))
    parser.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD_PATH))
    return parser.parse_args()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def selected_row(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path)
    selected = df.loc[df["is_selected_primary_model"] == True]  # noqa: E712
    if selected.empty:
        raise ValueError(f"No selected model row found in: {csv_path}")
    return selected.iloc[0]


def build_regimes(base_config: dict, shap_features: list[str]) -> list[tuple[str, list[str]]]:
    regimes: list[tuple[str, list[str]]] = [("full_reference", [])]
    for feature in SHORT_HORIZON_FEATURES + SUSPICIOUS_MEDIUM_FEATURES:
        if feature in shap_features and feature not in base_config["feature_exclusions"]["explicit"]:
            regimes.append((f"drop_{feature}", [feature]))
    if shap_features:
        regimes.append(("drop_top_shap_feature", [shap_features[0]]))
    if len(shap_features) >= 3:
        regimes.append(("drop_top_3_shap_features", shap_features[:3]))
    top_short_features = [
        feature for feature in shap_features[:8] if feature in set(SHORT_HORIZON_FEATURES + SUSPICIOUS_MEDIUM_FEATURES)
    ]
    if top_short_features:
        regimes.append(("drop_top_suspicious_cluster", top_short_features))
    return regimes


def build_markdown(result_df: pd.DataFrame) -> str:
    baseline = result_df.loc[result_df["regime_name"] == "full_reference"].iloc[0]
    best = result_df.sort_values(
        ["holdout_auc", "cv_auc_mean", "cv_log_loss_mean"],
        ascending=[False, False, True],
    ).iloc[0]
    lines = [
        "# Quarterly Feature Ablation",
        "",
        "## Summary",
        "",
        f"- Reference regime: CV AUC `{format_metric(baseline['cv_auc_mean'])}`, holdout AUC `{format_metric(baseline['holdout_auc'])}`.",
        f"- Best ablation regime by holdout/CV ordering: `{best['regime_name']}` with holdout AUC `{format_metric(best['holdout_auc'])}`.",
        "",
        "## Regime Comparison",
        "",
        "| Regime | Added Exclusions | Model | CV AUC | CV Log Loss | Holdout AUC | Holdout Log Loss | Delta Holdout AUC | Top SHAP Feature |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in result_df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["regime_name"]),
                    str(row["added_exclusions"]),
                    str(row["model_name"]),
                    format_metric(row["cv_auc_mean"]),
                    format_metric(row["cv_log_loss_mean"]),
                    format_metric(row["holdout_auc"]),
                    format_metric(row["holdout_log_loss"]),
                    format_metric(row["holdout_auc_delta_vs_reference"]),
                    str(row["top_shap_feature"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Readout", ""])
    lines.append(
        "- Use this report to demote short-horizon proxy features only when retraining shows they do not support the 63-day holdout."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    base_config_path = Path(args.base_config)
    output_csv_path = Path(args.output_csv)
    output_md_path = Path(args.output_md)
    ensure_parent_dir(output_csv_path)
    ensure_parent_dir(output_md_path)

    base_config = load_yaml(base_config_path)
    base_shap_csv = Path(base_config["outputs"]["shap_csv"])
    shap_features = pd.read_csv(base_shap_csv)["feature"].tolist()
    regimes = build_regimes(base_config, shap_features)

    tmp_config_path = output_csv_path.with_suffix(".tmp.yaml")
    rows = []
    for regime_name, additions in regimes:
        config = load_yaml(base_config_path)
        config["feature_exclusions"]["explicit"] = list(
            dict.fromkeys(list(config["feature_exclusions"]["explicit"]) + additions)
        )
        config["panel"]["name"] = f"{config['panel']['name']}_{regime_name}"
        config["metadata"]["report_title"] = f"{config['metadata']['report_title']} {regime_name}"
        config["outputs"]["csv"] = str(output_csv_path.parent / f"{regime_name}_benchmark.csv")
        config["outputs"]["markdown"] = str(output_csv_path.parent / f"{regime_name}_benchmark.md")
        config["outputs"]["shap_plot"] = str(output_csv_path.parent / f"{regime_name}_shap_summary.png")
        config["outputs"]["shap_csv"] = str(output_csv_path.parent / f"{regime_name}_shap_importance.csv")
        write_yaml(tmp_config_path, config)
        subprocess.run([sys.executable, "src/train_event_panel_v2.py", "--config", str(tmp_config_path)], check=True)
        selected = selected_row(Path(config["outputs"]["csv"]))
        shap_csv_path = Path(config["outputs"]["shap_csv"])
        top_shap_feature = pd.read_csv(shap_csv_path).iloc[0]["feature"] if shap_csv_path.exists() else "n/a"
        rows.append(
            {
                "regime_name": regime_name,
                "added_exclusions": json.dumps(additions),
                "model_name": selected["model_name"],
                "cv_auc_mean": float(selected["cv_auc_mean"]),
                "cv_log_loss_mean": float(selected["cv_log_loss_mean"]),
                "holdout_auc": float(selected["holdout_auc"]),
                "holdout_log_loss": float(selected["holdout_log_loss"]),
                "top_shap_feature": str(top_shap_feature),
            }
        )
    if tmp_config_path.exists():
        tmp_config_path.unlink()

    result_df = pd.DataFrame(rows)
    baseline = result_df.loc[result_df["regime_name"] == "full_reference"].iloc[0]
    result_df["holdout_auc_delta_vs_reference"] = result_df["holdout_auc"] - float(baseline["holdout_auc"])
    result_df = result_df.sort_values(
        ["holdout_auc", "cv_auc_mean", "cv_log_loss_mean"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    result_df.to_csv(output_csv_path, index=False)
    output_md_path.write_text(build_markdown(result_df), encoding="utf-8")

    print(f"Wrote quarterly feature ablation CSV to: {output_csv_path}")
    print(f"Wrote quarterly feature ablation Markdown to: {output_md_path}")


if __name__ == "__main__":
    main()
