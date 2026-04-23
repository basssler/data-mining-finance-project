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

from src.config_event_v1 import LAYER2_V2_FEATURE_COLUMNS
from src.quarterly_feature_design import build_feature_family_map
from src.train_event_panel_v2 import format_metric

DEFAULT_BASE_CONFIG_PATH = Path("configs") / "event_panel_v2_quarterly_feature_design_sentiment.yaml"
DEFAULT_OUTPUT_CSV_PATH = Path("reports") / "results" / "quarterly_feature_ablation.csv"
DEFAULT_OUTPUT_MD_PATH = Path("reports") / "results" / "quarterly_feature_ablation.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run family-based quarterly feature ablations.")
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


def build_regimes(base_config: dict) -> list[dict[str, object]]:
    del base_config
    family_map = build_feature_family_map()
    event_aware_market_features = family_map.loc[
        family_map["feature_family"].isin(["event_aware_market_pre_event", "event_aware_market_first_tradable"]),
        "feature_name",
    ].tolist()
    generic_market_features = list(LAYER2_V2_FEATURE_COLUMNS)
    regimes = [
        {
            "regime_name": "core_no_market",
            "add_exclusions": event_aware_market_features + generic_market_features,
            "remove_exclusions": [],
            "description": "Quarterly core stack with both generic Layer 2 and event-aware market features excluded.",
        },
        {
            "regime_name": "generic_market_only",
            "add_exclusions": event_aware_market_features,
            "remove_exclusions": generic_market_features,
            "description": "Quarterly core stack plus the old generic Layer 2 market controls only.",
        },
        {
            "regime_name": "event_aware_market_only",
            "add_exclusions": generic_market_features,
            "remove_exclusions": event_aware_market_features,
            "description": "Quarterly core stack plus event-aware pre-event and first-tradable market features only.",
        },
        {
            "regime_name": "generic_and_event_aware_market",
            "add_exclusions": [],
            "remove_exclusions": generic_market_features + event_aware_market_features,
            "description": "Quarterly core stack plus both generic Layer 2 and event-aware market feature blocks.",
        },
    ]
    return regimes


def build_markdown(result_df: pd.DataFrame) -> str:
    baseline = result_df.loc[result_df["regime_name"] == "core_no_market"].iloc[0]
    best = result_df.sort_values(
        ["holdout_auc", "cv_auc_mean", "cv_log_loss_mean"],
        ascending=[False, False, True],
    ).iloc[0]
    lines = [
        "# Quarterly Feature Ablation",
        "",
        "## Summary",
        "",
        f"- Baseline regime: `core_no_market` with CV AUC `{format_metric(baseline['cv_auc_mean'])}` and holdout AUC `{format_metric(baseline['holdout_auc'])}`.",
        f"- Best regime by holdout/CV ordering: `{best['regime_name']}` with holdout AUC `{format_metric(best['holdout_auc'])}`.",
        "",
        "## Regime Comparison",
        "",
        "| Regime | Description | Added Exclusions | Removed Exclusions | Model | CV AUC | CV Log Loss | Holdout AUC | Holdout Log Loss | Delta Holdout AUC | Top SHAP Feature |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in result_df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["regime_name"]),
                    str(row["regime_description"]),
                    str(row["added_exclusions"]),
                    str(row["removed_exclusions"]),
                    str(row["model_name"]),
                    format_metric(row["cv_auc_mean"]),
                    format_metric(row["cv_log_loss_mean"]),
                    format_metric(row["holdout_auc"]),
                    format_metric(row["holdout_log_loss"]),
                    format_metric(row["holdout_auc_delta_vs_core_no_market"]),
                    str(row["top_shap_feature"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Readout", ""])
    lines.append(
        "- Use this ladder to test whether the new event-aware market block outperforms the old generic Layer 2 market set and whether combining them adds incremental signal."
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
    tmp_config_path = output_csv_path.with_suffix(".tmp.yaml")
    rows = []

    family_map = build_feature_family_map()
    family_map_path = output_csv_path.with_name(f"{output_csv_path.stem}_feature_family_map.csv")
    family_map.to_csv(family_map_path, index=False)

    for regime in build_regimes(base_config):
        config = load_yaml(base_config_path)
        regime_name = str(regime["regime_name"])
        additions = list(regime["add_exclusions"])
        removals = list(regime["remove_exclusions"])
        description = str(regime["description"])
        explicit_exclusions = [feature for feature in list(config["feature_exclusions"]["explicit"]) if feature not in removals]
        config["feature_exclusions"]["explicit"] = list(dict.fromkeys(explicit_exclusions + additions))
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
                "regime_description": description,
                "added_exclusions": json.dumps(additions),
                "removed_exclusions": json.dumps(removals),
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
    baseline = result_df.loc[result_df["regime_name"] == "core_no_market"].iloc[0]
    result_df["holdout_auc_delta_vs_core_no_market"] = result_df["holdout_auc"] - float(baseline["holdout_auc"])
    result_df = result_df.sort_values(
        ["holdout_auc", "cv_auc_mean", "cv_log_loss_mean"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    result_df.to_csv(output_csv_path, index=False)
    output_md_path.write_text(build_markdown(result_df), encoding="utf-8")

    print(f"Wrote quarterly feature ablation CSV to: {output_csv_path}")
    print(f"Wrote quarterly feature ablation Markdown to: {output_md_path}")
    print(f"Wrote quarterly feature family map to: {family_map_path}")


if __name__ == "__main__":
    main()
