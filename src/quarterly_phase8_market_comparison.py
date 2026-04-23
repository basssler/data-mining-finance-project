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
from src.quarterly_feature_design import (
    build_event_aware_market_feature_group_map,
    build_feature_family_map,
)
from src.train_event_panel_v2 import format_metric

DEFAULT_BASE_CONFIG_PATH = Path("configs") / "quarterly" / "quarterly_final_core_confirmation_v1.yaml"
DEFAULT_COMPARISON_CSV_PATH = Path("reports") / "results" / "quarterly_phase8_market_comparison.csv"
DEFAULT_COMPARISON_MD_PATH = Path("reports") / "results" / "quarterly_phase8_market_comparison.md"
DEFAULT_USABLE_COUNTS_PATH = Path("reports") / "results" / "quarterly_phase8_usable_feature_counts_by_group.csv"
DEFAULT_SELECTED_FOLD_SUMMARY_PATH = Path("reports") / "results" / "quarterly_phase8_selected_fold_summary.csv"
DEFAULT_SURVIVORS_PATH = Path("reports") / "results" / "quarterly_phase8_market_feature_survivors.csv"
DEFAULT_SUMMARY_MD_PATH = Path("reports") / "results" / "quarterly_phase8_summary.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 8 quarterly market comparison.")
    parser.add_argument("--base-config", default=str(DEFAULT_BASE_CONFIG_PATH))
    parser.add_argument("--comparison-csv", default=str(DEFAULT_COMPARISON_CSV_PATH))
    parser.add_argument("--comparison-md", default=str(DEFAULT_COMPARISON_MD_PATH))
    parser.add_argument("--usable-counts-csv", default=str(DEFAULT_USABLE_COUNTS_PATH))
    parser.add_argument("--selected-fold-summary-csv", default=str(DEFAULT_SELECTED_FOLD_SUMMARY_PATH))
    parser.add_argument("--survivors-csv", default=str(DEFAULT_SURVIVORS_PATH))
    parser.add_argument("--summary-md", default=str(DEFAULT_SUMMARY_MD_PATH))
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


def build_regimes() -> list[dict[str, object]]:
    family_map = build_feature_family_map()
    event_aware_market_features = family_map.loc[
        family_map["feature_family"].isin(["event_aware_market_pre_event", "event_aware_market_first_tradable"]),
        "feature_name",
    ].tolist()
    generic_market_features = list(LAYER2_V2_FEATURE_COLUMNS)
    return [
        {
            "regime_name": "core_no_market",
            "add_exclusions": event_aware_market_features + generic_market_features,
            "remove_exclusions": [],
            "description": "Quarterly core stack with both generic Layer 2 and Phase 8 event-aware market features excluded.",
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
            "description": "Quarterly core stack plus Phase 8 event-aware market features only.",
        },
        {
            "regime_name": "generic_and_event_aware_market",
            "add_exclusions": [],
            "remove_exclusions": generic_market_features + event_aware_market_features,
            "description": "Quarterly core stack plus both generic Layer 2 and Phase 8 event-aware market features.",
        },
    ]


def _usable_feature_counts_by_group(
    benchmark_name: str,
    selected_model_name: str,
    usable_features: list[str],
    family_map: pd.DataFrame,
    market_group_map: pd.DataFrame,
) -> list[dict[str, object]]:
    family_lookup = dict(zip(family_map["feature_name"], family_map["feature_family"]))
    market_group_lookup = dict(zip(market_group_map["feature_name"], market_group_map["market_feature_group"]))
    rows: list[dict[str, object]] = []

    family_counts: dict[str, int] = {}
    market_counts: dict[str, int] = {}
    for feature in usable_features:
        family = family_lookup.get(feature, "base_or_other")
        family_counts[family] = family_counts.get(family, 0) + 1
        market_group = market_group_lookup.get(feature, "non_event_aware_market")
        market_counts[market_group] = market_counts.get(market_group, 0) + 1

    for group_name, count in sorted(family_counts.items()):
        rows.append(
            {
                "benchmark_name": benchmark_name,
                "selected_model_name": selected_model_name,
                "group_type": "feature_family",
                "group_name": group_name,
                "usable_feature_count": int(count),
            }
        )
    for group_name, count in sorted(market_counts.items()):
        rows.append(
            {
                "benchmark_name": benchmark_name,
                "selected_model_name": selected_model_name,
                "group_type": "market_group",
                "group_name": group_name,
                "usable_feature_count": int(count),
            }
        )
    return rows


def _selected_fold_rows(validation_dir: Path, benchmark_name: str, selected_model_name: str) -> pd.DataFrame:
    fold_summary_path = validation_dir / "fold_summary.csv"
    fold_summary_df = pd.read_csv(fold_summary_path)
    filtered = fold_summary_df.loc[fold_summary_df["model_name"] == selected_model_name].copy()
    filtered.insert(0, "benchmark_name", benchmark_name)
    return filtered


def _build_survivor_rows(
    benchmark_name: str,
    winning_regime_name: str,
    selected_model_name: str,
    usable_features: list[str],
    family_map: pd.DataFrame,
    market_group_map: pd.DataFrame,
) -> pd.DataFrame:
    family_lookup = dict(zip(family_map["feature_name"], family_map["feature_family"]))
    market_group_lookup = dict(zip(market_group_map["feature_name"], market_group_map["market_feature_group"]))
    rows = []
    for feature in usable_features:
        if feature not in market_group_lookup:
            continue
        rows.append(
            {
                "winning_benchmark_name": benchmark_name,
                "winning_regime_name": winning_regime_name,
                "selected_model_name": selected_model_name,
                "feature_name": feature,
                "feature_family": family_lookup.get(feature, "unknown"),
                "market_feature_group": market_group_lookup.get(feature, "unknown"),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "winning_benchmark_name",
                "winning_regime_name",
                "selected_model_name",
                "feature_name",
                "feature_family",
                "market_feature_group",
            ]
        )
    return pd.DataFrame(rows).sort_values(["market_feature_group", "feature_name"]).reset_index(drop=True)


def _choose_winner(result_df: pd.DataFrame) -> pd.Series:
    return result_df.sort_values(
        ["holdout_auc", "cv_auc_mean", "cv_auc_std", "worst_fold_auc"],
        ascending=[False, False, True, False],
    ).iloc[0]


def _bool_answer(value: bool) -> str:
    return "Yes" if value else "No"


def build_markdown(result_df: pd.DataFrame, survivor_df: pd.DataFrame, winner_row: pd.Series) -> str:
    rows = [
        "# Quarterly Phase 8 Market Comparison",
        "",
        "## Setup",
        "",
        "- Frozen quarterly anchor family: `levels_plus_deltas_plus_cross_sectional`.",
        "- Frozen label anchor: `21d_excess_thresholded`.",
        "- Event timing, purged walk-forward validation, and 2024 holdout were kept unchanged from the frozen anchor config.",
        "",
        "## Selected Models",
        "",
        "| Setup | Selected Model | CV AUC Mean | CV AUC Std | Worst Fold AUC | Holdout AUC | Holdout Rows | Usable Features | Pre-Event Features | First-Tradable Features |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in result_df.iterrows():
        rows.append(
            "| "
            + " | ".join(
                [
                    str(row["regime_name"]),
                    str(row["selected_model_name"]),
                    format_metric(row["cv_auc_mean"]),
                    format_metric(row["cv_auc_std"]),
                    format_metric(row["worst_fold_auc"]),
                    format_metric(row["holdout_auc"]),
                    str(int(row["holdout_row_count"])),
                    str(int(row["usable_feature_count_last_fold"])),
                    str(int(row["pre_event_feature_count"])),
                    str(int(row["first_tradable_feature_count"])),
                ]
            )
            + " |"
        )

    rows.extend(
        [
            "",
            "## Direct Answers",
            "",
            f"- Does `event_aware_market_only` beat `generic_market_only`? `{_bool_answer(bool(result_df.set_index('regime_name').loc['event_aware_market_only', 'holdout_auc'] > result_df.set_index('regime_name').loc['generic_market_only', 'holdout_auc']))}`",
            f"- Does `generic_and_event_aware_market` beat `core_no_market`? `{_bool_answer(bool(result_df.set_index('regime_name').loc['generic_and_event_aware_market', 'holdout_auc'] > result_df.set_index('regime_name').loc['core_no_market', 'holdout_auc']))}`",
            f"- Are first-tradable-session features helping? `{_bool_answer(int(winner_row['first_tradable_feature_count']) > 0)}`",
            f"- Should Phase 8 stay in the stack before Phase 9? `{_bool_answer(str(winner_row['regime_name']) in {'event_aware_market_only', 'generic_and_event_aware_market'})}`",
            "",
            "## Winner",
            "",
            f"- Winning setup: `{winner_row['regime_name']}` with `{winner_row['selected_model_name']}`.",
            f"- Winner metrics: CV AUC `{format_metric(winner_row['cv_auc_mean'])}`, CV AUC std `{format_metric(winner_row['cv_auc_std'])}`, worst fold AUC `{format_metric(winner_row['worst_fold_auc'])}`, holdout AUC `{format_metric(winner_row['holdout_auc'])}`.",
            "",
            "## Event-Aware Survivors In Winner",
            "",
        ]
    )

    if survivor_df.empty:
        rows.append("- No event-aware market features survived into the selected winner.")
    else:
        for market_group, group_df in survivor_df.groupby("market_feature_group", sort=True):
            feature_list = ", ".join(group_df["feature_name"].tolist())
            rows.append(f"- `{market_group}`: {feature_list}")

    return "\n".join(rows) + "\n"


def build_summary_markdown(result_df: pd.DataFrame, winner_row: pd.Series) -> str:
    keep_phase8 = str(winner_row["regime_name"]) in {"event_aware_market_only", "generic_and_event_aware_market"}
    lines = [
        "# Quarterly Phase 8 Summary",
        "",
        "## What Changed",
        "",
        "- Rebuilt the quarterly feature-design panel with Phase 8 event-aware market features.",
        "- Re-ran the frozen 21d thresholded quarterly benchmark contract across four market setups only.",
        "",
        "## Which Setup Won",
        "",
        f"- Winner: `{winner_row['regime_name']}` using `{winner_row['selected_model_name']}`.",
        f"- Holdout AUC: `{format_metric(winner_row['holdout_auc'])}`.",
        "",
        "## Keep Or Drop",
        "",
        f"- Keep Phase 8: `{_bool_answer(keep_phase8)}`.",
        f"- Move to Phase 9 now: `{_bool_answer(keep_phase8)}`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    base_config_path = Path(args.base_config)
    comparison_csv_path = Path(args.comparison_csv)
    comparison_md_path = Path(args.comparison_md)
    usable_counts_csv_path = Path(args.usable_counts_csv)
    selected_fold_summary_csv_path = Path(args.selected_fold_summary_csv)
    survivors_csv_path = Path(args.survivors_csv)
    summary_md_path = Path(args.summary_md)

    for path in [
        comparison_csv_path,
        comparison_md_path,
        usable_counts_csv_path,
        selected_fold_summary_csv_path,
        survivors_csv_path,
        summary_md_path,
    ]:
        ensure_parent_dir(path)

    tmp_config_path = comparison_csv_path.with_suffix(".tmp.yaml")
    base_config = load_yaml(base_config_path)

    print("Rebuilding quarterly feature-design panel with Phase 8 event-aware market features...")
    subprocess.run([sys.executable, "src/build_event_panel_v2_quarterly_feature_design.py"], check=True)

    family_map = build_feature_family_map()
    market_group_map = build_event_aware_market_feature_group_map()

    comparison_rows: list[dict[str, object]] = []
    usable_count_rows: list[dict[str, object]] = []
    selected_fold_frames: list[pd.DataFrame] = []

    for regime in build_regimes():
        config = load_yaml(base_config_path)
        regime_name = str(regime["regime_name"])
        additions = list(regime["add_exclusions"])
        removals = list(regime["remove_exclusions"])
        description = str(regime["description"])

        explicit_exclusions = [feature for feature in list(config["feature_exclusions"]["explicit"]) if feature not in removals]
        config["feature_exclusions"]["explicit"] = list(dict.fromkeys(explicit_exclusions + additions))
        config["panel"]["name"] = f"quarterly_phase8_{regime_name}"
        config["metadata"]["report_title"] = f"Quarterly Phase 8 {regime_name}"
        config["outputs"]["csv"] = str(comparison_csv_path.parent / f"quarterly_phase8_{regime_name}_benchmark.csv")
        config["outputs"]["markdown"] = str(comparison_csv_path.parent / f"quarterly_phase8_{regime_name}_benchmark.md")
        config["outputs"]["shap_plot"] = str(comparison_csv_path.parent / f"quarterly_phase8_{regime_name}_shap_summary.png")
        config["outputs"]["shap_csv"] = str(comparison_csv_path.parent / f"quarterly_phase8_{regime_name}_shap_importance.csv")
        config["outputs"]["validation_dir"] = str(Path("outputs") / "quarterly" / "validation" / f"quarterly_phase8_{regime_name}")
        write_yaml(tmp_config_path, config)

        benchmark_csv_path = Path(config["outputs"]["csv"])
        validation_dir = Path(str(config["outputs"]["validation_dir"]))
        fold_summary_path = validation_dir / "fold_summary.csv"
        if benchmark_csv_path.exists() and fold_summary_path.exists():
            print(f"Reusing existing Phase 8 benchmark artifacts for regime: {regime_name}")
        else:
            print(f"Running Phase 8 benchmark for regime: {regime_name}")
            subprocess.run([sys.executable, "src/train_event_panel_v2.py", "--config", str(tmp_config_path)], check=True)

        selected = selected_row(benchmark_csv_path)
        selected_model_name = str(selected["model_name"])
        usable_features = json.loads(str(selected["usable_feature_columns_last_fold"]))
        selected_fold_frames.append(_selected_fold_rows(validation_dir, regime_name, selected_model_name))
        usable_count_rows.extend(
            _usable_feature_counts_by_group(
                benchmark_name=regime_name,
                selected_model_name=selected_model_name,
                usable_features=usable_features,
                family_map=family_map,
                market_group_map=market_group_map,
            )
        )
        pre_event_feature_count = sum(feature in set(market_group_map.loc[market_group_map["market_feature_group"] == "pre_event", "feature_name"]) for feature in usable_features)
        first_tradable_feature_count = sum(feature in set(market_group_map.loc[market_group_map["market_feature_group"] == "first_tradable", "feature_name"]) for feature in usable_features)
        comparison_rows.append(
            {
                "regime_name": regime_name,
                "regime_description": description,
                "selected_model_name": selected_model_name,
                "cv_auc_mean": float(selected["cv_auc_mean"]),
                "cv_auc_std": float(selected["cv_auc_std"]),
                "worst_fold_auc": float(selected["worst_fold_auc"]),
                "holdout_auc": float(selected["holdout_auc"]),
                "holdout_row_count": int(selected["holdout_row_count"]),
                "usable_feature_count_last_fold": int(selected["usable_feature_count_last_fold"]),
                "pre_event_feature_count": int(pre_event_feature_count),
                "first_tradable_feature_count": int(first_tradable_feature_count),
                "added_exclusions": json.dumps(additions),
                "removed_exclusions": json.dumps(removals),
                "usable_feature_columns_last_fold": json.dumps(usable_features),
            }
        )

    if tmp_config_path.exists():
        tmp_config_path.unlink()

    result_df = pd.DataFrame(comparison_rows).sort_values(
        ["holdout_auc", "cv_auc_mean", "cv_auc_std", "worst_fold_auc"],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)
    winner_row = _choose_winner(result_df)
    survivor_df = _build_survivor_rows(
        benchmark_name=str(winner_row["regime_name"]),
        winning_regime_name=str(winner_row["regime_name"]),
        selected_model_name=str(winner_row["selected_model_name"]),
        usable_features=json.loads(str(winner_row["usable_feature_columns_last_fold"])),
        family_map=family_map,
        market_group_map=market_group_map,
    )

    result_df.to_csv(comparison_csv_path, index=False)
    pd.DataFrame(usable_count_rows).to_csv(usable_counts_csv_path, index=False)
    pd.concat(selected_fold_frames, ignore_index=True).to_csv(selected_fold_summary_csv_path, index=False)
    survivor_df.to_csv(survivors_csv_path, index=False)
    comparison_md_path.write_text(build_markdown(result_df, survivor_df, winner_row), encoding="utf-8")
    summary_md_path.write_text(build_summary_markdown(result_df, winner_row), encoding="utf-8")

    print(f"Wrote Phase 8 comparison CSV to: {comparison_csv_path}")
    print(f"Wrote Phase 8 comparison Markdown to: {comparison_md_path}")
    print(f"Wrote usable feature counts to: {usable_counts_csv_path}")
    print(f"Wrote selected fold summary to: {selected_fold_summary_csv_path}")
    print(f"Wrote market feature survivors to: {survivors_csv_path}")
    print(f"Wrote Phase 8 summary to: {summary_md_path}")


if __name__ == "__main__":
    main()
