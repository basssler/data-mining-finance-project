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

from src.quarterly_feature_design import build_sentiment_group_map
from src.train_event_panel_v2 import format_metric

DEFAULT_BASE_CONFIG_PATH = Path("configs") / "quarterly" / "quarterly_core_no_market_anchor_v1.yaml"
DEFAULT_COMPARISON_CSV_PATH = Path("reports") / "results" / "quarterly_phase9_sentiment_comparison.csv"
DEFAULT_COMPARISON_MD_PATH = Path("reports") / "results" / "quarterly_phase9_sentiment_comparison.md"
DEFAULT_USABLE_COUNTS_PATH = Path("reports") / "results" / "quarterly_phase9_usable_feature_counts_by_group.csv"
DEFAULT_SELECTED_FOLD_SUMMARY_PATH = Path("reports") / "results" / "quarterly_phase9_selected_fold_summary.csv"
DEFAULT_SURVIVORS_PATH = Path("reports") / "results" / "quarterly_phase9_sentiment_feature_survivors.csv"
DEFAULT_SUMMARY_MD_PATH = Path("reports") / "results" / "quarterly_phase9_summary.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 9 quarterly sentiment comparison.")
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


def build_regimes(sentiment_group_map: pd.DataFrame) -> list[dict[str, object]]:
    broad_features = sorted(
        sentiment_group_map.loc[
            sentiment_group_map["sentiment_group"] == "broad_filing_sentiment",
            "feature_name",
        ].unique().tolist()
    )
    event_features = sorted(
        sentiment_group_map.loc[
            sentiment_group_map["sentiment_group"] == "event_specific_sentiment",
            "feature_name",
        ].unique().tolist()
    )
    return [
        {
            "regime_name": "core_no_sentiment",
            "add_exclusions": broad_features + event_features,
            "remove_exclusions": [],
            "description": "Frozen quarterly core anchor with both broad filing sentiment and Phase 9 event-specific sentiment excluded.",
        },
        {
            "regime_name": "broad_filing_sentiment_only",
            "add_exclusions": event_features,
            "remove_exclusions": broad_features,
            "description": "Frozen quarterly core anchor plus broad filing sentiment only.",
        },
        {
            "regime_name": "event_specific_sentiment_only",
            "add_exclusions": [],
            "remove_exclusions": event_features,
            "description": "Frozen quarterly core anchor plus Phase 9 event-specific sentiment only.",
        },
        {
            "regime_name": "combined_sentiment_block",
            "add_exclusions": [],
            "remove_exclusions": broad_features + event_features,
            "description": "Frozen quarterly core anchor plus both broad filing sentiment and Phase 9 event-specific sentiment.",
        },
    ]


def _selected_fold_rows(validation_dir: Path, benchmark_name: str, selected_model_name: str) -> pd.DataFrame:
    fold_summary_path = validation_dir / "fold_summary.csv"
    fold_summary_df = pd.read_csv(fold_summary_path)
    filtered = fold_summary_df.loc[fold_summary_df["model_name"] == selected_model_name].copy()
    filtered.insert(0, "benchmark_name", benchmark_name)
    return filtered


def _usable_feature_counts_by_group(
    benchmark_name: str,
    selected_model_name: str,
    usable_features: list[str],
    sentiment_group_map: pd.DataFrame,
) -> list[dict[str, object]]:
    sentiment_lookup: dict[str, set[str]] = {}
    for row in sentiment_group_map.itertuples(index=False):
        sentiment_lookup.setdefault(str(row.feature_name), set()).add(str(row.sentiment_group))

    counts = {
        "broad_filing_sentiment": 0,
        "event_specific_sentiment": 0,
        "combined_sentiment_block": 0,
    }
    for feature in usable_features:
        for group_name in sentiment_lookup.get(feature, set()):
            counts[group_name] = counts.get(group_name, 0) + 1

    rows: list[dict[str, object]] = []
    for group_name, count in sorted(counts.items()):
        rows.append(
            {
                "benchmark_name": benchmark_name,
                "selected_model_name": selected_model_name,
                "group_type": "sentiment_group",
                "group_name": group_name,
                "usable_feature_count": int(count),
            }
        )
    return rows


def _build_survivor_rows(
    benchmark_name: str,
    winning_regime_name: str,
    selected_model_name: str,
    usable_features: list[str],
    sentiment_group_map: pd.DataFrame,
) -> pd.DataFrame:
    sentiment_lookup = (
        sentiment_group_map.groupby("feature_name", sort=True)["sentiment_group"]
        .apply(lambda series: "|".join(sorted(series.astype(str).unique())))
        .to_dict()
    )
    rows = []
    for feature in usable_features:
        sentiment_group = sentiment_lookup.get(feature)
        if sentiment_group is None:
            continue
        rows.append(
            {
                "winning_benchmark_name": benchmark_name,
                "winning_regime_name": winning_regime_name,
                "selected_model_name": selected_model_name,
                "feature_name": feature,
                "sentiment_group": sentiment_group,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "winning_benchmark_name",
                "winning_regime_name",
                "selected_model_name",
                "feature_name",
                "sentiment_group",
            ]
        )
    return pd.DataFrame(rows).sort_values(["sentiment_group", "feature_name"]).reset_index(drop=True)


def _choose_winner(result_df: pd.DataFrame) -> pd.Series:
    return result_df.sort_values(
        ["holdout_auc", "cv_auc_mean", "cv_auc_std", "worst_fold_auc"],
        ascending=[False, False, True, False],
    ).iloc[0]


def _bool_answer(value: bool) -> str:
    return "Yes" if value else "No"


def _float_value(result_df: pd.DataFrame, regime_name: str, column: str) -> float:
    return float(result_df.set_index("regime_name").loc[regime_name, column])


def build_markdown(result_df: pd.DataFrame, survivor_df: pd.DataFrame, winner_row: pd.Series) -> str:
    event_only_beats_broad = _float_value(result_df, "event_specific_sentiment_only", "holdout_auc") > _float_value(
        result_df, "broad_filing_sentiment_only", "holdout_auc"
    )
    combined_beats_core = _float_value(result_df, "combined_sentiment_block", "holdout_auc") > _float_value(
        result_df, "core_no_sentiment", "holdout_auc"
    )
    winner_survivor_groups = set()
    if not survivor_df.empty:
        for value in survivor_df["sentiment_group"].astype(str):
            winner_survivor_groups.update(value.split("|"))
    event_families_helping = any(group in winner_survivor_groups for group in ["event_specific_sentiment", "combined_sentiment_block"])

    rows = [
        "# Quarterly Phase 9 Sentiment Comparison",
        "",
        "## Setup",
        "",
        "- Base config: frozen quarterly benchmark anchor `quarterly_core_no_market_anchor_v1`.",
        "- Label contract: `21d_excess_thresholded` via the frozen prebuilt label map.",
        "- Event timing, purged walk-forward validation, and 2024 holdout were kept unchanged.",
        "- Phase 8 market features remained excluded in every setup.",
        "",
        "## Selected Models",
        "",
        "| Setup | Selected Model | CV AUC Mean | CV AUC Std | Worst Fold AUC | Holdout AUC | Holdout Rows | Broad Usable | Event-Specific Usable | Combined Usable |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    counts_lookup = (
        result_df.set_index("regime_name")[
            [
                "broad_filing_sentiment_count",
                "event_specific_sentiment_count",
                "combined_sentiment_block_count",
            ]
        ]
        .to_dict(orient="index")
    )
    for _, row in result_df.iterrows():
        counts = counts_lookup[str(row["regime_name"])]
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
                    str(int(counts["broad_filing_sentiment_count"])),
                    str(int(counts["event_specific_sentiment_count"])),
                    str(int(counts["combined_sentiment_block_count"])),
                ]
            )
            + " |"
        )

    rows.extend(
        [
            "",
            "## Direct Answers",
            "",
            f"- Does `event_specific_sentiment_only` beat `broad_filing_sentiment_only`? `{_bool_answer(event_only_beats_broad)}`",
            f"- Does `combined_sentiment_block` beat `core_no_sentiment`? `{_bool_answer(combined_beats_core)}`",
            f"- Are delta/surprise/dispersion/attention features helping, or just adding noise? `{'Helping' if event_families_helping and event_only_beats_broad else 'Mostly noise or unproven in this pass'}`",
            f"- Is Phase 9 strong enough to keep as part of the benchmark stack before Phase 10? `{_bool_answer(str(winner_row['regime_name']) in {'event_specific_sentiment_only', 'combined_sentiment_block'})}`",
            "",
            "## Winner",
            "",
            f"- Winning setup: `{winner_row['regime_name']}` with `{winner_row['selected_model_name']}`.",
            f"- Winner metrics: CV AUC `{format_metric(winner_row['cv_auc_mean'])}`, CV AUC std `{format_metric(winner_row['cv_auc_std'])}`, worst fold AUC `{format_metric(winner_row['worst_fold_auc'])}`, holdout AUC `{format_metric(winner_row['holdout_auc'])}`.",
            "",
            "## Sentiment Survivors In Winner",
            "",
        ]
    )

    if survivor_df.empty:
        rows.append("- No sentiment features survived into the selected winner.")
    else:
        for sentiment_group, group_df in survivor_df.groupby("sentiment_group", sort=True):
            feature_list = ", ".join(group_df["feature_name"].tolist())
            rows.append(f"- `{sentiment_group}`: {feature_list}")

    return "\n".join(rows) + "\n"


def build_summary_markdown(result_df: pd.DataFrame, winner_row: pd.Series) -> str:
    event_only_beats_broad = _float_value(result_df, "event_specific_sentiment_only", "holdout_auc") > _float_value(
        result_df, "broad_filing_sentiment_only", "holdout_auc"
    )
    combined_beats_core = _float_value(result_df, "combined_sentiment_block", "holdout_auc") > _float_value(
        result_df, "core_no_sentiment", "holdout_auc"
    )
    keep_phase9 = str(winner_row["regime_name"]) in {"event_specific_sentiment_only", "combined_sentiment_block"}
    lines = [
        "# Quarterly Phase 9 Summary",
        "",
        "## What Changed",
        "",
        "- Rebuilt the quarterly feature-design panel with Phase 9 event-specific sentiment features.",
        "- Re-ran the frozen 21d thresholded quarterly benchmark contract across four sentiment setups only.",
        "",
        "## Which Setup Won",
        "",
        f"- Winner: `{winner_row['regime_name']}` using `{winner_row['selected_model_name']}`.",
        f"- Holdout AUC: `{format_metric(winner_row['holdout_auc'])}`.",
        "",
        "## Direct Decision",
        "",
        f"- Event-specific only beats broad filing only: `{_bool_answer(event_only_beats_broad)}`.",
        f"- Combined sentiment beats no-sentiment core: `{_bool_answer(combined_beats_core)}`.",
        f"- Keep Phase 9 in the benchmark stack: `{_bool_answer(keep_phase9)}`.",
        f"- Move on to Phase 10 now: `{_bool_answer(keep_phase9)}`.",
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

    print("Rebuilding quarterly feature-design panel with Phase 9 sentiment features...")
    subprocess.run([sys.executable, "src/build_event_panel_v2_quarterly_feature_design.py"], check=True)

    sentiment_group_map = build_sentiment_group_map()
    comparison_rows: list[dict[str, object]] = []
    usable_count_rows: list[dict[str, object]] = []
    selected_fold_frames: list[pd.DataFrame] = []

    for regime in build_regimes(sentiment_group_map):
        config = load_yaml(base_config_path)
        regime_name = str(regime["regime_name"])
        additions = list(regime["add_exclusions"])
        removals = list(regime["remove_exclusions"])
        description = str(regime["description"])

        explicit_exclusions = [feature for feature in list(config["feature_exclusions"]["explicit"]) if feature not in removals]
        config["feature_exclusions"]["explicit"] = list(dict.fromkeys(explicit_exclusions + additions))
        config["panel"]["name"] = f"quarterly_phase9_{regime_name}"
        config["metadata"]["report_title"] = f"Quarterly Phase 9 {regime_name}"
        config["metadata"]["panel_display_name"] = f"quarterly_phase9_{regime_name}"
        config["metadata"]["setup_note"] = description
        config["outputs"]["csv"] = str(comparison_csv_path.parent / f"quarterly_phase9_{regime_name}_benchmark.csv")
        config["outputs"]["markdown"] = str(comparison_csv_path.parent / f"quarterly_phase9_{regime_name}_benchmark.md")
        config["outputs"]["shap_plot"] = str(comparison_csv_path.parent / f"quarterly_phase9_{regime_name}_shap_summary.png")
        config["outputs"]["shap_csv"] = str(comparison_csv_path.parent / f"quarterly_phase9_{regime_name}_shap_importance.csv")
        config["outputs"]["validation_dir"] = str(Path("outputs") / "quarterly" / "validation" / f"quarterly_phase9_{regime_name}")
        write_yaml(tmp_config_path, config)

        benchmark_csv_path = Path(config["outputs"]["csv"])
        validation_dir = Path(str(config["outputs"]["validation_dir"]))
        fold_summary_path = validation_dir / "fold_summary.csv"
        print(f"Running Phase 9 benchmark for regime: {regime_name}")
        subprocess.run([sys.executable, "src/train_event_panel_v2.py", "--config", str(tmp_config_path)], check=True)

        if not benchmark_csv_path.exists() or not fold_summary_path.exists():
            raise FileNotFoundError(f"Expected benchmark artifacts were not created for regime: {regime_name}")

        selected = selected_row(benchmark_csv_path)
        selected_model_name = str(selected["model_name"])
        usable_features = json.loads(str(selected["usable_feature_columns_last_fold"]))
        usable_count_rows.extend(
            _usable_feature_counts_by_group(
                benchmark_name=regime_name,
                selected_model_name=selected_model_name,
                usable_features=usable_features,
                sentiment_group_map=sentiment_group_map,
            )
        )
        selected_fold_frames.append(_selected_fold_rows(validation_dir, regime_name, selected_model_name))
        regime_counts = pd.DataFrame(
            _usable_feature_counts_by_group(
                benchmark_name=regime_name,
                selected_model_name=selected_model_name,
                usable_features=usable_features,
                sentiment_group_map=sentiment_group_map,
            )
        )
        count_lookup = dict(zip(regime_counts["group_name"], regime_counts["usable_feature_count"]))
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
                "broad_filing_sentiment_count": int(count_lookup.get("broad_filing_sentiment", 0)),
                "event_specific_sentiment_count": int(count_lookup.get("event_specific_sentiment", 0)),
                "combined_sentiment_block_count": int(count_lookup.get("combined_sentiment_block", 0)),
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
        sentiment_group_map=sentiment_group_map,
    )

    result_df.to_csv(comparison_csv_path, index=False)
    pd.DataFrame(usable_count_rows).to_csv(usable_counts_csv_path, index=False)
    pd.concat(selected_fold_frames, ignore_index=True).to_csv(selected_fold_summary_csv_path, index=False)
    survivor_df.to_csv(survivors_csv_path, index=False)
    comparison_md_path.write_text(build_markdown(result_df, survivor_df, winner_row), encoding="utf-8")
    summary_md_path.write_text(build_summary_markdown(result_df, winner_row), encoding="utf-8")

    print(f"Wrote Phase 9 comparison CSV to: {comparison_csv_path}")
    print(f"Wrote Phase 9 comparison Markdown to: {comparison_md_path}")
    print(f"Wrote usable feature counts to: {usable_counts_csv_path}")
    print(f"Wrote selected fold summary to: {selected_fold_summary_csv_path}")
    print(f"Wrote sentiment feature survivors to: {survivors_csv_path}")
    print(f"Wrote Phase 9 summary to: {summary_md_path}")


if __name__ == "__main__":
    main()
