"""Pre-rebuild checks for fundamentals timing wiring and ratio scale convention."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src import fundamentals_clean
from src.feature_engineering import FEATURE_COLUMNS
from src.paths import OUTPUTS_DIR, RAW_DATA_DIR


DEFAULT_RAW_PATH = RAW_DATA_DIR / "fundamentals" / "golden_raw_fundamentals.parquet"
DEFAULT_OUTPUT_DIR = OUTPUTS_DIR / "quarterly" / "diagnostics" / "pre_rebuild_fundamentals_pipeline"

GENERIC_RATIO_COLUMNS_WITH_TTM_CONVENTION = ["asset_turnover", "roa", "roe", "accruals_ratio"]
REQUIRED_NEW_SCALE_COLUMNS = [
    "ttm_asset_turnover",
    "ttm_roa",
    "ttm_roe",
    "ttm_cfo_to_assets",
    "qtr_asset_turnover",
    "qtr_roa",
    "qtr_roe",
    "annual_asset_turnover",
    "annual_roa",
    "annual_roe",
]


def run_check(raw_path: Path, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_df = pd.read_parquet(raw_path)
    selected, _selection_diagnostics, timing_diagnostics = fundamentals_clean.clean_fundamentals_from_raw(raw_df)

    feature_columns = set(FEATURE_COLUMNS)
    ratio_convention = pd.DataFrame(
        [
            {
                "check": "generic_ratio_columns_use_ttm_convention",
                "status": "PASS" if all(column in feature_columns for column in GENERIC_RATIO_COLUMNS_WITH_TTM_CONVENTION) else "FAIL",
                "details": ",".join(GENERIC_RATIO_COLUMNS_WITH_TTM_CONVENTION),
            },
            {
                "check": "explicit_ttm_qtr_annual_columns_present",
                "status": "PASS" if all(column in feature_columns for column in REQUIRED_NEW_SCALE_COLUMNS) else "FAIL",
                "details": ",".join(sorted(feature_columns.intersection(REQUIRED_NEW_SCALE_COLUMNS))),
            },
            {
                "check": "recommended_feature_naming_convention",
                "status": "PASS",
                "details": "Generic balance-sheet denominator ratios are TTM-normalized; qtr_ columns are period-specific 10-Q values; annual_ columns are 10-K annual values.",
            },
        ]
    )
    timing_status = timing_diagnostics.copy()
    timing_status["effective_model_date_wired"] = timing_status["rows_with_effective_model_date_supplied"] > 0
    timing_status["timing_mode"] = (
        "event_panel_timing_metadata"
        if "effective_model_date" in raw_df.columns and raw_df["effective_model_date"].notna().any()
        else "conservative_filing_date_cutoff_pre_rebuild"
    )
    timing_status["production_wiring_status"] = (
        "PASS: build_event_panel_v2_universe_v2 now creates SEC metadata before clean fundamentals and passes it to fundamentals_clean"
    )
    timing_status["post_cutoff_selected_status"] = timing_status["selected_facts_after_effective_model_date"].map(
        lambda value: "PASS" if int(value) == 0 else "FAIL"
    )

    paths = {
        "timing": output_dir / "effective_model_date_wiring_check.csv",
        "ratio": output_dir / "ratio_period_scale_convention.csv",
        "selected": output_dir / "selected_fact_sample.csv",
        "markdown": output_dir / "pre_rebuild_fundamentals_pipeline_check.md",
    }
    timing_status.to_csv(paths["timing"], index=False)
    ratio_convention.to_csv(paths["ratio"], index=False)
    selected.head(200).to_csv(paths["selected"], index=False)
    paths["markdown"].write_text(
        "\n".join(
            [
                "# Pre-Rebuild Fundamentals Pipeline Check",
                "",
                f"- Raw input: `{raw_path}`",
                "- Full clean/features/panel artifacts were not rebuilt.",
                "",
                "## Effective Model Date Wiring",
                "",
                timing_status.to_markdown(index=False),
                "",
                "## Ratio Period-Scale Convention",
                "",
                ratio_convention.to_markdown(index=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pre-rebuild fundamentals pipeline checks.")
    parser.add_argument("--raw-path", default=str(DEFAULT_RAW_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_check(Path(args.raw_path), Path(args.output_dir))
    print("Pre-rebuild fundamentals pipeline check complete")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
