from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.config_event_v1 import PRICE_INPUT_PATH
from src.event_panel_v2_schema import assert_matches_canonical_base_contract, order_columns_with_canonical_base_first
from src.label_comparison_event_v2 import load_event_panel
from src.panel_builder import prepare_prices
from src.paths import INTERIM_DATA_DIR
from src.paths import QUARTERLY_OUTPUTS_DIAGNOSTICS_DIR
from src.pipeline_utils import write_missingness_report
from src.quarterly_feature_design import (
    QUARTERLY_FEATURE_COLUMNS,
    build_event_aware_market_feature_group_map,
    build_cross_sectional_coverage,
    build_event_sentiment_coverage_diagnostics,
    build_feature_family_coverage,
    build_feature_family_map,
    build_sentiment_group_map,
    build_quarterly_feature_design_panel,
)

INPUT_PANEL_PATH = INTERIM_DATA_DIR / "event_panel_v2_phase6b_alpha_vantage.parquet"
OUTPUT_PANEL_PATH = INTERIM_DATA_DIR / "event_panel_v2_quarterly_feature_design.parquet"
DEFAULT_MISSINGNESS_PATH = QUARTERLY_OUTPUTS_DIAGNOSTICS_DIR / "quarterly_feature_design_missingness.csv"
DEFAULT_FAMILY_MAP_PATH = QUARTERLY_OUTPUTS_DIAGNOSTICS_DIR / "quarterly_feature_design_family_map.csv"
DEFAULT_FAMILY_COVERAGE_PATH = QUARTERLY_OUTPUTS_DIAGNOSTICS_DIR / "quarterly_feature_design_family_coverage.csv"
DEFAULT_CROSS_SECTIONAL_COVERAGE_PATH = QUARTERLY_OUTPUTS_DIAGNOSTICS_DIR / "quarterly_feature_design_cross_sectional_coverage.csv"
DEFAULT_MARKET_GROUP_MAP_PATH = QUARTERLY_OUTPUTS_DIAGNOSTICS_DIR / "quarterly_event_aware_market_feature_groups.csv"
DEFAULT_EVENT_SENTIMENT_COVERAGE_PATH = QUARTERLY_OUTPUTS_DIAGNOSTICS_DIR / "quarterly_event_sentiment_coverage.csv"
DEFAULT_SENTIMENT_GROUP_MAP_PATH = QUARTERLY_OUTPUTS_DIAGNOSTICS_DIR / "quarterly_sentiment_group_map.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the quarterly feature-design event panel.")
    parser.add_argument("--input-path", default=str(INPUT_PANEL_PATH))
    parser.add_argument("--price-path", default=str(PRICE_INPUT_PATH))
    parser.add_argument("--output-path", default=str(OUTPUT_PANEL_PATH))
    parser.add_argument("--missingness-path", default=str(DEFAULT_MISSINGNESS_PATH))
    parser.add_argument("--family-map-path", default=str(DEFAULT_FAMILY_MAP_PATH))
    parser.add_argument("--family-coverage-path", default=str(DEFAULT_FAMILY_COVERAGE_PATH))
    parser.add_argument("--cross-sectional-coverage-path", default=str(DEFAULT_CROSS_SECTIONAL_COVERAGE_PATH))
    parser.add_argument("--market-group-map-path", default=str(DEFAULT_MARKET_GROUP_MAP_PATH))
    parser.add_argument("--event-sentiment-coverage-path", default=str(DEFAULT_EVENT_SENTIMENT_COVERAGE_PATH))
    parser.add_argument("--sentiment-group-map-path", default=str(DEFAULT_SENTIMENT_GROUP_MAP_PATH))
    return parser.parse_args()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    price_path = Path(args.price_path)
    output_path = Path(args.output_path)
    missingness_path = Path(args.missingness_path)
    family_map_path = Path(args.family_map_path)
    family_coverage_path = Path(args.family_coverage_path)
    cross_sectional_coverage_path = Path(args.cross_sectional_coverage_path)
    market_group_map_path = Path(args.market_group_map_path)
    event_sentiment_coverage_path = Path(args.event_sentiment_coverage_path)
    sentiment_group_map_path = Path(args.sentiment_group_map_path)

    panel_df = load_event_panel(input_path)
    panel_df = panel_df.drop(columns=["date"], errors="ignore")
    price_df = prepare_prices(
        pd.read_parquet(price_path, columns=["ticker", "date", "open", "close", "adj_close", "volume"])
    )
    canonical_columns = assert_matches_canonical_base_contract(
        panel_df,
        panel_name="event_panel_v2_quarterly_feature_design input panel",
        allowed_extra_prefixes=("av_",),
    )

    enriched = build_quarterly_feature_design_panel(panel_df, price_df=price_df)
    assert_matches_canonical_base_contract(
        enriched,
        panel_name="event_panel_v2_quarterly_feature_design output panel",
        allowed_extra_prefixes=("av_", "qfd_"),
    )
    enriched = order_columns_with_canonical_base_first(enriched, canonical_columns=canonical_columns)
    ensure_parent_dir(output_path)
    enriched.to_parquet(output_path, index=False)

    family_map = build_feature_family_map()
    family_coverage = build_feature_family_coverage(enriched).round(2)
    cross_sectional_coverage = build_cross_sectional_coverage(enriched).round(2)
    market_group_map = build_event_aware_market_feature_group_map()
    event_sentiment_coverage = build_event_sentiment_coverage_diagnostics(enriched).round(2)
    sentiment_group_map = build_sentiment_group_map()
    write_missingness_report(enriched, QUARTERLY_FEATURE_COLUMNS, missingness_path)
    ensure_parent_dir(family_map_path)
    ensure_parent_dir(family_coverage_path)
    ensure_parent_dir(cross_sectional_coverage_path)
    ensure_parent_dir(market_group_map_path)
    ensure_parent_dir(event_sentiment_coverage_path)
    ensure_parent_dir(sentiment_group_map_path)
    family_map.to_csv(family_map_path, index=False)
    family_coverage.to_csv(family_coverage_path, index=False)
    cross_sectional_coverage.to_csv(cross_sectional_coverage_path, index=False)
    market_group_map.to_csv(market_group_map_path, index=False)
    event_sentiment_coverage.to_csv(event_sentiment_coverage_path, index=False)
    sentiment_group_map.to_csv(sentiment_group_map_path, index=False)

    missingness = enriched[QUARTERLY_FEATURE_COLUMNS].isna().mean().mul(100).round(2).sort_values(ascending=False)
    print(f"Saved quarterly feature-design panel to: {output_path}")
    print(f"Rows: {len(enriched):,}")
    print(f"Saved feature missingness diagnostics to: {missingness_path}")
    print(f"Saved feature family map to: {family_map_path}")
    print(f"Saved feature family coverage to: {family_coverage_path}")
    print(f"Saved cross-sectional coverage to: {cross_sectional_coverage_path}")
    print(f"Saved event-aware market feature groups to: {market_group_map_path}")
    print(f"Saved event-specific sentiment coverage to: {event_sentiment_coverage_path}")
    print(f"Saved sentiment ablation group map to: {sentiment_group_map_path}")
    print("Top quarterly-feature missingness:")
    for feature, pct in missingness.head(10).items():
        print(f"{feature:<45} {pct:>6.2f}%")


if __name__ == "__main__":
    main()
