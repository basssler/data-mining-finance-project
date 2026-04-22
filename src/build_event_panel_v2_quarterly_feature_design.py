from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.event_panel_v2_schema import assert_matches_canonical_base_contract, order_columns_with_canonical_base_first
from src.label_comparison_event_v2 import load_event_panel
from src.paths import INTERIM_DATA_DIR
from src.quarterly_feature_design import QUARTERLY_FEATURE_COLUMNS, build_quarterly_feature_design_panel

INPUT_PANEL_PATH = INTERIM_DATA_DIR / "event_panel_v2_phase6b_alpha_vantage.parquet"
OUTPUT_PANEL_PATH = INTERIM_DATA_DIR / "event_panel_v2_quarterly_feature_design.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the quarterly feature-design event panel.")
    parser.add_argument("--input-path", default=str(INPUT_PANEL_PATH))
    parser.add_argument("--output-path", default=str(OUTPUT_PANEL_PATH))
    return parser.parse_args()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    panel_df = load_event_panel(input_path)
    panel_df = panel_df.drop(columns=["date"], errors="ignore")
    canonical_columns = assert_matches_canonical_base_contract(
        panel_df,
        panel_name="event_panel_v2_quarterly_feature_design input panel",
        allowed_extra_prefixes=("av_",),
    )

    enriched = build_quarterly_feature_design_panel(panel_df)
    assert_matches_canonical_base_contract(
        enriched,
        panel_name="event_panel_v2_quarterly_feature_design output panel",
        allowed_extra_prefixes=("av_", "qfd_"),
    )
    enriched = order_columns_with_canonical_base_first(enriched, canonical_columns=canonical_columns)
    ensure_parent_dir(output_path)
    enriched.to_parquet(output_path, index=False)

    missingness = (
        enriched[QUARTERLY_FEATURE_COLUMNS].isna().mean().mul(100).round(2).sort_values(ascending=False)
    )
    print(f"Saved quarterly feature-design panel to: {output_path}")
    print(f"Rows: {len(enriched):,}")
    print("Top quarterly-feature missingness:")
    for feature, pct in missingness.head(10).items():
        print(f"{feature:<45} {pct:>6.2f}%")


if __name__ == "__main__":
    main()
