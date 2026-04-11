"""Build the Phase 6B Alpha Vantage additive panel on top of locked event_panel_v2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.alpha_vantage_earnings_phase6b import (
    AlphaVantageExhaustedError,
    FEATURE_BLOCK_PATH,
    MANIFEST_PATH,
    MERGED_PANEL_PATH,
    NORMALIZED_EARNINGS_PATH,
    NORMALIZED_ESTIMATES_PATH,
    build_event_level_feature_block,
    ensure_output_dirs,
    fetch_raw_payloads,
    merge_feature_block_onto_panel,
    normalize_earnings_from_manifest,
    normalize_estimates_from_manifest,
    save_manifest,
    save_normalized_tables,
)
from src.config_event_v1 import PRICE_INPUT_PATH
from src.label_comparison_event_v2 import load_event_panel
from src.labels_event_v1 import load_price_data, normalize_price_data
from src.paths import INTERIM_DATA_DIR
from src.universe import get_layer1_tickers

BASE_PANEL_PATH = INTERIM_DATA_DIR / "event_panel_v2.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Phase 6B Alpha Vantage additive panel.")
    parser.add_argument("--panel-path", default=str(BASE_PANEL_PATH))
    parser.add_argument("--prices-path", default=str(PRICE_INPUT_PATH))
    parser.add_argument("--mode", choices=["backfill", "refresh"], default="backfill")
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--manifest-path", default=str(MANIFEST_PATH))
    parser.add_argument("--estimates-path", default=str(NORMALIZED_ESTIMATES_PATH))
    parser.add_argument("--earnings-path", default=str(NORMALIZED_EARNINGS_PATH))
    parser.add_argument("--feature-path", default=str(FEATURE_BLOCK_PATH))
    parser.add_argument("--output-path", default=str(MERGED_PANEL_PATH))
    return parser.parse_args()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def summarize_manifest(manifest_df: pd.DataFrame) -> None:
    status_counts = manifest_df["status"].value_counts(dropna=False).sort_index()
    print("\nAlpha Vantage Manifest Summary")
    print("-" * 60)
    for status, count in status_counts.items():
        print(f"{status:<10} {int(count):>4}")


def summarize_features(feature_df: pd.DataFrame, diagnostics: dict) -> None:
    print("\nPhase 6B Alpha Vantage Feature Summary")
    print("-" * 60)
    print(f"Rows: {len(feature_df):,}")
    print(f"Matched rows: {diagnostics['matched_rows']:,}")
    print(f"Unmatched rows: {diagnostics['unmatched_rows']:,}")
    print("Top missingness:")
    for name, pct in list(diagnostics["feature_missingness_pct"].items())[:10]:
        print(f"{name:<45} {pct:>6.2f}%")


def main() -> None:
    args = parse_args()
    ensure_output_dirs()

    panel_path = Path(args.panel_path)
    prices_path = Path(args.prices_path)
    manifest_path = Path(args.manifest_path)
    estimates_path = Path(args.estimates_path)
    earnings_path = Path(args.earnings_path)
    feature_path = Path(args.feature_path)
    output_path = Path(args.output_path)

    tickers = get_layer1_tickers()
    print(f"Locked Phase 6B universe size: {len(tickers)}")

    if args.skip_fetch:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest does not exist for --skip-fetch: {manifest_path}")
        manifest_df = pd.DataFrame(json.loads(manifest_path.read_text(encoding='utf-8')))
        key_manager_summary = []
    else:
        try:
            manifest_df, key_manager = fetch_raw_payloads(
                tickers=tickers,
                mode=args.mode,
                force_refresh=args.force_refresh,
                manifest_path=manifest_path,
            )
            key_manager_summary = key_manager.masked_key_summary()
        except AlphaVantageExhaustedError as exc:
            print(f"\nStopped cleanly because all keys are exhausted: {exc}")
            manifest_df = pd.DataFrame(json.loads(manifest_path.read_text(encoding="utf-8")))
            summarize_manifest(manifest_df)
            raise

    summarize_manifest(manifest_df)
    if key_manager_summary:
        print("\nMasked key usage summary")
        print("-" * 60)
        for row in key_manager_summary:
            print(row)

    estimates_df = normalize_estimates_from_manifest(manifest_df)
    earnings_df = normalize_earnings_from_manifest(manifest_df)
    save_normalized_tables(estimates_df, earnings_df, estimates_path=estimates_path, earnings_path=earnings_path)
    print(f"\nSaved normalized estimates to: {estimates_path}")
    print(f"Saved normalized earnings to: {earnings_path}")

    print(f"\nLoading locked base panel from: {panel_path}")
    panel_df = load_event_panel(panel_path)
    print(f"Loading prices from: {prices_path}")
    prices_df = normalize_price_data(load_price_data(prices_path))

    feature_df, diagnostics = build_event_level_feature_block(panel_df, prices_df, estimates_df, earnings_df)
    ensure_parent_dir(feature_path)
    feature_df.to_parquet(feature_path, index=False)
    print(f"Saved Phase 6B Alpha Vantage feature block to: {feature_path}")
    summarize_features(feature_df, diagnostics)

    merged_panel = merge_feature_block_onto_panel(panel_df, feature_df)
    ensure_parent_dir(output_path)
    merged_panel.to_parquet(output_path, index=False)
    print(f"\nSaved merged Phase 6B panel to: {output_path}")

    diagnostics_path = feature_path.with_suffix(".diagnostics.json")
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(f"Saved diagnostics JSON to: {diagnostics_path}")


if __name__ == "__main__":
    main()
