"""Build additive event_v1 modeling panels without touching the benchmark path."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.config_event_v1 import (
    ANALYST_EVENT_FEATURE_COLUMNS,
    ANALYST_EVENT_V1_OUTPUT_PATH,
    EVENT_V1_FULL_PANEL_PATH,
    EVENT_V1_LAYER1_ANALYST_PANEL_PATH,
    EVENT_V1_LAYER1_LAYER2_PANEL_PATH,
    EVENT_V1_LAYER1_PANEL_PATH,
    IDENTIFIER_COLUMNS,
    LABEL_COLUMNS,
    LABEL_OUTPUT_PATH,
    LAYER1_BASE_PANEL_PATH,
    LAYER1_FEATURE_COLUMNS,
    LAYER1_METADATA_COLUMNS,
    LAYER2_V2_FEATURE_COLUMNS,
    LAYER3_EVENT_FEATURE_COLUMNS,
    LAYER3_EVENT_INTERACTION_COLUMNS,
    MARKET_FEATURE_V2_OUTPUT_PATH,
    PANEL_CHOICES,
    SENTIMENT_EVENT_V1_OUTPUT_PATH,
    TARGET_COLUMN,
    ensure_event_v1_directories,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI options for event_v1 panel assembly."""
    parser = argparse.ArgumentParser(description="Build event_v1 modeling panels.")
    parser.add_argument("--panel", choices=PANEL_CHOICES, required=True)
    parser.add_argument("--labels-path", default=str(LABEL_OUTPUT_PATH))
    parser.add_argument("--analyst-path", default=str(ANALYST_EVENT_V1_OUTPUT_PATH))
    parser.add_argument("--layer2-path", default=str(MARKET_FEATURE_V2_OUTPUT_PATH))
    parser.add_argument("--layer3-path", default=str(SENTIMENT_EVENT_V1_OUTPUT_PATH))
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--drop-neutral-band", action="store_true")
    return parser.parse_args()


def load_parquet(path: Path, required_columns: list[str], dataset_name: str) -> pd.DataFrame:
    """Load a parquet file and validate required columns."""
    if not path.exists():
        raise FileNotFoundError(f"{dataset_name} file was not found: {path}")

    df = pd.read_parquet(path)
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"{dataset_name} file is missing required columns: " + ", ".join(missing_columns)
        )
    return df.copy()


def prepare_layer1_base_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only Layer 1 metadata plus engineered feature columns."""
    keep_columns = [column for column in LAYER1_METADATA_COLUMNS if column in df.columns]
    keep_columns += [column for column in LAYER1_FEATURE_COLUMNS if column in df.columns]

    prepared = df[keep_columns].copy()
    prepared["ticker"] = prepared["ticker"].astype("string")
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")

    if "filing_date" in prepared.columns:
        prepared["filing_date"] = pd.to_datetime(prepared["filing_date"], errors="coerce")
    if "period_end" in prepared.columns:
        prepared["period_end"] = pd.to_datetime(prepared["period_end"], errors="coerce")

    for column in LAYER1_FEATURE_COLUMNS:
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    prepared = prepared.dropna(subset=["ticker", "date"]).copy()
    prepared = prepared.drop_duplicates(subset=IDENTIFIER_COLUMNS).copy()
    prepared = prepared.sort_values(["ticker", "date"]).reset_index(drop=True)
    return prepared


def prepare_label_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the event_v1 label table."""
    prepared = df.copy()
    prepared["ticker"] = prepared["ticker"].astype("string")
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    prepared["forward_return_5d"] = pd.to_numeric(prepared["forward_return_5d"], errors="coerce")
    prepared["benchmark_forward_return_5d"] = pd.to_numeric(
        prepared["benchmark_forward_return_5d"],
        errors="coerce",
    )
    prepared["excess_forward_return_5d"] = pd.to_numeric(
        prepared["excess_forward_return_5d"],
        errors="coerce",
    )
    prepared[TARGET_COLUMN] = pd.to_numeric(prepared[TARGET_COLUMN], errors="coerce").astype("Int64")
    prepared["within_neutral_band"] = prepared["within_neutral_band"].fillna(False).astype(bool)

    prepared = prepared.dropna(subset=["ticker", "date"]).copy()
    prepared = prepared.drop_duplicates(subset=IDENTIFIER_COLUMNS).copy()
    prepared = prepared.sort_values(["ticker", "date"]).reset_index(drop=True)
    return prepared


def prepare_daily_feature_table(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Normalize a ticker/date daily feature table before merging."""
    prepared = df.copy()
    prepared["ticker"] = prepared["ticker"].astype("string")
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")

    for column in feature_columns:
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    prepared = prepared.dropna(subset=["ticker", "date"]).copy()
    prepared = prepared.drop_duplicates(subset=IDENTIFIER_COLUMNS).copy()
    prepared = prepared.sort_values(["ticker", "date"]).reset_index(drop=True)
    return prepared


def add_layer3_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Create the first-pass Layer 3 x market-control interaction terms."""
    panel = df.copy()
    panel["sec_event_delta_x_vol21"] = panel["sec_event_delta_prev"] * panel["realized_vol_21d"]
    panel["sec_event_magnitude_x_days_since"] = (
        panel["sec_event_abs_latest"] * panel["sec_event_days_since_filing"]
    )
    negative_sentiment_magnitude = (-panel["sec_event_score_latest"]).clip(lower=0)
    panel["sec_event_negative_x_abnormal_volume"] = (
        negative_sentiment_magnitude * panel["abnormal_volume_flag"]
    )
    return panel


def build_event_v1_panel(
    layer1_path: str,
    labels_path: str,
    analyst_path: str | None = None,
    layer2_path: str | None = None,
    layer3_path: str | None = None,
    output_path: str | None = None,
    drop_neutral_band: bool = False,
) -> pd.DataFrame:
    """Build an event_v1 panel with optional Layer 2 and Layer 3 additions."""
    base_panel_df = prepare_layer1_base_panel(
        load_parquet(
            Path(layer1_path),
            ["ticker", "date"] + LAYER1_FEATURE_COLUMNS,
            "Layer 1 base panel",
        )
    )
    label_df = prepare_label_data(
        load_parquet(
            Path(labels_path),
            ["ticker", "date"] + LABEL_COLUMNS,
            "event_v1 label",
        )
    )

    panel_df = base_panel_df.merge(
        label_df[IDENTIFIER_COLUMNS + LABEL_COLUMNS],
        on=IDENTIFIER_COLUMNS,
        how="left",
        validate="one_to_one",
    )

    if analyst_path is not None:
        analyst_df = prepare_daily_feature_table(
            load_parquet(
                Path(analyst_path),
                ["ticker", "date"] + ANALYST_EVENT_FEATURE_COLUMNS,
                "event_v1 analyst layer",
            ),
            ANALYST_EVENT_FEATURE_COLUMNS,
        )
        panel_df = panel_df.merge(
            analyst_df[IDENTIFIER_COLUMNS + ANALYST_EVENT_FEATURE_COLUMNS],
            on=IDENTIFIER_COLUMNS,
            how="left",
            validate="one_to_one",
        )

    if layer2_path is not None:
        layer2_df = prepare_daily_feature_table(
            load_parquet(
                Path(layer2_path),
                ["ticker", "date"] + LAYER2_V2_FEATURE_COLUMNS,
                "event_v1 Layer 2",
            ),
            LAYER2_V2_FEATURE_COLUMNS,
        )
        panel_df = panel_df.merge(
            layer2_df[IDENTIFIER_COLUMNS + LAYER2_V2_FEATURE_COLUMNS],
            on=IDENTIFIER_COLUMNS,
            how="left",
            validate="one_to_one",
        )

    if layer3_path is not None:
        layer3_df = prepare_daily_feature_table(
            load_parquet(
                Path(layer3_path),
                ["ticker", "date"] + LAYER3_EVENT_FEATURE_COLUMNS,
                "event_v1 Layer 3",
            ),
            LAYER3_EVENT_FEATURE_COLUMNS,
        )
        panel_df = panel_df.merge(
            layer3_df[IDENTIFIER_COLUMNS + LAYER3_EVENT_FEATURE_COLUMNS],
            on=IDENTIFIER_COLUMNS,
            how="left",
            validate="one_to_one",
        )
        panel_df = add_layer3_interactions(panel_df)

    panel_df = panel_df.sort_values(["date", "ticker"]).reset_index(drop=True)

    if drop_neutral_band:
        panel_df = panel_df[~panel_df["within_neutral_band"]].copy()

    panel_df = panel_df.dropna(subset=[TARGET_COLUMN]).copy()
    panel_df[TARGET_COLUMN] = pd.to_numeric(panel_df[TARGET_COLUMN], errors="coerce").astype("Int64")
    panel_df = panel_df.sort_values(["date", "ticker"]).reset_index(drop=True)

    if output_path is not None:
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        panel_df.to_parquet(output_path_obj, index=False)

    return panel_df


def print_panel_diagnostics(df: pd.DataFrame, feature_columns: list[str], panel_name: str) -> None:
    """Print event_v1 panel diagnostics for auditability."""
    available_feature_columns = [column for column in feature_columns if column in df.columns]
    missingness = df[available_feature_columns].isna().mean().mul(100).sort_values(ascending=False)
    target_balance = df[TARGET_COLUMN].astype(int).value_counts(normalize=True).sort_index()

    print(f"\n{panel_name} Panel Diagnostics")
    print("-" * 60)
    print(f"Rows: {len(df):,}")
    print(f"Tickers: {df['ticker'].nunique():,}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Feature count: {len(available_feature_columns)}")
    print(f"Target class 0: {float(target_balance.get(0, 0.0) * 100):.2f}%")
    print(f"Target class 1: {float(target_balance.get(1, 0.0) * 100):.2f}%")

    print("\nTop missingness columns")
    print("-" * 60)
    for column_name, percentage in missingness.head(12).items():
        print(f"{column_name:<30} {percentage:>8.2f}%")


def main() -> None:
    """Build the requested event_v1 panel."""
    args = parse_args()
    ensure_event_v1_directories()

    default_output_paths = {
        "event_v1_layer1": EVENT_V1_LAYER1_PANEL_PATH,
        "event_v1_layer1_analyst": EVENT_V1_LAYER1_ANALYST_PANEL_PATH,
        "event_v1_layer1_layer2": EVENT_V1_LAYER1_LAYER2_PANEL_PATH,
        "event_v1_full": EVENT_V1_FULL_PANEL_PATH,
    }
    output_path = Path(args.output_path) if args.output_path else default_output_paths[args.panel]

    analyst_path = None
    layer2_path = None
    layer3_path = None
    feature_columns = list(LAYER1_FEATURE_COLUMNS)

    if args.panel == "event_v1_layer1_analyst":
        analyst_path = args.analyst_path
        feature_columns += ANALYST_EVENT_FEATURE_COLUMNS

    if args.panel in {"event_v1_layer1_layer2", "event_v1_full"}:
        layer2_path = args.layer2_path
        feature_columns += LAYER2_V2_FEATURE_COLUMNS

    if args.panel == "event_v1_full":
        layer3_path = args.layer3_path
        feature_columns += LAYER3_EVENT_FEATURE_COLUMNS + LAYER3_EVENT_INTERACTION_COLUMNS

    print(f"Building {args.panel} panel...")
    panel_df = build_event_v1_panel(
        layer1_path=str(LAYER1_BASE_PANEL_PATH),
        labels_path=args.labels_path,
        analyst_path=analyst_path,
        layer2_path=layer2_path,
        layer3_path=layer3_path,
        output_path=str(output_path),
        drop_neutral_band=args.drop_neutral_band,
    )

    print(f"Saved {args.panel} panel to: {output_path}")
    print_panel_diagnostics(panel_df, feature_columns=feature_columns, panel_name=args.panel)


if __name__ == "__main__":
    main()
