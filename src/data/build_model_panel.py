"""Merge labels and feature layers into one modeling panel."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.pipeline_utils import configure_logging, write_missingness_report
from src.project_config import (
    FEATURE_DICTIONARY_PATH,
    FUNDAMENTAL_FEATURE_PATH,
    LABELED_PANEL_PATH,
    MARKET_FEATURE_PATH,
    MISSINGNESS_REPORT_PATH,
    MODEL_PANEL_PATH,
    PEER_RELATIVE_FEATURE_PATH,
    dump_json,
    ensure_stock_prediction_directories,
)

LOGGER = configure_logging("build_model_panel")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load labels and currently implemented layers."""
    labels = pd.read_parquet(LABELED_PANEL_PATH)
    fundamentals = pd.read_parquet(FUNDAMENTAL_FEATURE_PATH)
    market = pd.read_parquet(MARKET_FEATURE_PATH)
    peer = pd.read_parquet(PEER_RELATIVE_FEATURE_PATH)
    return labels.copy(), fundamentals.copy(), market.copy(), peer.copy()


def main() -> None:
    """Build and save the final modeling table for the implemented layers."""
    ensure_stock_prediction_directories()
    labels, fundamentals, market, peer = load_inputs()
    panel = labels.merge(fundamentals, on=["date", "permno", "gvkey", "ticker"], how="left")
    panel = panel.merge(market, on=["date", "permno", "ticker"], how="left")
    panel = panel.merge(peer, on=["date", "permno", "ticker"], how="left")

    duplicate_count = int(panel.duplicated(subset=["permno", "date"]).sum())
    if duplicate_count:
        raise ValueError(f"Model panel has {duplicate_count} duplicate (permno, date) rows.")

    panel.to_parquet(MODEL_PANEL_PATH, index=False)
    feature_columns = [column for column in panel.columns if column not in {"date", "permno", "gvkey", "ticker"}]
    write_missingness_report(panel, feature_columns, MISSINGNESS_REPORT_PATH)
    dump_json(
        {
            "identifier_columns": ["date", "permno", "gvkey", "ticker"],
            "feature_columns": feature_columns,
        },
        FEATURE_DICTIONARY_PATH,
    )
    LOGGER.info("Saved model panel to %s", MODEL_PANEL_PATH)


if __name__ == "__main__":
    main()
