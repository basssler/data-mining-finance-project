"""Create forward-return labels from CRSP daily prices without lookahead leakage."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.pipeline_utils import configure_logging, log_frame_diagnostics
from src.project_config import LABELED_PANEL_PATH, PREDICTION_HORIZONS, SECURITY_MASTER_PATH, WRDS_CRSP_DAILY_PATH, ensure_stock_prediction_directories

LOGGER = configure_logging("build_labels")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load CRSP daily data and the security master."""
    prices = pd.read_parquet(WRDS_CRSP_DAILY_PATH)
    master = pd.read_parquet(SECURITY_MASTER_PATH)
    return prices.copy(), master.copy()


def normalize_inputs(prices: pd.DataFrame, master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize key columns."""
    prices["permno"] = pd.to_numeric(prices["permno"], errors="coerce").astype("Int64")
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices["price"] = pd.to_numeric(prices["price"], errors="coerce")
    prices["volume"] = pd.to_numeric(prices["volume"], errors="coerce")
    prices["market_cap"] = pd.to_numeric(prices["market_cap"], errors="coerce")
    prices = prices.sort_values(["permno", "date"]).dropna(subset=["permno", "date", "price"]).copy()

    master["permno"] = pd.to_numeric(master["permno"], errors="coerce").astype("Int64")
    master["date"] = pd.to_datetime(master["date"], errors="coerce")
    return prices, master


def build_labels(prices: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    """Compute forward returns and binary labels."""
    panel = prices.merge(
        master[["permno", "date", "gvkey", "ticker", "sic", "gics_sector", "gics_industry"]],
        on=["permno", "date"],
        how="left",
    )
    panel["close"] = panel["price"]
    panel["ret_1d"] = panel["total_ret"]

    for horizon in PREDICTION_HORIZONS:
        future_price = panel.groupby("permno")["close"].shift(-horizon)
        panel[f"fwd_ret_{horizon}d"] = future_price / panel["close"] - 1.0
        label_values = pd.Series(pd.NA, index=panel.index, dtype="Int64")
        valid = panel[f"fwd_ret_{horizon}d"].notna()
        label_values.loc[valid] = (panel.loc[valid, f"fwd_ret_{horizon}d"] > 0).astype("int64")
        panel[f"label_up_{horizon}d"] = label_values

    required_future_columns = [f"fwd_ret_{horizon}d" for horizon in PREDICTION_HORIZONS]
    panel = panel.dropna(subset=required_future_columns, how="all").copy()
    return panel[
        [
            "date",
            "permno",
            "gvkey",
            "ticker",
            "sic",
            "gics_sector",
            "gics_industry",
            "market_cap",
            "close",
            "volume",
            "ret_1d",
        ]
        + [f"fwd_ret_{horizon}d" for horizon in PREDICTION_HORIZONS]
        + [f"label_up_{horizon}d" for horizon in PREDICTION_HORIZONS]
    ]


def main() -> None:
    """Build and save the labeled stock-date panel."""
    ensure_stock_prediction_directories()
    prices, master = normalize_inputs(*load_inputs())
    labeled_panel = build_labels(prices, master)
    labeled_panel.to_parquet(LABELED_PANEL_PATH, index=False)
    log_frame_diagnostics(
        LOGGER,
        labeled_panel,
        name="Labeled panel",
        date_column="date",
        key_columns=["permno", "gvkey", "ticker"],
    )
    LOGGER.info("Saved labeled panel to %s", LABELED_PANEL_PATH)


if __name__ == "__main__":
    main()
