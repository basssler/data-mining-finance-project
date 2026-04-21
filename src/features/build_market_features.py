"""Build leak-free rolling market features from CRSP daily data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.pipeline_utils import (
    configure_logging,
    rolling_max_drawdown,
    rolling_sign_flip_count,
    safe_divide,
    winsorize_series,
    write_missingness_report,
)
from src.project_config import LABELED_PANEL_PATH, MARKET_FEATURE_PATH, MARKET_MISSINGNESS_REPORT_PATH, ensure_stock_prediction_directories

LOGGER = configure_logging("build_market_features")

FEATURE_COLUMNS = [
    "ret_5d",
    "ret_10d",
    "ret_21d",
    "ret_63d",
    "trend_efficiency_21d",
    "sign_flip_count_21d",
    "rolling_abs_return_sum_21d",
    "vol_21d",
    "downside_semivariance_21d",
    "max_drawdown_63d",
    "vol_of_vol_21d",
    "volume_over_20d_avg",
    "dollar_volume_log",
    "obv_21d_change",
    "distance_from_63d_high",
    "distance_from_252d_high",
    "drawdown_from_peak_21d",
    "ret_21d_minus_market",
    "ret_21d_minus_sector",
]


def load_panel() -> pd.DataFrame:
    """Load the labeled panel with market fields."""
    return pd.read_parquet(
        LABELED_PANEL_PATH,
        columns=[
            "date",
            "permno",
            "ticker",
            "gics_sector",
            "gics_industry",
            "close",
            "volume",
            "ret_1d",
            "market_cap",
        ],
    ).copy()


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling market features without future data."""
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["permno", "date"]).copy()
    grouped = df.groupby("permno")

    for window in [5, 10, 21, 63]:
        df[f"ret_{window}d"] = grouped["close"].pct_change(window)

    path_distance = grouped["close"].diff(21).abs()
    abs_return_sum = grouped["ret_1d"].rolling(21, min_periods=21).apply(lambda x: np.abs(x).sum(), raw=True)
    abs_return_sum = abs_return_sum.reset_index(level=0, drop=True)
    df["rolling_abs_return_sum_21d"] = abs_return_sum
    df["trend_efficiency_21d"] = safe_divide(path_distance, abs_return_sum)
    df["sign_flip_count_21d"] = grouped["ret_1d"].apply(lambda x: rolling_sign_flip_count(x, 21)).reset_index(level=0, drop=True)
    df["vol_21d"] = grouped["ret_1d"].rolling(21, min_periods=21).std().reset_index(level=0, drop=True)

    downside_returns = df["ret_1d"].where(df["ret_1d"] < 0, 0.0)
    df["downside_semivariance_21d"] = (
        downside_returns.groupby(df["permno"])
        .rolling(21, min_periods=21)
        .apply(lambda x: np.mean(np.square(x)), raw=True)
        .reset_index(level=0, drop=True)
    )

    df["max_drawdown_63d"] = grouped["close"].apply(lambda x: rolling_max_drawdown(x, 63)).reset_index(level=0, drop=True)
    rolling_vol = grouped["ret_1d"].rolling(21, min_periods=21).std().reset_index(level=0, drop=True)
    df["vol_of_vol_21d"] = rolling_vol.groupby(df["permno"]).rolling(21, min_periods=21).std().reset_index(level=0, drop=True)

    volume_avg_20d = grouped["volume"].rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)
    df["volume_over_20d_avg"] = safe_divide(df["volume"], volume_avg_20d)
    df["dollar_volume_log"] = np.log1p(df["close"] * df["volume"])
    obv_increment = np.sign(df["ret_1d"].fillna(0.0)) * df["volume"].fillna(0.0)
    obv = obv_increment.groupby(df["permno"]).cumsum()
    df["obv_21d_change"] = obv.groupby(df["permno"]).diff(21)

    rolling_high_63d = grouped["close"].rolling(63, min_periods=63).max().reset_index(level=0, drop=True)
    rolling_high_252d = grouped["close"].rolling(252, min_periods=252).max().reset_index(level=0, drop=True)
    rolling_high_21d = grouped["close"].rolling(21, min_periods=21).max().reset_index(level=0, drop=True)
    df["distance_from_63d_high"] = safe_divide(df["close"], rolling_high_63d) - 1.0
    df["distance_from_252d_high"] = safe_divide(df["close"], rolling_high_252d) - 1.0
    df["drawdown_from_peak_21d"] = safe_divide(df["close"], rolling_high_21d) - 1.0

    market_ret_21d = df.groupby("date")["ret_21d"].transform("mean")
    sector_ret_21d = df.groupby(["date", "gics_sector"])["ret_21d"].transform("mean")
    df["ret_21d_minus_market"] = df["ret_21d"] - market_ret_21d
    df["ret_21d_minus_sector"] = df["ret_21d"] - sector_ret_21d

    for column in FEATURE_COLUMNS:
        df[column] = winsorize_series(df[column], 0.01, 0.99)

    return df[["date", "permno", "ticker"] + FEATURE_COLUMNS]


def main() -> None:
    """Build and save rolling market features."""
    ensure_stock_prediction_directories()
    market_features = build_features(load_panel())
    market_features.to_parquet(MARKET_FEATURE_PATH, index=False)
    write_missingness_report(market_features, FEATURE_COLUMNS, MARKET_MISSINGNESS_REPORT_PATH)
    LOGGER.info("Saved market features to %s", MARKET_FEATURE_PATH)


if __name__ == "__main__":
    main()
