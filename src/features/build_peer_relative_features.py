"""Build peer-relative features cross-sectionally by date with fallback groups."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.pipeline_utils import configure_logging, robust_zscore
from src.project_config import (
    FUNDAMENTAL_FEATURE_PATH,
    LABELED_PANEL_PATH,
    MARKET_FEATURE_PATH,
    PEER_RELATIVE_DICTIONARY_PATH,
    PEER_RELATIVE_FEATURE_PATH,
    dump_json,
    ensure_stock_prediction_directories,
)

LOGGER = configure_logging("build_peer_relative_features")
MIN_GROUP_SIZE = 5
RAW_FEATURE_COLUMNS = [
    "ret_21d",
    "vol_21d",
    "distance_from_63d_high",
    "debt_to_equity",
    "roa",
    "revenue_growth_yoy",
]


def load_inputs() -> pd.DataFrame:
    """Load the base panel and feature layers needed for peer transforms."""
    panel = pd.read_parquet(
        LABELED_PANEL_PATH,
        columns=["date", "permno", "ticker", "gics_sector", "gics_industry", "market_cap"],
    )
    fundamentals = pd.read_parquet(
        FUNDAMENTAL_FEATURE_PATH,
        columns=["date", "permno", "debt_to_equity", "roa", "revenue_growth_yoy"],
    )
    market = pd.read_parquet(
        MARKET_FEATURE_PATH,
        columns=["date", "permno", "ret_21d", "vol_21d", "distance_from_63d_high"],
    )
    merged = panel.merge(fundamentals, on=["date", "permno"], how="left")
    merged = merged.merge(market, on=["date", "permno"], how="left")
    return merged


def assign_size_bucket(df: pd.DataFrame) -> pd.Series:
    """Assign daily market-cap quintiles."""
    return (
        df.groupby("date")["market_cap"]
        .transform(lambda x: pd.qcut(x.rank(method="first"), 5, labels=False, duplicates="drop"))
        .astype("Int64")
    )


def add_relative_features(df: pd.DataFrame, group_column: str, prefix: str) -> pd.DataFrame:
    """Create percentile and robust-z transforms for one peer grouping."""
    output = df.copy()
    for feature in RAW_FEATURE_COLUMNS:
        counts = output.groupby(["date", group_column])[feature].transform("count")
        valid = counts >= MIN_GROUP_SIZE
        rank_column = f"{prefix}_{feature}_pct"
        z_column = f"{prefix}_{feature}_rz"
        output[rank_column] = np.nan
        output[z_column] = np.nan
        output.loc[valid, rank_column] = (
            output.loc[valid].groupby(["date", group_column])[feature].rank(pct=True, method="average")
        )
        output.loc[valid, z_column] = (
            output.loc[valid].groupby(["date", group_column])[feature].transform(robust_zscore)
        )
    return output


def build_peer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build relative features with hierarchical fallbacks."""
    output = df.copy()
    output["size_bucket"] = assign_size_bucket(output)
    output["sector_x_size_bucket"] = (
        output["gics_sector"].astype("string").fillna("unknown")
        + "_"
        + output["size_bucket"].astype("string").fillna("unknown")
    )

    output = add_relative_features(output, "gics_industry", "industry")
    output = add_relative_features(output, "gics_sector", "sector")
    output = add_relative_features(output, "size_bucket", "size")
    output = add_relative_features(output, "sector_x_size_bucket", "sector_size")

    for feature in RAW_FEATURE_COLUMNS:
        output[f"peer_{feature}_pct"] = output[f"industry_{feature}_pct"].fillna(output[f"sector_{feature}_pct"])
        output[f"peer_{feature}_pct"] = output[f"peer_{feature}_pct"].fillna(output[f"size_{feature}_pct"])
        output[f"peer_{feature}_pct"] = output[f"peer_{feature}_pct"].fillna(output.groupby("date")[feature].rank(pct=True))

        output[f"peer_{feature}_rz"] = output[f"industry_{feature}_rz"].fillna(output[f"sector_{feature}_rz"])
        output[f"peer_{feature}_rz"] = output[f"peer_{feature}_rz"].fillna(output[f"size_{feature}_rz"])
        output[f"peer_{feature}_rz"] = output[f"peer_{feature}_rz"].fillna(output.groupby("date")[feature].transform(robust_zscore))

    keep_columns = ["date", "permno", "ticker"] + [column for column in output.columns if column.startswith("peer_")]
    return output[keep_columns]


def main() -> None:
    """Build and save peer-relative features."""
    ensure_stock_prediction_directories()
    peer_features = build_peer_features(load_inputs())
    peer_features.to_parquet(PEER_RELATIVE_FEATURE_PATH, index=False)
    dump_json(
        {
            "min_group_size": MIN_GROUP_SIZE,
            "raw_features": RAW_FEATURE_COLUMNS,
            "fallback_order": ["industry", "sector", "size_bucket", "market"],
        },
        PEER_RELATIVE_DICTIONARY_PATH,
    )
    LOGGER.info("Saved peer-relative features to %s", PEER_RELATIVE_FEATURE_PATH)


if __name__ == "__main__":
    main()
