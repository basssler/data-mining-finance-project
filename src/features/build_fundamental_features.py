"""Build daily lagged fundamental features from WRDS Compustat fundamentals."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.pipeline_utils import configure_logging, safe_divide
from src.project_config import FUNDAMENTAL_FEATURE_PATH, LABELED_PANEL_PATH, WRDS_FUNDAMENTALS_PATH, ensure_stock_prediction_directories

LOGGER = configure_logging("build_fundamental_features")

FEATURE_COLUMNS = [
    "current_ratio",
    "quick_ratio",
    "debt_to_equity",
    "debt_to_assets",
    "interest_coverage",
    "gross_margin",
    "operating_margin",
    "net_margin",
    "roa",
    "roe",
    "asset_turnover",
    "revenue_growth_yoy",
    "earnings_growth_yoy",
    "accruals_ratio",
    "cfo_to_net_income",
]


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw fundamentals and the daily panel keys."""
    fundamentals = pd.read_parquet(WRDS_FUNDAMENTALS_PATH)
    panel = pd.read_parquet(LABELED_PANEL_PATH, columns=["date", "permno", "gvkey", "ticker"])
    return fundamentals.copy(), panel.copy()


def normalize_inputs(fundamentals: pd.DataFrame, panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize dtypes and availability dates."""
    fundamentals["gvkey"] = fundamentals["gvkey"].astype("string")
    fundamentals["ticker"] = fundamentals["ticker"].astype("string")
    fundamentals["datadate"] = pd.to_datetime(fundamentals["datadate"], errors="coerce")
    fundamentals["rdq"] = pd.to_datetime(fundamentals["rdq"], errors="coerce")
    fundamentals["availability_date"] = fundamentals["rdq"].fillna(fundamentals["datadate"] + pd.Timedelta(days=60))
    fundamentals = fundamentals.sort_values(["gvkey", "availability_date", "datadate"]).copy()

    for column in fundamentals.columns:
        if column in {"gvkey", "ticker", "company_name", "gics_sector", "gics_industry", "datadate", "rdq", "availability_date"}:
            continue
        fundamentals[column] = pd.to_numeric(fundamentals[column], errors="coerce")

    panel["gvkey"] = panel["gvkey"].astype("string")
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel = panel.sort_values(["gvkey", "date"]).copy()
    return fundamentals, panel


def compute_quarterly_features(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Create quarterly ratios before the as-of join to daily rows."""
    features = fundamentals.copy()
    total_debt = features["ltq"]
    features["current_ratio"] = safe_divide(features["actq"], features["lctq"])
    features["quick_ratio"] = safe_divide(features["actq"] - features["invtq"], features["lctq"])
    features["debt_to_equity"] = safe_divide(total_debt, features["ceqq"])
    features["debt_to_assets"] = safe_divide(total_debt, features["atq"])
    features["interest_coverage"] = safe_divide(features["oibdpq"], features["xintq"])
    features["gross_margin"] = safe_divide(features["revtq"] - features["cogsq"], features["revtq"])
    features["operating_margin"] = safe_divide(features["oibdpq"], features["revtq"])
    features["net_margin"] = safe_divide(features["niq"], features["revtq"])
    features["roa"] = safe_divide(features["niq"], features["atq"])
    features["roe"] = safe_divide(features["niq"], features["ceqq"])
    features["asset_turnover"] = safe_divide(features["revtq"], features["atq"])
    features["revenue_growth_yoy"] = features.groupby("gvkey")["revtq"].pct_change(periods=4)
    features["earnings_growth_yoy"] = features.groupby("gvkey")["niq"].pct_change(periods=4)
    features["accruals_ratio"] = safe_divide(features["niq"] - features["oancfy"], features["atq"])
    features["cfo_to_net_income"] = safe_divide(features["oancfy"], features["niq"])
    return features


def align_to_daily_panel(fundamentals: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill the latest available quarter to each stock-date row."""
    base_columns = ["gvkey", "availability_date", "datadate"] + FEATURE_COLUMNS
    quarter_features = fundamentals[base_columns].sort_values(["gvkey", "availability_date"]).copy()

    merged = pd.merge_asof(
        panel.sort_values(["gvkey", "date"]),
        quarter_features,
        left_on="date",
        right_on="availability_date",
        by="gvkey",
        direction="backward",
        allow_exact_matches=True,
    )
    output_columns = ["date", "permno", "gvkey", "ticker", "datadate", "availability_date"] + FEATURE_COLUMNS
    return merged[output_columns].rename(columns={"datadate": "fundamental_period_end"})


def main() -> None:
    """Build and save lagged daily fundamentals."""
    ensure_stock_prediction_directories()
    fundamentals, panel = normalize_inputs(*load_inputs())
    quarter_features = compute_quarterly_features(fundamentals)
    daily_features = align_to_daily_panel(quarter_features, panel)
    daily_features.to_parquet(FUNDAMENTAL_FEATURE_PATH, index=False)
    LOGGER.info("Saved daily fundamental features to %s", FUNDAMENTAL_FEATURE_PATH)


if __name__ == "__main__":
    main()
