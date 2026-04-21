"""Build a canonical security master using CCM link windows and CRSP metadata."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.pipeline_utils import configure_logging, log_frame_diagnostics
from src.project_config import (
    SECURITY_MASTER_PATH,
    WRDS_CCM_LINKS_PATH,
    WRDS_CRSP_DAILY_PATH,
    WRDS_FUNDAMENTALS_PATH,
    ensure_stock_prediction_directories,
)

LOGGER = configure_logging("build_security_master")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load WRDS source tables."""
    ccm = pd.read_parquet(WRDS_CCM_LINKS_PATH)
    crsp = pd.read_parquet(WRDS_CRSP_DAILY_PATH)
    fundamentals = pd.read_parquet(WRDS_FUNDAMENTALS_PATH)
    return ccm.copy(), crsp.copy(), fundamentals.copy()


def normalize_inputs(
    ccm: pd.DataFrame,
    crsp: pd.DataFrame,
    fundamentals: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare dtypes and date windows."""
    ccm["permno"] = pd.to_numeric(ccm["permno"], errors="coerce").astype("Int64")
    ccm["gvkey"] = ccm["gvkey"].astype("string")
    ccm["linkdt"] = pd.to_datetime(ccm["linkdt"], errors="coerce")
    ccm["linkenddt"] = pd.to_datetime(ccm["linkenddt"], errors="coerce").fillna(pd.Timestamp("2099-12-31"))

    crsp["permno"] = pd.to_numeric(crsp["permno"], errors="coerce").astype("Int64")
    crsp["date"] = pd.to_datetime(crsp["date"], errors="coerce")
    crsp["ticker"] = crsp["ticker"].astype("string")

    fundamentals["gvkey"] = fundamentals["gvkey"].astype("string")
    fundamentals["ticker"] = fundamentals["ticker"].astype("string")
    fundamentals["datadate"] = pd.to_datetime(fundamentals["datadate"], errors="coerce")
    return ccm, crsp, fundamentals


def build_master(ccm: pd.DataFrame, crsp: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Create a row-level valid mapping between CRSP dates and CCM links."""
    crsp_keys = (
        crsp[["permno", "date", "ticker", "sic", "exchcd", "shrcd", "market_cap"]]
        .drop_duplicates(subset=["permno", "date"])
        .copy()
    )

    fundamentals_meta = (
        fundamentals[["gvkey", "ticker", "gics_sector", "gics_industry", "sic"]]
        .sort_values(["gvkey"])
        .drop_duplicates(subset=["gvkey"], keep="last")
        .rename(columns={"ticker": "fundamental_ticker", "sic": "fundamental_sic"})
    )

    merged = crsp_keys.merge(ccm, on="permno", how="left")
    valid_mask = merged["date"].between(merged["linkdt"], merged["linkenddt"], inclusive="both")
    merged = merged.loc[valid_mask].copy()
    merged["link_window_days"] = (merged["linkenddt"] - merged["linkdt"]).dt.days
    merged = merged.sort_values(
        ["permno", "date", "linkprim", "link_window_days"],
        ascending=[True, True, True, False],
    )
    merged["mapping_rank"] = merged.groupby(["permno", "date"]).cumcount() + 1
    merged["is_ambiguous_mapping"] = merged.groupby(["permno", "date"])["gvkey"].transform("nunique").gt(1)
    merged = merged.loc[merged["mapping_rank"] == 1].copy()
    merged = merged.merge(fundamentals_meta, on="gvkey", how="left")
    merged["canonical_ticker"] = merged["ticker"].fillna(merged["fundamental_ticker"])
    merged["canonical_sic"] = merged["sic"].fillna(merged["fundamental_sic"])
    return merged[
        [
            "permno",
            "date",
            "gvkey",
            "canonical_ticker",
            "canonical_sic",
            "gics_sector",
            "gics_industry",
            "exchcd",
            "shrcd",
            "market_cap",
            "linkdt",
            "linkenddt",
            "linktype",
            "linkprim",
            "is_ambiguous_mapping",
        ]
    ].rename(columns={"canonical_ticker": "ticker", "canonical_sic": "sic"})


def main() -> None:
    """Build and save the security master."""
    ensure_stock_prediction_directories()
    ccm, crsp, fundamentals = normalize_inputs(*load_inputs())
    security_master = build_master(ccm, crsp, fundamentals)
    duplicate_count = int(security_master.duplicated(subset=["permno", "date"]).sum())
    if duplicate_count:
        raise ValueError(f"Security master still has {duplicate_count} duplicate (permno, date) rows.")
    security_master.to_parquet(SECURITY_MASTER_PATH, index=False)
    log_frame_diagnostics(
        LOGGER,
        security_master,
        name="Security master",
        date_column="date",
        key_columns=["permno", "gvkey", "ticker"],
    )
    LOGGER.info(
        "Security master ambiguous mapping share: %.2f%%",
        float(security_master["is_ambiguous_mapping"].mean() * 100),
    )
    LOGGER.info("Saved security master to %s", SECURITY_MASTER_PATH)


if __name__ == "__main__":
    main()
