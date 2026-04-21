"""Pull quarterly Compustat fundamentals from WRDS and save them to parquet."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.pipeline_utils import configure_logging, log_frame_diagnostics
from src.project_config import DATE_RANGE, UNIVERSE_CONFIG, WRDS_FUNDAMENTALS_PATH, ensure_stock_prediction_directories

LOGGER = configure_logging("wrds_pull_fundamentals")


def build_ticker_filter() -> str:
    """Build an optional ticker filter clause from the YAML config."""
    tickers = UNIVERSE_CONFIG.get("tickers", [])
    if not tickers:
        return ""
    quoted = ", ".join(f"'{ticker}'" for ticker in sorted(set(tickers)))
    return f"and tic in ({quoted})"


def query_fundamentals() -> pd.DataFrame:
    """Run the Compustat query through the WRDS Python client."""
    try:
        import wrds
    except ImportError as exc:
        raise ImportError("The `wrds` package is required for WRDS pulls.") from exc

    ticker_filter = build_ticker_filter()
    sql = f"""
        select
            gvkey,
            datadate,
            rdq,
            tic as ticker,
            cik,
            conm as company_name,
            gsector as gics_sector,
            gind as gics_industry,
            sic,
            fyearq,
            fqtr,
            fyr,
            atq,
            ltq,
            ceqq,
            saleq,
            revtq,
            niq,
            oibdpq,
            ibq,
            cheq,
            actq,
            lctq,
            dlttq,
            cogsq,
            xsgaq,
            xintq,
            rectq,
            invtq,
            cshoq,
            oancfy
        from comp.fundq
        where datadate between '{DATE_RANGE["start"]}' and '{DATE_RANGE["end"]}'
          and indfmt = 'INDL'
          and datafmt = 'STD'
          and consol = 'C'
          and popsrc = 'D'
          {ticker_filter}
    """
    with wrds.Connection() as connection:
        return connection.raw_sql(sql, date_cols=["datadate", "rdq"])


def main() -> None:
    """Pull and save fundamentals."""
    ensure_stock_prediction_directories()
    LOGGER.info("Querying WRDS Compustat fundamentals...")
    fundamentals = query_fundamentals()
    WRDS_FUNDAMENTALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fundamentals.to_parquet(WRDS_FUNDAMENTALS_PATH, index=False)
    log_frame_diagnostics(
        LOGGER,
        fundamentals,
        name="WRDS fundamentals",
        date_column="datadate",
        key_columns=["gvkey", "ticker", "rdq"],
    )
    LOGGER.info("Saved fundamentals to %s", WRDS_FUNDAMENTALS_PATH)


if __name__ == "__main__":
    main()
