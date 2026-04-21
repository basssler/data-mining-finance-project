"""Pull CRSP daily equity data from WRDS and save it to parquet."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.pipeline_utils import configure_logging, log_frame_diagnostics
from src.project_config import DATE_RANGE, UNIVERSE_CONFIG, WRDS_CRSP_DAILY_PATH, ensure_stock_prediction_directories

LOGGER = configure_logging("wrds_pull_crsp_daily")


def build_ticker_filter() -> str:
    """Build an optional CRSP ticker filter."""
    tickers = UNIVERSE_CONFIG.get("tickers", [])
    if not tickers:
        return ""
    quoted = ", ".join(f"'{ticker}'" for ticker in sorted(set(tickers)))
    return f"and n.ticker in ({quoted})"


def query_crsp_daily() -> pd.DataFrame:
    """Run the CRSP daily query."""
    try:
        import wrds
    except ImportError as exc:
        raise ImportError("The `wrds` package is required for WRDS pulls.") from exc

    ticker_filter = build_ticker_filter()
    sql = f"""
        select
            d.permno,
            d.permco,
            d.date,
            n.ticker,
            n.shrcd,
            n.exchcd,
            n.siccd as sic,
            d.ret,
            d.retx,
            d.prc,
            d.vol,
            d.shrout,
            d.cfacpr,
            d.cfacshr,
            dl.dlret
        from crsp.dsf as d
        inner join crsp.msenames as n
            on d.permno = n.permno
           and d.date between n.namedt and coalesce(n.nameendt, '9999-12-31')
        left join crsp.dsedelist as dl
            on d.permno = dl.permno
           and d.date = dl.dlstdt
        where d.date between '{DATE_RANGE["start"]}' and '{DATE_RANGE["end"]}'
          and n.shrcd in (10, 11)
          {ticker_filter}
    """
    with wrds.Connection() as connection:
        return connection.raw_sql(sql, date_cols=["date"])


def finalize_crsp_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add core derived fields used later in the pipeline."""
    output = df.copy()
    output["price"] = pd.to_numeric(output["prc"], errors="coerce").abs()
    output["shares_outstanding"] = pd.to_numeric(output["shrout"], errors="coerce") * 1000.0
    output["market_cap"] = output["price"] * output["shares_outstanding"]
    output["volume"] = pd.to_numeric(output["vol"], errors="coerce")
    output["ret"] = pd.to_numeric(output["ret"], errors="coerce")
    output["dlret"] = pd.to_numeric(output["dlret"], errors="coerce")
    output["total_ret"] = (1.0 + output["ret"]).fillna(1.0) * (1.0 + output["dlret"]).fillna(1.0) - 1.0
    return output


def main() -> None:
    """Pull and save CRSP daily data."""
    ensure_stock_prediction_directories()
    LOGGER.info("Querying WRDS CRSP daily prices...")
    crsp_daily = finalize_crsp_frame(query_crsp_daily())
    WRDS_CRSP_DAILY_PATH.parent.mkdir(parents=True, exist_ok=True)
    crsp_daily.to_parquet(WRDS_CRSP_DAILY_PATH, index=False)
    log_frame_diagnostics(
        LOGGER,
        crsp_daily,
        name="WRDS CRSP daily",
        date_column="date",
        key_columns=["permno", "ticker", "ret"],
    )
    LOGGER.info("Saved CRSP daily data to %s", WRDS_CRSP_DAILY_PATH)


if __name__ == "__main__":
    main()
