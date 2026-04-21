"""Pull CCM links from WRDS and save them to parquet."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.pipeline_utils import configure_logging, log_frame_diagnostics
from src.project_config import WRDS_CCM_LINKS_PATH, ensure_stock_prediction_directories

LOGGER = configure_logging("wrds_pull_ccm_links")


def query_ccm_links() -> pd.DataFrame:
    """Pull the CRSP-Compustat link table."""
    try:
        import wrds
    except ImportError as exc:
        raise ImportError("The `wrds` package is required for WRDS pulls.") from exc

    sql = """
        select
            gvkey,
            lpermno as permno,
            lpermco as permco,
            linktype,
            linkprim,
            liid,
            linkdt,
            linkenddt
        from crsp.ccmxpf_linktable
        where lpermno is not null
          and linktype in ('LU', 'LC', 'LS')
          and linkprim in ('P', 'C')
    """
    with wrds.Connection() as connection:
        return connection.raw_sql(sql, date_cols=["linkdt", "linkenddt"])


def main() -> None:
    """Pull and save CCM links."""
    ensure_stock_prediction_directories()
    LOGGER.info("Querying WRDS CCM links...")
    ccm_links = query_ccm_links()
    WRDS_CCM_LINKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ccm_links.to_parquet(WRDS_CCM_LINKS_PATH, index=False)
    log_frame_diagnostics(
        LOGGER,
        ccm_links,
        name="WRDS CCM links",
        date_column="linkdt",
        key_columns=["gvkey", "permno", "linktype"],
    )
    LOGGER.info("Saved CCM links to %s", WRDS_CCM_LINKS_PATH)


if __name__ == "__main__":
    main()
