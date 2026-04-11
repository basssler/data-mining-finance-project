"""Helpers for defining and validating project ticker universes."""

import csv
from pathlib import Path
from typing import Iterable, List

from src.paths import DATA_DIR

# Chosen project universe:
# 34 large-cap Consumer Staples names. KVUE is excluded because its public
# trading history is much shorter than the full 2015-2024 project horizon.
#
# Note on ticker formatting:
# Brown-Forman Class B is stored as BF-B instead of BF.B because the dash
# format is more reliable across data providers and is easier to reuse later
# for price data.
LAYER1_TICKERS = [
    "WMT",
    "COST",
    "PG",
    "KO",
    "PM",
    "MDLZ",
    "PEP",
    "MO",
    "CL",
    "TGT",
    "MNST",
    "KR",
    "KDP",
    "ADM",
    "SYY",
    "KMB",
    "HSY",
    "DG",
    "CHD",
    "STZ",
    "DLTR",
    "GIS",
    "KHC",
    "TSN",
    "EL",
    "BG",
    "CLX",
    "MKC",
    "SJM",
    "CAG",
    "TAP",
    "HRL",
    "BF-B",
    "CPB",
]

UNIVERSE_V2_TICKERS_PATH = DATA_DIR / "reference" / "universe_v2_tickers.csv"


def normalize_tickers(tickers: Iterable[str]) -> List[str]:
    """Return a clean uppercase ticker list with blanks removed."""
    cleaned = []
    for ticker in tickers:
        if ticker is None:
            continue
        value = str(ticker).strip().upper()
        if value:
            cleaned.append(value)
    return cleaned


def get_layer1_tickers() -> List[str]:
    """Return the cleaned Layer 1 ticker universe."""
    return normalize_tickers(LAYER1_TICKERS)


def load_tickers_from_csv(path: Path) -> List[str]:
    """Load a ticker list from a CSV file with a required `ticker` column."""
    if not path.exists():
        raise FileNotFoundError(f"Ticker universe file was not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "ticker" not in reader.fieldnames:
            raise ValueError(f"Ticker universe file must include a `ticker` column: {path}")
        return normalize_tickers(row.get("ticker") for row in reader)


def get_universe_v2_tickers(path: Path | None = None) -> List[str]:
    """Return the cleaned cross-sector universe_v2 ticker list."""
    return load_tickers_from_csv(path or UNIVERSE_V2_TICKERS_PATH)
