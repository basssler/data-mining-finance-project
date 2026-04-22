"""Helpers for defining and validating project ticker universes."""

import csv
from pathlib import Path
from typing import Dict, Iterable, List

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
LAYER1_DEFAULT_SECTOR = "Consumer Staples"


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


def get_layer1_sector_map() -> Dict[str, str]:
    """Return the Layer 1 sector map, treating the full set as Consumer Staples."""
    return {ticker: LAYER1_DEFAULT_SECTOR for ticker in get_layer1_tickers()}


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


def load_universe_v2_sector_map(path: Path | None = None) -> Dict[str, str]:
    """Load the universe_v2 ticker-to-sector mapping from CSV."""
    resolved_path = path or UNIVERSE_V2_TICKERS_PATH
    if not resolved_path.exists():
        raise FileNotFoundError(f"Ticker universe file was not found: {resolved_path}")

    with resolved_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"ticker", "sector"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"Ticker universe file must include `ticker` and `sector` columns: {resolved_path}"
            )

        sector_map: Dict[str, str] = {}
        for row in reader:
            ticker = normalize_tickers([row.get("ticker")])
            if not ticker:
                continue
            sector = str(row.get("sector") or "").strip()
            if not sector:
                raise ValueError(f"Ticker universe file contains a blank sector value: {resolved_path}")
            sector_map[ticker[0]] = sector
        return sector_map


def get_project_sector_map(path: Path | None = None) -> Dict[str, str]:
    """Return the combined local ticker-to-sector mapping used by benchmark labels."""
    sector_map = get_layer1_sector_map()
    sector_map.update(load_universe_v2_sector_map(path))
    return sector_map
