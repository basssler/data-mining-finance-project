"""Helpers for defining and validating the project ticker universe.

This file stores the source-of-truth ticker list for the current project.
Right now the chosen universe is a Consumer Staples subset that is large
enough for the course project while still staying reasonably comparable.
"""

from typing import Iterable, List

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
