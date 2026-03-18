"""Helpers for defining which companies belong in the project universe.

Later, this file can hold logic for loading a ticker list, validating it,
or mapping tickers to SEC identifiers such as CIK values.
"""

from typing import Iterable, List


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
