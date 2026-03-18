"""Price-related utilities used only for labels or alignment later on.

This project is not building market feature layers right now. This file
exists so price-loading code needed for future target labels can stay
separate from fundamentals logic.
"""

from pathlib import Path


def get_price_file_path(filename: str) -> Path:
    """Build a path for a saved price file in the raw data folder."""
    from src.paths import RAW_DATA_DIR

    return RAW_DATA_DIR / filename
