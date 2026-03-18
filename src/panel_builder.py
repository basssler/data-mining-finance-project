"""Utilities for assembling the modeling panel.

The panel is the final table where each row represents a company and a
date, along with the features that will later be used for training.
"""

import pandas as pd


def combine_frames(left: pd.DataFrame, right: pd.DataFrame, on: list[str]) -> pd.DataFrame:
    """Merge two DataFrames on shared key columns."""
    if left is None or right is None:
        raise ValueError("Both input DataFrames are required.")
    return left.merge(right, on=on, how="inner")
