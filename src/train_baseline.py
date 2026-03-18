"""Starter baseline model training utilities.

Later, this file can hold a simple scikit-learn baseline model that takes
the final panel and predicts the stock direction label.
"""

import pandas as pd


def split_features_and_target(df: pd.DataFrame, target_column: str):
    """Separate a DataFrame into feature matrix X and target series y."""
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' was not found.")
    x = df.drop(columns=[target_column])
    y = df[target_column]
    return x, y
